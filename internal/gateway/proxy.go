package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/cduggn/inference-gateway/internal/pending"
	"github.com/cduggn/inference-gateway/internal/router"
	"github.com/cduggn/inference-gateway/internal/state"
	"github.com/cduggn/inference-gateway/internal/stats"
)

// maxUpstreamBody bounds a non-streamed upstream response. Replicas are trusted
// but not unbounded; a runaway generation should not be able to exhaust gateway
// memory.
const maxUpstreamBody = 32 << 20

func routerChoice(res pending.Result) router.Choice {
	return router.Choice{ReplicaID: res.ReplicaID, MatchTokens: res.MatchTokens}
}

func readAll(r io.Reader) ([]byte, error) { return io.ReadAll(r) }

// upstreamBody rewrites the client's body for the replica.
//
// Two edits, matching the Python original: deadline_ms is a gateway-level
// concept the replica does not understand, and the priority hint is promoted
// from a header into the body where vLLM expects it. Everything else is
// forwarded byte-for-byte, so parameters this gateway has never heard of still
// reach the model.
func upstreamBody(raw []byte, pr state.Request) ([]byte, error) {
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		return nil, err
	}
	delete(body, "deadline_ms")
	if pr.Priority != nil {
		body["priority"] = *pr.Priority
	}
	return json.Marshal(body)
}

// proxy forwards the request to the chosen replica and relays the response.
func (s *Server) proxy(w http.ResponseWriter, r *http.Request, pr state.Request, d dispatch, raw []byte, stream bool) {
	start := time.Now()
	outcome, reason := "ok", ""
	var ttft time.Duration

	// finish runs exactly once, on every exit path including a client
	// disconnect mid-stream, so in-flight accounting and the stats record can
	// never be skipped.
	defer func() {
		d.replica.Completed.Add(1)
		e := stats.Event{
			Kind: stats.Completed, Tenant: pr.Tenant, Replica: d.choice.ReplicaID,
			NIn: pr.NIn, NOut: pr.NOut, Outcome: outcome, Reason: reason,
			TotalMs:           float64(time.Since(start).Milliseconds()),
			PrefixMatchTokens: d.choice.MatchTokens,
		}
		if ttft > 0 {
			e.TTFTMs = float64(ttft.Milliseconds())
		}
		s.stats.Emit(e)
	}()

	body, err := upstreamBody(raw, pr)
	if err != nil {
		outcome, reason = "error", "bad_body"
		writeJSON(w, http.StatusBadRequest, map[string]any{
			"error": map[string]any{"type": "invalid_request", "reason": "malformed JSON body"},
		})
		return
	}

	url := strings.TrimSuffix(d.replica.URL, "/") + "/v1/chat/completions"
	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		outcome, reason = "error", "bad_upstream_request"
		s.upstreamError(w, d, http.StatusBadGateway)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	if stream {
		req.Header.Set("Accept", "text/event-stream")
	}

	resp, err := s.client.Do(req)
	if err != nil {
		// A cancelled client shows up here as a context error. Distinguish it
		// from a genuine upstream failure so the stats do not blame the replica.
		if r.Context().Err() != nil {
			outcome, reason = "cancelled", "client_disconnect"
			return
		}
		outcome, reason = "error", "upstream_error"
		s.log.Warn("upstream request failed", "replica", d.choice.ReplicaID, "err", err)
		s.upstreamError(w, d, http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		outcome, reason = "error", "upstream_"+strconv.Itoa(resp.StatusCode)
	}

	setDispatchHeaders(w.Header(), d, outcome, reason)

	if stream && resp.StatusCode < 400 {
		ttft = s.relayStream(w, r.Context(), resp.Body, start, &outcome, &reason)
		return
	}

	ttft = time.Since(start)
	payload, err := io.ReadAll(io.LimitReader(resp.Body, maxUpstreamBody))
	if err != nil {
		outcome, reason = "error", "upstream_read"
		s.upstreamError(w, d, http.StatusBadGateway)
		return
	}
	// The upstream body is relayed as-is. That is safe specifically because
	// replicas come from configuration and are trusted; a gateway fronting
	// caller-supplied backends should not do this.
	if ct := resp.Header.Get("Content-Type"); ct != "" {
		w.Header().Set("Content-Type", ct)
	}
	w.WriteHeader(resp.StatusCode)
	_, _ = w.Write(payload)
}

// relayStream pipes server-sent events through to the client, recording time to
// first byte and stopping early if the client disconnects.
func (s *Server) relayStream(w http.ResponseWriter, ctx context.Context, body io.Reader, start time.Time, outcome, reason *string) time.Duration {
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	flusher, canFlush := w.(http.Flusher)
	if canFlush {
		flusher.Flush()
	}

	var ttft time.Duration
	buf := make([]byte, 8<<10)
	for {
		// Checking the context each iteration is what the Python version was
		// doing with request.is_disconnected(); here cancellation also
		// propagates into the upstream read, so the replica stops generating
		// instead of finishing work nobody will read.
		if ctx.Err() != nil {
			*outcome, *reason = "cancelled", "client_disconnect"
			return ttft
		}
		n, err := body.Read(buf)
		if n > 0 {
			if ttft == 0 {
				ttft = time.Since(start)
			}
			if _, werr := w.Write(buf[:n]); werr != nil {
				*outcome, *reason = "cancelled", "client_disconnect"
				return ttft
			}
			if canFlush {
				flusher.Flush()
			}
		}
		if err != nil {
			if err != io.EOF {
				*outcome, *reason = "error", "upstream_stream"
			}
			return ttft
		}
	}
}

// upstreamError returns a generic failure. Upstream transport detail goes to
// the log, not to the caller.
func (s *Server) upstreamError(w http.ResponseWriter, d dispatch, status int) {
	setDispatchHeaders(w.Header(), d, "error", "upstream_error")
	writeJSON(w, status, map[string]any{
		"error": map[string]any{"type": "upstream", "reason": "upstream request failed"},
	})
}

// setDispatchHeaders exposes the routing decision to the caller. The agent
// driver reads these to show which replica served a turn and how much of the
// prompt was already cache-warm.
func setDispatchHeaders(h http.Header, d dispatch, outcome, reason string) {
	h.Set("X-Replica", d.choice.ReplicaID)
	h.Set("X-Prefix-Match-Tokens", strconv.Itoa(d.choice.MatchTokens))
	h.Set("X-Outcome", outcome)
	if reason != "" {
		h.Set("X-Reason", reason)
	}
}
