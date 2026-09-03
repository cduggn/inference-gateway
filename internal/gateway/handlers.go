package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/cduggn/inference-gateway/internal/admission"
	"github.com/cduggn/inference-gateway/internal/pending"
	"github.com/cduggn/inference-gateway/internal/refuse"
	"github.com/cduggn/inference-gateway/internal/state"
	"github.com/cduggn/inference-gateway/internal/stats"
	"github.com/cduggn/inference-gateway/internal/tokenize"
)

// chatRequest is the subset of the OpenAI chat-completions body the gateway
// reasons about. The full body is forwarded upstream unchanged apart from the
// two fields noted in proxy.go, so unknown fields are preserved rather than
// dropped.
type chatRequest struct {
	Model               string             `json:"model"`
	Messages            []tokenize.Message `json:"messages"`
	MaxTokens           int                `json:"max_tokens"`
	MaxCompletionTokens int                `json:"max_completion_tokens"`
	Stream              bool               `json:"stream"`
	DeadlineMs          *int               `json:"deadline_ms"`
}

// maxTenantLen bounds a tenant id before it is ever used as a map key, logged,
// or written into a stats event.
const maxTenantLen = 64

// normalizeTenant sanitises the X-Tenant-Id header.
//
// The header is caller-controlled and ends up as a quota map key and a log
// field, so it is restricted to a conservative charset and length here. Unknown
// tenants are kept (sanitised) for observability but, per bucket.Buckets, share
// a single quota bucket rather than each receiving a fresh allowance.
func normalizeTenant(raw string) string {
	if raw == "" {
		return "default"
	}
	if len(raw) > maxTenantLen {
		raw = raw[:maxTenantLen]
	}
	var b strings.Builder
	for _, r := range raw {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_', r == '.':
			b.WriteRune(r)
		default:
			// drop anything else: no control characters or separators in logs
		}
	}
	if b.Len() == 0 {
		return "default"
	}
	return b.String()
}

// deadlineFor resolves the latency objective for this request.
//
// A client may ask for a tighter deadline than its tier default, which is
// useful: a caller that will time out in 800ms would rather be shed now. It may
// not ask for a longer one. The Python version took deadline_ms from the body
// unclamped, so a client could opt out of the deadline check entirely by
// sending a large value -- and a negative value made every request instantly
// unmeetable.
func (s *Server) deadlineFor(req *chatRequest, tenant string) time.Duration {
	d := s.cfg.DeadlineFor(tenant)
	if req.DeadlineMs == nil {
		return d
	}
	asked := time.Duration(*req.DeadlineMs) * time.Millisecond
	if asked <= 0 {
		return d
	}
	return min(asked, min(d, s.cfg.MaxDeadline))
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	// Bound the body before reading a single byte of it. The Python version
	// parsed the request JSON with no size limit at all.
	r.Body = http.MaxBytesReader(w, r.Body, s.cfg.MaxRequestBytes)
	raw, err := readAll(r.Body)
	if err != nil {
		writeJSON(w, http.StatusRequestEntityTooLarge, map[string]any{
			"error": map[string]any{"type": "invalid_request", "reason": "body too large"},
		})
		return
	}

	var req chatRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{
			"error": map[string]any{"type": "invalid_request", "reason": "malformed JSON body"},
		})
		return
	}
	if len(req.Messages) == 0 {
		writeJSON(w, http.StatusBadRequest, map[string]any{
			"error": map[string]any{"type": "invalid_request", "reason": "messages must not be empty"},
		})
		return
	}

	tenant := normalizeTenant(r.Header.Get("X-Tenant-Id"))
	tokens := s.tok.Encode(tokenize.MessagesToText(req.Messages))

	nOut := req.MaxTokens
	if nOut <= 0 {
		nOut = req.MaxCompletionTokens
	}
	if nOut <= 0 {
		nOut = s.cfg.DefaultMaxTokens
	}

	pr := state.Request{
		Tenant:   tenant,
		NIn:      len(tokens),
		NOut:     nOut,
		TokenIDs: tokens,
		Deadline: s.deadlineFor(&req, tenant),
		Priority: parsePriority(r.Header.Get("X-Priority")),
	}

	s.stats.Emit(stats.Event{
		Kind: stats.Recv, Tenant: tenant, NIn: pr.NIn, NOut: pr.NOut,
		DeadlineMs: float64(pr.Deadline.Milliseconds()),
	})

	// 1. Quota. Charged for the whole request, prompt plus the completion it
	// has reserved, so a caller cannot dodge accounting with a long generation.
	if !s.quota.Allow(tenant, float64(pr.NIn+pr.NOut)) {
		s.refuse(w, refuse.Quota, pr)
		return
	}

	// 2. Admission control.
	if s.cfg.AdmissionEnabled {
		snap := s.fleet.Snapshot(time.Now())
		if reason := admission.Decide(s.cfg, snap, pr); reason != "" {
			s.refuse(w, reason, pr)
			return
		}
	}
	s.stats.Emit(stats.Event{Kind: stats.Admitted, Tenant: tenant, NIn: pr.NIn, NOut: pr.NOut})

	// 3. Capacity: straight through, or via the priority queue.
	d, err := s.acquire(r.Context(), pr)
	if err != nil {
		var re *refuse.Error
		if errors.As(err, &re) {
			s.refuse(w, re.Reason, pr)
			return
		}
		// Context cancelled: the caller hung up. Nothing to write.
		s.stats.Emit(stats.Event{
			Kind: stats.Refused, Reason: string(refuse.ClientCancelled),
			Tenant: tenant, Outcome: "cancelled",
		})
		return
	}
	defer d.release()

	s.stats.Emit(stats.Event{
		Kind: stats.Dispatched, Tenant: tenant, Replica: d.choice.ReplicaID,
		PrefixMatchTokens: d.choice.MatchTokens,
	})

	// 4. Proxy to the chosen replica.
	s.proxy(w, r, pr, d, raw, req.Stream)
}

// acquire gets a capacity reservation, going through the queue when enabled.
func (s *Server) acquire(ctx context.Context, pr state.Request) (dispatch, error) {
	if !s.cfg.QueueEnabled {
		return s.assign(ctx, pr)
	}

	item := pending.NewItem(pr, time.Now())
	if err := s.queue.Put(item); err != nil {
		return dispatch{}, err
	}
	s.stats.Emit(stats.Event{
		Kind: stats.Enqueued, Tenant: pr.Tenant, NIn: pr.NIn, NOut: pr.NOut,
		QueueDepth: s.queue.Len(),
	})

	select {
	case res := <-item.Ready():
		if res.Err != nil {
			return dispatch{}, res.Err
		}
		return dispatch{
			choice:  routerChoice(res),
			replica: s.replicaByID[res.ReplicaID],
			release: res.Release,
		}, nil

	case <-ctx.Done():
		// The caller gave up. If the item is still queued we own it and can
		// drop it outright. If a dispatcher already took it, a reservation is
		// in flight and has to be returned, or the semaphore leaks a slot.
		if !s.queue.Remove(item) {
			go func() {
				if res := <-item.Ready(); res.Release != nil {
					res.Release()
				}
			}()
		}
		return dispatch{}, ctx.Err()
	}
}

// refuse writes a structured refusal and records it.
func (s *Server) refuse(w http.ResponseWriter, reason refuse.Reason, pr state.Request) {
	snap := s.fleet.Snapshot(time.Now())
	est := snap.EstimateTotal(pr.NIn, pr.NOut)

	s.stats.Emit(stats.Event{
		Kind: stats.Refused, Reason: string(reason), Tenant: pr.Tenant,
		NIn: pr.NIn, NOut: pr.NOut, Outcome: "refused",
		EstimatedTotalMs: float64(est.Milliseconds()),
		DeadlineMs:       float64(pr.Deadline.Milliseconds()),
	})

	w.Header().Set("Retry-After", strconv.Itoa(int(reason.RetryAfter().Seconds())))
	w.Header().Set("X-Reason", string(reason))
	writeJSON(w, reason.HTTPStatus(), map[string]any{
		"error": map[string]any{
			"type":               "request_refused",
			"reason":             string(reason),
			"estimated_total_ms": est.Milliseconds(),
			"deadline_ms":        pr.Deadline.Milliseconds(),
		},
	})
}

// parsePriority reads the optional X-Priority hint.
//
// Returns nil for anything unparseable. The Python version called int() on the
// header directly, so `X-Priority: abc` raised ValueError inside the handler
// and surfaced as a 500 -- a caller-triggered server error.
func parsePriority(raw string) *int {
	if raw == "" {
		return nil
	}
	v, err := strconv.Atoi(raw)
	if err != nil {
		return nil
	}
	return &v
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	snap := s.fleet.Snapshot(time.Now())
	urls := make([]string, 0, len(snap.Replicas))
	for _, r := range snap.Replicas {
		urls = append(urls, r.URL)
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"ok":          true,
		"replicas":    urls,
		"stale_for_s": snap.StaleFor.Seconds(),
		"queue_depth": s.queue.Len(),
		"admission":   s.cfg.AdmissionEnabled,
		"queue":       s.cfg.QueueEnabled,
		"prefix":      s.cfg.UsePrefixRouting,
		"p2c":         s.cfg.UseP2C,
	})
}

func (s *Server) handleModels(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data":   []any{map[string]any{"id": s.cfg.ServedModelName, "object": "model"}},
	})
}

func (s *Server) handleStats(w http.ResponseWriter, _ *http.Request) {
	snap := s.fleet.Snapshot(time.Now())
	replicas := make([]map[string]any, 0, len(snap.Replicas))
	for _, r := range snap.Replicas {
		replicas = append(replicas, map[string]any{
			"id": r.ID, "url": r.URL,
			"waiting": r.Waiting, "running": r.Running, "kv_usage": r.KVUsage,
			"in_flight": r.InFlight, "completed": r.Completed,
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"events":      s.stats.Snapshot(),
		"queue_depth": s.queue.Len(),
		"trie_nodes":  s.router.Trie().Size(),
		"fleet": map[string]any{
			"stale_for_s":      snap.StaleFor.Seconds(),
			"kv_usage_max":     snap.KVUsageMax,
			"waiting_total":    snap.WaitingTotal,
			"running_total":    snap.RunningTotal,
			"in_flight_total":  snap.InFlightTotal,
			"headroom_tokens":  snap.HeadroomTokens,
			"queue_wait_s":     snap.QueueWait.Seconds(),
			"prefill_tokens_s": snap.PrefillTokensPerS,
			"itl_s":            snap.InterTokenLatency.Seconds(),
			"replicas":         replicas,
		},
	})
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}
