package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cduggn/inference-gateway/internal/config"
	"github.com/cduggn/inference-gateway/internal/scrape"
)

// fakeReplica stands in for a vLLM process: it answers chat completions and
// exposes the Prometheus metrics the scraper reads.
type fakeReplica struct {
	*httptest.Server
	id    string
	calls atomic.Int64
	// status, when non-zero, is returned instead of a completion.
	status atomic.Int64
}

func newFakeReplica(t *testing.T, id string, kvUsage float64) *fakeReplica {
	t.Helper()
	f := &fakeReplica{id: id}
	mux := http.NewServeMux()

	mux.HandleFunc("POST /v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		f.calls.Add(1)
		if s := f.status.Load(); s != 0 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(int(s))
			_, _ = io.WriteString(w, `{"error":{"message":"upstream said no"}}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":     "cmpl-test",
			"object": "chat.completion",
			"choices": []any{map[string]any{
				"index":         0,
				"message":       map[string]any{"role": "assistant", "content": "served by " + id},
				"finish_reason": "stop",
			}},
		})
	})

	mux.HandleFunc("GET /metrics", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4")
		fmt.Fprintf(w, `# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="lab"} 1
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="lab"} 0
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="lab"} %v
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{model_name="lab",le="+Inf"} 10
vllm:time_to_first_token_seconds_sum{model_name="lab"} 0.5
vllm:time_to_first_token_seconds_count{model_name="lab"} 10
`, kvUsage)
	})

	f.Server = httptest.NewServer(mux)
	t.Cleanup(f.Close)
	return f
}

// newTestServer wires a gateway in front of the given fakes. Background work is
// left to the caller: most tests do not need the scraper or dispatcher.
func newTestServer(t *testing.T, tune func(*config.Config), replicas ...*fakeReplica) (*Server, *httptest.Server) {
	t.Helper()

	cfg := config.Default()
	cfg.ReplicaURLs = nil
	for _, r := range replicas {
		cfg.ReplicaURLs = append(cfg.ReplicaURLs, r.URL)
	}
	if tune != nil {
		tune(&cfg)
	}

	srv, err := New(&cfg, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	front := httptest.NewServer(srv.Handler())
	t.Cleanup(front.Close)
	return srv, front
}

// chat posts a chat completion and returns the response.
func chat(t *testing.T, base, prompt string, headers map[string]string, extra map[string]any) *http.Response {
	t.Helper()

	body := map[string]any{
		"model":      "lab",
		"messages":   []any{map[string]any{"role": "user", "content": prompt}},
		"max_tokens": 32,
	}
	for k, v := range extra {
		body[k] = v
	}
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}

	req, err := http.NewRequest(http.MethodPost, base+"/v1/chat/completions", bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { resp.Body.Close() })
	return resp
}

// longPrompt is sized to produce several whole 16-token trie blocks under the
// fold-4 tokenizer, so prefix matching has something to match on.
var longPrompt = strings.Repeat("explain paged attention and kv cache reuse. ", 12)

func TestChatCompletionReachesAReplica(t *testing.T) {
	r0 := newFakeReplica(t, "r0", 0.1)
	r1 := newFakeReplica(t, "r1", 0.1)
	_, front := newTestServer(t, nil, r0, r1)

	resp := chat(t, front.URL, "hello", nil, nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if resp.Header.Get("X-Replica") == "" {
		t.Error("response did not report which replica served it")
	}
	if total := r0.calls.Load() + r1.calls.Load(); total != 1 {
		t.Errorf("replicas saw %d calls, want exactly 1", total)
	}
}

// TestPrefixRoutingSticksToTheWarmReplica is the behaviour the routing layer
// exists for: a repeated prompt should return to the replica that already has
// it in KV cache, and say how much of it matched.
func TestPrefixRoutingSticksToTheWarmReplica(t *testing.T) {
	r0 := newFakeReplica(t, "r0", 0.1)
	r1 := newFakeReplica(t, "r1", 0.1)
	_, front := newTestServer(t, func(c *config.Config) {
		c.UsePrefixRouting = true
	}, r0, r1)

	first := chat(t, front.URL, longPrompt, nil, nil)
	if got := first.Header.Get("X-Prefix-Match-Tokens"); got != "0" {
		t.Errorf("cold request reported %s matched tokens, want 0", got)
	}
	firstReplica := first.Header.Get("X-Replica")

	second := chat(t, front.URL, longPrompt, nil, nil)
	if got := second.Header.Get("X-Replica"); got != firstReplica {
		t.Errorf("repeat prompt routed to %s, want %s", got, firstReplica)
	}
	matched, _ := strconv.Atoi(second.Header.Get("X-Prefix-Match-Tokens"))
	if matched <= 0 {
		t.Errorf("repeat prompt matched %d tokens, want > 0", matched)
	}
}

func TestQuotaExhaustionReturns429(t *testing.T) {
	r0 := newFakeReplica(t, "r0", 0.1)
	_, front := newTestServer(t, func(c *config.Config) {
		c.BucketRateTokensPerS = 0 // no refill
		c.BucketBurstTokens = 1    // smaller than any real request
	}, r0)

	resp := chat(t, front.URL, "hello", map[string]string{"X-Tenant-Id": "agent"}, nil)
	if resp.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want 429", resp.StatusCode)
	}
	if resp.Header.Get("Retry-After") == "" {
		t.Error("refusal did not tell the caller when to retry")
	}
	if got := decodeReason(t, resp); got != "quota" {
		t.Errorf("reason = %q, want \"quota\"", got)
	}
	if r0.calls.Load() != 0 {
		t.Error("a request refused for quota still reached a replica")
	}
}

// TestStaleFleetShedsBeforeDispatch: with admission on and no scraper running,
// the gateway has never seen fleet state and must fail closed.
func TestStaleFleetShedsBeforeDispatch(t *testing.T) {
	r0 := newFakeReplica(t, "r0", 0.1)
	_, front := newTestServer(t, func(c *config.Config) {
		c.AdmissionEnabled = true
	}, r0)

	resp := chat(t, front.URL, "hello", nil, nil)
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503", resp.StatusCode)
	}
	if got := decodeReason(t, resp); got != "no_signal" {
		t.Errorf("reason = %q, want \"no_signal\"", got)
	}
	if r0.calls.Load() != 0 {
		t.Error("a shed request still reached a replica")
	}
}

// TestMalformedPriorityHeaderIsIgnored is a regression guard: the Python
// implementation called int() on this header directly, so a non-numeric value
// raised inside the handler and surfaced as a caller-triggered 500.
func TestMalformedPriorityHeaderIsIgnored(t *testing.T) {
	r0 := newFakeReplica(t, "r0", 0.1)
	_, front := newTestServer(t, nil, r0)

	resp := chat(t, front.URL, "hello", map[string]string{"X-Priority": "not-a-number"}, nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200 (a bad hint must not fail the request)", resp.StatusCode)
	}
}

// TestClientCannotExtendItsDeadline: a caller may ask to be shed sooner, never
// later. The Python version took deadline_ms from the body unclamped, letting a
// client opt out of the deadline check.
func TestClientCannotExtendItsDeadline(t *testing.T) {
	r0 := newFakeReplica(t, "r0", 0.1)
	srv, _ := newTestServer(t, nil, r0)

	tierDefault := srv.cfg.DeadlineFor("agent")

	greedy := &chatRequest{DeadlineMs: ptr(3_600_000)}
	if got := srv.deadlineFor(greedy, "agent"); got > tierDefault {
		t.Errorf("client stretched its deadline to %s, capped at %s", got, tierDefault)
	}
	tighter := &chatRequest{DeadlineMs: ptr(250)}
	if got := srv.deadlineFor(tighter, "agent"); got != 250*time.Millisecond {
		t.Errorf("client asked for a tighter %s deadline, got %s", 250*time.Millisecond, got)
	}
	negative := &chatRequest{DeadlineMs: ptr(-1)}
	if got := srv.deadlineFor(negative, "agent"); got != tierDefault {
		t.Errorf("negative deadline produced %s, want the tier default %s", got, tierDefault)
	}
}

func TestOversizedBodyIsRejected(t *testing.T) {
	r0 := newFakeReplica(t, "r0", 0.1)
	_, front := newTestServer(t, func(c *config.Config) { c.MaxRequestBytes = 256 }, r0)

	resp := chat(t, front.URL, strings.Repeat("x", 4096), nil, nil)
	if resp.StatusCode != http.StatusRequestEntityTooLarge {
		t.Fatalf("status = %d, want 413", resp.StatusCode)
	}
	if r0.calls.Load() != 0 {
		t.Error("an oversized request still reached a replica")
	}
}

func TestMalformedJSONIsRejected(t *testing.T) {
	r0 := newFakeReplica(t, "r0", 0.1)
	_, front := newTestServer(t, nil, r0)

	resp, err := http.Post(front.URL+"/v1/chat/completions", "application/json", strings.NewReader("{not json"))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", resp.StatusCode)
	}
}

// TestQueuedRequestsAreDispatched exercises the queue path end to end: the
// handler parks the request, the dispatch loop reserves capacity and hands back
// a replica, and the handler proxies to it.
func TestQueuedRequestsAreDispatched(t *testing.T) {
	r0 := newFakeReplica(t, "r0", 0.1)
	srv, front := newTestServer(t, func(c *config.Config) {
		c.QueueEnabled = true
		c.UsePrefixRouting = true
	}, r0)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go srv.dispatchLoop(ctx)

	resp := chat(t, front.URL, "hello from the queue", nil, nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if r0.calls.Load() != 1 {
		t.Errorf("replica saw %d calls, want 1", r0.calls.Load())
	}
	if srv.queue.Len() != 0 {
		t.Errorf("queue still holds %d items after dispatch", srv.queue.Len())
	}
}

// TestDispatchSlotsAreReturned guards the semaphore accounting: if a slot leaked
// per request, the gateway would wedge after `capacity` requests.
func TestDispatchSlotsAreReturned(t *testing.T) {
	r0 := newFakeReplica(t, "r0", 0.1)
	srv, front := newTestServer(t, func(c *config.Config) {
		c.DefaultMaxNumSeqs = 1
		c.DispatchOvershoot = 0
	}, r0)

	for i := range 5 {
		resp := chat(t, front.URL, "hello", nil, nil)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("request %d: status = %d, want 200", i, resp.StatusCode)
		}
	}
	if got := len(srv.slots); got != 0 {
		t.Errorf("%d dispatch slots still held after all requests completed", got)
	}
}

func TestUpstreamFailureIsReportedAsBadGateway(t *testing.T) {
	r0 := newFakeReplica(t, "r0", 0.1)
	r0.status.Store(http.StatusInternalServerError)
	_, front := newTestServer(t, nil, r0)

	resp := chat(t, front.URL, "hello", nil, nil)
	if resp.StatusCode != http.StatusInternalServerError {
		t.Fatalf("status = %d, want the upstream's 500 relayed", resp.StatusCode)
	}
	if got := resp.Header.Get("X-Outcome"); got != "error" {
		t.Errorf("X-Outcome = %q, want \"error\"", got)
	}
}

func TestScraperPopulatesFleetState(t *testing.T) {
	r0 := newFakeReplica(t, "r0", 0.42)
	srv, _ := newTestServer(t, nil, r0)

	scraped := scrapeOnce(t, srv)
	if !scraped {
		t.Fatal("scrape round reported failure against a healthy fake replica")
	}
	snap := srv.fleet.Snapshot(time.Now())
	if got := snap.KVUsageMax; got < 0.41 || got > 0.43 {
		t.Errorf("kv usage = %v, want ~0.42 from the scraped metrics", got)
	}
	if snap.StaleFor > time.Second {
		t.Errorf("fleet still looks stale (%s) after a successful scrape", snap.StaleFor)
	}
}

func TestHealthAndStatsEndpoints(t *testing.T) {
	r0 := newFakeReplica(t, "r0", 0.1)
	_, front := newTestServer(t, nil, r0)

	for _, path := range []string{"/health", "/_stats", "/v1/models"} {
		resp, err := http.Get(front.URL + path)
		if err != nil {
			t.Fatalf("GET %s: %v", path, err)
		}
		if resp.StatusCode != http.StatusOK {
			t.Errorf("GET %s: status = %d, want 200", path, resp.StatusCode)
		}
		var body map[string]any
		if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
			t.Errorf("GET %s: response was not JSON: %v", path, err)
		}
		resp.Body.Close()
	}
}

func TestNormalizeTenantStripsHostileInput(t *testing.T) {
	tests := []struct{ in, want string }{
		{"", "default"},
		{"agent", "agent"},
		{"agent\nX-Injected: 1", "agentX-Injected1"}, // control chars and spaces dropped
		{"!!!", "default"},
		{strings.Repeat("a", 200), strings.Repeat("a", maxTenantLen)},
	}
	for _, tt := range tests {
		if got := normalizeTenant(tt.in); got != tt.want {
			t.Errorf("normalizeTenant(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

// scrapeOnce runs a single scrape round against the fakes, standing in for the
// background scraper that Run would otherwise start.
func scrapeOnce(t *testing.T, srv *Server) bool {
	t.Helper()
	sc := scrape.NewScraper(srv.fleet, srv.cfg.ScrapeInterval, slog.New(slog.NewTextHandler(io.Discard, nil)))
	return sc.Once(context.Background())
}

func decodeReason(t *testing.T, resp *http.Response) string {
	t.Helper()
	var body struct {
		Error struct {
			Reason string `json:"reason"`
		} `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode refusal body: %v", err)
	}
	return body.Error.Reason
}

func ptr[T any](v T) *T { return &v }
