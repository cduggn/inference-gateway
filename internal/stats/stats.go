// Package stats is a fixed-size in-memory event ring for the request lifecycle.
//
// The Python version emitted arbitrary **kwargs into a deque, which is flexible
// but means no consumer can rely on a field existing. Here an Event is a struct:
// the /_stats consumer and the tests both get a fixed schema, and adding a field
// is a compile-time decision rather than a runtime surprise.
package stats

import (
	"sync"
	"time"
)

// Kind names a point in the request lifecycle.
type Kind string

const (
	Recv       Kind = "recv"
	Admitted   Kind = "admitted"
	Enqueued   Kind = "enqueued"
	Dequeued   Kind = "dequeued"
	Dispatched Kind = "dispatched"
	Completed  Kind = "completed"
	Refused    Kind = "refused"
	Expired    Kind = "expired"
)

// Event is one lifecycle record. Zero values mean "not applicable at this
// stage" rather than "zero" -- a recv event has no TTFT, for instance.
type Event struct {
	TS      time.Time `json:"ts"`
	Kind    Kind      `json:"event"`
	Tenant  string    `json:"tenant,omitempty"`
	Reason  string    `json:"reason,omitempty"`
	Outcome string    `json:"outcome,omitempty"`
	Replica string    `json:"replica,omitempty"`

	NIn  int `json:"n_in,omitempty"`
	NOut int `json:"n_out,omitempty"`

	TTFTMs            float64 `json:"ttft_ms,omitempty"`
	TotalMs           float64 `json:"total_ms,omitempty"`
	EstimatedTotalMs  float64 `json:"estimated_total_ms,omitempty"`
	DeadlineMs        float64 `json:"deadline_ms,omitempty"`
	PrefixMatchTokens int     `json:"prefix_match_tokens,omitempty"`
	QueueDepth        int     `json:"queue_depth,omitempty"`
}

// Ring is a bounded, concurrency-safe event buffer. Go has no equivalent of
// Python's collections.deque(maxlen=N), so the wrap-around is hand-rolled over
// a preallocated slice -- which also means Emit never allocates.
type Ring struct {
	mu    sync.Mutex
	buf   []Event
	next  int
	count int
}

// NewRing returns a ring holding at most size events.
func NewRing(size int) *Ring {
	if size < 1 {
		size = 1
	}
	return &Ring{buf: make([]Event, size)}
}

// Emit appends an event, evicting the oldest once the ring is full. The
// timestamp is stamped here so callers cannot forget it.
func (r *Ring) Emit(e Event) {
	if e.TS.IsZero() {
		e.TS = time.Now()
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.buf[r.next] = e
	r.next = (r.next + 1) % len(r.buf)
	if r.count < len(r.buf) {
		r.count++
	}
}

// Snapshot returns the buffered events, oldest first.
func (r *Ring) Snapshot() []Event {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]Event, 0, r.count)
	start := (r.next - r.count + len(r.buf)) % len(r.buf)
	for i := 0; i < r.count; i++ {
		out = append(out, r.buf[(start+i)%len(r.buf)])
	}
	return out
}

// Len reports how many events are currently buffered.
func (r *Ring) Len() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.count
}
