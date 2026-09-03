// Package state holds live fleet telemetry: what each vLLM replica is doing
// right now, and the derived rates the gateway uses to predict latency.
//
// Concurrency note, and the sharpest difference from the Python original:
// there, FleetState was a plain dataclass mutated by the scrape task and read
// by request handlers with no synchronisation at all. That is safe in Python
// only because asyncio runs one coroutine at a time on a single thread, so a
// handler can never observe a half-written update. Go actually runs these
// concurrently, so the same design would be a data race. Reads therefore go
// through Snapshot, which copies under an RLock; writers hold the write lock.
package state

import (
	"sync"
	"sync/atomic"
	"time"
)

// Replica is one upstream vLLM process.
//
// InFlight and Completed are atomics rather than mutex-guarded fields because
// they are touched on every request dispatch and completion, while the rest of
// the struct is written only by the metrics scraper.
type Replica struct {
	ID               string
	URL              string
	MaxNumSeqs       int
	KVCapacityTokens int

	InFlight  atomic.Int64
	Completed atomic.Int64

	// Fields below are written by the scraper under Fleet's write lock.
	Waiting       int
	Running       int
	KVUsage       float64
	PrefixQueries int
	PrefixHits    int
	Preemptions   int
	TTFTSum       float64
	TTFTCount     float64
	ITLSum        float64
	ITLCount      float64
	QueueTimeSum  float64
	QueueTimeCnt  float64
}

// Fleet aggregates every replica plus fleet-wide derived rates.
type Fleet struct {
	mu           sync.RWMutex
	replicas     []*Replica
	scrapeOKAt   time.Time
	prefillRate  float64       // tokens/sec, estimated from observed TTFT
	interTokenLa time.Duration // estimated from observed ITL
	queueWait    time.Duration // estimated wait implied by current queue depth
}

// NewFleet builds a Fleet from replica URLs. Seed rates come from config so
// admission control has something usable before the first scrape lands.
func NewFleet(urls []string, maxNumSeqs, kvCapacity int, seedPrefill float64, seedITL time.Duration) *Fleet {
	f := &Fleet{prefillRate: seedPrefill, interTokenLa: seedITL}
	f.replicas = make([]*Replica, 0, len(urls))
	for i, u := range urls {
		f.replicas = append(f.replicas, &Replica{
			ID:               "r" + itoa(i),
			URL:              u,
			MaxNumSeqs:       maxNumSeqs,
			KVCapacityTokens: kvCapacity,
		})
	}
	return f
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b [20]byte
	pos := len(b)
	for i > 0 {
		pos--
		b[pos] = byte('0' + i%10)
		i /= 10
	}
	return string(b[pos:])
}

// Replicas exposes the replica handles. The slice itself is never mutated after
// construction, so handing it out is safe; the atomics on each Replica are the
// only fields callers may touch without a lock.
func (f *Fleet) Replicas() []*Replica { return f.replicas }

// ByID returns a replica handle, or nil.
func (f *Fleet) ByID(id string) *Replica {
	for _, r := range f.replicas {
		if r.ID == id {
			return r
		}
	}
	return nil
}

// Update applies scraped metrics to a replica under the write lock.
func (f *Fleet) Update(id string, apply func(*Replica)) {
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, r := range f.replicas {
		if r.ID == id {
			apply(r)
			return
		}
	}
}

// MarkScraped records a successful scrape round and refreshes derived rates
// from the cumulative TTFT/ITL histograms vLLM exposes.
func (f *Fleet) MarkScraped(now time.Time) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.scrapeOKAt = now

	var ttftSum, ttftCount, itlSum, itlCount float64
	for _, r := range f.replicas {
		ttftSum += r.TTFTSum
		ttftCount += r.TTFTCount
		itlSum += r.ITLSum
		itlCount += r.ITLCount
	}
	if ttftCount > 0 && ttftSum > 0 {
		// Mean TTFT stands in for prefill throughput against a nominal
		// 200-token prompt: tokens/sec = 200 / mean_ttft_seconds.
		meanTTFT := ttftSum / ttftCount
		if rate := 200.0 / meanTTFT; rate > 1 {
			f.prefillRate = rate
		}
	}
	if itlCount > 0 {
		f.interTokenLa = time.Duration(itlSum / itlCount * float64(time.Second))
	}
}

// SetQueueWait records the wait implied by the current gateway queue depth.
func (f *Fleet) SetQueueWait(depth int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	slots := 0
	for _, r := range f.replicas {
		slots += r.MaxNumSeqs
	}
	if slots < 1 {
		slots = 1
	}
	// Nominal service time for one request: prefill of ~400 tokens plus ~200
	// decoded tokens. Crude on purpose; it only needs to be monotonic in depth.
	svc := 400.0/max64(f.prefillRate, 1) + 200*f.interTokenLa.Seconds()
	f.queueWait = time.Duration(float64(depth) * svc / float64(slots) * float64(time.Second))
}

func max64(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

// ReplicaSnapshot is an immutable copy of one replica's state.
type ReplicaSnapshot struct {
	ID         string
	URL        string
	Waiting    int
	Running    int
	KVUsage    float64
	MaxNumSeqs int
	InFlight   int64
	Completed  int64
}

// Load is the router's load signal: queued plus running work, normalised by the
// replica's own concurrency limit.
func (r ReplicaSnapshot) Load() float64 {
	den := r.MaxNumSeqs
	if den < 1 {
		den = 1
	}
	return float64(r.Waiting+r.Running) / float64(den)
}

// Snapshot is a consistent point-in-time view of the whole fleet. Admission and
// routing are pure functions of this value, which makes both trivially testable
// and removes any chance of them observing a torn read.
type Snapshot struct {
	Replicas          []ReplicaSnapshot
	StaleFor          time.Duration
	KVUsageMax        float64
	WaitingTotal      int
	RunningTotal      int
	InFlightTotal     int64
	HeadroomTokens    int
	QueueWait         time.Duration
	PrefillTokensPerS float64
	InterTokenLatency time.Duration
}

// Snapshot copies current fleet state under a read lock.
func (f *Fleet) Snapshot(now time.Time) Snapshot {
	f.mu.RLock()
	defer f.mu.RUnlock()

	s := Snapshot{
		Replicas:          make([]ReplicaSnapshot, 0, len(f.replicas)),
		QueueWait:         f.queueWait,
		PrefillTokensPerS: f.prefillRate,
		InterTokenLatency: f.interTokenLa,
	}
	// A zero scrapeOKAt means "never scraped successfully". Report it as
	// effectively infinite staleness so admission control sheds rather than
	// trusting a fleet it has never seen.
	if f.scrapeOKAt.IsZero() {
		s.StaleFor = time.Duration(1<<62 - 1)
	} else {
		s.StaleFor = now.Sub(f.scrapeOKAt)
	}
	for _, r := range f.replicas {
		rs := ReplicaSnapshot{
			ID:         r.ID,
			URL:        r.URL,
			Waiting:    r.Waiting,
			Running:    r.Running,
			KVUsage:    r.KVUsage,
			MaxNumSeqs: r.MaxNumSeqs,
			InFlight:   r.InFlight.Load(),
			Completed:  r.Completed.Load(),
		}
		s.Replicas = append(s.Replicas, rs)
		s.WaitingTotal += rs.Waiting
		s.RunningTotal += rs.Running
		s.InFlightTotal += rs.InFlight
		if rs.KVUsage > s.KVUsageMax {
			s.KVUsageMax = rs.KVUsage
		}
		s.HeadroomTokens += int((1.0 - rs.KVUsage) * float64(r.KVCapacityTokens))
	}
	return s
}

// Capacity is the total number of requests the fleet will accept in flight,
// including the configured overshoot.
func (s Snapshot) Capacity(overshoot int) int64 {
	var c int64
	for _, r := range s.Replicas {
		c += int64(r.MaxNumSeqs)
	}
	c += int64(overshoot)
	if c < 1 {
		c = 1
	}
	return c
}

// EstimateTotal predicts end-to-end latency for a request of the given shape:
// queue wait, then prefill of nIn tokens, then nOut decode steps.
func (s Snapshot) EstimateTotal(nIn, nOut int) time.Duration {
	prefill := float64(nIn) / max64(s.PrefillTokensPerS, 1)
	decode := float64(nOut) * s.InterTokenLatency.Seconds()
	return time.Duration((prefill+decode)*float64(time.Second)) + s.QueueWait
}

// Request is the shape of an inbound request as the policy layers see it. It
// deliberately carries no HTTP types so admission, routing and queueing stay
// testable without spinning up a server.
type Request struct {
	Tenant   string
	NIn      int
	NOut     int
	TokenIDs []uint32
	Deadline time.Duration // latency objective, relative
	Priority *int          // optional client hint, forwarded to vLLM
}
