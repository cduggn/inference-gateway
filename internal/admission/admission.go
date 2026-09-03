// Package admission decides whether the fleet can take a request at all.
//
// This is load shedding, not rate limiting: quota asks "is this tenant over
// budget", admission asks "can the fleet meet this request's objective right
// now". A shed request is refused in microseconds instead of queueing behind
// work that will blow its deadline anyway -- cheaper for the caller and it
// stops the fleet spending KV cache on requests nobody will wait for.
package admission

import (
	"time"

	"github.com/cduggn/inference-gateway/internal/config"
	"github.com/cduggn/inference-gateway/internal/refuse"
	"github.com/cduggn/inference-gateway/internal/state"
)

// Decide reports the reason to shed, or "" to admit.
//
// A pure function of a fleet snapshot and the request shape: no locks, no I/O,
// no clock reads beyond what the caller passed in. That makes every branch
// below directly testable, which is why the checks live here rather than inline
// in the HTTP handler as they did in the Python original.
//
// Order matters and is deliberate, cheapest and most decisive first:
//  1. no_signal            - we are blind; refusing beats guessing
//  2. kv_pressure          - the cache is nearly full, admitting causes preemption
//  3. queue_depth          - vLLM's own queues are backing up
//  4. no_headroom          - this specific request does not fit
//  5. deadline_unmeetable  - it fits, but not in time to be useful
func Decide(cfg *config.Config, snap state.Snapshot, req state.Request) refuse.Reason {
	// Stale metrics mean every check below is reasoning about a fleet we can no
	// longer see. Fail closed.
	if snap.StaleFor > cfg.StaleCeiling {
		return refuse.NoSignal
	}
	if snap.KVUsageMax > cfg.KVCeiling {
		return refuse.KVPressure
	}
	replicas := len(snap.Replicas)
	if replicas < 1 {
		replicas = 1
	}
	if snap.WaitingTotal > cfg.WaitingCeilingPerReplica*replicas {
		return refuse.QueueDepth
	}
	// Headroom is fleet-wide free KV measured in tokens. A request needs room
	// for its prompt and everything it will generate.
	if snap.HeadroomTokens < req.NIn+req.NOut {
		return refuse.NoHeadroom
	}
	if snap.EstimateTotal(req.NIn, req.NOut) > req.Deadline {
		return refuse.DeadlineUnmeetable
	}
	return ""
}

// Estimate exposes the latency prediction used by the deadline check, so
// refusal responses can tell the caller what the gateway thought would happen.
func Estimate(snap state.Snapshot, req state.Request) time.Duration {
	return snap.EstimateTotal(req.NIn, req.NOut)
}
