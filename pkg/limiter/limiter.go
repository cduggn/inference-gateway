// Package limiter is the client-side request limiter used by the agent driver.
//
// It is deliberately separate from the gateway's own per-tenant quota: this one
// counts requests and lives in the caller's process, so an agent throttles
// itself before a request ever leaves. The two are independent gates, not a
// shared budget -- a request can pass this and still be refused for quota at
// the gateway.
//
// The Go standard ecosystem does have golang.org/x/time/rate, which is a
// better-tested version of exactly this. It is hand-rolled here only to keep
// the mapping to the Python lab's limiter.py one-to-one; production code should
// prefer x/time/rate.
package limiter

import (
	"context"
	"errors"
	"sync"
	"time"
)

// ErrRateLimited is returned when a caller declines to wait for capacity.
var ErrRateLimited = errors.New("client rate limit exceeded")

// Limiter is a token bucket over request count.
type Limiter struct {
	mu     sync.Mutex
	rps    float64
	burst  float64
	tokens float64
	last   time.Time
}

// New returns a limiter allowing rps requests per second with the given burst.
func New(rps, burst float64) *Limiter {
	return &Limiter{rps: rps, burst: burst, tokens: burst, last: time.Now()}
}

// RPS reports the configured steady-state rate.
func (l *Limiter) RPS() float64 { return l.rps }

// Burst reports the configured burst ceiling.
func (l *Limiter) Burst() float64 { return l.burst }

// TryAcquire takes one token if available, without blocking.
func (l *Limiter) TryAcquire() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	now := time.Now()
	l.tokens = min(l.burst, l.tokens+now.Sub(l.last).Seconds()*l.rps)
	l.last = now
	if l.tokens < 1 {
		return false
	}
	l.tokens--
	return true
}

// Wait blocks until a token is available or ctx is done. Unlike the Python
// version's busy sleep loop, this respects cancellation, so a shutdown does not
// have to wait out the backoff.
func (l *Limiter) Wait(ctx context.Context) error {
	const poll = 20 * time.Millisecond
	t := time.NewTicker(poll)
	defer t.Stop()
	for {
		if l.TryAcquire() {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-t.C:
		}
	}
}
