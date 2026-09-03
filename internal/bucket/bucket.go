// Package bucket implements per-tenant token-bucket quota.
package bucket

import (
	"sync"
	"time"
)

// TokenBucket is a classic token bucket refilled lazily on each call rather
// than by a background ticker, so an idle bucket costs nothing.
type TokenBucket struct {
	mu     sync.Mutex
	rate   float64 // tokens added per second
	burst  float64 // ceiling
	tokens float64
	last   time.Time
	now    func() time.Time // injectable for tests
}

// NewTokenBucket returns a bucket that starts full.
func NewTokenBucket(rate, burst float64) *TokenBucket {
	return newTokenBucketAt(rate, burst, time.Now)
}

// newTokenBucketAt pins the clock. Refill is a function of elapsed time, so the
// initial timestamp and every later reading have to come from the same clock or
// the first call sees a nonsense interval.
func newTokenBucketAt(rate, burst float64, clock func() time.Time) *TokenBucket {
	return &TokenBucket{rate: rate, burst: burst, tokens: burst, last: clock(), now: clock}
}

// Allow debits n tokens if available, reporting whether the request may proceed.
func (b *TokenBucket) Allow(n float64) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	now := b.now()
	b.tokens = min(b.burst, b.tokens+now.Sub(b.last).Seconds()*b.rate)
	b.last = now
	if b.tokens < n {
		return false
	}
	b.tokens -= n
	return true
}

// Buckets holds one bucket per known tenant.
//
// SECURITY FIX relative to the Python original. There, Buckets.allow created a
// fresh full-burst TokenBucket for any unseen X-Tenant-Id header value. Two
// consequences: (1) the map grew without bound, one entry per distinct header
// an attacker cared to send, and (2) quota was trivially bypassed by rotating
// the tenant id, since each new identity arrived with a full burst allowance.
//
// Here the tenant set is fixed at construction from configuration. Anything not
// on that list shares a single bucket, so unknown callers are rate-limited in
// aggregate and the map size is bounded by config, not by traffic.
type Buckets struct {
	known  map[string]*TokenBucket
	shared *TokenBucket
}

// New returns quota state for the configured tenants. Every tenant in known
// gets its own budget; all other callers share one.
func New(rate, burst float64, known []string) *Buckets {
	b := &Buckets{
		known:  make(map[string]*TokenBucket, len(known)),
		shared: NewTokenBucket(rate, burst),
	}
	for _, t := range known {
		b.known[t] = NewTokenBucket(rate, burst)
	}
	return b
}

// Allow debits n tokens from the tenant's bucket, or from the shared bucket if
// the tenant is not one the gateway was configured to know about.
func (b *Buckets) Allow(tenant string, n float64) bool {
	if tb, ok := b.known[tenant]; ok {
		return tb.Allow(n)
	}
	return b.shared.Allow(n)
}
