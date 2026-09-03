package bucket

import (
	"testing"
	"time"
)

func TestBucketSpendsBurstThenRefills(t *testing.T) {
	now := time.Now()
	b := newTokenBucketAt(10, 100, func() time.Time { return now }) // 10 tokens/sec, burst 100

	if !b.Allow(100) {
		t.Fatal("full bucket refused a request within burst")
	}
	if b.Allow(1) {
		t.Fatal("empty bucket allowed a request")
	}
	now = now.Add(time.Second) // +10 tokens
	if !b.Allow(10) {
		t.Fatal("refilled bucket refused a request it should cover")
	}
	if b.Allow(1) {
		t.Fatal("bucket allowed more than it had refilled")
	}
}

// TestUnknownTenantsShareOneBucket pins the fix for the quota-bypass the Python
// implementation had: there, an unrecognised X-Tenant-Id minted a brand new
// full-burst bucket, so rotating the header both dodged quota entirely and grew
// the map without bound.
func TestUnknownTenantsShareOneBucket(t *testing.T) {
	b := New(0, 10, []string{"agent"}) // no refill, burst of 10

	if !b.Allow("attacker-1", 10) {
		t.Fatal("first unknown tenant should get the shared burst")
	}
	// A different unknown tenant must draw on the same exhausted bucket.
	if b.Allow("attacker-2", 1) {
		t.Fatal("rotating the tenant id bypassed quota")
	}
	// A configured tenant keeps its own budget and is unaffected.
	if !b.Allow("agent", 10) {
		t.Fatal("known tenant was starved by unknown-tenant traffic")
	}
}

func TestKnownTenantsAreIsolated(t *testing.T) {
	b := New(0, 5, []string{"chat", "agent"})
	if !b.Allow("chat", 5) {
		t.Fatal("chat should spend its own burst")
	}
	if b.Allow("chat", 1) {
		t.Fatal("chat exceeded its burst")
	}
	if !b.Allow("agent", 5) {
		t.Fatal("agent's budget was consumed by chat")
	}
}
