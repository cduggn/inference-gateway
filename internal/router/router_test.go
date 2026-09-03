package router

import (
	"testing"
	"time"

	"github.com/cduggn/inference-gateway/internal/config"
	"github.com/cduggn/inference-gateway/internal/state"
	"github.com/cduggn/inference-gateway/internal/trie"
)

func tokens(n int) []uint32 {
	out := make([]uint32, n)
	for i := range out {
		out[i] = uint32(i + 1)
	}
	return out
}

func replicas(loads ...float64) []state.ReplicaSnapshot {
	out := make([]state.ReplicaSnapshot, 0, len(loads))
	for i, l := range loads {
		out = append(out, state.ReplicaSnapshot{
			ID:         "r" + string(rune('0'+i)),
			MaxNumSeqs: 8,
			Running:    int(l * 8),
		})
	}
	return out
}

func TestPrefixAffinityWinsBetweenEquallyLoadedReplicas(t *testing.T) {
	cfg := config.Default()
	cfg.UsePrefixRouting = true

	r := New(trie.New(time.Minute, cfg.BlockSize))
	req := state.Request{TokenIDs: tokens(64)}

	// Warm r1 by routing the same prompt there once.
	r.Trie().Insert("r1", req.TokenIDs)

	got, ok := r.Pick(&cfg, replicas(0, 0), req)
	if !ok {
		t.Fatal("Pick reported no replicas")
	}
	if got.ReplicaID != "r1" {
		t.Errorf("routed to %s, want r1 (the cache-warm replica)", got.ReplicaID)
	}
	if got.MatchTokens != 64 {
		t.Errorf("MatchTokens = %d, want 64", got.MatchTokens)
	}
}

// TestLoadOutweighsPrefixAffinity encodes the deliberate weighting: TTFT saved
// by a cache hit is smaller than TTFT lost to queueing, so a saturated replica
// must lose even holding a perfect prefix.
func TestLoadOutweighsPrefixAffinity(t *testing.T) {
	cfg := config.Default()
	cfg.UsePrefixRouting = true

	r := New(trie.New(time.Minute, cfg.BlockSize))
	req := state.Request{TokenIDs: tokens(64)}
	r.Trie().Insert("r0", req.TokenIDs) // r0 is warm but about to be busy

	got, _ := r.Pick(&cfg, replicas(2.0, 0), req)
	if got.ReplicaID != "r1" {
		t.Errorf("routed to %s, want r1 (idle) despite r0 being cache-warm", got.ReplicaID)
	}
}

func TestBaselineIsRoundRobin(t *testing.T) {
	cfg := config.Default()
	cfg.UsePrefixRouting = false // baseline preset

	r := New(trie.New(time.Minute, cfg.BlockSize))
	req := state.Request{TokenIDs: tokens(64)}

	seen := map[string]int{}
	for range 4 {
		got, _ := r.Pick(&cfg, replicas(0, 0), req)
		seen[got.ReplicaID]++
	}
	if seen["r0"] != 2 || seen["r1"] != 2 {
		t.Errorf("distribution = %v, want an even 2/2 split", seen)
	}
}

func TestPickRecordsPromptAgainstChosenReplica(t *testing.T) {
	cfg := config.Default()
	cfg.UsePrefixRouting = true

	r := New(trie.New(time.Minute, cfg.BlockSize))
	req := state.Request{TokenIDs: tokens(64)}

	first, _ := r.Pick(&cfg, replicas(0, 0), req)
	if first.MatchTokens != 0 {
		t.Fatalf("cold trie reported %d matched tokens", first.MatchTokens)
	}
	// The second identical turn should stick to the replica that served the
	// first -- this is the behaviour the whole routing layer exists to produce.
	second, _ := r.Pick(&cfg, replicas(0, 0), req)
	if second.ReplicaID != first.ReplicaID {
		t.Errorf("second turn routed to %s, want %s", second.ReplicaID, first.ReplicaID)
	}
	if second.MatchTokens != 64 {
		t.Errorf("second turn matched %d tokens, want 64", second.MatchTokens)
	}
}

func TestPickWithNoReplicas(t *testing.T) {
	cfg := config.Default()
	r := New(trie.New(time.Minute, cfg.BlockSize))
	if _, ok := r.Pick(&cfg, nil, state.Request{}); ok {
		t.Fatal("Pick returned ok with an empty fleet")
	}
}

func TestP2CSamplesTwoDistinctReplicas(t *testing.T) {
	cfg := config.Default()
	cfg.UsePrefixRouting = true
	cfg.UseP2C = true

	r := New(trie.New(time.Minute, cfg.BlockSize))
	req := state.Request{TokenIDs: tokens(32)}

	// With four replicas and one heavily loaded, P2C should essentially never
	// settle on the busiest one across many draws.
	busy := 0
	for range 200 {
		got, _ := r.Pick(&cfg, replicas(2.0, 0, 0, 0), req)
		if got.ReplicaID == "r0" {
			busy++
		}
	}
	if busy > 20 {
		t.Errorf("busiest replica chosen %d/200 times, expected it to lose nearly always", busy)
	}
}
