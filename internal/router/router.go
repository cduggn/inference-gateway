// Package router picks which replica serves a request.
//
// Two signals, traded off against each other: how much of this prompt the
// replica has already got in KV cache (prefix affinity, worth real TTFT), and
// how loaded it is (worth real queueing delay). Sending every request to the
// warmest replica would hot-spot it; ignoring warmth throws away prefill work
// the fleet already paid for.
package router

import (
	"math/rand/v2"
	"sync/atomic"

	"github.com/cduggn/inference-gateway/internal/config"
	"github.com/cduggn/inference-gateway/internal/state"
	"github.com/cduggn/inference-gateway/internal/trie"
)

// Choice is a routing decision.
type Choice struct {
	ReplicaID   string
	MatchTokens int // leading tokens believed to be cache-warm on that replica
	Score       float64
}

// Router holds the prefix trie and the round-robin cursor used to break ties.
type Router struct {
	trie *trie.Trie
	rr   atomic.Uint64
}

// New returns a Router over the given trie.
func New(t *trie.Trie) *Router { return &Router{trie: t} }

// Trie exposes the underlying trie, for pruning and stats.
func (r *Router) Trie() *trie.Trie { return r.trie }

// Score is the routing objective: prefix match earns credit, load costs.
//
// With the shipped weights (WPrefix 1, WLoad 64) one cache-warm 16-token block
// is worth 16 points while a fully saturated replica costs 64, so affinity only
// decides between replicas of comparable load -- a busy replica loses even with
// a perfect prefix hit. That ordering is the point: TTFT saved by cache reuse
// is smaller than TTFT lost to queueing.
func Score(cfg *config.Config, r state.ReplicaSnapshot, matchTokens int) float64 {
	load := min(r.Load(), cfg.LoadCeiling)
	return cfg.WPrefix*float64(matchTokens) - cfg.WLoad*load
}

// Pick chooses a replica for req and records the prompt against it, so a
// follow-up turn in the same conversation is drawn to the same replica.
//
// Returns ok=false only when there are no replicas at all; the caller decides
// whether that is a 503 or a startup bug.
func (r *Router) Pick(cfg *config.Config, replicas []state.ReplicaSnapshot, req state.Request) (Choice, bool) {
	if len(replicas) == 0 {
		return Choice{}, false
	}

	candidates := replicas
	if cfg.UseP2C && len(replicas) >= 2 {
		// Power of two choices: sample two at random and take the better one.
		// Cheaper than scoring the whole fleet and, more importantly, it stops
		// every request in a burst stampeding onto whichever single replica
		// looked best at the instant the burst began.
		i := rand.IntN(len(replicas))
		j := rand.IntN(len(replicas) - 1)
		if j >= i {
			j++
		}
		candidates = []state.ReplicaSnapshot{replicas[i], replicas[j]}
	}

	best := Choice{Score: -1e18}
	var tied []Choice
	for _, rep := range candidates {
		match := r.trie.Match(rep.ID, req.TokenIDs)
		score := 0.0
		if cfg.UsePrefixRouting {
			score = Score(cfg, rep, match)
		}
		c := Choice{ReplicaID: rep.ID, MatchTokens: match, Score: score}
		switch {
		case score > best.Score:
			best = c
			tied = tied[:0]
			tied = append(tied, c)
		case score == best.Score:
			tied = append(tied, c)
		}
	}

	// Rotate through equally-scored replicas. With prefix routing off every
	// score is 0, so this degenerates to plain round-robin -- which is exactly
	// what the baseline preset is meant to be.
	chosen := tied[int(r.rr.Add(1)-1)%len(tied)]
	r.trie.Insert(chosen.ReplicaID, req.TokenIDs)
	return chosen, true
}
