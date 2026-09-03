package trie

import (
	"testing"
	"time"
)

func seq(n int) []uint32 {
	out := make([]uint32, n)
	for i := range out {
		out[i] = uint32(i)
	}
	return out
}

func TestBlockHashesDropsPartialBlock(t *testing.T) {
	// 40 tokens at block size 16 is two whole blocks; the trailing 8 tokens are
	// not a reusable cache block and must not produce a hash.
	if got := len(BlockHashes(seq(40), 16)); got != 2 {
		t.Fatalf("blocks = %d, want 2", got)
	}
	if got := BlockHashes(seq(15), 16); got != nil {
		t.Fatalf("short input produced %v, want nil", got)
	}
}

func TestMatchCountsOnlySharedPrefix(t *testing.T) {
	tr := New(time.Minute, 16)
	tr.Insert("r0", seq(64))

	// Identical prompt: every block matches.
	if got, want := tr.Match("r0", seq(64)), 64; got != want {
		t.Errorf("identical prompt matched %d tokens, want %d", got, want)
	}
	// Longer prompt sharing a 64-token prefix: only the shared part matches.
	if got, want := tr.Match("r0", seq(96)), 64; got != want {
		t.Errorf("extended prompt matched %d tokens, want %d", got, want)
	}
	// A different replica has cached nothing.
	if got := tr.Match("r1", seq(64)); got != 0 {
		t.Errorf("unrelated replica matched %d tokens, want 0", got)
	}
}

func TestMatchDivergesAtFirstDifferentBlock(t *testing.T) {
	tr := New(time.Minute, 16)
	tr.Insert("r0", seq(64))

	// Same first 32 tokens, then different content.
	diverged := append(seq(32), make([]uint32, 32)...)
	for i := 32; i < 64; i++ {
		diverged[i] = 9999
	}
	if got, want := tr.Match("r0", diverged), 32; got != want {
		t.Errorf("matched %d tokens past divergence, want %d", got, want)
	}
}

func TestExpiredEntriesStopMatching(t *testing.T) {
	now := time.Now()
	tr := New(time.Minute, 16)
	tr.now = func() time.Time { return now }
	tr.Insert("r0", seq(32))

	now = now.Add(90 * time.Second) // past the TTL
	if got := tr.Match("r0", seq(32)); got != 0 {
		t.Fatalf("expired prefix matched %d tokens, want 0", got)
	}
}

func TestPruneReclaimsExpiredNodes(t *testing.T) {
	now := time.Now()
	tr := New(time.Minute, 16)
	tr.now = func() time.Time { return now }
	tr.Insert("r0", seq(160))

	before := tr.Size()
	if before < 10 {
		t.Fatalf("expected a populated trie, got %d nodes", before)
	}
	now = now.Add(2 * time.Minute)
	if reclaimed := tr.Prune(); reclaimed == 0 {
		t.Fatal("prune reclaimed nothing after TTL elapsed")
	}
	if after := tr.Size(); after >= before {
		t.Fatalf("size %d after prune, was %d before", after, before)
	}
}
