// Package trie tracks, per replica, which prompt prefixes that replica has
// recently served -- the gateway's model of where vLLM's KV cache is warm.
package trie

import (
	"encoding/binary"
	"sync"
	"time"

	"golang.org/x/crypto/blake2s"
)

// BlockHashes fingerprints a token sequence in fixed-size blocks, mirroring how
// vLLM's paged KV cache is addressed: a prefix is reusable only in whole
// blocks, so matching at finer granularity would claim hits that do not exist.
// A trailing partial block is dropped for the same reason.
//
// Digest note: Python used blake2s with digest_size=8. Go's x/crypto/blake2s
// exposes only 256-bit output, so this truncates Sum256. BLAKE2 folds the
// output length into its IV, so these are NOT the same 64 bits Python produces
// -- which is fine, because the trie is per-process state and the hash is only
// ever compared against other hashes from this same build.
func BlockHashes(tokenIDs []uint32, blockSize int) []uint64 {
	if blockSize <= 0 || len(tokenIDs) < blockSize {
		return nil
	}
	n := len(tokenIDs) - (len(tokenIDs) % blockSize)
	out := make([]uint64, 0, n/blockSize)
	raw := make([]byte, blockSize*4)
	for i := 0; i < n; i += blockSize {
		for j, tok := range tokenIDs[i : i+blockSize] {
			binary.LittleEndian.PutUint32(raw[j*4:], tok)
		}
		sum := blake2s.Sum256(raw)
		out = append(out, binary.LittleEndian.Uint64(sum[:8]))
	}
	return out
}

type node struct {
	children  map[uint64]*node
	expiresAt time.Time
}

// Trie maps replica id -> tree of block-hash paths.
//
// Guarded by a mutex: unlike the Python original, which was implicitly
// serialised by the asyncio event loop, this is read and written from every
// request goroutine concurrently.
type Trie struct {
	mu        sync.Mutex
	roots     map[string]*node
	ttl       time.Duration
	blockSize int
	now       func() time.Time // injectable for tests
}

// New returns an empty trie. ttl is how long a prefix is assumed to stay warm
// in the replica's KV cache; blockSize should match vLLM's block size.
func New(ttl time.Duration, blockSize int) *Trie {
	return &Trie{
		roots:     make(map[string]*node),
		ttl:       ttl,
		blockSize: blockSize,
		now:       time.Now,
	}
}

// Insert records that replica served this token sequence, refreshing the TTL
// along the whole path.
func (t *Trie) Insert(replica string, tokenIDs []uint32) {
	hashes := BlockHashes(tokenIDs, t.blockSize)
	if len(hashes) == 0 {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	now := t.now()
	exp := now.Add(t.ttl)
	root, ok := t.roots[replica]
	if !ok {
		root = &node{children: make(map[uint64]*node)}
		t.roots[replica] = root
	}
	root.expiresAt = exp
	cur := root
	for _, h := range hashes {
		next, ok := cur.children[h]
		if !ok {
			next = &node{children: make(map[uint64]*node)}
			cur.children[h] = next
		}
		next.expiresAt = exp
		cur = next
	}
}

// Match returns how many leading tokens of tokenIDs this replica is believed to
// have cached. Walking stops at the first block that is absent or expired,
// since KV reuse is only valid for an unbroken prefix.
func (t *Trie) Match(replica string, tokenIDs []uint32) int {
	hashes := BlockHashes(tokenIDs, t.blockSize)
	if len(hashes) == 0 {
		return 0
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	cur, ok := t.roots[replica]
	if !ok {
		return 0
	}
	now := t.now()
	matched := 0
	for _, h := range hashes {
		next, ok := cur.children[h]
		if !ok || next.expiresAt.Before(now) {
			break
		}
		matched += t.blockSize
		cur = next
	}
	return matched
}

// Prune drops expired subtrees and reports how many nodes were reclaimed.
//
// The Python original had no equivalent: expired nodes stopped matching but
// stayed resident forever, so memory grew with the number of distinct prefixes
// the gateway had ever seen. Run this periodically.
func (t *Trie) Prune() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	now := t.now()
	removed := 0
	for id, root := range t.roots {
		removed += pruneNode(root, now)
		if len(root.children) == 0 && root.expiresAt.Before(now) {
			delete(t.roots, id)
			removed++
		}
	}
	return removed
}

func pruneNode(n *node, now time.Time) int {
	removed := 0
	for h, child := range n.children {
		removed += pruneNode(child, now)
		// A node is only reclaimable once it is expired AND has no live
		// descendants, otherwise a still-warm deeper prefix becomes unreachable.
		if child.expiresAt.Before(now) && len(child.children) == 0 {
			delete(n.children, h)
			removed++
		}
	}
	return removed
}

// Size reports the number of live nodes, for tests and /_stats.
func (t *Trie) Size() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	total := 0
	for _, root := range t.roots {
		total += countNode(root)
	}
	return total
}

func countNode(n *node) int {
	total := 1
	for _, c := range n.children {
		total += countNode(c)
	}
	return total
}
