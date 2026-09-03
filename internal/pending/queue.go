// Package pending is the gateway's bounded admission queue.
//
// It sits between "we accepted this request" and "a replica has capacity for
// it". Bounded on purpose: an unbounded queue converts an overload into
// unbounded latency, which is worse than a fast refusal, so Put fails with
// queue_full rather than growing.
package pending

import (
	"context"
	"slices"
	"sync"
	"time"

	"github.com/cduggn/inference-gateway/internal/config"
	"github.com/cduggn/inference-gateway/internal/refuse"
	"github.com/cduggn/inference-gateway/internal/state"
)

// Result is what the dispatcher hands back to a waiting request.
type Result struct {
	ReplicaID   string
	MatchTokens int
	Err         error

	// Release undoes the capacity reservation the dispatcher took out on the
	// caller's behalf. It travels with the result because the dispatcher
	// acquires the slot but the requesting goroutine is the one that finishes
	// using it -- and must return it even if it has since given up. Nil on a
	// refusal, and safe to call exactly once.
	Release func()
}

// Item is one queued request plus its scheduling bookkeeping.
type Item struct {
	Req        state.Request
	EnqueuedAt time.Time
	DeadlineAt time.Time

	// passedOver counts how many times this item was skipped in favour of a
	// higher-priority one. Guarded by Queue.mu.
	passedOver int

	// ready carries exactly one Result. Buffered so the sender never blocks if
	// the waiter has already given up.
	//
	// This replaces the asyncio.Future the Python version used: same one-shot
	// handoff, but a channel composes with select, so the waiter can abandon it
	// on client disconnect without any extra machinery.
	ready chan Result
}

// NewItem builds a queue item for a request.
func NewItem(req state.Request, now time.Time) *Item {
	return &Item{
		Req:        req,
		EnqueuedAt: now,
		DeadlineAt: now.Add(req.Deadline),
		ready:      make(chan Result, 1),
	}
}

// Ready is the channel the caller waits on.
func (i *Item) Ready() <-chan Result { return i.ready }

// Queue is a priority queue with deadline expiry.
type Queue struct {
	cfg *config.Config

	mu    sync.Mutex
	items []*Item

	// notify is a "something changed" signal, capacity 1 so a burst of Puts
	// collapses into one wakeup. Chosen over sync.Cond because a Cond cannot be
	// selected on alongside ctx.Done().
	notify chan struct{}

	onDepth  func(int)        // called with the new depth after every change
	onExpire func(*Item)      // called for each item dropped at its deadline
	now      func() time.Time // injectable for tests
}

// SweepInterval bounds how long an expired item can sit unnoticed when the
// queue is otherwise idle: without it a queue with no arrivals would never
// re-check deadlines.
const SweepInterval = 50 * time.Millisecond

// New returns an empty queue. onDepth and onExpire may be nil.
func New(cfg *config.Config, onDepth func(int), onExpire func(*Item)) *Queue {
	if onDepth == nil {
		onDepth = func(int) {}
	}
	if onExpire == nil {
		onExpire = func(*Item) {}
	}
	return &Queue{
		cfg:      cfg,
		notify:   make(chan struct{}, 1),
		onDepth:  onDepth,
		onExpire: onExpire,
		now:      time.Now,
	}
}

// Len reports the current depth.
func (q *Queue) Len() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.items)
}

// Put enqueues an item, or returns a queue_full refusal.
func (q *Queue) Put(it *Item) error {
	q.mu.Lock()
	if len(q.items) >= q.cfg.QueueMaxSize {
		q.mu.Unlock()
		return refuse.Err(refuse.QueueFull)
	}
	q.items = append(q.items, it)
	depth := len(q.items)
	q.mu.Unlock()

	q.onDepth(depth)
	q.signal()
	return nil
}

// Remove drops an item whose caller has abandoned it, freeing the slot
// immediately. The Python version had no equivalent: a disconnected client's
// request occupied queue capacity until its deadline elapsed.
//
// The bool matters for correctness, not just for stats. False means a
// dispatcher already popped this item and a Result is inbound, so the caller
// still has to collect it and undo the reservation it carries.
func (q *Queue) Remove(target *Item) bool {
	q.mu.Lock()
	idx := slices.Index(q.items, target)
	if idx < 0 {
		q.mu.Unlock()
		return false
	}
	q.items = slices.Delete(q.items, idx, idx+1)
	depth := len(q.items)
	q.mu.Unlock()
	q.onDepth(depth)
	return true
}

// Get blocks until an item is ready to dispatch or ctx is done.
func (q *Queue) Get(ctx context.Context) (*Item, error) {
	t := time.NewTicker(SweepInterval)
	defer t.Stop()
	for {
		if it := q.pop(); it != nil {
			return it, nil
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-q.notify:
		case <-t.C:
		}
	}
}

func (q *Queue) signal() {
	select {
	case q.notify <- struct{}{}:
	default: // a wakeup is already pending; one is enough
	}
}

// pop expires stale items, then removes and returns the highest-priority one.
func (q *Queue) pop() *Item {
	q.mu.Lock()
	now := q.now()
	expired := q.expireLocked(now)

	if len(q.items) == 0 {
		depth := 0
		q.mu.Unlock()
		q.drain(expired, depth)
		return nil
	}

	// Re-sorted on every dequeue rather than kept in a heap, because the keys
	// are not stable: slack shrinks as the clock advances and passedOver grows
	// as items are overtaken, so a heap's invariant would silently rot.
	slices.SortStableFunc(q.items, func(a, b *Item) int { return q.compare(a, b, now) })

	it := q.items[0]
	q.items = slices.Delete(q.items, 0, 1)
	// Everything left just lost a round; age them so they cannot starve.
	for _, other := range q.items {
		other.passedOver++
	}
	depth := len(q.items)
	q.mu.Unlock()

	q.drain(expired, depth)
	return it
}

// drain reports depth once and notifies abandoned waiters, outside the lock.
func (q *Queue) drain(expired []*Item, depth int) {
	for _, it := range expired {
		it.ready <- Result{Err: refuse.Err(refuse.ExpiredInQueue)}
		q.onExpire(it)
	}
	if len(expired) > 0 || depth >= 0 {
		q.onDepth(depth)
	}
}

// expireLocked removes items past their deadline. Caller holds q.mu.
func (q *Queue) expireLocked(now time.Time) []*Item {
	var expired []*Item
	keep := q.items[:0]
	for _, it := range q.items {
		if !it.DeadlineAt.After(now) {
			expired = append(expired, it)
			continue
		}
		keep = append(keep, it)
	}
	q.items = keep
	return expired
}

// compare implements the scheduling policy. Ordering, in priority order:
//
//  1. long: short prompts first. A long prompt monopolises prefill and pushes
//     out many small ones, so it yields -- but only until it has been overtaken
//     MaxOvertakes times, after which this term is forced to 0 and it competes
//     on equal footing. That cutoff is what stops the policy from starving
//     large requests indefinitely.
//  2. slack: least remaining time to deadline wins, i.e. earliest-deadline-
//     first, minus an aging credit of AgingGain per overtake so a repeatedly
//     skipped item climbs.
//  3. enqueuedAt: FIFO as the final tiebreak, so the policy stays deterministic.
func (q *Queue) compare(a, b *Item, now time.Time) int {
	if la, lb := q.longFlag(a), q.longFlag(b); la != lb {
		return la - lb
	}
	if sa, sb := q.slack(a, now), q.slack(b, now); sa != sb {
		if sa < sb {
			return -1
		}
		return 1
	}
	return a.EnqueuedAt.Compare(b.EnqueuedAt)
}

func (q *Queue) longFlag(it *Item) int {
	starved := it.passedOver >= q.cfg.MaxOvertakes
	if starved || it.Req.NIn < q.cfg.LongPromptTokens {
		return 0
	}
	return 1
}

func (q *Queue) slack(it *Item, now time.Time) time.Duration {
	return it.DeadlineAt.Sub(now) - time.Duration(it.passedOver)*q.cfg.AgingGain
}

// Complete hands a dispatch decision back to the waiting request.
func (it *Item) Complete(res Result) { it.ready <- res }
