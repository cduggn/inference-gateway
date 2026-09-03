package pending

import (
	"context"
	"testing"
	"time"

	"github.com/cduggn/inference-gateway/internal/config"
	"github.com/cduggn/inference-gateway/internal/refuse"
	"github.com/cduggn/inference-gateway/internal/state"
)

func testQueue(t *testing.T) (*Queue, *time.Time) {
	t.Helper()
	cfg := config.Default()
	cfg.QueueMaxSize = 4
	now := time.Now()
	q := New(&cfg, nil, nil)
	q.now = func() time.Time { return now }
	return q, &now
}

func item(nIn int, deadline time.Duration, now time.Time) *Item {
	return NewItem(state.Request{NIn: nIn, NOut: 64, Deadline: deadline}, now)
}

func TestEarliestDeadlineFirst(t *testing.T) {
	q, now := testQueue(t)

	relaxed := item(10, 5*time.Second, *now)
	urgent := item(10, 1*time.Second, *now)
	if err := q.Put(relaxed); err != nil {
		t.Fatal(err)
	}
	if err := q.Put(urgent); err != nil {
		t.Fatal(err)
	}

	got, err := q.Get(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if got != urgent {
		t.Error("dequeued the relaxed request first; expected earliest deadline to win")
	}
}

func TestLongPromptsYieldToShortOnes(t *testing.T) {
	q, now := testQueue(t)

	// The long prompt is enqueued first and has the same deadline, so only the
	// long-prompt rule can reorder them.
	long := item(4096, 5*time.Second, *now)
	short := item(64, 5*time.Second, *now)
	_ = q.Put(long)
	_ = q.Put(short)

	got, _ := q.Get(context.Background())
	if got != short {
		t.Error("long prompt went first; expected it to yield to the short one")
	}
}

// TestStarvedLongPromptStopsYielding pins the anti-starvation cutoff: after
// MaxOvertakes skips, a long prompt competes on equal footing again.
func TestStarvedLongPromptStopsYielding(t *testing.T) {
	cfg := config.Default()
	cfg.QueueMaxSize = 64
	cfg.MaxOvertakes = 2
	q := New(&cfg, nil, nil)
	now := time.Now()
	q.now = func() time.Time { return now }

	long := item(4096, 10*time.Second, now)
	_ = q.Put(long)

	// Feed short prompts past it until it has been overtaken twice.
	for range 2 {
		short := item(64, 10*time.Second, now)
		_ = q.Put(short)
		got, _ := q.Get(context.Background())
		if got == long {
			t.Fatal("long prompt dequeued before reaching the starvation cutoff")
		}
	}

	// Now it is starved; a freshly arrived short prompt must not overtake again.
	fresh := item(64, 10*time.Second, now)
	_ = q.Put(fresh)
	got, _ := q.Get(context.Background())
	if got != long {
		t.Error("starved long prompt was overtaken again past MaxOvertakes")
	}
}

func TestExpiredItemsAreRefusedNotDispatched(t *testing.T) {
	q, now := testQueue(t)

	doomed := item(10, 500*time.Millisecond, *now)
	_ = q.Put(doomed)

	*now = now.Add(time.Second) // deadline elapsed while queued

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	// Nothing dispatchable remains, so Get blocks until the context expires...
	if _, err := q.Get(ctx); err == nil {
		t.Fatal("Get returned an item that was already past its deadline")
	}
	// ...and the waiter is told why.
	select {
	case res := <-doomed.Ready():
		var re *refuse.Error
		if !asRefusal(res.Err, &re) || re.Reason != refuse.ExpiredInQueue {
			t.Fatalf("waiter got %v, want %s", res.Err, refuse.ExpiredInQueue)
		}
	default:
		t.Fatal("expired item never notified its waiter")
	}
}

func TestPutRefusesWhenFull(t *testing.T) {
	q, now := testQueue(t)
	for range 4 {
		if err := q.Put(item(10, time.Minute, *now)); err != nil {
			t.Fatalf("unexpected refusal below capacity: %v", err)
		}
	}
	err := q.Put(item(10, time.Minute, *now))
	var re *refuse.Error
	if !asRefusal(err, &re) || re.Reason != refuse.QueueFull {
		t.Fatalf("Put past capacity returned %v, want %s", err, refuse.QueueFull)
	}
}

// TestRemoveReportsOwnership covers the handoff that keeps the dispatch
// semaphore balanced: true means the caller owned the item and no result is
// coming, false means one is already in flight and must still be collected.
func TestRemoveReportsOwnership(t *testing.T) {
	q, now := testQueue(t)

	queued := item(10, time.Minute, *now)
	_ = q.Put(queued)
	if !q.Remove(queued) {
		t.Error("Remove returned false for an item still in the queue")
	}
	if q.Remove(queued) {
		t.Error("Remove returned true for an item it had already removed")
	}

	taken := item(10, time.Minute, *now)
	_ = q.Put(taken)
	if _, err := q.Get(context.Background()); err != nil {
		t.Fatal(err)
	}
	if q.Remove(taken) {
		t.Error("Remove claimed ownership of an item a dispatcher already took")
	}
}

func TestGetRespectsContextCancellation(t *testing.T) {
	q, _ := testQueue(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := q.Get(ctx); err == nil {
		t.Fatal("Get ignored a cancelled context")
	}
}

func asRefusal(err error, target **refuse.Error) bool {
	if err == nil {
		return false
	}
	re, ok := err.(*refuse.Error)
	if ok {
		*target = re
	}
	return ok
}
