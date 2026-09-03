package admission

import (
	"testing"
	"time"

	"github.com/cduggn/inference-gateway/internal/config"
	"github.com/cduggn/inference-gateway/internal/refuse"
	"github.com/cduggn/inference-gateway/internal/state"
)

// healthy is a fleet with plenty of room, so each test can spoil exactly one
// dimension and assert which reason fires.
func healthy() state.Snapshot {
	return state.Snapshot{
		Replicas: []state.ReplicaSnapshot{
			{ID: "r0", MaxNumSeqs: 8},
			{ID: "r1", MaxNumSeqs: 8},
		},
		StaleFor:          100 * time.Millisecond,
		KVUsageMax:        0.10,
		WaitingTotal:      0,
		HeadroomTokens:    100_000,
		QueueWait:         0,
		PrefillTokensPerS: 10_000,
		InterTokenLatency: time.Millisecond,
	}
}

func req() state.Request {
	return state.Request{Tenant: "agent", NIn: 500, NOut: 128, Deadline: 5 * time.Second}
}

func TestAdmitsHealthyFleet(t *testing.T) {
	cfg := config.Default()
	if got := Decide(&cfg, healthy(), req()); got != "" {
		t.Fatalf("healthy fleet shed with reason %q", got)
	}
}

func TestShedReasons(t *testing.T) {
	cfg := config.Default()

	tests := []struct {
		name  string
		spoil func(*state.Snapshot)
		want  refuse.Reason
	}{
		{
			name:  "stale metrics fail closed",
			spoil: func(s *state.Snapshot) { s.StaleFor = 5 * time.Second },
			want:  refuse.NoSignal,
		},
		{
			name:  "kv cache above ceiling",
			spoil: func(s *state.Snapshot) { s.KVUsageMax = 0.95 },
			want:  refuse.KVPressure,
		},
		{
			name:  "vllm queues backing up",
			spoil: func(s *state.Snapshot) { s.WaitingTotal = 100 },
			want:  refuse.QueueDepth,
		},
		{
			name:  "request larger than free kv",
			spoil: func(s *state.Snapshot) { s.HeadroomTokens = 10 },
			want:  refuse.NoHeadroom,
		},
		{
			name:  "cannot finish before the deadline",
			spoil: func(s *state.Snapshot) { s.InterTokenLatency = time.Second },
			want:  refuse.DeadlineUnmeetable,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			snap := healthy()
			tt.spoil(&snap)
			if got := Decide(&cfg, snap, req()); got != tt.want {
				t.Errorf("Decide() = %q, want %q", got, tt.want)
			}
		})
	}
}

// TestCheckOrderIsStable guards the documented precedence: when the fleet is
// bad in several ways at once, the earliest check must win, because "we are
// blind" is more actionable than "this request does not fit".
func TestCheckOrderIsStable(t *testing.T) {
	cfg := config.Default()
	snap := healthy()
	snap.StaleFor = 10 * time.Second
	snap.KVUsageMax = 0.99
	snap.HeadroomTokens = 0

	if got := Decide(&cfg, snap, req()); got != refuse.NoSignal {
		t.Fatalf("Decide() = %q, want %q to take precedence", got, refuse.NoSignal)
	}
}
