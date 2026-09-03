// Package config holds every tunable in one explicit struct.
//
// Deliberate deviation from the Python original: there, gateway/config.py was a
// module of mutable globals that main.py reassigned at startup and every other
// module read live via cfg.SOMETHING. That is workable in Python but is a data
// race in Go the moment two goroutines touch it, and it makes tests order
// dependent. Here the config is constructed once, validated, and passed down by
// value/pointer. Nothing mutates it after Server construction.
package config

import (
	"fmt"
	"net/url"
	"time"
)

// Preset names a bundle of feature flags, mirroring config.apply_preset in the
// Python original. Presets exist so the lab can be run in four postures and the
// difference in behaviour attributed to a specific mechanism.
type Preset string

const (
	// PresetBaseline: dumb round-robin proxy. No admission, no queue, no
	// prefix affinity. The control group.
	PresetBaseline Preset = "baseline"
	// PresetRoute: adds prefix-aware routing only.
	PresetRoute Preset = "route"
	// PresetQueue: adds the bounded priority queue on top of routing.
	PresetQueue Preset = "queue"
	// PresetFull: everything, including admission control / load shedding.
	PresetFull Preset = "full"
)

// Config is the fully resolved gateway configuration.
type Config struct {
	// --- serving ---
	Addr            string        // listen address for the gateway itself
	ReplicaURLs     []string      // upstream vLLM replicas, fixed at startup
	ServedModelName string        // model id echoed on /v1/models
	HTTPTimeout     time.Duration // per-request upstream timeout
	MaxRequestBytes int64         // hard cap on inbound body size

	// --- model shape, used for capacity math ---
	DefaultMaxNumSeqs  int // vLLM --max-num-seqs per replica
	DefaultMaxModelLen int
	KVCapacityTokens   int // assumed KV cache capacity per replica, in tokens
	DefaultMaxTokens   int // n_out when the client does not say
	BlockSize          int // tokens per prefix-trie block; mirror vLLM's block size

	// --- tenancy ---
	TenantTiers    map[string]string        // tenant id -> tier name
	DeadlineByTier map[string]time.Duration // tier name -> latency objective
	MaxDeadline    time.Duration            // clamp on a client-supplied deadline

	// --- quota (token bucket, per tenant) ---
	BucketRateTokensPerS float64
	BucketBurstTokens    float64

	// --- admission control ---
	AdmissionEnabled         bool
	StaleCeiling             time.Duration // refuse if fleet metrics are older than this
	KVCeiling                float64       // refuse above this KV utilisation
	WaitingCeilingPerReplica int
	InitPrefillTokensPerS    float64       // seed estimate before metrics arrive
	InitInterTokenLatency    time.Duration // seed estimate before metrics arrive

	// --- gateway-side queue ---
	QueueEnabled      bool
	QueueMaxSize      int
	AgingGain         time.Duration // slack credited per time an item is overtaken
	MaxOvertakes      int           // starvation cutoff
	LongPromptTokens  int           // prompts at or above this are deprioritised
	DispatchOvershoot int           // in-flight slack above summed max-num-seqs

	// --- routing ---
	UsePrefixRouting bool
	UseP2C           bool // power-of-two-choices sampling
	WPrefix          float64
	WLoad            float64
	LoadCeiling      float64
	TrieTTL          time.Duration

	// --- observability ---
	ScrapeInterval time.Duration
	StatsRing      int
}

// Default returns the same values gateway/config.py ships with, so behaviour is
// comparable between the two implementations out of the box.
func Default() Config {
	return Config{
		Addr:            "127.0.0.1:8080",
		ReplicaURLs:     []string{"http://127.0.0.1:8001", "http://127.0.0.1:8002"},
		ServedModelName: "lab",
		HTTPTimeout:     60 * time.Second,
		MaxRequestBytes: 1 << 20, // 1 MiB; the Python version read the body unbounded

		DefaultMaxNumSeqs:  8,
		DefaultMaxModelLen: 16384,
		KVCapacityTokens:   16384,
		DefaultMaxTokens:   256,
		BlockSize:          16,

		TenantTiers: map[string]string{
			"default": "interactive",
			"chat":    "interactive",
			"agent":   "agentic",
		},
		DeadlineByTier: map[string]time.Duration{
			"interactive": 2 * time.Second,
			"agentic":     5 * time.Second,
			"batch":       8 * time.Second,
		},
		MaxDeadline: 30 * time.Second,

		BucketRateTokensPerS: 50_000,
		BucketBurstTokens:    200_000,

		AdmissionEnabled:         false,
		StaleCeiling:             2 * time.Second,
		KVCeiling:                0.85,
		WaitingCeilingPerReplica: 4,
		InitPrefillTokensPerS:    4_000,
		InitInterTokenLatency:    8 * time.Millisecond,

		QueueEnabled:      false,
		QueueMaxSize:      64,
		AgingGain:         150 * time.Millisecond,
		MaxOvertakes:      8,
		LongPromptTokens:  1024,
		DispatchOvershoot: 4,

		UsePrefixRouting: false,
		UseP2C:           false,
		WPrefix:          1.0,
		WLoad:            64.0,
		LoadCeiling:      2.0,
		TrieTTL:          90 * time.Second,

		ScrapeInterval: 250 * time.Millisecond,
		StatsRing:      8192,
	}
}

// ApplyPreset flips the three feature flags to a named combination.
func (c *Config) ApplyPreset(p Preset) error {
	switch p {
	case PresetBaseline:
		c.AdmissionEnabled, c.QueueEnabled, c.UsePrefixRouting = false, false, false
	case PresetRoute:
		c.AdmissionEnabled, c.QueueEnabled, c.UsePrefixRouting = false, false, true
	case PresetQueue:
		c.AdmissionEnabled, c.QueueEnabled, c.UsePrefixRouting = false, true, true
	case PresetFull:
		c.AdmissionEnabled, c.QueueEnabled, c.UsePrefixRouting = true, true, true
	default:
		return fmt.Errorf("unknown preset %q (want baseline|route|queue|full)", p)
	}
	return nil
}

// TierFor resolves a tenant to its tier, falling back to the interactive tier
// for anything unrecognised. Callers must have already normalised the tenant id
// (see gateway.normalizeTenant) so this is never handed raw header bytes.
func (c *Config) TierFor(tenant string) string {
	if tier, ok := c.TenantTiers[tenant]; ok {
		return tier
	}
	return "interactive"
}

// DeadlineFor returns the latency objective for a tenant.
func (c *Config) DeadlineFor(tenant string) time.Duration {
	if d, ok := c.DeadlineByTier[c.TierFor(tenant)]; ok {
		return d
	}
	return 2 * time.Second
}

// Validate rejects a configuration that cannot serve traffic. Upstream URLs are
// checked here, at startup, so a malformed replica address fails loudly instead
// of turning into a per-request error later. Because replicas come only from
// config and never from a request, the proxy has no SSRF surface.
func (c *Config) Validate() error {
	if len(c.ReplicaURLs) == 0 {
		return fmt.Errorf("no replica URLs configured")
	}
	for _, raw := range c.ReplicaURLs {
		u, err := url.Parse(raw)
		if err != nil {
			return fmt.Errorf("replica %q: %w", raw, err)
		}
		if u.Scheme != "http" && u.Scheme != "https" {
			return fmt.Errorf("replica %q: scheme must be http or https", raw)
		}
		if u.Host == "" {
			return fmt.Errorf("replica %q: missing host", raw)
		}
	}
	if c.BlockSize <= 0 {
		return fmt.Errorf("block size must be positive, got %d", c.BlockSize)
	}
	if c.QueueMaxSize <= 0 {
		return fmt.Errorf("queue max size must be positive, got %d", c.QueueMaxSize)
	}
	if c.MaxRequestBytes <= 0 {
		return fmt.Errorf("max request bytes must be positive, got %d", c.MaxRequestBytes)
	}
	return nil
}
