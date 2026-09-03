// Package gateway is the HTTP front end: one OpenAI-compatible endpoint in
// front of N vLLM replicas, with quota, admission control, a priority queue and
// prefix-aware routing between them.
package gateway

import (
	"context"
	"errors"
	"log/slog"
	"net"
	"net/http"
	"sync"
	"time"

	"github.com/cduggn/inference-gateway/internal/bucket"
	"github.com/cduggn/inference-gateway/internal/config"
	"github.com/cduggn/inference-gateway/internal/pending"
	"github.com/cduggn/inference-gateway/internal/refuse"
	"github.com/cduggn/inference-gateway/internal/router"
	"github.com/cduggn/inference-gateway/internal/scrape"
	"github.com/cduggn/inference-gateway/internal/state"
	"github.com/cduggn/inference-gateway/internal/stats"
	"github.com/cduggn/inference-gateway/internal/tokenize"
	"github.com/cduggn/inference-gateway/internal/trie"
)

// Server holds everything the request path touches. Constructed once; none of
// these fields are reassigned after New returns, so no lock guards the struct
// itself -- only the state each component owns internally.
type Server struct {
	cfg    *config.Config
	log    *slog.Logger
	fleet  *state.Fleet
	stats  *stats.Ring
	quota  *bucket.Buckets
	router *router.Router
	queue  *pending.Queue
	tok    tokenize.Tokenizer
	client *http.Client

	// slots is a counting semaphore over fleet-wide concurrency. The Python
	// version polled a counter every 5ms to the same end; a buffered channel
	// blocks precisely and wakes exactly one waiter, with no polling interval to
	// tune. Capacity is fixed at startup because max-num-seqs is configuration,
	// not telemetry.
	slots chan struct{}

	replicaByID map[string]*state.Replica
}

// New builds a server from configuration. It does not start any background
// work; call Run for that.
func New(cfg *config.Config, log *slog.Logger) (*Server, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	fleet := state.NewFleet(cfg.ReplicaURLs, cfg.DefaultMaxNumSeqs, cfg.KVCapacityTokens,
		cfg.InitPrefillTokensPerS, cfg.InitInterTokenLatency)

	known := make([]string, 0, len(cfg.TenantTiers))
	for t := range cfg.TenantTiers {
		known = append(known, t)
	}

	s := &Server{
		cfg:    cfg,
		log:    log,
		fleet:  fleet,
		stats:  stats.NewRing(cfg.StatsRing),
		quota:  bucket.New(cfg.BucketRateTokensPerS, cfg.BucketBurstTokens, known),
		router: router.New(trie.New(cfg.TrieTTL, cfg.BlockSize)),
		tok:    tokenize.Fold4{},
		client: &http.Client{
			Timeout: cfg.HTTPTimeout,
			// Replicas are fixed by configuration and are expected to answer
			// directly. Following a redirect would let a compromised or
			// misconfigured upstream point the gateway somewhere else.
			CheckRedirect: func(*http.Request, []*http.Request) error {
				return http.ErrUseLastResponse
			},
		},
		slots:       make(chan struct{}, capacityFor(cfg)),
		replicaByID: make(map[string]*state.Replica),
	}
	s.queue = pending.New(cfg, fleet.SetQueueWait, s.onExpire)
	for _, r := range fleet.Replicas() {
		s.replicaByID[r.ID] = r
	}
	return s, nil
}

func capacityFor(cfg *config.Config) int {
	c := len(cfg.ReplicaURLs)*cfg.DefaultMaxNumSeqs + cfg.DispatchOvershoot
	if c < 1 {
		c = 1
	}
	return c
}

// Handler returns the routed HTTP handler.
//
// Uses the standard library's method-and-pattern routing (Go 1.22+) rather than
// a third-party router: there is no middleware chain here deep enough to earn
// the dependency. FastAPI's decorator registration maps to these explicit
// Handle calls.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", s.handleChat)
	mux.HandleFunc("GET /v1/models", s.handleModels)
	mux.HandleFunc("GET /health", s.handleHealth)
	mux.HandleFunc("GET /_stats", s.handleStats)
	return s.recoverer(mux)
}

// recoverer keeps one panicking request from taking down the process, which
// matters more here than in Python: an unrecovered panic in any goroutine is
// fatal to the whole program, whereas an unhandled Python exception in one
// coroutine only fails that request.
func (s *Server) recoverer(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if rec := recover(); rec != nil {
				s.log.Error("panic in handler", "path", r.URL.Path, "recover", rec)
				// The response may be partially written; only try if not.
				http.Error(w, `{"error":{"type":"internal"}}`, http.StatusInternalServerError)
			}
		}()
		next.ServeHTTP(w, r)
	})
}

// Run starts background workers and serves until ctx is cancelled, then shuts
// down gracefully. Every goroutine it starts is joined before it returns, so a
// caller that sees Run return knows nothing is still running.
func (s *Server) Run(ctx context.Context) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	var wg sync.WaitGroup

	scraper := scrape.NewScraper(s.fleet, s.cfg.ScrapeInterval, s.log)
	wg.Add(1)
	go func() { defer wg.Done(); scraper.Run(ctx) }()

	if s.cfg.QueueEnabled {
		wg.Add(1)
		go func() { defer wg.Done(); s.dispatchLoop(ctx) }()
	}

	wg.Add(1)
	go func() { defer wg.Done(); s.pruneLoop(ctx) }()

	srv := &http.Server{
		Addr:    s.cfg.Addr,
		Handler: s.Handler(),
		// Slowloris protection: a client must send its headers promptly.
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       30 * time.Second,
		IdleTimeout:       120 * time.Second,
		// No WriteTimeout on purpose: a streamed completion is a long-lived
		// response, and a write deadline would sever it mid-generation. The
		// per-request context and the upstream client timeout bound it instead.
		BaseContext: func(net.Listener) context.Context { return ctx },
	}

	errCh := make(chan error, 1)
	go func() {
		s.log.Info("gateway listening",
			"addr", s.cfg.Addr,
			"replicas", s.cfg.ReplicaURLs,
			"admission", s.cfg.AdmissionEnabled,
			"queue", s.cfg.QueueEnabled,
			"prefix_routing", s.cfg.UsePrefixRouting,
			"p2c", s.cfg.UseP2C,
		)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errCh <- err
		}
		close(errCh)
	}()

	select {
	case err := <-errCh:
		cancel()
		wg.Wait()
		return err
	case <-ctx.Done():
	}

	shutdownCtx, stop := context.WithTimeout(context.WithoutCancel(ctx), 10*time.Second)
	defer stop()
	err := srv.Shutdown(shutdownCtx)
	cancel()
	wg.Wait()
	s.log.Info("gateway stopped")
	return err
}

// dispatchLoop drains the priority queue, reserving fleet capacity for each
// item before handing it to the goroutine that is waiting on it.
func (s *Server) dispatchLoop(ctx context.Context) {
	for {
		item, err := s.queue.Get(ctx)
		if err != nil {
			return // context cancelled: shutting down
		}
		d, err := s.assign(ctx, item.Req)
		if err != nil {
			item.Complete(pending.Result{Err: err})
			continue
		}
		s.stats.Emit(stats.Event{
			Kind: stats.Dequeued, Tenant: item.Req.Tenant,
			NIn: item.Req.NIn, NOut: item.Req.NOut, Replica: d.choice.ReplicaID,
		})
		item.Complete(pending.Result{
			ReplicaID:   d.choice.ReplicaID,
			MatchTokens: d.choice.MatchTokens,
			Release:     d.release,
		})
	}
}

// pruneLoop reclaims expired prefix-trie nodes. Without it the trie is a slow
// memory leak: entries stop matching at their TTL but stay resident, which is
// exactly what the Python implementation does.
func (s *Server) pruneLoop(ctx context.Context) {
	interval := s.cfg.TrieTTL / 2
	if interval < time.Second {
		interval = time.Second
	}
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			if n := s.router.Trie().Prune(); n > 0 {
				s.log.Debug("pruned prefix trie", "nodes", n)
			}
		}
	}
}

// dispatch is a granted capacity reservation plus the replica it is for.
type dispatch struct {
	choice  router.Choice
	replica *state.Replica
	release func()
}

// assign waits for fleet capacity, then picks a replica.
func (s *Server) assign(ctx context.Context, req state.Request) (dispatch, error) {
	select {
	case s.slots <- struct{}{}:
	case <-ctx.Done():
		return dispatch{}, ctx.Err()
	}

	snap := s.fleet.Snapshot(time.Now())
	choice, ok := s.router.Pick(s.cfg, snap.Replicas, req)
	if !ok {
		<-s.slots
		return dispatch{}, refuse.Err(refuse.NoSignal)
	}
	replica := s.replicaByID[choice.ReplicaID]
	replica.InFlight.Add(1)

	var once sync.Once
	return dispatch{
		choice:  choice,
		replica: replica,
		release: func() {
			once.Do(func() {
				replica.InFlight.Add(-1)
				<-s.slots
			})
		},
	}, nil
}

func (s *Server) onExpire(item *pending.Item) {
	s.stats.Emit(stats.Event{
		Kind: stats.Expired, Reason: string(refuse.ExpiredInQueue),
		Tenant: item.Req.Tenant, NIn: item.Req.NIn, NOut: item.Req.NOut,
		Outcome: "expired",
	})
}
