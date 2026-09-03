// Command agent drives the gateway with a small langchaingo pipeline, so the
// serving path is exercised by agent-shaped traffic: several sequential model
// calls per user query, with overlapping prompts.
//
//	agent --repeat 5 "What is paged attention?"
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/cduggn/inference-gateway/internal/agent"
	"github.com/cduggn/inference-gateway/internal/config"
	"github.com/cduggn/inference-gateway/pkg/limiter"
)

// defaultQueries are prompts that deliberately overlap in phrasing, so repeated
// runs exercise the prefix cache instead of always missing.
var defaultQueries = []string{
	"What is paged attention in LLM inference?",
	"Why does TTFT jump when the KV cache fills?",
	"What is prefix caching and when does it miss?",
	"How does a gateway decide to shed load?",
	"What is power of two choices in replica routing?",
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "agent:", err)
		os.Exit(1)
	}
}

func run() error {
	cfg := config.Default()

	var (
		endpoint  = flag.String("gateway", envOr("GATEWAY_URL", "http://"+cfg.Addr), "gateway base URL")
		tenant    = flag.String("tenant", "agent", "value sent as X-Tenant-Id")
		single    = flag.Bool("single", false, "one model call per query instead of researcher+writer")
		repeat    = flag.Int("repeat", 1, "number of queries to send")
		maxTokens = flag.Int("max-tokens", 128, "max tokens per model call")
		rps       = flag.Float64("rps", 4, "client-side request rate limit")
		burst     = flag.Float64("burst", 8, "client-side burst allowance")
	)
	flag.Parse()

	lim := limiter.New(*rps, *burst)
	client := agent.NewClient(*endpoint, cfg.ServedModelName,
		agent.WithTenant(*tenant),
		agent.WithMaxTokens(*maxTokens),
		agent.WithLimiter(lim),
	)

	crew, err := agent.NewCrew(client, !*single)
	if err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	queries := buildQueries(flag.Args(), *repeat)
	calls := 1
	if crew.Dual() {
		calls = 2
	}
	fmt.Printf("agent -> limiter (%.0f rps, burst %.0f) -> %s -> vLLM  [%d call(s) per query]\n\n",
		*rps, *burst, *endpoint, calls)

	var refused int
	for i, q := range queries {
		if ctx.Err() != nil {
			break
		}
		fmt.Printf("USER [%d/%d]: %s\n", i+1, len(queries), q)

		start := time.Now()
		res := crew.Run(ctx, client, q)

		switch {
		case res.Err == nil:
			fmt.Println(res.Answer)
		default:
			refused++
			var refusal *agent.RefusedError
			if errors.As(res.Err, &refusal) {
				// A refusal is the gateway working as designed, not a crash.
				// Report it as backpressure so the two are never conflated.
				fmt.Printf("REFUSED: %s (retry after %s)\n", refusal.Reason, refusal.RetryAfter)
			} else {
				fmt.Println("ERROR:", res.Err)
			}
		}
		if m := res.Last; m != nil {
			fmt.Printf("  [replica=%s prefix_match=%d tokens  last_call=%dms  total=%dms]\n",
				orDash(m.Replica), m.PrefixMatch,
				m.TotalLatency.Milliseconds(), time.Since(start).Milliseconds())
		}
		fmt.Println()
	}

	fmt.Printf("%d/%d queries refused\n", refused, len(queries))
	return nil
}

func buildQueries(args []string, repeat int) []string {
	if repeat < 1 {
		repeat = 1
	}
	pool := defaultQueries
	if len(args) > 0 {
		pool = []string{strings.Join(args, " ")}
	}
	out := make([]string, 0, repeat)
	for i := 0; i < repeat; i++ {
		out = append(out, pool[i%len(pool)])
	}
	return out
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func orDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}
