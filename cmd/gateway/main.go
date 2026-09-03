// Command gateway serves an OpenAI-compatible endpoint in front of one or more
// vLLM replicas, applying quota, admission control, queueing and prefix-aware
// routing.
//
//	gateway --preset full --replicas http://127.0.0.1:8001,http://127.0.0.1:8002
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/cduggn/inference-gateway/internal/config"
	"github.com/cduggn/inference-gateway/internal/gateway"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "gateway:", err)
		os.Exit(1)
	}
}

func run() error {
	cfg := config.Default()

	var (
		replicas = flag.String("replicas", strings.Join(cfg.ReplicaURLs, ","), "comma-separated vLLM replica base URLs")
		addr     = flag.String("addr", cfg.Addr, "listen address")
		preset   = flag.String("preset", "", "feature preset: baseline|route|queue|full (overrides individual flags)")
		logLevel = flag.String("log-level", "info", "log level: debug|info|warn|error")

		admission  = flag.Bool("admission", cfg.AdmissionEnabled, "enable admission control / load shedding")
		queue      = flag.Bool("queue", cfg.QueueEnabled, "enable the bounded priority queue")
		prefix     = flag.Bool("prefix-routing", cfg.UsePrefixRouting, "route by prefix-cache affinity")
		p2c        = flag.Bool("p2c", cfg.UseP2C, "sample two replicas per decision instead of scoring all")
		queueMax   = flag.Int("queue-max", cfg.QueueMaxSize, "maximum queued requests before shedding")
		maxNumSeqs = flag.Int("max-num-seqs", cfg.DefaultMaxNumSeqs, "per-replica concurrency, should match vLLM --max-num-seqs")
	)
	flag.Parse()

	cfg.Addr = *addr
	cfg.ReplicaURLs = splitAndTrim(*replicas)
	cfg.QueueMaxSize = *queueMax
	cfg.DefaultMaxNumSeqs = *maxNumSeqs

	// A preset is a shorthand for the three feature flags, so it wins over
	// them; mixing both silently would make the running posture ambiguous.
	if *preset != "" {
		if err := cfg.ApplyPreset(config.Preset(*preset)); err != nil {
			return err
		}
	} else {
		cfg.AdmissionEnabled = *admission
		cfg.QueueEnabled = *queue
		cfg.UsePrefixRouting = *prefix
	}
	cfg.UseP2C = *p2c

	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: parseLevel(*logLevel)}))

	srv, err := gateway.New(&cfg, log)
	if err != nil {
		return err
	}

	// Cancel on SIGINT/SIGTERM so in-flight requests get a chance to finish
	// rather than being cut off mid-stream.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	return srv.Run(ctx)
}

func splitAndTrim(s string) []string {
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

func parseLevel(s string) slog.Level {
	switch strings.ToLower(s) {
	case "debug":
		return slog.LevelDebug
	case "warn":
		return slog.LevelWarn
	case "error":
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}
