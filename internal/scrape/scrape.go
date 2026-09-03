// Package scrape pulls vLLM's Prometheus metrics into fleet state.
//
// The Python original hand-rolled a regex over the exposition format. Go has
// the actual upstream parser -- github.com/prometheus/common/expfmt, the same
// code Prometheus itself uses -- so this is one place the port is strictly
// better than the original: histograms, label sets and type information are
// handled properly instead of being flattened by a line regex.
package scrape

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	dto "github.com/prometheus/client_model/go"
	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"

	"github.com/cduggn/inference-gateway/internal/state"
)

// Metrics is a parsed /metrics response, reduced to the aggregates the gateway
// cares about. Values are summed across label sets, since the gateway scrapes
// one replica at a time and does not care which model label served what.
type Metrics map[string]sample

type sample struct {
	Sum   float64
	Count float64
}

// Parse reads Prometheus text exposition into aggregated samples.
//
// The recover is defence in depth. This parses output from a separate process
// on a background goroutine, and in Go an unrecovered panic anywhere takes down
// the entire gateway -- unlike Python, where an exception in the scrape task
// would only have killed that task. A malformed /metrics response must degrade
// to "this replica looks stale", never to a dead gateway. (expfmt does panic on
// a zero-value TextParser, which is why it is constructed explicitly below.)
func Parse(r io.Reader) (m Metrics, err error) {
	defer func() {
		if rec := recover(); rec != nil {
			m, err = nil, fmt.Errorf("malformed metrics payload: %v", rec)
		}
	}()

	// Must be constructed, not zero-valued: as of prometheus/common v0.71 the
	// parser carries its own name-validation scheme and a zero value panics.
	// UTF8 accepts vLLM's colon-separated names and the legacy charset alike.
	p := expfmt.NewTextParser(model.UTF8Validation)
	families, err := p.TextToMetricFamilies(r)
	if err != nil {
		return nil, fmt.Errorf("parse metrics: %w", err)
	}
	out := make(Metrics, len(families))
	for name, mf := range families {
		var s sample
		for _, m := range mf.GetMetric() {
			switch mf.GetType() {
			case dto.MetricType_GAUGE:
				s.Sum += m.GetGauge().GetValue()
				s.Count++
			case dto.MetricType_COUNTER:
				s.Sum += m.GetCounter().GetValue()
				s.Count++
			case dto.MetricType_HISTOGRAM:
				s.Sum += m.GetHistogram().GetSampleSum()
				s.Count += float64(m.GetHistogram().GetSampleCount())
			case dto.MetricType_SUMMARY:
				s.Sum += m.GetSummary().GetSampleSum()
				s.Count += float64(m.GetSummary().GetSampleCount())
			default:
				s.Sum += m.GetUntyped().GetValue()
				s.Count++
			}
		}
		out[name] = s
	}
	return out, nil
}

// value returns the summed value of the first metric name that is present.
// Several names are tried because vLLM has renamed metrics across releases
// (inter_token_latency_seconds became time_per_output_token_seconds), and a
// gateway that silently reads zero from a renamed metric makes confidently
// wrong admission decisions.
func (m Metrics) value(names ...string) (sample, bool) {
	for _, n := range names {
		if s, ok := m[n]; ok {
			return s, true
		}
		// Tolerate pre-flattened histogram families (name_sum / name_count).
		if s, ok := m[n+"_sum"]; ok {
			c := m[n+"_count"]
			return sample{Sum: s.Sum, Count: c.Sum}, true
		}
	}
	return sample{}, false
}

// Apply folds a scraped metric set into a replica's state.
func Apply(r *state.Replica, m Metrics) {
	if s, ok := m.value("vllm:num_requests_running"); ok {
		r.Running = int(s.Sum)
	}
	if s, ok := m.value("vllm:num_requests_waiting"); ok {
		r.Waiting = int(s.Sum)
	}
	if s, ok := m.value("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"); ok {
		// vLLM has reported this both as a 0..1 ratio and as a 0..100
		// percentage depending on version. Normalise to a ratio.
		v := s.Sum
		if v > 1.0 {
			v /= 100.0
		}
		r.KVUsage = v
	}
	if s, ok := m.value("vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries"); ok {
		r.PrefixQueries = int(s.Sum)
	}
	if s, ok := m.value("vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"); ok {
		r.PrefixHits = int(s.Sum)
	}
	if s, ok := m.value("vllm:num_preemptions_total", "vllm:num_preemptions"); ok {
		r.Preemptions = int(s.Sum)
	}
	if s, ok := m.value("vllm:time_to_first_token_seconds"); ok {
		r.TTFTSum, r.TTFTCount = s.Sum, s.Count
	}
	if s, ok := m.value("vllm:inter_token_latency_seconds", "vllm:time_per_output_token_seconds"); ok {
		r.ITLSum, r.ITLCount = s.Sum, s.Count
	}
	if s, ok := m.value("vllm:request_queue_time_seconds"); ok {
		r.QueueTimeSum, r.QueueTimeCnt = s.Sum, s.Count
	}
}

// Scraper polls every replica's /metrics endpoint on an interval.
type Scraper struct {
	fleet    *state.Fleet
	client   *http.Client
	interval time.Duration
	log      *slog.Logger
}

// NewScraper wires a scraper. The client should have a short timeout: a hung
// metrics endpoint must not delay the next scrape round.
func NewScraper(f *state.Fleet, interval time.Duration, log *slog.Logger) *Scraper {
	return &Scraper{
		fleet:    f,
		client:   &http.Client{Timeout: 2 * time.Second},
		interval: interval,
		log:      log,
	}
}

// Once scrapes all replicas, reporting whether every one succeeded. Fleet
// staleness is only reset on a fully successful round, so a single unreachable
// replica makes the whole fleet look stale -- which is what drives admission
// control to fail closed rather than route on a partial picture.
func (s *Scraper) Once(ctx context.Context) bool {
	ok := true
	for _, r := range s.fleet.Replicas() {
		m, err := s.fetch(ctx, r.URL)
		if err != nil {
			ok = false
			s.log.Debug("scrape failed", "replica", r.ID, "err", err)
			continue
		}
		s.fleet.Update(r.ID, func(rep *state.Replica) { Apply(rep, m) })
	}
	if ok {
		s.fleet.MarkScraped(time.Now())
	}
	return ok
}

func (s *Scraper) fetch(ctx context.Context, base string) (Metrics, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimSuffix(base, "/")+"/metrics", nil)
	if err != nil {
		return nil, err
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("metrics endpoint returned %s", resp.Status)
	}
	// Bound the read: a misbehaving upstream should not be able to exhaust
	// gateway memory through the metrics path.
	return Parse(io.LimitReader(resp.Body, 8<<20))
}

// Run scrapes until ctx is cancelled.
func (s *Scraper) Run(ctx context.Context) {
	t := time.NewTicker(s.interval)
	defer t.Stop()
	s.Once(ctx)
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			s.Once(ctx)
		}
	}
}
