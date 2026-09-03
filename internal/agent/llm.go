// Package agent is the client half of the lab: an agent pipeline whose every
// model call goes through the gateway, so agent traffic exercises quota,
// admission, queueing and routing the way real workloads would.
//
// LIBRARY GAP. The Python lab uses CrewAI (agents with roles, tasks, a crew,
// sequential process). Go has no CrewAI. github.com/tmc/langchaingo is the
// closest analogue and is what this package builds on: it supplies the
// llms.Model interface, prompt templates and sequential chains. What it does
// not supply is CrewAI's role/goal/backstory framing or its task-delegation
// model, so "researcher" and "writer" here are two chained LLM calls with
// distinct prompts rather than two agents in a crew. The observable serving
// behaviour -- two sequential calls per query, the second sharing a prefix with
// the first -- is the same, which is what the gateway is being measured on.
package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/tmc/langchaingo/llms"

	"github.com/cduggn/inference-gateway/pkg/limiter"
)

// Meta is what the gateway reported about the most recent call. The gateway
// surfaces its routing decision in response headers; capturing it here is what
// makes prefix-cache behaviour observable from the client side.
type Meta struct {
	Status       int
	Reason       string
	Replica      string
	PrefixMatch  int
	TotalLatency time.Duration
}

// Client is an llms.Model backed by the gateway.
//
// It implements langchaingo's model interface, so it drops into any chain or
// agent in that ecosystem -- the same role GatewayLLM(BaseLLM) plays for CrewAI
// in the Python lab.
type Client struct {
	endpoint  string
	model     string
	tenant    string
	maxTokens int
	limiter   *limiter.Limiter
	hc        *http.Client

	last atomic.Pointer[Meta]
}

// Option configures a Client.
type Option func(*Client)

// WithTenant sets the X-Tenant-Id sent with every call, which selects the
// gateway-side quota bucket and latency tier.
func WithTenant(t string) Option { return func(c *Client) { c.tenant = t } }

// WithMaxTokens caps generation length per call.
func WithMaxTokens(n int) Option { return func(c *Client) { c.maxTokens = n } }

// WithLimiter attaches a client-side rate limiter.
func WithLimiter(l *limiter.Limiter) Option { return func(c *Client) { c.limiter = l } }

// WithHTTPClient overrides the HTTP client, mainly for tests.
func WithHTTPClient(h *http.Client) Option { return func(c *Client) { c.hc = h } }

// NewClient returns a gateway-backed model.
func NewClient(endpoint, model string, opts ...Option) *Client {
	c := &Client{
		endpoint:  strings.TrimSuffix(endpoint, "/"),
		model:     model,
		tenant:    "agent",
		maxTokens: 128,
		hc:        &http.Client{Timeout: 60 * time.Second},
	}
	for _, o := range opts {
		o(c)
	}
	return c
}

// LastMeta returns the routing metadata from the most recent call, or nil.
func (c *Client) LastMeta() *Meta { return c.last.Load() }

// compile-time check that the gateway client satisfies langchaingo's interface.
var _ llms.Model = (*Client)(nil)

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model     string        `json:"model"`
	Messages  []chatMessage `json:"messages"`
	MaxTokens int           `json:"max_tokens"`
	Stream    bool          `json:"stream"`
}

type chatResponse struct {
	Choices []struct {
		Message      chatMessage `json:"message"`
		FinishReason string      `json:"finish_reason"`
	} `json:"choices"`
	Error *struct {
		Type   string `json:"type"`
		Reason string `json:"reason"`
	} `json:"error"`
}

// GenerateContent implements llms.Model.
func (c *Client) GenerateContent(ctx context.Context, messages []llms.MessageContent, options ...llms.CallOption) (*llms.ContentResponse, error) {
	opts := llms.CallOptions{MaxTokens: c.maxTokens}
	for _, o := range options {
		o(&opts)
	}

	// Throttle before the request is built, not after: the point of a
	// client-side limiter is to avoid spending a round trip to be told no.
	if c.limiter != nil && !c.limiter.TryAcquire() {
		return nil, fmt.Errorf("client limiter: %.0f rps / burst %.0f exceeded: %w",
			c.limiter.RPS(), c.limiter.Burst(), limiter.ErrRateLimited)
	}

	body, err := json.Marshal(chatRequest{
		Model:     c.model,
		Messages:  toChatMessages(messages),
		MaxTokens: opts.MaxTokens,
		Stream:    false,
	})
	if err != nil {
		return nil, fmt.Errorf("encode request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Tenant-Id", c.tenant)

	start := time.Now()
	resp, err := c.hc.Do(req)
	if err != nil {
		return nil, fmt.Errorf("gateway unreachable: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	var parsed chatResponse
	// A refusal body is JSON too, so decode before branching on status.
	_ = json.Unmarshal(raw, &parsed)

	meta := &Meta{
		Status:       resp.StatusCode,
		Replica:      resp.Header.Get("X-Replica"),
		PrefixMatch:  atoiSafe(resp.Header.Get("X-Prefix-Match-Tokens")),
		TotalLatency: time.Since(start),
	}
	if parsed.Error != nil {
		meta.Reason = parsed.Error.Reason
	}
	if meta.Reason == "" {
		meta.Reason = resp.Header.Get("X-Reason")
	}
	c.last.Store(meta)

	if resp.StatusCode >= 400 {
		return nil, &RefusedError{
			Status:     resp.StatusCode,
			Reason:     meta.Reason,
			RetryAfter: parseRetryAfter(resp.Header.Get("Retry-After")),
		}
	}
	if len(parsed.Choices) == 0 {
		return nil, fmt.Errorf("gateway returned no choices")
	}

	return &llms.ContentResponse{
		Choices: []*llms.ContentChoice{{
			Content:    parsed.Choices[0].Message.Content,
			StopReason: parsed.Choices[0].FinishReason,
			GenerationInfo: map[string]any{
				"replica":             meta.Replica,
				"prefix_match_tokens": meta.PrefixMatch,
				"total_ms":            meta.TotalLatency.Milliseconds(),
			},
		}},
	}, nil
}

// Call implements the deprecated half of llms.Model. langchaingo still requires
// it on the interface, so it delegates rather than duplicating logic.
func (c *Client) Call(ctx context.Context, prompt string, options ...llms.CallOption) (string, error) {
	return llms.GenerateFromSinglePrompt(ctx, c, prompt, options...)
}

// RefusedError is a structured gateway refusal: which policy said no, and how
// long the caller was asked to wait. Returned as an error value so callers can
// use errors.As to distinguish backpressure from a genuine failure -- the
// distinction that matters when deciding whether to retry.
type RefusedError struct {
	Status     int
	Reason     string
	RetryAfter time.Duration
}

func (e *RefusedError) Error() string {
	if e.Reason == "" {
		return fmt.Sprintf("gateway refused with status %d", e.Status)
	}
	return fmt.Sprintf("gateway refused: %s (status %d)", e.Reason, e.Status)
}

func toChatMessages(messages []llms.MessageContent) []chatMessage {
	out := make([]chatMessage, 0, len(messages))
	for _, m := range messages {
		var text strings.Builder
		for _, part := range m.Parts {
			if tc, ok := part.(llms.TextContent); ok {
				text.WriteString(tc.Text)
			}
			// Non-text parts (images, tool calls) are dropped: the lab's model
			// is text-only, and silently forwarding a stringified blob would
			// distort the token counts the gateway meters on.
		}
		out = append(out, chatMessage{Role: roleFor(m.Role), Content: text.String()})
	}
	return out
}

func roleFor(t llms.ChatMessageType) string {
	switch t {
	case llms.ChatMessageTypeSystem:
		return "system"
	case llms.ChatMessageTypeAI:
		return "assistant"
	default:
		return "user"
	}
}

func atoiSafe(s string) int {
	v, err := strconv.Atoi(s)
	if err != nil {
		return 0
	}
	return v
}

func parseRetryAfter(s string) time.Duration {
	if s == "" {
		return 0
	}
	secs, err := strconv.Atoi(s)
	if err != nil {
		return 0
	}
	return time.Duration(secs) * time.Second
}
