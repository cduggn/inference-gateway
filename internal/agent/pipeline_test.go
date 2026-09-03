package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/memory"
	"github.com/tmc/langchaingo/schema"
)

// stubChain records what it was handed and emits one key.
type stubChain struct {
	in, out string
	seen    map[string]any
}

func (s *stubChain) Call(_ context.Context, inputs map[string]any, _ ...chains.ChainCallOption) (map[string]any, error) {
	s.seen = inputs
	return map[string]any{s.out: "from-" + s.out}, nil
}
func (s *stubChain) GetMemory() schema.Memory { return memory.NewSimple() }
func (s *stubChain) GetInputKeys() []string   { return []string{s.in} }
func (s *stubChain) GetOutputKeys() []string  { return []string{s.out} }

// TestPipelineAccumulatesValues is the reason this type exists: langchaingo's
// SequentialChain replaces the value map between steps, so a later step cannot
// read the original input. Every step here must still see user_query.
func TestPipelineAccumulatesValues(t *testing.T) {
	first := &stubChain{in: keyQuery, out: keyResearch}
	second := &stubChain{in: keyResearch, out: keyAnswer}

	p := newPipeline([]string{keyQuery}, []string{keyAnswer}, first, second)
	out, err := chains.Call(context.Background(), p, map[string]any{keyQuery: "why is TTFT high?"})
	if err != nil {
		t.Fatalf("pipeline failed: %v", err)
	}

	if _, ok := second.seen[keyQuery]; !ok {
		t.Error("second step could not see the original input; values were replaced, not merged")
	}
	if got := second.seen[keyResearch]; got != "from-"+keyResearch {
		t.Errorf("second step saw research = %v, want the first step's output", got)
	}
	if got := out[keyAnswer]; got != "from-"+keyAnswer {
		t.Errorf("answer = %v, want the last step's output", got)
	}
}

// TestPipelineErrorsNameTheFailingStep: two near-identical LLM calls are hard to
// tell apart in a stack trace, so the step index has to be in the message.
func TestPipelineErrorsNameTheFailingStep(t *testing.T) {
	ok := &stubChain{in: keyQuery, out: keyResearch}
	broken := &stubChain{in: "never-provided", out: keyAnswer}

	p := newPipeline([]string{keyQuery}, []string{keyAnswer}, ok, broken)
	_, err := chains.Call(context.Background(), p, map[string]any{keyQuery: "q"})
	if err == nil {
		t.Fatal("expected the second step to fail on its missing input")
	}
	if !strings.Contains(err.Error(), "step 2/2") {
		t.Errorf("error %q does not identify which step failed", err)
	}
}
