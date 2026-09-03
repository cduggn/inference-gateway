package agent

import (
	"context"
	"fmt"
	"maps"

	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/memory"
	"github.com/tmc/langchaingo/schema"
)

// pipeline runs chains in order, threading an accumulating value map so every
// step can see the original inputs as well as everything produced before it.
//
// This exists because langchaingo's own chains.SequentialChain does not do
// that. Its Call loop ends each iteration with `inputs = outputs`, replacing the
// value map rather than merging into it, so step two cannot read the value that
// was passed to step one -- here, the writer could not see user_query and failed
// with "missing key in input values". LangChain's Python SequentialChain
// accumulates, so this is a genuine behavioural difference between the two
// libraries and not a porting mistake.
//
// Implementing chains.Chain rather than hand-rolling a loop keeps the result
// composable: it still works with chains.Call, chains.Run and memory.
type pipeline struct {
	steps      []chains.Chain
	inputKeys  []string
	outputKeys []string
	mem        schema.Memory
}

// newPipeline builds an accumulating sequential chain.
func newPipeline(inputKeys, outputKeys []string, steps ...chains.Chain) *pipeline {
	return &pipeline{
		steps:      steps,
		inputKeys:  inputKeys,
		outputKeys: outputKeys,
		mem:        memory.NewSimple(),
	}
}

// Call implements chains.Chain.
func (p *pipeline) Call(ctx context.Context, inputs map[string]any, options ...chains.ChainCallOption) (map[string]any, error) {
	values := maps.Clone(inputs)
	if values == nil {
		values = map[string]any{}
	}
	for i, step := range p.steps {
		out, err := chains.Call(ctx, step, values, options...)
		if err != nil {
			// Name the step: with two near-identical LLM calls, "step 2 failed"
			// is the difference between a one-minute and a ten-minute debug.
			return nil, fmt.Errorf("pipeline step %d/%d: %w", i+1, len(p.steps), err)
		}
		maps.Copy(values, out)
	}
	return values, nil
}

// GetMemory implements chains.Chain.
func (p *pipeline) GetMemory() schema.Memory { return p.mem }

// GetInputKeys implements chains.Chain.
func (p *pipeline) GetInputKeys() []string { return p.inputKeys }

// GetOutputKeys implements chains.Chain.
func (p *pipeline) GetOutputKeys() []string { return p.outputKeys }

var _ chains.Chain = (*pipeline)(nil)
