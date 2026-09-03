package agent

import (
	"context"

	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/prompts"
)

// Keys used to thread values between chain steps.
const (
	keyQuery    = "user_query"
	keyResearch = "research"
	keyAnswer   = "answer"
)

// Crew runs a query through one or two model calls via the gateway.
type Crew struct {
	chain chains.Chain
	// dual reports whether this crew makes two calls per query.
	dual bool
}

// NewCrew builds the pipeline.
//
// The two-step variant is the interesting one for the gateway: the writer's
// prompt repeats the user's question verbatim, so its prefix overlaps the
// researcher's. With prefix routing on, both calls should land on the same
// replica and the second should report a non-zero X-Prefix-Match-Tokens. That
// is the signal the whole routing layer exists to produce, and it only shows up
// if the second call is actually similar to the first -- which is why the
// prompts share a prefix by construction rather than by luck.
func NewCrew(model llms.Model, dual bool) (*Crew, error) {
	if !dual {
		single := chains.NewLLMChain(model, prompts.NewPromptTemplate(
			"You are an inference assistant. Answer through the local serving gateway.\n\n"+
				"The user asked: {{.user_query}}\n\nGive a short, helpful answer in 1-3 sentences.",
			[]string{keyQuery},
		))
		single.OutputKey = keyAnswer
		return &Crew{chain: single, dual: false}, nil
	}

	researcher := chains.NewLLMChain(model, prompts.NewPromptTemplate(
		"You are a research analyst. Extract key facts.\n\n"+
			"The user asked: {{.user_query}}\n\nList 3 key facts about this topic as bullet points.",
		[]string{keyQuery},
	))
	researcher.OutputKey = keyResearch

	writer := chains.NewLLMChain(model, prompts.NewPromptTemplate(
		"You are a technical writer. Turn research notes into a clear answer.\n\n"+
			"The user asked: {{.user_query}}\n\nResearch notes:\n{{.research}}\n\n"+
			"Using the notes, write a clear 2-sentence answer.",
		[]string{keyQuery, keyResearch},
	))
	writer.OutputKey = keyAnswer

	// Not chains.NewSequentialChain: it replaces the value map between steps,
	// so the writer would not be able to see user_query. See pipeline.go.
	seq := newPipeline([]string{keyQuery}, []string{keyAnswer}, researcher, writer)
	return &Crew{chain: seq, dual: true}, nil
}

// Result is one query's outcome.
type Result struct {
	Query  string
	Answer string
	Err    error
	// Last is the gateway's routing metadata for the final call.
	Last *Meta
}

// Run executes the pipeline for one query.
func (c *Crew) Run(ctx context.Context, client *Client, query string) Result {
	out, err := chains.Call(ctx, c.chain, map[string]any{keyQuery: query})
	res := Result{Query: query, Err: err, Last: client.LastMeta()}
	if err != nil {
		return res
	}
	if answer, ok := out[keyAnswer].(string); ok {
		res.Answer = answer
	}
	return res
}

// Dual reports whether this crew issues two model calls per query.
func (c *Crew) Dual() bool { return c.dual }
