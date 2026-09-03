# inference-gateway

An LLM inference gateway in Go: one OpenAI-compatible endpoint in front of N vLLM
replicas, applying per-tenant quota, admission control, a bounded priority queue,
and prefix-cache-aware routing — plus a langchaingo agent pipeline to drive it.

It is a port of a Python teaching lab (FastAPI + CrewAI) rewritten to idiomatic
Go. The interesting parts are the places where the two languages disagree about
how to express the same idea, and the places where Go has no equivalent library
at all. Both are documented below and in the code.

## The request path

```
agent (langchaingo)                 gateway                          vLLM
─────────────────────    ────────────────────────────────    ──────────────────
Researcher ──┐
             ├─► client limiter ─► 1. tokenize ─► 2. quota ─► 3. admission ──┐
Writer ──────┘   (4 rps/burst 8)      │             │            │           │
                                      │          429 quota    503 no_signal  │
                                      │                       kv_pressure    │
                                      │                       queue_depth    │
                                      │                       no_headroom    │
                                      │                    deadline_unmeetable
                                      ▼                                      │
                              4. priority queue ─► 5. route ─► 6. proxy ──────┤
                                 │                  │                         │
                          503 queue_full      prefix trie                r0 / r1
                          503 expired_in_queue + load score
```

Every gate can only refuse a request sooner or make it wait longer; nothing
downstream can un-refuse it. Refusals carry a machine-readable `reason`, an
HTTP status (429 for your own budget, 503 for the fleet's), and a `Retry-After`.

Six stages:

| # | Stage | What it does |
|---|-------|--------------|
| 1 | tokenize | Count prompt tokens and derive the latency objective from the tenant's tier. |
| 2 | quota | Per-tenant token bucket, charged `n_in + n_out` up front. Always on. |
| 3 | admission | Five ordered checks against live fleet telemetry. Sheds what cannot be served in time. |
| 4 | queue | Bounded priority queue: earliest-deadline-first, short prompts favoured, with an anti-starvation cutoff. |
| 5 | route | Prefix-trie affinity scored against replica load, optionally sampling two replicas (P2C). |
| 6 | proxy | Forward to the chosen replica, relay SSE, record TTFT. |

## Quick start

```bash
# two vLLM replicas on :8001 and :8002
make build

./bin/gateway --preset full --p2c          # gateway on :8080
./bin/agent   --repeat 5                   # drive it with the agent pipeline

curl -s localhost:8080/health | jq
curl -s localhost:8080/_stats | jq '.fleet'
```

Presets bundle the three feature flags so a behaviour change can be attributed
to a specific mechanism:

| Preset | admission | queue | prefix routing |
|--------|-----------|-------|----------------|
| `baseline` | off | off | off |
| `route` | off | off | **on** |
| `queue` | off | **on** | **on** |
| `full` | **on** | **on** | **on** |

## Library mapping: Python → Go

Where a real equivalent exists it is used. Where none does, it is called out.

| Python | Go | Notes |
|--------|-----|-------|
| FastAPI | `net/http` + `http.ServeMux` | Go 1.22+ method-and-pattern routing. No dependency earns its place for four routes. Decorator registration becomes explicit `mux.HandleFunc` calls. |
| uvicorn | `net/http.Server` | No separate ASGI server; the server is the stdlib. |
| `httpx.AsyncClient` | `net/http.Client` | Direct equivalent. |
| `asyncio` | goroutines + channels | Different model, not a port: asyncio is cooperative on one thread, goroutines are preemptive and genuinely parallel. Every place the Python relied on the event loop for mutual exclusion needed a real lock here. |
| `asyncio.Condition` | channel + `sync.Mutex` | A `sync.Cond` maps 1:1 but cannot be `select`ed on alongside `ctx.Done()`, so the queue uses a capacity-1 notify channel instead. |
| `asyncio.Future` | buffered channel | One-shot handoff from dispatcher to waiter. |
| `collections.deque(maxlen=N)` | hand-rolled ring | **No stdlib equivalent.** ~30 lines over a preallocated slice. |
| `hashlib.blake2s` | `golang.org/x/crypto/blake2s` | Same primitive, but Go exposes only 256-bit output, so the 64-bit block hash truncates `Sum256`. Not bit-identical to Python's `digest_size=8`; irrelevant, the trie is per-process. |
| `struct.pack` | `encoding/binary` | Direct equivalent. |
| regex over `/metrics` | `github.com/prometheus/common/expfmt` | **Better than the original.** The real upstream parser instead of a hand-rolled line regex: histograms, label sets and metric types handled properly. |
| `argparse` | `flag` | Stdlib to stdlib. |
| `dataclasses` | structs | Direct. |
| `@property` | methods | Direct. |
| `pydantic` | `encoding/json` + explicit validation | **No equivalent.** Nothing in Go does runtime coercion, defaulting and validation from type annotations. Validation here is hand-written, which is more code but makes every rejection an explicit decision. |
| `transformers.AutoTokenizer` | *(none — see below)* | **No equivalent.** |
| CrewAI | `github.com/tmc/langchaingo` | **Closest analogue, not an equivalent — see below.** |

### Gap 1: no HuggingFace tokenizer

Go has no tokenizer of comparable fidelity. The options, none drop-in:

- `github.com/daulet/tokenizers` — cgo bindings to HuggingFace's Rust library. Faithful, drags cgo into the build.
- `github.com/sugarme/tokenizer` — pure Go, partial coverage, lags upstream.
- `github.com/tiktoken-go/tokenizer` — correct, but OpenAI BPE only.

The gateway needs token *counts* (quota, capacity) and *stable ids* (prefix block
hashing); it never detokenizes. So `internal/tokenize` ships the same
deterministic 4-rune fold the Python lab falls back to, behind a `Tokenizer`
interface. Swapping in a real tokenizer touches one line of wiring.

### Gap 2: no CrewAI

There is no Go equivalent of CrewAI's agents-with-roles / tasks / crew model.
`langchaingo` is the closest thing and is what `internal/agent` builds on — it
supplies the `llms.Model` interface, prompt templates and sequential chains.
What it does not supply is the role/goal/backstory framing or task delegation, so
"researcher" and "writer" are two chained LLM calls with distinct prompts rather
than two agents in a crew.

That is fine for what this repo measures: the gateway sees the same traffic
shape either way — two sequential calls per query, the second sharing a prompt
prefix with the first, which is exactly what makes prefix routing observable.

`agent.Client` implements `llms.Model`, so it drops into any langchaingo chain
or agent — the same role `GatewayLLM(BaseLLM)` plays for CrewAI in the original.
Note that langchaingo pulls a heavy transitive tree (sprig, gonja, starlark,
tiktoken-go) for what is ultimately two prompt templates.

One langchaingo behaviour worth knowing before you build on it:
`chains.SequentialChain` **replaces** the value map between steps
(`inputs = outputs`) rather than merging into it, so step two cannot read what
step one was given. LangChain's Python `SequentialChain` accumulates. In
practice the writer step could not see `user_query` and failed with
`missing key in input values`. `internal/agent/pipeline.go` is a ~40-line
`chains.Chain` implementation that threads an accumulating map instead; it
stays composable with `chains.Call`, `chains.Run` and memory.

## Deliberate deviations from the original

Not everything was ported literally. Where the Python shape would have been
unidiomatic or unsafe in Go:

- **Config is a struct, not module globals.** The original mutated
  `gateway/config.py`'s globals at startup and read them live from every module.
  In Go that is a data race and makes tests order-dependent, so `config.Config`
  is built once, validated, and passed down.
- **Policy layers are pure functions of a snapshot.** `admission.Decide` and
  `router.Pick` take a `state.Snapshot` copied under a read lock. The original
  read live mutable objects, which was safe only because of the event loop.
- **Errors are values.** Refusals are a typed `refuse.Reason` returned as an
  error, not `QueueFull`/`Expired` exceptions. Each reason declares its own HTTP
  status and `Retry-After` in one place, so a typo becomes a compile error.
- **Dispatch capacity is a semaphore, not a poll loop.** The original polled an
  in-flight counter every 5 ms; a buffered channel blocks precisely and wakes one
  waiter, with no interval to tune.
- **Cancellation is a `context.Context`.** Client disconnect propagates to the
  upstream request, so the replica stops generating tokens nobody will read. The
  original checked `request.is_disconnected()` between chunks and left the
  upstream running.
- **Abandoned queue slots are reclaimed.** A disconnected client's request is
  removed from the queue immediately rather than occupying capacity until its
  deadline.
- **The prefix trie is pruned.** The original never evicted expired nodes — they
  stopped matching but stayed resident, so memory grew with every distinct prefix
  ever seen. A background loop reclaims them.
- **Panics are contained.** An unrecovered panic in any goroutine kills the whole
  Go process, whereas an unhandled Python exception in one coroutine only failed
  that request. Hence the handler recover and the guard around metrics parsing.

## Security hardening

Issues in the original that the port fixes, and the defensive additions:

| Issue | Original behaviour | Here |
|-------|-------------------|------|
| **Quota bypass + unbounded map** | Any unseen `X-Tenant-Id` minted a *new full-burst* bucket, so rotating the header dodged quota entirely and grew the map without bound. | Tenant set fixed by config; unknown callers share one bucket. Pinned by `TestUnknownTenantsShareOneBucket`. |
| **Caller-triggered 500** | `X-Priority: abc` reached `int()` and raised inside the handler. | Unparseable hints are ignored. Pinned by `TestMalformedPriorityHeaderIsIgnored`. |
| **Deadline opt-out** | Client-supplied `deadline_ms` was used unclamped, so a large value disabled the deadline check and a negative one made every request unmeetable. | Clamped to the tenant's tier: a client may ask to be shed *sooner*, never later. Pinned by `TestClientCannotExtendItsDeadline`. |
| **Unbounded request body** | `await request.json()` with no size limit. | `http.MaxBytesReader`, 1 MiB default. |
| Header injection into logs | Raw tenant header used as a map key and log field. | Sanitised to a conservative charset, length-capped. |
| Slowloris | No server timeouts set. | `ReadHeaderTimeout`, `ReadTimeout`, `IdleTimeout`. No `WriteTimeout` by design — it would sever a streamed completion. |
| Upstream redirects | Followed by default. | `CheckRedirect` refuses them; replicas must answer directly. |
| Unbounded upstream reads | — | Response and metrics reads are `io.LimitReader`-bounded. |

Replica URLs come only from configuration and are validated at startup, so the
proxy has no SSRF surface. Upstream response bodies are relayed verbatim, which
is safe *specifically because* the upstreams are trusted; a gateway fronting
caller-supplied backends should not do that.

## Layout

```
cmd/gateway        gateway server entrypoint
cmd/agent          langchaingo pipeline that drives the gateway
internal/config    all tunables, presets, validation
internal/gateway   HTTP server, handlers, proxy and streaming
internal/admission load-shedding decision (pure)
internal/router    replica selection (prefix affinity vs load)
internal/trie      per-replica prefix trie with TTL and pruning
internal/pending   bounded priority queue with deadline expiry
internal/bucket    per-tenant token-bucket quota
internal/state     fleet telemetry and consistent snapshots
internal/scrape    Prometheus /metrics collection
internal/stats     fixed-size lifecycle event ring
internal/tokenize  Tokenizer interface + fallback implementation
internal/refuse    refusal reasons, statuses, retry hints
internal/agent     langchaingo llms.Model over the gateway
pkg/limiter        client-side request limiter
```

## Testing

```bash
make test     # unit + integration
make race     # race detector
make cover    # coverage summary
```

Policy layers are pure functions, so their tests are table-driven and instant.
`internal/gateway` runs the full request path against fake vLLM replicas that
serve both chat completions and Prometheus metrics, covering prefix affinity
across turns, each refusal path, semaphore accounting, and the hardening above.

## API

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/chat/completions` | OpenAI-compatible, streaming and non-streaming |
| `GET /v1/models` | Served model id |
| `GET /health` | Liveness plus current feature posture |
| `GET /_stats` | Lifecycle event ring and fleet telemetry |

Response headers on a dispatched request: `X-Replica`,
`X-Prefix-Match-Tokens`, `X-Outcome`, and `X-Reason` when something went wrong.
