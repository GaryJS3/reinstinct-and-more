# OpenAI API compatibility enhancements

Last updated: 2026-08-19

## Objective

Make ReInstinct usable as an OpenAI-compatible inference backend for real
self-hosted applications, initially:

- Paperless-ngx AI suggestions, document chat, and (later) RAG;
- paperless-gpt metadata extraction and vision OCR;
- Home Assistant local-LLM conversation integrations;
- SparkyFitness AI features;
- ordinary OpenAI SDK clients configured with a custom `base_url`.

Compatibility is defined by those clients completing their real workflows,
not merely by exposing similarly named routes or publishing an OpenAPI file.
The OpenAPI 3.1 document remains the machine-readable description of the
subset that has actually passed its compatibility gates.

## Rules for every implementation session

1. Read this file, `docs/API_COMPATIBILITY_PLAN.md`, and the relevant server
   section of `MANUAL.md` before editing.
2. Preserve unrelated working-tree changes. This roadmap and the existing API
   work may be uncommitted when another session begins.
3. Do not describe an ignored field as supported when client-visible behavior
   depends on it. For example, accepting `response_format` while returning
   arbitrary prose is not structured-output compatibility.
4. Add a Rust parser/shaper test and a black-box C# contract test for every
   server behavior change.
5. Test through HTTP. Unit tests alone do not establish client compatibility.
6. Do not restart or modify the Furnace service. Use an isolated checkout and
   port for MI50 validation.
7. Preserve the existing image limits: one user image, JPEG/PNG data URL,
   decoded size at most 6 MiB, and request body at most 8 MiB.
8. Keep unsupported features explicit in errors, status JSON, documentation,
   and the OpenAPI contract.

## Progress legend

- `[x]` implemented and locally validated;
- `[~]` partially implemented or implemented without its final live/client gate;
- `[ ]` not implemented;
- `[blocked]` depends on a separate runtime or a product decision.

## Current implementation inventory

### HTTP and discovery

- [x] `GET /healthz` liveness response.
- [x] `GET /metrics` Prometheus counters.
- [x] `GET /v1/models` with one advertised model per enabled LLM port.
- [x] OpenAI-shaped JSON errors for known request failures.
- [x] Finite HTTP limits: 8 MiB body, bounded headers, socket timeouts, bounded
  JSON nesting.
- [~] `GET /openapi.json` publishes OpenAPI 3.1, but it currently documents
  only the existing subset and has not been validated with an external schema
  validator or generated SDK.
- [~] `GET /docs` serves Swagger UI using CDN assets. The raw contract works
  offline; the interactive UI does not.

### Status and operations UI

- [x] `GET /` browser status page.
- [x] `GET /api/status` with service phase, endpoint, model path/ID, context
  size, drafter, vision configuration, queue depth, active request, counters,
  HIP device name/architecture, and VRAM use.
- [x] Bounded, memory-only run history (`GET /api/runs` and
  `GET /api/runs/{id}`) records queued/active/completed/error state, client IP,
  request and response captures, token counts, prefill, TTFT, generation time,
  and prompt/generation throughput. The dashboard polls it every two seconds
  and provides a compact list with a detail dialog.
- [x] Request captures are limited to 64 KiB, image data URL payloads are
  redacted, and only the latest 128 runs are retained. History resets when the
  process restarts; persistent storage and cross-restart aggregation are
  deliberately deferred.
- [x] `GET /api/status` and the dashboard report aggregate and last-completion
  prompt/generation tokens per second plus prefill, TTFT, and generation
  timing. Active generation progress is approximate until final usage is
  available.
- [ ] Add last error and prefix-cache statistics.
- [x] Add collapsible/filterable bounded engine logs (`GET /api/logs`) and a
  collapsible run section with retained/active/error counts.
- [x] Add Linux network receive/transmit telemetry plus model-load elapsed,
  estimated progress, and ETA based on model bytes and observed receive bytes.
- [x] Add a canonical-root model catalog (`GET /api/models`) with file size and
  projector association, plus idle-only in-process switching
  (`POST /api/models/switch`) with rollback to the previous model on failure.
- [x] Make catalog scans cacheable/refreshable, make model switching asynchronous
  with a status endpoint, add a readiness probe, and reject inference while a
  model is loading or switching.
- [ ] Add optional Linux hwmon temperature, power input/cap, and clock data.
  Missing hwmon fields must result in `null`, never endpoint failure.

### Completions

- [x] `POST /v1/completions` raw text, non-streaming and SSE.
- [x] `POST /v1/chat/completions` text chat with server-side templates.
- [x] Chat roles: `system`, `user`, and `assistant`.
- [x] Non-streaming `chat.completion` response with usage.
- [x] SSE `chat.completion.chunk`, initial assistant role, content deltas,
  finish reason, optional usage chunk, and `data: [DONE]`.
- [x] Common sampling fields, seed, timeouts, logprobs, and ReInstinct
  speculative-decoding extensions.
- [x] One Qwen JPEG/PNG data-URL image when vision startup options are present.
- [~] Common request fields are now classified for the first compatibility
  slice: `max_completion_tokens`, up to four `stop` strings, and `n=1` are
  implemented; `user`, `metadata`, and bearer auth are accepted/ignored;
  tools, functions, `parallel_tool_calls`, and `response_format` are rejected
  explicitly until their work packages land. The full field matrix remains.
- [ ] Function/tool calling.
- [ ] Enforced structured JSON output.
- [x] Stop sequences, `n`, `user`, and `metadata` now have explicit behavior;
  `n>1` and tool/structured-output fields return stable 400 errors.
- [ ] `service_tier` and other commonly emitted SDK fields still need an
  explicit behavior matrix.

### Other OpenAI API families

- [blocked] `POST /v1/embeddings` currently returns 503. A real embedding model
  runtime is required; synthetic vectors would silently corrupt Paperless RAG.
- [ ] `POST /v1/responses` is not implemented.
- [ ] Responses typed SSE events are not implemented.
- [ ] Responses retrieval/deletion/cancellation are not implemented.
- [ ] `/v1/conversations` and conversation items are not implemented.
- [ ] Audio, video, files, batches, fine tuning, moderation, and hosted tools
  are out of scope for the named applications.

## Application acceptance matrix

| Application | Required surface | Important behaviors | Current state |
|---|---|---|---|
| Paperless-ngx suggestions/chat | Chat Completions | long text prompts, JSON results, timeout up to 120 s, configurable model/base URL | Partial: Paperless-AI custom-provider validation passes live; JSON enforcement unproven |
| Paperless-ngx RAG | Embeddings plus Chat Completions | deterministic fixed-width vectors, batch input, correct model/dimensions | Blocked: embeddings returns 503 |
| paperless-gpt metadata | Chat Completions | generic `OPENAI_BASE_URL`, reliable JSON, longer documents | Partial |
| paperless-gpt OCR/vision | Chat Completions image input | base64 image content, JSON result, predictable error behavior | Partial: one JPEG/PNG works; real client gate missing |
| Home Assistant Local OpenAI LLM | Streaming Chat Completions and tools | `tools`, `tool_choice`, multi-round tool results, streamed `tool_calls`, long contexts | Missing tool calling |
| Extended OpenAI Conversation | Chat Completions functions/tools | legacy `functions`/`function_call` may be sent | Missing |
| SparkyFitness | Chat Completions, likely JSON | exact source fixtures still required | Discovery incomplete |
| OpenAI SDK smoke | Models and Chat Completions | auth header tolerated, sync/async, stream parsing, error decoding | Not yet run |

Reference findings:

- Paperless-AI's custom-provider startup probe sends only the configured model
  and one user message containing `Test`. That exact fixture passes against
  the isolated port-8006 deployment, including generated multibyte Unicode.
- Paperless-ngx calls its backend `openai-like` and separately configures an
  OpenAI-compatible embeddings backend for RAG.
- paperless-gpt explicitly supports custom OpenAI-compatible base URLs.
- Home Assistant's official OpenAI integration is restricted to OpenAI's own
  endpoint; validation should target a local-compatible integration such as
  Local OpenAI LLM or Home LLM. These require tool calling for entity control.

## Work package 0: freeze client fixtures

**Priority:** P0. **Can run independently:** yes. **Primary files:** new files
under `tests/api-contract/fixtures/` and this roadmap only.

- [ ] Pin tested versions/commits for Paperless-ngx, paperless-gpt, Local
  OpenAI LLM/Home LLM, and SparkyFitness.
- [ ] Extract sanitized request bodies and relevant headers directly from
  source or captured local traffic.
- [ ] Record whether each caller appends `/v1`, which model-discovery route it
  calls, and whether it sends a dummy bearer token.
- [ ] Record exact `response_format`, tool schema, stop, token-limit, image,
  and streaming fields.
- [ ] Store one fixture per workflow with secrets and document content removed.
- [ ] Add a short provenance manifest with project, version, source path, and
  expected behavior.

**Gate:** every row in the acceptance matrix has either a concrete fixture or
an explicitly documented reason it cannot be obtained.

## Work package 1: request compatibility and error semantics

**Priority:** P0. **Depends on:** WP0 fixtures. **Primary file:**
`src/serve/mod.rs`. Avoid editing `src/serve/api.rs` in this package.

- [x] Introduce typed internal request options for the implemented common
  fields instead of silently coercing them.
- [x] Classify the first common fields into supported, harmlessly ignored, and
  rejected categories; the remaining SDK field matrix is still open.
- [x] Accept both `max_tokens` and `max_completion_tokens`, rejecting requests
  that send both.
- [x] Implement up to four text stop strings, stopping without returning the
  matched suffix, including token-boundary and streamed-boundary matches.
- [x] Require `n` to be absent or `1`; larger values return a stable 400.
- [x] Accept `user` and `metadata` attribution fields as ignored, because they
  do not affect local inference semantics.
- [x] Validate a supplied requested `model` against the exact loaded ID; model
  omission remains accepted for legacy clients.
- [x] Accept `Authorization: Bearer ...` without requiring a key when auth is
  disabled. Authentication policy remains a reverse-proxy responsibility.
- [x] Add `GET /v1/models/{model}` with exact-ID lookup and `model_not_found`.
- [x] Ensure all generated 4xx/5xx JSON replies have `error.message`,
  `error.type`, `error.param`, and `error.code` fields.

**Gate:** all fixture requests parse as intended, invalid variants produce
stable OpenAI error objects, and existing text/image/SSE tests remain green.

## Work package 2: structured JSON output

**Priority:** P0 for Paperless. **Depends on:** WP1 parser. **Primary files:**
new `src/serve/structured.rs`, focused integration points in `src/serve/mod.rs`.

- [ ] Support `response_format: {"type":"json_object"}`.
- [ ] Determine which target applications require
  `response_format: {"type":"json_schema", ...}` and record their schemas.
- [ ] Add a grammar/token-mask mechanism that guarantees syntactically valid
  JSON during decode. Prompting alone is not compatibility.
- [ ] For `json_schema`, implement only a documented JSON Schema subset first:
  object, array, string, number/integer, boolean, null, enum, required,
  properties, items, and `additionalProperties: false`.
- [ ] Reject unsupported schema keywords with a precise 400.
- [ ] Ensure thinking-marker filtering cannot corrupt structured output.
- [ ] Define streaming behavior: deltas must concatenate into the same valid
  final JSON produced by non-streaming mode.
- [ ] Add representative Paperless metadata fixtures and malformed-model-output
  tests.

**Gate:** 100 repeated deterministic fixture runs parse as JSON and satisfy the
supported schema; no post-hoc repair is needed.

## Work package 3: OpenAI tool/function calling

**Priority:** P0 for Home Assistant. **Depends on:** WP1; may share grammar
infrastructure with WP2. **Primary files:** new `src/serve/tools.rs`, chat
template code, sampling/decode integration, and response shapers.

- [ ] Parse modern `tools: [{"type":"function","function":...}]`.
- [ ] Parse `tool_choice` values `none`, `auto`, `required`, and named function.
- [ ] Parse `parallel_tool_calls`; initially support false or a single call and
  return a clear compatibility error when multiple calls are requested.
- [ ] Accept assistant messages containing `tool_calls` and subsequent
  `role: "tool"` messages with `tool_call_id`.
- [ ] Add legacy `functions` and `function_call` translation if WP0 shows an
  active target still uses them.
- [ ] Extend chat templates with the model-family-specific tool definitions and
  tool-result turns. Do not inject a generic format without testing the Qwen
  and Gemma templates/models.
- [ ] Parse or constrain model output into OpenAI tool calls with stable IDs,
  names, and JSON argument strings.
- [ ] Non-streaming responses must use `finish_reason: "tool_calls"` and
  `message.tool_calls` with `content: null` when appropriate.
- [ ] Streaming must emit `delta.tool_calls[index].{index,id,type,function}` and
  incremental `function.arguments`, followed by `finish_reason: "tool_calls"`.
- [ ] Preserve the worker across the multi-round sequence; the client remains
  responsible for executing tools and sending results back.
- [ ] Add Home Assistant-shaped fixtures with many entity tools and long system
  prompts. Measure prompt tokens and TTFT separately from decode.

**Gate:** a real Home Assistant local-compatible integration can read an
entity, invoke an allowed service through its own tool executor, receive the
tool result, and produce a final streamed natural-language response.

## Work package 4: multimodal client parity

**Priority:** P0 for paperless-gpt. **Can run alongside:** WP2/WP3 if it avoids
their parser regions. **Primary files:** multimodal parser/tests/docs.

- [ ] Capture paperless-gpt's exact image content shape and MIME behavior.
- [ ] Accept equivalent OpenAI content naming used by current SDKs where safe,
  including `detail` as a harmless ignored hint if present.
- [ ] Verify multi-page workflows: keep one image per request unless the client
  truly requires multiple images; do not raise safety limits preemptively.
- [ ] Confirm PNG, JPEG, grayscale, rotated scan, large document page, and
  client timeout behavior.
- [ ] Pair vision with structured JSON output for OCR/metadata workflows.
- [ ] Ensure image errors preserve OpenAI error shape and do not poison the next
  request.

**Gate:** a pinned paperless-gpt build completes one OCR and one metadata update
against isolated ReInstinct without request rewriting.

## Work package 5: embeddings

**Priority:** P1; required for full Paperless-ngx RAG. **Independent runtime
spike:** yes. **Primary files:** new encoder model/runtime modules; minimize
overlap with `src/serve/mod.rs` until the runtime contract is settled.

- [ ] Select a supported embedding architecture/model that fits alongside the
  serving LLM or document the need for a separate process/GPU.
- [ ] Implement tokenizer, pooling, and normalization matching the selected
  model's reference implementation.
- [ ] Support string input and arrays of strings.
- [ ] Return `object: "list"`, ordered `data[].{object,index,embedding}`,
  `model`, and usage.
- [ ] Decide support for token-array input only after fixtures show a need.
- [ ] Validate vector dimension, normalization, determinism, ordering, empty
  input errors, batching, and context truncation/rejection.
- [ ] Measure memory residency and interference with LLM TTFT/decode.
- [ ] Replace the current unconditional 503 only after numerical parity tests
  pass against the reference encoder.

**Gate:** Paperless-ngx builds an index, retrieves a known relevant document in
a deterministic corpus, and answers a RAG question using ReInstinct for both
embeddings and generation.

## Work package 6: Responses API adapter

**Priority:** P2 unless a pinned target client proves it is needed earlier.
**Depends on:** stable chat, structured output, and tool-call internal types.

- [ ] Implement `POST /v1/responses` for string input and message-item arrays.
- [ ] Map `instructions` to the internal system/developer instruction layer.
- [ ] Return a proper Response object with `status`, `output` message items,
  `output_text` content, usage, and errors.
- [ ] Implement typed SSE at minimum for response created/in-progress,
  output-item/content-part creation, output-text delta/done, item done, and
  response completed/failed.
- [ ] Add tool-call items only after WP3 is complete.
- [ ] Consider `previous_response_id` using a bounded in-memory store with
  eviction; do not imply durable storage.
- [ ] Implement retrieval/deletion only if state is stored.
- [ ] Keep `/v1/conversations` separate. It is durable application state, not a
  prerequisite for stateless Responses compatibility.

**Gate:** official OpenAI SDK sync, async, and streaming Responses calls pass
against the supported subset without custom response parsing.

## Work package 7: contract, generated spec, and client harnesses

**Priority:** continuous. **Primary ownership:** `tests/api-contract/`, new
client harness directories, `src/serve/api.rs`.

- [ ] Split the C# black-box suite into named groups: core, image, structured,
  tools, embeddings, Responses, resilience, and application fixtures.
- [ ] Add CLI switches so sessions can run a safe subset against endpoints
  without vision or embeddings.
- [ ] Add an official OpenAI .NET SDK smoke harness; keep credentials out of
  command-line arguments and logs.
- [ ] Add a small Go harness only if needed to exercise paperless-gpt's exact SDK
  behavior; otherwise replay its captured HTTP fixtures from C#.
- [ ] Validate `openapi.json` with an OpenAPI 3.1 validator.
- [ ] Add explicit schemas for successful responses, SSE event payloads,
  errors, status, tools, and embeddings. Do not leave `{}` placeholders for
  completed work packages.
- [ ] Ensure the OpenAPI document lists limitations and vendor extensions
  (`x-reinstinct-*`) without presenting them as OpenAI fields.
- [ ] Record application/version/result in the matrix after each real-client
  run.

## Work package 8: isolated MI50 deployment and soak

**Priority:** final gate per milestone. **Do not modify Furnace.**

- [ ] Build in an isolated remote checkout and serve on an unused port.
- [ ] Verify the NFS model mount before diagnosing load hangs.
- [ ] Retain the host's reversible power cap, per-request junction monitoring,
  cooldowns, and 88 C internal cutoff for large-image tests.
- [ ] Run cold start, warm text, image, structured JSON, tools, timeout,
  disconnect, and restart recovery.
- [ ] Run each named application against the isolated endpoint.
- [ ] Run a bounded soak with mixed requests; report queue latency, prefill,
  TTFT, decode, failures, VRAM, temperature, and power separately.
- [ ] Verify shutdown releases VRAM.
- [ ] Update the dashboard/status endpoint from observed operational gaps.

## Recommended parallel session allocation

To reduce merge conflicts, use these boundaries:

1. **Client-fixture session:** WP0 and application matrix only; owns
   `tests/api-contract/fixtures/` and this file.
2. **Core-request session:** WP1; owns parser and error portions of
   `src/serve/mod.rs` plus focused Rust tests.
3. **Structured-output design session:** WP2; starts in a new module and sends
   a narrow integration patch after its decoder tests pass.
4. **Tool-calling research session:** WP3 template/output protocol and fixtures;
   avoid editing the same parser lines until WP1 lands.
5. **Vision parity session:** WP4 and paperless-gpt live fixture; owns vision
   tests/docs, coordinating any parser changes with the core session.
6. **Embeddings spike:** WP5 model selection and numerical oracle; should not
   change the live `/v1/embeddings` route until the runtime is credible.
7. **Contract/spec session:** WP7 C# harness organization and OpenAPI schemas;
   consume landed server behavior rather than advertising planned behavior.

Only one session at a time should own `src/serve/mod.rs`. New functionality
should live in focused modules so later integration patches stay small.

## Milestones

### M1: Paperless generation and paperless-gpt

- WP0 fixtures complete for both projects.
- WP1 request compatibility complete.
- WP2 JSON object/schema subset complete for actual fixtures.
- WP4 real image workflows pass.
- Official OpenAI SDK Chat Completions smoke passes.

### M2: Home Assistant control

- WP3 modern tools complete.
- Legacy functions added only if an active integration needs them.
- Real streamed read/action/result/final-response workflow passes.
- Long tool prompts stay within configured context and have measured TTFT.

### M3: Paperless RAG

- WP5 numerical parity and deployment policy complete.
- Real indexing, retrieval, and document-chat workflow passes.

### M4: Responses compatibility

- WP6 supported subset complete.
- Official OpenAI SDK Responses sync/async/streaming tests pass.
- OpenAPI contract accurately describes the completed surface.

## Evidence required before claiming compatibility

For each application, record:

- application version or commit;
- ReInstinct commit;
- endpoint/port and model/projector IDs;
- relevant application configuration with secrets removed;
- request family and streaming mode;
- pass/fail for discovery, generation, JSON parsing, tools, images, embeddings,
  timeout recovery, and disconnect recovery as applicable;
- prompt tokens, prefill, TTFT, decode throughput, total wall time, and VRAM;
- known limitations and whether the application can operate within them.

The headline may say "OpenAI-compatible" only with an immediately adjacent
supported-surface statement. Never imply compatibility with the complete
OpenAI platform.
