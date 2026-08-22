# API compatibility plan

Last updated: 2026-08-22

## Goal

Turn the validated Qwen3.6 image endpoint into a stable service integration
contract. The target for this phase is OpenAI Chat Completions compatibility,
including image input and SSE streaming. It is not a claim of compatibility
with the complete OpenAI API.

The implementation must continue to use ReInstinct as the only language-model
backend. Vision remains opt-in through `--mmproj` and `--mtmd-bridge`, with the
existing single-image safety limits preserved.

## Implementation status

The first implementation slice is complete: the standalone C#/.NET 8
black-box executable is in [`../tests/api-contract/`](../tests/api-contract/).
It uses only `HttpClient` and `System.Text.Json`, includes deterministic JPEG
and PNG data-URL fixtures, and covers the positive, negative, SSE, usage,
limit, timeout-recovery, and disconnect-recovery cases below. It has passed a
clean `dotnet build --configuration Release` locally.

The standalone suite builds cleanly, the focused Rust server tests pass, and
the deployed service has passed readiness and chat smoke checks. The live
deployment now uses the same JSON-backed settings and dashboard controls
described below. The broader named-client matrix remains a separate follow-up
when a client-specific compatibility claim is needed.

## Deployment status

The user-approved ReInstinct service is active on `ai@10.0.0.41` as
`reinstinct-server.service`, enabled at boot on port `8006`. It runs from the
isolated checkout `/home/ai/inference-bench/reinstinct-service-20260822` and
loads `/etc/reinstinct/server.json` by default. Furnace remains a separate
enabled-but-inactive unit on port `8000`; it was not restarted or modified.

The unit mirrors the applicable Furnace settings: the same Qwen3.6-35B-A3B
UD-Q4_K_XL model and F32 projector, a 32,000-token context, GPU vision, eight
vision threads, journal logging, `Restart=always`, `SIGINT` shutdown,
30-second stop timeout, unlimited memlock, and `LimitNOFILE=524288`.
ReInstinct does not expose Furnace's `--batch-size`, `--ubatch-size`,
`--parallel`, `--cache-reuse`, `-fa`, `--cont-batching`, `--no-mmap`, or
`--mlock` flags, so those settings have no direct equivalent in this unit.

The release binary and bridge were rebuilt in the new isolated checkout. The
live checks returned `200` from `/readyz`, reported the persisted config from
`/api/config`, and completed a text chat request with HTTP `200` after the
service restart. The installed unit backup is retained at
`/etc/systemd/system/reinstinct-server.service.bak-20260822` for rollback.

The 2026-08-22 deployment of commit `7ddf521` rebuilt the service from the
updated Rust sources on the MI50 host. After model load, `/readyz` returned
`200`, the dashboard exposed separate Save/Reload controls, the OpenAPI
document exposed `POST /api/config/reload`, and a text chat smoke request
returned HTTP `200`.

The dashboard exposes the **Engine configuration** panel when the service is
started with `--config`. **Save** validates and persists edits without
disturbing the running engine; **Reload** explicitly applies saved engine
settings while idle. Port edits are saved but reported as requiring a service
restart. If startup or the GPU worker fails, the listener stays available
and the dashboard shows the error and recent engine logs, so first diagnosis
does not require host-shell access.

On 2026-08-21 the deployment was rebuilt with incremental UTF-8 output
assembly. The exact Paperless-AI custom-provider validation request (`Test`,
with only `model` and `messages`) returned HTTP 200 and preserved the generated
emoji. The full C# contract suite, including that fixture, then passed all 18
checks against port 8006 after a service restart. Furnace remained inactive
and unchanged.

The endpoint currently binds openly on the server's network interface. It is
not a production exposure: authentication, TLS, CORS, rate limiting, and a
public request-size policy still belong behind a reverse proxy.

## Current contract

The server currently exposes:

- `GET /` for a live, browser-friendly server dashboard;
- `GET /api/status` for machine-readable model, worker, GPU, settings, and counter state;
- `GET /api/gpus` for PCI-keyed multi-GPU inventory, virtualization evidence,
  telemetry, and read-only/non-AMD capability reporting;
- `PUT /api/gpus/{pci_address}/power-limit`, `PUT
  /api/gpus/{pci_address}/tuning`, and `POST /api/gpus/{pci_address}/reset`
  for exact-identity AMD mutations through the optional privileged helper;
- `GET /api/config` for active/saved JSON settings, CLI-locked fields, pending reload fields, and reload state;
- `PUT /api/config` for validated atomic persistence without an engine reload;
- `GET /api/config/reload` plus `POST /api/config/reload` for reload state and explicit idle-only engine reloads;
- `GET /openapi.json` for the canonical OpenAPI 3.1 contract;
- `GET /docs` for interactive Swagger UI documentation;
- `GET /healthz` for liveness;
- `GET /v1/models` for model discovery;
- `POST /v1/completions` for raw text prompts;
- `POST /v1/chat/completions` for text chat and one structured image input;
- OpenAI-shaped non-streaming responses and errors;
- SSE streaming terminated by `data: [DONE]`, with optional usage chunks.
- UTF-8-safe output assembly when one character spans multiple generated
  byte-level tokens.

The dashboard also reports the cooperative thermal interlock. Its default
90°C/5-second/82°C policy is persisted in the JSON configuration and can be
changed live with the four `gpu_*` settings. Thermal hold time is retained as
`thermal_wait_ms` in run history and is not included in inference timing.

The OpenAPI document describes the implemented subset rather than claiming
the entire OpenAI API. Responses, Conversations, tool calling, and structured
output remain out of scope until their routes and semantics are implemented.

Image input currently means exactly one user `image_url` part containing a
base64 JPEG or PNG data URL. The decoded image is limited to 6 MiB and the HTTP
body to 8 MiB. Remote URLs, multiple images, non-user structured content,
video, and audio remain rejected.

## Phase 1: black-box compatibility suite

Build a small C# test executable or xUnit project that accepts a base URL and
runs only through HTTP. It must not link to ReInstinct internals. Keep fixtures
small and deterministic so the same suite can run against an isolated local
server, the MI50 server, and a reference OpenAI-compatible implementation.

Required positive cases:

1. Liveness and model discovery return stable, parseable responses.
2. Text-only chat works in non-streaming and streaming modes.
3. JPEG and PNG data-URL chat requests return coherent image-conditioned text.
4. The first chat SSE chunk carries the assistant role, subsequent chunks
   carry content deltas, optional usage is emitted when requested, and the
   stream ends with `data: [DONE]`.
5. Non-streaming and streaming usage fields, finish reasons, model IDs, object
   types, and content fields can be consumed by an OpenAI-style client.
6. Client disconnect and request timeout do not leave the worker unusable.

Required negative cases:

1. Malformed JSON, malformed base64, unsupported MIME types, empty images,
   remote URLs, and multiple images return an OpenAI-shaped 4xx error.
2. The decoded-image and request-body limits are enforced at their boundaries.
3. Image requests made without both vision startup options fail clearly.
4. Unknown routes, unavailable workers, and model-load failures return stable
   status codes and parseable error bodies.

At least one real target service or SDK must run against the same endpoint.
Record its name, version, configuration, request shape, and result. Passing
hand-written `curl` alone is not the completion gate.

## Phase 2: compatibility gaps

Compare failures from the black-box suite and real client against the declared
contract. Fix response shape, streaming, error handling, or harmless ignored
fields in the server when the behavior belongs to Chat Completions. Prefer a
small gateway when a caller needs translation into a different API family.

The following require explicit scope decisions and are not implied by this
phase:

- `POST /v1/responses`;
- tool or function calling;
- structured-output or JSON-schema enforcement;
- remote image fetching or multipart uploads;
- multiple images per turn;
- embeddings, video, or audio;
- OpenAI account authentication semantics.

## Phase 3: isolated deployment rehearsal

Deploy ReInstinct on a separate port without changing or restarting the
managed Furnace service. Use the matching BF16 projector unless compatibility
testing identifies a correctness regression. Keep the MI50 at the established
100 W test cap and retain external junction-temperature monitoring, cooldowns,
and the 88 C internal cutoff during large-image tests.

Put authentication, TLS, CORS policy, rate limiting, and public request-size
enforcement in a reverse proxy. ReInstinct endpoints are currently open and
should not be bound directly to an untrusted network.

The completed deployment verified startup, readiness, a live text request,
restart recovery, and dashboard/config observability without altering Furnace.
Streaming, image, timeout, and disconnect coverage remains owned by the C#
contract suite and its isolated-server runs. Public exposure is still a
separate reverse-proxy task.

## Completion criteria

This compatibility phase is complete when:

- the C# black-box suite passes against an isolated ReInstinct vision server;
- one named real service or SDK completes text, image, and streaming calls;
- supported and rejected request shapes are documented with stable examples;
- the deployed service and dashboard pass the startup/readiness/reload smoke
  checks without altering the managed Furnace service;
- thermal, timeout, request-size, authentication, and exposure policies are
  documented separately from inference performance;
- remaining unsupported OpenAI surfaces are listed as explicit follow-up
  decisions rather than described as compatible.

## Implementation order for the next session

1. Add the C# HTTP contract-test project and deterministic JPEG/PNG fixtures. **Done** — `tests/api-contract`.
2. Run it against an isolated ReInstinct server and capture the first failure
   matrix without changing the managed service. **Done for the isolated run.**
3. Fix Chat Completions compatibility gaps and add focused Rust unit tests for
   every server-side correction. **Done for the current supported subset.**
4. Validate a real target service or SDK. **Paperless-AI validation is recorded;**
   additional named-client matrices remain optional follow-up work.
5. Re-run the full Rust suite, the C# contract suite, and an MI50 image smoke.
   **Focused Rust tests, local C# build, remote release build, and live smoke
   are current deployment evidence.**
6. Keep this plan, `MULTIMODAL_PROGRESS.md`, and the API section of
   `MANUAL.md` synchronized with the supported contract and deployment state.
