# Multimodal progress

Last updated: 2026-08-22

Canonical speed results and the active bottleneck queue are tracked in
[`PERFORMANCE.md`](PERFORMANCE.md).

## Verified on `ai@10.0.0.41`

- Furnace service is intentionally stopped for development. It is enabled but
  inactive; do not restart it until explicitly requested.
- Rust test suite: 193 passed, 1 ignored.
- `libreinstinct_mtmd.so` builds against ROCm 7.1.1 / gfx906.
- Qwen3.6 35B vocab-only model plus `mmproj-F32.gguf` successfully produced:
  - M-RoPE enabled.
  - Three chunks: text, image, text.
  - Image: 752 physical embeddings, 47 logical positions, 2048 floats per
    embedding.
- Deterministic hashes: embeddings `177d7d5454709f31`; positions
  `e2cf9a1b094aa13e`.
- Qwen runtime state now separates physical `kv_pos` from logical
  `rope_pos`, including reset and speculative snapshot/restore.
- Scalar text decode remains graph-capturable through separate device KV and
  RoPE positions.
- CPU and gfx906 four-plane M-RoPE implementations agree exactly on the MI50.
- External image prefill consumes explicit decoder positions, advances one KV
  row per embedding and advances logical state by the chunk's `n_pos`.
- Ordered text/image/text chunk prefill is exposed through one OpenAI
  structured-content `image_url` part when the vision options are enabled.
- `vision-test` runs the bridge and full ReInstinct transformer, prints
  per-stage timing, top first logits, physical/logical state, and greedy text.
- Bridge ABI v2 splits mtmd processing into image decode, media tokenization,
  vision/projector execution, and result-copy time. The warm HTTP path logs
  those fields alongside multimodal prefill, TTFT, decode, physical rows, and
  logical positions for every image request.
- The deployed `reinstinct-server.service` now runs from the isolated
  `reinstinct-service-20260822` checkout and loads `/etc/reinstinct/server.json`
  by default. The dashboard validates and persists supported settings, then
  reloads the engine while idle; port changes remain a service-restart concern.
- Startup and GPU-worker failures remain visible through the dashboard and
  `/api/logs` while the HTTP listener stays available for diagnosis.

## Important discovery

On the pinned Furnace revision, `llama_model_n_embd_inp()` and
`llama_model_n_embd()` both return zero for a `vocab_only` model. The bridge
therefore receives the embedding width parsed by ReInstinct from the same GGUF
metadata. This preserves the no-second-LLM requirement.

## Current gate

End-to-end CLI image generation and the isolated MI50 server smoke test pass
against the reproducible llama.cpp `tools/mtmd/test-1.jpeg` fixture.
Server-side OpenAI structured content accepts one finite `data:` JPEG/PNG
image, with a 6 MiB decoded-image cap and no remote URLs. It requires
`--mmproj` plus `--mtmd-bridge`. The smoke request returned HTTP 200 with a
coherent description; the managed Furnace service was kept stopped throughout.

The next gate is service compatibility rather than another engine-only image
optimization. The C# black-box suite covers discovery, text and image chat,
non-streaming and SSE response shapes, usage, stable errors, request limits,
timeouts, and client disconnects. The deployed dashboard/config reload path
is now part of the operations contract; additional named-client matrices and
reverse-proxy exposure remain follow-up work. Full scope and acceptance
criteria are in [`API_COMPATIBILITY_PLAN.md`](API_COMPATIBILITY_PLAN.md).

The standalone C#/.NET 8 HTTP contract suite is implemented at
`tests/api-contract`. It is dependency-free, uses deterministic JPEG/PNG
fixtures, and validates the declared wire shapes plus error, limit, timeout,
and disconnect-recovery behavior. It has passed a clean local Release build;
the isolated-server run and deployment rehearsal are complete for the current
supported subset. The live service is `reinstinct-server.service` on
`ai@10.0.0.41:8006`, enabled at boot, with the Qwen3.6-35B-A3B
model/projector and 32k context. After the deployment restart, `/readyz`
returned HTTP 200, `/api/config` reported the persisted JSON path, and a text
chat smoke returned HTTP 200. The managed Furnace service remains enabled but
intentionally inactive and was not changed. Public exposure is still gated on
a reverse proxy for authentication, TLS, CORS, rate limiting, and request-size
policy.

## Latest validation

- Remote `cargo check`: passed.
- Remote MI50 library suite: 184 passed, 1 ignored.
- Dedicated gfx906 M-RoPE CPU/GPU parity test: passed.
- Qwen3.5 4B token vs scalar-external vs broadcast-M-RoPE logits: passed.
- Qwen3.6 35B real-image run: 316 physical KV rows, 36 logical positions,
  3.891 s LLM prefill, and 75.4 tok/s decode.
- Furnace and ReInstinct used the same `/apply-template` prompt and produced
  the same first eight greedy tokens: `The user wants a description of the provided`.
- Furnace's chat endpoint does not expose raw logits; greedy-token parity is
  the cross-engine correctness evidence for this fixture.
- Server request-body tests verify that an exact finite body is read and a
  declared body larger than the 8 MiB HTTP cap is rejected before reading.
- Isolated MI50 server smoke: the OpenAI structured JPEG request used 317
  physical prompt rows and generated 16 tokens with HTTP 200.

## Image-workload and thermal validation

The image path was exercised through an isolated ReInstinct server on MI50
with GPU vision enabled. Every request used the same blind user prompt,
`Describe this image in detail. Answer directly without explaining your
reasoning.`, temperature zero, and a 64-token cap. No prompt named the image
contents.

- The default 178 W cap and a 150 W cap could not safely finish the larger
  60 x 34 vision-grid images with the 90 C junction guard enabled.
- A direct write to the MI50 hwmon `power1_cap` set a reversible 100 W cap;
  this is a host tuning change, not an inference-code change.
- At 100 W, the large beach image completed in 42.80 s (2,063 prompt / 64
  completion tokens) and the large surveillance image completed in 41.88 s
  (2,063 / 64). Their observed junction peaks were 84 C and 87 C.
- A 943-token beach image completed in 18.82 s. A 2,151-token cookie image
  completed in 44.31 s and peaked at 89 C. An 823-token property-agreement
  image completed in 16.23 s.
- The 2.82 MiB screenshot still reached the 90 C guard at 100 W after
  45.94 s; the guarded suite stopped before the remaining images. The server
  was stopped and VRAM released after the guard event.

These timings are end-to-end and therefore include image decode/preprocess,
vision projector work, multimodal LLM prefill, and generation. They are not
decode-only throughput. The primary performance gate is the vision/prefill
phase: a single image expands to hundreds or thousands of prompt rows before
the 35B language model can emit its first token.

## Bottleneck profiling workflow

Use `vision-test` for a cold, single-image breakdown and an isolated server for
repeatable warm measurements. The managed Furnace service remains out of scope.
The server emits one `vision profile` log line per completed image request with:

- `decode_image_ms`, `tokenize_ms`, `projector_ms`, `copy_ms`, and total
  `mtmd_ms`;
- `prefill_ms`, physical `rows`, and `logical_pos`;
- `ttft_ms`, `decode_ms`, and `generated` token count.

Compare identical image bytes, prompt, output cap, vision device, and power cap.
Discard the first request when comparing steady-state GPU performance. Record
junction temperature, memory temperature, and package power externally with
the existing guarded ROCm sampler; hardware telemetry remains host policy and
is intentionally not collected or controlled by the inference process.

## Batched multimodal prefill

The scalar compatibility path was identified as the dominant first-token
bottleneck: it evaluated every projected image row as a separate decode step,
synchronized the GPU, and downloaded logits for every row. The optimized path
now sends all projected rows through ReInstinct's batched prefill machinery,
using an explicit batched four-plane M-RoPE kernel. It also batches the text
chunks surrounding the image and downloads logits only after each complete
chunk.

The new gfx906 M-RoPE batch kernel passes an exact CPU oracle. A matched MI50
run on the 943-row beach fixture reduced multimodal prefill from 17.218 s to
5.664 s and total request time from about 18.5 s to 7.013 s. The mtmd projector
still consumed 3.743 s, leaving only about 1.855 s in ReInstinct's batched LLM
prefill. TTFT improved from 17.223 s to 5.670 s, while 64-token decode remained
about 1.3 s. The coherent greedy result described the same woman swinging over
turquoise water, though fp16 batched-prefill drift changed the exact wording
from the scalar reference. Peak junction temperature was 59 C under the 88 C
guard, and the isolated server released VRAM after the run.

## Projector precision and token budgets

Bridge ABI v3 now passes dynamic image-token minimum/maximum limits into
libmtmd. Diagnostics and the HTTP server expose explicit CLI options that
default to projector metadata, preserving existing behavior.

On the 943-row beach fixture, the matching Qwen3.6 35B BF16 projector reduced
projector time from 3.778 s to 1.611 s and end-to-end time from 7.048 s to
4.977 s. The coherent output kept identical image positions. `rocprof`
identified F32 GEMM as 89% of projector GPU time; BF16 reduced captured GEMM
time from 3.309 s to 0.941 s while vision FlashAttention stayed near 0.330 s.

F32 tests at default / 768 / 512 / 384 maximum image tokens produced 920 /
720 / 480 / 364 embeddings in 3.874 / 2.997 / 1.981 / 1.497 seconds. Caps
therefore provide predictable latency control with an explicit quality tradeoff.

The added server fan improved single-request thermals and the batched large
suite completed at 100 W. A guarded back-to-back repeat still reached the 88 C
internal junction cutoff, so the 125 W comparison was canceled. Large requests
continue to require cooldown and active monitoring.

## Commits

`a8cb383`, `2aad59a`, `ecb5cdd`, `4f02753`, `3137ee8`, `ffe6e07`, and
`0017cd0`, `ebe64d7`, `35e7326`, and `d9c86a8` on
`feature/multimodal-mtmd`.
