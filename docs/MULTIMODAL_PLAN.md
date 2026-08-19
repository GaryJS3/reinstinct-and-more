# Multimodal implementation plan

## Objective

Accept one Qwen3.6 image and prompt while keeping ReInstinct as the only LLM
transformer backend. Furnace's pinned `libmtmd` owns image decoding,
preprocessing, vision encoding, projection, media tokenization, and position
metadata; it must not evaluate language-model layers.

## Completed foundation

- Pin `third_party/llama.cpp` to Furnace `5013b9f91d3cbe627a758336802b1a2223427b49`.
- Build the optional `libreinstinct_mtmd` C ABI bridge and load it through
  Rust `libloading`.
- Load the text GGUF vocab-only; pass ReInstinct's parsed hidden width because
  this pinned llama.cpp revision leaves its embedding-width getters unset in
  vocab-only mode.
- Decode an encoded image buffer, tokenize media markers, encode/project to
  copied FP32 embeddings, and return text/image chunks plus M-RoPE positions.
- Add `mtmd-test` and external-embedding parity coverage.

## Position-correct external prefill

Completed locally and validated on gfx906:

1. Split `Qwen35GpuState.pos` into physical `kv_pos` and logical `rope_pos`.
   Split device `d_pos` so graph-captured KV writes use physical position while
   RoPE reads independently supplied logical positions.
2. Preserve the contiguous scalar text fast path. Add explicit image position
   buffers only for multimodal external embeddings.
3. Implement the pinned llama.cpp Qwen four-plane M-RoPE semantics using
   `rope.dimension_sections`; CPU and gfx906 tests cover text-broadcast and
   independent positions. Captured-image parity remains pending.
4. Process bridge chunks in order: token prefill for text and external
   embedding prefill for image. Advance physical position by `n_tokens` and
   logical position by `n_pos`. Keep MTP/speculative decode disabled.

## Completed end-to-end path

- `vision-test` now reports bridge/vision, model load, per-chunk prefill, first
  logits, decode, and physical/logical position diagnostics.
- Deterministic chunk metadata and temperature-zero greedy tokens match the
  equivalent Furnace prompt on the reproducible llama.cpp image fixture.
- OpenAI structured content accepts text plus exactly one `data:` JPEG/PNG
  image. Remote URLs and additional images are rejected; the HTTP body and
  decoded image have independent finite limits.
- The isolated MI50 server smoke test returned HTTP 200 for the reproducible
  image fixture without starting the managed Furnace service.

## Rollout status

The image path is available when a Qwen server is explicitly started with
`--mmproj` and `--mtmd-bridge`. The managed Furnace service remains stopped
for development; production rollout, multi-image input, and remote image
fetching are separate decisions.

## Next performance and thermal work

End-to-end GPU-vision validation exposed a practical deployment constraint on
the MI50: large images expand to roughly 2,000 prompt rows and make vision
encoding plus multimodal prefill dominate request latency and thermals. A
temporary 100 W hardware cap allowed several representative images to finish
below a 90 C junction guard, but the largest screenshot still tripped that
guard. This is not a correctness failure and must not be hidden by changing
the model prompt or reducing output quality.

Before changing model behavior, add an explicit per-request timing breakdown
(implemented in bridge ABI v2 and the warm server request log):

1. Image decode/preprocess and projector encoding time.
2. ReInstinct multimodal prefill time and physical/logical row counts.
3. Decode time, generated-token count, and time to first generated token.
4. Peak junction, memory temperature, and package power for the request.

Items 1-3 are implemented. Item 4 remains deliberately external to the server:
pair each stress request with the guarded ROCm telemetry sampler so a profiler
cannot silently change fan or power policy and a wedged request can still be
terminated independently.

The first engine optimization is validated: projected image embeddings use
the same batched transformer prefill strategy as text, extended with explicit
per-row four-plane M-RoPE positions. This removes one full model invocation,
GPU synchronization, and logits download per image row. On the matched 943-row
MI50 fixture it cut LLM prefill from roughly 13.3 s to 1.86 s and total request
latency from roughly 18.5 s to 7.0 s. The projector is now the dominant TTFT
stage at 3.74 s. Further gains should target projector execution or an explicit
image-token budget; the latter is a quality tradeoff and must remain observable
and benchmarked against the uncapped path.

Use that data to choose the first optimization. Likely candidates are
adaptive image resolution/token budgeting and projector-path optimization;
both must preserve the existing single-image OpenAI API contract and be
compared at matched image inputs and output caps. Keep power-limit tuning as
host configuration, outside inference code, and retain a thermal guard for
stress runs.

## Non-goals for v1

Multiple images, video, audio, remote image fetches, non-causal projectors,
image prefix caching, and shared HIP buffers remain deferred.
