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

## After correctness

- Add `vision-test` with image/preprocess, projector, H2D, prefill, and decode
  timing breakdowns.
- Compare deterministic chunk hashes, positions, first logits, and greedy
  tokens with Furnace at temperature zero.
- Add OpenAI structured text plus one `data:` JPEG/PNG image after CLI output
  matches Furnace. Reject remote URLs and preserve finite body limits.

## Non-goals for v1

Multiple images, video, audio, remote image fetches, non-causal projectors,
image prefix caching, and shared HIP buffers remain deferred.
