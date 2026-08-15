# Multimodal progress

Last updated: 2026-08-15

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
- Ordered text/image/text chunk prefill is implemented but is not exposed by
  the server.

## Important discovery

On the pinned Furnace revision, `llama_model_n_embd_inp()` and
`llama_model_n_embd()` both return zero for a `vocab_only` model. The bridge
therefore receives the embedding width parsed by ReInstinct from the same GGUF
metadata. This preserves the no-second-LLM requirement.

## Current blocker

The position split and M-RoPE kernel are implemented and pass synthetic and
text-broadcast parity tests. The next correctness gate is running the captured
752-row image fixture through the full Qwen3.6 model and comparing first
logits and greedy tokens with Furnace. Do not enable HTTP image requests until
that end-to-end comparison passes.

## Latest validation

- Remote `cargo check`: passed.
- Remote MI50 library suite: 184 passed, 1 ignored.
- Dedicated gfx906 M-RoPE CPU/GPU parity test: passed.
- Qwen3.5 4B token vs scalar-external vs broadcast-M-RoPE logits: passed.

## Commits

`a8cb383`, `2aad59a`, `ecb5cdd`, `4f02753`, `3137ee8`, `ffe6e07`, and
`0017cd0` on `feature/multimodal-mtmd`.
