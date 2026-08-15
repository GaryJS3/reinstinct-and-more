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

## Important discovery

On the pinned Furnace revision, `llama_model_n_embd_inp()` and
`llama_model_n_embd()` both return zero for a `vocab_only` model. The bridge
therefore receives the embedding width parsed by ReInstinct from the same GGUF
metadata. This preserves the no-second-LLM requirement.

## Current blocker

The bridge boundary is correct, but ReInstinct still assumes a single scalar
position for KV placement and RoPE. Do not enable end-to-end image generation
until physical KV positions and four-plane M-RoPE are separated and tested.

## Commits

`a8cb383`, `2aad59a`, `ecb5cdd`, `4f02753`, `3137ee8`, `ffe6e07`, and
`0017cd0` on `feature/multimodal-mtmd`.
