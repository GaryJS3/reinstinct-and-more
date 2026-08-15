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
- Ordered text/image/text chunk prefill is exposed through one OpenAI
  structured-content `image_url` part when the vision options are enabled.
- `vision-test` runs the bridge and full ReInstinct transformer, prints
  per-stage timing, top first logits, physical/logical state, and greedy text.

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

## Commits

`a8cb383`, `2aad59a`, `ecb5cdd`, `4f02753`, `3137ee8`, `ffe6e07`, and
`0017cd0`, `ebe64d7`, `35e7326`, and `d9c86a8` on
`feature/multimodal-mtmd`.
