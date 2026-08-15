# Multimodal port: Furnace/libmtmd to ReInstinct

## Frozen baseline

- ReInstinct: `5d79eca88ede998a1712a01ed7c73ca2a687564b`.
- Furnace/libmtmd: `5013b9f91d3cbe627a758336802b1a2223427b49` on branch `gfx906-perf`.
- Target model: `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`.
- Projector: `mmproj-F32.gguf`.
- Reference Furnace server: `llama-server -ngl 999 --ctx-size 32000 --batch-size 4096 --ubatch-size 2048 -fa on --reasoning off --mmproj mmproj-F32.gguf`.

The bridge loads the text GGUF with `vocab_only=true`, then owns the mtmd
context and image preprocessing/vision/projector work. It must never create a
llama context or call `llama_decode`; ReInstinct evaluates all language-model
layers and generation.

## Position audit

| Existing value | Meaning today | Required multimodal meaning |
| --- | --- | --- |
| `GpuKvCache.len` / batched `base_pos` | Physical KV cache index | Advances by every text token and image embedding. |
| `Qwen35GpuState.pos` | Decode position and diagnostic counter | Must be split into physical KV position and logical RoPE position. |
| `GpuQwen35.d_pos` | Device value shared by RoPE, KV write, and decode attention | Must become separate device values before M-RoPE is enabled. |
| `mtmd_input_chunk_get_n_pos` | Logical mtmd continuation count | Advances logical Qwen position after an image. |

For text `n_tokens == n_pos`. An M-RoPE image can have many physical
embeddings but fewer logical positions. The pinned Furnace helper advances its
`n_past` by `mtmd_input_chunk_get_n_pos`, while its K/V batch has one row per
embedding. This distinction is a correctness gate, not an optimization.

## Current milestones

1. The pinned submodule and `mtmd-bridge` expose stable copied text/image
   chunks. `reinstinct-engine mtmd-test` validates the image frontend without
   loading ReInstinct transformer weights.
2. `GpuQwen35::forward_embeddings_mrope` provides the sequential,
   position-correct injection point for external FP32 rows. Physical KV and
   logical RoPE state advance independently, and ordered bridge chunks are
   accepted by the runtime.
3. Separate KV/RoPE state and the four-plane Qwen M-RoPE kernel pass CPU/GPU
   parity on gfx906. `vision-test` also matches Furnace's first eight greedy
   tokens on a real image at temperature zero. HTTP image requests remain
   disabled pending structured-content parsing and finite-body tests.

## Validation commands

```bash
scripts/build-mtmd-bridge.sh
reinstinct-engine mtmd-test \
  --model /mnt/ai-models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --mmproj /mnt/ai-models/unsloth/Qwen3.6-35B-A3B-GGUF/mmproj-F32.gguf \
  --image test.jpg --prompt 'Describe this image.' \
  --bridge build/mtmd-bridge/libreinstinct_mtmd.so
```

The command reports deterministic xxh3 hashes for text tokens, FP32 image
embeddings, and M-RoPE position data. Compare those fields with an equivalent
Furnace diagnostic before changing transformer kernels.
