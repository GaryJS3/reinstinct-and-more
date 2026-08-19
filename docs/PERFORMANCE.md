# Multimodal performance

Last updated: 2026-08-18

This is the canonical ledger for ReInstinct image-path speed, bottlenecks, and
thermal constraints. Keep correctness and deployment status in
`MULTIMODAL_PROGRESS.md`; put every comparable performance result here.

## Reference configuration

- Host: `ai@10.0.0.41`
- GPU: AMD Instinct MI50, gfx906, 32 GiB
- Model: Qwen3.6-35B-A3B-UD-Q4_K_XL
- Projector: `mmproj-F32.gguf`, GPU vision, 8 threads
- Prompt: `Describe this image in detail. Answer directly without explaining
  your reasoning.`
- Sampling: temperature 0, 64 completion-token cap
- Current test power cap: 100 W; hardware default: 178 W
- Thermal policy: stop at an internal 88 C junction reading for a requested
  90 C ceiling

Always verify the live power cap before adding a row. Results at different
power limits, image bytes, prompts, output caps, or vision devices are not
directly comparable.

## Current headline result

The 943-row beach fixture is the primary matched regression workload.

| Metric | Scalar image prefill | Batched image prefill | Change |
|---|---:|---:|---:|
| Physical rows | 943 | 943 | same work |
| Logical positions | 63 | 63 | same positions |
| Image decode | 14.7 ms | 15.3 ms | noise |
| Media tokenization | 39.8 ms | 44.8 ms | noise |
| Vision/projector | 3.863 s | 3.743 s | 1.03x |
| mtmd total | 3.923 s | 3.809 s | 1.03x |
| ReInstinct LLM prefill | ~13.295 s | ~1.855 s | **7.17x** |
| TTFT | 17.223 s | 5.670 s | **3.04x** |
| Decode, 64 tokens | 1.279 s | 1.329 s | 0.96x |
| Full request | ~18.50 s | 7.013 s | **2.64x** |
| End-to-end completion rate | ~3.46 tok/s | 9.13 tok/s | **2.64x** |
| Peak junction | 88 C cutoff after request | 59 C | safer/shorter load |

The batched result is commit `7d2942b`; profiling instrumentation is
`4830c81`. The result remained coherent and described the same woman swinging
over turquoise water. Exact greedy wording changed because the optimized
prefill uses the existing fp16 batched math instead of the scalar fp32 decode
path.

## Current request breakdown

For the optimized 7.013-second request:

| Stage | Time | Share |
|---|---:|---:|
| Vision/projector | 3.743 s | 53% of wall, 66% of TTFT |
| ReInstinct batched LLM prefill | ~1.855 s | 26% of wall, 33% of TTFT |
| Decode | 1.329 s | 19% of wall |
| Decode/tokenize/copy and server overhead | ~86 ms | ~1% of wall |

The projector is now the dominant bottleneck. The engine-side image prefill is
no longer the first optimization target.

## Historical image results

These rows are useful trend data but are not all matched comparisons.

| Workload | Vision | Rows | Output | Total | Rate | Thermal/power notes |
|---|---|---:|---:|---:|---:|---|
| llama.cpp `test-1.jpeg` | GPU | 316 | not recorded | prefill 3.891 s | decode 75.4 tok/s | early scalar correctness run |
| beach-pic-2 | GPU | 943 | 64 | 16.14 s | 4.0 tok/s | older uninstrumented run |
| beach-pic-2 | CPU | 943 | 64 | 37.51 s | 1.7 tok/s | same image; GPU vision was 2.3x faster |
| beach-pic-1 | GPU | 2,063 | 64 | 42.80 s | 1.50 tok/s | 100 W, 84 C peak |
| surveillance | GPU | 2,063 | 64 | 41.88 s | 1.53 tok/s | 100 W, 87 C peak |
| cookie pizza | GPU | 2,151 | 64 | 44.31 s | 1.44 tok/s | 100 W, 89 C peak |
| property agreement | GPU | 823 | 64 | 16.23 s | 3.94 tok/s | 100 W |
| screenshot | GPU | not completed | 64 cap | stopped at 45.94 s | n/a | 100 W, 90 C guard |

The large-image rows predate batched image prefill and should be rerun before
using them to predict current latency.

## Bottleneck queue

1. **Vision/projector execution — 3.743 s.** Investigate libmtmd/ggml HIP
   profiling and image-token limits. An image-token cap can reduce both
   projector and LLM work, but it is an observable quality tradeoff and must
   be compared against identical uncapped images.
2. **Batched LLM prefill — 1.855 s for 943 rows.** Already improved 7.17x.
   Track milliseconds per physical row (~1.97 ms/row) across image sizes to
   identify nonlinear attention or allocation costs.
3. **Decode — 1.329 s for 64 tokens (~48.2 decode tok/s).** This is no longer
   on the TTFT critical path; optimize only after projector work unless the
   workload shifts toward long outputs.
4. **Cold model startup.** Multi-minute GGUF/NFS load time is excluded from
   request measurements. Keep the production model resident; report startup
   separately when deployment behavior is under test.

## Measurement protocol

For every new comparable result:

1. Record commit, exact image filename/hash, model, projector, prompt, output
   cap, GPU/CPU vision, vision threads, power cap, and thermal cutoff.
2. Use an isolated port and leave the managed Furnace service unchanged.
3. Start thermal sampling before the request. Record peak junction, memory
   temperature, and package power; stop immediately at the cutoff.
4. Discard model startup from request latency. Label first-request versus warm
   results, and use at least two warm repetitions when thermally safe.
5. Capture `decode_image_ms`, `tokenize_ms`, `projector_ms`, `copy_ms`,
   `mtmd_ms`, `prefill_ms`, `ttft_ms`, `decode_ms`, rows, logical positions,
   and generated tokens from the `vision profile` log line.
6. Report model output or a correctness comparison. Speed without coherent
   image understanding is not a valid win.
7. Stop the isolated server and confirm VRAM is released.

Derived fields:

- ReInstinct LLM prefill = `prefill_ms - mtmd_ms`
- Decode tok/s = `generated * 1000 / decode_ms`
- End-to-end completion rate = `generated / request_wall_seconds`
- Projector share of TTFT = `projector_ms / ttft_ms`
