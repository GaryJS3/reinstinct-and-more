# Multimodal performance

Last updated: 2026-08-19

This is the canonical ledger for ReInstinct image-path speed, bottlenecks, and
thermal constraints. Keep correctness and deployment status in
`MULTIMODAL_PROGRESS.md`; put every comparable performance result here.

API compatibility is the immediate project priority. The engine optimization
queue below remains valid, but resumes after the contract suite and isolated
deployment gates in `API_COMPATIBILITY_PLAN.md` unless compatibility testing
finds a performance blocker.

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

### BF16 projector result

The matching `mmproj-Qwen3.6-35B-A3B-BF16.gguf` preserves the 920 image
embeddings / 943 total prompt rows and 40 image positions. At 100 W:

| Metric | F32 projector | BF16 projector | Change |
|---|---:|---:|---:|
| Projector | 3.778 s | 1.611 s | **2.35x** |
| mtmd total | 3.846 s | 1.714 s | **2.24x** |
| ReInstinct LLM prefill | 1.867 s | 1.834 s | same work |
| TTFT | 5.719 s | 3.561 s | **1.61x** |
| Decode, 64 tokens | 1.315 s | 1.403 s | load dependent |
| Full request | 7.048 s | 4.977 s | **1.42x** |
| End-to-end completion rate | 9.1 tok/s | 12.9 tok/s | **1.42x** |

The BF16 result remained coherent, describing the same woman on a rope swing
over turquoise water. Its embedding hash differs from F32, as expected for a
precision change; its position hash is identical. Peak junction was 63 C.

### Dynamic image-token budget

Bridge ABI v3 exposes libmtmd's dynamic-resolution limits through
`--image-min-tokens` / `--image-max-tokens` on diagnostics and
`--vision-min-tokens` / `--vision-max-tokens` on the server:

| Maximum | Image embeddings | Logical positions | Decode/tokenize/encode | Change |
|---:|---:|---:|---:|---:|
| model default | 920 | 40 | 3.874 s | baseline |
| 768 | 720 | 36 | 2.997 s | 1.29x |
| 512 | 480 | 30 | 1.981 s | 1.96x |
| 384 | 364 | 26 | 1.497 s | 2.59x |

Scaling is close to linear. Caps reduce both projector execution and following
LLM prefill, but remain an explicit quality/latency tradeoff.

### HIP profile

An F32 `rocprof --stats` capture attributed 89.0% of GPU kernel time to 112
F32 GEMMs and 8.9% to 27 vision FlashAttention kernels. Every other kernel
family was below 1%. With BF16, GEMM fell from 3.309 s to 0.941 s of captured
kernel time while FlashAttention stayed near 0.330 s. Projector precision and
GEMM throughput—not copies, graph setup, or preprocessing—are the primary target.

### Matched Furnace comparison

Furnace revision `5013b9f` and ReInstinct used the same MI50, 100 W cap,
Qwen3.6 35B model, image bytes, prompt, temperature zero, 64-token output cap,
GPU projector, and eight vision threads. Furnace ran as an isolated one-slot
server with FlashAttention enabled; its managed service remained inactive.
Furnace counts four additional prompt bookkeeping tokens, while both engines
receive the same number of image embeddings.

| Projector / image cap | Furnace prompt | Furnace total | ReInstinct TTFT | ReInstinct total | Faster total |
|---|---:|---:|---:|---:|---:|
| F32 / default | 5.223 s | 6.413 s | 5.719 s | 7.048 s | Furnace 9.0% |
| BF16 / default | 2.965 s | 4.142 s | 3.561 s | 4.977 s | Furnace 16.8% |
| BF16 / 768 | 2.368 s | 3.588 s | 2.748 s | 4.170 s | Furnace 14.0% |
| BF16 / 512 | 1.717 s | 2.951 s | 1.860 s | 3.146 s | Furnace 6.2% |
| BF16 / 384 | 1.397 s | 2.586 s | 1.508 s | 2.792 s | Furnace 7.4% |

The BF16 projector itself is effectively tied at the default resolution:
Furnace reports 1.566 s and ReInstinct 1.611 s. Furnace wins in the subsequent
multimodal LLM prefill and usually in decode (about 54.4 versus 45.6 tok/s in
the default BF16 repetition). ReInstinct closes most of the gap at 512--384
image tokens but is not yet faster end to end for this image workload.

Selected matched F32 large-image results show the gap grows again with rows:

| Workload | Image rows | Furnace total | ReInstinct total | Faster total | Furnace peak junction |
|---|---:|---:|---:|---:|---:|
| beach-pic-1 | 2,040 | 13.339 s | 14.869 s | Furnace 10.3% | 63 C |
| driveway surveillance | 4,080 | 28.475 s | 34.026 s | Furnace 16.3% | 71 C |
| screenshot | 2,714 | 17.982 s | 20.609 s | Furnace 12.7% | 73 C |

Rows in this table exclude each engine's surrounding text tokens. Requests
were separated by cooldown and all descriptions remained coherent.

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
| beach-pic-1, batched | GPU | 2,063 | 64 | 14.87 s | 4.3 tok/s | 100 W, new fan |
| driveway surveillance, batched | GPU | 4,103 | 64 | 34.03 s | 1.9 tok/s | 100 W, new fan |
| cookie pizza, batched | GPU | 2,151 | 64 | 15.78 s | 4.1 tok/s | 100 W, new fan |
| property agreement, batched | GPU | 823 | 64 | 6.15 s | 10.4 tok/s | 100 W, new fan |
| screenshot, batched | GPU | 2,737 | 64 | 20.61 s | 3.1 tok/s | 100 W, new fan |

The new fan allowed the guarded suite to complete once, but a telemetry-verified
repeat reached the 88 C internal cutoff (76 C memory) at 100 W. Do not run
sustained back-to-back large requests or raise the power cap. Cool down and
retain the per-request guard even with the additional airflow.

## Bottleneck queue

1. **Batched LLM prefill — 1.834 s after BF16 projection for 920 image rows.**
   Furnace spends roughly 1.399 s after its effectively tied projector. Profile
   ReInstinct's attention, MoE, and chunk/allocation behavior against Furnace;
   this is now the primary cross-engine gap.
2. **Decode — 1.403 s for 64 tokens (~45.6 tok/s) in the matched BF16 run.**
   Furnace reached 54.4 tok/s. Recheck warm repetitions and graph state, then
   compare against the established text-only decode benchmark.
3. **Vision/projector execution — 1.611 s with BF16.** Make the matching BF16
   projector the preferred artifact after broader correctness coverage.
   Furnace's 1.566 s is effectively tied; FlashAttention is the remaining
   shared non-GEMM projector target rather than a ReInstinct-specific gap.
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
