// Four-plane Qwen M-RoPE for one input row.
//
// Frequencies remain indexed globally across the rotary dimension. The
// section table only chooses which logical position plane supplies theta.
// Cos/sin rows are shared with the scalar text fast path.

#include <hip/hip_runtime.h>

extern "C" __global__
void rope_apply_mrope_f32(float*       __restrict__ x,
                          const float* __restrict__ cos,
                          const float* __restrict__ sin,
                          unsigned int head_dim,
                          unsigned int rotary_dim,
                          unsigned int n_heads,
                          const unsigned int* __restrict__ positions,
                          const unsigned int* __restrict__ sections)
{
    const unsigned int half = rotary_dim >> 1;
    const unsigned int h = blockIdx.y;
    if (h >= n_heads) return;
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half) return;

    unsigned int plane = 0;
    unsigned int boundary = sections[0];
    while (i >= boundary && plane < 3) {
        ++plane;
        boundary += sections[plane];
    }
    const unsigned int pos = positions[plane];
    const float* cr = cos + (size_t)pos * rotary_dim;
    const float* sr = sin + (size_t)pos * rotary_dim;
    float* head = x + (size_t)h * head_dim;
    const float a = head[i];
    const float b = head[i + half];
    head[i] = a * cr[i] - b * sr[i];
    head[i + half] = b * cr[i + half] + a * sr[i + half];
}
