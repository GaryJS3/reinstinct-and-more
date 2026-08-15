//! Rotary Position Embedding cache (partial, half-split rotation).
//!
//! Qwen 3.5 0.8B uses `rope.dimension_count = 64` of `head_dim = 256` —
//! only the first 64 elements of each Q/K head are rotated, the remaining
//! 192 pass through unchanged. The rotation is the standard llama/HF
//! "half-split" form (`rotate_half` flips the two halves with sign change).
//!
//! M-RoPE (mrope_section [11, 11, 10]) is the multimodal extension for
//! vision tokens. For pure-text tokens all three position axes share the
//! same position id, so the M-RoPE freq table collapses to standard RoPE.
//! The scalar cache remains the text fast path. [`apply_mrope`] is the
//! four-plane reference used to validate multimodal GPU kernels.

/// Precomputed cos/sin tables for positions 0..max_seq_len.
///
/// Layout: `cos[pos * rotary_dim + i]` for `i in 0..rotary_dim`. The table
/// duplicates each frequency value into both halves (`cos[i] == cos[i + half]`)
/// to keep the rotation kernel branch-free.
pub struct RopeCache {
    pub rotary_dim: usize,
    pub max_seq_len: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl RopeCache {
    pub fn new(rotary_dim: usize, max_seq_len: usize, freq_base: f32) -> Self {
        Self::new_proportional(rotary_dim, rotary_dim / 2, max_seq_len, freq_base)
    }

    /// HF-style "proportional" RoPE: kernel pairs `(i, i + rotary_dim/2)`
    /// for every i in `[0, rotary_dim/2)`, BUT inv_freq is non-zero only
    /// for the first `rope_angles` pairs and zero (= cos 1, sin 0,
    /// pass-through) for the rest. This is the convention HF uses for
    /// `rope_type="proportional"` (gemma4 full attention with
    /// `partial_rotary_factor=0.25` → `rope_angles=64` of 256 pairs).
    /// Standard fully-rotated RoPE is `rope_angles == rotary_dim/2`.
    pub fn new_proportional(rotary_dim: usize, rope_angles: usize,
                            max_seq_len: usize, freq_base: f32) -> Self
    {
        assert!(rotary_dim % 2 == 0, "rotary_dim must be even");
        let half = rotary_dim / 2;
        let rope_angles = rope_angles.min(half);
        let mut inv_freq = vec![0.0_f32; half];
        for i in 0..rope_angles {
            inv_freq[i] = freq_base.powf(-2.0 * i as f32 / rotary_dim as f32);
        }
        // i in [rope_angles, half) stay at inv_freq[i]=0 → theta=0 →
        // cos=1, sin=0 → identity (the HF "proportional" tail).
        let mut cos = vec![1.0_f32; max_seq_len * rotary_dim];
        let mut sin = vec![0.0_f32; max_seq_len * rotary_dim];
        for pos in 0..max_seq_len {
            for i in 0..half {
                let theta = pos as f32 * inv_freq[i];
                let c = theta.cos();
                let s = theta.sin();
                cos[pos * rotary_dim + i]        = c;
                cos[pos * rotary_dim + i + half] = c;
                sin[pos * rotary_dim + i]        = s;
                sin[pos * rotary_dim + i + half] = s;
            }
        }
        Self { rotary_dim, max_seq_len, cos, sin }
    }

    /// Slice the (cos, sin) row for a given position.
    pub fn get(&self, position: usize) -> (&[f32], &[f32]) {
        let off = position * self.rotary_dim;
        (
            &self.cos[off..off + self.rotary_dim],
            &self.sin[off..off + self.rotary_dim],
        )
    }
}

/// Apply RoPE in place to a single head.
///
/// `head` has length `head_dim`; the first `rope_cache.rotary_dim` elements
/// are rotated using the half-split convention:
///   half = rotary_dim / 2
///   y[i]        = x[i]        * cos[i]        - x[i + half] * sin[i]
///   y[i + half] = x[i + half] * cos[i + half] + x[i]        * sin[i + half]
/// Elements beyond `rotary_dim` are passed through unchanged.
pub fn apply_rope(head: &mut [f32], rope_cache: &RopeCache, position: usize) {
    let rd = rope_cache.rotary_dim;
    assert!(head.len() >= rd, "head_dim {} < rotary_dim {}", head.len(), rd);
    let half = rd / 2;
    let (cos, sin) = rope_cache.get(position);

    // Snapshot the rotated portion first (rotation reads both halves).
    // Sized dynamically — Gemma 4 rotates up to 512 dims, Qwen 3.5 only 64.
    let buf: Vec<f32> = head[..rd].to_vec();

    for i in 0..half {
        head[i]        = buf[i]        * cos[i]        - buf[i + half] * sin[i];
        head[i + half] = buf[i + half] * cos[i + half] + buf[i]        * sin[i + half];
    }
}

/// Apply Qwen's four-plane, NEOX-ordered M-RoPE to one attention head.
///
/// `sections` counts frequency pairs assigned to the t/x/y/z planes and
/// must cover exactly `rotary_dim / 2` pairs. Frequencies do not restart at
/// section boundaries: pair `i` always uses `base^(-2*i/rotary_dim)`, matching
/// `ggml_mrope_cache_init` with independent-section mode disabled.
pub fn apply_mrope(
    head: &mut [f32],
    rotary_dim: usize,
    freq_base: f32,
    sections: &[u32],
    positions: [u32; 4],
) {
    assert!(rotary_dim % 2 == 0, "rotary_dim must be even");
    assert!(head.len() >= rotary_dim,
        "head_dim {} < rotary_dim {}", head.len(), rotary_dim);
    assert_eq!(sections.len(), 4, "M-RoPE requires four sections");
    let half = rotary_dim / 2;
    let section_total: usize = sections.iter().map(|&v| v as usize).sum();
    assert_eq!(section_total, half,
        "M-RoPE sections must sum to rotary_dim / 2");

    let input = head[..rotary_dim].to_vec();
    let mut boundary = sections[0] as usize;
    let mut plane = 0usize;
    for i in 0..half {
        while i >= boundary && plane < 3 {
            plane += 1;
            boundary += sections[plane] as usize;
        }
        let inv_freq = freq_base.powf(-2.0 * i as f32 / rotary_dim as f32);
        let theta = positions[plane] as f32 * inv_freq;
        let (sin, cos) = theta.sin_cos();
        let a = input[i];
        let b = input[i + half];
        head[i] = a * cos - b * sin;
        head[i + half] = b * cos + a * sin;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    #[test]
    fn position_zero_is_identity() {
        // At position 0, cos = 1 and sin = 0, so rotation is a no-op.
        let cache = RopeCache::new(64, 16, 10000.0);
        let mut head = vec![0.0_f32; 256];
        for i in 0..256 { head[i] = (i as f32) * 0.01; }
        let original = head.clone();
        apply_rope(&mut head, &cache, 0);
        for i in 0..256 {
            assert!(approx_eq(head[i], original[i], 1e-6),
                "i={i} got {} expected {}", head[i], original[i]);
        }
    }

    #[test]
    fn pass_through_dims_unchanged() {
        // For partial RoPE (rotary_dim=64 of head_dim=256), elements >= 64
        // must be byte-equal to input regardless of position.
        let cache = RopeCache::new(64, 16, 10000.0);
        let mut head = vec![0.0_f32; 256];
        for i in 0..256 { head[i] = (i as f32 + 1.0).sqrt(); }
        let original = head.clone();
        apply_rope(&mut head, &cache, 5);
        for i in 64..256 {
            assert_eq!(head[i].to_bits(), original[i].to_bits(), "i={i}");
        }
    }

    #[test]
    fn rotation_at_position_one_matches_handcomputed() {
        // For freq_base=10000, rotary_dim=4:
        //   inv_freq[0] = 10000^0 = 1
        //   inv_freq[1] = 10000^(-2/4) = 1 / sqrt(10000) = 0.01
        // At position 1:
        //   theta_0 = 1 * 1 = 1 → cos=cos(1), sin=sin(1)
        //   theta_1 = 1 * 0.01 = 0.01 → cos≈cos(0.01), sin≈sin(0.01)
        // For input x = [a, b, c, d]:
        //   y[0] = a*cos(1) - c*sin(1)
        //   y[1] = b*cos(0.01) - d*sin(0.01)
        //   y[2] = c*cos(1) + a*sin(1)
        //   y[3] = d*cos(0.01) + b*sin(0.01)
        let cache = RopeCache::new(4, 4, 10000.0);
        let mut head = vec![1.0_f32, 2.0, 3.0, 4.0];
        apply_rope(&mut head, &cache, 1);

        let c0 = (1.0_f32).cos();
        let s0 = (1.0_f32).sin();
        let c1 = (0.01_f32).cos();
        let s1 = (0.01_f32).sin();
        assert!(approx_eq(head[0], 1.0 * c0 - 3.0 * s0, 1e-6));
        assert!(approx_eq(head[1], 2.0 * c1 - 4.0 * s1, 1e-6));
        assert!(approx_eq(head[2], 3.0 * c0 + 1.0 * s0, 1e-6));
        assert!(approx_eq(head[3], 4.0 * c1 + 2.0 * s1, 1e-6));
    }

    #[test]
    fn rotation_preserves_norm() {
        // Rotation is a unitary transformation → ||x||² preserved on the
        // rotated portion. Use head_dim = rotary_dim so we measure the whole vector.
        let cache = RopeCache::new(8, 32, 10000.0);
        let head_init: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let norm_in: f32 = head_init.iter().map(|v| v * v).sum::<f32>().sqrt();
        for pos in 0..32 {
            let mut head = head_init.clone();
            apply_rope(&mut head, &cache, pos);
            let norm_out: f32 = head.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(approx_eq(norm_in, norm_out, 1e-6),
                "pos={pos}: norm changed {} → {}", norm_in, norm_out);
        }
    }

    #[test]
    fn mrope_text_broadcast_matches_scalar_rope() {
        let rotary_dim = 64;
        let freq_base = 1_000_000.0;
        let sections = [11, 11, 10, 0];
        let position = 17;
        let cache = RopeCache::new(rotary_dim, position + 1, freq_base);
        let input: Vec<f32> = (0..128).map(|i| (i as f32 * 0.17).sin()).collect();
        let mut scalar = input.clone();
        let mut multi = input;
        apply_rope(&mut scalar, &cache, position);
        apply_mrope(&mut multi, rotary_dim, freq_base, &sections,
                    [position as u32; 4]);
        for i in 0..multi.len() {
            assert!(approx_eq(multi[i], scalar[i], 1e-6),
                "i={i}: mrope={} scalar={}", multi[i], scalar[i]);
        }
    }

    #[test]
    fn mrope_uses_the_position_selected_by_each_section() {
        let mut head = vec![0.0; 8];
        head[..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        apply_mrope(&mut head, 8, 10_000.0, &[1, 1, 1, 1], [0, 1, 2, 3]);
        assert_eq!(head[0], 1.0); // t=0 is identity for pair 0
        assert_eq!(head[4], 0.0);
        for (i, &position) in [1_u32, 2, 3].iter().enumerate() {
            let pair = i + 1;
            let theta = position as f32 * 10_000.0_f32.powf(-2.0 * pair as f32 / 8.0);
            let (sin, cos) = theta.sin_cos();
            assert!(approx_eq(head[pair], (pair + 1) as f32 * cos, 1e-6));
            assert!(approx_eq(head[pair + 4], (pair + 1) as f32 * sin, 1e-6));
        }
    }
}
