/*
 * shaders.metal — Metal compute shaders for 4-bit quantized MoE inference
 *
 * Core operations:
 *   1. dequant_matvec_4bit: 4-bit affine dequantized matrix-vector multiply
 *   2. swiglu_fused: SwiGLU activation (silu(gate) * up)
 *   3. weighted_sum: combine expert outputs with routing weights
 *   4. rms_norm: RMS normalization
 *
 * Quantization format (MLX affine 4-bit, group_size=64):
 *   - Weights stored as uint32, each holding 8 x 4-bit values
 *   - Per-group scale and bias in bfloat16
 *   - Dequantized value = uint4_val * scale + bias
 *   - Groups of 64 elements share one (scale, bias) pair
 *
 * Matrix layout for expert projections:
 *   gate_proj/up_proj: [1024, 512] uint32 = [1024, 4096] logical (out=1024, in=4096)
 *   down_proj: [4096, 128] uint32 = [4096, 1024] logical (out=4096, in=1024)
 *
 *   Scales/biases: [out_dim, in_dim/group_size]
 *   gate/up scales: [1024, 64]   (4096/64 = 64 groups)
 *   down scales:    [4096, 16]   (1024/64 = 16 groups)
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// BFloat16 helpers
// ============================================================================

// Convert bfloat16 (stored as uint16) to float32
inline float bf16_to_f32(uint16_t bf16) {
    // bfloat16 is the upper 16 bits of float32
    return as_type<float>(uint(bf16) << 16);
}

// Convert float32 to bfloat16 (stored as uint16)
inline uint16_t f32_to_bf16(float f) {
    return uint16_t(as_type<uint>(f) >> 16);
}


// ============================================================================
// Kernel 1: 4-bit dequantized matrix-vector multiply
// ============================================================================
//
// Computes: out[row] = dot(dequant(W[row, :]), x[:])
//
// For a single expert, single output row:
//   W_packed[row, :] has in_dim/8 uint32 values
//   scales[row, :] has in_dim/64 bf16 values
//   biases[row, :] has in_dim/64 bf16 values
//   x[:] has in_dim float values
//
// Each thread processes one output row.
// Within a row, we iterate over groups of 64 input elements.
// Each group = 8 uint32 values (8 * 8 = 64 4-bit values) + 1 scale + 1 bias.
//
// For gate/up_proj: out_dim=1024, in_dim=4096
//   W_packed: [1024, 512], scales: [1024, 64], biases: [1024, 64]
//
// For down_proj: out_dim=4096, in_dim=1024
//   W_packed: [4096, 128], scales: [4096, 16], biases: [4096, 16]

kernel void dequant_matvec_4bit(
    device const uint32_t* W_packed   [[buffer(0)]],  // [out_dim, in_dim/8]
    device const uint16_t* scales     [[buffer(1)]],  // [out_dim, num_groups] bf16
    device const uint16_t* biases     [[buffer(2)]],  // [out_dim, num_groups] bf16
    device const float*    x          [[buffer(3)]],  // [in_dim]
    device float*          out        [[buffer(4)]],  // [out_dim]
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;

    uint num_groups = in_dim / group_size;  // groups per row
    uint packed_per_group = group_size / 8; // uint32 values per group (64/8 = 8)
    uint packed_cols = in_dim / 8;          // total uint32 cols per row

    float acc = 0.0f;

    // Pointer to this row's packed weights
    device const uint32_t* w_row = W_packed + tid * packed_cols;
    device const uint16_t* s_row = scales + tid * num_groups;
    device const uint16_t* b_row = biases + tid * num_groups;

    for (uint g = 0; g < num_groups; g++) {
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint base_packed = g * packed_per_group;
        uint base_x = g * group_size;

        // Each uint32 holds 8 x 4-bit values (LSB first)
        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t packed = w_row[base_packed + p];
            uint x_base = base_x + p * 8;

            // Unroll 8 nibbles from the uint32
            // Nibble i = bits [4*i .. 4*i+3]
            for (uint n = 0; n < 8; n++) {
                uint nibble = (packed >> (n * 4)) & 0xF;
                float w_val = float(nibble) * scale + bias;
                acc += w_val * x[x_base + n];
            }
        }
    }

    out[tid] = acc;
}


// ============================================================================
// Kernel 1b: 4-bit dequant matvec — SIMD-optimized with threadgroup reduction
// ============================================================================
//
// Each threadgroup handles one output row.
// Threads within the group split the input dimension, then reduce.
// This gives much better utilization for large in_dim (4096).

kernel void dequant_matvec_4bit_fast(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= out_dim) return;

    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;

    device const uint32_t* w_row = W_packed + tgid * packed_cols;
    device const uint16_t* s_row = scales + tgid * num_groups;
    device const uint16_t* b_row = biases + tgid * num_groups;

    // Each thread handles a subset of groups
    float acc = 0.0f;
    for (uint g = lid; g < num_groups; g += tg_size) {
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint base_packed = g * packed_per_group;
        uint base_x = g * group_size;

        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t packed = w_row[base_packed + p];
            uint x_base = base_x + p * 8;

            // Manual unroll: 8 nibbles
            acc += (float((packed >>  0) & 0xF) * scale + bias) * x[x_base + 0];
            acc += (float((packed >>  4) & 0xF) * scale + bias) * x[x_base + 1];
            acc += (float((packed >>  8) & 0xF) * scale + bias) * x[x_base + 2];
            acc += (float((packed >> 12) & 0xF) * scale + bias) * x[x_base + 3];
            acc += (float((packed >> 16) & 0xF) * scale + bias) * x[x_base + 4];
            acc += (float((packed >> 20) & 0xF) * scale + bias) * x[x_base + 5];
            acc += (float((packed >> 24) & 0xF) * scale + bias) * x[x_base + 6];
            acc += (float((packed >> 28) & 0xF) * scale + bias) * x[x_base + 7];
        }
    }

    // SIMD reduction within the threadgroup
    // Use simd_sum for warp-level reduction, then threadgroup shared memory
    threadgroup float shared[32];  // max 32 simd groups per threadgroup
    float simd_val = simd_sum(acc);

    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;
    uint num_simd_groups = (tg_size + 31) / 32;

    if (simd_lane == 0) {
        shared[simd_group] = simd_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First simd group reduces the partial sums
    if (simd_group == 0 && simd_lane < num_simd_groups) {
        float val = shared[simd_lane];
        val = simd_sum(val);
        if (simd_lane == 0) {
            out[tgid] = val;
        }
    }
}


// ============================================================================
// Kernel 2: SwiGLU activation
// ============================================================================
//
// out[i] = silu(gate[i]) * up[i]
// where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))

kernel void swiglu_fused(
    device const float* gate [[buffer(0)]],  // [dim]
    device const float* up   [[buffer(1)]],  // [dim]
    device float*       out  [[buffer(2)]],  // [dim]
    constant uint&      dim  [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float g = gate[tid];
    float silu_g = g / (1.0f + exp(-g));  // silu = x * sigmoid(x)
    out[tid] = silu_g * up[tid];
}


// ============================================================================
// Kernel 3: Weighted sum of expert outputs
// ============================================================================
//
// out[d] = sum_k( weights[k] * expert_outs[k * dim + d] )
// Used to combine top-K expert outputs with routing weights.

kernel void weighted_sum(
    device const float* expert_outs [[buffer(0)]],  // [K, dim]
    device const float* weights     [[buffer(1)]],  // [K]
    device float*       out         [[buffer(2)]],  // [dim]
    constant uint&      K           [[buffer(3)]],
    constant uint&      dim         [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float acc = 0.0f;
    for (uint k = 0; k < K; k++) {
        acc += weights[k] * expert_outs[k * dim + tid];
    }
    out[tid] = acc;
}


// ============================================================================
// Kernel 4: RMS Normalization
// ============================================================================
//
// out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]
// Two-pass: first compute sum of squares, then normalize.

// Pass 1: compute sum of x^2 (partial sums per thread)
kernel void rms_norm_sum_sq(
    device const float* x       [[buffer(0)]],  // [dim]
    device float*       sum_sq  [[buffer(1)]],  // [1] output: sum of squares
    constant uint&      dim     [[buffer(2)]],
    uint tid  [[thread_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shared[32];

    float acc = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float val = x[i];
        acc += val * val;
    }

    float simd_val = simd_sum(acc);
    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;

    if (simd_lane == 0) {
        shared[simd_group] = simd_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float val = (simd_lane < (tg_size + 31) / 32) ? shared[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            sum_sq[0] = val;
        }
    }
}

// Pass 2: normalize
kernel void rms_norm_apply(
    device const float* x       [[buffer(0)]],
    device const float* weight  [[buffer(1)]],  // [dim] gamma
    device const float* sum_sq  [[buffer(2)]],  // [1]
    device float*       out     [[buffer(3)]],
    constant uint&      dim     [[buffer(4)]],
    constant float&     eps     [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float rms = rsqrt(sum_sq[0] / float(dim) + eps);
    out[tid] = x[tid] * rms * weight[tid];
}
