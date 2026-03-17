/*
 * main.m — Pure C/Metal 4-bit dequantized MoE expert computation engine
 *
 * Phase 1: Standalone benchmark of 4-bit dequant matvec via Metal compute shaders.
 * Loads expert weights from packed binary files, runs the Metal shader, verifies output.
 *
 * This is the foundation for a full llama.cpp-style inference engine for
 * Qwen3.5-397B-A17B running on Apple Silicon with SSD-streamed expert weights.
 *
 * Build: make
 * Run:   ./metal_infer [--layer N] [--expert E] [--benchmark]
 *
 * What it does:
 *   1. Creates Metal device, command queue, loads compute shaders
 *   2. Opens packed expert file for the specified layer
 *   3. pread()s one expert's weights (7.08 MB) into Metal shared buffers
 *   4. Runs the full MoE expert forward pass:
 *      - gate_proj matvec (4096 -> 1024)
 *      - up_proj matvec (4096 -> 1024)
 *      - SwiGLU activation
 *      - down_proj matvec (1024 -> 4096)
 *   5. Reports timing and throughput
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>
#include <getopt.h>

// ============================================================================
// Constants matching the Qwen3.5-397B packed expert layout
// ============================================================================

#define HIDDEN_DIM       4096
#define INTERMEDIATE_DIM 1024
#define GROUP_SIZE       64
#define BITS             4
#define NUM_EXPERTS      512
#define NUM_LAYERS       60

// Expert component sizes (from layout.json)
#define GATE_W_OFFSET    0
#define GATE_W_SIZE      2097152   // [1024, 512] uint32
#define GATE_S_OFFSET    2097152
#define GATE_S_SIZE      131072    // [1024, 64] uint16 (bf16)
#define GATE_B_OFFSET    2228224
#define GATE_B_SIZE      131072

#define UP_W_OFFSET      2359296
#define UP_W_SIZE        2097152
#define UP_S_OFFSET      4456448
#define UP_S_SIZE        131072
#define UP_B_OFFSET      4587520
#define UP_B_SIZE        131072

#define DOWN_W_OFFSET    4718592
#define DOWN_W_SIZE      2097152   // [4096, 128] uint32
#define DOWN_S_OFFSET    6815744
#define DOWN_S_SIZE      131072    // [4096, 16] uint16 (bf16)
#define DOWN_B_OFFSET    6946816
#define DOWN_B_SIZE      131072

#define EXPERT_SIZE      7077888   // Total bytes per expert

// Default model path
#define MODEL_PATH "/Users/danielwoods/.cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit/snapshots/39159bd8aa74f5c8446d2b2dc584f62bb51cb0d3"

// ============================================================================
// Timing helper
// ============================================================================

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// ============================================================================
// bf16 <-> f32 conversion (CPU side, for reference/verification)
// ============================================================================

static float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// ============================================================================
// CPU reference: 4-bit dequant matvec (for verification)
// ============================================================================

static void cpu_dequant_matvec_4bit(
    const uint32_t *W_packed,   // [out_dim, in_dim/8]
    const uint16_t *scales,     // [out_dim, num_groups]
    const uint16_t *biases,     // [out_dim, num_groups]
    const float *x,             // [in_dim]
    float *out,                 // [out_dim]
    uint32_t out_dim,
    uint32_t in_dim,
    uint32_t group_size
) {
    uint32_t num_groups = in_dim / group_size;
    uint32_t packed_per_group = group_size / 8;
    uint32_t packed_cols = in_dim / 8;

    for (uint32_t row = 0; row < out_dim; row++) {
        float acc = 0.0f;
        const uint32_t *w_row = W_packed + row * packed_cols;
        const uint16_t *s_row = scales + row * num_groups;
        const uint16_t *b_row = biases + row * num_groups;

        for (uint32_t g = 0; g < num_groups; g++) {
            float scale = bf16_to_f32(s_row[g]);
            float bias  = bf16_to_f32(b_row[g]);

            uint32_t base_packed = g * packed_per_group;
            uint32_t base_x = g * group_size;

            for (uint32_t p = 0; p < packed_per_group; p++) {
                uint32_t packed = w_row[base_packed + p];
                uint32_t x_base = base_x + p * 8;

                for (uint32_t n = 0; n < 8; n++) {
                    uint32_t nibble = (packed >> (n * 4)) & 0xF;
                    float w_val = (float)nibble * scale + bias;
                    acc += w_val * x[x_base + n];
                }
            }
        }
        out[row] = acc;
    }
}

// ============================================================================
// CPU reference: SwiGLU
// ============================================================================

static void cpu_swiglu(const float *gate, const float *up, float *out, uint32_t dim) {
    for (uint32_t i = 0; i < dim; i++) {
        float g = gate[i];
        float silu_g = g / (1.0f + expf(-g));
        out[i] = silu_g * up[i];
    }
}

// ============================================================================
// Metal setup and shader management
// ============================================================================

typedef struct {
    id<MTLDevice>              device;
    id<MTLCommandQueue>        queue;
    id<MTLLibrary>             library;
    id<MTLComputePipelineState> matvec_naive;
    id<MTLComputePipelineState> matvec_fast;
    id<MTLComputePipelineState> swiglu;
    id<MTLComputePipelineState> weighted_sum;
    id<MTLComputePipelineState> rms_norm_sum;
    id<MTLComputePipelineState> rms_norm_apply;
} MetalContext;

static MetalContext *metal_init(void) {
    MetalContext *ctx = calloc(1, sizeof(MetalContext));
    if (!ctx) { fprintf(stderr, "ERROR: alloc MetalContext\n"); return NULL; }

    // Get default Metal device
    ctx->device = MTLCreateSystemDefaultDevice();
    if (!ctx->device) {
        fprintf(stderr, "ERROR: No Metal device found\n");
        free(ctx);
        return NULL;
    }
    printf("[metal] Device: %s\n", [[ctx->device name] UTF8String]);
    printf("[metal] Unified memory: %s\n", [ctx->device hasUnifiedMemory] ? "YES" : "NO");
    printf("[metal] Max buffer size: %.0f MB\n", [ctx->device maxBufferLength] / (1024.0 * 1024.0));

    // Create command queue
    ctx->queue = [ctx->device newCommandQueue];
    if (!ctx->queue) {
        fprintf(stderr, "ERROR: Failed to create command queue\n");
        free(ctx);
        return NULL;
    }

    // Load shader source and compile at runtime
    // (Metal offline compiler may not be available on all systems)
    NSError *error = nil;

    // Try loading pre-compiled metallib first, then fall back to source compilation
    NSString *execPath = [[NSBundle mainBundle] executablePath];
    NSString *execDir = [execPath stringByDeletingLastPathComponent];

    // Search for metallib
    NSArray *metallib_paths = @[
        [execDir stringByAppendingPathComponent:@"shaders.metallib"],
        @"shaders.metallib",
        @"metal_infer/shaders.metallib"
    ];
    for (NSString *libPath in metallib_paths) {
        if ([[NSFileManager defaultManager] fileExistsAtPath:libPath]) {
            NSURL *libURL = [NSURL fileURLWithPath:libPath];
            ctx->library = [ctx->device newLibraryWithURL:libURL error:&error];
            if (ctx->library) {
                printf("[metal] Loaded pre-compiled shader library: %s\n", [libPath UTF8String]);
                break;
            }
        }
    }

    // Fall back: compile from source
    if (!ctx->library) {
        NSArray *source_paths = @[
            [execDir stringByAppendingPathComponent:@"shaders.metal"],
            @"shaders.metal",
            @"metal_infer/shaders.metal"
        ];
        NSString *shaderSource = nil;
        NSString *foundPath = nil;
        for (NSString *srcPath in source_paths) {
            shaderSource = [NSString stringWithContentsOfFile:srcPath
                                                    encoding:NSUTF8StringEncoding
                                                       error:&error];
            if (shaderSource) {
                foundPath = srcPath;
                break;
            }
        }
        if (!shaderSource) {
            fprintf(stderr, "ERROR: Could not find shaders.metal or shaders.metallib\n");
            free(ctx);
            return NULL;
        }

        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        opts.mathMode = MTLMathModeFast;
        opts.languageVersion = MTLLanguageVersion3_1;

        printf("[metal] Compiling shaders from source: %s ...\n", [foundPath UTF8String]);
        double t_compile = now_ms();
        ctx->library = [ctx->device newLibraryWithSource:shaderSource
                                                 options:opts
                                                   error:&error];
        if (!ctx->library) {
            fprintf(stderr, "ERROR: Shader compilation failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            free(ctx);
            return NULL;
        }
        printf("[metal] Shader compilation: %.0f ms\n", now_ms() - t_compile);
    }

    // Create pipeline states for each kernel
    // Helper block to create a pipeline from a function name
    id<MTLComputePipelineState> (^makePipeline)(NSString *) = ^(NSString *name) {
        id<MTLFunction> fn = [ctx->library newFunctionWithName:name];
        if (!fn) {
            fprintf(stderr, "ERROR: Shader function '%s' not found\n", [name UTF8String]);
            return (id<MTLComputePipelineState>)nil;
        }
        NSError *pipeError = nil;
        id<MTLComputePipelineState> ps =
            [ctx->device newComputePipelineStateWithFunction:fn error:&pipeError];
        if (!ps) {
            fprintf(stderr, "ERROR: Failed to create pipeline for '%s': %s\n",
                    [name UTF8String], [[pipeError localizedDescription] UTF8String]);
            return (id<MTLComputePipelineState>)nil;
        }
        printf("[metal] Pipeline '%s': maxTotalThreadsPerThreadgroup=%lu\n",
               [name UTF8String], (unsigned long)[ps maxTotalThreadsPerThreadgroup]);
        return ps;
    };

    ctx->matvec_naive = makePipeline(@"dequant_matvec_4bit");
    ctx->matvec_fast  = makePipeline(@"dequant_matvec_4bit_fast");
    ctx->swiglu       = makePipeline(@"swiglu_fused");
    ctx->weighted_sum = makePipeline(@"weighted_sum");
    ctx->rms_norm_sum = makePipeline(@"rms_norm_sum_sq");
    ctx->rms_norm_apply = makePipeline(@"rms_norm_apply");

    if (!ctx->matvec_naive || !ctx->matvec_fast || !ctx->swiglu ||
        !ctx->weighted_sum || !ctx->rms_norm_sum || !ctx->rms_norm_apply) {
        free(ctx);
        return NULL;
    }

    return ctx;
}

static void metal_destroy(MetalContext *ctx) {
    if (ctx) free(ctx);
}

// ============================================================================
// Metal buffer helpers
// ============================================================================

// Create a shared-memory Metal buffer (CPU and GPU see the same memory)
static id<MTLBuffer> metal_buf_shared(MetalContext *ctx, size_t size) {
    id<MTLBuffer> buf = [ctx->device newBufferWithLength:size
                                                options:MTLResourceStorageModeShared];
    if (!buf) {
        fprintf(stderr, "ERROR: Failed to allocate Metal buffer of %zu bytes\n", size);
    }
    return buf;
}

// Create a shared buffer and fill it with pread from fd
static id<MTLBuffer> metal_buf_pread(MetalContext *ctx, int fd, size_t size, off_t offset) {
    id<MTLBuffer> buf = metal_buf_shared(ctx, size);
    if (!buf) return nil;

    ssize_t nread = pread(fd, [buf contents], size, offset);
    if (nread != (ssize_t)size) {
        fprintf(stderr, "ERROR: pread returned %zd, expected %zu (errno=%d)\n",
                nread, size, errno);
        return nil;
    }
    return buf;
}

// ============================================================================
// Run a single dequant matvec on Metal (supports buffer offsets)
// ============================================================================

static void metal_dequant_matvec_offset(
    MetalContext *ctx,
    id<MTLCommandBuffer> cmdbuf,
    id<MTLBuffer> W_packed,  NSUInteger w_offset,
    id<MTLBuffer> scales,    NSUInteger s_offset,
    id<MTLBuffer> biases,    NSUInteger b_offset,
    id<MTLBuffer> x,         NSUInteger x_offset,
    id<MTLBuffer> out,       NSUInteger o_offset,
    uint32_t out_dim,
    uint32_t in_dim,
    uint32_t group_size,
    int use_fast
) {
    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
    id<MTLComputePipelineState> pipeline = use_fast ? ctx->matvec_fast : ctx->matvec_naive;
    [enc setComputePipelineState:pipeline];

    [enc setBuffer:W_packed offset:w_offset atIndex:0];
    [enc setBuffer:scales   offset:s_offset atIndex:1];
    [enc setBuffer:biases   offset:b_offset atIndex:2];
    [enc setBuffer:x        offset:x_offset atIndex:3];
    [enc setBuffer:out      offset:o_offset atIndex:4];
    [enc setBytes:&out_dim    length:sizeof(uint32_t) atIndex:5];
    [enc setBytes:&in_dim     length:sizeof(uint32_t) atIndex:6];
    [enc setBytes:&group_size length:sizeof(uint32_t) atIndex:7];

    if (use_fast) {
        NSUInteger tg_size = MIN(64, [pipeline maxTotalThreadsPerThreadgroup]);
        MTLSize numGroups = MTLSizeMake(out_dim, 1, 1);
        MTLSize tgSize = MTLSizeMake(tg_size, 1, 1);
        [enc dispatchThreadgroups:numGroups threadsPerThreadgroup:tgSize];
    } else {
        NSUInteger tg_size = MIN(256, [pipeline maxTotalThreadsPerThreadgroup]);
        MTLSize tgSize = MTLSizeMake(tg_size, 1, 1);
        MTLSize numGroups = MTLSizeMake((out_dim + tg_size - 1) / tg_size, 1, 1);
        [enc dispatchThreadgroups:numGroups threadsPerThreadgroup:tgSize];
    }

    [enc endEncoding];
}

// Convenience wrapper with zero offsets
static void metal_dequant_matvec(
    MetalContext *ctx,
    id<MTLCommandBuffer> cmdbuf,
    id<MTLBuffer> W_packed,
    id<MTLBuffer> scales,
    id<MTLBuffer> biases,
    id<MTLBuffer> x,
    id<MTLBuffer> out,
    uint32_t out_dim,
    uint32_t in_dim,
    uint32_t group_size,
    int use_fast
) {
    metal_dequant_matvec_offset(ctx, cmdbuf, W_packed, 0, scales, 0, biases, 0,
                                 x, 0, out, 0, out_dim, in_dim, group_size, use_fast);
}

// ============================================================================
// Run SwiGLU on Metal
// ============================================================================

static void metal_swiglu(
    MetalContext *ctx,
    id<MTLCommandBuffer> cmdbuf,
    id<MTLBuffer> gate,
    id<MTLBuffer> up,
    id<MTLBuffer> out,
    uint32_t dim
) {
    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
    [enc setComputePipelineState:ctx->swiglu];
    [enc setBuffer:gate offset:0 atIndex:0];
    [enc setBuffer:up   offset:0 atIndex:1];
    [enc setBuffer:out  offset:0 atIndex:2];
    [enc setBytes:&dim length:sizeof(uint32_t) atIndex:3];

    NSUInteger tg_size = MIN(256, [ctx->swiglu maxTotalThreadsPerThreadgroup]);
    MTLSize numGroups = MTLSizeMake((dim + tg_size - 1) / tg_size, 1, 1);
    [enc dispatchThreadgroups:numGroups threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
    [enc endEncoding];
}

// ============================================================================
// Full expert forward pass: gate/up -> SwiGLU -> down
// ============================================================================

typedef struct {
    double io_ms;       // time to pread expert weights
    double compute_ms;  // time for Metal compute
    double total_ms;    // end-to-end
    size_t io_bytes;    // bytes read from SSD
} ExpertTiming;

// Original version: 9 separate preads, 9 separate Metal buffers
static ExpertTiming run_expert_forward(
    MetalContext *ctx,
    int packed_fd,
    int expert_idx,
    id<MTLBuffer> x_buf,       // [HIDDEN_DIM] float input
    id<MTLBuffer> out_buf,     // [HIDDEN_DIM] float output
    int use_fast
) {
    ExpertTiming timing = {0};
    double t0 = now_ms();

    // ---- I/O: pread all 9 components for this expert ----
    off_t expert_offset = (off_t)expert_idx * EXPERT_SIZE;

    double t_io_start = now_ms();

    id<MTLBuffer> gate_w = metal_buf_pread(ctx, packed_fd, GATE_W_SIZE, expert_offset + GATE_W_OFFSET);
    id<MTLBuffer> gate_s = metal_buf_pread(ctx, packed_fd, GATE_S_SIZE, expert_offset + GATE_S_OFFSET);
    id<MTLBuffer> gate_b = metal_buf_pread(ctx, packed_fd, GATE_B_SIZE, expert_offset + GATE_B_OFFSET);
    id<MTLBuffer> up_w   = metal_buf_pread(ctx, packed_fd, UP_W_SIZE,   expert_offset + UP_W_OFFSET);
    id<MTLBuffer> up_s   = metal_buf_pread(ctx, packed_fd, UP_S_SIZE,   expert_offset + UP_S_OFFSET);
    id<MTLBuffer> up_b   = metal_buf_pread(ctx, packed_fd, UP_B_SIZE,   expert_offset + UP_B_OFFSET);
    id<MTLBuffer> down_w = metal_buf_pread(ctx, packed_fd, DOWN_W_SIZE, expert_offset + DOWN_W_OFFSET);
    id<MTLBuffer> down_s = metal_buf_pread(ctx, packed_fd, DOWN_S_SIZE, expert_offset + DOWN_S_OFFSET);
    id<MTLBuffer> down_b = metal_buf_pread(ctx, packed_fd, DOWN_B_SIZE, expert_offset + DOWN_B_OFFSET);

    double t_io_end = now_ms();
    timing.io_ms = t_io_end - t_io_start;
    timing.io_bytes = EXPERT_SIZE;

    if (!gate_w || !gate_s || !gate_b || !up_w || !up_s || !up_b ||
        !down_w || !down_s || !down_b) {
        fprintf(stderr, "ERROR: Failed to load expert %d weights\n", expert_idx);
        timing.total_ms = now_ms() - t0;
        return timing;
    }

    // ---- Compute: gate/up matvecs -> SwiGLU -> down matvec ----

    // Intermediate buffers
    id<MTLBuffer> gate_out = metal_buf_shared(ctx, INTERMEDIATE_DIM * sizeof(float));
    id<MTLBuffer> up_out   = metal_buf_shared(ctx, INTERMEDIATE_DIM * sizeof(float));
    id<MTLBuffer> act_out  = metal_buf_shared(ctx, INTERMEDIATE_DIM * sizeof(float));

    uint32_t hidden = HIDDEN_DIM;
    uint32_t inter  = INTERMEDIATE_DIM;
    uint32_t gs     = GROUP_SIZE;

    double t_compute_start = now_ms();

    // Create a single command buffer for the full expert pipeline
    id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];

    // gate_proj: [4096] -> [1024]
    metal_dequant_matvec(ctx, cmdbuf, gate_w, gate_s, gate_b, x_buf, gate_out,
                         inter, hidden, gs, use_fast);

    // up_proj: [4096] -> [1024]
    metal_dequant_matvec(ctx, cmdbuf, up_w, up_s, up_b, x_buf, up_out,
                         inter, hidden, gs, use_fast);

    // SwiGLU: silu(gate) * up -> [1024]
    metal_swiglu(ctx, cmdbuf, gate_out, up_out, act_out, inter);

    // down_proj: [1024] -> [4096]
    metal_dequant_matvec(ctx, cmdbuf, down_w, down_s, down_b, act_out, out_buf,
                         hidden, inter, gs, use_fast);

    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];

    double t_compute_end = now_ms();
    timing.compute_ms = t_compute_end - t_compute_start;
    timing.total_ms = now_ms() - t0;

    return timing;
}

// ============================================================================
// Optimized expert forward: single pread, buffer offsets
// ============================================================================
// One 7.08 MB pread per expert instead of 9 separate reads.
// Uses Metal buffer offsets to point shaders at the right data within the buffer.
// Eliminates 9x Metal buffer allocation overhead per expert.

static ExpertTiming run_expert_forward_fast(
    MetalContext *ctx,
    int packed_fd,
    int expert_idx,
    id<MTLBuffer> expert_buf,  // Pre-allocated buffer for one expert (EXPERT_SIZE bytes)
    id<MTLBuffer> x_buf,       // [HIDDEN_DIM] float input
    id<MTLBuffer> gate_out,    // [INTERMEDIATE_DIM] float scratch
    id<MTLBuffer> up_out,      // [INTERMEDIATE_DIM] float scratch
    id<MTLBuffer> act_out,     // [INTERMEDIATE_DIM] float scratch
    id<MTLBuffer> out_buf,     // [HIDDEN_DIM] float output
    int use_fast
) {
    ExpertTiming timing = {0};
    double t0 = now_ms();

    // ---- I/O: single pread for the entire expert ----
    off_t expert_offset = (off_t)expert_idx * EXPERT_SIZE;

    double t_io_start = now_ms();
    ssize_t nread = pread(packed_fd, [expert_buf contents], EXPERT_SIZE, expert_offset);
    double t_io_end = now_ms();

    timing.io_ms = t_io_end - t_io_start;
    timing.io_bytes = EXPERT_SIZE;

    if (nread != EXPERT_SIZE) {
        fprintf(stderr, "ERROR: pread expert %d: got %zd, expected %d\n",
                expert_idx, nread, EXPERT_SIZE);
        timing.total_ms = now_ms() - t0;
        return timing;
    }

    // ---- Compute using buffer offsets ----
    uint32_t hidden = HIDDEN_DIM;
    uint32_t inter  = INTERMEDIATE_DIM;
    uint32_t gs     = GROUP_SIZE;

    double t_compute_start = now_ms();
    id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];

    // gate_proj: [4096] -> [1024]
    metal_dequant_matvec_offset(ctx, cmdbuf,
        expert_buf, GATE_W_OFFSET,
        expert_buf, GATE_S_OFFSET,
        expert_buf, GATE_B_OFFSET,
        x_buf, 0,
        gate_out, 0,
        inter, hidden, gs, use_fast);

    // up_proj: [4096] -> [1024]
    metal_dequant_matvec_offset(ctx, cmdbuf,
        expert_buf, UP_W_OFFSET,
        expert_buf, UP_S_OFFSET,
        expert_buf, UP_B_OFFSET,
        x_buf, 0,
        up_out, 0,
        inter, hidden, gs, use_fast);

    // SwiGLU
    metal_swiglu(ctx, cmdbuf, gate_out, up_out, act_out, inter);

    // down_proj: [1024] -> [4096]
    metal_dequant_matvec_offset(ctx, cmdbuf,
        expert_buf, DOWN_W_OFFSET,
        expert_buf, DOWN_S_OFFSET,
        expert_buf, DOWN_B_OFFSET,
        act_out, 0,
        out_buf, 0,
        hidden, inter, gs, use_fast);

    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];

    double t_compute_end = now_ms();
    timing.compute_ms = t_compute_end - t_compute_start;
    timing.total_ms = now_ms() - t0;

    return timing;
}

// ============================================================================
// Run full MoE: K experts, weighted combination
// ============================================================================

typedef struct {
    double io_ms;
    double compute_ms;
    double combine_ms;
    double total_ms;
    size_t io_bytes;
} MoETiming;

static MoETiming run_moe_forward(
    MetalContext *ctx,
    int packed_fd,
    const int *expert_indices,   // [K]
    const float *expert_weights, // [K]
    int K,
    id<MTLBuffer> x_buf,        // [HIDDEN_DIM] input
    id<MTLBuffer> moe_out_buf,  // [HIDDEN_DIM] output
    int use_fast
) {
    MoETiming timing = {0};
    double t0 = now_ms();

    // Allocate per-expert output buffers (use NSMutableArray for ARC)
    NSMutableArray<id<MTLBuffer>> *expert_outs = [NSMutableArray arrayWithCapacity:K];
    for (int k = 0; k < K; k++) {
        [expert_outs addObject:metal_buf_shared(ctx, HIDDEN_DIM * sizeof(float))];
    }

    // Run each expert
    for (int k = 0; k < K; k++) {
        ExpertTiming et = run_expert_forward(ctx, packed_fd, expert_indices[k],
                                              x_buf, expert_outs[k], use_fast);
        timing.io_ms += et.io_ms;
        timing.compute_ms += et.compute_ms;
        timing.io_bytes += et.io_bytes;
    }

    // Combine: out = sum(w[k] * expert_out[k])
    double t_combine = now_ms();

    // Stack expert outputs into contiguous buffer for the weighted_sum kernel
    id<MTLBuffer> stacked = metal_buf_shared(ctx, K * HIDDEN_DIM * sizeof(float));
    float *stacked_ptr = (float *)[stacked contents];
    for (int k = 0; k < K; k++) {
        memcpy(stacked_ptr + k * HIDDEN_DIM,
               [expert_outs[k] contents],
               HIDDEN_DIM * sizeof(float));
    }

    // Upload weights
    id<MTLBuffer> w_buf = metal_buf_shared(ctx, K * sizeof(float));
    memcpy([w_buf contents], expert_weights, K * sizeof(float));

    // Run weighted sum kernel
    uint32_t k_val = (uint32_t)K;
    uint32_t dim_val = HIDDEN_DIM;

    id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
    [enc setComputePipelineState:ctx->weighted_sum];
    [enc setBuffer:stacked    offset:0 atIndex:0];
    [enc setBuffer:w_buf      offset:0 atIndex:1];
    [enc setBuffer:moe_out_buf offset:0 atIndex:2];
    [enc setBytes:&k_val   length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&dim_val length:sizeof(uint32_t) atIndex:4];

    NSUInteger tg_size = MIN(256, [ctx->weighted_sum maxTotalThreadsPerThreadgroup]);
    MTLSize numGroups = MTLSizeMake((HIDDEN_DIM + tg_size - 1) / tg_size, 1, 1);
    [enc dispatchThreadgroups:numGroups threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
    [enc endEncoding];

    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];

    timing.combine_ms = now_ms() - t_combine;
    timing.total_ms = now_ms() - t0;

    return timing;
}

// ============================================================================
// CPU reference for full expert forward (for verification)
// ============================================================================

static void cpu_expert_forward(
    int packed_fd,
    int expert_idx,
    const float *x,     // [HIDDEN_DIM]
    float *out          // [HIDDEN_DIM]
) {
    off_t expert_offset = (off_t)expert_idx * EXPERT_SIZE;

    // Read all components
    uint32_t gate_w[GATE_W_SIZE / 4];
    uint16_t gate_s[GATE_S_SIZE / 2];
    uint16_t gate_b[GATE_B_SIZE / 2];
    uint32_t up_w[UP_W_SIZE / 4];
    uint16_t up_s[UP_S_SIZE / 2];
    uint16_t up_b[UP_B_SIZE / 2];
    uint32_t down_w[DOWN_W_SIZE / 4];
    uint16_t down_s[DOWN_S_SIZE / 2];
    uint16_t down_b[DOWN_B_SIZE / 2];

    pread(packed_fd, gate_w, GATE_W_SIZE, expert_offset + GATE_W_OFFSET);
    pread(packed_fd, gate_s, GATE_S_SIZE, expert_offset + GATE_S_OFFSET);
    pread(packed_fd, gate_b, GATE_B_SIZE, expert_offset + GATE_B_OFFSET);
    pread(packed_fd, up_w,   UP_W_SIZE,   expert_offset + UP_W_OFFSET);
    pread(packed_fd, up_s,   UP_S_SIZE,   expert_offset + UP_S_OFFSET);
    pread(packed_fd, up_b,   UP_B_SIZE,   expert_offset + UP_B_OFFSET);
    pread(packed_fd, down_w, DOWN_W_SIZE, expert_offset + DOWN_W_OFFSET);
    pread(packed_fd, down_s, DOWN_S_SIZE, expert_offset + DOWN_S_OFFSET);
    pread(packed_fd, down_b, DOWN_B_SIZE, expert_offset + DOWN_B_OFFSET);

    // gate_proj: [4096] -> [1024]
    float gate_out[INTERMEDIATE_DIM];
    cpu_dequant_matvec_4bit(gate_w, gate_s, gate_b, x, gate_out,
                            INTERMEDIATE_DIM, HIDDEN_DIM, GROUP_SIZE);

    // up_proj: [4096] -> [1024]
    float up_out[INTERMEDIATE_DIM];
    cpu_dequant_matvec_4bit(up_w, up_s, up_b, x, up_out,
                            INTERMEDIATE_DIM, HIDDEN_DIM, GROUP_SIZE);

    // SwiGLU
    float act_out[INTERMEDIATE_DIM];
    cpu_swiglu(gate_out, up_out, act_out, INTERMEDIATE_DIM);

    // down_proj: [1024] -> [4096]
    cpu_dequant_matvec_4bit(down_w, down_s, down_b, act_out, out,
                            HIDDEN_DIM, INTERMEDIATE_DIM, GROUP_SIZE);
}

// ============================================================================
// Main
// ============================================================================

static void print_usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  --layer N        Layer index (default: 0)\n");
    printf("  --expert E       Expert index (default: 0)\n");
    printf("  --benchmark      Run timing benchmark (10 iterations)\n");
    printf("  --moe            Run full MoE with 10 experts\n");
    printf("  --verify         Verify Metal output against CPU reference\n");
    printf("  --fast           Use threadgroup-optimized shader\n");
    printf("  --model PATH     Model path (default: built-in)\n");
    printf("  --help           This message\n");
}

int main(int argc, char **argv) {
    @autoreleasepool {
        int layer_idx = 0;
        int expert_idx = 0;
        int do_benchmark = 0;
        int do_moe = 0;
        int do_verify = 0;
        int use_fast = 0;
        const char *model_path = MODEL_PATH;

        static struct option long_options[] = {
            {"layer",     required_argument, 0, 'l'},
            {"expert",    required_argument, 0, 'e'},
            {"benchmark", no_argument,       0, 'b'},
            {"moe",       no_argument,       0, 'm'},
            {"verify",    no_argument,       0, 'v'},
            {"fast",      no_argument,       0, 'f'},
            {"model",     required_argument, 0, 'p'},
            {"help",      no_argument,       0, 'h'},
            {0, 0, 0, 0}
        };

        int c;
        while ((c = getopt_long(argc, argv, "l:e:bmvfp:h", long_options, NULL)) != -1) {
            switch (c) {
                case 'l': layer_idx = atoi(optarg); break;
                case 'e': expert_idx = atoi(optarg); break;
                case 'b': do_benchmark = 1; break;
                case 'm': do_moe = 1; break;
                case 'v': do_verify = 1; break;
                case 'f': use_fast = 1; break;
                case 'p': model_path = optarg; break;
                case 'h': print_usage(argv[0]); return 0;
                default:  print_usage(argv[0]); return 1;
            }
        }

        printf("=== metal_infer: 4-bit dequant MoE engine ===\n");
        printf("Layer: %d, Expert: %d, Fast: %s, Benchmark: %s, MoE: %s, Verify: %s\n",
               layer_idx, expert_idx,
               use_fast ? "YES" : "NO",
               do_benchmark ? "YES" : "NO",
               do_moe ? "YES" : "NO",
               do_verify ? "YES" : "NO");

        // ---- Initialize Metal ----
        MetalContext *ctx = metal_init();
        if (!ctx) return 1;

        // ---- Open packed expert file ----
        char packed_path[1024];
        snprintf(packed_path, sizeof(packed_path),
                 "%s/packed_experts/layer_%02d.bin", model_path, layer_idx);

        printf("[io] Opening: %s\n", packed_path);
        int packed_fd = open(packed_path, O_RDONLY);
        if (packed_fd < 0) {
            fprintf(stderr, "ERROR: Cannot open %s: %s\n", packed_path, strerror(errno));
            metal_destroy(ctx);
            return 1;
        }
        printf("[io] Opened layer %d packed file (fd=%d)\n", layer_idx, packed_fd);

        // ---- Create input vector with deterministic values ----
        // Use realistic magnitude (~1.0) to stress-test numerical accuracy
        id<MTLBuffer> x_buf = metal_buf_shared(ctx, HIDDEN_DIM * sizeof(float));
        float *x_data = (float *)[x_buf contents];
        for (int i = 0; i < HIDDEN_DIM; i++) {
            x_data[i] = 0.1f * sinf((float)i * 0.1f + 0.3f);
        }
        printf("[init] Input vector: x[0..3] = [%.6f, %.6f, %.6f, %.6f]\n",
               x_data[0], x_data[1], x_data[2], x_data[3]);

        // ---- Output buffer ----
        id<MTLBuffer> out_buf = metal_buf_shared(ctx, HIDDEN_DIM * sizeof(float));

        // ========== Single expert forward ==========
        if (!do_moe) {
            printf("\n--- Single expert forward (expert %d) ---\n", expert_idx);

            ExpertTiming et = run_expert_forward(ctx, packed_fd, expert_idx,
                                                  x_buf, out_buf, use_fast);

            float *out_data = (float *)[out_buf contents];
            printf("[result] out[0..7] = [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f]\n",
                   out_data[0], out_data[1], out_data[2], out_data[3],
                   out_data[4], out_data[5], out_data[6], out_data[7]);
            printf("[timing] I/O: %.2f ms (%.1f GB/s), Compute: %.2f ms, Total: %.2f ms\n",
                   et.io_ms, et.io_bytes / (et.io_ms * 1e6),
                   et.compute_ms, et.total_ms);

            // ---- Verify against CPU ----
            if (do_verify) {
                printf("\n--- CPU verification ---\n");
                float *cpu_out = calloc(HIDDEN_DIM, sizeof(float));
                double t_cpu = now_ms();
                cpu_expert_forward(packed_fd, expert_idx, x_data, cpu_out);
                double cpu_ms = now_ms() - t_cpu;

                printf("[cpu] Time: %.2f ms\n", cpu_ms);
                printf("[cpu] out[0..7] = [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f]\n",
                       cpu_out[0], cpu_out[1], cpu_out[2], cpu_out[3],
                       cpu_out[4], cpu_out[5], cpu_out[6], cpu_out[7]);

                // Compare
                float max_diff = 0.0f;
                float max_rel_diff = 0.0f;
                int worst_idx = 0;
                for (int i = 0; i < HIDDEN_DIM; i++) {
                    float diff = fabsf(out_data[i] - cpu_out[i]);
                    float rel = (fabsf(cpu_out[i]) > 1e-6f) ? diff / fabsf(cpu_out[i]) : diff;
                    if (diff > max_diff) {
                        max_diff = diff;
                        worst_idx = i;
                    }
                    if (rel > max_rel_diff) max_rel_diff = rel;
                }
                printf("[verify] Max abs diff: %.6f at index %d (GPU=%.6f, CPU=%.6f)\n",
                       max_diff, worst_idx, out_data[worst_idx], cpu_out[worst_idx]);
                printf("[verify] Max rel diff: %.6f\n", max_rel_diff);
                printf("[verify] %s (threshold: 0.01)\n",
                       max_rel_diff < 0.01f ? "PASS" : "FAIL");
                printf("[verify] GPU speedup: %.1fx vs CPU\n", cpu_ms / et.compute_ms);
                free(cpu_out);
            }

            // ---- Benchmark ----
            if (do_benchmark) {
                printf("\n--- Benchmark (10 iterations) ---\n");
                double io_sum = 0, compute_sum = 0, total_sum = 0;
                int N = 10;
                for (int i = 0; i < N; i++) {
                    ExpertTiming bt = run_expert_forward(ctx, packed_fd, expert_idx,
                                                          x_buf, out_buf, use_fast);
                    io_sum += bt.io_ms;
                    compute_sum += bt.compute_ms;
                    total_sum += bt.total_ms;
                    printf("  [%d] io=%.2f ms, compute=%.2f ms, total=%.2f ms\n",
                           i, bt.io_ms, bt.compute_ms, bt.total_ms);
                }
                printf("[bench] Average: io=%.2f ms, compute=%.2f ms, total=%.2f ms\n",
                       io_sum / N, compute_sum / N, total_sum / N);
                printf("[bench] I/O throughput: %.1f GB/s\n",
                       EXPERT_SIZE * N / (io_sum * 1e6));
            }
        }

        // ========== Full MoE forward (10 experts) ==========
        if (do_moe) {
            printf("\n--- Full MoE forward (10 experts) ---\n");

            // Simulated routing: experts 0-9 with uniform weights
            int K = 10;
            int moe_experts[10] = {0, 42, 100, 155, 200, 256, 300, 350, 400, 450};
            float moe_weights[10];
            float wsum = 0.0f;
            for (int k = 0; k < K; k++) {
                moe_weights[k] = 1.0f / (float)(k + 1);
                wsum += moe_weights[k];
            }
            // Normalize weights (as softmax would)
            for (int k = 0; k < K; k++) {
                moe_weights[k] /= wsum;
            }

            printf("[moe] Experts: ");
            for (int k = 0; k < K; k++) printf("%d(%.3f) ", moe_experts[k], moe_weights[k]);
            printf("\n");

            id<MTLBuffer> moe_out = metal_buf_shared(ctx, HIDDEN_DIM * sizeof(float));

            MoETiming mt = run_moe_forward(ctx, packed_fd, moe_experts, moe_weights, K,
                                            x_buf, moe_out, use_fast);

            float *moe_data = (float *)[moe_out contents];
            printf("[result] out[0..7] = [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f]\n",
                   moe_data[0], moe_data[1], moe_data[2], moe_data[3],
                   moe_data[4], moe_data[5], moe_data[6], moe_data[7]);
            printf("[timing] I/O: %.2f ms (%.1f GB/s)\n",
                   mt.io_ms, (EXPERT_SIZE * K) / (mt.io_ms * 1e6));
            printf("[timing] Compute: %.2f ms\n", mt.compute_ms);
            printf("[timing] Combine: %.2f ms\n", mt.combine_ms);
            printf("[timing] Total: %.2f ms\n", mt.total_ms);
            printf("[timing] Experts/sec: %.0f\n", K / (mt.total_ms / 1000.0));

            // Benchmark MoE
            if (do_benchmark) {
                printf("\n--- MoE Benchmark (10 iterations) ---\n");
                int N = 10;
                double total_time = 0;
                for (int i = 0; i < N; i++) {
                    MoETiming bt = run_moe_forward(ctx, packed_fd, moe_experts, moe_weights, K,
                                                    x_buf, moe_out, use_fast);
                    total_time += bt.total_ms;
                    printf("  [%d] io=%.2f compute=%.2f combine=%.2f total=%.2f ms\n",
                           i, bt.io_ms, bt.compute_ms, bt.combine_ms, bt.total_ms);
                }
                printf("[bench] Average MoE time: %.2f ms (%.0f experts/sec)\n",
                       total_time / N, K * N / (total_time / 1000.0));
                printf("[bench] If this were the whole token: %.1f tok/s\n",
                       1000.0 / (total_time / N));
            }
        }

        // Cleanup
        close(packed_fd);
        metal_destroy(ctx);

        printf("\nDone.\n");
        return 0;
    }
}
