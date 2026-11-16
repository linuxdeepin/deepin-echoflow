#include "common.h"

// ── CrispASR patch (PR #07-metal-aa-snake-beta) ─────────────────────
// Fused BigVGAN v2 anti-aliased SnakeBeta. One threadgroup per
// (channel × seq-chunk × batch). Each thread owns AA_BUFFER_SIZE
// output samples and keeps the upsample/intermediate/output arrays
// in thread-private memory (no threadgroup memory, no global round-
// trips between the upsample / snake / downsample stages).
//
// Ported from upstream IndexTTS CUDA reference
// (anti_alias_activation_cuda.cu, Apache 2.0). Layout mirrors that
// kernel exactly — kept literal so the next NVIDIA bump is mergable.
// MUST RE-APPLY after every ggml bump.
constant constexpr int AA_FILTER_SIZE      = 12;
constant constexpr int AA_HALF_FILTER_SIZE = 6;
constant constexpr int AA_BUFFER_SIZE      = 32;
constant constexpr int AA_UP_REP_PAD       = 5;
constant constexpr int AA_DS_REP_PAD_LEFT  = 5;
constant constexpr int AA_DS_REP_PAD_RIGHT = 6;
constant constexpr int AA_THREADS_PER_TGRP = 128;

template <typename T>
kernel void kernel_aa_snake_beta_impl(
        constant ggml_metal_kargs_aa_snake_beta & args      [[buffer(0)]],
        device const T                          * src       [[buffer(1)]],
        device const T                          * log_alpha [[buffer(2)]],
        device const T                          * log_beta  [[buffer(3)]],
        device const T                          * up_ftr    [[buffer(4)]],
        device const T                          * down_ftr  [[buffer(5)]],
        device T                                * dst       [[buffer(6)]],
        uint3                                     tgpig     [[threadgroup_position_in_grid]],
        uint3                                     tpitg     [[thread_position_in_threadgroup]]) {

    const int seq_blk = (int) tgpig.x;   // chunk index
    const int chan    = (int) tgpig.y;   // channel
    const int tid     = (int) tpitg.x;
    const int seq_len = args.T;

    // Each thread covers [seq_offset, seq_offset + AA_BUFFER_SIZE).
    const int seq_offset = seq_blk * AA_THREADS_PER_TGRP * AA_BUFFER_SIZE
                         + tid * AA_BUFFER_SIZE;
    const int intermediate_seq_offset = seq_offset * 2;

    // Pointer to this thread's start in the per-channel slice
    // (layout matches ggml's [T, C] — T-fastest, channel-major).
    const int channel_offset = chan * seq_len;
    device const T * src_ch  = src + channel_offset;
    device       T * dst_ch  = dst + channel_offset;

    // Replicate-pad reference values (always defined; safe even when
    // seq_offset >= seq_len because we bounds-check writes below).
    const float seq_left  = (float) src_ch[0];
    const float seq_right = (float) src_ch[seq_len - 1];

    // Per-channel snake-beta gains.
    const float alpha_val = exp((float) log_alpha[chan]);
    const float beta_val  = exp((float) log_beta[chan]);

    // Filter taps copied into thread-private storage. Upstream applies
    // the ×2 zero-stuff gain on the element loads, not the filter, so
    // we mirror that to keep the GGUF filter untouched.
    float up_filter[AA_FILTER_SIZE];
    float down_filter[AA_FILTER_SIZE];
    #pragma unroll
    for (int k = 0; k < AA_FILTER_SIZE; k++) {
        up_filter[k]   = (float) up_ftr[k];
        down_filter[k] = (float) down_ftr[k];
    }

    // Thread-private workspaces. Size matches upstream exactly so the
    // semantics stay literal — see the upstream comment about the
    // DOWNSAMPLE_REPLICATION_PAD_LEFT headroom in `intermediates`.
    float elements[2 * AA_FILTER_SIZE + 2 * AA_BUFFER_SIZE + 2 * AA_UP_REP_PAD] = {0};
    float intermediates[2 * AA_FILTER_SIZE + 2 * AA_BUFFER_SIZE
                        + AA_DS_REP_PAD_LEFT + AA_DS_REP_PAD_RIGHT]              = {0};
    float output[AA_BUFFER_SIZE] = {0};

    // ── 1. Load elements with replicate-pad and zero-stuff. ─────────
    // Writes go to even indices `2*(HALF + it)`; odd indices stay 0.
    // The ×2 factor on the loaded value is the zero-stuff gain.
    #pragma unroll
    for (int it = -AA_HALF_FILTER_SIZE; it < AA_BUFFER_SIZE + AA_HALF_FILTER_SIZE; it++) {
        const int element_index = seq_offset + it;
        const int slot = 2 * (AA_HALF_FILTER_SIZE + it);
        if ((element_index < 0) && (element_index >= -AA_UP_REP_PAD)) {
            elements[slot] = 2.0f * seq_left;
        } else if ((element_index >= seq_len) && (element_index < seq_len + AA_UP_REP_PAD)) {
            elements[slot] = 2.0f * seq_right;
        } else if ((element_index >= 0) && (element_index < seq_len)) {
            elements[slot] = 2.0f * (float) src_ch[element_index];
        }
    }

    // ── 2. Upsample FIR. Writes go into `intermediates` with the left
    // downsample-pad headroom reserved.
    #pragma unroll
    for (int it = 0; it < (2 * AA_BUFFER_SIZE + 2 * AA_FILTER_SIZE); it++) {
        float acc = 0.0f;
        const int element_index = intermediate_seq_offset + it;
        #pragma unroll
        for (int f = 0; f < AA_FILTER_SIZE; f++) {
            if ((element_index + f) >= 0) {
                acc += up_filter[f] * elements[it + f];
            }
        }
        intermediates[it + AA_DS_REP_PAD_LEFT] = acc;
    }

    // ── 3. SnakeBeta in place on the active range.
    const float inv_beta = 1.0f / (beta_val + 1e-9f);
    #pragma unroll
    for (int it = 0; it < 2 * AA_BUFFER_SIZE + 2 * AA_FILTER_SIZE; it++) {
        const float v = intermediates[it + AA_DS_REP_PAD_LEFT];
        const float s = sin(alpha_val * v);
        intermediates[it + AA_DS_REP_PAD_LEFT] = v + inv_beta * s * s;
    }

    // ── 4. Replicate-pad left/right for the downsample FIR.
    #pragma unroll
    for (int it = 0; it < AA_DS_REP_PAD_LEFT; it++) {
        intermediates[it] = intermediates[AA_DS_REP_PAD_LEFT];
    }
    const int tail_first = AA_DS_REP_PAD_LEFT + 2 * AA_BUFFER_SIZE + 2 * AA_FILTER_SIZE;
    #pragma unroll
    for (int it = 0; it < AA_DS_REP_PAD_RIGHT; it++) {
        intermediates[tail_first + it] = intermediates[tail_first - 1];
    }

    // ── 5. Downsample FIR (stride 2). The `+ AA_DS_REP_PAD_RIGHT`
    // offset matches the upstream torch slice — DO NOT collapse.
    #pragma unroll
    for (int it = 0; it < AA_BUFFER_SIZE; it++) {
        float acc = 0.0f;
        #pragma unroll
        for (int f = 0; f < AA_FILTER_SIZE; f++) {
            acc += down_filter[f] * intermediates[it * 2 + f + AA_DS_REP_PAD_RIGHT];
        }
        output[it] = acc;
    }

    // ── 6. Write back (bounds-checked at the tail thread).
    #pragma unroll
    for (int it = 0; it < AA_BUFFER_SIZE; it++) {
        const int element_index = seq_offset + it;
        if (element_index < seq_len) {
            dst_ch[element_index] = (T) output[it];
        }
    }
}

typedef decltype(kernel_aa_snake_beta_impl<float>) aa_snake_beta_t;
template [[host_name("kernel_aa_snake_beta_f32")]]
kernel aa_snake_beta_t kernel_aa_snake_beta_impl<float>;


// ── CrispASR patch (PR #07-metal-aa-snake-beta) — opt-in variants ───
// Two alternative kernels kept reachable via INDEXTTS_AA_METAL_VARIANT
// so they can be re-benched on different Apple GPU families. Both
// produce numerically-identical output to the default v1 kernel above
// (verified: WAV rmsdiff == 0). On M1 both were measured slower than
// the default during the Phase 3 sweep (see commit notes); on M3+ or
// future architectures the tradeoff may flip.
//
// MUST RE-APPLY after every ggml bump.

// Variant: polyphase upsample with a pre-loaded per-thread input
// window. Halves the upsample mul count (6-tap × 2 phases vs 12-tap
// zero-stuff FIR) and eliminates the ~392 B/thread `elements[]`
// array. On M1 the irregular base-index pattern in the polyphase
// loop empirically regressed th_max from 832 to 768 and wall-clock
// by ~25 %.
template <typename T>
kernel void kernel_aa_snake_beta_polyphase_impl(
        constant ggml_metal_kargs_aa_snake_beta & args      [[buffer(0)]],
        device const T                          * src       [[buffer(1)]],
        device const T                          * log_alpha [[buffer(2)]],
        device const T                          * log_beta  [[buffer(3)]],
        device const T                          * up_ftr    [[buffer(4)]],
        device const T                          * down_ftr  [[buffer(5)]],
        device T                                * dst       [[buffer(6)]],
        uint3                                     tgpig     [[threadgroup_position_in_grid]],
        uint3                                     tpitg     [[thread_position_in_threadgroup]]) {

    const int seq_blk = (int) tgpig.x;
    const int chan    = (int) tgpig.y;
    const int tid     = (int) tpitg.x;
    const int seq_len = args.T;

    const int seq_offset = seq_blk * AA_THREADS_PER_TGRP * AA_BUFFER_SIZE
                         + tid * AA_BUFFER_SIZE;

    const int channel_offset = chan * seq_len;
    device const T * src_ch  = src + channel_offset;
    device       T * dst_ch  = dst + channel_offset;

    const float seq_left  = (float) src_ch[0];
    const float seq_right = (float) src_ch[seq_len - 1];

    const float alpha_val = exp((float) log_alpha[chan]);
    const float beta_val  = exp((float) log_beta[chan]);
    const float inv_beta  = 1.0f / (beta_val + 1e-9f);

    // Bake the ×2 zero-stuff gain into the upsample filter.
    float up_filter[AA_FILTER_SIZE];
    float down_filter[AA_FILTER_SIZE];
    #pragma unroll
    for (int k = 0; k < AA_FILTER_SIZE; k++) {
        up_filter[k]   = 2.0f * (float) up_ftr[k];
        down_filter[k] = (float) down_ftr[k];
    }

    // input_window[j] = input(seq_offset − HALF + j) with replicate-pad.
    constexpr int AA_INPUT_WIN = AA_BUFFER_SIZE + 2 * AA_HALF_FILTER_SIZE;
    float input_window[AA_INPUT_WIN] = {0};
    #pragma unroll
    for (int j = 0; j < AA_INPUT_WIN; j++) {
        const int idx = seq_offset - AA_HALF_FILTER_SIZE + j;
        if (idx >= 0 && idx < seq_len) {
            input_window[j] = (float) src_ch[idx];
        } else if (idx < 0 && idx >= -AA_UP_REP_PAD) {
            input_window[j] = seq_left;
        } else if (idx >= seq_len && idx < seq_len + AA_UP_REP_PAD) {
            input_window[j] = seq_right;
        }
    }

    float intermediates[2 * AA_FILTER_SIZE + 2 * AA_BUFFER_SIZE
                        + AA_DS_REP_PAD_LEFT + AA_DS_REP_PAD_RIGHT] = {0};

    #pragma unroll
    for (int it = 0; it < (2 * AA_BUFFER_SIZE + 2 * AA_FILTER_SIZE); it++) {
        const int p = it & 1;
        const int base = (it + p) / 2;
        float acc = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < AA_FILTER_SIZE / 2; kk++) {
            acc += up_filter[p + 2 * kk] * input_window[base + kk];
        }
        intermediates[it + AA_DS_REP_PAD_LEFT] = acc;
    }

    #pragma unroll
    for (int it = 0; it < 2 * AA_BUFFER_SIZE + 2 * AA_FILTER_SIZE; it++) {
        const float v = intermediates[it + AA_DS_REP_PAD_LEFT];
        const float s = sin(alpha_val * v);
        intermediates[it + AA_DS_REP_PAD_LEFT] = v + inv_beta * s * s;
    }

    #pragma unroll
    for (int it = 0; it < AA_DS_REP_PAD_LEFT; it++) {
        intermediates[it] = intermediates[AA_DS_REP_PAD_LEFT];
    }
    const int tail_first_p = AA_DS_REP_PAD_LEFT + 2 * AA_BUFFER_SIZE + 2 * AA_FILTER_SIZE;
    #pragma unroll
    for (int it = 0; it < AA_DS_REP_PAD_RIGHT; it++) {
        intermediates[tail_first_p + it] = intermediates[tail_first_p - 1];
    }

    float output_p[AA_BUFFER_SIZE];
    #pragma unroll
    for (int it = 0; it < AA_BUFFER_SIZE; it++) {
        float acc = 0.0f;
        #pragma unroll
        for (int f = 0; f < AA_FILTER_SIZE; f++) {
            acc += down_filter[f] * intermediates[it * 2 + f + AA_DS_REP_PAD_RIGHT];
        }
        output_p[it] = acc;
    }

    #pragma unroll
    for (int it = 0; it < AA_BUFFER_SIZE; it++) {
        const int abs_idx = seq_offset + it;
        if (abs_idx < seq_len) {
            dst_ch[abs_idx] = (T) output_p[it];
        }
    }
}

template [[host_name("kernel_aa_snake_beta_polyphase_f32")]]
kernel aa_snake_beta_t kernel_aa_snake_beta_polyphase_impl<float>;


// Variant: threadgroup-shared input tile (cooperative load). Reduces
// device-memory bandwidth to one chunk-wide read per threadgroup, at
// the cost of a ~16 KB threadgroup allocation. On M1 the shared
// allocation regressed th_max from 832 to 640 (the SM's register
// budget competes with shared mem on Apple's tiled architecture) and
// wall-clock by ~30 %.
constant constexpr int AA_TILE_PAD_LEFT  = AA_HALF_FILTER_SIZE;
constant constexpr int AA_TILE_PAD_RIGHT = AA_HALF_FILTER_SIZE + AA_FILTER_SIZE / 2;
constant constexpr int AA_TILE_TOTAL     = AA_THREADS_PER_TGRP * AA_BUFFER_SIZE
                                           + AA_TILE_PAD_LEFT + AA_TILE_PAD_RIGHT + 16;
constant constexpr int AA_TILE_LOAD_ITERS = (AA_TILE_TOTAL + AA_THREADS_PER_TGRP - 1)
                                           / AA_THREADS_PER_TGRP;

template <typename T>
kernel void kernel_aa_snake_beta_tgmem_impl(
        constant ggml_metal_kargs_aa_snake_beta & args      [[buffer(0)]],
        device const T                          * src       [[buffer(1)]],
        device const T                          * log_alpha [[buffer(2)]],
        device const T                          * log_beta  [[buffer(3)]],
        device const T                          * up_ftr    [[buffer(4)]],
        device const T                          * down_ftr  [[buffer(5)]],
        device T                                * dst       [[buffer(6)]],
        uint3                                     tgpig     [[threadgroup_position_in_grid]],
        uint3                                     tpitg     [[thread_position_in_threadgroup]]) {

    const int seq_blk = (int) tgpig.x;
    const int chan    = (int) tgpig.y;
    const int tid     = (int) tpitg.x;
    const int seq_len = args.T;

    const int chunk_start = seq_blk * AA_THREADS_PER_TGRP * AA_BUFFER_SIZE;
    const int local_start = tid * AA_BUFFER_SIZE;
    const int seq_offset  = chunk_start + local_start;

    const int channel_offset = chan * seq_len;
    device const T * src_ch  = src + channel_offset;
    device       T * dst_ch  = dst + channel_offset;

    threadgroup float input_tile[AA_TILE_TOTAL];

    {
        const float seq_left  = (float) src_ch[0];
        const float seq_right = (float) src_ch[seq_len - 1];
        #pragma unroll
        for (int it = 0; it < AA_TILE_LOAD_ITERS; it++) {
            const int local_idx = it * AA_THREADS_PER_TGRP + tid;
            if (local_idx < AA_TILE_TOTAL) {
                const int abs_idx = chunk_start + local_idx - AA_TILE_PAD_LEFT;
                float val;
                if (abs_idx >= 0 && abs_idx < seq_len) {
                    val = (float) src_ch[abs_idx];
                } else if (abs_idx < 0 && abs_idx >= -AA_UP_REP_PAD) {
                    val = seq_left;
                } else if (abs_idx >= seq_len && abs_idx < seq_len + AA_UP_REP_PAD) {
                    val = seq_right;
                } else {
                    val = 0.0f;
                }
                input_tile[local_idx] = val;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float alpha_val = exp((float) log_alpha[chan]);
    const float beta_val  = exp((float) log_beta[chan]);
    const float inv_beta  = 1.0f / (beta_val + 1e-9f);

    float up_filter[AA_FILTER_SIZE];
    float down_filter[AA_FILTER_SIZE];
    #pragma unroll
    for (int k = 0; k < AA_FILTER_SIZE; k++) {
        up_filter[k]   = 2.0f * (float) up_ftr[k];
        down_filter[k] = (float) down_ftr[k];
    }

    float intermediates[2 * AA_FILTER_SIZE + 2 * AA_BUFFER_SIZE
                        + AA_DS_REP_PAD_LEFT + AA_DS_REP_PAD_RIGHT] = {0};

    // AA_TILE_PAD_LEFT == AA_HALF_FILTER_SIZE so the (-HALF, +TILE_PAD)
    // offsets cancel and tile_base reduces to local_start.
    #pragma unroll
    for (int it = 0; it < (2 * AA_BUFFER_SIZE + 2 * AA_FILTER_SIZE); it++) {
        const int p = it & 1;
        const int base = local_start + (it + p) / 2;
        float acc = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < AA_FILTER_SIZE / 2; kk++) {
            acc += up_filter[p + 2 * kk] * input_tile[base + kk];
        }
        intermediates[it + AA_DS_REP_PAD_LEFT] = acc;
    }

    #pragma unroll
    for (int it = 0; it < 2 * AA_BUFFER_SIZE + 2 * AA_FILTER_SIZE; it++) {
        const float v = intermediates[it + AA_DS_REP_PAD_LEFT];
        const float s = sin(alpha_val * v);
        intermediates[it + AA_DS_REP_PAD_LEFT] = v + inv_beta * s * s;
    }

    #pragma unroll
    for (int it = 0; it < AA_DS_REP_PAD_LEFT; it++) {
        intermediates[it] = intermediates[AA_DS_REP_PAD_LEFT];
    }
    const int tail_first_t = AA_DS_REP_PAD_LEFT + 2 * AA_BUFFER_SIZE + 2 * AA_FILTER_SIZE;
    #pragma unroll
    for (int it = 0; it < AA_DS_REP_PAD_RIGHT; it++) {
        intermediates[tail_first_t + it] = intermediates[tail_first_t - 1];
    }

    float output_t[AA_BUFFER_SIZE];
    #pragma unroll
    for (int it = 0; it < AA_BUFFER_SIZE; it++) {
        float acc = 0.0f;
        #pragma unroll
        for (int f = 0; f < AA_FILTER_SIZE; f++) {
            acc += down_filter[f] * intermediates[it * 2 + f + AA_DS_REP_PAD_RIGHT];
        }
        output_t[it] = acc;
    }

    #pragma unroll
    for (int it = 0; it < AA_BUFFER_SIZE; it++) {
        const int abs_idx = seq_offset + it;
        if (abs_idx < seq_len) {
            dst_ch[abs_idx] = (T) output_t[it];
        }
    }
}

template [[host_name("kernel_aa_snake_beta_tgmem_f32")]]
kernel aa_snake_beta_t kernel_aa_snake_beta_tgmem_impl<float>;
