// piano_transcription.cpp — Piano transcription backend
// Port of ByteDance/Kong piano_transcription_inference (Apache-2.0)
//
// Architecture: Regress_onset_offset_frame_velocity_CRNN
//   Input: 16 kHz mono → STFT(n_fft=2048, hop=160) → LogMel(229 bins) → BN
//   4× AcousticModelCRnn8Dropout (frame/onset/offset/velocity):
//     4× ConvBlock(Conv2d 3×3 + BN2d + ReLU, AvgPool2d(1,2))
//     FC(1792→768) + BN1d + ReLU → BiGRU(2 layers, 256 hidden) → FC(512→88) → sigmoid
//   Onset refinement: cat(onset, sqrt(onset)*velocity) → BiGRU(1 layer, 256) → FC → sigmoid
//   Frame refinement: cat(frame, onset, offset) → BiGRU(1 layer, 256) → FC → sigmoid
//   Post-processing: regression binarization → note detection → MIDI events
//
// The Conv/FC pipeline uses a ggml computation graph. The BiGRU is computed
// directly in C++ because unrolling 1000+ timesteps in a ggml graph would
// create 100k+ nodes and exhaust graph capacity.

#include "piano_transcription.h"

#include "core/crispasr_env.h"
#include "core/fft.h"
#include "core/gguf_loader.h"
#include "core/mel.h"
#include "core/parallel_for.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include "core/ggml_cpu_backend.h"

// ─── Hyperparameters ────────────────────────────────────────────────────────

struct piano_hparams {
    uint32_t sample_rate = 16000;
    uint32_t n_fft = 2048;
    uint32_t hop_size = 160;
    uint32_t n_mels = 229;
    uint32_t fmin = 30;
    uint32_t fmax = 8000;
    uint32_t frames_per_second = 100;
    uint32_t classes_num = 88;
    uint32_t begin_note = 21; // MIDI note of A0

    // Conv block channels: [48, 64, 96, 128]
    uint32_t conv_channels[4] = {48, 64, 96, 128};
    uint32_t gru_hidden = 256;
    uint32_t fc5_out = 768;
    uint32_t midfeat = 1792; // 128 * 14

    static constexpr float velocity_scale = 128.0f;
};

// ─── Weight structures ──────────────────────────────────────────────────────

struct piano_bn_weights {
    ggml_tensor* weight = nullptr; // gamma
    ggml_tensor* bias = nullptr;   // beta
    ggml_tensor* running_mean = nullptr;
    ggml_tensor* running_var = nullptr;
};

struct piano_conv_block {
    ggml_tensor* conv1_w = nullptr; // [OC, IC, 3, 3]
    piano_bn_weights bn1;
    ggml_tensor* conv2_w = nullptr; // [OC, OC, 3, 3]
    piano_bn_weights bn2;
};

struct piano_gru_layer {
    // weight_ih: [3*hidden, input_size], weight_hh: [3*hidden, hidden]
    // bias_ih, bias_hh: [3*hidden]
    ggml_tensor* weight_ih = nullptr;
    ggml_tensor* weight_hh = nullptr;
    ggml_tensor* bias_ih = nullptr;
    ggml_tensor* bias_hh = nullptr;
    // Reverse direction
    ggml_tensor* weight_ih_r = nullptr;
    ggml_tensor* weight_hh_r = nullptr;
    ggml_tensor* bias_ih_r = nullptr;
    ggml_tensor* bias_hh_r = nullptr;
};

struct piano_acoustic_model {
    piano_conv_block conv_blocks[4];
    ggml_tensor* fc5_w = nullptr; // [768, 1792]
    piano_bn_weights bn5;
    piano_gru_layer gru[2];      // 2-layer BiGRU
    ggml_tensor* fc_w = nullptr; // [88, 512]
    ggml_tensor* fc_b = nullptr; // [88]
};

struct piano_weights {
    // Note model
    piano_bn_weights note_bn0;
    ggml_tensor* mel_w = nullptr; // [1025, 229]
    piano_acoustic_model frame;
    piano_acoustic_model onset;
    piano_acoustic_model offset;
    piano_acoustic_model velocity;

    // Onset refinement
    piano_gru_layer onset_refine_gru;         // 1-layer BiGRU, input=176
    ggml_tensor* onset_refine_fc_w = nullptr; // [88, 512]
    ggml_tensor* onset_refine_fc_b = nullptr; // [88]

    // Frame refinement
    piano_gru_layer frame_refine_gru;         // 1-layer BiGRU, input=264
    ggml_tensor* frame_refine_fc_w = nullptr; // [88, 512]
    ggml_tensor* frame_refine_fc_b = nullptr; // [88]

    // Pedal model
    piano_bn_weights pedal_bn0;
    piano_acoustic_model pedal_onset;
    piano_acoustic_model pedal_offset;
    piano_acoustic_model pedal_frame;
};

// ─── Context ────────────────────────────────────────────────────────────────

struct piano_transcription_ctx {
    piano_hparams hp;
    piano_weights weights;
    piano_transcription_params params;

    // GGUF-loaded weight context
    ggml_context* w_ctx = nullptr;
    ggml_backend_buffer_t w_buf = nullptr;
    ggml_backend_t backend = nullptr;

    // Mel filterbank (computed once at init or loaded from GGUF)
    std::vector<float> mel_fb; // [n_mels * n_freqs]
    std::vector<float> hann;   // [n_fft] window
};

// ─── FFT callback for core_mel ──────────────────────────────────────────────

static void piano_fft_r2c(const float* in, int N, float* out) {
    std::vector<float> re(N), im(N, 0.0f);
    std::memcpy(re.data(), in, N * sizeof(float));
    core_fft::fft_radix2_inplace(re.data(), im.data(), N);
    for (int i = 0; i < N; i++) {
        out[2 * i + 0] = re[i];
        out[2 * i + 1] = im[i];
    }
}

// ─── Helper: read tensor data as float ──────────────────────────────────────

static const float* tensor_f32_data(ggml_tensor* t) {
    assert(t->type == GGML_TYPE_F32);
    return (const float*)t->data;
}

static std::vector<float> tensor_to_f32(ggml_tensor* t) {
    int64_t n = ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t->data, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t* src = (const ggml_fp16_t*)t->data;
        for (int64_t i = 0; i < n; i++) {
            out[i] = ggml_fp16_to_fp32(src[i]);
        }
    } else {
        fprintf(stderr, "piano: unsupported tensor type %d\n", t->type);
    }
    return out;
}

// ─── BatchNorm inference (fused: scale = weight/sqrt(var+eps), shift = bias - mean*scale) ──

struct bn_fused {
    std::vector<float> scale;
    std::vector<float> shift;
};

// BatchNorm epsilon — NOT the PyTorch default for this model.
//
// Upstream builds every 2-D BN as `nn.BatchNorm2d(out_channels, momentum)`
// (models.py:74-75, 186). BatchNorm2d's SECOND POSITIONAL parameter is `eps`,
// not `momentum`, so that intended-momentum 0.01 actually lands on eps and
// momentum keeps its 0.1 default. bn5 is built as
// `nn.BatchNorm1d(768, momentum=momentum)` with an explicit keyword, so it
// keeps eps = 1e-5. Confirmed on the loaded checkpoint: 33 BatchNorm2d at
// eps=0.01, 4 BatchNorm1d at eps=1e-5.
//
// It is an upstream slip, but the weights were TRAINED with it, so it must be
// reproduced exactly — the same reasoning as BTC's trailing ReLU. It matters a
// lot here because the running variances are tiny (~0.003), so eps dominates
// the denominator: sqrt(0.00295 + 1e-5) = 0.0544 vs sqrt(0.00295 + 0.01) =
// 0.1138, a 2.09x error per layer. Using 1e-5 inflated conv-block-1 output by
// 2.44x and dragged the final onset head to cos 0.72 against the reference.
static constexpr float PIANO_BN2D_EPS = 0.01f; // ALL BatchNorm2d (bn0 + conv blocks)
static constexpr float PIANO_BN1D_EPS = 1e-5f; // BatchNorm1d (bn5) only

// #305: everything below the mel front-end (which core_mel already threads) ran
// on one core — the 3x3 convs, the dense layers and the two GRU directions.
// Each is split along an axis whose iterations write disjoint outputs and only
// READ shared inputs, and the innermost accumulation order is untouched, so the
// result is bit-identical to the serial path. Opt out with
// CRISPASR_PIANO_SERIAL=1.
//
// Deliberately NOT parallelized: the four acoustic_model_forward calls in
// piano_transcription_transcribe. Running those concurrently would hold four
// sets of conv activations at once, and the box this ships on has 8 GB.
static int piano_nthreads() {
    static int v = -1;
    if (v < 0) {
        if (crispasr_env::get("CRISPASR_PIANO_SERIAL") != nullptr) {
            v = 1;
        } else {
            unsigned hw = std::thread::hardware_concurrency();
            v = (hw == 0) ? 1 : (int)std::min(hw, 8u);
        }
    }
    return v;
}

static bn_fused fuse_bn(const piano_bn_weights& bn, float eps) {
    int C = (int)ggml_nelements(bn.weight);
    auto w = tensor_to_f32(bn.weight);
    auto b = tensor_to_f32(bn.bias);
    auto rm = tensor_to_f32(bn.running_mean);
    auto rv = tensor_to_f32(bn.running_var);
    bn_fused out;
    out.scale.resize(C);
    out.shift.resize(C);
    for (int i = 0; i < C; i++) {
        float s = w[i] / std::sqrt(rv[i] + eps);
        out.scale[i] = s;
        out.shift[i] = b[i] - rm[i] * s;
    }
    return out;
}

// Apply BN2d on [C, H, W] data (channel-first, H=T, W=freq).
// In memory: [W, H, C] (ggml layout).
// BN2d normalizes over (N, H, W) per channel.
static void apply_bn2d(float* data, int W, int H, int C, const bn_fused& bn) {
    // data layout: data[c * H * W + h * W + w] in channel-first,
    // but in ggml memory it's data[w + h*W + c*H*W] = same thing!
    for (int c = 0; c < C; c++) {
        float s = bn.scale[c];
        float b = bn.shift[c];
        for (int hw = 0; hw < H * W; hw++) {
            data[c * H * W + hw] = data[c * H * W + hw] * s + b;
        }
    }
}

// Apply BN1d on [C, T] data.
static void apply_bn1d(float* data, int T, int C, const bn_fused& bn) {
    for (int c = 0; c < C; c++) {
        float s = bn.scale[c];
        float b = bn.shift[c];
        for (int t = 0; t < T; t++) {
            data[c * T + t] = data[c * T + t] * s + b;
        }
    }
}

// ─── Conv2d (manual im2col + GEMM, since ggml_conv_2d needs graph) ──────────

// Conv2d with kernel 3x3, stride 1x1, padding 1x1, no bias
// Input: [IC, H, W] in channel-first order (= [W, H, IC] in ggml mem)
// Weight: [OC, IC, 3, 3]
// Output: [OC, H, W]
static std::vector<float> conv2d_3x3_pad1(const float* input, int IC, int H, int W, const float* weight, int OC) {
    std::vector<float> output(OC * H * W, 0.0f);

    // Split over oc: each output channel owns a disjoint [H*W] slab of `output`
    // and only reads `input`/`weight`. The ic/kh/kw accumulation below is
    // untouched, so every `sum` is summed in the same order as the serial loop.
    core_parallel::for_each_chunk(OC, piano_nthreads(), [&](int oc0, int oc1) {
        for (int oc = oc0; oc < oc1; oc++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    float sum = 0.0f;
                    for (int ic = 0; ic < IC; ic++) {
                        for (int kh = 0; kh < 3; kh++) {
                            for (int kw = 0; kw < 3; kw++) {
                                int ih = h + kh - 1;
                                int iw = w + kw - 1;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    // Weight layout: [OC, IC, KH, KW]
                                    // Input layout: [IC, H, W]
                                    sum += weight[oc * IC * 9 + ic * 9 + kh * 3 + kw] * input[ic * H * W + ih * W + iw];
                                }
                            }
                        }
                    }
                    output[oc * H * W + h * W + w] = sum;
                }
            }
        }
    });

    return output;
}

// AvgPool2d with kernel (1, 2) — only pools the width (freq) axis
static std::vector<float> avg_pool2d_1x2(const float* input, int C, int H, int W) {
    int W_out = W / 2;
    std::vector<float> output(C * H * W_out);
    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W_out; w++) {
                output[c * H * W_out + h * W_out + w] =
                    0.5f * (input[c * H * W + h * W + 2 * w] + input[c * H * W + h * W + 2 * w + 1]);
            }
        }
    }
    return output;
}

// ReLU in-place
static void relu_inplace(float* data, int n) {
    for (int i = 0; i < n; i++) {
        if (data[i] < 0.0f)
            data[i] = 0.0f;
    }
}

// Sigmoid in-place
static void sigmoid_inplace(float* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = 1.0f / (1.0f + std::exp(-data[i]));
    }
}

// ─── GRU computation ────────────────────────────────────────────────────────

// Single-direction GRU forward pass.
// x: [input_size, T] column-major (column t = x[t * input_size .. (t+1)*input_size])
// Returns: [hidden_size, T]
static std::vector<float> gru_forward(const float* x, int input_size, int T, int hidden_size, const float* W_ih,
                                      const float* W_hh, const float* b_ih, const float* b_hh, bool reverse) {
    // GRU gate size = 3 * hidden_size
    int gate_size = 3 * hidden_size;

    std::vector<float> h(hidden_size, 0.0f); // hidden state
    std::vector<float> output(hidden_size * T);
    std::vector<float> gates_ih(gate_size);
    std::vector<float> gates_hh(gate_size);

    for (int step = 0; step < T; step++) {
        int t = reverse ? (T - 1 - step) : step;
        const float* x_t = x + t * input_size;

        // gates_ih = W_ih @ x_t + b_ih
        for (int g = 0; g < gate_size; g++) {
            float sum = b_ih[g];
            for (int i = 0; i < input_size; i++) {
                sum += W_ih[g * input_size + i] * x_t[i];
            }
            gates_ih[g] = sum;
        }

        // gates_hh = W_hh @ h + b_hh
        for (int g = 0; g < gate_size; g++) {
            float sum = b_hh[g];
            for (int i = 0; i < hidden_size; i++) {
                sum += W_hh[g * hidden_size + i] * h[i];
            }
            gates_hh[g] = sum;
        }

        // PyTorch GRU gate order: [r, z, n] (reset, update, new)
        // r = sigmoid(gates_ih[0:H] + gates_hh[0:H])        (reset gate)
        // z = sigmoid(gates_ih[H:2H] + gates_hh[H:2H])      (update gate)
        // n = tanh(gates_ih[2H:3H] + r * gates_hh[2H:3H])   (new gate)
        // h = (1 - z) * n + z * h_prev
        int H = hidden_size;
        for (int i = 0; i < H; i++) {
            float r = 1.0f / (1.0f + std::exp(-(gates_ih[i] + gates_hh[i])));
            float z = 1.0f / (1.0f + std::exp(-(gates_ih[H + i] + gates_hh[H + i])));
            float n = std::tanh(gates_ih[2 * H + i] + r * gates_hh[2 * H + i]);
            h[i] = (1.0f - z) * n + z * h[i];
        }

        // Store h at time t
        std::memcpy(output.data() + t * hidden_size, h.data(), hidden_size * sizeof(float));
    }

    return output; // [T, hidden_size] row-major (each row = h_t)
}

// Bidirectional GRU — concatenates forward and reverse hidden states.
// x: [T, input_size] row-major
// Returns: [T, 2*hidden_size] row-major
static std::vector<float> bigru_forward(const float* x, int T, int input_size, int hidden_size,
                                        const piano_gru_layer& layer) {
    // Extract weight data as F32
    auto wih = tensor_to_f32(layer.weight_ih);
    auto whh = tensor_to_f32(layer.weight_hh);
    auto bih = tensor_to_f32(layer.bias_ih);
    auto bhh = tensor_to_f32(layer.bias_hh);
    auto wih_r = tensor_to_f32(layer.weight_ih_r);
    auto whh_r = tensor_to_f32(layer.weight_hh_r);
    auto bih_r = tensor_to_f32(layer.bias_ih_r);
    auto bhh_r = tensor_to_f32(layer.bias_hh_r);

    // Transpose x from [T, input_size] to [input_size, T] for gru_forward
    // (gru_forward expects column-major: x_t at offset t * input_size)
    // Actually our gru_forward reads x + t * input_size — so [T, input_size] row-major
    // is exactly right as long as x_t starts at x + t * input_size. ✓

    // The two directions share no state — each recurrence carries its own `h`
    // and writes its own output vector — so they run concurrently. Two tasks
    // only; with piano_nthreads() == 1 for_each_task runs them in the original
    // forward-then-reverse order.
    std::vector<float> h_fwd, h_rev;
    core_parallel::for_each_task(2, piano_nthreads(), [&](int dir, int) {
        if (dir == 0) {
            h_fwd = gru_forward(x, input_size, T, hidden_size, wih.data(), whh.data(), bih.data(), bhh.data(), false);
        } else {
            h_rev = gru_forward(x, input_size, T, hidden_size, wih_r.data(), whh_r.data(), bih_r.data(), bhh_r.data(),
                                true);
        }
    });

    // Concatenate: [T, 2*hidden_size]
    std::vector<float> out(T * 2 * hidden_size);
    for (int t = 0; t < T; t++) {
        std::memcpy(out.data() + t * 2 * hidden_size, h_fwd.data() + t * hidden_size, hidden_size * sizeof(float));
        std::memcpy(out.data() + t * 2 * hidden_size + hidden_size, h_rev.data() + t * hidden_size,
                    hidden_size * sizeof(float));
    }
    return out;
}

// Multi-layer BiGRU: 2 layers, each bidirectional
// x: [T, input_size] → [T, 2*hidden_size] after 2 layers
static std::vector<float> bigru_2layer(const float* x, int T, int input_size, int hidden_size,
                                       const piano_gru_layer layers[2]) {
    auto h = bigru_forward(x, T, input_size, hidden_size, layers[0]);
    // Layer 1 input_size = 2 * hidden_size
    h = bigru_forward(h.data(), T, 2 * hidden_size, hidden_size, layers[1]);
    return h; // [T, 2*hidden_size]
}

// FC (linear): y = x @ W^T + b
// x: [T, in_feat], W: [out_feat, in_feat], b: [out_feat]
// output: [T, out_feat]
static std::vector<float> linear(const float* x, int T, int in_feat, const float* W, const float* b, int out_feat) {
    std::vector<float> y(T * out_feat);
    // Split over t: each row owns a disjoint [out_feat] slab of `y`, and the
    // dot product over `in_feat` keeps its serial order.
    core_parallel::for_each_chunk(T, piano_nthreads(), [&](int t0, int t1) {
        for (int t = t0; t < t1; t++) {
            for (int o = 0; o < out_feat; o++) {
                float sum = b ? b[o] : 0.0f;
                for (int i = 0; i < in_feat; i++) {
                    sum += W[o * in_feat + i] * x[t * in_feat + i];
                }
                y[t * out_feat + o] = sum;
            }
        }
    });
    return y;
}

// ─── Acoustic model forward ─────────────────────────────────────────────────

// Run one AcousticModelCRnn8Dropout in inference mode (no dropout).
// Input: [T, 229] mel spectrogram after BN0.
// Output: [T, 88] sigmoid-activated.
// Optional intermediate taps for the diff harness. All nullptr in normal use,
// so the inference path is unchanged. Named for the reference dumper's stage
// names (tools/reference_backends/piano_transcription.py) so the mapping is
// obvious at the call site.
struct piano_acoustic_taps {
    std::vector<float>* conv_block_output = nullptr; // [C=128, T, W=14], channel-first
    std::vector<float>* fc5_output = nullptr;        // [T, 768] post BN1d+ReLU
    std::vector<float>* gru_output = nullptr;        // [T, 512] BiGRU output
};

static std::vector<float> acoustic_model_forward(const float* mel_bn, int T, int n_mels,
                                                 const piano_acoustic_model& model, int hidden_size,
                                                 piano_acoustic_taps* taps = nullptr) {
    // mel_bn: [T, n_mels] row-major

    // Convert to [1, T, n_mels] → treat as [C=1, H=T, W=n_mels]
    // (channel-first: data[c*H*W + h*W + w])
    int C = 1, H = T, W = n_mels;
    std::vector<float> x(C * H * W);
    std::memcpy(x.data(), mel_bn, C * H * W * sizeof(float));

    // 4 ConvBlocks
    int conv_channels[5] = {1, 48, 64, 96, 128};
    for (int blk = 0; blk < 4; blk++) {
        int IC = conv_channels[blk];
        int OC = conv_channels[blk + 1];

        auto conv1_w = tensor_to_f32(model.conv_blocks[blk].conv1_w);
        auto conv2_w = tensor_to_f32(model.conv_blocks[blk].conv2_w);
        auto bn1 = fuse_bn(model.conv_blocks[blk].bn1, PIANO_BN2D_EPS);
        auto bn2 = fuse_bn(model.conv_blocks[blk].bn2, PIANO_BN2D_EPS);

        // Conv1: [IC, H, W] → [OC, H, W]
        auto h = conv2d_3x3_pad1(x.data(), IC, H, W, conv1_w.data(), OC);
        apply_bn2d(h.data(), W, H, OC, bn1);
        relu_inplace(h.data(), OC * H * W);

        // Conv2: [OC, H, W] → [OC, H, W]
        auto h2 = conv2d_3x3_pad1(h.data(), OC, H, W, conv2_w.data(), OC);
        apply_bn2d(h2.data(), W, H, OC, bn2);
        relu_inplace(h2.data(), OC * H * W);

        // AvgPool2d(1, 2) — pools only the W (frequency) axis
        x = avg_pool2d_1x2(h2.data(), OC, H, W);
        C = OC;
        W = W / 2;
    }

    if (taps && taps->conv_block_output)
        *taps->conv_block_output = x; // [C, T, W] channel-first, as the reference dumps it

    // After 4 ConvBlocks: x is [C=128, H=T, W=n_mels/16]
    // W should be 229/16 = 14 (229 → 114 → 57 → 28 → 14)
    int W_final = W;

    // Transpose and flatten: [C=128, H=T, W=14] → [T, 128*14=1792]
    // In channel-first: x[c, h, w] = x[c*H*W_final + h*W_final + w]
    // We need row-major [T, 1792] where row t = x[:, t, :].flatten()
    int midfeat = C * W_final;
    std::vector<float> flat(T * midfeat);
    for (int t = 0; t < T; t++) {
        for (int c = 0; c < C; c++) {
            for (int w = 0; w < W_final; w++) {
                flat[t * midfeat + c * W_final + w] = x[c * H * W_final + t * W_final + w];
            }
        }
    }

    // FC5: [T, 1792] → [T, 768]
    auto fc5_w = tensor_to_f32(model.fc5_w);
    auto fc5_out = linear(flat.data(), T, midfeat, fc5_w.data(), nullptr, 768);

    // BN1d: normalize over T per channel (768 channels)
    // fc5_out is [T, 768]. BN1d expects [N, C, T] = [1, 768, T].
    // Transpose to [768, T], apply BN, transpose back.
    auto bn5 = fuse_bn(model.bn5, PIANO_BN1D_EPS); // BatchNorm1d — keeps the 1e-5 default
    std::vector<float> transposed(768 * T);
    for (int t = 0; t < T; t++) {
        for (int c = 0; c < 768; c++) {
            transposed[c * T + t] = fc5_out[t * 768 + c];
        }
    }
    apply_bn1d(transposed.data(), T, 768, bn5);
    // Transpose back to [T, 768] and apply ReLU
    for (int t = 0; t < T; t++) {
        for (int c = 0; c < 768; c++) {
            float v = transposed[c * T + t];
            fc5_out[t * 768 + c] = v > 0.0f ? v : 0.0f;
        }
    }

    // BiGRU (2 layers): [T, 768] → [T, 512]
    if (taps && taps->fc5_output)
        *taps->fc5_output = fc5_out; // [T, 768] after BN1d + ReLU

    auto gru_out = bigru_2layer(fc5_out.data(), T, 768, hidden_size, model.gru);

    if (taps && taps->gru_output)
        *taps->gru_output = gru_out; // [T, 512]

    // FC: [T, 512] → [T, 88] + sigmoid
    auto fc_w = tensor_to_f32(model.fc_w);
    auto fc_b = tensor_to_f32(model.fc_b);
    auto output = linear(gru_out.data(), T, 2 * hidden_size, fc_w.data(), fc_b.data(), 88);
    sigmoid_inplace(output.data(), T * 88);

    return output; // [T, 88]
}

// ─── Post-processing ────────────────────────────────────────────────────────

static bool is_monotonic_neighbour(const float* x, int n, int neighbour, int len) {
    for (int i = 0; i < neighbour; i++) {
        if (n - i - 1 < 0 || n + i + 1 >= len)
            return false;
        if (x[n - i] < x[n - i - 1])
            return false;
        if (x[n + i] < x[n + i + 1])
            return false;
    }
    return true;
}

// Convert regression output to binary onset/offset indicators.
// reg_output: [T, 88], threshold, neighbour
// binary_output, shift_output: [T, 88]
static void binarize_regression(const float* reg_output, int T, int classes_num, float threshold, int neighbour,
                                std::vector<float>& binary_output, std::vector<float>& shift_output) {
    binary_output.assign(T * classes_num, 0.0f);
    shift_output.assign(T * classes_num, 0.0f);

    for (int k = 0; k < classes_num; k++) {
        for (int n = neighbour; n < T - neighbour; n++) {
            float val = reg_output[n * classes_num + k];
            if (val > threshold) {
                // Check monotonicity — extract column k
                // We need x[t] = reg_output[t * classes_num + k]
                // is_monotonic_neighbour works on a contiguous array,
                // so extract the column first.
                bool mono = true;
                for (int i = 0; i < neighbour && mono; i++) {
                    float cur = reg_output[(n - i) * classes_num + k];
                    float prev = reg_output[(n - i - 1) * classes_num + k];
                    if (cur < prev)
                        mono = false;
                    float next = reg_output[(n + i + 1) * classes_num + k];
                    float now = reg_output[(n + i) * classes_num + k];
                    if (now < next)
                        mono = false;
                }
                if (mono) {
                    binary_output[n * classes_num + k] = 1.0f;
                    float x_prev = reg_output[(n - 1) * classes_num + k];
                    float x_next = reg_output[(n + 1) * classes_num + k];
                    float x_n = val;
                    float shift;
                    if (x_prev > x_next) {
                        shift = (x_next - x_prev) / (x_n - x_next) / 2.0f;
                    } else {
                        shift = (x_next - x_prev) / (x_n - x_prev) / 2.0f;
                    }
                    shift_output[n * classes_num + k] = shift;
                }
            }
        }
    }
}

struct note_tuple {
    int bgn;
    int fin;
    float onset_shift;
    float offset_shift;
    float velocity;
};

static std::vector<note_tuple> note_detection(const float* frame_output,    // [T, 88] — per-class column
                                              const float* onset_output,    // [T, 88] binary
                                              const float* onset_shift,     // [T, 88]
                                              const float* offset_output,   // [T, 88] binary
                                              const float* offset_shift,    // [T, 88]
                                              const float* velocity_output, // [T, 88]
                                              int T, int classes_num, float frame_threshold) {
    std::vector<note_tuple> all_tuples;

    for (int k = 0; k < classes_num; k++) {
        int bgn = -1;
        int frame_disappear = -1;
        int offset_occur = -1;

        for (int i = 0; i < T; i++) {
            if (onset_output[i * classes_num + k] == 1.0f) {
                if (bgn >= 0) {
                    int fin = std::max(i - 1, 0);
                    all_tuples.push_back(
                        {bgn, fin, onset_shift[bgn * classes_num + k], 0.0f, velocity_output[bgn * classes_num + k]});
                    frame_disappear = -1;
                    offset_occur = -1;
                }
                bgn = i;
            }

            if (bgn >= 0 && i > bgn) {
                if (frame_output[i * classes_num + k] <= frame_threshold && frame_disappear < 0) {
                    frame_disappear = i;
                }
                if (offset_output[i * classes_num + k] == 1.0f && offset_occur < 0) {
                    offset_occur = i;
                }
                if (frame_disappear >= 0) {
                    int fin;
                    if (offset_occur >= 0 && offset_occur - bgn > frame_disappear - offset_occur) {
                        fin = offset_occur;
                    } else {
                        fin = frame_disappear;
                    }
                    all_tuples.push_back({bgn, fin, onset_shift[bgn * classes_num + k],
                                          offset_shift[fin * classes_num + k], velocity_output[bgn * classes_num + k]});
                    bgn = -1;
                    frame_disappear = -1;
                    offset_occur = -1;
                }
                if (bgn >= 0 && (i - bgn >= 600 || i == T - 1)) {
                    int fin = i;
                    all_tuples.push_back({bgn, fin, onset_shift[bgn * classes_num + k],
                                          offset_shift[fin * classes_num + k], velocity_output[bgn * classes_num + k]});
                    bgn = -1;
                    frame_disappear = -1;
                    offset_occur = -1;
                }
            }
        }

        // Tag each tuple with the MIDI note
        // (encoded in velocity field temporarily — we'll extract it later)
    }

    return all_tuples;
}

// ─── Weight loading ─────────────────────────────────────────────────────────

static void load_bn(core_gguf::tensor_map& tensors, const std::string& prefix, piano_bn_weights& bn) {
    bn.weight = core_gguf::try_get(tensors, (prefix + ".weight").c_str());
    bn.bias = core_gguf::try_get(tensors, (prefix + ".bias").c_str());
    bn.running_mean = core_gguf::try_get(tensors, (prefix + ".running_mean").c_str());
    bn.running_var = core_gguf::try_get(tensors, (prefix + ".running_var").c_str());
}

static void load_gru_layer(core_gguf::tensor_map& tensors, const std::string& prefix, int layer_idx,
                           piano_gru_layer& gru) {
    std::string l = std::to_string(layer_idx);
    gru.weight_ih = core_gguf::try_get(tensors, (prefix + ".weight_ih_l" + l).c_str());
    gru.weight_hh = core_gguf::try_get(tensors, (prefix + ".weight_hh_l" + l).c_str());
    gru.bias_ih = core_gguf::try_get(tensors, (prefix + ".bias_ih_l" + l).c_str());
    gru.bias_hh = core_gguf::try_get(tensors, (prefix + ".bias_hh_l" + l).c_str());
    gru.weight_ih_r = core_gguf::try_get(tensors, (prefix + ".weight_ih_l" + l + "_reverse").c_str());
    gru.weight_hh_r = core_gguf::try_get(tensors, (prefix + ".weight_hh_l" + l + "_reverse").c_str());
    gru.bias_ih_r = core_gguf::try_get(tensors, (prefix + ".bias_ih_l" + l + "_reverse").c_str());
    gru.bias_hh_r = core_gguf::try_get(tensors, (prefix + ".bias_hh_l" + l + "_reverse").c_str());
}

static void load_acoustic_model(core_gguf::tensor_map& tensors, const std::string& prefix,
                                piano_acoustic_model& model) {
    for (int i = 0; i < 4; i++) {
        std::string cb = prefix + ".conv_block" + std::to_string(i + 1);
        model.conv_blocks[i].conv1_w = core_gguf::try_get(tensors, (cb + ".conv1.weight").c_str());
        load_bn(tensors, cb + ".bn1", model.conv_blocks[i].bn1);
        model.conv_blocks[i].conv2_w = core_gguf::try_get(tensors, (cb + ".conv2.weight").c_str());
        load_bn(tensors, cb + ".bn2", model.conv_blocks[i].bn2);
    }
    model.fc5_w = core_gguf::try_get(tensors, (prefix + ".fc5.weight").c_str());
    load_bn(tensors, prefix + ".bn5", model.bn5);
    load_gru_layer(tensors, prefix + ".gru", 0, model.gru[0]);
    load_gru_layer(tensors, prefix + ".gru", 1, model.gru[1]);
    model.fc_w = core_gguf::try_get(tensors, (prefix + ".fc.weight").c_str());
    model.fc_b = core_gguf::try_get(tensors, (prefix + ".fc.bias").c_str());
}

// ─── Public API ─────────────────────────────────────────────────────────────

struct piano_transcription_params piano_transcription_default_params(void) {
    return {
        /* n_threads              */ 4,
        /* verbosity              */ 1,
        /* use_gpu                */ false,
        /* onset_threshold        */ 0.3f,
        /* offset_threshold       */ 0.3f,
        /* frame_threshold        */ 0.1f,
        /* pedal_offset_threshold */ 0.2f,
    };
}

struct piano_transcription_ctx* piano_transcription_init_from_file(const char* path,
                                                                   struct piano_transcription_params params) {
    auto* ctx = new piano_transcription_ctx();
    ctx->params = params;

    // Pass 1: metadata
    auto* meta = core_gguf::open_metadata(path);
    if (!meta) {
        delete ctx;
        return nullptr;
    }

    auto& hp = ctx->hp;
    hp.sample_rate = core_gguf::kv_u32(meta, "piano.sample_rate", hp.sample_rate);
    hp.n_fft = core_gguf::kv_u32(meta, "piano.n_fft", hp.n_fft);
    hp.hop_size = core_gguf::kv_u32(meta, "piano.hop_size", hp.hop_size);
    hp.n_mels = core_gguf::kv_u32(meta, "piano.n_mels", hp.n_mels);
    hp.fmin = core_gguf::kv_u32(meta, "piano.fmin", hp.fmin);
    hp.fmax = core_gguf::kv_u32(meta, "piano.fmax", hp.fmax);
    hp.frames_per_second = core_gguf::kv_u32(meta, "piano.frames_per_second", hp.frames_per_second);
    hp.classes_num = core_gguf::kv_u32(meta, "piano.classes_num", hp.classes_num);
    hp.begin_note = core_gguf::kv_u32(meta, "piano.begin_note", hp.begin_note);
    hp.gru_hidden = core_gguf::kv_u32(meta, "piano.gru_hidden", hp.gru_hidden);
    hp.fc5_out = core_gguf::kv_u32(meta, "piano.fc5_out", hp.fc5_out);
    hp.midfeat = core_gguf::kv_u32(meta, "piano.midfeat", hp.midfeat);

    core_gguf::free_metadata(meta);

    // Pass 2: load weights
    ctx->backend = core_cpu_backend::init();
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx->backend, "piano", wl)) {
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }
    ctx->w_ctx = wl.ctx;
    ctx->w_buf = wl.buf;

    auto& w = ctx->weights;
    auto& tm = wl.tensors;

    // Note model weights
    load_bn(tm, "piano.note.bn0", w.note_bn0);
    w.mel_w = core_gguf::try_get(tm, "piano.note.logmel.melW");

    load_acoustic_model(tm, "piano.note.frame", w.frame);
    load_acoustic_model(tm, "piano.note.onset", w.onset);
    load_acoustic_model(tm, "piano.note.offset", w.offset);
    load_acoustic_model(tm, "piano.note.velocity", w.velocity);

    // Onset refinement GRU (1 layer)
    load_gru_layer(tm, "piano.note.onset_refine_gru", 0, w.onset_refine_gru);
    w.onset_refine_fc_w = core_gguf::try_get(tm, "piano.note.onset_refine_fc.weight");
    w.onset_refine_fc_b = core_gguf::try_get(tm, "piano.note.onset_refine_fc.bias");

    // Frame refinement GRU (1 layer)
    load_gru_layer(tm, "piano.note.frame_refine_gru", 0, w.frame_refine_gru);
    w.frame_refine_fc_w = core_gguf::try_get(tm, "piano.note.frame_refine_fc.weight");
    w.frame_refine_fc_b = core_gguf::try_get(tm, "piano.note.frame_refine_fc.bias");

    // Pedal model weights
    load_bn(tm, "piano.pedal.bn0", w.pedal_bn0);
    load_acoustic_model(tm, "piano.pedal.onset", w.pedal_onset);
    load_acoustic_model(tm, "piano.pedal.offset", w.pedal_offset);
    load_acoustic_model(tm, "piano.pedal.frame", w.pedal_frame);

    // Build mel filterbank from the loaded melW tensor
    // The checkpoint's melW is [1025, 229] — this is the mel basis transposed
    // relative to core_mel's convention. We'll use it directly.
    if (w.mel_w) {
        auto melW = tensor_to_f32(w.mel_w);
        int n_freqs = hp.n_fft / 2 + 1; // 1025
        // melW is [n_freqs, n_mels] = [1025, 229]
        // core_mel expects [n_mels, n_freqs] for MelsFreqs layout
        ctx->mel_fb.resize(hp.n_mels * n_freqs);
        for (uint32_t m = 0; m < hp.n_mels; m++) {
            for (int f = 0; f < n_freqs; f++) {
                ctx->mel_fb[m * n_freqs + f] = melW[f * hp.n_mels + m];
            }
        }
    } else {
        // Fall back to computing mel filterbank
        ctx->mel_fb = core_mel::build_htk_fb(hp.sample_rate, hp.n_fft, hp.n_mels, (float)hp.fmin, (float)hp.fmax,
                                             core_mel::FbLayout::MelsFreqs);
    }

    // Build Hann window
    ctx->hann.resize(hp.n_fft);
    for (uint32_t i = 0; i < hp.n_fft; i++) {
        ctx->hann[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / hp.n_fft));
    }

    if (params.verbosity >= 1) {
        fprintf(stderr, "piano: loaded model (%u mels, %u classes, %u fps)\n", hp.n_mels, hp.classes_num,
                hp.frames_per_second);
    }

    return ctx;
}

void piano_transcription_free(struct piano_transcription_ctx* ctx) {
    if (!ctx)
        return;
    if (ctx->w_buf)
        core_gguf::release_weight_buffer(ctx->w_buf);
    if (ctx->w_ctx)
        ggml_free(ctx->w_ctx);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

uint32_t piano_transcription_sample_rate(const struct piano_transcription_ctx* ctx) {
    return ctx ? ctx->hp.sample_rate : 16000;
}

// ─── Mel spectrogram ────────────────────────────────────────────────────────

static std::vector<float> compute_mel_with_bn0(piano_transcription_ctx* ctx, const float* pcm, int n_samples,
                                               int& T_out) {
    auto& hp = ctx->hp;
    int n_freqs = hp.n_fft / 2 + 1;

    // Use core_mel::compute for STFT + mel projection + log
    core_mel::Params mel_params;
    mel_params.n_fft = hp.n_fft;
    mel_params.hop_length = hp.hop_size;
    mel_params.win_length = hp.n_fft;
    mel_params.n_mels = hp.n_mels;
    mel_params.log_base = core_mel::LogBase::Log10;
    mel_params.log_guard = core_mel::LogGuard::MaxClip;
    mel_params.log_eps = 1e-10f;
    mel_params.spec_kind = core_mel::SpecKind::Power;
    mel_params.norm = core_mel::Normalization::None;
    mel_params.layout = core_mel::Layout::TimeMels; // [T, n_mels]
    mel_params.fb_layout = core_mel::FbLayout::MelsFreqs;
    mel_params.matmul = core_mel::MatmulPrecision::Float;
    mel_params.center_pad = true;
    mel_params.center_pad_reflect = true; // librosa reflect padding

    auto mel = core_mel::compute(pcm, n_samples, ctx->hann.data(), hp.n_fft, ctx->mel_fb.data(), n_freqs, piano_fft_r2c,
                                 mel_params, T_out);
    // core_mel with Log10 + MaxClip produces log10(max(power, 1e-10)).
    // torchlibrosa's power_to_db uses 10 * log10(max(S, amin)) — the standard
    // dB conversion. Scale by 10 to match.
    for (int i = 0; i < T_out * (int)hp.n_mels; i++) {
        mel[i] *= 10.0f;
    }

    // Apply BN0: the original model transposes to (1, 229, T, 1) for BN2d
    // then back. For our [T, n_mels] layout, BN operates per-mel-bin across T.
    // This is equivalent to BN1d with C=n_mels.
    auto bn0 = fuse_bn(ctx->weights.note_bn0, PIANO_BN2D_EPS);
    // mel is [T, n_mels] row-major. We need to apply scale[m] * x[t,m] + shift[m]
    for (int t = 0; t < T_out; t++) {
        for (uint32_t m = 0; m < hp.n_mels; m++) {
            mel[t * hp.n_mels + m] = mel[t * hp.n_mels + m] * bn0.scale[m] + bn0.shift[m];
        }
    }

    return mel; // [T, n_mels]
}

// ─── Transcribe ─────────────────────────────────────────────────────────────

int piano_transcription_transcribe(struct piano_transcription_ctx* ctx, const float* pcm, int n_samples,
                                   struct piano_transcription_result* result) {
    if (!ctx || !pcm || n_samples <= 0 || !result)
        return -1;
    std::memset(result, 0, sizeof(*result));

    auto& hp = ctx->hp;
    auto& w = ctx->weights;
    auto& p = ctx->params;

    int segment_samples = hp.sample_rate * 10; // 10 seconds

    // Pad to multiple of segment_samples
    int padded_len = ((n_samples + segment_samples - 1) / segment_samples) * segment_samples;
    std::vector<float> audio(padded_len, 0.0f);
    std::memcpy(audio.data(), pcm, n_samples * sizeof(float));

    // Process in 10s segments with 50% overlap
    int hop_samples = segment_samples / 2;
    int n_segments = 0;
    {
        int ptr = 0;
        while (ptr + segment_samples <= padded_len) {
            n_segments++;
            ptr += hop_samples;
        }
    }

    if (p.verbosity >= 1) {
        fprintf(stderr, "piano: %d segments (%.1fs audio)\n", n_segments, (float)n_samples / hp.sample_rate);
    }

    // Process each segment
    struct segment_outputs {
        std::vector<float> frame; // [T_seg, 88]
        std::vector<float> onset;
        std::vector<float> offset;
        std::vector<float> velocity;
        int T_seg;
    };
    std::vector<segment_outputs> seg_outputs(n_segments);

    int ptr = 0;
    for (int seg = 0; seg < n_segments; seg++) {
        if (p.verbosity >= 1) {
            fprintf(stderr, "piano: segment %d / %d\n", seg, n_segments);
        }

        const float* seg_pcm = audio.data() + ptr;

        // Compute mel spectrogram with BN0
        int T_mel;
        auto mel_bn = compute_mel_with_bn0(ctx, seg_pcm, segment_samples, T_mel);

        // Run 4 acoustic models
        auto frame_out = acoustic_model_forward(mel_bn.data(), T_mel, hp.n_mels, w.frame, hp.gru_hidden);
        auto onset_out = acoustic_model_forward(mel_bn.data(), T_mel, hp.n_mels, w.onset, hp.gru_hidden);
        auto offset_out = acoustic_model_forward(mel_bn.data(), T_mel, hp.n_mels, w.offset, hp.gru_hidden);
        auto velocity_out = acoustic_model_forward(mel_bn.data(), T_mel, hp.n_mels, w.velocity, hp.gru_hidden);

        // Onset refinement: cat(onset, sqrt(onset) * velocity) → BiGRU → FC → sigmoid
        // Input: [T, 88*2 = 176]
        int refine_input = hp.classes_num * 2;
        std::vector<float> onset_refine_input(T_mel * refine_input);
        for (int t = 0; t < T_mel; t++) {
            for (uint32_t k = 0; k < hp.classes_num; k++) {
                float o = onset_out[t * hp.classes_num + k];
                float v = velocity_out[t * hp.classes_num + k];
                onset_refine_input[t * refine_input + k] = o;
                onset_refine_input[t * refine_input + hp.classes_num + k] = std::sqrt(o) * v;
            }
        }
        auto onset_gru_out =
            bigru_forward(onset_refine_input.data(), T_mel, refine_input, hp.gru_hidden, w.onset_refine_gru);
        auto onset_fc_w = tensor_to_f32(w.onset_refine_fc_w);
        auto onset_fc_b = tensor_to_f32(w.onset_refine_fc_b);
        onset_out = linear(onset_gru_out.data(), T_mel, 2 * hp.gru_hidden, onset_fc_w.data(), onset_fc_b.data(),
                           hp.classes_num);
        sigmoid_inplace(onset_out.data(), T_mel * hp.classes_num);

        // Frame refinement: cat(frame, onset, offset) → BiGRU → FC → sigmoid
        // Input: [T, 88*3 = 264]
        int frame_refine_input = hp.classes_num * 3;
        std::vector<float> frame_refine_in(T_mel * frame_refine_input);
        for (int t = 0; t < T_mel; t++) {
            for (uint32_t k = 0; k < hp.classes_num; k++) {
                frame_refine_in[t * frame_refine_input + k] = frame_out[t * hp.classes_num + k];
                frame_refine_in[t * frame_refine_input + hp.classes_num + k] = onset_out[t * hp.classes_num + k];
                frame_refine_in[t * frame_refine_input + 2 * hp.classes_num + k] = offset_out[t * hp.classes_num + k];
            }
        }
        auto frame_gru_out =
            bigru_forward(frame_refine_in.data(), T_mel, frame_refine_input, hp.gru_hidden, w.frame_refine_gru);
        auto frame_fc_w = tensor_to_f32(w.frame_refine_fc_w);
        auto frame_fc_b = tensor_to_f32(w.frame_refine_fc_b);
        frame_out = linear(frame_gru_out.data(), T_mel, 2 * hp.gru_hidden, frame_fc_w.data(), frame_fc_b.data(),
                           hp.classes_num);
        sigmoid_inplace(frame_out.data(), T_mel * hp.classes_num);

        seg_outputs[seg].frame = std::move(frame_out);
        seg_outputs[seg].onset = std::move(onset_out);
        seg_outputs[seg].offset = std::move(offset_out);
        seg_outputs[seg].velocity = std::move(velocity_out);
        seg_outputs[seg].T_seg = T_mel;

        ptr += hop_samples;
    }

    // Deframe segments (overlap-add style from upstream)
    // Each segment has T_seg frames. With 50% overlap:
    // - First segment: use frames [0, 0.75*T)
    // - Middle segments: use frames [0.25*T, 0.75*T)
    // - Last segment: use frames [0.25*T, T)
    auto deframe = [&](const std::string& which) -> std::vector<float> {
        if (n_segments == 1) {
            if (which == "frame")
                return seg_outputs[0].frame;
            if (which == "onset")
                return seg_outputs[0].onset;
            if (which == "offset")
                return seg_outputs[0].offset;
            return seg_outputs[0].velocity;
        }

        std::vector<float> combined;
        for (int seg = 0; seg < n_segments; seg++) {
            int T_seg = seg_outputs[seg].T_seg;
            // Remove last frame (center=True artifact)
            int T_use = T_seg - 1;
            int start_frame = 0;
            int end_frame = T_use;

            if (seg == 0) {
                end_frame = (int)(T_use * 0.75);
            } else if (seg == n_segments - 1) {
                start_frame = (int)(T_use * 0.25);
            } else {
                start_frame = (int)(T_use * 0.25);
                end_frame = (int)(T_use * 0.75);
            }

            const std::vector<float>* src = nullptr;
            if (which == "frame")
                src = &seg_outputs[seg].frame;
            else if (which == "onset")
                src = &seg_outputs[seg].onset;
            else if (which == "offset")
                src = &seg_outputs[seg].offset;
            else
                src = &seg_outputs[seg].velocity;

            for (int t = start_frame; t < end_frame && t < T_seg; t++) {
                combined.insert(combined.end(), src->data() + t * hp.classes_num,
                                src->data() + (t + 1) * hp.classes_num);
            }
        }
        return combined;
    };

    auto frame_all = deframe("frame");
    auto onset_all = deframe("onset");
    auto offset_all = deframe("offset");
    auto velocity_all = deframe("velocity");

    int T_total = (int)(frame_all.size() / hp.classes_num);
    // Trim to original audio length in frames
    int audio_frames = (int)std::ceil((float)n_samples / hp.hop_size);
    if (T_total > audio_frames)
        T_total = audio_frames;

    // Post-processing: binarize onsets and offsets
    std::vector<float> onset_binary, onset_shift, offset_binary, offset_shift;
    binarize_regression(onset_all.data(), T_total, hp.classes_num, p.onset_threshold, 2, onset_binary, onset_shift);
    binarize_regression(offset_all.data(), T_total, hp.classes_num, p.offset_threshold, 4, offset_binary, offset_shift);

    // Detect notes
    auto tuples = note_detection(frame_all.data(), onset_binary.data(), onset_shift.data(), offset_binary.data(),
                                 offset_shift.data(), velocity_all.data(), T_total, hp.classes_num, p.frame_threshold);

    // Convert tuples to note events
    // tuples contain per-class detections but we need to track which class each belongs to.
    // The note_detection function processes all classes in sequence, so we need to
    // re-run per class to know the MIDI note.

    // Actually, let me re-implement note detection properly per-class:
    std::vector<piano_note_event> events;
    for (uint32_t k = 0; k < hp.classes_num; k++) {
        int bgn = -1;
        int frame_disappear = -1;
        int offset_occur = -1;

        for (int i = 0; i < T_total; i++) {
            if (onset_binary[i * hp.classes_num + k] == 1.0f) {
                if (bgn >= 0) {
                    int fin = std::max(i - 1, 0);
                    float onset_time = (bgn + onset_shift[bgn * hp.classes_num + k]) / hp.frames_per_second;
                    float offset_time = (float)fin / hp.frames_per_second;
                    int vel = (int)(velocity_all[bgn * hp.classes_num + k] * piano_hparams::velocity_scale);
                    vel = std::clamp(vel, 0, 127);
                    events.push_back({onset_time, offset_time, (int)(k + hp.begin_note), vel});
                    frame_disappear = -1;
                    offset_occur = -1;
                }
                bgn = i;
            }

            if (bgn >= 0 && i > bgn) {
                if (frame_all[i * hp.classes_num + k] <= p.frame_threshold && frame_disappear < 0)
                    frame_disappear = i;
                if (offset_binary[i * hp.classes_num + k] == 1.0f && offset_occur < 0)
                    offset_occur = i;

                if (frame_disappear >= 0) {
                    int fin;
                    if (offset_occur >= 0 && offset_occur - bgn > frame_disappear - offset_occur)
                        fin = offset_occur;
                    else
                        fin = frame_disappear;

                    float onset_time = (bgn + onset_shift[bgn * hp.classes_num + k]) / hp.frames_per_second;
                    float offset_time = (fin + offset_shift[fin * hp.classes_num + k]) / hp.frames_per_second;
                    int vel = (int)(velocity_all[bgn * hp.classes_num + k] * piano_hparams::velocity_scale);
                    vel = std::clamp(vel, 0, 127);
                    events.push_back({onset_time, offset_time, (int)(k + hp.begin_note), vel});
                    bgn = -1;
                    frame_disappear = -1;
                    offset_occur = -1;
                }

                if (bgn >= 0 && (i - bgn >= 600 || i == T_total - 1)) {
                    int fin = i;
                    float onset_time = (bgn + onset_shift[bgn * hp.classes_num + k]) / hp.frames_per_second;
                    float offset_time = (fin + offset_shift[fin * hp.classes_num + k]) / hp.frames_per_second;
                    int vel = (int)(velocity_all[bgn * hp.classes_num + k] * piano_hparams::velocity_scale);
                    vel = std::clamp(vel, 0, 127);
                    events.push_back({onset_time, offset_time, (int)(k + hp.begin_note), vel});
                    bgn = -1;
                    frame_disappear = -1;
                    offset_occur = -1;
                }
            }
        }
    }

    // Sort by onset time
    std::sort(events.begin(), events.end(),
              [](const piano_note_event& a, const piano_note_event& b) { return a.onset_time < b.onset_time; });

    // Copy to result
    result->n_notes = (int)events.size();
    if (result->n_notes > 0) {
        result->note_events = (piano_note_event*)malloc(result->n_notes * sizeof(piano_note_event));
        std::memcpy(result->note_events, events.data(), result->n_notes * sizeof(piano_note_event));
    }

    // TODO: pedal detection (similar to note detection but simpler)
    result->n_pedals = 0;
    result->pedal_events = nullptr;

    // Optionally store raw outputs
    if (p.verbosity >= 2) {
        result->n_frames = T_total;
        result->n_classes = hp.classes_num;
        result->frame_output = (float*)malloc(T_total * hp.classes_num * sizeof(float));
        result->onset_output = (float*)malloc(T_total * hp.classes_num * sizeof(float));
        result->offset_output = (float*)malloc(T_total * hp.classes_num * sizeof(float));
        result->velocity_output = (float*)malloc(T_total * hp.classes_num * sizeof(float));
        std::memcpy(result->frame_output, frame_all.data(), T_total * hp.classes_num * sizeof(float));
        std::memcpy(result->onset_output, onset_all.data(), T_total * hp.classes_num * sizeof(float));
        std::memcpy(result->offset_output, offset_all.data(), T_total * hp.classes_num * sizeof(float));
        std::memcpy(result->velocity_output, velocity_all.data(), T_total * hp.classes_num * sizeof(float));
    }

    if (p.verbosity >= 1) {
        fprintf(stderr, "piano: detected %d notes in %d frames\n", result->n_notes, T_total);
    }

    return 0;
}

void piano_transcription_result_free(struct piano_transcription_result* result) {
    if (!result)
        return;
    free(result->note_events);
    free(result->pedal_events);
    free(result->frame_output);
    free(result->onset_output);
    free(result->offset_output);
    free(result->velocity_output);
    std::memset(result, 0, sizeof(*result));
}

// ─── Diff harness helpers ───────────────────────────────────────────────────

float* piano_transcription_mel_spectrogram(struct piano_transcription_ctx* ctx, const float* pcm, int n_samples,
                                           int* out_n) {
    if (!ctx || !pcm || n_samples <= 0 || !out_n)
        return nullptr;
    int T;
    auto mel = compute_mel_with_bn0(ctx, pcm, n_samples, T);
    *out_n = (int)mel.size();
    float* out = (float*)malloc(mel.size() * sizeof(float));
    std::memcpy(out, mel.data(), mel.size() * sizeof(float));
    return out;
}

float* piano_transcription_acoustic_model(struct piano_transcription_ctx* ctx, const float* mel, int n_mel_elements,
                                          const char* model_name, int* out_n) {
    if (!ctx || !mel || n_mel_elements <= 0 || !model_name || !out_n)
        return nullptr;

    int T = n_mel_elements / ctx->hp.n_mels;
    const piano_acoustic_model* model = nullptr;
    std::string name(model_name);
    if (name == "frame")
        model = &ctx->weights.frame;
    else if (name == "onset")
        model = &ctx->weights.onset;
    else if (name == "offset")
        model = &ctx->weights.offset;
    else if (name == "velocity")
        model = &ctx->weights.velocity;
    else
        return nullptr;

    auto out = acoustic_model_forward(mel, T, ctx->hp.n_mels, *model, ctx->hp.gru_hidden);
    *out_n = (int)out.size();
    float* ret = (float*)malloc(out.size() * sizeof(float));
    std::memcpy(ret, out.data(), out.size() * sizeof(float));
    return ret;
}

// ───────────────────────────────────────────────────────────────────────────
// Diff harness — per-stage parity against tools/reference_backends/piano_transcription.py
//
// NOTE ON INPUT ALIGNMENT. Unlike the btc and mel-band-roformer harnesses, the
// piano reference dump carries NO input_audio stage, so this cannot replay the
// reference's exact samples — it recomputes the front end from the same WAV.
// That is fine while both sides read an already-16 kHz mono file (no resampler
// in the path), and `mel_spectrogram` is compared FIRST precisely so a
// front-end difference shows up as itself rather than as a model failure.
// If the reference ever gains an input_audio stage, replay it instead.
//
// The reference runs ONE forward pass over the whole clip. transcribe() instead
// splits audio into overlapping segments, so this deliberately calls the
// internals directly rather than going through transcribe() — comparing against
// a segmented+deframed result would diff two different computations.
// ───────────────────────────────────────────────────────────────────────────

namespace {

double piano_cosine(const float* a, const float* b, int64_t n) {
    double dot = 0, na = 0, nb = 0;
    for (int64_t i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    const double den = std::sqrt(na) * std::sqrt(nb);
    return den > 0 ? dot / den : (na == 0 && nb == 0 ? 1.0 : 0.0);
}

double piano_max_abs(const float* a, const float* b, int64_t n) {
    double m = 0;
    for (int64_t i = 0; i < n; i++)
        m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}

double piano_l2(const float* a, int64_t n) {
    double s = 0;
    for (int64_t i = 0; i < n; i++)
        s += (double)a[i] * a[i];
    return std::sqrt(s);
}

bool piano_ref_get(core_gguf::WeightLoad& rw, const char* name, std::vector<float>& out) {
    auto it = rw.tensors.find(name);
    if (it == rw.tensors.end() || !it->second)
        return false;
    const int64_t n = ggml_nelements(it->second);
    out.resize((size_t)n);
    ggml_backend_tensor_get(it->second, out.data(), 0, (size_t)n * sizeof(float));
    return true;
}

} // namespace

int piano_transcription_diff(const char* model_gguf, const char* ref_gguf, const float* pcm_16k, int n_samples,
                             int verbosity) {
    piano_transcription_params p = piano_transcription_default_params();
    p.verbosity = 0;
    piano_transcription_ctx* ctx = piano_transcription_init_from_file(model_gguf, p);
    if (!ctx) {
        fprintf(stderr, "piano_diff: failed to load model %s\n", model_gguf);
        return 2;
    }
    core_gguf::WeightLoad rw;
    if (!core_gguf::load_weights(ref_gguf, ctx->backend, "piano_ref", rw)) {
        fprintf(stderr, "piano_diff: failed to load reference %s\n", ref_gguf);
        piano_transcription_free(ctx);
        return 2;
    }

    const auto& hp = ctx->hp;
    const auto& w = ctx->weights;
    int n_fail = 0;
    const double COS_MIN = 0.999;

    auto report = [&](const char* stage, const std::vector<float>& mine, const std::vector<float>& ref) {
        const int64_t n = (int64_t)std::min(mine.size(), ref.size());
        const double cos = piano_cosine(mine.data(), ref.data(), n);
        const double mad = piano_max_abs(mine.data(), ref.data(), n);
        const bool ok = cos >= COS_MIN && mine.size() == ref.size();
        if (!ok)
            n_fail++;
        if (verbosity >= 1 || !ok) {
            fprintf(stderr, "  %-20s %s cos=%.6f max_abs=%.3e  (mine=%zu ref=%zu)", stage, ok ? "PASS" : "FAIL", cos,
                    mad, mine.size(), ref.size());
            if (verbosity >= 2)
                // Absolute magnitudes next to the cosine: cosine is scale-blind,
                // so a uniform gain error passes it silently.
                fprintf(stderr, "  |mine|=%.6f |ref|=%.6f", piano_l2(mine.data(), n), piano_l2(ref.data(), n));
            fprintf(stderr, "\n");
        }
    };

    int T = 0;
    auto mel_bn = compute_mel_with_bn0(ctx, pcm_16k, n_samples, T);
    fprintf(stderr, "piano_transcription diff (n_samples=%d, T=%d, n_mels=%u, classes=%u):\n", n_samples, T, hp.n_mels,
            hp.classes_num);

    {
        std::vector<float> ref_mel;
        if (piano_ref_get(rw, "mel_spectrogram", ref_mel))
            report("mel_spectrogram", mel_bn, ref_mel);
        else
            fprintf(stderr, "  mel_spectrogram      SKIP (absent from reference)\n");
    }

    // The reference dumps conv/fc5/gru for the ONSET model as representative.
    piano_acoustic_taps taps;
    std::vector<float> conv_out, fc5_out, gru_out;
    taps.conv_block_output = &conv_out;
    taps.fc5_output = &fc5_out;
    taps.gru_output = &gru_out;

    auto onset_raw = acoustic_model_forward(mel_bn.data(), T, hp.n_mels, w.onset, hp.gru_hidden, &taps);
    auto frame_raw = acoustic_model_forward(mel_bn.data(), T, hp.n_mels, w.frame, hp.gru_hidden);
    auto offset_out = acoustic_model_forward(mel_bn.data(), T, hp.n_mels, w.offset, hp.gru_hidden);
    auto velocity_out = acoustic_model_forward(mel_bn.data(), T, hp.n_mels, w.velocity, hp.gru_hidden);

    {
        std::vector<float> r;
        if (piano_ref_get(rw, "conv_block_output", r))
            report("conv_block_output", conv_out, r);
        if (piano_ref_get(rw, "fc5_output", r))
            report("fc5_output", fc5_out, r);
        if (piano_ref_get(rw, "gru_output", r))
            report("gru_output", gru_out, r);
        // offset/velocity are NOT refined upstream, so the raw head IS the output.
        if (piano_ref_get(rw, "offset_output", r))
            report("offset_output", offset_out, r);
        if (piano_ref_get(rw, "velocity_output", r))
            report("velocity_output", velocity_out, r);
    }

    // Onset refinement: cat(onset, sqrt(onset)*velocity) -> BiGRU -> FC -> sigmoid
    const int refine_in = (int)hp.classes_num * 2;
    std::vector<float> onset_refine_input((size_t)T * refine_in);
    for (int t = 0; t < T; t++)
        for (uint32_t k = 0; k < hp.classes_num; k++) {
            const float o = onset_raw[(size_t)t * hp.classes_num + k];
            const float v = velocity_out[(size_t)t * hp.classes_num + k];
            onset_refine_input[(size_t)t * refine_in + k] = o;
            onset_refine_input[(size_t)t * refine_in + hp.classes_num + k] = std::sqrt(o) * v;
        }
    auto onset_gru = bigru_forward(onset_refine_input.data(), T, refine_in, hp.gru_hidden, w.onset_refine_gru);
    auto o_fc_w = tensor_to_f32(w.onset_refine_fc_w);
    auto o_fc_b = tensor_to_f32(w.onset_refine_fc_b);
    auto onset_final =
        linear(onset_gru.data(), T, 2 * (int)hp.gru_hidden, o_fc_w.data(), o_fc_b.data(), (int)hp.classes_num);
    sigmoid_inplace(onset_final.data(), (size_t)T * hp.classes_num);

    // Frame refinement: cat(frame, onset_final, offset) -> BiGRU -> FC -> sigmoid
    const int frame_refine_in = (int)hp.classes_num * 3;
    std::vector<float> frame_in((size_t)T * frame_refine_in);
    for (int t = 0; t < T; t++)
        for (uint32_t k = 0; k < hp.classes_num; k++) {
            frame_in[(size_t)t * frame_refine_in + k] = frame_raw[(size_t)t * hp.classes_num + k];
            frame_in[(size_t)t * frame_refine_in + hp.classes_num + k] = onset_final[(size_t)t * hp.classes_num + k];
            frame_in[(size_t)t * frame_refine_in + 2 * hp.classes_num + k] = offset_out[(size_t)t * hp.classes_num + k];
        }
    auto frame_gru = bigru_forward(frame_in.data(), T, frame_refine_in, hp.gru_hidden, w.frame_refine_gru);
    auto f_fc_w = tensor_to_f32(w.frame_refine_fc_w);
    auto f_fc_b = tensor_to_f32(w.frame_refine_fc_b);
    auto frame_final =
        linear(frame_gru.data(), T, 2 * (int)hp.gru_hidden, f_fc_w.data(), f_fc_b.data(), (int)hp.classes_num);
    sigmoid_inplace(frame_final.data(), (size_t)T * hp.classes_num);

    {
        std::vector<float> r;
        if (piano_ref_get(rw, "onset_output", r))
            report("onset_output", onset_final, r);
        if (piano_ref_get(rw, "frame_output", r))
            report("frame_output", frame_final, r);
    }

    // Declare the coverage gap rather than implying full coverage: the pedal
    // sub-model is dumped by the reference but this runtime path does not
    // produce it here.
    {
        std::vector<float> tmp;
        std::string present;
        for (const char* nm : {"pedal_onset_output", "pedal_offset_output", "pedal_frame_output"})
            if (piano_ref_get(rw, nm, tmp))
                present += (present.empty() ? "" : ", ") + std::string(nm);
        if (!present.empty())
            fprintf(stderr, "  NOTE: in the reference but not compared: %s\n", present.c_str());
    }

    core_gguf::free_weights(rw);
    piano_transcription_free(ctx);
    fprintf(stderr, "piano_transcription diff: %d stage(s) FAILED.\n", n_fail);
    return n_fail == 0 ? 0 : 1;
}
