// voxtral_tts.cpp — Voxtral-4B-TTS runtime (mistralai/Voxtral-4B-TTS-2603).
//
// Three-component TTS pipeline:
//   1. Ministral-3B AR backbone (26L GQA) — predicts semantic tokens per frame
//   2. Acoustic FM transformer (3L bidirectional, 8-step Euler ODE) — predicts
//      36 acoustic FSQ values per frame
//   3. Voxtral codec decoder (4 conv+transformer blocks) — decodes to 24 kHz PCM
//
// Reuses:
//   - core_attn::llama_self_attn_kv() for the LLM backbone (GQA + RoPE + KV cache)
//   - core_ffn::swiglu() for SwiGLU FFN in both LLM and FM
//   - Tekken BPE tokenizer pattern from voxtral4b.cpp
//
// Env knobs:
//   CRISPASR_VOXTRAL_TTS_TIMING=1    print per-stage LLM/FM ms/frame
//   CRISPASR_VOXTRAL_TTS_FM_STEPS=N  Euler ODE step count (default 8 → 7 intervals).
//                                    EXPERIMENTAL quality/speed lever. On a clean CUDA
//                                    GPU (P100) 7 steps is ~11% faster end-to-end, but
//                                    it is a QUALITY TRADEOFF not free speed: fewer
//                                    steps change the output (prosody/duration) via the
//                                    acoustic feedback loop, so validate by ear before
//                                    lowering. The model is calibrated for 8.
//                                    (FM-on-CPU was tried and dropped: 18x slower on
//                                    CUDA, slower on Metal — a real GPU always wins.)
//   CRISPASR_VOXTRAL_TTS_DEBUG=1     per-frame code dump + semantic-argmax diag
//   CRISPASR_VOXTRAL_TTS_SEMANTIC_CB=<f32 blob>  side-load codec.semantic_cb
//   CRISPASR_VOXTRAL_TTS_CODEC_FROM_FILE=<codes> codec-only diff-harness mode

#include "voxtral_tts.h"

#include "voxtral_tekken_vocab.h" // #338 active-vocabulary bound

#include "core/attention.h"
#include "core/conv.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"
#include "core/crispasr_env.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem> // crispasr-diff LLM stage temp dir (portable; replaces POSIX mkdtemp/unistd.h — no unistd.h on MSVC, breaks the Windows build)
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include "core/ggml_cpu_backend.h"

// ---------------------------------------------------------------------------
// Hyperparameters
// ---------------------------------------------------------------------------

struct voxtral_tts_hparams {
    // LLM backbone (Ministral 3B)
    int llm_n_layers = 26;
    int llm_dim = 3072;
    int llm_n_heads = 32;
    int llm_n_kv = 8;
    int llm_head_dim = 128;
    int llm_ff_dim = 9216;
    float llm_rope_theta = 1000000.0f;
    float llm_norm_eps = 1e-5f;
    int llm_vocab_size = 131072;
    bool tied_embeddings = true;

    // FM transformer (acoustic flow-matching)
    int fm_n_layers = 3;
    int fm_dim = 3072;
    int fm_n_heads = 32;
    int fm_n_kv = 8;
    int fm_head_dim = 128;
    int fm_ff_dim = 9216;
    float fm_rope_theta = 10000.0f;
    float fm_sigma = 1e-5f;
    float fm_sigma_max = 1.0f;

    // Audio encoding
    int semantic_cb_size = 8192;
    int acoustic_cb_size = 21; // FSQ levels
    int n_acoustic_cb = 36;
    int n_codebooks = 37; // 1 semantic + 36 acoustic
    int sample_rate = 24000;
    float frame_rate = 12.5f;

    // Special tokens
    int audio_token_id = 24;
    int begin_audio_token_id = 25;
    int cond_dropped_token_id = 42;
    int bos_token_id = 1;

    // Codec decoder
    int codec_dim = 1024;
    int codec_hidden_dim = 4096;
    int codec_n_heads = 8;
    int codec_n_kv = 8;
    int codec_head_dim = 128;
    int codec_semantic_dim = 256;
    int codec_acoustic_dim = 36;
    int codec_patch_size = 240;
    int codec_patch_kernel = 7;
    int codec_attn_window = 16;
    float codec_norm_eps = 0.01f;
    float codec_qk_norm_eps = 1e-6f;
    bool codec_qk_norm = true;
    bool codec_layer_scale = true;
    std::vector<int> codec_conv_strides; // {1, 2, 2, 2}
    std::vector<int> codec_conv_kernels; // {3, 4, 4, 4}
    std::vector<int> codec_tfm_lengths;  // {2, 2, 2, 2}
};

// ---------------------------------------------------------------------------
// Tekken BPE tokenizer (shared pattern with voxtral4b.cpp)
// ---------------------------------------------------------------------------

struct voxtral_tts_vocab {
    std::string pre_pattern;
    std::vector<std::string> specials;
    int n_specials = 0;
    int n_vocab = 0;

    // BPE merge table: pair → merged piece
    std::vector<std::pair<std::string, std::string>> merges;
    // Piece → token ID (specials first, then BPE vocab)
    std::map<std::string, int> piece_to_id;
    // Token ID → piece
    std::vector<std::string> id_to_piece;

    std::vector<uint8_t> tekken_vocab_blob;
};

// ---------------------------------------------------------------------------
// Model weights
// ---------------------------------------------------------------------------

struct voxtral_tts_llm_layer {
    ggml_tensor* attn_norm = nullptr;
    ggml_tensor* attn_q = nullptr;
    ggml_tensor* attn_k = nullptr;
    ggml_tensor* attn_v = nullptr;
    ggml_tensor* attn_o = nullptr;
    ggml_tensor* ffn_norm = nullptr;
    ggml_tensor* ffn_gate = nullptr;
    ggml_tensor* ffn_up = nullptr;
    ggml_tensor* ffn_down = nullptr;
};

struct voxtral_tts_fm_layer {
    ggml_tensor* attn_norm = nullptr;
    ggml_tensor* attn_q = nullptr;
    ggml_tensor* attn_k = nullptr;
    ggml_tensor* attn_v = nullptr;
    ggml_tensor* attn_o = nullptr;
    ggml_tensor* ffn_norm = nullptr;
    ggml_tensor* ffn_gate = nullptr;
    ggml_tensor* ffn_up = nullptr;
    ggml_tensor* ffn_down = nullptr;
};

struct voxtral_tts_codec_tfm_layer {
    ggml_tensor* attn_norm = nullptr;
    ggml_tensor* attn_q = nullptr;
    ggml_tensor* attn_k = nullptr;
    ggml_tensor* attn_v = nullptr;
    ggml_tensor* attn_o = nullptr;
    ggml_tensor* q_norm = nullptr;
    ggml_tensor* k_norm = nullptr;
    ggml_tensor* ffn_norm = nullptr;
    ggml_tensor* ffn_gate = nullptr;
    ggml_tensor* ffn_up = nullptr;
    ggml_tensor* ffn_down = nullptr;
    ggml_tensor* attn_scale = nullptr; // layer_scale
    ggml_tensor* ffn_scale = nullptr;
};

struct voxtral_tts_codec_conv {
    ggml_tensor* weight = nullptr;
    ggml_tensor* bias = nullptr;
};

struct voxtral_tts_model {
    // LLM backbone
    ggml_tensor* token_embd = nullptr; // (vocab_size, dim)
    ggml_tensor* audio_embd = nullptr; // (9088, dim) — combined audio codebook embeddings
    ggml_tensor* output_norm = nullptr;
    std::vector<voxtral_tts_llm_layer> llm_layers;

    // FM transformer
    ggml_tensor* fm_input_proj = nullptr;           // (dim, 36)
    ggml_tensor* fm_llm_proj = nullptr;             // (dim, dim)
    ggml_tensor* fm_time_proj = nullptr;            // (dim, dim)
    ggml_tensor* fm_semantic_output = nullptr;      // (8320, dim)
    ggml_tensor* fm_semantic_output_bias = nullptr; // (8320,) — optional
    ggml_tensor* fm_acoustic_output = nullptr;      // (36, dim)
    ggml_tensor* fm_norm = nullptr;
    std::vector<voxtral_tts_fm_layer> fm_layers;

    // Codec decoder
    std::vector<voxtral_tts_codec_conv> codec_convs;                        // 4 conv layers
    std::vector<std::vector<voxtral_tts_codec_tfm_layer>> codec_tfm_blocks; // 4 blocks × 2 layers
    voxtral_tts_codec_conv codec_output;
    voxtral_tts_codec_conv codec_patch_proj;
    ggml_tensor* codec_semantic_cb = nullptr; // (8192, 256)

    // Voice embeddings
    std::map<std::string, ggml_tensor*> voice_tensors;
};

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

struct voxtral_tts_context {
    voxtral_tts_hparams hp;
    voxtral_tts_model model;
    voxtral_tts_vocab vocab;
    voxtral_tts_context_params params;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_context* ctx_w = nullptr; // weight context

    // Compute scheduler + graph metadata scratch (shared across LLM/FM/codec graphs)
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    // Cached FM velocity graph (fixed shape → built + allocated once via a dedicated
    // gallocr, single backend, reused every eval). Computes cond+uncond in one pass.
    ggml_context* fm_ctx = nullptr;
    ggml_cgraph* fm_gf = nullptr;
    ggml_gallocr_t fm_alloc = nullptr;
    std::vector<uint8_t> fm_meta;

    // Euler ODE step count (default 8 → 7 intervals). CRISPASR_VOXTRAL_TTS_FM_STEPS=N
    // overrides — experimental quality/speed lever (see file-header note).
    int fm_flow_steps = 8; // = VTTS_FLOW_STEPS

    // KV cache for LLM backbone
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr; // (head_dim, max_ctx, n_kv, n_layers)
    ggml_tensor* kv_v = nullptr;
    int kv_used = 0;

    // Voice name list (for list_voices API)
    std::vector<std::string> voice_names;
    std::vector<const char*> voice_name_ptrs;

    // Flow-matching noise RNG (xorshift64, matching the reference C port so the
    // acoustic sample is reproducible; the semantic path is deterministic argmax).
    uint64_t rng_state = 0x12345678ABCDEF01ULL;

    // Semantic VQ codebook as host F32 (8192 × 256). The codec's 292-d input is
    // [semantic_cb[sem] (256) ; FSQ(acoustic) (36)]. Populated from the GGUF
    // tensor if present, else side-loaded from CRISPASR_VOXTRAL_TTS_SEMANTIC_CB
    // (a raw float32 blob) since older GGUFs dropped codec.semantic_cb.weight.
    std::vector<float> semantic_cb_host;

    int verbosity = 1;
};

// xorshift64 + Box-Muller, bit-identical to github.com/mudler/voxtral-tts.c so a
// fixed seed reproduces the same acoustic sample.
static inline uint64_t vtts_xorshift64(uint64_t* s) {
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    return x;
}
static inline float vtts_uniform01(uint64_t* s) {
    return (float)(vtts_xorshift64(s) >> 11) * (1.0f / 9007199254740992.0f);
}
static inline float vtts_randn(uint64_t* s) {
    float u1, u2;
    do {
        u1 = vtts_uniform01(s);
    } while (u1 < 1e-30f);
    u2 = vtts_uniform01(s);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853071795864f * u2);
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static bool env_bool(const char* name) {
    const char* v = crispasr_env::get(name);
    return v && (*v == '1' || *v == 'y' || *v == 'Y');
}

// Env int with fallback (returns def if unset/unparseable/<=0).
static int env_int(const char* name, int def) {
    const char* v = crispasr_env::get(name);
    if (!v || !*v)
        return def;
    int n = atoi(v);
    return n > 0 ? n : def;
}

static constexpr int VTTS_FLOW_STEPS = 8; // FM timesteps → 7 Euler intervals (default)

static std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> out;
    size_t start = 0;
    while (start < s.size()) {
        size_t end = s.find(',', start);
        if (end == std::string::npos)
            end = s.size();
        out.push_back(std::stoi(s.substr(start, end - start)));
        start = end + 1;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Tekken BPE tokenizer (adapted from voxtral4b.cpp)
// ---------------------------------------------------------------------------

// #338: `llm_vocab_size` is the width of the embedding table, and the Tekken
// blob is allowed to be LONGER than it — the tail entries are inert padding
// this checkpoint never activates. Admitting them to the encoder's map lets it
// emit ids past the table, which reaches `ggml_get_rows(token_embd, …)` out of
// bounds. The bound and the decode both live in voxtral_tekken_vocab.h so this
// runtime and the diff harness's reference tokenizer cannot drift apart again.
static void tekken_build_vocab(voxtral_tts_vocab& v, int llm_vocab_size, int verbosity) {
    const int active_limit = voxtral_tekken::active_bpe_count(llm_vocab_size, v.n_specials);
    const auto st = voxtral_tekken::decode_blob(v.tekken_vocab_blob, v.n_specials, v.specials, active_limit,
                                                v.id_to_piece, v.piece_to_id);
    if (st.n_inactive > 0 && verbosity >= 1) {
        fprintf(stderr,
                "voxtral_tts: tekken vocab: %d active BPE pieces, %d inactive tail entries ignored "
                "(llm_vocab_size=%d, %d specials)\n",
                st.n_active, st.n_inactive, llm_vocab_size, v.n_specials);
    }
}

static void tekken_bpe_encode(const voxtral_tts_vocab& v, const uint8_t* data, size_t len, std::vector<int32_t>& out) {
    if (len == 0)
        return;

    // Initialize: each byte is its own piece
    std::vector<std::string> pieces;
    for (size_t i = 0; i < len; i++) {
        pieces.push_back(std::string(1, (char)data[i]));
    }

    // Greedy BPE merge: repeatedly find the highest-priority merge
    while (pieces.size() > 1) {
        int best_pos = -1;
        int best_id = INT32_MAX;
        for (int i = 0; i < (int)pieces.size() - 1; i++) {
            std::string merged = pieces[i] + pieces[i + 1];
            auto it = v.piece_to_id.find(merged);
            if (it != v.piece_to_id.end() && it->second < best_id) {
                best_id = it->second;
                best_pos = i;
            }
        }
        if (best_pos < 0)
            break;
        pieces[best_pos] = pieces[best_pos] + pieces[best_pos + 1];
        pieces.erase(pieces.begin() + best_pos + 1);
    }

    // Map pieces to IDs
    for (auto& pc : pieces) {
        auto it = v.piece_to_id.find(pc);
        if (it != v.piece_to_id.end()) {
            out.push_back(it->second);
        } else {
            // Fallback: encode each byte as its own token
            for (unsigned char c : pc) {
                std::string s(1, (char)c);
                auto it2 = v.piece_to_id.find(s);
                if (it2 != v.piece_to_id.end()) {
                    out.push_back(it2->second);
                }
            }
        }
    }
}

// Pre-tokenizer lives in voxtral_tekken_vocab.h so the runtime and its
// tests share one implementation (the #338 lesson).
static std::vector<std::string> tekken_pre_tokenize(const std::string& text) {
    return voxtral_tekken::pre_tokenize(text);
}

static std::vector<int32_t> voxtral_tts_tokenize(voxtral_tts_context* ctx, const std::string& text) {
    std::vector<int32_t> ids;
    auto& v = ctx->vocab;

    // Check for special tokens in the text
    size_t pos = 0;
    while (pos < text.size()) {
        // Look for special token at current position
        bool found_special = false;
        for (int si = 0; si < (int)v.specials.size(); si++) {
            const auto& sp = v.specials[si];
            if (sp.empty())
                continue;
            if (text.compare(pos, sp.size(), sp) == 0) {
                ids.push_back(si);
                pos += sp.size();
                found_special = true;
                break;
            }
        }
        if (found_special)
            continue;

        // Find the next special token
        size_t next_special = text.size();
        for (const auto& sp : v.specials) {
            if (sp.empty())
                continue;
            size_t f = text.find(sp, pos);
            if (f != std::string::npos && f < next_special)
                next_special = f;
        }

        // BPE encode the text between specials
        if (pos < next_special) {
            auto pre_tokens = tekken_pre_tokenize(text.substr(pos, next_special - pos));
            for (auto& pt : pre_tokens) {
                tekken_bpe_encode(v, (const uint8_t*)pt.data(), pt.size(), ids);
            }
        }
        pos = next_special;
    }
    // #338 belt-and-braces: the bounded vocab above should make this
    // unreachable, but an out-of-range id here becomes an out-of-bounds row in
    // `ggml_get_rows(token_embd, …)` — a CPU assertion, or NaN-from-frame-0 and
    // runaway generation on CUDA. Drop it loudly instead of handing it to ggml;
    // a missing token mangles one word, a bad row index takes down the process
    // or the whole utterance.
    {
        const int vs = (int)ctx->hp.llm_vocab_size;
        size_t n_bad = 0;
        std::vector<int32_t> kept;
        kept.reserve(ids.size());
        for (int32_t id : ids) {
            if (voxtral_tekken::token_id_in_range(id, vs)) {
                kept.push_back(id);
            } else {
                n_bad++;
            }
        }
        if (n_bad > 0) {
            fprintf(stderr,
                    "voxtral_tts: dropped %zu token id(s) outside [0, %d) before embedding lookup — the tokenizer "
                    "produced an id this checkpoint cannot address (please report with the input text)\n",
                    n_bad, vs);
            ids.swap(kept);
        }
    }
    return ids;
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

extern "C" voxtral_tts_context_params voxtral_tts_context_default_params(void) {
    voxtral_tts_context_params p{};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    p.temperature = 0.0f;
    p.n_ode_steps = 0;  // default = 8
    p.cfg_alpha = 0.0f; // default = 1.2
    return p;
}

extern "C" voxtral_tts_context* voxtral_tts_init_from_file(const char* path_model, voxtral_tts_context_params params) {
    if (!path_model || !*path_model)
        return nullptr;
    auto* ctx = new voxtral_tts_context();
    ctx->params = params;
    ctx->verbosity = params.verbosity;
    auto& hp = ctx->hp;

    // Open GGUF metadata
    gguf_context* gctx = core_gguf::open_metadata(path_model);
    if (!gctx) {
        fprintf(stderr, "voxtral_tts: failed to open '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }

    // Read hyperparameters
    hp.llm_n_layers = core_gguf::kv_u32(gctx, "voxtral_tts.llm.n_layers", hp.llm_n_layers);
    hp.llm_dim = core_gguf::kv_u32(gctx, "voxtral_tts.llm.dim", hp.llm_dim);
    hp.llm_n_heads = core_gguf::kv_u32(gctx, "voxtral_tts.llm.n_heads", hp.llm_n_heads);
    hp.llm_n_kv = core_gguf::kv_u32(gctx, "voxtral_tts.llm.n_kv_heads", hp.llm_n_kv);
    hp.llm_head_dim = core_gguf::kv_u32(gctx, "voxtral_tts.llm.head_dim", hp.llm_head_dim);
    hp.llm_ff_dim = core_gguf::kv_u32(gctx, "voxtral_tts.llm.hidden_dim", hp.llm_ff_dim);
    hp.llm_rope_theta = core_gguf::kv_f32(gctx, "voxtral_tts.llm.rope_theta", hp.llm_rope_theta);
    hp.llm_norm_eps = core_gguf::kv_f32(gctx, "voxtral_tts.llm.norm_eps", hp.llm_norm_eps);
    hp.llm_vocab_size = core_gguf::kv_u32(gctx, "voxtral_tts.llm.vocab_size", hp.llm_vocab_size);
    hp.tied_embeddings = core_gguf::kv_bool(gctx, "voxtral_tts.llm.tied_embeddings", hp.tied_embeddings);

    hp.fm_n_layers = core_gguf::kv_u32(gctx, "voxtral_tts.fm.n_layers", hp.fm_n_layers);
    hp.fm_dim = core_gguf::kv_u32(gctx, "voxtral_tts.fm.dim", hp.fm_dim);
    hp.fm_n_heads = core_gguf::kv_u32(gctx, "voxtral_tts.fm.n_heads", hp.fm_n_heads);
    hp.fm_n_kv = core_gguf::kv_u32(gctx, "voxtral_tts.fm.n_kv_heads", hp.fm_n_kv);
    hp.fm_head_dim = core_gguf::kv_u32(gctx, "voxtral_tts.fm.head_dim", hp.fm_head_dim);
    hp.fm_ff_dim = core_gguf::kv_u32(gctx, "voxtral_tts.fm.hidden_dim", hp.fm_ff_dim);
    hp.fm_rope_theta = core_gguf::kv_f32(gctx, "voxtral_tts.fm.rope_theta", hp.fm_rope_theta);
    hp.fm_sigma = core_gguf::kv_f32(gctx, "voxtral_tts.fm.sigma", hp.fm_sigma);
    hp.fm_sigma_max = core_gguf::kv_f32(gctx, "voxtral_tts.fm.sigma_max", hp.fm_sigma_max);

    hp.semantic_cb_size = core_gguf::kv_u32(gctx, "voxtral_tts.semantic_codebook_size", hp.semantic_cb_size);
    hp.acoustic_cb_size = core_gguf::kv_u32(gctx, "voxtral_tts.acoustic_codebook_size", hp.acoustic_cb_size);
    hp.n_acoustic_cb = core_gguf::kv_u32(gctx, "voxtral_tts.n_acoustic_codebook", hp.n_acoustic_cb);
    hp.n_codebooks = core_gguf::kv_u32(gctx, "voxtral_tts.n_codebooks", hp.n_codebooks);
    hp.sample_rate = core_gguf::kv_u32(gctx, "voxtral_tts.sample_rate", hp.sample_rate);
    hp.frame_rate = core_gguf::kv_f32(gctx, "voxtral_tts.frame_rate", hp.frame_rate);

    hp.audio_token_id = core_gguf::kv_u32(gctx, "voxtral_tts.audio_token_id", hp.audio_token_id);
    hp.begin_audio_token_id = core_gguf::kv_u32(gctx, "voxtral_tts.begin_audio_token_id", hp.begin_audio_token_id);
    hp.cond_dropped_token_id =
        core_gguf::kv_u32(gctx, "voxtral_tts.condition_dropped_token_id", hp.cond_dropped_token_id);
    hp.bos_token_id = core_gguf::kv_u32(gctx, "voxtral_tts.bos_token_id", hp.bos_token_id);

    hp.codec_dim = core_gguf::kv_u32(gctx, "voxtral_tts.codec.dim", hp.codec_dim);
    hp.codec_hidden_dim = core_gguf::kv_u32(gctx, "voxtral_tts.codec.hidden_dim", hp.codec_hidden_dim);
    hp.codec_n_heads = core_gguf::kv_u32(gctx, "voxtral_tts.codec.n_heads", hp.codec_n_heads);
    hp.codec_n_kv = core_gguf::kv_u32(gctx, "voxtral_tts.codec.n_kv_heads", hp.codec_n_kv);
    hp.codec_head_dim = core_gguf::kv_u32(gctx, "voxtral_tts.codec.head_dim", hp.codec_head_dim);
    hp.codec_semantic_dim = core_gguf::kv_u32(gctx, "voxtral_tts.codec.semantic_dim", hp.codec_semantic_dim);
    hp.codec_acoustic_dim = core_gguf::kv_u32(gctx, "voxtral_tts.codec.acoustic_dim", hp.codec_acoustic_dim);
    hp.codec_patch_size = core_gguf::kv_u32(gctx, "voxtral_tts.codec.patch_size", hp.codec_patch_size);
    hp.codec_patch_kernel = core_gguf::kv_u32(gctx, "voxtral_tts.codec.patch_proj_kernel", hp.codec_patch_kernel);
    hp.codec_attn_window = core_gguf::kv_u32(gctx, "voxtral_tts.codec.attn_window", hp.codec_attn_window);
    hp.codec_norm_eps = core_gguf::kv_f32(gctx, "voxtral_tts.codec.norm_eps", hp.codec_norm_eps);
    hp.codec_qk_norm_eps = core_gguf::kv_f32(gctx, "voxtral_tts.codec.qk_norm_eps", hp.codec_qk_norm_eps);
    hp.codec_qk_norm = core_gguf::kv_bool(gctx, "voxtral_tts.codec.qk_norm", hp.codec_qk_norm);
    hp.codec_layer_scale = core_gguf::kv_bool(gctx, "voxtral_tts.codec.layer_scale", hp.codec_layer_scale);

    std::string strides_str = core_gguf::kv_str(gctx, "voxtral_tts.codec.conv_strides", "1,2,2,2");
    std::string kernels_str = core_gguf::kv_str(gctx, "voxtral_tts.codec.conv_kernels", "3,4,4,4");
    std::string tfm_lens_str = core_gguf::kv_str(gctx, "voxtral_tts.codec.tfm_lengths", "2,2,2,2");
    hp.codec_conv_strides = parse_int_list(strides_str);
    hp.codec_conv_kernels = parse_int_list(kernels_str);
    hp.codec_tfm_lengths = parse_int_list(tfm_lens_str);

    // Voice names
    auto voice_names = core_gguf::kv_str_array(gctx, "voxtral_tts.voice_names");
    ctx->voice_names = voice_names;
    for (auto& vn : ctx->voice_names) {
        ctx->voice_name_ptrs.push_back(vn.c_str());
    }
    ctx->voice_name_ptrs.push_back(nullptr);

    // Tekken tokenizer
    ctx->vocab.pre_pattern = core_gguf::kv_str(gctx, "tokenizer.tekken.pattern", "");
    ctx->vocab.specials = core_gguf::kv_str_array(gctx, "tokenizer.tekken.specials");
    ctx->vocab.n_specials = core_gguf::kv_u32(gctx, "tokenizer.tekken.n_specials", 1000);
    ctx->vocab.n_vocab = core_gguf::kv_u32(gctx, "tokenizer.tekken.n_vocab", 150000);

    core_gguf::free_metadata(gctx);

    // Load weights
    ggml_backend_t be = core_cpu_backend::init();
    ctx->backend_cpu = be;
    if (params.use_gpu) {
        ggml_backend_t gpu = crispasr_init_gpu_backend();
        if (gpu) {
            ctx->backend = gpu;
            be = gpu;
        }
    }
    if (!ctx->backend)
        ctx->backend = be;

    if (ctx->backend_cpu && params.n_threads > 0)
        core_cpu_backend::set_n_threads(ctx->backend_cpu, params.n_threads);

    // FM_STEPS: experimental Euler ODE step-count override (default 8). See header.
    ctx->fm_flow_steps = env_int("CRISPASR_VOXTRAL_TTS_FM_STEPS", VTTS_FLOW_STEPS);

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, ctx->backend, "voxtral_tts", wl)) {
        fprintf(stderr, "voxtral_tts: failed to load weights from '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }
    ctx->ctx_w = wl.ctx;
    ctx->buf = wl.buf;
    if (ctx->verbosity >= 1 && ctx->fm_flow_steps != VTTS_FLOW_STEPS)
        fprintf(stderr, "voxtral_tts: FM ode_steps=%d (non-default)\n", ctx->fm_flow_steps);

    // Compute scheduler (weighted-model inference; sched handles weight + compute
    // buffers across the GPU/CPU split). 16384-node graph budget covers the 26-layer
    // LLM prefill plus per-frame FM/codec graphs.
    {
        int n_be = 0;
        ggml_backend_t backends[2];
        backends[n_be++] = ctx->backend;
        if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
            backends[n_be++] = ctx->backend_cpu;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
    }
    ctx->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

    auto get = [&](const std::string& name) -> ggml_tensor* {
        auto it = wl.tensors.find(name);
        if (it == wl.tensors.end())
            return nullptr;
        return it->second;
    };

    // Bind LLM weights
    ctx->model.token_embd = get("token_embd.weight");
    ctx->model.audio_embd = get("audio_embd.weight");
    ctx->model.output_norm = get("output_norm.weight");
    ctx->model.llm_layers.resize(hp.llm_n_layers);
    for (int i = 0; i < hp.llm_n_layers; i++) {
        auto& l = ctx->model.llm_layers[i];
        auto b = [&](const std::string& s) { return get("blk." + std::to_string(i) + "." + s); };
        l.attn_norm = b("attn_norm.weight");
        l.attn_q = b("attn_q.weight");
        l.attn_k = b("attn_k.weight");
        l.attn_v = b("attn_v.weight");
        l.attn_o = b("attn_output.weight");
        l.ffn_norm = b("ffn_norm.weight");
        l.ffn_gate = b("ffn_gate.weight");
        l.ffn_up = b("ffn_up.weight");
        l.ffn_down = b("ffn_down.weight");
    }

    // Bind FM weights
    ctx->model.fm_input_proj = get("fm.input_proj.weight");
    ctx->model.fm_llm_proj = get("fm.llm_proj.weight");
    ctx->model.fm_time_proj = get("fm.time_proj.weight");
    ctx->model.fm_semantic_output = get("fm.semantic_output.weight");
    ctx->model.fm_semantic_output_bias = get("fm.semantic_output.bias"); // optional
    ctx->model.fm_acoustic_output = get("fm.acoustic_output.weight");
    ctx->model.fm_norm = get("fm.norm.weight");
    ctx->model.fm_layers.resize(hp.fm_n_layers);
    for (int i = 0; i < hp.fm_n_layers; i++) {
        auto& l = ctx->model.fm_layers[i];
        auto b = [&](const std::string& s) { return get("fm.blk." + std::to_string(i) + "." + s); };
        l.attn_norm = b("attn_norm.weight");
        l.attn_q = b("attn_q.weight");
        l.attn_k = b("attn_k.weight");
        l.attn_v = b("attn_v.weight");
        l.attn_o = b("attn_output.weight");
        l.ffn_norm = b("ffn_norm.weight");
        l.ffn_gate = b("ffn_gate.weight");
        l.ffn_up = b("ffn_up.weight");
        l.ffn_down = b("ffn_down.weight");
    }

    // Bind codec weights
    int n_conv_blocks = (int)hp.codec_conv_strides.size();
    ctx->model.codec_convs.resize(n_conv_blocks);
    for (int i = 0; i < n_conv_blocks; i++) {
        auto pfx = "codec.dec.conv." + std::to_string(i);
        ctx->model.codec_convs[i].weight = get(pfx + ".weight");
        ctx->model.codec_convs[i].bias = get(pfx + ".bias");
    }

    int n_tfm_blocks = (int)hp.codec_tfm_lengths.size();
    ctx->model.codec_tfm_blocks.resize(n_tfm_blocks);
    for (int bi = 0; bi < n_tfm_blocks; bi++) {
        int n_layers = hp.codec_tfm_lengths[bi];
        ctx->model.codec_tfm_blocks[bi].resize(n_layers);
        for (int li = 0; li < n_layers; li++) {
            auto pfx = "codec.dec.tfm." + std::to_string(bi) + ".blk." + std::to_string(li);
            auto& l = ctx->model.codec_tfm_blocks[bi][li];
            l.attn_norm = get(pfx + ".attn_norm.weight");
            l.attn_q = get(pfx + ".attn_q.weight");
            l.attn_k = get(pfx + ".attn_k.weight");
            l.attn_v = get(pfx + ".attn_v.weight");
            l.attn_o = get(pfx + ".attn_o.weight");
            l.q_norm = get(pfx + ".q_norm.weight");
            l.k_norm = get(pfx + ".k_norm.weight");
            l.ffn_norm = get(pfx + ".ffn_norm.weight");
            l.ffn_gate = get(pfx + ".ffn_gate.weight");
            l.ffn_up = get(pfx + ".ffn_up.weight");
            l.ffn_down = get(pfx + ".ffn_down.weight");
            l.attn_scale = get(pfx + ".attn_scale");
            l.ffn_scale = get(pfx + ".ffn_scale");
        }
    }
    ctx->model.codec_output.weight = get("codec.output.weight");
    ctx->model.codec_output.bias = get("codec.output.bias");
    ctx->model.codec_patch_proj.weight = get("codec.patch_proj.weight");
    ctx->model.codec_patch_proj.bias = get("codec.patch_proj.bias");
    ctx->model.codec_semantic_cb = get("codec.semantic_cb.weight");

    // Load the semantic VQ codebook into host memory (8192 × 256). Prefer the
    // GGUF tensor; fall back to a side-load file for GGUFs that dropped it.
    {
        const int rows = hp.semantic_cb_size, cols = hp.codec_semantic_dim;
        if (ctx->model.codec_semantic_cb) {
            ctx->semantic_cb_host.resize((size_t)rows * cols);
            ggml_backend_tensor_get(ctx->model.codec_semantic_cb, ctx->semantic_cb_host.data(), 0,
                                    ctx->semantic_cb_host.size() * sizeof(float));
        } else if (const char* p = std::getenv("CRISPASR_VOXTRAL_TTS_SEMANTIC_CB")) {
            FILE* fp = fopen(p, "rb");
            if (fp) {
                ctx->semantic_cb_host.resize((size_t)rows * cols);
                size_t n = fread(ctx->semantic_cb_host.data(), sizeof(float), ctx->semantic_cb_host.size(), fp);
                fclose(fp);
                if (n != ctx->semantic_cb_host.size()) {
                    fprintf(stderr, "voxtral_tts: semantic_cb side-load short read (%zu/%zu)\n", n,
                            ctx->semantic_cb_host.size());
                    ctx->semantic_cb_host.clear();
                } else if (ctx->verbosity >= 1) {
                    fprintf(stderr, "voxtral_tts: semantic_cb side-loaded from %s\n", p);
                }
            }
        }
    }

    // Bind voice embeddings
    for (auto& [name, tensor] : wl.tensors) {
        if (name.substr(0, 6) == "voice.") {
            ctx->model.voice_tensors[name.substr(6)] = tensor;
        }
    }

    // Load Tekken vocab blob
    {
        ggml_tensor* vt = get("tokenizer.tekken.vocab_tensor");
        if (vt) {
            int n = (int)ggml_nelements(vt);
            std::vector<float> tmp(n);
            ggml_backend_tensor_get(vt, tmp.data(), 0, n * sizeof(float));
            ctx->vocab.tekken_vocab_blob.resize(n);
            for (int i = 0; i < n; i++) {
                ctx->vocab.tekken_vocab_blob[i] = (uint8_t)(int)tmp[i];
            }
        }
    }
    tekken_build_vocab(ctx->vocab, (int)hp.llm_vocab_size, ctx->verbosity);

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "voxtral_tts: LLM %dL d=%d heads=%d/%d\n", hp.llm_n_layers, hp.llm_dim, hp.llm_n_heads,
                hp.llm_n_kv);
        fprintf(stderr, "voxtral_tts: FM  %dL d=%d heads=%d/%d rope_theta=%.0f\n", hp.fm_n_layers, hp.fm_dim,
                hp.fm_n_heads, hp.fm_n_kv, hp.fm_rope_theta);
        fprintf(stderr, "voxtral_tts: Codec d=%d semantic_cb=%d acoustic_fsq=%d×%d\n", hp.codec_dim,
                hp.semantic_cb_size, hp.acoustic_cb_size, hp.n_acoustic_cb);
        fprintf(stderr, "voxtral_tts: %d voices, %d tokens loaded\n", (int)ctx->model.voice_tensors.size(),
                (int)ctx->vocab.id_to_piece.size());
        fprintf(stderr, "voxtral_tts: loaded '%s'\n", path_model);
    }

    return ctx;
}

// ---------------------------------------------------------------------------
// KV cache
// ---------------------------------------------------------------------------

static bool voxtral_tts_kv_init(voxtral_tts_context* ctx, int max_ctx) {
    if (ctx->kv_k)
        return true;
    const auto& hp = ctx->hp;
    ggml_init_params ip = {2 * ggml_tensor_overhead(), nullptr, true};
    ctx->kv_ctx = ggml_init(ip);
    const auto kv_pair = core_attn::kv_dtype_pair_from_env("voxtral_tts");
    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, kv_pair.k, hp.llm_head_dim, max_ctx, hp.llm_n_kv, hp.llm_n_layers);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, kv_pair.v, hp.llm_head_dim, max_ctx, hp.llm_n_kv, hp.llm_n_layers);
    ggml_backend_t kv_be = core_attn::kv_backend_from_env(ctx->backend, ctx->backend_cpu, "voxtral_tts");
    ctx->kv_buf = ggml_backend_alloc_ctx_tensors(ctx->kv_ctx, kv_be);
    if (!ctx->kv_buf) {
        fprintf(stderr, "voxtral_tts: kv cache alloc failed (max_ctx=%d)\n", max_ctx);
        return false;
    }
    ggml_backend_buffer_clear(ctx->kv_buf, 0);
    if (ctx->verbosity >= 1) {
        size_t total = ggml_nbytes(ctx->kv_k) + ggml_nbytes(ctx->kv_v);
        fprintf(stderr, "voxtral_tts: kv cache %.0f MiB (max_ctx=%d)\n", total / 1048576.0, max_ctx);
    }
    return true;
}

static void voxtral_tts_kv_reset(voxtral_tts_context* ctx) {
    if (ctx->kv_buf)
        ggml_backend_buffer_clear(ctx->kv_buf, 0);
    ctx->kv_used = 0;
}

// ---------------------------------------------------------------------------
// Prompt embedding: [voice frames] ++ [text token embeddings]
// ---------------------------------------------------------------------------
// Returns a (dim × T_prompt) column-major F32 buffer (frame-contiguous). Voice
// frames are the pre-summed conditioning embeddings stored in the GGUF; text
// tokens are embedded via token_embd. `ggml_get_rows` dequantises any quant
// type to F32, so this is robust whether the GGUF is F16 or Q4_K.
// mistral_common encode_speech_request framing. The voice AUDIO-token positions
// are replaced by the pre-summed voice embeddings, so the token-id stream splits
// into a prefix ([BOS][BEGIN_AUDIO]) and a suffix ([/INST] text [INST][BEGIN_AUDIO])
// with the voice frames spliced between. IDs match the reference C port
// (github.com/mudler/voxtral-tts.c): /INST=36, [INST]=35.
static constexpr int32_t VTTS_TOK_INST_END = 36; // [/INST]
static constexpr int32_t VTTS_TOK_INST = 35;     // [INST]

static std::vector<float> voxtral_tts_build_prompt_embeds(voxtral_tts_context* ctx, ggml_tensor* voice_t,
                                                          const std::vector<int32_t>& text_ids, int* out_T_prompt) {
    const int d = ctx->hp.llm_dim;
    const int T_voice = (int)voice_t->ne[1];
    const int32_t bos = ctx->hp.bos_token_id;
    const int32_t begin_audio = ctx->hp.begin_audio_token_id;

    // Prompt = [BOS][BEGIN_AUDIO] [voice×N] [/INST] text [INST][BEGIN_AUDIO]
    std::vector<int32_t> prefix_ids = {bos, begin_audio};
    std::vector<int32_t> suffix_ids;
    suffix_ids.push_back(VTTS_TOK_INST_END);
    suffix_ids.insert(suffix_ids.end(), text_ids.begin(), text_ids.end());
    suffix_ids.push_back(VTTS_TOK_INST);
    suffix_ids.push_back(begin_audio);

    const int n_pre = (int)prefix_ids.size();
    const int n_suf = (int)suffix_ids.size();
    const int T_prompt = n_pre + T_voice + n_suf;
    if (out_T_prompt)
        *out_T_prompt = T_prompt;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph(ctx0);

    ggml_tensor* pre_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pre);
    ggml_set_name(pre_idx, "pre_idx");
    ggml_set_input(pre_idx);
    ggml_tensor* pre_emb = ggml_get_rows(ctx0, ctx->model.token_embd, pre_idx);

    // Voice frames: identity get_rows dequantises (D, T_voice) → F32.
    ggml_tensor* v_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T_voice);
    ggml_set_name(v_idx, "v_idx");
    ggml_set_input(v_idx);
    ggml_tensor* voice_f32 = ggml_get_rows(ctx0, voice_t, v_idx);

    ggml_tensor* suf_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_suf);
    ggml_set_name(suf_idx, "suf_idx");
    ggml_set_input(suf_idx);
    ggml_tensor* suf_emb = ggml_get_rows(ctx0, ctx->model.token_embd, suf_idx);

    ggml_tensor* out = ggml_concat(ctx0, pre_emb, voice_f32, 1); // along time (ne[1])
    out = ggml_concat(ctx0, out, suf_emb, 1);
    ggml_set_name(out, "prompt_embeds");
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        ggml_free(ctx0);
        return {};
    }

    std::vector<int32_t> vidx(T_voice);
    for (int i = 0; i < T_voice; i++)
        vidx[i] = i;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pre_idx"), prefix_ids.data(), 0, n_pre * sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "v_idx"), vidx.data(), 0, vidx.size() * sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "suf_idx"), suffix_ids.data(), 0, n_suf * sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx0);
        return {};
    }

    std::vector<float> buf((size_t)d * T_prompt);
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "prompt_embeds"), buf.data(), 0, buf.size() * sizeof(float));
    ggml_free(ctx0);
    return buf;
}

// Embed a list of token ids via token_embd → (dim × n) F32 (dequantises any quant).
static std::vector<float> voxtral_tts_embed_ids(voxtral_tts_context* ctx, const std::vector<int32_t>& ids) {
    const int d = ctx->hp.llm_dim;
    const int n = (int)ids.size();
    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph(ctx0);
    ggml_tensor* idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n);
    ggml_set_name(idx, "idx");
    ggml_set_input(idx);
    ggml_tensor* emb = ggml_get_rows(ctx0, ctx->model.token_embd, idx);
    ggml_set_name(emb, "emb");
    ggml_set_output(emb);
    ggml_build_forward_expand(gf, emb);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        ggml_free(ctx0);
        return {};
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "idx"), ids.data(), 0, n * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx0);
        return {};
    }
    std::vector<float> buf((size_t)d * n);
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "emb"), buf.data(), 0, buf.size() * sizeof(float));
    ggml_free(ctx0);
    return buf;
}

// ---------------------------------------------------------------------------
// LLM AR backbone (Ministral-3B) — KV-cached GQA + SwiGLU, NEOX RoPE.
// ---------------------------------------------------------------------------
// Architecture is identical to voxtral4b's LLM (mirror of voxtral4b_build_graph_llm_kv)
// with two differences: (1) no ada_rms_norm time conditioning, and (2) the graph
// outputs the hidden state after the final RMSNorm+output_norm (the FM head's
// input), NOT the tied-embedding logits.
static ggml_cgraph* voxtral_tts_build_graph_llm(voxtral_tts_context* ctx, int n_past, int n_tokens) {
    const auto& hp = ctx->hp;
    const auto& m = ctx->model;
    const int d = hp.llm_dim;
    const int n_q = hp.llm_n_heads;
    const int n_kv = hp.llm_n_kv;
    const int hd = hp.llm_head_dim;
    const int n_layers = hp.llm_n_layers;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, n_tokens);
    ggml_set_name(embeds, "inputs_embeds");
    ggml_set_input(embeds);

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    ggml_tensor* causal_mask = nullptr;
    if (n_tokens > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_past + n_tokens, n_tokens);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    ggml_tensor* cur = embeds;

    const core_attn::KvSelfAttnParams kvp = {
        /*n_heads*/ n_q,
        /*n_kv_heads*/ n_kv,
        /*head_dim*/ hd,
        /*n_kv_grp*/ n_q / n_kv,
        /*n_ctx_orig*/ 0,
        /*rope_theta*/ hp.llm_rope_theta,
        /*rope_beta_fast*/ 0.0f,
        /*rope_beta_slow*/ 0.0f,
        /*attn_scale*/ 1.0f / std::sqrt((float)hd),
        /*qk_norm_eps*/ 0.0f,
        /*gqa_mode*/ core_attn::GQA_MANUAL_NOCONT,
        // Raw Mistral `consolidated` weights use adjacent-pair (GPT-J / NORMAL)
        // RoPE and the converter does NOT permute Q/K for NEOX, so the runtime
        // must apply NORMAL RoPE (matches the reference tts_apply_rope). Using
        // NEOX here rotates the wrong dim pairs → semantically wrong hidden state
        // (right magnitude, wrong direction) → degenerate/non-terminating decode.
        /*rope_type*/ GGML_ROPE_TYPE_NORMAL,
    };

    for (int il = 0; il < n_layers; il++) {
        const auto& b = m.llm_layers[il];
        ggml_tensor* residual = cur;

        cur = ggml_rms_norm(ctx0, cur, hp.llm_norm_eps);
        cur = ggml_mul(ctx0, cur, b.attn_norm);

        ggml_tensor* attn = core_attn::kv_self_attn(ctx0, gf, cur, b.attn_q, b.attn_k, b.attn_v, b.attn_o,
                                                    /*q_norm_w*/ nullptr, /*k_norm_w*/ nullptr, positions, causal_mask,
                                                    ctx->kv_k, ctx->kv_v, il, n_past, kvp);
        cur = ggml_add(ctx0, residual, attn);

        residual = cur;
        cur = ggml_rms_norm(ctx0, cur, hp.llm_norm_eps);
        cur = ggml_mul(ctx0, cur, b.ffn_norm);
        ggml_tensor* ffn = core_ffn::swiglu(ctx0, cur, b.ffn_gate, b.ffn_up, b.ffn_down);
        cur = ggml_add(ctx0, residual, ffn);

        // Per-stage diff harness: expose every LLM layer output for a per-layer cos
        // comparison vs the reference (CRISPASR_VOXTRAL_TTS_DIFF_DUMP). set_output is
        // required or ggml reuses the buffer and later layers overwrite it.
        if (std::getenv("CRISPASR_VOXTRAL_TTS_DIFF_DUMP")) {
            char nm[24];
            snprintf(nm, sizeof(nm), "llm_L%d", il);
            ggml_set_name(cur, nm);
            ggml_set_output(cur);
        }
    }

    // Final RMSNorm + output_norm → hidden state (FM head input, not logits).
    cur = ggml_rms_norm(ctx0, cur, hp.llm_norm_eps);
    cur = ggml_mul(ctx0, cur, m.output_norm);
    ggml_set_name(cur, "hidden");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// Run the LLM over `n_tokens` prompt embeddings starting at `n_past`, writing K/V
// into the cache. Returns the LAST position's hidden state (dim floats) — the
// conditioning `h` for the first FM frame. Empty vector on failure.
static std::vector<float> voxtral_tts_run_llm(voxtral_tts_context* ctx, const float* embeds, int n_tokens, int n_past) {
    const int d = ctx->hp.llm_dim;

    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        positions[i] = n_past + i;

    std::vector<ggml_fp16_t> mask;
    if (n_tokens > 1) {
        const int Lk = n_past + n_tokens;
        mask.resize((size_t)n_tokens * Lk, ggml_fp32_to_fp16(0.0f));
        const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < n_tokens; q++)
            for (int k = 0; k < Lk; k++)
                if (k > n_past + q)
                    mask[(size_t)q * Lk + k] = ninf;
    }

    ggml_cgraph* gf = voxtral_tts_build_graph_llm(ctx, n_past, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
        return {};

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inputs_embeds"), embeds, 0,
                            (size_t)d * n_tokens * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "positions"), positions.data(), 0,
                            positions.size() * sizeof(int32_t));
    if (n_tokens > 1)
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "causal_mask"), mask.data(), 0,
                                mask.size() * sizeof(ggml_fp16_t));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
        return {};

    ggml_tensor* h = ggml_graph_get_tensor(gf, "hidden");
    std::vector<float> out(d);
    ggml_backend_tensor_get(h, out.data(), (size_t)(n_tokens - 1) * d * sizeof(float), (size_t)d * sizeof(float));

    // Per-stage diff harness: dump the LAST-position d-vector of the input embed +
    // every LLM layer + the final hidden, per run_llm call, as raw f32. call 0 =
    // prompt prefill, call 1 = frame-0 (h0) decode, etc. Compared per stage vs the
    // reference dumps (same layout) to find the first divergent layer.
    if (const char* dd = std::getenv("CRISPASR_VOXTRAL_TTS_DIFF_DUMP")) {
        static int call = 0;
        auto dump = [&](const char* stage, ggml_tensor* t) {
            if (!t)
                return;
            std::vector<float> v(d);
            ggml_backend_tensor_get(t, v.data(), (size_t)(n_tokens - 1) * d * sizeof(float), (size_t)d * sizeof(float));
            char p[512];
            snprintf(p, sizeof(p), "%s/mine.c%d.%s.bin", dd, call, stage);
            if (FILE* fp = fopen(p, "wb")) {
                fwrite(v.data(), sizeof(float), d, fp);
                fclose(fp);
            }
        };
        // "embed": write the RAW input, not the post-compute inputs_embeds tensor —
        // its buffer is reused as scratch after layer 0 (gallocr/sched), so reading it
        // back here yields llm_L0's value. `embeds` is the caller's untouched input.
        {
            char p[512];
            snprintf(p, sizeof(p), "%s/mine.c%d.embed.bin", dd, call);
            if (FILE* fp = fopen(p, "wb")) {
                fwrite(embeds + (size_t)(n_tokens - 1) * d, sizeof(float), d, fp);
                fclose(fp);
            }
        }
        for (int il = 0; il < ctx->hp.llm_n_layers; il++) {
            char nm[24];
            snprintf(nm, sizeof(nm), "llm_L%d", il);
            dump(nm, ggml_graph_get_tensor(gf, nm));
        }
        dump("hidden", h);
        call++;
    }
    return out;
}

// ===========================================================================
// Stage 2 — flow-matching acoustic transformer
// ===========================================================================
// Blueprint: github.com/mudler/voxtral-tts.c (MIT, validated vs vLLM-Omni). Per
// frame: a semantic token (greedy argmax of fm.semantic_output @ h) plus 36
// acoustic FSQ codes from an 8-timestep (7-interval) Euler flow-matching ODE
// with classifier-free guidance (α=1.2). The FM transformer runs over exactly 3
// tokens [input_proj(x_t), time_proj(time_emb(t)), llm_proj(h)] and is
// bidirectional with NO positional encoding.

static constexpr float VTTS_CFG_ALPHA = 1.2f;
static constexpr float VTTS_NOISE_SCALE = 1.0f;
static constexpr int VTTS_FSQ_LEVELS = 21;
static constexpr int VTTS_ACOUSTIC_DIM = 36;
static constexpr int VTTS_SPECIAL_COUNT = 2; // [EMPTY]=0, [END]=1
static constexpr int VTTS_AUDIO_END = 1;
static constexpr int VTTS_AUDIO_EMPTY = 0;

// cos-first sinusoidal time embedding: [cos(t·invf) ; sin(t·invf)],
// invf[i] = exp(-log(1e4)·i/(d/2)).
static std::vector<float> vtts_fm_time_embed(int dim, float t) {
    std::vector<float> e(dim);
    const int half = dim / 2;
    for (int i = 0; i < half; i++) {
        float invf = std::exp(-std::log(10000.0f) * (float)i / (float)half);
        float a = t * invf;
        e[i] = std::cos(a);
        e[half + i] = std::sin(a);
    }
    return e;
}

// Computes BOTH the conditional (h) and unconditional (h=0) FM velocities in ONE
// graph: a 6-token sequence [cond: input_proj(x), time_proj(t), llm_proj(h);
// uncond: input_proj(x), time_proj(t), 0] with a block-diagonal mask so the two
// 3-token CFG groups don't attend to each other — mathematically identical to
// running them separately. This halves the per-frame graph dispatch count (the
// FM is 14 tiny dispatches/frame and dominates generation time). The fixed-shape
// graph is built + allocated once (dedicated gallocr, single backend) and reused
// every eval. llm_proj(0)==0, so the uncond third token is the cond one scaled by
// 0. Velocities come from token 0 (cond) and token 3 (uncond).
static bool vtts_fm_predict_velocity_cfg(voxtral_tts_context* ctx, const float* x_t, const float* h, float t,
                                         std::vector<float>& v_cond, std::vector<float>& v_uncond) {
    const auto& hp = ctx->hp;
    const auto& m = ctx->model;
    const int d = hp.fm_dim;
    const int n_q = hp.fm_n_heads, n_kv = hp.fm_n_kv, hd = hp.fm_head_dim, grp = n_q / n_kv;
    const int T = 6; // 2 CFG groups × 3 tokens
    const float scale = 1.0f / std::sqrt((float)hd);
    std::vector<float> time_emb = vtts_fm_time_embed(d, t);

    if (!ctx->fm_gf) {
        if (ctx->fm_meta.empty())
            ctx->fm_meta.resize(ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(4096, false));
        ggml_init_params ip = {ctx->fm_meta.size(), ctx->fm_meta.data(), true};
        ctx->fm_ctx = ggml_init(ip);
        ggml_context* ctx0 = ctx->fm_ctx;
        ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

        ggml_tensor* xt = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, VTTS_ACOUSTIC_DIM);
        ggml_set_name(xt, "xt");
        ggml_set_input(xt);
        ggml_tensor* hin = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, d);
        ggml_set_name(hin, "hin");
        ggml_set_input(hin);
        ggml_tensor* tin = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, d);
        ggml_set_name(tin, "tin");
        ggml_set_input(tin);
        ggml_tensor* mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, T, T); // (kv, q) block-diagonal
        ggml_set_name(mask, "mask");
        ggml_set_input(mask);
        ggml_tensor* velidx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 2); // tokens {0, 3}
        ggml_set_name(velidx, "velidx");
        ggml_set_input(velidx);

        ggml_tensor* tok0 = ggml_reshape_2d(ctx0, ggml_mul_mat(ctx0, m.fm_input_proj, xt), d, 1);
        ggml_tensor* tok1 = ggml_reshape_2d(ctx0, ggml_mul_mat(ctx0, m.fm_time_proj, tin), d, 1);
        ggml_tensor* tok2c = ggml_reshape_2d(ctx0, ggml_mul_mat(ctx0, m.fm_llm_proj, hin), d, 1);
        ggml_tensor* tok2u = ggml_scale(ctx0, tok2c, 0.0f); // llm_proj(0) == 0
        // [c0, c1, c2, u0, u1, u2]  (u0==c0, u1==c1)
        ggml_tensor* cur = ggml_concat(ctx0, tok0, tok1, 1);
        cur = ggml_concat(ctx0, cur, tok2c, 1);
        cur = ggml_concat(ctx0, cur, tok0, 1);
        cur = ggml_concat(ctx0, cur, tok1, 1);
        cur = ggml_concat(ctx0, cur, tok2u, 1); // (d, 6)

        for (int il = 0; il < hp.fm_n_layers; il++) {
            const auto& l = m.fm_layers[il];
            ggml_tensor* res = cur;
            ggml_tensor* x = ggml_mul(ctx0, ggml_rms_norm(ctx0, cur, hp.llm_norm_eps), l.attn_norm);

            ggml_tensor* Q = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, l.attn_q, x), hd, n_q, T);
            ggml_tensor* K = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, l.attn_k, x), hd, n_kv, T);
            ggml_tensor* V = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, l.attn_v, x), hd, n_kv, T);

            // GQA interleave K,V to n_q heads (each kv head repeated grp times contiguously)
            K = ggml_reshape_4d(ctx0, K, hd, 1, n_kv, T);
            K = ggml_repeat(ctx0, K, ggml_new_tensor_4d(ctx0, K->type, hd, grp, n_kv, T));
            K = ggml_reshape_3d(ctx0, K, hd, n_q, T);
            V = ggml_reshape_4d(ctx0, V, hd, 1, n_kv, T);
            V = ggml_repeat(ctx0, V, ggml_new_tensor_4d(ctx0, V->type, hd, grp, n_kv, T));
            V = ggml_reshape_3d(ctx0, V, hd, n_q, T);

            Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));      // (hd, T, nh)
            K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));      // (hd, T, nh)
            ggml_tensor* scores = ggml_mul_mat(ctx0, K, Q);              // (T, T, nh)
            scores = ggml_soft_max_ext(ctx0, scores, mask, scale, 0.0f); // block-diagonal (per CFG group)
            V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));      // (hd, T, nh)
            V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 0, 2, 3));      // (T, hd, nh)
            ggml_tensor* attn = ggml_mul_mat(ctx0, V, scores);           // (hd, T, nh)
            attn = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, attn, 0, 2, 1, 3)), n_q * hd, T);
            cur = ggml_add(ctx0, res, ggml_mul_mat(ctx0, l.attn_o, attn));

            res = cur;
            x = ggml_mul(ctx0, ggml_rms_norm(ctx0, cur, hp.llm_norm_eps), l.ffn_norm);
            cur = ggml_add(ctx0, res, core_ffn::swiglu(ctx0, x, l.ffn_gate, l.ffn_up, l.ffn_down));
        }

        // Velocities from token 0 (cond) and token 3 (uncond): final RMSNorm + fm.norm + acoustic head.
        ggml_tensor* t03 = ggml_get_rows(ctx0, cur, velidx); // (d, 2)
        t03 = ggml_mul(ctx0, ggml_rms_norm(ctx0, t03, hp.llm_norm_eps), m.fm_norm);
        ggml_tensor* vel = ggml_mul_mat(ctx0, m.fm_acoustic_output, t03); // (36, 2)
        ggml_set_name(vel, "vel");
        ggml_set_output(vel);
        ggml_build_forward_expand(gf, vel);

        ctx->fm_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
        if (!ctx->fm_alloc || !ggml_gallocr_alloc_graph(ctx->fm_alloc, gf)) {
            fprintf(stderr, "voxtral_tts: FM graph alloc failed\n");
            return false;
        }
        ctx->fm_gf = gf;
    }

    ggml_cgraph* gf = ctx->fm_gf;
    // ALL inputs must be re-set every call: gallocr reuses input buffers as scratch
    // after their last use, so the (constant) mask + velidx are re-written too.
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "xt"), x_t, 0, VTTS_ACOUSTIC_DIM * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "hin"), h, 0, (size_t)d * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "tin"), time_emb.data(), 0, (size_t)d * sizeof(float));
    {
        std::vector<ggml_fp16_t> mbuf((size_t)T * T);
        const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < T; q++)
            for (int k = 0; k < T; k++)
                mbuf[(size_t)q * T + k] = ((q < 3) == (k < 3)) ? z : ninf; // same CFG group → attend
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mask"), mbuf.data(), 0, mbuf.size() * sizeof(ggml_fp16_t));
        const int32_t vi[2] = {0, 3};
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "velidx"), vi, 0, sizeof(vi));
    }
    if (ggml_backend_graph_compute(ctx->backend, gf) != GGML_STATUS_SUCCESS)
        return false;
    v_cond.resize(VTTS_ACOUSTIC_DIM);
    v_uncond.resize(VTTS_ACOUSTIC_DIM);
    ggml_tensor* velt = ggml_graph_get_tensor(gf, "vel");
    ggml_backend_tensor_get(velt, v_cond.data(), 0, VTTS_ACOUSTIC_DIM * sizeof(float)); // col 0
    ggml_backend_tensor_get(velt, v_uncond.data(), VTTS_ACOUSTIC_DIM * sizeof(float),
                            VTTS_ACOUSTIC_DIM * sizeof(float)); // col 1
    return true;
}

// Greedy semantic token: argmax of fm.semantic_output @ h (no positional path).
// [EMPTY]=0 and the padded tail (≥ special+cb) are masked out; [END]=1 is allowed
// (it stops generation).
static int vtts_fm_semantic_argmax(voxtral_tts_context* ctx, const float* h) {
    const int d = ctx->hp.fm_dim;
    const int n_out = (int)ctx->model.fm_semantic_output->ne[1]; // 8320
    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph(ctx0);
    ggml_tensor* hin = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, d);
    ggml_set_name(hin, "hin");
    ggml_set_input(hin);
    ggml_tensor* logits = ggml_mul_mat(ctx0, ctx->model.fm_semantic_output, hin);
    if (ctx->model.fm_semantic_output_bias)
        logits = ggml_add(ctx0, logits, ctx->model.fm_semantic_output_bias);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        ggml_free(ctx0);
        return VTTS_AUDIO_END;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "hin"), h, 0, (size_t)d * sizeof(float));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx0);
        return VTTS_AUDIO_END;
    }
    std::vector<float> lg(n_out);
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "logits"), lg.data(), 0, lg.size() * sizeof(float));
    ggml_free(ctx0);

    lg[VTTS_AUDIO_EMPTY] = -1e30f; // [EMPTY] never emitted
    for (int i = VTTS_SPECIAL_COUNT + ctx->hp.semantic_cb_size; i < n_out; i++)
        lg[i] = -1e30f; // padded tail
    int best = 0;
    for (int i = 1; i < n_out; i++)
        if (lg[i] > lg[best])
            best = i;

    static bool logged = false;
    if (!logged && env_bool("CRISPASR_VOXTRAL_TTS_DEBUG")) {
        logged = true;
        // rank of [END]=1 and margin of the winner vs runner-up
        float second = -1e30f;
        for (int i = 1; i < n_out; i++)
            if (i != best && lg[i] > second)
                second = lg[i];
        int end_rank = 0;
        for (int i = 1; i < n_out; i++)
            if (lg[i] > lg[VTTS_AUDIO_END])
                end_rank++;
        fprintf(stderr, "voxtral_tts[sem]: best=%d logit=%.3f runnerup=%.3f margin=%.3f | END(1) logit=%.3f rank=%d\n",
                best, lg[best], second, lg[best] - second, lg[VTTS_AUDIO_END], end_rank);
    }
    return best;
}

// One frame → 37 codes (codes[0]=semantic incl. specials; codes[1..36]=acoustic
// FSQ+special offset). Returns semantic==[END] to signal end of audio.
static std::vector<int> vtts_acoustic_forward(voxtral_tts_context* ctx, const std::vector<float>& h) {
    std::vector<int> codes(1 + VTTS_ACOUSTIC_DIM, 0);
    codes[0] = vtts_fm_semantic_argmax(ctx, h.data());
    if (codes[0] == VTTS_AUDIO_END) {
        for (int i = 1; i < (int)codes.size(); i++)
            codes[i] = VTTS_AUDIO_EMPTY + VTTS_SPECIAL_COUNT;
        return codes;
    }

    // Euler flow-matching ODE from Gaussian noise.
    std::vector<float> x(VTTS_ACOUSTIC_DIM);
    for (int i = 0; i < VTTS_ACOUSTIC_DIM; i++)
        x[i] = vtts_randn(&ctx->rng_state) * VTTS_NOISE_SCALE;
    const int n_steps = ctx->fm_flow_steps; // = VTTS_FLOW_STEPS unless CRISPASR_VOXTRAL_TTS_FM_STEPS
    for (int step = 0; step < n_steps - 1; step++) {
        float t = (float)step / (float)(n_steps - 1);
        float dt = (float)(step + 1) / (float)(n_steps - 1) - t;
        std::vector<float> v_cond, v_uncond;
        if (!vtts_fm_predict_velocity_cfg(ctx, x.data(), h.data(), t, v_cond, v_uncond))
            break;
        for (int i = 0; i < VTTS_ACOUSTIC_DIM; i++) {
            float v = VTTS_CFG_ALPHA * v_cond[i] + (1.0f - VTTS_CFG_ALPHA) * v_uncond[i];
            x[i] += v * dt;
        }
    }

    // FSQ quantize → [0, levels-1], offset by special count.
    for (int i = 0; i < VTTS_ACOUSTIC_DIM; i++) {
        float val = std::max(-1.0f, std::min(1.0f, x[i]));
        int code = (int)(((val + 1.0f) / 2.0f) * (float)(VTTS_FSQ_LEVELS - 1) + 0.5f);
        code = std::max(0, std::min(VTTS_FSQ_LEVELS - 1, code));
        codes[i + 1] = code + VTTS_SPECIAL_COUNT;
    }
    return codes;
}

// Embed the 37 generated codes back into LLM input space: sum of one row per
// codebook from audio_embd. Offsets follow MultiVocabEmbeddings (pad_to_multiple
// =None): semantic codebook size 8194 (8192+2), acoustic 23 (21+2) each.
static std::vector<float> vtts_embed_audio_codes(voxtral_tts_context* ctx, const std::vector<int>& codes) {
    const int d = ctx->hp.llm_dim;
    const int sem_size = ctx->hp.semantic_cb_size + VTTS_SPECIAL_COUNT; // 8194
    const int acou_size = VTTS_FSQ_LEVELS + VTTS_SPECIAL_COUNT;         // 23
    const int n_cb = (int)codes.size();                                 // 37

    std::vector<int32_t> idx(n_cb);
    int off = 0;
    for (int cb = 0; cb < n_cb; cb++) {
        idx[cb] = off + codes[cb];
        off += (cb == 0) ? sem_size : acou_size;
    }

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph(ctx0);
    ggml_tensor* ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_cb);
    ggml_set_name(ids, "ids");
    ggml_set_input(ids);
    ggml_tensor* rows = ggml_get_rows(ctx0, ctx->model.audio_embd, ids); // (d, n_cb)
    ggml_set_name(rows, "rows");
    ggml_set_output(rows);
    ggml_build_forward_expand(gf, rows);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        ggml_free(ctx0);
        return {};
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "ids"), idx.data(), 0, idx.size() * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx0);
        return {};
    }
    std::vector<float> rowbuf((size_t)d * n_cb);
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "rows"), rowbuf.data(), 0, rowbuf.size() * sizeof(float));
    ggml_free(ctx0);

    std::vector<float> sum(d, 0.0f);
    for (int cb = 0; cb < n_cb; cb++)
        for (int j = 0; j < d; j++)
            sum[j] += rowbuf[(size_t)cb * d + j];
    return sum;
}

// ===========================================================================
// Stage 3 — Voxtral codec decoder
// ===========================================================================
// codes[37/frame] → [292,T] (256 semantic VQ + 36 FSQ) → causal conv(292→1024,k3)
// → 4×[2-layer ALiBi transformer + (first 3) ConvTranspose1d(1024→1024,k4,s2)] →
// causal conv(1024→240,k7) → 240 PCM/frame @ 24 kHz. ALiBi slopes 2^-(h+1),
// sliding windows {2,4,8,16}, QK-norm, layer_scale. Blueprint: mudler/voxtral-tts.c.

static constexpr int VTTS_CODEC_WINDOWS[4] = {2, 4, 8, 16};

// 292-d codec input on the host, laid out for a ggml (T, 292) tensor
// (flat = c*T + t): semantic 256-d VQ lookup + acoustic 36-d FSQ decode.
static std::vector<float> vtts_codec_embed(voxtral_tts_context* ctx, const std::vector<int>& codes, int n_frames) {
    const int Ds = ctx->hp.codec_semantic_dim; // 256
    const int Da = VTTS_ACOUSTIC_DIM;          // 36
    const int Dc = Ds + Da;                    // 292
    std::vector<float> emb((size_t)Dc * n_frames, 0.0f);
    const float* cb = ctx->semantic_cb_host.data();
    for (int t = 0; t < n_frames; t++) {
        const int* fc = &codes[(size_t)t * (1 + VTTS_ACOUSTIC_DIM)];
        int sem = fc[0] - VTTS_SPECIAL_COUNT;
        sem = std::max(0, std::min(ctx->hp.semantic_cb_size - 1, sem));
        const float* row = cb + (size_t)sem * Ds;
        for (int d = 0; d < Ds; d++)
            emb[(size_t)d * n_frames + t] = row[d];
        for (int i = 0; i < Da; i++) {
            int ac = fc[i + 1] - VTTS_SPECIAL_COUNT;
            ac = std::max(0, std::min(VTTS_FSQ_LEVELS - 1, ac));
            emb[(size_t)(Ds + i) * n_frames + t] = (float)ac * 2.0f / (float)(VTTS_FSQ_LEVELS - 1) - 1.0f;
        }
    }
    return emb;
}

// ALiBi + causal + sliding-window bias for one stage, laid out for a ggml scores
// tensor (n_kv=T, n_q=T, heads): flat = h*T*T + i*T + j (i=query, j=key).
static std::vector<float> vtts_codec_alibi_bias(int T, int window, int n_heads) {
    std::vector<float> bias((size_t)n_heads * T * T);
    for (int h = 0; h < n_heads; h++) {
        float slope = std::pow(2.0f, -(float)(h + 1)); // 2^-(h+1)
        for (int i = 0; i < T; i++)
            for (int j = 0; j < T; j++)
                bias[(size_t)h * T * T + (size_t)i * T + j] =
                    (j <= i && j > i - window) ? slope * (float)(j - i) : -INFINITY;
    }
    return bias;
}

// One codec transformer layer (ALiBi attention + QK-norm + layer_scale + SwiGLU).
// x is (D, T); bias is the precomputed (T, T, heads) ALiBi/window tensor.
static ggml_tensor* vtts_codec_tfm_layer(ggml_context* ctx0, ggml_tensor* x, const voxtral_tts_codec_tfm_layer& l,
                                         ggml_tensor* bias, const voxtral_tts_hparams& hp) {
    const int D = hp.codec_dim, nh = hp.codec_n_heads, hd = hp.codec_head_dim;
    const int T = (int)x->ne[1];
    const float scale = 1.0f / std::sqrt((float)hd);

    ggml_tensor* res = x;
    ggml_tensor* xn = ggml_mul(ctx0, ggml_rms_norm(ctx0, x, hp.codec_norm_eps), l.attn_norm);
    ggml_tensor* Q = ggml_mul_mat(ctx0, l.attn_q, xn);
    ggml_tensor* K = ggml_mul_mat(ctx0, l.attn_k, xn);
    ggml_tensor* V = ggml_mul_mat(ctx0, l.attn_v, xn);
    if (l.q_norm)
        Q = ggml_mul(ctx0, ggml_rms_norm(ctx0, Q, hp.codec_qk_norm_eps), l.q_norm);
    if (l.k_norm)
        K = ggml_mul(ctx0, ggml_rms_norm(ctx0, K, hp.codec_qk_norm_eps), l.k_norm);
    Q = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q, hd, nh, T), 0, 2, 1, 3)); // (hd,T,nh)
    K = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, K, hd, nh, T), 0, 2, 1, 3));
    ggml_tensor* V3 = ggml_reshape_3d(ctx0, V, hd, nh, T);
    ggml_tensor* scores = ggml_mul_mat(ctx0, K, Q); // (T,T,nh)
    scores = ggml_scale(ctx0, scores, scale);
    scores = ggml_add(ctx0, scores, bias); // ALiBi + causal + window
    scores = ggml_soft_max(ctx0, scores);
    V3 = ggml_cont(ctx0, ggml_permute(ctx0, V3, 0, 2, 1, 3)); // (hd,T,nh)
    V3 = ggml_cont(ctx0, ggml_permute(ctx0, V3, 1, 0, 2, 3)); // (T,hd,nh)
    ggml_tensor* attn = ggml_mul_mat(ctx0, V3, scores);       // (hd,T,nh)
    attn = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, attn, 0, 2, 1, 3)), D, T);
    ggml_tensor* proj = ggml_mul_mat(ctx0, l.attn_o, attn); // (D,T)
    if (l.attn_scale)
        proj = ggml_mul(ctx0, proj, ggml_reshape_2d(ctx0, l.attn_scale, D, 1));
    x = ggml_add(ctx0, res, proj);

    res = x;
    xn = ggml_mul(ctx0, ggml_rms_norm(ctx0, x, hp.codec_norm_eps), l.ffn_norm);
    ggml_tensor* ffn = core_ffn::swiglu(ctx0, xn, l.ffn_gate, l.ffn_up, l.ffn_down);
    if (l.ffn_scale)
        ffn = ggml_mul(ctx0, ffn, ggml_reshape_2d(ctx0, l.ffn_scale, D, 1));
    return ggml_add(ctx0, res, ffn);
}

// Causal reflect-padded Conv1d: x=(T,Cin), weight=(K,Cin,Cout) → (T,Cout).
static ggml_tensor* vtts_codec_causal_conv(ggml_context* ctx0, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    const int K = (int)w->ne[0];
    ggml_tensor* xp = ggml_pad_reflect_1d(ctx0, x, K - 1, 0); // stride 1 → left_pad = K-1
    ggml_tensor* y = ggml_conv_1d(ctx0, w, xp, 1, 0, 1);      // (T, Cout)
    if (b)
        y = ggml_add(ctx0, y, ggml_reshape_2d(ctx0, b, 1, (int)b->ne[0]));
    return y;
}

// Full codec decode → interleaved PCM (T_final × 240 samples). Runs as one graph.
static std::vector<float> vtts_codec_decode(voxtral_tts_context* ctx, const std::vector<int>& codes, int n_frames) {
    const auto& hp = ctx->hp;
    const auto& m = ctx->model;
    const int D = hp.codec_dim;
    const int Dc = hp.codec_semantic_dim + VTTS_ACOUSTIC_DIM; // 292
    const int nh = hp.codec_n_heads;

    if (ctx->semantic_cb_host.empty()) {
        fprintf(stderr, "voxtral_tts: codec needs semantic_cb (GGUF tensor or CRISPASR_VOXTRAL_TTS_SEMANTIC_CB)\n");
        return {};
    }

    std::vector<float> emb = vtts_codec_embed(ctx, codes, n_frames);
    int Ts[4];
    Ts[0] = n_frames;
    for (int s = 1; s < 4; s++)
        Ts[s] = Ts[s - 1] * hp.codec_conv_strides[s]; // strides {1,2,2,2}
    std::vector<std::vector<float>> biases(4);
    for (int s = 0; s < 4; s++)
        biases[s] = vtts_codec_alibi_bias(Ts[s], VTTS_CODEC_WINDOWS[s], nh);

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* emb_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_frames, Dc); // (T,292)
    ggml_set_name(emb_in, "emb");
    ggml_set_input(emb_in);
    ggml_tensor* bias_in[4];
    for (int s = 0; s < 4; s++) {
        bias_in[s] = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, Ts[s], Ts[s], nh);
        ggml_set_name(bias_in[s], ("bias" + std::to_string(s)).c_str());
        ggml_set_input(bias_in[s]);
    }

    // Input conv (292→1024, k3) → (1024, T)
    ggml_tensor* x = vtts_codec_causal_conv(ctx0, emb_in, m.codec_convs[0].weight, m.codec_convs[0].bias); // (T,1024)
    x = ggml_cont(ctx0, ggml_transpose(ctx0, x));                                                          // (1024,T)

    for (int stage = 0; stage < 4; stage++) {
        for (int li = 0; li < hp.codec_tfm_lengths[stage]; li++)
            x = vtts_codec_tfm_layer(ctx0, x, m.codec_tfm_blocks[stage][li], bias_in[stage], hp);
        if (stage < 3) {
            // Causal ConvTranspose1d (1024→1024, k4, s2): crop trailing K-stride=2.
            auto& c = m.codec_convs[stage + 1];
            x = core_convt::convt1d_crop(ctx0, x, c.weight, c.bias, hp.codec_conv_strides[stage + 1], 0,
                                         hp.codec_conv_kernels[stage + 1] - hp.codec_conv_strides[stage + 1]);
        }
    }

    // Output conv (1024→240, k7) on (T_final,1024) → (240, T_final)
    ggml_tensor* xt = ggml_cont(ctx0, ggml_transpose(ctx0, x));                                    // (T_final, 1024)
    ggml_tensor* y = vtts_codec_causal_conv(ctx0, xt, m.codec_output.weight, m.codec_output.bias); // (T_final,240)
    y = ggml_cont(ctx0, ggml_transpose(ctx0, y)); // (240, T_final) → flat = t*240 + h = interleaved PCM
    ggml_set_name(y, "pcm");
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        ggml_free(ctx0);
        return {};
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "emb"), emb.data(), 0, emb.size() * sizeof(float));
    for (int s = 0; s < 4; s++)
        ggml_backend_tensor_set(bias_in[s], biases[s].data(), 0, biases[s].size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx0);
        return {};
    }
    ggml_tensor* pcm_t = ggml_graph_get_tensor(gf, "pcm");
    const int n_samples = (int)ggml_nelements(pcm_t);
    std::vector<float> pcm(n_samples);
    ggml_backend_tensor_get(pcm_t, pcm.data(), 0, (size_t)n_samples * sizeof(float));
    ggml_free(ctx0);
    return pcm;
}

// ---------------------------------------------------------------------------
// Synthesize (LLM AR + FM ODE + codec decode)
// ---------------------------------------------------------------------------

extern "C" float* voxtral_tts_synthesize(voxtral_tts_context* ctx, const char* text, const char* voice,
                                         int* out_n_samples) {
    if (!ctx || !text || !out_n_samples)
        return nullptr;
    *out_n_samples = 0;

    const auto& m = ctx->model;

    // Diff-harness codec isolation: decode a codes file (37 ints/line) directly,
    // bypassing the LLM+FM, so my codec can be compared to the reference codec on
    // IDENTICAL codes. Gated by CRISPASR_VOXTRAL_TTS_CODEC_FROM_FILE.
    if (const char* cf = std::getenv("CRISPASR_VOXTRAL_TTS_CODEC_FROM_FILE")) {
        FILE* fp = fopen(cf, "r");
        if (!fp)
            return nullptr;
        std::vector<int> codes;
        int n_frames = 0;
        char line[8192];
        while (fgets(line, sizeof(line), fp)) {
            int k = 0;
            for (char* p = strtok(line, ",\n"); p && k < 1 + VTTS_ACOUSTIC_DIM; p = strtok(nullptr, ",\n")) {
                codes.push_back(atoi(p));
                k++;
            }
            if (k == 1 + VTTS_ACOUSTIC_DIM)
                n_frames++;
            else
                codes.resize((size_t)n_frames * (1 + VTTS_ACOUSTIC_DIM));
        }
        fclose(fp);
        fprintf(stderr, "voxtral_tts: codec-only decode of %d frames from %s\n", n_frames, cf);
        std::vector<float> pcm = vtts_codec_decode(ctx, codes, n_frames);
        if (pcm.empty())
            return nullptr;
        float* out = (float*)malloc(pcm.size() * sizeof(float));
        std::copy(pcm.begin(), pcm.end(), out);
        *out_n_samples = (int)pcm.size();
        return out;
    }

    // Step 1: Tokenize text
    std::vector<int32_t> text_ids = voxtral_tts_tokenize(ctx, text);
    if (ctx->verbosity >= 1) {
        fprintf(stderr, "voxtral_tts: tokenized %d tokens from \"%s\"\n", (int)text_ids.size(), text);
        if (env_bool("CRISPASR_VOXTRAL_TTS_DEBUG")) {
            fprintf(stderr, "voxtral_tts: token ids =");
            for (int32_t id : text_ids)
                fprintf(stderr, " %d", id);
            fprintf(stderr, "\n");
        }
    }

    // Step 2: Get voice embedding
    std::string voice_name = voice ? voice : "neutral_female";
    auto vit = m.voice_tensors.find(voice_name);
    if (vit == m.voice_tensors.end() && !m.voice_tensors.empty()) {
        if (ctx->verbosity >= 1)
            fprintf(stderr, "voxtral_tts: voice '%s' not found, using first available\n", voice_name.c_str());
        vit = m.voice_tensors.begin();
        voice_name = vit->first;
    }
    if (vit == m.voice_tensors.end()) {
        fprintf(stderr, "voxtral_tts: no voice embeddings available\n");
        return nullptr;
    }

    // Voice tensor is pre-summed (T_voice, dim) embeddings
    ggml_tensor* voice_t = vit->second;
    int T_voice = (int)voice_t->ne[1];
    if (ctx->verbosity >= 1) {
        fprintf(stderr, "voxtral_tts: voice '%s' (%d frames)\n", voice_name.c_str(), T_voice);
    }

    // Step 3: Prompt embeddings — [BOS][BEGIN_AUDIO][voice×N][/INST]text[INST][BEGIN_AUDIO]
    int T_prompt = 0;
    std::vector<float> prompt = voxtral_tts_build_prompt_embeds(ctx, voice_t, text_ids, &T_prompt);
    if (prompt.empty()) {
        fprintf(stderr, "voxtral_tts: failed to build prompt embeddings\n");
        return nullptr;
    }

    // Step 4: KV cache + LLM prefill, then feed one AUDIO token (24) to obtain the
    // frame-0 conditioning hidden state (matching the reference AR loop: after
    // prefill the first decode step embeds the AUDIO placeholder). max_ctx budgets
    // the prompt plus up to 2048 generated audio frames (~164 s at 12.5 Hz).
    if (!voxtral_tts_kv_init(ctx, T_prompt + 2048))
        return nullptr;
    voxtral_tts_kv_reset(ctx);

    std::vector<float> h_prefill = voxtral_tts_run_llm(ctx, prompt.data(), T_prompt, /*n_past*/ 0);
    if (h_prefill.empty()) {
        fprintf(stderr, "voxtral_tts: LLM prefill failed\n");
        return nullptr;
    }
    ctx->kv_used = T_prompt;

    std::vector<float> audio_emb = voxtral_tts_embed_ids(ctx, {ctx->hp.audio_token_id});
    std::vector<float> h0 = voxtral_tts_run_llm(ctx, audio_emb.data(), /*n_tokens*/ 1, /*n_past*/ T_prompt);
    if (h0.empty()) {
        fprintf(stderr, "voxtral_tts: LLM frame-0 decode failed\n");
        return nullptr;
    }
    ctx->kv_used = T_prompt + 1;

    if (ctx->verbosity >= 1)
        fprintf(stderr, "voxtral_tts: prefill done (T_prompt=%d), generating audio codes...\n", T_prompt);

    // Step 5: Autoregressive audio-code generation.
    //   frame: acoustic_forward(h) → 37 codes; stop at [END]; else embed codes
    //   back (audio_embd sum) → next LLM input → hidden for the next frame.
    const int max_frames = 2000; // ~160 s at 12.5 Hz
    std::vector<int> all_codes;
    std::vector<float> h = h0;
    int n_past = T_prompt + 1;
    int n_frames = 0;
    bool got_end = false;

    const bool dbg = env_bool("CRISPASR_VOXTRAL_TTS_DEBUG");
    const bool timing = dbg || env_bool("CRISPASR_VOXTRAL_TTS_TIMING");
    int64_t t_fm_us = 0, t_llm_us = 0;
    for (int frame = 0; frame < max_frames; frame++) {
        if (dbg && frame == 0) {
            FILE* fp = fopen("/tmp/vtts_h0.f32.bin", "wb");
            if (fp) {
                fwrite(h.data(), sizeof(float), h.size(), fp);
                fclose(fp);
            }
        }
        int64_t _t0 = ggml_time_us();
        std::vector<int> codes = vtts_acoustic_forward(ctx, h);
        t_fm_us += ggml_time_us() - _t0;
        if (dbg) {
            double sq = 0.0;
            for (float x : h)
                sq += (double)x * x;
            fprintf(stderr, "MINE frame %d |h|=%.4f codes=", frame, std::sqrt(sq));
            for (int c : codes)
                fprintf(stderr, "%d,", c);
            fprintf(stderr, "\n");
        } else if (ctx->verbosity >= 1 && frame < 3) {
            double sq = 0.0;
            for (float x : h)
                sq += (double)x * x;
            fprintf(stderr, "voxtral_tts: frame %d |h|=%.3f sem=%d ac[0..2]=%d,%d,%d\n", frame, std::sqrt(sq), codes[0],
                    codes[1], codes[2], codes[3]);
        }
        if (codes[0] == VTTS_AUDIO_END) {
            got_end = true;
            break;
        }
        all_codes.insert(all_codes.end(), codes.begin(), codes.end());
        n_frames++;

        std::vector<float> next_emb = vtts_embed_audio_codes(ctx, codes);
        if (next_emb.empty())
            break;
        int64_t _t1 = ggml_time_us();
        h = voxtral_tts_run_llm(ctx, next_emb.data(), /*n_tokens*/ 1, n_past);
        t_llm_us += ggml_time_us() - _t1;
        if (h.empty())
            break;
        n_past++;
        ctx->kv_used = n_past;
    }

    if (ctx->verbosity >= 1)
        fprintf(stderr, "voxtral_tts: generated %d frames (%.2f s)%s\n", n_frames, n_frames / ctx->hp.frame_rate,
                got_end ? " [END]" : " [max_frames]");
    if (timing && n_frames > 0)
        fprintf(stderr, "voxtral_tts[timing]: LLM-decode %.1f ms/frame, FM %.1f ms/frame (%.1f s tot)\n",
                t_llm_us / 1000.0 / n_frames, t_fm_us / 1000.0 / n_frames, t_fm_us / 1e6);

    if (n_frames == 0) {
        fprintf(stderr, "voxtral_tts: no audio frames produced\n");
        return nullptr;
    }

    // Step 6: Codec decode → 24 kHz PCM.
    std::vector<float> pcm = vtts_codec_decode(ctx, all_codes, n_frames);
    if (pcm.empty()) {
        fprintf(stderr, "voxtral_tts: codec decode failed\n");
        return nullptr;
    }
    if (ctx->verbosity >= 1) {
        float peak = 0.0f;
        for (float s : pcm)
            peak = std::max(peak, std::fabs(s));
        fprintf(stderr, "voxtral_tts: codec decoded %d samples (%.2f s), peak=%.4f\n", (int)pcm.size(),
                pcm.size() / (float)ctx->hp.sample_rate, peak);
    }

    float* out = (float*)malloc(pcm.size() * sizeof(float));
    if (!out)
        return nullptr;
    std::copy(pcm.begin(), pcm.end(), out);
    *out_n_samples = (int)pcm.size();
    return out;
}

// ---------------------------------------------------------------------------
// Cleanup + API
// ---------------------------------------------------------------------------

extern "C" void voxtral_tts_pcm_free(float* pcm) {
    free(pcm);
}

extern "C" int voxtral_tts_sample_rate(void) {
    return 24000;
}

extern "C" void voxtral_tts_set_seed(voxtral_tts_context* ctx, uint64_t seed) {
    if (ctx)
        ctx->rng_state = seed ? seed : 0x12345678ABCDEF01ULL;
}

extern "C" const char* const* voxtral_tts_list_voices(voxtral_tts_context* ctx, int* out_n_voices) {
    if (!ctx || !out_n_voices)
        return nullptr;
    *out_n_voices = (int)ctx->voice_names.size();
    return ctx->voice_name_ptrs.data();
}

extern "C" void voxtral_tts_free(voxtral_tts_context* ctx) {
    if (!ctx)
        return;
    if (ctx->fm_alloc)
        ggml_gallocr_free(ctx->fm_alloc);
    if (ctx->fm_ctx)
        ggml_free(ctx->fm_ctx);
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->buf)
        core_gguf::release_weight_buffer(ctx->buf);
    if (ctx->ctx_w)
        ggml_free(ctx->ctx_w);
    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

// crispasr-diff LLM stage: per-layer frame-0 cos vs the reference GGUF produced by
// tools/reference_backends/voxtral_tts.py (manual PyTorch forward, no vllm). Runs the
// real synth path with CRISPASR_VOXTRAL_TTS_DIFF_DUMP so the per-layer dumps come from
// the actual runtime graph, then compares embed + each LLM layer + hidden against the
// ref tensors of the same name. Prints the FIRST divergent stage (structural bug) or
// confirms the LLM path is correct (so any code divergence is downstream/stochastic).
extern "C" int voxtral_tts_llm_diff(const char* model_gguf, const char* ref_gguf, int verbosity) {
    voxtral_tts_context_params p = voxtral_tts_context_default_params();
    p.verbosity = 0;
    voxtral_tts_context* ctx = voxtral_tts_init_from_file(model_gguf, p);
    if (!ctx) {
        fprintf(stderr, "voxtral_tts_llm_diff: failed to load model %s\n", model_gguf);
        return 2;
    }
    core_gguf::WeightLoad rw;
    if (!core_gguf::load_weights(ref_gguf, ctx->backend, "ref", rw)) {
        fprintf(stderr, "voxtral_tts_llm_diff: failed to load reference %s\n", ref_gguf);
        voxtral_tts_free(ctx);
        return 2;
    }
    const int D = ctx->hp.llm_dim;

    // Portable unique temp dir (replaces POSIX mkdtemp + hardcoded "/tmp" — Windows has neither).
    std::error_code _ec;
    std::filesystem::path dir =
        std::filesystem::temp_directory_path(_ec) / ("vtts_diff_" + std::to_string((unsigned long long)ggml_time_us()));
    if (_ec || (!std::filesystem::create_directories(dir, _ec) && _ec)) {
        voxtral_tts_free(ctx);
        return 2;
    }
    const std::string dir_s = dir.string();
#if defined(_WIN32)
    _putenv_s("CRISPASR_VOXTRAL_TTS_DIFF_DUMP", dir_s.c_str());
#else
    setenv("CRISPASR_VOXTRAL_TTS_DIFF_DUMP", dir_s.c_str(), 1);
#endif
    const char* text = crispasr_env::get("CRISPASR_VOXTRAL_TTS_TEXT") ? crispasr_env::get("CRISPASR_VOXTRAL_TTS_TEXT")
                                                                      : "Hello world.";
    const char* voice = crispasr_env::get("CRISPASR_VOXTRAL_TTS_VOICE")
                            ? crispasr_env::get("CRISPASR_VOXTRAL_TTS_VOICE")
                            : "neutral_female";
    int n_samples = 0;
    float* pcm = voxtral_tts_synthesize(ctx, text, voice, &n_samples); // dumps mine.c1.<stage> (frame 0)
    if (pcm)
        voxtral_tts_pcm_free(pcm);
#if defined(_WIN32)
    _putenv_s("CRISPASR_VOXTRAL_TTS_DIFF_DUMP", "");
#else
    unsetenv("CRISPASR_VOXTRAL_TTS_DIFF_DUMP");
#endif

    std::vector<std::string> stages = {"embed"};
    for (int i = 0; i < ctx->hp.llm_n_layers; i++)
        stages.push_back("llm_L" + std::to_string(i));
    stages.push_back("hidden");

    int n_fail = 0;
    std::string first_bad;
    for (const auto& s : stages) {
        std::vector<float> a(D), b(D);
        char pth[512];
        snprintf(pth, sizeof(pth), "%s/mine.c1.%s.bin", dir_s.c_str(), s.c_str());
        FILE* fp = fopen(pth, "rb");
        // Use load_weights' canonical name→tensor map, not ggml_get_tensor(rw.ctx, ...)
        // which scans the ctx and (observed) could return the wrong tensor's data for
        // the "embed" stage while the layer stages resolved fine.
        auto rit = rw.tensors.find(s);
        ggml_tensor* rt = (rit != rw.tensors.end()) ? rit->second : nullptr;
        bool have_mine = fp && fread(a.data(), sizeof(float), D, fp) == (size_t)D;
        if (fp)
            fclose(fp);
        remove(pth);
        if (!have_mine || !rt || ggml_nelements(rt) != D) {
            printf("voxtral-tts llm %-8s  (missing: %s%s)\n", s.c_str(), have_mine ? "" : "runtime ",
                   (rt && ggml_nelements(rt) == D) ? "" : "ref");
            continue;
        }
        ggml_backend_tensor_get(rt, b.data(), 0, (size_t)D * sizeof(float));
        double dot = 0, na = 0, nb = 0, maxd = 0;
        for (int i = 0; i < D; i++) {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
            double dd = std::fabs((double)a[i] - b[i]);
            if (dd > maxd)
                maxd = dd;
        }
        double cos = (na > 0 && nb > 0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
        bool pass = cos > 0.99;
        if (!pass) {
            n_fail++;
            if (first_bad.empty())
                first_bad = s;
        }
        printf("voxtral-tts llm %-8s  cos=%.6f  max_abs=%.5f  |mine|=%.3f |ref|=%.3f  %s\n", s.c_str(), cos, maxd,
               std::sqrt(na), std::sqrt(nb), pass ? "PASS" : "FAIL");
    }
    if (!first_bad.empty())
        printf("\nFIRST DIVERGENCE (cos<0.99): %s  → structural bug in that layer.\n", first_bad.c_str());
    else
        printf("\nALL LLM STAGES cos>=0.99 — the deterministic LLM path matches the reference; any "
               "per-frame code divergence is downstream (stochastic acoustic sampler).\n");
    (void)verbosity;
    voxtral_tts_free(ctx);
    return n_fail > 0 ? 1 : 0;
}
