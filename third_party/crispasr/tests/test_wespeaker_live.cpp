// tests/test_wespeaker_live.cpp — integration tests for the WeSpeaker
// ResNet34-LM speaker embedder (#324).
//
// Live tests: require a model on disk. Gated behind [.live].
// Run: ctest -R test-wespeaker --output-on-failure
//
// Env vars (see tests/env-live-tests.sh):
//   CRISPASR_MODEL_WESPEAKER — GGUF path

#include <catch2/catch_test_macros.hpp>

#include "wespeaker.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "portable_env.h"

static std::string get_env(const char* name, const char* fallback = "") {
    const char* v = std::getenv(name);
    return v ? v : fallback;
}

static std::vector<float> load_wav_16k_mono(const std::string& path) {
    std::vector<float> pcm;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f)
        return pcm;
    char riff[12];
    if (fread(riff, 1, 12, f) != 12 || memcmp(riff, "RIFF", 4) != 0 || memcmp(riff + 8, "WAVE", 4) != 0) {
        fclose(f);
        return pcm;
    }
    int channels = 0, bits = 0;
    int32_t data_size = 0;
    bool found_fmt = false, found_data = false;
    while (!found_data) {
        char id[4];
        int32_t sz;
        if (fread(id, 1, 4, f) != 4 || fread(&sz, 4, 1, f) != 1)
            break;
        if (memcmp(id, "fmt ", 4) == 0) {
            char fmt[16];
            if (sz < 16 || fread(fmt, 1, 16, f) != 16)
                break;
            channels = *(int16_t*)(fmt + 2);
            bits = *(int16_t*)(fmt + 14);
            found_fmt = true;
            if (sz > 16)
                fseek(f, sz - 16, SEEK_CUR);
        } else if (memcmp(id, "data", 4) == 0) {
            data_size = sz;
            found_data = true;
        } else {
            fseek(f, sz, SEEK_CUR);
        }
    }
    if (!found_fmt || !found_data || bits != 16 || channels < 1) {
        fclose(f);
        return pcm;
    }
    const int n = data_size / (channels * (bits / 8));
    std::vector<int16_t> raw((size_t)n * channels);
    if (fread(raw.data(), sizeof(int16_t), raw.size(), f) != raw.size()) {
        fclose(f);
        return {};
    }
    fclose(f);
    pcm.resize((size_t)n);
    for (int i = 0; i < n; i++) {
        float s = 0;
        for (int c = 0; c < channels; c++)
            s += raw[(size_t)i * channels + c];
        pcm[(size_t)i] = s / (channels * 32768.0f);
    }
    return pcm;
}

static double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

// Embed [t0, t0+dur) seconds of `pcm`.
static std::vector<float> embed_window(wespeaker_context* ctx, const std::vector<float>& pcm, double t0, double dur) {
    const size_t a = (size_t)(t0 * 16000), b = (size_t)((t0 + dur) * 16000);
    if (b > pcm.size() || a >= b)
        return {};
    std::vector<float> emb((size_t)wespeaker_embed_dim(ctx));
    if (wespeaker_embed(ctx, pcm.data() + a, (int)(b - a), emb.data()) != 0)
        return {};
    return emb;
}

static wespeaker_context* open_model() {
    const std::string model = get_env("CRISPASR_MODEL_WESPEAKER");
    if (model.empty())
        return nullptr;
    wespeaker_context_params cp = wespeaker_context_default_params();
    cp.n_threads = 2;
    cp.verbosity = 0;
    return wespeaker_init_from_file(model.c_str(), cp);
}

TEST_CASE("wespeaker: init reports the model's shape", "[wespeaker][.live]") {
    if (get_env("CRISPASR_MODEL_WESPEAKER").empty())
        SKIP("CRISPASR_MODEL_WESPEAKER not set");
    wespeaker_context* ctx = open_model();
    REQUIRE(ctx != nullptr);
    CHECK(wespeaker_embed_dim(ctx) == 256);
    CHECK(wespeaker_sample_rate(ctx) == 16000);
    CHECK(wespeaker_n_mels(ctx) == 80);
    CHECK(wespeaker_min_samples(ctx) > 0);
    wespeaker_free(ctx);
}

TEST_CASE("wespeaker: embedding is finite and non-degenerate", "[wespeaker][.live]") {
    if (get_env("CRISPASR_MODEL_WESPEAKER").empty())
        SKIP("CRISPASR_MODEL_WESPEAKER not set");
    auto pcm = load_wav_16k_mono("samples/jfk.wav");
    REQUIRE(!pcm.empty());
    wespeaker_context* ctx = open_model();
    REQUIRE(ctx != nullptr);

    std::vector<float> emb((size_t)wespeaker_embed_dim(ctx));
    REQUIRE(wespeaker_embed(ctx, pcm.data(), (int)pcm.size(), emb.data()) == 0);

    double norm = 0;
    for (float v : emb) {
        CHECK(std::isfinite(v));
        norm += (double)v * v;
    }
    norm = std::sqrt(norm);
    INFO("|emb| = " << norm);
    // The model emits raw (un-normalised) embeddings; an all-zero or wildly
    // scaled vector would mean the head or the pooling is broken.
    CHECK(norm > 0.1);
    CHECK(norm < 100.0);
    wespeaker_free(ctx);
}

TEST_CASE("wespeaker: embedding is deterministic", "[wespeaker][.live]") {
    if (get_env("CRISPASR_MODEL_WESPEAKER").empty())
        SKIP("CRISPASR_MODEL_WESPEAKER not set");
    auto pcm = load_wav_16k_mono("samples/jfk.wav");
    REQUIRE(!pcm.empty());
    wespeaker_context* ctx = open_model();
    REQUIRE(ctx != nullptr);
    auto a = embed_window(ctx, pcm, 0.0, 3.0);
    auto b = embed_window(ctx, pcm, 0.0, 3.0);
    REQUIRE(a.size() == 256);
    REQUIRE(b.size() == 256);
    for (size_t i = 0; i < a.size(); i++)
        REQUIRE(a[i] == b[i]);
    wespeaker_free(ctx);
}

TEST_CASE("wespeaker: audio below the minimum is rejected", "[wespeaker][.live]") {
    if (get_env("CRISPASR_MODEL_WESPEAKER").empty())
        SKIP("CRISPASR_MODEL_WESPEAKER not set");
    wespeaker_context* ctx = open_model();
    REQUIRE(ctx != nullptr);
    std::vector<float> tiny((size_t)wespeaker_min_samples(ctx) / 2, 0.0f);
    std::vector<float> emb(256);
    // Must fail cleanly rather than emit a garbage embedding: the time axis is
    // downsampled 8x, so a too-short clip leaves the final map empty.
    CHECK(wespeaker_embed(ctx, tiny.data(), (int)tiny.size(), emb.data()) != 0);
    wespeaker_free(ctx);
}

// The property the whole diarizer rests on: two windows of the SAME speaker
// must sit closer in cosine than a window of a DIFFERENT speaker. This is what
// distinguishes a working embedder from one that merely produces finite
// numbers — a transposed feature map or a mis-ordered TSTP flatten passes every
// test above and fails this one.
TEST_CASE("wespeaker: same speaker is closer than a different speaker", "[wespeaker][.live]") {
    if (get_env("CRISPASR_MODEL_WESPEAKER").empty())
        SKIP("CRISPASR_MODEL_WESPEAKER not set");
    auto jfk = load_wav_16k_mono("samples/jfk.wav");
    auto multi = load_wav_16k_mono("samples/multispeaker.wav");
    REQUIRE(jfk.size() > 16000 * 6);
    REQUIRE(multi.size() > 16000 * 17); // need audio past the embedded jfk prefix

    wespeaker_context* ctx = open_model();
    REQUIRE(ctx != nullptr);

    // Two disjoint windows of the same speaker.
    auto a = embed_window(ctx, jfk, 0.5, 2.5);
    auto b = embed_window(ctx, jfk, 5.0, 2.5);
    // A window from a DIFFERENT speaker. ⚠ samples/multispeaker.wav opens with
    // all 11 s of samples/jfk.wav verbatim (176,000 identical samples) and only
    // then changes speaker — so any window before 11 s here is the SAME audio as
    // `a`/`b` and scores cos = 1.000000 exactly. Take the window well past that
    // boundary.
    auto c = embed_window(ctx, multi, 14.0, 2.5);
    REQUIRE(a.size() == 256);
    REQUIRE(b.size() == 256);
    REQUIRE(c.size() == 256);

    const double same = cosine(a, b);
    const double diff = std::max(cosine(a, c), cosine(b, c));
    INFO("cos(same-speaker) = " << same << "   cos(cross-speaker) = " << diff);
    CHECK(same > diff);
    // Generous absolute floors: the point is the ORDERING, not a WER-style
    // threshold. A margin this wide still catches a scrambled embedding.
    CHECK(same > 0.3);
    wespeaker_free(ctx);
}

// #324 perf: wespeaker_embed_batch must reproduce the per-window path — that
// is its entire contract. Mixed window lengths exercise the exact-T bucketing
// (three 1.2 s windows batch together, the 0.9 s tail becomes a singleton and
// takes the per-window graph). A tiny batch cap forces the chunking code too.
TEST_CASE("wespeaker: batched embedding matches per-window", "[wespeaker][.live]") {
    if (get_env("CRISPASR_MODEL_WESPEAKER").empty())
        SKIP("CRISPASR_MODEL_WESPEAKER not set");
    auto pcm = load_wav_16k_mono("samples/jfk.wav");
    REQUIRE(pcm.size() > 16000 * 8);
    wespeaker_context* ctx = open_model();
    REQUIRE(ctx != nullptr);

    const int sr = wespeaker_sample_rate(ctx);
    const int dim = wespeaker_embed_dim(ctx);
    const int64_t offsets[] = {(int64_t)(0.5 * sr), (int64_t)(1.1 * sr), (int64_t)(3.0 * sr), (int64_t)(5.2 * sr),
                               (int64_t)(6.8 * sr)};
    const int lengths[] = {(int)(1.2 * sr), (int)(1.2 * sr), (int)(1.2 * sr), (int)(1.2 * sr), (int)(0.9 * sr)};
    const int n_win = 5;

    std::vector<float> batched((size_t)n_win * dim);
    setenv("CRISPASR_WESPEAKER_BATCH", "2", 1); // force >1 chunk per T group
    const int rc = wespeaker_embed_batch(ctx, pcm.data(), (int64_t)pcm.size(), offsets, lengths, n_win, batched.data());
    unsetenv("CRISPASR_WESPEAKER_BATCH");
    REQUIRE(rc == 0);

    for (int i = 0; i < n_win; i++) {
        std::vector<float> ref((size_t)dim);
        REQUIRE(wespeaker_embed(ctx, pcm.data() + offsets[i], lengths[i], ref.data()) == 0);
        std::vector<float> got(batched.begin() + (size_t)i * dim, batched.begin() + (size_t)(i + 1) * dim);
        const double cos = cosine(ref, got);
        INFO("window " << i << " cos(batched, per-window) = " << cos);
        CHECK(cos > 0.999999);
    }
    wespeaker_free(ctx);
}

// #324 perf: the im2col conv lowering must be numerically interchangeable with
// the direct conv it replaces — same convolution, different summation order.
// The bar is the same one the port itself was held to against the Python
// oracle (cosine ~0.999997), an order of magnitude above what a wrong stride,
// pad, or kernel layout would score.
TEST_CASE("wespeaker: im2col conv matches direct conv", "[wespeaker][.live]") {
    if (get_env("CRISPASR_MODEL_WESPEAKER").empty())
        SKIP("CRISPASR_MODEL_WESPEAKER not set");
    auto pcm = load_wav_16k_mono("samples/jfk.wav");
    REQUIRE(pcm.size() > 16000 * 6);

    setenv("CRISPASR_WESPEAKER_CONV", "direct", 1);
    wespeaker_context* ctx_d = open_model();
    REQUIRE(ctx_d != nullptr);
    auto a = embed_window(ctx_d, pcm, 0.5, 2.5);
    wespeaker_free(ctx_d);

    setenv("CRISPASR_WESPEAKER_CONV", "im2col", 1);
    wespeaker_context* ctx_i = open_model();
    REQUIRE(ctx_i != nullptr);
    auto b = embed_window(ctx_i, pcm, 0.5, 2.5);
    wespeaker_free(ctx_i);
    unsetenv("CRISPASR_WESPEAKER_CONV");

    REQUIRE(a.size() == 256);
    REQUIRE(b.size() == 256);
    const double cos = cosine(a, b);
    INFO("cos(direct, im2col) = " << cos);
    CHECK(cos > 0.99999);
}
