// tests/test_canary_qwen_live.cpp — integration tests for the canary-qwen
// (nvidia/canary-qwen-2.5b SALM) backend.
//
// Live tests: require the model + samples/jfk.wav on disk. Gated behind
// [.live]. Run:
//   CRISPASR_MODEL_CANARY_QWEN=/path/canary-qwen-2.5b-q8_0.gguf \
//     ctest -R test-canary-qwen-live --output-on-failure
//
// #247 regression: on a too-short audio window the SALM decoder echoes its task
// framing as a meta word ("Transcript"/"Transcription"/"PASS") instead of
// transcribing. This is genuine model behaviour (the NeMo SALM reference emits
// the identical tokens), so the runtime gates degenerate windows and strips the
// echoes. These tests assert (a) a normal utterance transcribes verbatim and
// (b) a sub-0.5 s window produces NO instruction-echo leak.

#include <catch2/catch_test_macros.hpp>
#include "canary_qwen.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::string get_env(const char* name, const char* fallback = "") {
    const char* v = std::getenv(name);
    return v ? v : fallback;
}

// Chunk-skipping 16-bit PCM WAV loader (mono, 16 kHz expected).
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
        char chunk_id[4];
        int32_t chunk_size;
        if (fread(chunk_id, 1, 4, f) != 4)
            break;
        if (fread(&chunk_size, 4, 1, f) != 1)
            break;
        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            char fmt[16];
            if (chunk_size < 16 || fread(fmt, 1, 16, f) != 16)
                break;
            channels = *(int16_t*)(fmt + 2);
            bits = *(int16_t*)(fmt + 14);
            found_fmt = true;
            if (chunk_size > 16)
                fseek(f, chunk_size - 16, SEEK_CUR);
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            data_size = chunk_size;
            found_data = true;
        } else {
            fseek(f, chunk_size, SEEK_CUR);
        }
    }

    if (!found_fmt || !found_data || bits != 16 || channels < 1) {
        fclose(f);
        return pcm;
    }

    int n_samples = data_size / (channels * (bits / 8));
    std::vector<int16_t> raw(n_samples * channels);
    fread(raw.data(), sizeof(int16_t), raw.size(), f);
    fclose(f);

    pcm.resize(n_samples);
    for (int i = 0; i < n_samples; i++) {
        float sum = 0;
        for (int c = 0; c < channels; c++)
            sum += raw[i * channels + c];
        pcm[i] = sum / (channels * 32768.0f);
    }
    return pcm;
}

static bool contains_ci(const std::string& hay, const char* needle) {
    std::string h = hay, n = needle;
    for (auto& c : h)
        c = (char)tolower((unsigned char)c);
    for (auto& c : n)
        c = (char)tolower((unsigned char)c);
    return h.find(n) != std::string::npos;
}

static canary_qwen_context* open_ctx(const std::string& model) {
    canary_qwen_context_params cp = canary_qwen_context_default_params();
    cp.n_threads = 2;
    cp.verbosity = 0;
    return canary_qwen_init_from_file(model.c_str(), cp);
}

TEST_CASE("canary-qwen: init and free", "[canary-qwen][.live]") {
    std::string model = get_env("CRISPASR_MODEL_CANARY_QWEN");
    if (model.empty())
        SKIP("CRISPASR_MODEL_CANARY_QWEN not set");

    canary_qwen_context* ctx = open_ctx(model);
    REQUIRE(ctx != nullptr);
    canary_qwen_free(ctx);
}

TEST_CASE("canary-qwen: JFK English transcription (verbatim, no echo leak)", "[canary-qwen][.live]") {
    std::string model = get_env("CRISPASR_MODEL_CANARY_QWEN");
    if (model.empty())
        SKIP("CRISPASR_MODEL_CANARY_QWEN not set");

    auto pcm = load_wav_16k_mono("samples/jfk.wav");
    REQUIRE(!pcm.empty());

    canary_qwen_context* ctx = open_ctx(model);
    REQUIRE(ctx != nullptr);

    char* text = canary_qwen_transcribe(ctx, pcm.data(), (int)pcm.size());
    REQUIRE(text != nullptr);
    std::string got(text);
    std::free(text);
    INFO("Transcription: " << got);

    CHECK(contains_ci(got, "fellow americans"));
    CHECK(contains_ci(got, "country"));
    // #247: no instruction-echo leaked into a normal transcript.
    CHECK_FALSE(contains_ci(got, "transcript"));
    CHECK_FALSE(contains_ci(got, "pass"));

    canary_qwen_free(ctx);
}

TEST_CASE("canary-qwen #247: short window must not leak 'Transcript'/'PASS'", "[canary-qwen][.live]") {
    std::string model = get_env("CRISPASR_MODEL_CANARY_QWEN");
    if (model.empty())
        SKIP("CRISPASR_MODEL_CANARY_QWEN not set");

    auto pcm = load_wav_16k_mono("samples/jfk.wav");
    REQUIRE(!pcm.empty());

    canary_qwen_context* ctx = open_ctx(model);
    REQUIRE(ctx != nullptr);

    // Windows sliced from the start of the utterance. The <=0.35 s ones are the
    // degenerate windows (FastConformer 8x subsampling → T_enc<=5) that pre-fix
    // decoded to "Transcript"/"Okay" (verified identical in the NeMo reference);
    // the gate + echo strip must make them empty. The ~0.45 s window has enough
    // frames (T_enc>=6) to carry a real fragment ("And") — it must transcribe,
    // not be blanked — but must STILL never contain a meta echo. The one
    // invariant across ALL windows: no "Transcript"/"PASS" leak.
    const int SR = 16000;
    struct WinCase {
        double secs;
        bool degenerate; // below the frame gate → expect empty
    };
    for (WinCase w : {WinCase{0.10, true}, WinCase{0.15, true}, WinCase{0.20, true}, WinCase{0.30, true},
                      WinCase{0.35, true}, WinCase{0.45, false}}) {
        int n = (int)(w.secs * SR);
        if (n > (int)pcm.size())
            n = (int)pcm.size();
        canary_qwen_result* r = canary_qwen_transcribe_ex(ctx, pcm.data(), n);
        REQUIRE(r != nullptr);
        std::string got(r->text ? r->text : "");
        INFO("window " << w.secs << "s -> '" << got << "' (" << r->n_tokens << " tokens)");
        // Invariant for every window: no instruction-echo leak.
        CHECK_FALSE(contains_ci(got, "transcript"));
        CHECK_FALSE(contains_ci(got, "pass"));
        if (w.degenerate) {
            // Gated → empty transcript with no dangling echo tokens (#218: the
            // tokens array stays consistent with the text).
            CHECK(got.empty());
            CHECK(r->n_tokens == 0);
        }
        canary_qwen_result_free(r);
    }

    canary_qwen_free(ctx);
}
