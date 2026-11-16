// Issue #89 regression guard — parakeet-ja long-form coverage.
//
// The JA FastConformer encoder collapses past ~12 s of attention context on
// real speech and blanks utterances whenever enough context follows them.
// The shipped pipeline (VAD/energy slices capped at 12 s + one exact pass
// per slice + gap-fill second pass — see HISTORY 2026-07-04) recovered the
// reporter's clips from ~56 % to 96-97 % content recall. This test pins the
// library-level (session ABI) behavior: the reazon baseball fixture
// concatenated ×3 (42.2 s — crosses the 30 s auto-chunk threshold and the
// 12 s slice cap) must keep all three repetitions of the keyword 岡本 and
// keep emitting words into the third repetition. Before the fix, the same
// input recovered 1/3 (single-pass, streamed) or 1-2/3 (chunked).
//
// Requires:
//   CRISPASR_MODEL_PARAKEET_JA    — parakeet-tdt-0.6b-ja GGUF
//   CRISPASR_FIXTURE_PARAKEET_JA  — reazon_baseball_14s/audio.wav from
//       cstr/crispasr-regression-fixtures (hf download
//       cstr/crispasr-regression-fixtures
//       parakeet-tdt-0.6b-ja/reazon_baseball_14s/audio.wav)
// SKIPs cleanly when either is missing.

#include <catch2/catch_test_macros.hpp>

#include "crispasr_session.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static std::vector<float> load_wav_16k(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f)
        return {};
    fseek(f, 0, SEEK_END);
    long sz = ftell(f) - 44;
    fseek(f, 44, SEEK_SET);
    std::vector<int16_t> raw(sz / 2);
    size_t n = fread(raw.data(), 2, raw.size(), f);
    (void)n;
    fclose(f);
    std::vector<float> pcm(raw.size());
    for (size_t i = 0; i < raw.size(); i++)
        pcm[i] = raw[i] / 32768.0f;
    return pcm;
}

static int count_occurrences(const std::string& hay, const std::string& needle) {
    int n = 0;
    for (size_t pos = hay.find(needle); pos != std::string::npos; pos = hay.find(needle, pos + needle.size()))
        n++;
    return n;
}

TEST_CASE("parakeet-ja long-form keeps interior content (issue #89)", "[integration][parakeet-ja-longform]") {
    const char* model_path = std::getenv("CRISPASR_MODEL_PARAKEET_JA");
    if (!model_path || !*model_path) {
        SKIP("CRISPASR_MODEL_PARAKEET_JA not set");
    }
    const char* wav_path = std::getenv("CRISPASR_FIXTURE_PARAKEET_JA");
    if (!wav_path || !*wav_path) {
        SKIP("CRISPASR_FIXTURE_PARAKEET_JA not set");
    }
    {
        FILE* f = fopen(model_path, "rb");
        if (!f)
            SKIP("model file not present: " << model_path);
        fclose(f);
        f = fopen(wav_path, "rb");
        if (!f)
            SKIP("fixture wav not present: " << wav_path);
        fclose(f);
    }

    auto one = load_wav_16k(wav_path);
    REQUIRE(one.size() > 16000 * 10); // fixture is 14.06 s
    std::vector<float> pcm;
    pcm.reserve(one.size() * 3);
    for (int i = 0; i < 3; i++)
        pcm.insert(pcm.end(), one.begin(), one.end());
    const double dur_s = (double)pcm.size() / 16000.0;

    crispasr_session* s = crispasr_session_open(model_path, 4);
    REQUIRE(s != nullptr);

    crispasr_session_result* r = crispasr_session_transcribe(s, pcm.data(), (int)pcm.size());
    REQUIRE(r != nullptr);

    std::string text;
    int64_t last_t1_cs = 0;
    const int n_segs = crispasr_session_result_n_segments(r);
    for (int i = 0; i < n_segs; i++) {
        const char* t = crispasr_session_result_segment_text(r, i);
        if (t)
            text += t;
        last_t1_cs = std::max(last_t1_cs, crispasr_session_result_segment_t1(r, i));
    }
    INFO("duration=" << dur_s << "s  transcript(" << text.size() << " bytes): " << text);

    // All three repetitions of the keyword survive (was 1/3 pre-fix).
    CHECK(count_occurrences(text, "岡本") >= 3);
    // Words keep flowing into the third repetition (starts at ~28.1 s).
    CHECK(last_t1_cs >= 2900);
    // Content floor: one repetition is ~50 JA chars (~150 bytes UTF-8);
    // require well over two repetitions' worth so a silent interior drop
    // of a whole repetition fails the test.
    CHECK(text.size() >= 350);

    crispasr_session_result_free(r);
    crispasr_session_close(s);
}
