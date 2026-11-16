// Live end-to-end test for crispasr_diarize_segments_turns_abi (issue #395).
//
// The turn-forwarding ABI is what lets a caller split one of its OWN segments
// that spans a speaker change: labelling alone can never resolve finer than
// the segment grid the caller sent in. Only FoxNose derives turns, and FoxNose
// needs a real speaker embedder — so the parts that can be checked without a
// model live in tests/test-session-abi-nulls.cpp, and everything that needs
// actual turns lives here.
//
// What this pins, none of which the model-free contract test can reach:
//   - turns come back at all, well-formed, inside the audio, in order;
//   - the truncation protocol (rc 2 + full required count + the first
//     n_turns_cap turns still written);
//   - the count-only query agrees with the filled buffer;
//   - turns are on the SAME absolute timeline as the caller's segments —
//     shifting slice_t0_cs shifts every turn by exactly that much;
//   - the segment labels are byte-identical to crispasr_diarize_segments_abi,
//     i.e. this is additive plumbing and not a behaviour change.
//
// Opt-in; skips cleanly when either env var is unset:
//
//   CRISPASR_TEST_FOXNOSE_WAV=samples/multispeaker.wav                (required)
//   CRISPASR_TEST_FOXNOSE_EMBEDDER=~/.cache/crispasr/wespeaker-...gguf (required)
//
// The wav must be 16 kHz 16-bit PCM (mono or stereo; stereo is downmixed).

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

extern "C" {
// Hand-written mirrors of the ABI structs in src/crispasr_c_api.cpp, kept
// byte-identical on purpose (see test-session-abi-nulls.cpp for why).
struct diarize_seg_abi {
    int64_t t0_cs;
    int64_t t1_cs;
    int32_t speaker;
    int32_t _pad;
};
struct diarize_turn_abi {
    int64_t t0_cs;
    int64_t t1_cs;
    int32_t speaker;
    int32_t _pad;
};
struct diarize_opts_abi {
    int32_t method;
    int32_t n_threads;
    int64_t slice_t0_cs;
    const char* pyannote_model_path;
    const char* foxnose_embedder_path;
    int32_t min_speakers;
    int32_t max_speakers;
    int32_t num_speakers;
    int32_t _pad2;
};
int crispasr_diarize_segments_abi(const float* left_pcm, const float* right_pcm, int32_t n_samples, int32_t is_stereo,
                                  diarize_seg_abi* segs, int32_t n_segs, const diarize_opts_abi* opts);
int crispasr_diarize_segments_turns_abi(const float* left_pcm, const float* right_pcm, int32_t n_samples,
                                        int32_t is_stereo, diarize_seg_abi* segs, int32_t n_segs,
                                        const diarize_opts_abi* opts, diarize_turn_abi* out_turns, int32_t n_turns_cap,
                                        int32_t* out_n_turns);
}

namespace {

// Minimal 16-bit PCM WAV loader, mono/stereo → float32 mono at the file's
// native rate. Same shape as tests/test-diarize-pyannote-live.cpp.
bool load_wav_mono(const std::string& path, std::vector<float>& out, int& sample_rate) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f)
        return false;

    char riff[4], wave[4];
    uint32_t file_size = 0;
    if (std::fread(riff, 1, 4, f) != 4 || std::fread(&file_size, 4, 1, f) != 1 || std::fread(wave, 1, 4, f) != 4 ||
        std::memcmp(riff, "RIFF", 4) != 0 || std::memcmp(wave, "WAVE", 4) != 0) {
        std::fclose(f);
        return false;
    }

    int channels = 0, bits = 0;
    sample_rate = 0;
    std::vector<int16_t> pcm;

    while (true) {
        char id[4];
        uint32_t sz = 0;
        if (std::fread(id, 1, 4, f) != 4 || std::fread(&sz, 4, 1, f) != 1)
            break;
        if (std::memcmp(id, "fmt ", 4) == 0) {
            uint16_t fmt = 0, ch = 0;
            uint32_t sr = 0;
            uint16_t bps = 0;
            std::fread(&fmt, 2, 1, f);
            std::fread(&ch, 2, 1, f);
            std::fread(&sr, 4, 1, f);
            std::fseek(f, 4 + 2, SEEK_CUR);
            std::fread(&bps, 2, 1, f);
            channels = ch;
            sample_rate = (int)sr;
            bits = bps;
            if (sz > 16)
                std::fseek(f, sz - 16, SEEK_CUR);
        } else if (std::memcmp(id, "data", 4) == 0) {
            int n = (int)(sz / (bits / 8) / channels);
            pcm.resize((size_t)n * channels);
            std::fread(pcm.data(), bits / 8, (size_t)n * channels, f);
            break;
        } else {
            std::fseek(f, sz, SEEK_CUR);
        }
    }
    std::fclose(f);

    if (bits != 16 || channels < 1 || sample_rate <= 0 || pcm.empty())
        return false;

    out.resize(pcm.size() / channels);
    for (size_t i = 0; i < out.size(); i++) {
        int32_t acc = 0;
        for (int c = 0; c < channels; c++)
            acc += pcm[i * channels + c];
        out[i] = (float)acc / (float)channels / 32768.0f;
    }
    return true;
}

// The caller grid this test hands in: fixed 3 s blocks over the whole file.
// Deliberately coarse — that IS the failure mode #395 is about, and the turns
// are what a caller would use to cut these blocks up afterwards.
std::vector<diarize_seg_abi> block_grid(int64_t n_samples, int sr, int64_t slice_t0_cs) {
    std::vector<diarize_seg_abi> segs;
    const int64_t total_cs = n_samples * 100 / sr;
    for (int64_t t = 0; t + 300 <= total_cs; t += 300)
        segs.push_back({slice_t0_cs + t, slice_t0_cs + t + 300, -1, 0});
    return segs;
}

struct Fixture {
    std::vector<float> pcm;
    int sr = 0;
    std::string embedder;
};

// Returns false when the fixtures are not configured, so each test can SKIP.
bool load_fixture(Fixture& fx) {
    const char* wav = std::getenv("CRISPASR_TEST_FOXNOSE_WAV");
    const char* emb = std::getenv("CRISPASR_TEST_FOXNOSE_EMBEDDER");
    if (!wav || !*wav || !emb || !*emb)
        return false;
    if (!load_wav_mono(wav, fx.pcm, fx.sr))
        return false;
    fx.embedder = emb;
    return true;
}

diarize_opts_abi foxnose_opts(const Fixture& fx, int64_t slice_t0_cs) {
    diarize_opts_abi o = {};
    o.method = 4; // FoxNose
    o.n_threads = 4;
    o.slice_t0_cs = slice_t0_cs;
    o.foxnose_embedder_path = fx.embedder.c_str();
    o.num_speakers = 2; // pinned: the count estimator is not what's under test
    return o;
}

} // namespace

TEST_CASE("FoxNose turns reach the caller through the ABI", "[live][diarize]") {
    Fixture fx;
    if (!load_fixture(fx))
        SKIP("set CRISPASR_TEST_FOXNOSE_WAV + CRISPASR_TEST_FOXNOSE_EMBEDDER to run this");
    REQUIRE(fx.sr == 16000);

    auto segs = block_grid((int64_t)fx.pcm.size(), fx.sr, 0);
    REQUIRE(segs.size() >= 2);
    diarize_opts_abi opts = foxnose_opts(fx, 0);

    // One turn per 0.6 s of audio is well above what the pipeline can emit
    // (its embedding hop), so this cap cannot truncate.
    const int32_t cap = (int32_t)((int64_t)fx.pcm.size() / fx.sr / 0.6) + (int32_t)segs.size() + 16;
    std::vector<diarize_turn_abi> turns((size_t)cap);
    int32_t n_turns = -1;

    REQUIRE(crispasr_diarize_segments_turns_abi(fx.pcm.data(), nullptr, (int32_t)fx.pcm.size(), 0, segs.data(),
                                                (int32_t)segs.size(), &opts, turns.data(), cap, &n_turns) == 0);
    REQUIRE(n_turns > 0);
    REQUIRE(n_turns <= cap);

    const int64_t total_cs = (int64_t)fx.pcm.size() * 100 / fx.sr;
    int64_t prev_end = -1;
    std::set<int> speakers;
    for (int32_t i = 0; i < n_turns; i++) {
        const diarize_turn_abi& t = turns[(size_t)i];
        CHECK(t.t1_cs > t.t0_cs);
        CHECK(t.t0_cs >= 0);
        CHECK(t.t1_cs <= total_cs + 1); // +1 cs of rounding slack
        CHECK(t.speaker >= 0);          // turns are always labelled, unlike segs
        CHECK(t.t0_cs >= prev_end);     // sorted and non-overlapping
        prev_end = t.t1_cs;
        speakers.insert(t.speaker);
    }
    CHECK(speakers.size() == 2); // num_speakers was pinned to 2

    // The whole point of #395: at least one of the caller's segments covers
    // two DIFFERENT speakers, so labelling it can only ever be half right —
    // and the turns are what a caller splits it on. If this ever stops
    // holding for the fixture, the test has lost the case it was written for.
    int straddling = 0;
    for (const auto& g : segs) {
        std::set<int> spk;
        for (int32_t i = 0; i < n_turns; i++) {
            const diarize_turn_abi& t = turns[(size_t)i];
            const int64_t ov = std::min(g.t1_cs, t.t1_cs) - std::max(g.t0_cs, t.t0_cs);
            if (ov > 20) // >0.2 s of real overlap, not a boundary graze
                spk.insert(t.speaker);
        }
        if (spk.size() > 1)
            straddling++;
    }
    INFO("segments straddling a speaker change: " << straddling << " of " << segs.size());
    CHECK(straddling > 0);
}

TEST_CASE("a short turn buffer truncates with 2 and reports what was needed", "[live][diarize]") {
    Fixture fx;
    if (!load_fixture(fx))
        SKIP("set CRISPASR_TEST_FOXNOSE_WAV + CRISPASR_TEST_FOXNOSE_EMBEDDER to run this");

    auto segs = block_grid((int64_t)fx.pcm.size(), fx.sr, 0);
    diarize_opts_abi opts = foxnose_opts(fx, 0);

    // Pass 1: count only, no buffer. Nothing can be truncated, so rc 0.
    int32_t needed = -1;
    REQUIRE(crispasr_diarize_segments_turns_abi(fx.pcm.data(), nullptr, (int32_t)fx.pcm.size(), 0, segs.data(),
                                                (int32_t)segs.size(), &opts, nullptr, 0, &needed) == 0);
    REQUIRE(needed > 1);

    // Pass 2: deliberately one short. rc 2, the count is still the FULL
    // requirement (that is what makes size-and-retry possible), and the turns
    // that did fit are written.
    const int32_t cap = needed - 1;
    std::vector<diarize_turn_abi> turns((size_t)cap);
    int32_t n_turns = -1;
    for (auto& s : segs)
        s.speaker = -1;
    REQUIRE(crispasr_diarize_segments_turns_abi(fx.pcm.data(), nullptr, (int32_t)fx.pcm.size(), 0, segs.data(),
                                                (int32_t)segs.size(), &opts, turns.data(), cap, &n_turns) == 2);
    CHECK(n_turns == needed);
    CHECK(turns[0].t1_cs > turns[0].t0_cs);
    CHECK(turns[(size_t)cap - 1].t1_cs > turns[(size_t)cap - 1].t0_cs);

    // Truncation is about the TURNS only — the segments came back labelled.
    bool any_labelled = false;
    for (const auto& s : segs)
        any_labelled = any_labelled || s.speaker >= 0;
    CHECK(any_labelled);
}

TEST_CASE("turns share the caller's absolute timeline, not the buffer's", "[live][diarize]") {
    Fixture fx;
    if (!load_fixture(fx))
        SKIP("set CRISPASR_TEST_FOXNOSE_WAV + CRISPASR_TEST_FOXNOSE_EMBEDDER to run this");

    const int64_t shift_cs = 1000; // the buffer starts 10 s into the recording
    const int32_t cap = (int32_t)((int64_t)fx.pcm.size() / fx.sr / 0.6) + 64;

    auto segs_a = block_grid((int64_t)fx.pcm.size(), fx.sr, 0);
    diarize_opts_abi opts_a = foxnose_opts(fx, 0);
    std::vector<diarize_turn_abi> turns_a((size_t)cap);
    int32_t n_a = -1;
    REQUIRE(crispasr_diarize_segments_turns_abi(fx.pcm.data(), nullptr, (int32_t)fx.pcm.size(), 0, segs_a.data(),
                                                (int32_t)segs_a.size(), &opts_a, turns_a.data(), cap, &n_a) == 0);

    // Same audio, same segments, both shifted by shift_cs: identical work,
    // so the turns must come back shifted by exactly the same amount.
    auto segs_b = block_grid((int64_t)fx.pcm.size(), fx.sr, shift_cs);
    diarize_opts_abi opts_b = foxnose_opts(fx, shift_cs);
    std::vector<diarize_turn_abi> turns_b((size_t)cap);
    int32_t n_b = -1;
    REQUIRE(crispasr_diarize_segments_turns_abi(fx.pcm.data(), nullptr, (int32_t)fx.pcm.size(), 0, segs_b.data(),
                                                (int32_t)segs_b.size(), &opts_b, turns_b.data(), cap, &n_b) == 0);

    REQUIRE(n_a == n_b);
    for (int32_t i = 0; i < n_a; i++) {
        CHECK(turns_b[(size_t)i].t0_cs == turns_a[(size_t)i].t0_cs + shift_cs);
        CHECK(turns_b[(size_t)i].t1_cs == turns_a[(size_t)i].t1_cs + shift_cs);
        CHECK(turns_b[(size_t)i].speaker == turns_a[(size_t)i].speaker);
    }
}

TEST_CASE("asking for turns does not change the labels", "[live][diarize]") {
    Fixture fx;
    if (!load_fixture(fx))
        SKIP("set CRISPASR_TEST_FOXNOSE_WAV + CRISPASR_TEST_FOXNOSE_EMBEDDER to run this");

    auto segs_old = block_grid((int64_t)fx.pcm.size(), fx.sr, 0);
    auto segs_new = block_grid((int64_t)fx.pcm.size(), fx.sr, 0);
    diarize_opts_abi opts = foxnose_opts(fx, 0);

    REQUIRE(crispasr_diarize_segments_abi(fx.pcm.data(), nullptr, (int32_t)fx.pcm.size(), 0, segs_old.data(),
                                          (int32_t)segs_old.size(), &opts) == 0);

    const int32_t cap = (int32_t)((int64_t)fx.pcm.size() / fx.sr / 0.6) + 64;
    std::vector<diarize_turn_abi> turns((size_t)cap);
    int32_t n_turns = -1;
    REQUIRE(crispasr_diarize_segments_turns_abi(fx.pcm.data(), nullptr, (int32_t)fx.pcm.size(), 0, segs_new.data(),
                                                (int32_t)segs_new.size(), &opts, turns.data(), cap, &n_turns) == 0);

    REQUIRE(segs_old.size() == segs_new.size());
    for (size_t i = 0; i < segs_old.size(); i++)
        CHECK(segs_new[i].speaker == segs_old[i].speaker);
}
