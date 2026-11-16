/*
 * test-audio-formats.cpp — Catch2 tests for crispasr_audio_load with
 * extended format support: AU/SND (µ-law), AMR-NB, WebM (Opus).
 *
 * Each format is decoded via the crispasr_audio_load C ABI and compared
 * against the WAV reference (jfk.wav) decoded through the same path.
 * Lossy codecs won't be bit-exact; we check length similarity, non-silence,
 * and cross-correlation above a threshold.
 */

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

// The C ABI under test — declared in crispasr.h, but we forward-declare to
// avoid pulling the full header into a test TU that doesn't need it.
extern "C" int crispasr_audio_load(const char* path, float** out_pcm, int* out_samples, int* out_sample_rate);
extern "C" int crispasr_audio_load_stereo(const char* path, float** out_left, float** out_right, int* out_samples,
                                          int* out_sample_rate, int* out_channels);
extern "C" void crispasr_audio_free(float* pcm);

// Header-only WAV reader used directly by the indextts/voxcpm2 --voice paths
// (crispasr_audio_load routes WAV through miniaudio, not this), so its own
// malicious-size clamp needs a direct test.
#include "core/wav_reader.h"

// setenv/unsetenv are POSIX; MSVC and MinGW need the shim.
#include "portable_env.h"

// ── helpers ──────────────────────────────────────────────────────────

#ifndef SAMPLES_DIR
#define SAMPLES_DIR "."
#endif

static std::string sample(const char* name) {
    return std::string(SAMPLES_DIR) + "/" + name;
}

// Normalised cross-correlation between two float signals (peak over ±lag window).
static double cross_correlation(const float* a, int na, const float* b, int nb,
                                int max_lag = 480 /* 30 ms at 16 kHz */) {
    int n = std::min(na, nb);
    if (n == 0)
        return 0.0;

    double ea = 0, eb = 0;
    for (int i = 0; i < n; ++i) {
        ea += (double)a[i] * a[i];
        eb += (double)b[i] * b[i];
    }
    double norm = std::sqrt(ea * eb);
    if (norm < 1e-12)
        return 0.0;

    double best = -1.0;
    for (int lag = -max_lag; lag <= max_lag; ++lag) {
        double sum = 0;
        for (int i = 0; i < n; ++i) {
            int j = i + lag;
            if (j < 0 || j >= n)
                continue;
            sum += (double)a[i] * b[j];
        }
        double cc = sum / norm;
        if (cc > best)
            best = cc;
    }
    return best;
}

// Check that a signal is not silent.
static bool has_energy(const float* pcm, int n, double min_rms = 0.001) {
    if (n == 0)
        return false;
    double sum_sq = 0;
    for (int i = 0; i < n; ++i)
        sum_sq += (double)pcm[i] * pcm[i];
    return std::sqrt(sum_sq / n) >= min_rms;
}

// ── reference loader ─────────────────────────────────────────────────

struct AudioRef {
    float* pcm = nullptr;
    int samples = 0;
    int sample_rate = 0;

    ~AudioRef() {
        if (pcm)
            crispasr_audio_free(pcm);
    }
};

static AudioRef load_ref() {
    AudioRef ref;
    int rc = crispasr_audio_load(sample("jfk.wav").c_str(), &ref.pcm, &ref.samples, &ref.sample_rate);
    REQUIRE(rc == 0);
    REQUIRE(ref.pcm != nullptr);
    REQUIRE(ref.samples > 100000); // ~11s at 16kHz
    REQUIRE(ref.sample_rate == 16000);
    return ref;
}

// ── test cases ───────────────────────────────────────────────────────

TEST_CASE("crispasr_audio_load decodes WAV reference", "[audio][unit]") {
    auto ref = load_ref();
    REQUIRE(has_energy(ref.pcm, ref.samples));
}

TEST_CASE("crispasr_audio_load decodes AU (µ-law)", "[audio][unit][au]") {
    auto ref = load_ref();

    float* pcm = nullptr;
    int samples = 0, rate = 0;
    int rc = crispasr_audio_load(sample("jfk.au").c_str(), &pcm, &samples, &rate);
    REQUIRE(rc == 0);
    REQUIRE(pcm != nullptr);
    REQUIRE(rate == 16000);
    REQUIRE(has_energy(pcm, samples));

    // Length within 5% of reference (lossy + resample)
    double ratio = (double)samples / ref.samples;
    INFO("AU length ratio: " << ratio);
    REQUIRE(ratio > 0.90);
    REQUIRE(ratio < 1.10);

    // Cross-correlation — µ-law at 8kHz is lossy, 0.70 is reasonable
    double cc = cross_correlation(ref.pcm, ref.samples, pcm, samples);
    INFO("AU cross-correlation: " << cc);
    REQUIRE(cc > 0.70);

    crispasr_audio_free(pcm);
}

// Regression: a crafted Sun-AU header must not drive an unbounded allocation
// or an out-of-bounds read. crispasr_au_decode now clamps data_size to the
// real file size and rejects a data_offset past EOF. Pre-fix, section 1
// allocated ~4 GB from the untrusted data_size (DoS / unhandled bad_alloc)
// and section 2 underflowed end-cur into a huge size_t. We only require the
// process to survive and behave sanely — under ASan this also proves no OOB.
TEST_CASE("crispasr_audio_load rejects malicious AU sizes without over-allocating", "[audio][unit][au]") {
    auto put_be32 = [](std::vector<uint8_t>& b, uint32_t v) {
        b.push_back((uint8_t)(v >> 24));
        b.push_back((uint8_t)(v >> 16));
        b.push_back((uint8_t)(v >> 8));
        b.push_back((uint8_t)v);
    };
    auto write_file = [](const char* p, const std::vector<uint8_t>& bytes) {
        FILE* f = std::fopen(p, "wb");
        REQUIRE(f != nullptr);
        std::fwrite(bytes.data(), 1, bytes.size(), f);
        std::fclose(f);
    };
    const char* path = "crispasr_au_regression_tmp.au";

    SECTION("data_size claims ~4 GB in a tiny file") {
        std::vector<uint8_t> b;
        put_be32(b, 0x2e736e64u); // ".snd"
        put_be32(b, 24);          // data_offset
        put_be32(b, 0xFFFFFFFEu); // data_size ~4 GB (not the 0xFFFFFFFF sentinel)
        put_be32(b, 1);           // encoding = µ-law
        put_be32(b, 8000);        // sample_rate
        put_be32(b, 1);           // channels
        for (int i = 0; i < 8; i++)
            b.push_back(0x7F); // 8 bytes of audio
        write_file(path, b);

        float* pcm = nullptr;
        int samples = 0, rate = 0;
        // Must return without a 4 GB allocation; result is a tiny clip or an error.
        int rc = crispasr_audio_load(path, &pcm, &samples, &rate);
        if (rc == 0) {
            CHECK(samples < 1000);
            crispasr_audio_free(pcm);
        }
    }

    SECTION("data_offset past EOF (size_t underflow path)") {
        std::vector<uint8_t> b;
        put_be32(b, 0x2e736e64u);
        put_be32(b, 0xFFFFFF00u); // data_offset far past EOF
        put_be32(b, 0);           // data_size = 0 → "compute from file" branch
        put_be32(b, 1);
        put_be32(b, 8000);
        put_be32(b, 1);
        for (int i = 0; i < 8; i++)
            b.push_back(0x7F);
        write_file(path, b);

        float* pcm = nullptr;
        int samples = 0, rate = 0;
        int rc = crispasr_audio_load(path, &pcm, &samples, &rate);
        CHECK(rc != 0); // offset past EOF → clean rejection, no underflow
    }

    std::remove(path);
}

// Regression: read_wav_mono_pcm16 sized its int16 buffer from the untrusted
// `data` chunk_size — a tiny WAV claiming data_size ~2 GB forced a ~2 GB alloc.
// Now clamped to the real file size. Require survival + a tiny result.
TEST_CASE("read_wav_mono_pcm16 clamps malicious data_size without over-allocating", "[audio][unit]") {
    std::vector<uint8_t> w;
    auto le32 = [&](uint32_t v) {
        for (int i = 0; i < 4; i++)
            w.push_back((uint8_t)(v >> (8 * i)));
    };
    auto le16 = [&](uint16_t v) {
        w.push_back((uint8_t)v);
        w.push_back((uint8_t)(v >> 8));
    };
    auto tag = [&](const char* s) {
        for (int i = 0; i < 4; i++)
            w.push_back((uint8_t)s[i]);
    };
    tag("RIFF");
    le32(0x7FFFFFFFu);
    tag("WAVE");
    tag("fmt ");
    le32(16);
    le16(1);
    le16(1);
    le32(16000);
    le32(32000);
    le16(2);
    le16(16);
    tag("data");
    le32(0x7FFFFFFEu); // claims ~2 GB
    for (int i = 0; i < 8; i++)
        w.push_back(0x11); // but only 8 real bytes
    const char* path = "crispasr_wav_regression_tmp.wav";
    {
        FILE* f = std::fopen(path, "wb");
        REQUIRE(f != nullptr);
        std::fwrite(w.data(), 1, w.size(), f);
        std::fclose(f);
    }

    std::vector<float> pcm;
    int rate = 0;
    bool ok = crispasr::core::read_wav_mono_pcm16(path, pcm, rate);
    if (ok)
        CHECK(pcm.size() < 1000);
    std::remove(path);
}

// Reachability regression: a crafted MP4 whose stsz box claims a ~4 billion
// sample count must flow through crispasr_audio_load -> crispasr_m4a_decode
// (reached after AudioToolbox/AU/AMR/WebM reject it) without the multi-GB
// resize() (now clamped to box bytes). Builds the minimal
// ftyp + moov{trak{mdia{minf{stbl{stsz}}}}} nesting the box parser walks.
TEST_CASE("crispasr_audio_load survives a malicious MP4 stsz count", "[audio][unit]") {
    auto box = [](const char* type, const std::vector<uint8_t>& payload) {
        std::vector<uint8_t> b;
        uint32_t size = 8u + (uint32_t)payload.size();
        for (int i = 3; i >= 0; i--)
            b.push_back((uint8_t)(size >> (8 * i)));
        for (int i = 0; i < 4; i++)
            b.push_back((uint8_t)type[i]);
        b.insert(b.end(), payload.begin(), payload.end());
        return b;
    };
    auto be32 = [](std::vector<uint8_t>& b, uint32_t v) {
        for (int i = 3; i >= 0; i--)
            b.push_back((uint8_t)(v >> (8 * i)));
    };

    std::vector<uint8_t> stsz_p;
    be32(stsz_p, 0);           // version + flags
    be32(stsz_p, 0);           // sample_size = 0 (non-uniform)
    be32(stsz_p, 0xFFFFFFFFu); // sample_count ~4 billion, box has no entries
    std::vector<uint8_t> moov = box("moov", box("trak", box("mdia", box("minf", box("stbl", box("stsz", stsz_p))))));
    std::vector<uint8_t> ftyp_p = {'M', '4', 'A', ' ', 'i', 's', 'o', 'm'};
    std::vector<uint8_t> ftyp = box("ftyp", ftyp_p);

    std::vector<uint8_t> file;
    file.insert(file.end(), ftyp.begin(), ftyp.end());
    file.insert(file.end(), moov.begin(), moov.end());

    const char* path = "crispasr_mp4_regression_tmp.m4a";
    {
        FILE* f = std::fopen(path, "wb");
        REQUIRE(f != nullptr);
        std::fwrite(file.data(), 1, file.size(), f);
        std::fclose(f);
    }

    float* pcm = nullptr;
    int n = 0, sr = 0;
    // Must return (no ~16 GB resize / crash). Result is an error (no audio data).
    int rc = crispasr_audio_load(path, &pcm, &n, &sr);
    if (rc == 0)
        crispasr_audio_free(pcm);
    SUCCEED("survived malicious MP4 stsz without over-allocating");
    std::remove(path);
}

TEST_CASE("crispasr_audio_load decodes AMR-NB", "[audio][unit][amr]") {
    float* pcm = nullptr;
    int samples = 0, rate = 0;
    int rc = crispasr_audio_load(sample("jfk.amr").c_str(), &pcm, &samples, &rate);

    // AMR support is optional (CRISPASR_HAVE_AMR). If the build doesn't
    // include it, rc will be -2 and we skip the test gracefully.
    if (rc == -2) {
        WARN("AMR decoder not available (CRISPASR_HAVE_AMR not set) — skipping");
        return;
    }
    REQUIRE(rc == 0);
    REQUIRE(pcm != nullptr);
    REQUIRE(rate == 16000);
    REQUIRE(has_energy(pcm, samples));

    auto ref = load_ref();

    // AMR-NB at 12.2kbps is heavily lossy — length within 10%
    double ratio = (double)samples / ref.samples;
    INFO("AMR length ratio: " << ratio);
    REQUIRE(ratio > 0.90);
    REQUIRE(ratio < 1.10);

    // Cross-correlation — AMR is very lossy, 0.50 is a generous floor
    double cc = cross_correlation(ref.pcm, ref.samples, pcm, samples);
    INFO("AMR cross-correlation: " << cc);
    REQUIRE(cc > 0.50);

    crispasr_audio_free(pcm);
}

TEST_CASE("crispasr_audio_load decodes WebM (Opus)", "[audio][unit][webm]") {
    float* pcm = nullptr;
    int samples = 0, rate = 0;
    int rc = crispasr_audio_load(sample("jfk.webm").c_str(), &pcm, &samples, &rate);

    // WebM/Opus requires CRISPASR_HAVE_OPUS. The Opus custom backend might
    // handle it via miniaudio, or our EBML demuxer fallback kicks in.
    if (rc == -2) {
        WARN("WebM decoder not available — skipping");
        return;
    }
    REQUIRE(rc == 0);
    REQUIRE(pcm != nullptr);
    REQUIRE(rate == 16000);
    REQUIRE(has_energy(pcm, samples));

    auto ref = load_ref();

    // Length within 5%
    double ratio = (double)samples / ref.samples;
    INFO("WebM length ratio: " << ratio);
    REQUIRE(ratio > 0.90);
    REQUIRE(ratio < 1.10);

    // Cross-correlation — Opus is high quality, expect >0.80
    double cc = cross_correlation(ref.pcm, ref.samples, pcm, samples);
    INFO("WebM cross-correlation: " << cc);
    REQUIRE(cc > 0.80);

    crispasr_audio_free(pcm);
}

// Is the WebM/Opus path built at all? The live-WebM cases below must not
// silently skip on a decode failure — that is the very regression they guard —
// so they ask this first and only skip when the *plain* WebM sample fails too.
static bool webm_opus_available() {
    float* pcm = nullptr;
    int samples = 0, rate = 0;
    int rc = crispasr_audio_load(sample("jfk.webm").c_str(), &pcm, &samples, &rate);
    if (pcm)
        crispasr_audio_free(pcm);
    return rc == 0;
}

TEST_CASE("crispasr_audio_load decodes live/streaming WebM (Opus, unknown-size clusters)", "[audio][unit][webm]") {
    // Issue #417 — Chrome MediaRecorder (libwebm mkvmuxer against a
    // non-seekable writer) emits the Segment AND every Cluster with the EBML
    // "unknown size" marker, one Cluster per timeslice. The demuxer used to
    // bound each Cluster by that marker read as a literal length, so it stopped
    // at the first Cluster and returned ~0.1 s of an 11 s recording.
    if (!webm_opus_available()) {
        WARN("WebM decoder not available — skipping");
        return;
    }

    float* pcm = nullptr;
    int samples = 0, rate = 0;
    int rc = crispasr_audio_load(sample("jfk-live.webm").c_str(), &pcm, &samples, &rate);
    REQUIRE(rc == 0);
    REQUIRE(pcm != nullptr);
    REQUIRE(rate == 16000);
    REQUIRE(has_energy(pcm, samples));

    auto ref = load_ref();

    // The whole recording, not just the first cluster. The pre-fix decoder
    // produced a ratio of ~0.008 here, so this bound is far tighter than the
    // defect it guards.
    double ratio = (double)samples / ref.samples;
    INFO("live WebM length ratio: " << ratio);
    REQUIRE(ratio > 0.98);
    REQUIRE(ratio < 1.02);

    double cc = cross_correlation(ref.pcm, ref.samples, pcm, samples);
    INFO("live WebM cross-correlation: " << cc);
    REQUIRE(cc > 0.80);

    crispasr_audio_free(pcm);
}

TEST_CASE("crispasr_audio_load handles the 1-byte EBML unknown-size marker", "[audio][unit][webm]") {
    // The unknown-size marker is "all data bits set", which is
    // length-dependent: the 1-byte form is 0xFF, whose *value* is 127. A
    // value-based test misreads it as a 127-byte element. No shipping muxer
    // writes the short form today, so derive it from the live fixture rather
    // than carrying a second one.
    if (!webm_opus_available()) {
        WARN("WebM decoder not available — skipping");
        return;
    }

    std::vector<uint8_t> data;
    {
        FILE* f = std::fopen(sample("jfk-live.webm").c_str(), "rb");
        REQUIRE(f != nullptr);
        std::fseek(f, 0, SEEK_END);
        long n = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        REQUIRE(n > 0);
        data.resize((size_t)n);
        REQUIRE(std::fread(data.data(), 1, data.size(), f) == data.size());
        std::fclose(f);
    }

    static const uint8_t kWide[8] = {0x01, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    std::vector<uint8_t> narrow;
    narrow.reserve(data.size());
    int rewritten = 0;
    for (size_t i = 0; i < data.size();) {
        if (i + 8 <= data.size() && std::memcmp(data.data() + i, kWide, 8) == 0) {
            narrow.push_back(0xff);
            i += 8;
            ++rewritten;
        } else {
            narrow.push_back(data[i]);
            ++i;
        }
    }
    INFO("rewritten unknown-size markers: " << rewritten);
    REQUIRE(rewritten > 100); // Segment + one per cluster

    // Fixed name: this is the only writer, and the content is deterministic,
    // so a concurrent ctest shard racing us would write the same bytes.
    const std::string tmp = (std::filesystem::temp_directory_path() / "crispasr-live-unknown-1b.webm").string();
    {
        FILE* f = std::fopen(tmp.c_str(), "wb");
        REQUIRE(f != nullptr);
        REQUIRE(std::fwrite(narrow.data(), 1, narrow.size(), f) == narrow.size());
        std::fclose(f);
    }

    float* pcm = nullptr;
    int samples = 0, rate = 0;
    int rc = crispasr_audio_load(tmp.c_str(), &pcm, &samples, &rate);
    std::remove(tmp.c_str());
    REQUIRE(rc == 0);
    REQUIRE(rate == 16000);
    REQUIRE(has_energy(pcm, samples));

    auto ref = load_ref();
    double ratio = (double)samples / ref.samples;
    INFO("1-byte-marker live WebM length ratio: " << ratio);
    REQUIRE(ratio > 0.98);
    REQUIRE(ratio < 1.02);

    crispasr_audio_free(pcm);
}

TEST_CASE("crispasr_audio_load decodes WebM (Vorbis)", "[audio][unit][webm]") {
    float* pcm = nullptr;
    int samples = 0, rate = 0;
    int rc = crispasr_audio_load(sample("jfk-vorbis.webm").c_str(), &pcm, &samples, &rate);

    if (rc == -2) {
        WARN("WebM/Vorbis decoder not available — skipping");
        return;
    }
    REQUIRE(rc == 0);
    REQUIRE(pcm != nullptr);
    REQUIRE(rate == 16000);
    REQUIRE(has_energy(pcm, samples));

    auto ref = load_ref();

    double ratio = (double)samples / ref.samples;
    INFO("WebM/Vorbis length ratio: " << ratio);
    REQUIRE(ratio > 0.90);
    REQUIRE(ratio < 1.10);

    double cc = cross_correlation(ref.pcm, ref.samples, pcm, samples);
    INFO("WebM/Vorbis cross-correlation: " << cc);
    REQUIRE(cc > 0.70);

    crispasr_audio_free(pcm);
}

TEST_CASE("crispasr_audio_load decodes M4A (AAC)", "[audio][unit][m4a]") {
    float* pcm = nullptr;
    int samples = 0, rate = 0;
    int rc = crispasr_audio_load(sample("jfk.m4a").c_str(), &pcm, &samples, &rate);

    // M4A/AAC requires either CRISPASR_HAVE_FDK_AAC (Linux) or Apple AudioToolbox
    if (rc == -2) {
        WARN("M4A/AAC decoder not available — skipping");
        return;
    }
    REQUIRE(rc == 0);
    REQUIRE(pcm != nullptr);
    REQUIRE(rate == 16000);
    REQUIRE(has_energy(pcm, samples));

    auto ref = load_ref();

    double ratio = (double)samples / ref.samples;
    INFO("M4A length ratio: " << ratio);
    REQUIRE(ratio > 0.90);
    REQUIRE(ratio < 1.10);

    // AAC is good quality. Encoder priming is stripped via elst parsing,
    // but a small residual offset (~240 samples) may remain from SBR/
    // resampling rounding. Use a moderate lag window (500 samples = 31ms).
    double cc = cross_correlation(ref.pcm, ref.samples, pcm, samples, 500);
    INFO("M4A cross-correlation: " << cc);
    REQUIRE(cc > 0.85);

    crispasr_audio_free(pcm);
}

TEST_CASE("crispasr_audio_load decodes ADTS AAC (glint)", "[audio][unit][aac]") {
    // Raw ADTS .aac (ffmpeg-encoded AAC-LC) decoded by the in-tree glint
    // decoder — cross-platform and always available (no runtime lib), so unlike
    // M4A this must succeed everywhere. ffmpeg-encoded -> glint-decoded is the
    // cross-reference roundtrip (HARD RULE #3).
    float* pcm = nullptr;
    int samples = 0, rate = 0;
    int rc = crispasr_audio_load(sample("jfk.aac").c_str(), &pcm, &samples, &rate);
    REQUIRE(rc == 0);
    REQUIRE(pcm != nullptr);
    REQUIRE(rate == 16000);
    REQUIRE(has_energy(pcm, samples));

    auto ref = load_ref();

    double ratio = (double)samples / ref.samples;
    INFO("ADTS AAC length ratio: " << ratio);
    REQUIRE(ratio > 0.90);
    REQUIRE(ratio < 1.10);

    // AAC-LC encoder priming (~1024-2112 samples) shifts the stream; raw ADTS
    // carries no edit list to strip it (unlike M4A's elst), so use a wide lag
    // window (3000 samples = 188 ms) to locate the true correlation peak.
    double cc = cross_correlation(ref.pcm, ref.samples, pcm, samples, 3000);
    INFO("ADTS AAC cross-correlation: " << cc);
    REQUIRE(cc > 0.85);

    crispasr_audio_free(pcm);
}

TEST_CASE("crispasr_audio_load decodes Ogg Opus (glint)", "[audio][unit][opus]") {
    // Real libopus-encoded .opus decoded by the in-tree glint decoder (default
    // for Ogg Opus; RFC-conformant). libopus-encode -> glint-decode is the
    // cross-encoder validation (HARD RULE #3); glint is always available, so
    // unlike WebM this must succeed everywhere.
    float* pcm = nullptr;
    int samples = 0, rate = 0;
    int rc = crispasr_audio_load(sample("jfk.opus").c_str(), &pcm, &samples, &rate);
    REQUIRE(rc == 0);
    REQUIRE(pcm != nullptr);
    REQUIRE(rate == 16000);
    REQUIRE(has_energy(pcm, samples));

    auto ref = load_ref();

    double ratio = (double)samples / ref.samples;
    INFO("Ogg Opus length ratio: " << ratio);
    REQUIRE(ratio > 0.90);
    REQUIRE(ratio < 1.10);

    // Opus is high quality and the decoder applies the pre-skip edit list, so
    // alignment is tight; a moderate lag window suffices.
    double cc = cross_correlation(ref.pcm, ref.samples, pcm, samples, 500);
    INFO("Ogg Opus cross-correlation: " << cc);
    REQUIRE(cc > 0.80);

    crispasr_audio_free(pcm);
}

TEST_CASE("crispasr_audio_load_stereo decodes Ogg Opus (glint, stereo path)", "[audio][unit][opus]") {
    // The stereo loader routes Ogg Opus through glint too (stereo-preserving).
    // jfk.opus is mono, so we expect 1 channel with L == R; the point is to
    // exercise the stereo loader's glint interception and its resample path.
    float* left = nullptr;
    float* right = nullptr;
    int samples = 0, rate = 0, channels = 0;
    int rc = crispasr_audio_load_stereo(sample("jfk.opus").c_str(), &left, &right, &samples, &rate, &channels);
    REQUIRE(rc == 0);
    REQUIRE(left != nullptr);
    REQUIRE(right != nullptr);
    REQUIRE(rate == 16000);
    REQUIRE(samples > 100000);
    REQUIRE(has_energy(left, samples));

    auto ref = load_ref();
    double cc = cross_correlation(ref.pcm, ref.samples, left, samples, 500);
    INFO("stereo Ogg Opus (L) cross-correlation: " << cc);
    REQUIRE(cc > 0.80);

    crispasr_audio_free(left);
    crispasr_audio_free(right);
}

TEST_CASE("crispasr_audio_load rejects missing file", "[audio][unit]") {
    float* pcm = nullptr;
    int samples = 0, rate = 0;
    int rc = crispasr_audio_load("/nonexistent/file.wav", &pcm, &samples, &rate);
    REQUIRE(rc < 0);
    REQUIRE(pcm == nullptr);
}

TEST_CASE("crispasr_audio_load rejects null args", "[audio][unit]") {
    float* pcm = nullptr;
    int samples = 0;
    REQUIRE(crispasr_audio_load(nullptr, &pcm, &samples, nullptr) == -1);
    REQUIRE(crispasr_audio_load("test.wav", nullptr, &samples, nullptr) == -1);
    REQUIRE(crispasr_audio_load("test.wav", &pcm, nullptr, nullptr) == -1);
}

// Regression: a structurally valid WAV declaring an absurd sample rate made
// crispasr_audio_load allocate without bound.
//
// CI's seeded fuzzer found it (run 33840954769): 344 KB of PCM at a declared
// `sampleRate = 1` resamples 16000x to the 16 kHz target — 2.816e9 frames,
// 11.3 GB — and the three chunked-decode loops doubled their buffer with no
// ceiling. The input reaches every surface that accepts a user file.
// tests/fuzz/regressions/wav-1hz-resample-oom.wav replays the exact input;
// this pins the behaviour hermetically, at a size that fits a unit test.
//
// THE POSITIVE CONTROL IS THE POINT. Byte-for-byte the same payload, with only
// the declared sample rate changed, must LOAD — otherwise "rejected" would be
// equally explained by a test that writes an undecodable WAV, and the arm that
// matters would pass for the wrong reason.
TEST_CASE("crispasr_audio_load bounds resample amplification", "[audio][unit]") {
    // 2000 mono int16 samples. At sr=1 that is 2000 s of audio, which the
    // 16 kHz target turns into 32e6 frames from a ~4 KB file (8000x the
    // ~1.03e6-frame ceiling a 4 KB input earns). At sr=16000 it is 0.125 s.
    const uint32_t kSamples = 2000;
    auto make_wav = [&](uint32_t sample_rate) {
        std::vector<uint8_t> w;
        auto le16 = [&](uint16_t v) {
            w.push_back((uint8_t)v);
            w.push_back((uint8_t)(v >> 8));
        };
        auto le32 = [&](uint32_t v) {
            for (int i = 0; i < 4; i++)
                w.push_back((uint8_t)(v >> (8 * i)));
        };
        const uint32_t data_bytes = kSamples * 2;
        w.insert(w.end(), {'R', 'I', 'F', 'F'});
        le32(36 + data_bytes);
        w.insert(w.end(), {'W', 'A', 'V', 'E', 'f', 'm', 't', ' '});
        le32(16);
        le16(1);               // PCM
        le16(1);               // mono
        le32(sample_rate);     // <- the only difference between the arms
        le32(sample_rate * 2); // byte rate
        le16(2);               // block align
        le16(16);              // bits
        w.insert(w.end(), {'d', 'a', 't', 'a'});
        le32(data_bytes);
        for (uint32_t i = 0; i < kSamples; i++) {
            // A real 220 Hz-ish tone, not silence: a decoder that bails early
            // would otherwise be indistinguishable from one that decodes.
            const int16_t v = (int16_t)(8000.0 * std::sin(2.0 * 3.14159265358979 * 220.0 * (double)i / 16000.0));
            le16((uint16_t)v);
        }
        return w;
    };
    auto write_file = [](const char* p, const std::vector<uint8_t>& bytes) {
        FILE* f = std::fopen(p, "wb");
        REQUIRE(f != nullptr);
        std::fwrite(bytes.data(), 1, bytes.size(), f);
        std::fclose(f);
    };

    const char* path = "crispasr_resample_amplification_tmp.wav";
    unsetenv("CRISPASR_MAX_DECODED_FRAMES");

    SECTION("positive control: the same payload at a sane rate decodes") {
        write_file(path, make_wav(16000));
        float* pcm = nullptr;
        int samples = 0, rate = 0;
        const int rc = crispasr_audio_load(path, &pcm, &samples, &rate);
        REQUIRE(rc == 0);
        REQUIRE(pcm != nullptr);
        // No resampling at the target rate, so the count is the stored length
        // give or take a decoder-internal frame; the point of the bound is that
        // it is ~2000 and not ~32e6.
        CHECK(samples > 1900);
        CHECK(samples < 2100);
        crispasr_audio_free(pcm);
    }

    SECTION("sampleRate = 1 is rejected instead of allocating 32e6 frames") {
        write_file(path, make_wav(1));
        float* pcm = nullptr;
        int samples = 0, rate = 0;
        const int rc = crispasr_audio_load(path, &pcm, &samples, &rate);
        CHECK(rc != 0);
        CHECK(pcm == nullptr);
    }

    SECTION("CRISPASR_MAX_DECODED_FRAMES is live") {
        // Same file the positive control accepts, under a ceiling below its
        // length: if the knob were dead this would still return 0, so the two
        // sections together pin the bound in both directions.
        write_file(path, make_wav(16000));
        setenv("CRISPASR_MAX_DECODED_FRAMES", "1000", 1);
        float* pcm = nullptr;
        int samples = 0, rate = 0;
        const int rc = crispasr_audio_load(path, &pcm, &samples, &rate);
        unsetenv("CRISPASR_MAX_DECODED_FRAMES");
        CHECK(rc != 0);
        CHECK(pcm == nullptr);
    }

    unsetenv("CRISPASR_MAX_DECODED_FRAMES");
    std::remove(path);
}
