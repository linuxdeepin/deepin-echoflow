// #326 — embedding segments concurrently must not change the answer.
//
// Once pyannote segmentation was chunked, the speaker embedder became the
// dominant cost of diarization: one forward per segment, strictly sequential,
// and unhelped by `-t` (on TitaNet those per-segment graphs do not engage
// ggml's threads at all — user CPU stays within 3% of wall from -t 1 to -t 4).
// So the segments are now embedded across a few workers, each with its own
// model instance because a ggml backend is not safe for concurrent use.
//
// The risk in that change is not speed, it is order. The embeddings feed a
// clusterer, and the cluster a segment lands in depends on its position in the
// embedding array; if workers appended as they finished, results would vary run
// to run and speaker labels would shuffle. These cases pin the property that
// matters: identical output, whatever the worker count.
//
// Deterministic fake embedder — no model, no download, no ggml.
#include "crispasr_diarize_cli.h"
#include "crispasr_speaker_embedder.h"
#include "whisper_params.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include "portable_env.h"

namespace {

// Embedding is a pure function of the audio it is handed, so any reordering or
// mis-slicing shows up as a changed result rather than as noise. Sleeps a
// little so the workers genuinely interleave — without that, a fast fake could
// finish in submission order by luck and hide a real ordering bug.
class FakeEmbedder : public CrispasrSpeakerEmbedder {
public:
    explicit FakeEmbedder(int d = 8) : d_(d) {}

    int dim() const override { return d_; }
    const char* name() const override { return "fake"; }

    bool embed(const float* pcm, int n, float* out) override {
        if (!pcm || n <= 0 || !out)
            return false;
        // Reject one recognisable value so the "some segments fail" path is
        // exercised the same way a too-quiet segment would be.
        if (pcm[0] == kRejectMarker)
            return false;
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        double sum = 0.0;
        for (int i = 0; i < n; i++)
            sum += pcm[i];
        for (int k = 0; k < d_; k++)
            out[k] = (float)(sum / (double)n) + (float)k * 0.001f;
        return true;
    }

    std::unique_ptr<CrispasrSpeakerEmbedder> clone() const override { return std::make_unique<FakeEmbedder>(d_); }

    static constexpr float kRejectMarker = -12345.0f;

private:
    int d_;
};

struct Fixture {
    std::vector<crispasr_segment> segs;
    std::vector<float> audio;
};

// Segments of 1 s each, every one long enough to be embeddable, each carrying a
// distinct constant so a swap between two of them is detectable.
Fixture make_fixture(int n_segs, bool poison_one = false) {
    Fixture f;
    f.audio.resize((size_t)n_segs * 16000);
    for (int i = 0; i < n_segs; i++) {
        crispasr_segment s;
        s.t0 = (int64_t)i * 100; // centiseconds
        s.t1 = (int64_t)(i + 1) * 100;
        s.text = "seg" + std::to_string(i);
        f.segs.push_back(s);
        for (int j = 0; j < 16000; j++)
            f.audio[(size_t)i * 16000 + j] = 0.01f * (float)(i + 1);
    }
    if (poison_one && n_segs > 3)
        f.audio[(size_t)2 * 16000] = FakeEmbedder::kRejectMarker;
    return f;
}

std::vector<std::string> speakers_with_workers(int workers, int n_segs, bool poison = false) {
    if (workers > 0)
        setenv("CRISPASR_SPEAKER_EMBED_WORKERS", std::to_string(workers).c_str(), 1);
    else
        unsetenv("CRISPASR_SPEAKER_EMBED_WORKERS");

    Fixture f = make_fixture(n_segs, poison);
    FakeEmbedder emb;
    whisper_params params;
    crispasr_remap_speakers_via_embeddings(f.segs, f.audio.data(), (int)f.audio.size(), &emb, params);

    std::vector<std::string> out;
    out.reserve(f.segs.size());
    for (const auto& s : f.segs)
        out.push_back(s.speaker);
    unsetenv("CRISPASR_SPEAKER_EMBED_WORKERS");
    return out;
}

} // namespace

TEST_CASE("embed parallel: 4 workers give the same labels as 1", "[unit][diarize][issue-326]") {
    const auto serial = speakers_with_workers(1, 24);
    const auto parallel = speakers_with_workers(4, 24);
    REQUIRE(serial.size() == parallel.size());
    REQUIRE(serial == parallel);
}

TEST_CASE("embed parallel: more workers than segments is still correct", "[unit][diarize][issue-326]") {
    // The worker count is clamped against the job count; if that clamp were
    // wrong this would spin up idle threads or index past the job list.
    const auto serial = speakers_with_workers(1, 10);
    const auto many = speakers_with_workers(16, 10);
    REQUIRE(serial == many);
}

TEST_CASE("embed parallel: a failing segment is skipped identically", "[unit][diarize][issue-326]") {
    // A segment the embedder rejects must be skipped without shifting the
    // embeddings of the segments after it — that misalignment would attach
    // every later segment to the wrong cluster.
    const auto serial = speakers_with_workers(1, 24, /*poison=*/true);
    const auto parallel = speakers_with_workers(4, 24, /*poison=*/true);
    REQUIRE(serial == parallel);
}

TEST_CASE("embed parallel: repeated parallel runs are stable", "[unit][diarize][issue-326]") {
    // Thread scheduling differs between runs; the output must not.
    const auto a = speakers_with_workers(4, 24);
    const auto b = speakers_with_workers(4, 24);
    const auto c = speakers_with_workers(4, 24);
    REQUIRE(a == b);
    REQUIRE(b == c);
}

TEST_CASE("embed parallel: labels are actually assigned", "[unit][diarize][issue-326]") {
    // Guard against the whole comparison passing because both arms produced
    // nothing — two empty results compare equal.
    const auto serial = speakers_with_workers(1, 24);
    int labelled = 0;
    for (const auto& s : serial)
        if (!s.empty())
            labelled++;
    REQUIRE(labelled > 0);
}
