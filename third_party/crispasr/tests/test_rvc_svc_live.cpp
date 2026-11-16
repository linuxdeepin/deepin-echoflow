// RVC voice-conversion integration test.
//
// Requires CRISPASR_MODEL_RVC pointing at an rvc GGUF. SKIPs cleanly when unset.
//
// This drives the SESSION C ABI, not rvc_svc.h directly: RVC has no CLI verb
// (its input is ContentVec features, which CrispASR does not produce), so the
// session ABI is the ONLY surface a consumer has. It is therefore the one that
// must not rot.
//
// The determinism cases are the important ones. RVC inference is stochastic —
// two runs disagree by design — so the entire validation strategy for this port
// rests on being able to replay a noise draw. If that property breaks, every
// parity result becomes unreproducible and the per-stage harness stops meaning
// anything. These lock it in.

#include <catch2/catch_test_macros.hpp>

#include "crispasr_session.h"

#include <cmath>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kFrames = 32; // 100 Hz feature rate -> 0.32 s

crispasr_session* open_session(const char* model) {
    return crispasr_session_open_explicit(model, "rvc-svc", 2);
}

// Deterministic pseudo-inputs. Content is small-magnitude noise (ContentVec
// features are not wildly scaled) and F0 includes an UNVOICED stretch, which
// exercises the voicing-dependent branch in the sine source.
void make_inputs(int content_dim, std::vector<float>& content, std::vector<float>& f0) {
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 0.1f);
    content.resize((size_t)kFrames * content_dim);
    for (auto& v : content)
        v = nd(rng);
    f0.assign((size_t)kFrames, 0.0f);
    for (int i = 0; i < kFrames; i++)
        f0[(size_t)i] = (i >= kFrames / 3 && i < kFrames / 3 + 5) ? 0.0f : 120.0f + 40.0f * std::sin(i * 0.3f);
}

std::vector<float> draw(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v)
        x = nd(rng);
    return v;
}

} // namespace

TEST_CASE("rvc: session opens and reports checkpoint geometry", "[integration][rvc]") {
    const char* model = std::getenv("CRISPASR_MODEL_RVC");
    if (!model || !*model)
        SKIP("CRISPASR_MODEL_RVC not set");

    crispasr_session* s = open_session(model);
    REQUIRE(s != nullptr);

    // ContentVec dim is 256 (v1, layer 9 + final_proj) or 768 (v2, final layer)
    // and nothing else. This accessor exists so a consumer can refuse a v1/v2
    // mismatch, which is otherwise silent.
    const int cd = crispasr_session_convert_content_dim(s);
    REQUIRE((cd == 256 || cd == 768));
    REQUIRE(crispasr_session_convert_n_speakers(s) > 0);
    // The output rate is a property of the checkpoint, not a constant.
    const int sr = crispasr_session_convert_sample_rate(s);
    REQUIRE((sr == 32000 || sr == 40000 || sr == 48000));

    crispasr_session_close(s);
}

TEST_CASE("rvc: replayed noise makes conversion deterministic", "[integration][rvc]") {
    const char* model = std::getenv("CRISPASR_MODEL_RVC");
    if (!model || !*model)
        SKIP("CRISPASR_MODEL_RVC not set");

    crispasr_session* s = open_session(model);
    REQUIRE(s != nullptr);
    const int cd = crispasr_session_convert_content_dim(s);
    const int sr = crispasr_session_convert_sample_rate(s);
    const int upp = sr / 100; // the wire contract's 100 Hz feature rate

    std::vector<float> content, f0;
    make_inputs(cd, content, f0);
    // inter_channels is 192 for every shipped RVC config; the buffer only has
    // to match what convert() expects, and a wrong size would surface as a
    // failure here rather than as silent garbage.
    const std::vector<float> nz = draw((size_t)192 * kFrames, 7);
    const std::vector<float> ns = draw((size_t)kFrames * upp, 8);

    const int n1 = crispasr_session_convert(s, content.data(), kFrames, f0.data(), 0, nz.data(), ns.data());
    REQUIRE(n1 > 0);
    int c1 = 0;
    const float* p1 = crispasr_session_convert_audio(s, &c1);
    REQUIRE(p1 != nullptr);
    REQUIRE(c1 == n1);
    const std::vector<float> first(p1, p1 + c1);

    const int n2 = crispasr_session_convert(s, content.data(), kFrames, f0.data(), 0, nz.data(), ns.data());
    REQUIRE(n2 == n1);
    int c2 = 0;
    const float* p2 = crispasr_session_convert_audio(s, &c2);
    REQUIRE(c2 == c1);

    // BIT-IDENTICAL. Not "close": the whole point of injection over seeding is
    // that the same buffer must give the same samples, so a cross-
    // implementation harness can compare exactly.
    for (int i = 0; i < c1; i++)
        REQUIRE(first[(size_t)i] == p2[i]);

    // ...and the audio is real, not silence.
    double rms = 0.0;
    for (float v : first)
        rms += (double)v * v;
    rms = std::sqrt(rms / (double)first.size());
    INFO("output rms " << rms);
    REQUIRE(rms > 1e-4);
    REQUIRE(std::isfinite(rms));

    crispasr_session_close(s);
}

TEST_CASE("rvc: NULL noise draws fresh each call", "[integration][rvc]") {
    const char* model = std::getenv("CRISPASR_MODEL_RVC");
    if (!model || !*model)
        SKIP("CRISPASR_MODEL_RVC not set");

    crispasr_session* s = open_session(model);
    REQUIRE(s != nullptr);
    std::vector<float> content, f0;
    make_inputs(crispasr_session_convert_content_dim(s), content, f0);

    const int n1 = crispasr_session_convert(s, content.data(), kFrames, f0.data(), 0, nullptr, nullptr);
    REQUIRE(n1 > 0);
    int c1 = 0;
    const float* p1 = crispasr_session_convert_audio(s, &c1);
    const std::vector<float> first(p1, p1 + c1);

    REQUIRE(crispasr_session_convert(s, content.data(), kFrames, f0.data(), 0, nullptr, nullptr) == n1);
    int c2 = 0;
    const float* p2 = crispasr_session_convert_audio(s, &c2);

    // Two draws must DIFFER: the model is stochastic by design and production
    // should not be silently deterministic. If this ever passes trivially, the
    // noise path has been short-circuited.
    bool differs = false;
    for (int i = 0; i < c1 && !differs; i++)
        differs = first[(size_t)i] != p2[i];
    REQUIRE(differs);

    crispasr_session_close(s);
}

TEST_CASE("rvc: rejects bad arguments", "[integration][rvc]") {
    const char* model = std::getenv("CRISPASR_MODEL_RVC");
    if (!model || !*model)
        SKIP("CRISPASR_MODEL_RVC not set");

    crispasr_session* s = open_session(model);
    REQUIRE(s != nullptr);
    std::vector<float> content, f0;
    make_inputs(crispasr_session_convert_content_dim(s), content, f0);

    REQUIRE(crispasr_session_convert(s, nullptr, kFrames, f0.data(), 0, nullptr, nullptr) == -1);
    REQUIRE(crispasr_session_convert(s, content.data(), 0, f0.data(), 0, nullptr, nullptr) == -1);
    REQUIRE(crispasr_session_convert(s, content.data(), kFrames, nullptr, 0, nullptr, nullptr) == -1);
    REQUIRE(crispasr_session_convert(nullptr, content.data(), kFrames, f0.data(), 0, nullptr, nullptr) == -1);
    // Out-of-range speaker must be refused, not clamped: a silently wrong
    // speaker sounds like a bad conversion rather than an error.
    const int n_spk = crispasr_session_convert_n_speakers(s);
    REQUIRE(crispasr_session_convert(s, content.data(), kFrames, f0.data(), n_spk, nullptr, nullptr) == -1);
    REQUIRE(crispasr_session_convert(s, content.data(), kFrames, f0.data(), -1, nullptr, nullptr) == -1);

    REQUIRE(crispasr_session_convert_audio(nullptr, nullptr) == nullptr);
    REQUIRE(crispasr_session_convert_content_dim(nullptr) == 0);

    crispasr_session_close(s);
}
