// #334 — the reference-transcript cache is a contract between two layers.
//
// CosyVoice3's talker is conditioned on the (transcript, speech) pair and infers
// the speaker's rate from it, so a transcript that does not match the clip is
// worse than none: the decode stops at once or crams the requested line into far
// too few frames — the "chipmunk effect" in that report. The CLI therefore
// transcribes the reference itself when --ref-text is omitted, and caches the
// result beside the clip.
//
// The session C ABI cannot transcribe — the CLI helper constructs a second
// CrispasrBackend, a layer the ABI has no access to — but it CAN read that
// cache, so a clip prepared once through the CLI now clones through the session
// API too instead of failing with -2.
//
// That makes the cache a cross-layer contract: the CLI writes it
// (crispasr_tts_ref_text.h) and crispasr_c_api.cpp reads it, with the suffix and
// the tag agreed only by convention. Change either on one side and the session
// silently stops finding transcripts that are sitting right there — no error, no
// failing build, just the -2 coming back. These cases pin it.
//
// Header-only cache, no model and no backend needed.
#include "core/tts_ref_cache.h"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

namespace {

// THE shared constant both real sites now use — not a copy of it. An earlier
// version of this file declared its own literal, which made the round-trip case
// self-consistent and therefore blind: changing the suffix changed writer and
// reader together and every assertion still passed. Red-verifying caught that.
constexpr const char* kSuffix = crispasr_ref_cache::kCv3RefTextSuffix;

std::string temp_voice_path(const char* stem) {
    std::string p =
        std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") + "/crispasr-t334-" + stem + ".wav";
    FILE* f = std::fopen(p.c_str(), "wb");
    if (f) {
        std::fputs("not really a wav", f);
        std::fclose(f);
    }
    return p;
}

void write_cache(const std::string& voice, const std::string& text, const char* tag) {
    const std::string cp = crispasr_ref_cache::path_for(voice, kSuffix);
    crispasr_ref_cache::save(cp, tag, {(uint32_t)text.size()}, text.data(), text.size());
}

// Exactly what crispasr_c_api.cpp does on the read side.
bool read_cache(const std::string& voice, std::string& out) {
    const std::string cp = crispasr_ref_cache::path_for(voice, kSuffix);
    std::vector<uint32_t> shape;
    std::vector<uint8_t> payload;
    if (!crispasr_ref_cache::load(cp, voice, kSuffix, shape, payload) || payload.empty())
        return false;
    out.assign((const char*)payload.data(), payload.size());
    return true;
}

} // namespace

TEST_CASE("ref cache: what the CLI writes, the session reads", "[unit][tts][issue-334]") {
    const std::string voice = temp_voice_path("roundtrip");
    const std::string text = "The quick brown fox jumps over the lazy dog.";
    write_cache(voice, text, kSuffix);

    std::string got;
    REQUIRE(read_cache(voice, got));
    REQUIRE(got == text);

    std::remove(crispasr_ref_cache::path_for(voice, kSuffix).c_str());
    std::remove(voice.c_str());
}

TEST_CASE("ref cache: a mismatched tag is rejected, not misread", "[unit][tts][issue-334]") {
    // Two backends must never read each other's blob. A transcript from the
    // wrong model is exactly the failure mode #334 is about, so silently
    // accepting it would be worse than finding nothing.
    const std::string voice = temp_voice_path("tag");
    write_cache(voice, "some other backend's text", ".f5reftext");

    std::string got;
    REQUIRE_FALSE(read_cache(voice, got));

    std::remove(crispasr_ref_cache::path_for(voice, kSuffix).c_str());
    std::remove(voice.c_str());
}

TEST_CASE("ref cache: absent cache reports absent", "[unit][tts][issue-334]") {
    // The path that must still return -2 with the explanatory message.
    const std::string voice = temp_voice_path("missing");
    std::remove(crispasr_ref_cache::path_for(voice, kSuffix).c_str());

    std::string got;
    REQUIRE_FALSE(read_cache(voice, got));
    std::remove(voice.c_str());
}

TEST_CASE("ref cache: an edited clip invalidates its transcript", "[unit][tts][issue-334]") {
    // The whole hazard is a transcript that does not describe the audio. If the
    // clip is replaced, its cached transcript describes the old one and must not
    // be reused — mtime is what enforces that.
    const std::string voice = temp_voice_path("mtime");
    write_cache(voice, "transcript of the ORIGINAL clip", kSuffix);

    std::string got;
    REQUIRE(read_cache(voice, got));

    // Rewrite the clip with a newer mtime than the cache.
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    FILE* f = std::fopen(voice.c_str(), "wb");
    REQUIRE(f != nullptr);
    std::fputs("a completely different recording", f);
    std::fclose(f);

    REQUIRE_FALSE(read_cache(voice, got));

    std::remove(crispasr_ref_cache::path_for(voice, kSuffix).c_str());
    std::remove(voice.c_str());
}

TEST_CASE("ref cache: the cache path sits beside the clip", "[unit][tts][issue-334]") {
    // The error message in crispasr_c_api.cpp tells the user this exact path, so
    // it has to be derived the same way on both sides.
    const std::string voice = "/some/dir/reference.wav";
    REQUIRE(crispasr_ref_cache::path_for(voice, kSuffix) == voice + kSuffix);
}

TEST_CASE("ref cache: both layers share one suffix constant", "[unit][tts][issue-334]") {
    // The value itself is pinned because it names on-disk files users may
    // already have; changing it silently orphans every cached transcript.
    REQUIRE(std::string(crispasr_ref_cache::kCv3RefTextSuffix) == ".cv3reftext");
}
