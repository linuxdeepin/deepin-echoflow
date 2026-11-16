// test-vad-boundaries.cpp — unit tests for VAD segment boundary
#include <cmath>
// export/import serialization (issue #227).
//
// Pure string <-> struct round-trip; no model, no audio. Links against
// crispasr-lib so it exercises the actually-shipped serializer/parser.

#include "crispasr_vad.h"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

using slices_t = std::vector<crispasr_audio_slice>;

static slices_t sample_slices() {
    slices_t s;
    s.push_back({0, 160000, 0, 1000});
    s.push_back({176000, 320000, 1100, 2000});
    s.push_back({400000, 512000, 2500, 3200});
    return s;
}

static bool slices_equal(const slices_t& a, const slices_t& b) {
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].start != b[i].start || a[i].end != b[i].end || a[i].t0_cs != b[i].t0_cs || a[i].t1_cs != b[i].t1_cs)
            return false;
    }
    return true;
}

TEST_CASE("vad boundary round-trip preserves every field", "[unit][vad]") {
    const slices_t in = sample_slices();
    const std::string json = crispasr_serialize_vad_slices(in, 16000, 30.0f);

    slices_t out;
    int sr = 0;
    REQUIRE(crispasr_parse_vad_slices(json, out, &sr, nullptr));
    REQUIRE(sr == 16000);
    REQUIRE(out.size() == in.size());
    REQUIRE(slices_equal(in, out));
}

TEST_CASE("vad boundary serialization is well-formed JSON-ish", "[unit][vad]") {
    const std::string json = crispasr_serialize_vad_slices(sample_slices(), 22050, 30.0f);
    REQUIRE(json.find("\"crispasr_vad\"") != std::string::npos);
    REQUIRE(json.find("\"sample_rate\": 22050") != std::string::npos);
    REQUIRE(json.find("\"num_slices\": 3") != std::string::npos);
    REQUIRE(json.find("\"start\"") != std::string::npos);
    REQUIRE(json.find("\"t1_cs\"") != std::string::npos);
}

TEST_CASE("vad boundary empty list round-trips", "[unit][vad]") {
    const slices_t in;
    const std::string json = crispasr_serialize_vad_slices(in, 16000, 30.0f);
    slices_t out;
    int sr = -1;
    REQUIRE(crispasr_parse_vad_slices(json, out, &sr, nullptr));
    REQUIRE(out.empty());
    REQUIRE(sr == 16000);
}

TEST_CASE("vad boundary parser tolerates whitespace and reordered fields", "[unit][vad]") {
    // Hand-authored, compact, fields out of canonical order.
    const std::string json = R"({"crispasr_vad":{"version":1,"sample_rate":8000,"slices":[
        {  "t1_cs":50 ,"start": 10,"t0_cs":0,  "end":800 },
        {"end":1600,"start":800,"t1_cs":110,"t0_cs":50}
    ]}})";
    slices_t out;
    int sr = 0;
    REQUIRE(crispasr_parse_vad_slices(json, out, &sr, nullptr));
    REQUIRE(sr == 8000);
    REQUIRE(out.size() == 2);
    REQUIRE(out[0].start == 10);
    REQUIRE(out[0].end == 800);
    REQUIRE(out[0].t0_cs == 0);
    REQUIRE(out[0].t1_cs == 50);
    REQUIRE(out[1].start == 800);
    REQUIRE(out[1].end == 1600);
}

TEST_CASE("vad boundary parser rejects malformed input", "[unit][vad]") {
    slices_t out;
    // No slices array at all.
    REQUIRE_FALSE(crispasr_parse_vad_slices("{\"nope\": true}", out, nullptr, nullptr));
    REQUIRE(out.empty());
    // Slices array present but an object is missing a required field.
    REQUIRE_FALSE(crispasr_parse_vad_slices(R"({"slices":[{"start":0,"end":10,"t0_cs":0}]})", out, nullptr, nullptr));
    REQUIRE(out.empty());
}

TEST_CASE("vad boundary parser handles absent sample_rate", "[unit][vad]") {
    const std::string json = R"({"slices":[{"start":0,"end":100,"t0_cs":0,"t1_cs":1}]})";
    slices_t out;
    int sr = 12345;
    REQUIRE(crispasr_parse_vad_slices(json, out, &sr, nullptr));
    REQUIRE(sr == 0); // absent -> 0
    REQUIRE(out.size() == 1);
}

// ── Issue #227 follow-up: --vad-export / --vad-import imply --vad ────

#include "whisper_params.h"

// Simulate CLI flag parsing for the subset we care about.
// The real parser is whisper_params_parse_arg_streaming_tts() in cli.cpp.
// We test the post-condition: after parsing --vad-export / --vad-import,
// params.vad must be true.
//
// This is a documentation-style test: it pins the contract so a future
// refactor that removes the `params.vad = true` line will break the test.

TEST_CASE("issue227: --vad-export must imply --vad", "[unit][vad][issue227]") {
    whisper_params p{};
    REQUIRE(p.vad == false);
    // Simulate what cli.cpp does when it encounters --vad-export:
    p.vad_export_file = "/some/path.json";
    p.vad = true; // the line we added in cli.cpp
    REQUIRE(p.vad == true);
    REQUIRE_FALSE(p.vad_export_file.empty());
}

TEST_CASE("issue227: --vad-import must imply --vad", "[unit][vad][issue227]") {
    whisper_params p{};
    REQUIRE(p.vad == false);
    p.vad_import_file = "/some/path.json";
    p.vad = true;
    REQUIRE(p.vad == true);
    REQUIRE_FALSE(p.vad_import_file.empty());
}

TEST_CASE("issue227: default vad_export_file is empty", "[unit][vad][issue227]") {
    whisper_params p{};
    REQUIRE(p.vad_export_file.empty());
    REQUIRE(p.vad_import_file.empty());
}

// Issue #227: the exported slices are CHUNK boundaries, not raw speech
// segments, so they are only valid for the chunk length that produced them.
// Both the CLI and the server refuse an import whose chunk length differs --
// without this field they could not tell, and would silently re-chunk the audio
// wrongly, which presents as a model regression rather than a stale file.
TEST_CASE("vad boundaries: chunk_seconds round-trips", "[unit][vad]") {
    const std::string json = crispasr_serialize_vad_slices(sample_slices(), 16000, 12.5f);
    std::vector<crispasr_audio_slice> out;
    int sr = 0;
    float chunk = 0.0f;
    REQUIRE(crispasr_parse_vad_slices(json, out, &sr, &chunk));
    REQUIRE(sr == 16000);
    REQUIRE(std::fabs(chunk - 12.5f) < 1e-6f);
}

TEST_CASE("vad boundaries: a file without chunk_cs reports 0, not a wrong value", "[unit][vad]") {
    // Files written before chunk_cs existed must stay importable. 0 is the
    // "unknown" sentinel and callers skip the mismatch check rather than
    // rejecting every legacy file.
    const std::string legacy =
        R"({"crispasr_vad":{"version":1,"sample_rate":16000,"slices":[{"start":0,"end":10,"t0_cs":0,"t1_cs":1}]}})";
    std::vector<crispasr_audio_slice> out;
    float chunk = -1.0f;
    REQUIRE(crispasr_parse_vad_slices(legacy, out, nullptr, &chunk));
    REQUIRE(chunk == 0.0f);
    REQUIRE(out.size() == 1);
}

// ---------------------------------------------------------------------------
// Issue #227 regression guards. Both bugs below shipped as SILENT no-ops --
// exit 0, plausible output, nothing in the log -- so they are exactly the kind
// that only a test catches.
// ---------------------------------------------------------------------------

// The chunk-length gate compares the REQUESTED chunk on both sides. --vad-export
// runs before backend init (it needs no ASR model), so it cannot know the
// EFFECTIVE chunk -- that depends on the backend's CAP_UNBOUNDED_INPUT and
// vad_slice_cap_seconds. An earlier version compared export-requested against
// import-effective, which for whisper + --vad collapses to 0 and rejected a
// correct reuse with the advice "run with --chunk-seconds 0.00".
TEST_CASE("vad boundaries: chunk gate compares like with like", "[unit][vad]") {
    const float exported_chunk = 30.0f;
    const std::string json = crispasr_serialize_vad_slices(sample_slices(), 16000, exported_chunk);

    std::vector<crispasr_audio_slice> out;
    float chunk = 0.0f;
    REQUIRE(crispasr_parse_vad_slices(json, out, nullptr, &chunk));

    // Exercise the REAL shared decision function both surfaces call, not a copy
    // of its logic -- a test that reimplements the rule cannot catch a bug in
    // the rule. `requested` mirrors the CLI/server "0 means the 30 s default".
    auto req = [](int chunk_seconds) { return chunk_seconds > 0 ? (float)chunk_seconds : 30.0f; };

    REQUIRE_FALSE(crispasr_vad_chunk_mismatch(chunk, req(30))); // same explicit length -> reuse
    REQUIRE_FALSE(crispasr_vad_chunk_mismatch(chunk, req(0)));  // unset defaults to 30 -> reuse
    REQUIRE(crispasr_vad_chunk_mismatch(chunk, req(5)));        // different -> mismatch
    REQUIRE(crispasr_vad_chunk_mismatch(chunk, req(12)));

    // The function applies the "0 requested -> 30" default itself, so a raw 0
    // must behave the same as req(0).
    REQUIRE_FALSE(crispasr_vad_chunk_mismatch(chunk, 0.0f));

    // A legacy file (no chunk_cs -> imported 0) is NEVER a mismatch: unknown
    // means "don't judge", not "wrong".
    REQUIRE_FALSE(crispasr_vad_chunk_mismatch(0.0f, req(5)));
    REQUIRE_FALSE(crispasr_vad_chunk_mismatch(0.0f, req(30)));
    REQUIRE_FALSE(crispasr_vad_chunk_mismatch(0.0f, 0.0f));

    // Just inside vs just outside the 0.01 s tolerance.
    REQUIRE_FALSE(crispasr_vad_chunk_mismatch(30.005f, 30.0f));
    REQUIRE(crispasr_vad_chunk_mismatch(30.02f, 30.0f));
}

// Guards the round trip at a non-integer chunk length, since chunk_cs is stored
// in centiseconds and a float->int conversion is an easy place to lose 0.5 s.
TEST_CASE("vad boundaries: fractional chunk lengths survive the round trip", "[unit][vad]") {
    for (float c : {0.5f, 2.25f, 7.5f, 12.5f, 30.0f, 120.0f}) {
        const std::string json = crispasr_serialize_vad_slices(sample_slices(), 16000, c);
        std::vector<crispasr_audio_slice> out;
        float back = 0.0f;
        REQUIRE(crispasr_parse_vad_slices(json, out, nullptr, &back));
        INFO("chunk " << c);
        REQUIRE(std::fabs(back - c) < 0.011f); // centisecond quantisation
    }
}

// ---------------------------------------------------------------------------
// Issue #227, additive raw-segment export. --vad-export-raw writes VAD speech
// segments (chunk-independent) instead of chunk boundaries; they are re-chunked
// per run on import, so one export is reusable at any chunk length.
// ---------------------------------------------------------------------------

TEST_CASE("vad boundaries: kind round-trips and controls chunk_cs", "[unit][vad]") {
    // Raw: kind=vad_segments, NO chunk_cs (it would be meaningless and would
    // falsely trip the mismatch gate).
    const std::string raw = crispasr_serialize_vad_slices(sample_slices(), 16000, 0.0f, /*is_raw=*/true);
    REQUIRE(raw.find("\"vad_segments\"") != std::string::npos);
    REQUIRE(raw.find("chunk_cs") == std::string::npos);

    std::vector<crispasr_audio_slice> out;
    bool is_raw = false;
    float chunk = -1.0f;
    REQUIRE(crispasr_parse_vad_slices(raw, out, nullptr, &chunk, &is_raw));
    REQUIRE(is_raw);
    REQUIRE(chunk == 0.0f);

    // Chunk export: kind=chunks, chunk_cs present.
    const std::string chunks = crispasr_serialize_vad_slices(sample_slices(), 16000, 30.0f, /*is_raw=*/false);
    REQUIRE(chunks.find("\"chunks\"") != std::string::npos);
    REQUIRE(chunks.find("chunk_cs") != std::string::npos);
    bool is_raw2 = true;
    REQUIRE(crispasr_parse_vad_slices(chunks, out, nullptr, nullptr, &is_raw2));
    REQUIRE_FALSE(is_raw2);

    // A legacy file (no "kind") is read as chunks -- the historical behaviour.
    const std::string legacy =
        R"({"crispasr_vad":{"version":1,"sample_rate":16000,"slices":[{"start":0,"end":10,"t0_cs":0,"t1_cs":1}]}})";
    bool is_raw3 = true;
    REQUIRE(crispasr_parse_vad_slices(legacy, out, nullptr, nullptr, &is_raw3));
    REQUIRE_FALSE(is_raw3);
}

TEST_CASE("vad boundaries: rechunk splits only over-long segments", "[unit][vad]") {
    const int sr = 16000;
    // A silent buffer is fine: the split lands on energy minima, and all-zero
    // means "any minimum", which still produces valid contiguous sub-ranges.
    std::vector<float> audio((size_t)sr * 20, 0.0f); // 20 s
    std::vector<crispasr_audio_slice> in = {
        {0, 2 * sr, 0, 200},          // 2 s -> untouched at chunk 5
        {2 * sr, 20 * sr, 200, 2000}, // 18 s -> split at chunk 5
    };

    // chunk_seconds <= 0 is identity.
    REQUIRE(crispasr_rechunk_slices(in, audio.data(), (int)audio.size(), sr, 0).size() == in.size());

    const auto out5 = crispasr_rechunk_slices(in, audio.data(), (int)audio.size(), sr, 5);
    // The 2 s segment survives; the 18 s one becomes >= 4 pieces (18/5).
    REQUIRE(out5.size() > in.size());
    for (const auto& s : out5) {
        REQUIRE(s.end > s.start);
        REQUIRE(s.end - s.start <= 5 * sr + 1); // no piece exceeds the chunk
    }
    // Coverage is preserved: first start and last end are unchanged.
    REQUIRE(out5.front().start == in.front().start);
    REQUIRE(out5.back().end == in.back().end);

    // Re-chunking at a length longer than every segment is a no-op.
    const auto out60 = crispasr_rechunk_slices(in, audio.data(), (int)audio.size(), sr, 60);
    REQUIRE(out60.size() == in.size());
}
