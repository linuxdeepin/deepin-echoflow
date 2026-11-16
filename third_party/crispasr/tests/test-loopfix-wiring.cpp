// test-loopfix-wiring.cpp — do the CLI backend adapters actually CALL the
// n-gram loop collapse? (PLAN.md §W1b)
//
// This exists because firered-asr did not, for its entire life. The header had
// unit tests, the algorithm was correct, and `core_ngram::fix_loops` was simply
// never invoked on firered's output — the only mention in `firered_asr.cpp` is
// a comment at :2270 stating the absence out loud, used to justify the
// decode-time `core_repeat::tail_is_repetition` break as a stand-in. That break
// is wired into the `beam_size == 1` branch only, so a beam run had no guard at
// all. firered is Mandarin + 20+ Chinese dialects — the exact script §W1 had
// just taught the collapse to handle.
//
// A pure-predicate test cannot catch that: the predicate was always fine. This
// guards the JOIN. It is a source scan, so it costs nothing and needs no model.
//
// If you are adding an AR backend adapter that can loop, add it below. If one
// of these goes red, the call was deleted — restore it rather than shrinking
// the list.

#include <catch2/catch_test_macros.hpp>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef CRISPASR_SOURCE_DIR
#error "CRISPASR_SOURCE_DIR must be defined by the build"
#endif

namespace {

std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.good());
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Adapters observed to route their transcript through the collapse. This list
// pins existing wiring; it is not a claim that every other adapter needs it
// (CTC/RNN-T backends do not loop this way, and higgs-stt / moss-transcribe
// call fix_loops inside the backend rather than the adapter).
const char* kAdaptersWithLoopfix[] = {
    "crispasr_backend_canary_qwen.cpp", "crispasr_backend_cohere.cpp",  "crispasr_backend_firered_asr.cpp",
    "crispasr_backend_glm_asr.cpp",     "crispasr_backend_granite.cpp", "crispasr_backend_qwen3.cpp",
};

} // namespace

TEST_CASE("every AR CLI adapter on the list calls core_ngram::fix_loops", "[loopfix-wiring]") {
    const std::string dir = std::string(CRISPASR_SOURCE_DIR) + "/examples/cli/";
    for (const char* name : kAdaptersWithLoopfix) {
        const std::string src = slurp(dir + name);
        INFO("adapter: " << name);
        // Assert the CALL, not the #include — an include that nothing invokes
        // is exactly the state firered-asr shipped in.
        const bool calls = src.find("core_ngram::fix_loops(") != std::string::npos ||
                           src.find("core_ngram::fix_loops_keep_indices(") != std::string::npos;
        REQUIRE(calls);
    }
}

TEST_CASE("the shared collapse header is not silently unreferenced", "[loopfix-wiring]") {
    // Sanity: the backends that own the collapse in-library still include it.
    // Catches a refactor that moves the call out without moving the guard.
    const std::string root = CRISPASR_SOURCE_DIR;
    for (const char* rel : {"/src/moss_transcribe.cpp", "/src/higgs_stt.cpp", "/src/crispasr_c_api.cpp"}) {
        const std::string src = slurp(root + rel);
        INFO("source: " << rel);
        REQUIRE(src.find("core_ngram::fix_loops") != std::string::npos);
    }
}

TEST_CASE("firered-asr's decode-time break is not mistaken for the collapse", "[loopfix-wiring]") {
    // §W1b's real lesson: `core_repeat::tail_is_repetition` in firered_asr.cpp
    // is a COMPUTE saver on the greedy branch, not a substitute for the text
    // collapse. Both must be present. If the repeat-break is ever wired into
    // the beam branch too, this test is still correct — it only asserts that
    // neither guard has vanished.
    const std::string src = slurp(std::string(CRISPASR_SOURCE_DIR) + "/src/firered_asr.cpp");
    REQUIRE(src.find("core_repeat::tail_is_repetition") != std::string::npos);
}
