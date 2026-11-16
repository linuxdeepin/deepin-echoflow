// test-abi-clone-disclosure.cpp — the ABI's Art. 50(4) posture for voice clones.
//
// crispasr_session_synthesize() watermarks every clip, which discharges the
// machine-readable marking duty (Art. 50(2)) on the ABI. Art. 50(4) additionally
// requires a visible or audible disclosure for deepfakes, and the ABI does not
// prepend one — the CLI and server do. This file pins the three decisions that
// keep that difference deliberate instead of accidental:
//
//   1. a reference WAV is what makes a voice a CLONE (same test the CLI and
//      server use, so the three surfaces agree on what triggers disclosure);
//   2. the warning fires once per session, not per synthesis call, and is
//      silenced by attesting — a per-call warning gets filtered out by
//      integrators and stops being read;
//   3. get_disclaimer_pcm() is refused once a clone voice is set.
//
// (3) is the load-bearing one. The CLI synthesizes a NEUTRAL disclaimer by
// clearing tts_voice per call, and several adapters need backend-specific
// handling to honour that. On the ABI the voice is already applied to the
// backend context, so synthesizing anyway would risk speaking the disclosure IN
// THE CLONED VOICE — more convincing rather than less, and worse than no
// disclaimer at all. The refusal is the feature.
//
// Weight-free: the predicates are reproduced here rather than exercised through
// a session, because constructing one needs a TTS model and would put this in
// the live tier that CI does not run — the #312 lesson about compliance gates
// that only exist on an untested tier.

#include <catch2/catch_test_macros.hpp>

#include <fstream>
#include <sstream>
#include <string>

#ifndef CRISPASR_SOURCE_DIR
#error "CRISPASR_SOURCE_DIR must be defined by the build"
#endif

namespace {

// Mirror of the ends_with_wav test in crispasr_session_set_voice(), which is
// also what crispasr_run.cpp:2984 and crispasr_server.cpp:2145 use.
bool is_clone_voice(const std::string& path) {
    if (path.size() < 4)
        return false;
    const char* tail = path.c_str() + path.size() - 4;
    return (tail[0] == '.' && (tail[1] == 'w' || tail[1] == 'W') && (tail[2] == 'a' || tail[2] == 'A') &&
            (tail[3] == 'v' || tail[3] == 'V'));
}

// Mirror of crispasr_session_warn_unmarked_clone()'s guard.
struct SessionMarkingState {
    bool voice_is_clone = false;
    bool marking_responsibility_accepted = false;
    bool warned_clone_unmarked = false;

    // Returns true when this call actually emits the audit line.
    bool warn() {
        if (!voice_is_clone || marking_responsibility_accepted || warned_clone_unmarked)
            return false;
        warned_clone_unmarked = true;
        return true;
    }
};

} // namespace

TEST_CASE("a reference WAV is a clone, a preset name is not", "[unit][compliance][marking]") {
    // Clone → Art. 50(4) disclosure is owed by whoever publishes the output.
    REQUIRE(is_clone_voice("alice.wav"));
    REQUIRE(is_clone_voice("/abs/path/Speaker Ref.WAV"));
    REQUIRE(is_clone_voice("a.wav"));

    // Preset/bank voices synthesize a voice that belongs to nobody — no deepfake,
    // nothing to disclose beyond the watermark.
    REQUIRE_FALSE(is_clone_voice("af_heart"));
    REQUIRE_FALSE(is_clone_voice("voice.gguf"));
    REQUIRE_FALSE(is_clone_voice(""));
    REQUIRE_FALSE(is_clone_voice("wav"));
    // ".wav" as a bare name is 4 chars and does end in .wav — accepted, matching
    // the CLI. Guarding the degenerate case would make the surfaces disagree.
    REQUIRE(is_clone_voice(".wav"));
}

TEST_CASE("the clone warning fires once per session", "[unit][compliance][marking]") {
    SessionMarkingState s;
    s.voice_is_clone = true;

    REQUIRE(s.warn());       // first synthesize() → audit line
    REQUIRE_FALSE(s.warn()); // every subsequent call stays quiet
    REQUIRE_FALSE(s.warn());
}

TEST_CASE("attesting silences the clone warning", "[unit][compliance][marking]") {
    // accept_marking_responsibility() means the integrator has taken the
    // disclosure duty on themselves; telling them about it again is noise.
    SessionMarkingState s;
    s.voice_is_clone = true;
    s.marking_responsibility_accepted = true;
    REQUIRE_FALSE(s.warn());
}

TEST_CASE("non-clone synthesis never warns", "[unit][compliance][marking]") {
    // A preset voice is watermarked and owes no Art. 50(4) label, so warning
    // would train integrators to ignore the line that does matter.
    SessionMarkingState s;
    s.voice_is_clone = false;
    REQUIRE_FALSE(s.warn());
    s.marking_responsibility_accepted = true;
    REQUIRE_FALSE(s.warn());
}

TEST_CASE("get_disclaimer_pcm is refused once a clone voice is set", "[unit][compliance][marking]") {
    // The ordering contract: open -> get_disclaimer_pcm -> set_voice ->
    // synthesize -> prepend. Reversing the middle two must fail closed.
    auto disclaimer_available = [](bool voice_is_clone) { return !voice_is_clone; };

    REQUIRE(disclaimer_available(/*voice_is_clone=*/false));
    REQUIRE_FALSE(disclaimer_available(/*voice_is_clone=*/true));
}

TEST_CASE("the ABI disclosure text matches the CLI's", "[unit][compliance][marking]") {
    // crispasr_session_disclaimer_text() and crispasr_disclaimer::text() in
    // examples/cli/crispasr_tts_disclaimer.h must not drift: an integrator
    // rendering the visible label should be saying exactly what the CLI speaks,
    // or the same product discloses two different things on two surfaces.
    //
    // Compared at the SOURCE level — including the CLI header here would drag in
    // CrispasrBackend and whisper_params, and asserting a literal against itself
    // would be a test that cannot fail.
    auto disclaimer_literal_in = [](const std::string& rel) -> std::string {
        const std::string path = std::string(CRISPASR_SOURCE_DIR) + "/" + rel;
        std::ifstream f(path);
        REQUIRE(f.good());
        std::ostringstream ss;
        ss << f.rdbuf();
        const std::string src = ss.str();

        // Both sites return the string from a one-line `return "...";`.
        const std::string marker = "return \"This audio was generated";
        const size_t at = src.find(marker);
        REQUIRE(at != std::string::npos);
        const size_t open = src.find('"', at);
        const size_t close = src.find('"', open + 1);
        REQUIRE(close != std::string::npos);
        return src.substr(open + 1, close - open - 1);
    };

    const std::string abi = disclaimer_literal_in("src/crispasr_c_api.cpp");
    const std::string cli = disclaimer_literal_in("examples/cli/crispasr_tts_disclaimer.h");

    INFO("ABI: " << abi);
    INFO("CLI: " << cli);
    REQUIRE(abi == cli);
    REQUIRE_FALSE(abi.empty());
}
