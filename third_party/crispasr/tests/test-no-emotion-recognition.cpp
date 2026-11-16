// test-no-emotion-recognition.cpp — CrispASR must not ship voice-based emotion
// inference on any surface.
//
// Inferring a natural person's emotions from their voice makes the thing doing
// it an "emotion recognition system" (EU AI Act Art. 3(39)). That is PROHIBITED
// in workplace and education settings (Art. 5(1)(f), applicable since
// 2 Feb 2025) and HIGH-RISK everywhere else (Annex III(1)(c)) — and the AI
// Act's open-source exemption (Art. 2(12)) covers neither category, so shipping
// the capability at all would put the full Chapter III provider regime
// (risk management, conformity assessment, CE marking, EU-database
// registration, post-market monitoring) on this project. See docs/eu-ai-act.md.
//
// SenseVoice-Small is upstream an emotion classifier as well as an ASR model:
// its CTC head emits a `<|HAPPY|>` / `<|ANGRY|>` / `<|SAD|>` / ... marker in
// the annotation prefix of every utterance. We cannot stop the model emitting
// it, so the rule is that the value is parsed only in order to be discarded,
// and never reaches a struct field, a JSON key, or an ABI consumer.
//
// This is a SOURCE-level guard on purpose. The behavioural version would need
// the SenseVoice GGUF, which puts it in the live tier that CI does not run —
// and a compliance gate nobody can test on the tier CI actually runs is a gate
// that ships wrong (the lesson from #312's marking policy). Pure file
// inspection: no model, no linkage, runs in the unit tier.

#include <catch2/catch_test_macros.hpp>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef CRISPASR_SOURCE_DIR
#error "CRISPASR_SOURCE_DIR must be defined by the build"
#endif

namespace {

std::string read_file(const std::string& rel) {
    const std::string path = std::string(CRISPASR_SOURCE_DIR) + "/" + rel;
    std::ifstream f(path);
    REQUIRE(f.good());
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Strip //-comments so the explanatory prose above each removal (which has to
// name the thing it removed) doesn't count as a re-introduction. Block comments
// are not used for these notes in the files under test.
std::string strip_line_comments(const std::string& src) {
    std::istringstream in(src);
    std::string line, out;
    while (std::getline(in, line)) {
        const size_t c = line.find("//");
        out += (c == std::string::npos) ? line : line.substr(0, c);
        out += '\n';
    }
    return out;
}

bool contains(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

} // namespace

TEST_CASE("no emotion field on the SenseVoice C result struct", "[unit][compliance]") {
    // `struct sensevoice_result` is the public structured-output surface. A
    // `char* emotion` here is a per-utterance emotion inference handed to every
    // C/C++ consumer.
    const std::string code = strip_line_comments(read_file("src/sensevoice.h"));
    REQUIRE_FALSE(contains(code, "emotion"));
}

TEST_CASE("no emotion field on the CLI segment struct", "[unit][compliance]") {
    // crispasr_segment is what every ASR backend fills and every writer reads;
    // a field here would reach -oj/-ojf, SRT/VTT and the structured writers at
    // once.
    const std::string code = strip_line_comments(read_file("examples/cli/crispasr_backend.h"));
    REQUIRE_FALSE(contains(code, "emotion"));
}

TEST_CASE("no emotion key in any JSON writer", "[unit][compliance]") {
    for (const char* rel : {"examples/cli/cli.cpp", "examples/cli/crispasr_output.cpp"}) {
        const std::string code = strip_line_comments(read_file(rel));
        INFO("file: " << rel);
        REQUIRE_FALSE(contains(code, "\"emotion\""));
    }
}

TEST_CASE("the SenseVoice runtime parses the emotion marker only to drop it", "[unit][compliance]") {
    const std::string raw = read_file("src/sensevoice.cpp");
    const std::string code = strip_line_comments(raw);

    // The marker set must still be recognised — that is what keeps `<|HAPPY|>`
    // out of the transcript text. Removing the classifier would not remove the
    // capability, it would leak the raw token into every transcript instead.
    REQUIRE(contains(code, "\"HAPPY\""));

    // ...but the classified value must never be copied out to a caller.
    REQUIRE_FALSE(contains(code, "r->emotion"));
}

TEST_CASE("both SenseVoice entry points strip the annotation prefix", "[unit][compliance]") {
    // sensevoice_transcribe() is the one the session ABI uses, so it feeds
    // Python/Rust/Go/Dart/Java/C#/JS/WASM. It returned the prefix verbatim until
    // this audit, which put "<|en|><|HAPPY|><|Speech|><|withitn|>" at the front
    // of every binding's transcript. Both entry points must call the parser.
    const std::string code = strip_line_comments(read_file("src/sensevoice.cpp"));

    size_t n = 0;
    for (size_t p = code.find("sv_parse_prefix("); p != std::string::npos; p = code.find("sv_parse_prefix(", p + 1))
        ++n;
    // One definition + one call from each of the two entry points.
    REQUIRE(n >= 3);
}
