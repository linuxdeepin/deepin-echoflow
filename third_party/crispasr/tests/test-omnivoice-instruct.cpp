// test-omnivoice-instruct.cpp — OmniVoice voice-design instruct (#13273).
//
// Found while verifying the language fix against the blueprint: `_resolve_instruct`
// sits ten lines below `_resolve_language` and we mirrored neither. The instruct
// goes into the prompt literally, out of a CLOSED 48-item vocabulary, so an
// unnormalised value is conditioning the model on tokens it never saw there:
//
//     'Male, British Accent'  -> [151672, 36421, 11,  7855, 81809, 151673]
//     'male, british accent'  -> [151672, 36476, 11, 93927, 29100, 151673]
//
// Same two halves as the language guard: the PREDICATE (parity with the
// blueprint) and the JOINS (every surface actually calls it, and rejects).

#include "core/omnivoice_instruct.h"

#include <catch2/catch_test_macros.hpp>

#include <fstream>
#include <set>
#include <sstream>
#include <string>

#ifndef CRISPASR_SOURCE_DIR
#error "CRISPASR_SOURCE_DIR must be defined by the build"
#endif

namespace {

std::string read_file(const std::string& rel) {
    const std::string path = std::string(CRISPASR_SOURCE_DIR) + "/" + rel;
    std::ifstream f(path);
    INFO("reading " << rel);
    REQUIRE(f.good());
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

bool contains(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

using core_omnivoice_instruct::parse;
using core_omnivoice_instruct::render;
using core_omnivoice_instruct::Status;

// Convenience: parse + render against non-Chinese text, the common case.
std::string resolve_en(const std::string& s) {
    return render(parse(s), /*text_has_zh=*/false);
}

} // namespace

// ---------------------------------------------------------------------------
// The predicate.
// ---------------------------------------------------------------------------

// The defect in one assertion: what a user naturally types must come out in the
// spelling the model was trained on.
TEST_CASE("omnivoice instruct: casing is normalised to the trained spelling", "[unit][omnivoice]") {
    REQUIRE(resolve_en("Male, British Accent") == "male, british accent");
    REQUIRE(resolve_en("FEMALE, ELDERLY") == "female, elderly");
    REQUIRE(resolve_en("male, british accent") == "male, british accent");
}

TEST_CASE("omnivoice instruct: separators and whitespace are repaired", "[unit][omnivoice]") {
    // Upstream treats a wrong-width comma and stray spacing as "minor issues,
    // auto-fixed" — it splits on both and re-joins canonically.
    REQUIRE(resolve_en("male，elderly") == "male, elderly");
    REQUIRE(resolve_en("  male ,   elderly  ") == "male, elderly");
    REQUIRE(resolve_en(",male,,elderly,") == "male, elderly");
}

TEST_CASE("omnivoice instruct: empty means no voice design", "[unit][omnivoice]") {
    for (const char* v : {"", "   ", ",", " , "}) {
        INFO("value '" << v << "'");
        REQUIRE(parse(v).status == Status::cleared);
        REQUIRE(resolve_en(v).empty());
    }
}

// Upstream RAISES here; we reject with the same shape rather than degrading,
// because a voice-design request that silently does nothing is the bug.
TEST_CASE("omnivoice instruct: an unsupported item is REJECTED, not ignored", "[unit][omnivoice]") {
    for (const char* v : {"britsh accent", "gruff", "male, wizard", "sehr tief"}) {
        INFO("value '" << v << "'");
        const auto p = parse(v);
        REQUIRE(p.status == Status::unknown_item);
        REQUIRE(!p.error.empty());
        REQUIRE(render(p, false).empty());
    }
    // ...and the error names the offending item, with a near-miss when there is
    // one. The suggestion text is best-effort (we use an LCS ratio where
    // upstream uses difflib), so assert only that it fires.
    const auto p = parse("britsh accent");
    REQUIRE(contains(p.error, "britsh accent"));
    REQUIRE(contains(p.error, "did you mean"));
}

TEST_CASE("omnivoice instruct: a dialect cannot be mixed with an accent", "[unit][omnivoice]") {
    const auto p = parse("河南话, british accent");
    REQUIRE(p.status == Status::mixed_dialect_accent);
    REQUIRE(render(p, false).empty());
}

TEST_CASE("omnivoice instruct: two items from one category conflict", "[unit][omnivoice]") {
    REQUIRE(parse("male, female").status == Status::category_conflict);
    REQUIRE(parse("child, elderly").status == Status::category_conflict);
    REQUIRE(parse("low pitch, high pitch").status == Status::category_conflict);
    // Cross-language pairs are the same category upstream (each exclusivity
    // group holds both spellings), so this must conflict too.
    REQUIRE(parse("male, 女").status == Status::category_conflict);
    // Different categories are fine.
    REQUIRE(parse("male, elderly, high pitch, whisper").status == Status::ok);
}

// The text-dependent half of the blueprint: everything is unified to ONE
// language, chosen by the text unless a dialect or an accent forces it.
TEST_CASE("omnivoice instruct: unified to the language of the text", "[unit][omnivoice]") {
    const auto p = parse("male, elderly");
    REQUIRE(render(p, /*text_has_zh=*/false) == "male, elderly");
    REQUIRE(render(p, /*text_has_zh=*/true) == "男，老年");
    // Chinese input against English text goes the other way.
    REQUIRE(render(parse("男，老年"), false) == "male, elderly");
}

TEST_CASE("omnivoice instruct: a dialect forces Chinese, an accent forces English", "[unit][omnivoice]") {
    // A dialect wins even when the text is English...
    REQUIRE(render(parse("male, 河南话"), /*text_has_zh=*/false) == "男，河南话");
    // ...and an accent wins even when the text is Chinese.
    REQUIRE(render(parse("male, british accent"), /*text_has_zh=*/true) == "male, british accent");
}

TEST_CASE("omnivoice instruct: the separator follows the rendered language", "[unit][omnivoice]") {
    REQUIRE(contains(render(parse("male, elderly"), true), "，"));
    REQUIRE(contains(render(parse("male, elderly"), false), ", "));
}

TEST_CASE("omnivoice instruct: item order is preserved", "[unit][omnivoice]") {
    // The prompt is a token sequence; reordering it would change conditioning.
    REQUIRE(resolve_en("elderly, male") == "elderly, male");
    REQUIRE(resolve_en("male, elderly") == "male, elderly");
}

TEST_CASE("omnivoice instruct: the generated table is well-formed", "[unit][omnivoice]") {
    // Upstream derives 48 valid items across 6 categories; the generator
    // asserts that against _INSTRUCT_ALL_VALID, this asserts it did not collapse.
    REQUIRE(core_omnivoice_instruct::kInstructTableN == 48);
    REQUIRE(core_omnivoice_instruct::kInstructGroups == 6);

    std::set<std::string> seen;
    for (int i = 0; i < core_omnivoice_instruct::kInstructTableN; i++) {
        const auto& e = core_omnivoice_instruct::kInstructTable[i];
        INFO("row " << i << " '" << e.item << "'");
        REQUIRE(e.item != nullptr);
        REQUIRE(seen.insert(e.item).second); // no duplicates
        REQUIRE(core_omnivoice_instruct::is_valid_item(e.item));
        // A translatable item must round-trip: its counterpart is also a valid
        // item, in the same category, in the other language.
        if (e.counterpart) {
            REQUIRE(core_omnivoice_instruct::is_valid_item(e.counterpart));
            REQUIRE(render(parse(e.item), !e.is_zh) == e.counterpart);
        }
    }
}

// ---------------------------------------------------------------------------
// The joins.
// ---------------------------------------------------------------------------

TEST_CASE("omnivoice instruct: the runtime validates and renders per text", "[unit][omnivoice]") {
    const std::string src = read_file("src/omnivoice.cpp");
    REQUIRE(contains(src, "core/omnivoice_instruct.h"));
    // Validation at set time...
    REQUIRE(contains(src, "core_omnivoice_instruct::parse("));
    // ...and rendering at synthesis time, against the TARGET text. Baking the
    // rendered string at set time would freeze one line's EN/ZH choice onto
    // every later line on a reused server context.
    REQUIRE(contains(src, "core_omnivoice_instruct::render(ctx->instruct, "
                          "core_omnivoice_instruct::text_is_zh(text))"));
}

// Same per-call bug the language knob had: the server maps a per-request
// "instructions" field onto params.tts_instruct, and init()-only application
// left it dead on every line after the first.
TEST_CASE("omnivoice instruct: the CLI adapter applies it PER CALL", "[unit][omnivoice]") {
    const std::string src = read_file("examples/cli/crispasr_backend_omnivoice.cpp");
    const size_t synth = src.find("std::vector<float> synthesize(");
    REQUIRE(synth != std::string::npos);
    const std::string body = src.substr(synth);
    REQUIRE(contains(body, "omnivoice_set_instruct("));
    // And a rejection must abort the synthesis, not silently drop the design.
    REQUIRE(contains(body, "omnivoice_set_instruct(ctx_, params.tts_instruct.c_str()) != 0"));
}

TEST_CASE("omnivoice instruct: a bad --tts-instruct fails init", "[unit][omnivoice]") {
    const std::string src = read_file("examples/cli/crispasr_backend_omnivoice.cpp");
    const size_t init = src.find("bool init(");
    REQUIRE(init != std::string::npos);
    const size_t synth = src.find("std::vector<float> synthesize(");
    const std::string body = src.substr(init, synth - init);
    REQUIRE(contains(body, "omnivoice_set_instruct(ctx_, p.tts_instruct.c_str()) != 0"));
    REQUIRE(contains(body, "return false"));
}

// The third backend to be missing from a session-ABI dispatch for the same
// reason. `crispasr_session_set_instruct` existed and handled qwen3-tts and
// parler; omnivoice fell through to `return -3` — "this backend has no instruct
// contract" — which is why voice design was unreachable from every binding.
TEST_CASE("omnivoice instruct: the session ABI dispatches to omnivoice", "[unit][omnivoice]") {
    const std::string src = read_file("src/crispasr_c_api.cpp");
    const size_t fn = src.find("crispasr_session_set_instruct(crispasr_session* s, const char* instruct) {");
    INFO("crispasr_session_set_instruct not found");
    REQUIRE(fn != std::string::npos);
    // Bound the search to the function body so a mention elsewhere in this
    // 10k-line file cannot make the guard pass.
    const size_t end = src.find("\n}\n", fn);
    REQUIRE(end != std::string::npos);
    const std::string body = src.substr(fn, end - fn);
    REQUIRE(contains(body, "s->omnivoice_ctx"));
    REQUIRE(contains(body, "omnivoice_set_instruct(s->omnivoice_ctx, instruct)"));
}

TEST_CASE("omnivoice instruct: the server rejects a bad value with 400", "[unit][omnivoice]") {
    const std::string src = read_file("examples/cli/crispasr_server.cpp");
    REQUIRE(contains(src, "core/omnivoice_instruct.h"));
    REQUIRE(contains(src, "core_omnivoice_instruct::parse(instructions)"));
    REQUIRE(contains(src, "invalid_instructions"));
}
