// test-voxtral-pretokenize.cpp — the tekken regex, as the runtime implements it.
//
// #338 was a vocabulary bound; measuring that fix against `mistral-common`
// afterwards showed the pre-tokenizer disagreeing on a second axis entirely.
// Two defects, both about who owns a character at a boundary:
//
//   * a whitespace run must hand its LAST character to the following token
//     (`a  b` -> "a", " ", " b"), because the letter and punctuation
//     alternatives each open with an optional leading character. The runtime
//     swallowed the whole run instead.
//   * every byte >= 0x80 was classified as a letter, which is right for scripts
//     and wrong for the punctuation and symbols above ASCII — an en dash or a
//     curly quote split a `[^\s\p{L}\p{N}]+` run in two.
//
// The expectations below are GENERATED from `config.pattern` in the published
// tekken.json, run through Python's `regex` engine (which implements real
// \p{L}). That is an independent oracle, and it pins the SPLIT directly rather
// than comparing token ids — so this test needs neither the 15 MB vocabulary
// nor a network. The live id-level check against mistral-common is
// `tools/check-voxtral-tokenizer-parity.py`.

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include "voxtral_tekken_vocab.h"

// GENERATED from the tekken.json `config.pattern` via Python `regex`
// (tools/…/gen_golden.py). Do not hand-edit; regenerate if the pattern changes.
static const struct {
    const char* in;
    const char* pieces[16];
} kGolden[] = {
    {"a b", {"a", " b", nullptr}},
    {"a  b", {"a", " ", " b", nullptr}},
    {"a   b", {"a", "  ", " b", nullptr}},
    {"a 1", {"a", " ", "1", nullptr}},
    {"a  1", {"a", " ", " ", "1", nullptr}},
    {"a [", {"a", " [", nullptr}},
    {"a  [", {"a", " ", " [", nullptr}},
    {"a   [", {"a", "  ", " [", nullptr}},
    {"a - b", {"a", " -", " b", nullptr}},
    {"a[b", {"a", "[b", nullptr}},
    {" a", {" a", nullptr}},
    {"  a", {" ", " a", nullptr}},
    {"a ", {"a", " ", nullptr}},
    {"a  ", {"a", "  ", nullptr}},
    {"a\x09"
     "b",
     {"a",
      "\x09"
      "b",
      nullptr}},
    {"a \x09"
     "b",
     {"a", " ",
      "\x09"
      "b",
      nullptr}},
    {"a\x09 b", {"a", "\x09", " b", nullptr}},
    {"a\x09[", {"a", "\x09", "[", nullptr}},
    {"a  B", {"a", " ", " B", nullptr}},
    {"hello world", {"hello", " world", nullptr}},
    {"don\xe2\x80\x99t stop", {"don", "\xe2\x80\x99t", " stop", nullptr}},
    {"l\xe2\x80\x99uomo", {"l", "\xe2\x80\x99uomo", nullptr}},
    {"\xe2\x80\x9cquoted\xe2\x80\x9d", {"\xe2\x80\x9cquoted", "\xe2\x80\x9d", nullptr}},
    {"a\xe2\x80\x94"
     "b",
     {"a",
      "\xe2\x80\x94"
      "b",
      nullptr}},
    {"x\xe2\x80\xa6y", {"x", "\xe2\x80\xa6y", nullptr}},
    {"\xe2\x82\xac"
     "100",
     {"\xe2\x82\xac", "1", "0", "0", nullptr}},
    {"50\xc2\xb0"
     "C",
     {"5", "0",
      "\xc2\xb0"
      "C",
      nullptr}},
    {"\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e text", {"\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e", " text", nullptr}},
    {"emoji \xf0\x9f\x8e\x89 party", {"emoji", " \xf0\x9f\x8e\x89", " party", nullptr}},
    {"a/b/c", {"a", "/b", "/c", nullptr}},
    {"key=value; x=1", {"key", "=value", ";", " x", "=", "1", nullptr}},
    {"(paren) [brack] {brace}", {"(paren", ")", " [", "brack", "]", " {", "brace", "}", nullptr}},
    {"multiple   internal   spaces", {"multiple", "  ", " internal", "  ", " spaces", nullptr}},
    {"\xc2\xbb\xe2\x80\x9d.&", {"\xc2\xbb\xe2\x80\x9d.&", nullptr}},
    {"1234", {"1", "2", "3", "4", nullptr}},
    {"a1234", {"a", "1", "2", "3", "4", nullptr}},
};

TEST_CASE("tekken pre-tokenizer matches the published regex", "[unit][voxtral][pretokenize]") {
    for (const auto& g : kGolden) {
        std::vector<std::string> want;
        for (int i = 0; i < 16 && g.pieces[i]; i++)
            want.push_back(g.pieces[i]);
        INFO("input: " << g.in);
        CHECK(voxtral_tekken::pre_tokenize(g.in) == want);
    }
}

TEST_CASE("pre-tokenizer is total: pieces always rejoin to the input", "[unit][voxtral][pretokenize]") {
    // A tokenizer that drops or duplicates bytes is worse than one that splits
    // oddly — the model would be conditioned on text the caller never wrote.
    const char* inputs[] = {
        "",
        " ",
        "  ",
        "\t",
        "a",
        "abc",
        "\xc3\xa8",
        "\xe2\x80\x99",
        "\xf0\x9f\x8e\x89",
        "a \xe2\x80\x94 b",
        "  \t  ",
        "1a!",
        "\xff\xfe",
        "a\xc3",
        "\x80\x80",
    };
    for (const char* s : inputs) {
        std::string joined;
        for (const auto& p : voxtral_tekken::pre_tokenize(s))
            joined += p;
        INFO("input bytes: " << std::string(s).size());
        CHECK(joined == std::string(s));
    }
}

TEST_CASE("pre-tokenizer terminates on malformed UTF-8", "[unit][voxtral][pretokenize]") {
    // Truncated sequences and stray continuation bytes must still advance the
    // cursor; a stall here is an infinite loop inside synthesis.
    const char* bad[] = {"\xe2", "\xe2\x80", "\x80", "\xbf\xbf", "a\xf0\x9f", "\xf0\x9f\x8e"};
    for (const char* s : bad) {
        std::string joined;
        for (const auto& p : voxtral_tekken::pre_tokenize(s))
            joined += p;
        CHECK(joined == std::string(s));
    }
}
