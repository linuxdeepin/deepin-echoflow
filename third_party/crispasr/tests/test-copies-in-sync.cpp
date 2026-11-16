// test-copies-in-sync.cpp — every duplicated implementation must not drift.
//
// `crisp_punc/src/fireredpunc.cpp` is the shared library CrispASR normally links
// (and CrispEmbed consumes via add_subdirectory). `src/fireredpunc.cpp` is the
// fallback src/CMakeLists.txt builds when the crisp_punc/ directory is absent
// from a checkout. They are the same implementation twice.
//
// That duplication silently ate a bug fix. #308 fixed a capitalisation defect —
// a pending capitalisation was not cleared by an ALREADY-uppercase letter, so
// "And" became "ANd" on every backend whose model emits sentence-cased text —
// but the fix landed in the FALLBACK copy only. The shipping copy kept the bug
// for months, and every symptom pointed at the file that was already correct.
//
// So: assert the two are byte-identical apart from their one legitimate
// difference, the header they include. A future fix applied to one copy now
// fails here instead of appearing to work.
//
// EXTENDED 2026-08-03 from the punc pair to all 14 duplicated files, after a
// survey found the same bug already sitting in an uncovered one: `src/pcs.cpp`
// was missing the `__has_include("imatrix.h")` hook and the PCS_DUMP_LOGITS
// diff dump that `crisp_punc/src/pcs.cpp` carries — 32 lines — while the shared
// copy's own comment read "Do NOT let this file diverge between the two repos".
// A test that guards one file out of fourteen is a test that says the other
// thirteen are fine.
//
// SANCTIONED DIFFERENCES are listed per pair below, not tolerated globally. A
// blanket "ignore #include lines and #ifdefs" would have hidden the pcs.cpp
// drift, which is exactly an #include and an #ifdef.
//
// Pure file comparison — no model, no audio, no linkage against either copy.

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#ifndef CRISPASR_SOURCE_DIR
#error "CRISPASR_SOURCE_DIR must be defined by the build"
#endif

namespace {

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.good());
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// One duplicated file, and the differences that are allowed to remain.
struct Pair {
    const char* shared;   // the copy the shared library builds
    const char* fallback; // the copy src/CMakeLists.txt builds on a partial checkout
    // Applied to the SHARED copy before comparing. Each entry is a specific,
    // justified difference — never a general pattern.
    const char* from;
    const char* to;
    const char* why;
};

// Every .cpp/.h that exists twice. Enumerated rather than globbed: a new
// duplicated file should be a deliberate addition here, not something a glob
// silently starts or stops covering.
const Pair kPairs[] = {
    {"/crisp_punc/src/fireredpunc.cpp", "/src/fireredpunc.cpp", "#include \"crisp_punc.h\"",
     "#include \"fireredpunc.h\"", "each copy includes its own public header"},
    {"/crisp_punc/src/pcs.cpp", "/src/pcs.cpp", "#include \"crisp_punc.h\"", "#include \"pcs.h\"",
     "each copy includes its own public header"},

    // crisp_lid. The shared copies must compile inside CrispEmbed too, hence
    // the two structural allowances below; everything else must match.
    // Both copies declare `core_gguf::tensor_map`, which is what gguf_loader.h
    // asks for ("each repo's copy declares core_gguf::tensor_map so it tracks",
    // line 107). The fallback used to spell the std::map out and was the one
    // out of step — synced rather than normalised apart.
    {"/crisp_lid/src/lid_cld3.cpp", "/src/lid_cld3.cpp", nullptr, nullptr, nullptr},
    {"/crisp_lid/src/lid_fasttext.cpp", "/src/lid_fasttext.cpp", nullptr, nullptr, nullptr},
    {"/crisp_lid/src/lid_cld3.h", "/src/lid_cld3.h", nullptr, nullptr, nullptr},
    {"/crisp_lid/src/lid_fasttext.h", "/src/lid_fasttext.h", nullptr, nullptr, nullptr},
    // The shared copy wraps its CrispASR-only includes in #ifdef CRISPASR_BUILD
    // so it also compiles inside CrispEmbed. src/CMakeLists.txt:2425 defines
    // CRISPASR_BUILD for the fallback target, so those guards are a no-op there
    // — the copies are kept byte-identical rather than normalised apart.
    {"/crisp_lid/src/text_lid_dispatch.cpp", "/src/text_lid_dispatch.cpp", nullptr, nullptr, nullptr},
    {"/crisp_lid/src/text_lid_dispatch.h", "/src/text_lid_dispatch.h", nullptr, nullptr, nullptr},

    // crisp_truecase — currently byte-identical throughout.
    {"/crisp_truecase/src/truecaser.cpp", "/src/truecaser.cpp", nullptr, nullptr, nullptr},
    {"/crisp_truecase/src/truecaser.h", "/src/truecaser.h", nullptr, nullptr, nullptr},
    {"/crisp_truecase/src/truecaser_crf.cpp", "/src/truecaser_crf.cpp", nullptr, nullptr, nullptr},
    {"/crisp_truecase/src/truecaser_crf.h", "/src/truecaser_crf.h", nullptr, nullptr, nullptr},
    {"/crisp_truecase/src/truecaser_lstm.cpp", "/src/truecaser_lstm.cpp", nullptr, nullptr, nullptr},
    {"/crisp_truecase/src/truecaser_lstm.h", "/src/truecaser_lstm.h", nullptr, nullptr, nullptr},
};

std::string normalize(std::string s, const Pair& p) {
    if (!p.from)
        return s;
    const std::string from = p.from, to = p.to;
    size_t at = 0;
    while ((at = s.find(from, at)) != std::string::npos) {
        s.replace(at, from.size(), to);
        at += to.size();
    }
    return s;
}

size_t first_differing_line(const std::string& a, const std::string& b) {
    size_t line = 1, i = 0;
    for (; i < a.size() && i < b.size() && a[i] == b[i]; i++)
        if (a[i] == '\n')
            line++;
    return line;
}

} // namespace

TEST_CASE("every duplicated implementation is in sync", "[unit][punc]") {
    const std::string root = CRISPASR_SOURCE_DIR;
    for (const Pair& p : kPairs) {
        INFO("pair: " << p.shared << "  vs  " << p.fallback);
        const std::string shared = normalize(read_file(root + p.shared), p);
        const std::string fallback = read_file(root + p.fallback);
        if (shared != fallback) {
            FAIL("" << p.shared << " and " << p.fallback << " diverge at line "
                    << first_differing_line(shared, fallback)
                    << ". They are the same implementation twice (shared library vs "
                       "partial-checkout fallback) — apply the change to BOTH. #308's "
                       "capitalisation fix went to the fallback only and was dead code for "
                       "months; src/pcs.cpp was missing the imatrix hook and the logits dump "
                       "for as long again.");
        }
    }
    SUCCEED();
}

TEST_CASE("the sync list covers every file that exists twice", "[unit][punc]") {
    // The list above is hand-maintained, which is the point — but a new
    // duplicated file added without an entry would be silently unguarded, which
    // is the bug this whole test exists for. Cross-check it against the
    // directories rather than trusting it.
    const std::string root = CRISPASR_SOURCE_DIR;
    for (const char* dir : {"crisp_punc", "crisp_lid", "crisp_truecase"}) {
        const std::filesystem::path shared_dir = std::filesystem::path(root) / dir / "src";
        if (!std::filesystem::exists(shared_dir))
            continue; // partial checkout: nothing to compare against
        for (const auto& e : std::filesystem::directory_iterator(shared_dir)) {
            const std::string ext = e.path().extension().string();
            if (ext != ".cpp" && ext != ".h")
                continue;
            const std::string name = e.path().filename().string();
            if (!std::filesystem::exists(std::filesystem::path(root) / "src" / name))
                continue; // only exists in the shared lib — not a duplicate
            const std::string rel = std::string("/") + dir + "/src/" + name;
            bool listed = false;
            for (const Pair& p : kPairs)
                if (rel == p.shared)
                    listed = true;
            INFO("unlisted duplicate: " << rel);
            REQUIRE(listed);
        }
    }
}

TEST_CASE("both copies carry the #308 capitalisation guard", "[unit][punc]") {
    // A targeted check on top of the identity test: if someone "fixes" the
    // divergence by deleting the guard from both, identity still passes. This
    // pins the behaviour itself — an already-uppercase letter must satisfy the
    // pending capitalisation.
    const std::string root = CRISPASR_SOURCE_DIR;
    for (const char* rel : {"/crisp_punc/src/fireredpunc.cpp", "/src/fireredpunc.cpp"}) {
        const std::string src = read_file(root + rel);
        INFO("file: " << rel);
        REQUIRE(src.find("if (c >= 'A' && c <= 'Z') {") != std::string::npos);
    }
}
