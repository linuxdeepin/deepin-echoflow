// test-cohere-lang-gguf.cpp — round-trip the Cohere supported-language list
// through a real GGUF file.
//
// cohere_lang::resolve() is only as good as the list it is handed, and that
// list travels as a GGUF string array written by the converter and read by
// core_gguf::kv_str_array. This pins that seam: written array in, same array
// out, and a file WITHOUT the key must read back empty ("unknown") rather than
// as an empty whitelist that would reject every language.

#include <catch2/catch_test_macros.hpp>

#include "cohere-arch.h"
#include "cohere_lang.h"
#include "core/gguf_loader.h"
#include "gguf.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

std::string temp_gguf_path(const char* stem) {
    const char* dir = std::getenv("TMPDIR");
    std::string d = (dir && *dir) ? std::string(dir) : std::string("/tmp");
    if (!d.empty() && d.back() == '/')
        d.pop_back();
    return d + "/crispasr-test-" + stem + ".gguf";
}

// Write a metadata-only GGUF carrying `langs` under the converter's key.
void write_lang_gguf(const std::string& path, const std::vector<std::string>& langs) {
    gguf_context* ctx = gguf_init_empty();
    REQUIRE(ctx != nullptr);
    gguf_set_val_str(ctx, "general.architecture", "cohere-transcribe");
    if (!langs.empty()) {
        std::vector<const char*> ptrs;
        ptrs.reserve(langs.size());
        for (const auto& l : langs)
            ptrs.push_back(l.c_str());
        gguf_set_arr_str(ctx, CT_KEY_SUPPORTED_LANGS, ptrs.data(), ptrs.size());
    }
    REQUIRE(gguf_write_to_file(ctx, path.c_str(), /*only_meta=*/true));
    gguf_free(ctx);
}

} // namespace

TEST_CASE("cohere supported_languages survives a GGUF round-trip", "[unit][cohere]") {
    const std::vector<std::string> arabic = {"en", "ar"};
    const std::string path = temp_gguf_path("cohere-langs-ar");
    write_lang_gguf(path, arabic);

    gguf_context* g = core_gguf::open_metadata(path.c_str());
    REQUIRE(g != nullptr);
    const auto read_back = core_gguf::kv_str_array(g, CT_KEY_SUPPORTED_LANGS);
    core_gguf::free_metadata(g);
    std::remove(path.c_str());

    REQUIRE(read_back == arabic);

    // And the list actually drives the decision it exists for.
    REQUIRE(cohere_lang::resolve(read_back, "de").substituted);
    REQUIRE_FALSE(cohere_lang::resolve(read_back, "ar").substituted);
}

TEST_CASE("cohere GGUF without the language key reads back as unknown", "[unit][cohere]") {
    // Every GGUF published before the key existed lands here. Empty must mean
    // "no information", never "supports nothing" — otherwise the fallback would
    // rewrite every user's language.
    const std::string path = temp_gguf_path("cohere-langs-none");
    write_lang_gguf(path, {});

    gguf_context* g = core_gguf::open_metadata(path.c_str());
    REQUIRE(g != nullptr);
    const auto read_back = core_gguf::kv_str_array(g, CT_KEY_SUPPORTED_LANGS);
    core_gguf::free_metadata(g);
    std::remove(path.c_str());

    REQUIRE(read_back.empty());
    REQUIRE_FALSE(cohere_lang::resolve(read_back, "ru").substituted);
    REQUIRE(cohere_lang::resolve(read_back, "ru").lang == "ru");
}

TEST_CASE("cohere supported_languages round-trips the full base-model list", "[unit][cohere]") {
    const std::vector<std::string> base = {"en", "fr", "de", "es", "it", "pt", "nl",
                                           "pl", "el", "ar", "ja", "zh", "vi", "ko"};
    const std::string path = temp_gguf_path("cohere-langs-base");
    write_lang_gguf(path, base);

    gguf_context* g = core_gguf::open_metadata(path.c_str());
    REQUIRE(g != nullptr);
    const auto read_back = core_gguf::kv_str_array(g, CT_KEY_SUPPORTED_LANGS);
    core_gguf::free_metadata(g);
    std::remove(path.c_str());

    REQUIRE(read_back.size() == 14);
    REQUIRE(read_back == base);
}
