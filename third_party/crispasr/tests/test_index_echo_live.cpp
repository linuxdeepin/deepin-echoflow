#include <catch2/catch_test_macros.hpp>
#include "index_echo.h"
#include "core/wav_reader.h"
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

TEST_CASE("Index-Echo bilingual cues and file-history reset", "[integration][index-echo]") {
    const char* model = std::getenv("CRISPASR_MODEL_INDEX_ECHO");
    if (!model || !*model)
        SKIP("CRISPASR_MODEL_INDEX_ECHO not set; matching decoder companion required");
    auto params = index_echo_context_default_params();
    params.use_gpu = false;
    params.n_threads = 4;
    params.verbosity = 0;
    std::unique_ptr<index_echo_context, decltype(&index_echo_free)> ctx(index_echo_init_from_file(model, params),
                                                                        index_echo_free);
    REQUIRE(ctx != nullptr);
    CHECK_FALSE(index_echo_set_target_lang(ctx.get(), "invalid"));
    REQUIRE(index_echo_set_target_lang(ctx.get(), "en"));
    std::vector<float> pcm;
    int rate = 0;
    REQUIRE(crispasr::core::read_wav_mono_pcm16("samples/jfk.wav", pcm, rate));
    REQUIRE(rate == 16000);
    std::string previous;
    for (int repeat = 0; repeat < 2; ++repeat) {
        std::unique_ptr<index_echo_result, decltype(&index_echo_result_free)> result(
            index_echo_transcribe(ctx.get(), pcm.data(), (int)pcm.size()), index_echo_result_free);
        REQUIRE(result != nullptr);
        REQUIRE(result->n_cues == 3);
        CHECK(result->parse_warnings == 0);
        CHECK(std::string(result->cues[0].transcript) == "And so my fellow Americans,");
        CHECK(std::string(result->cues[0].translation) == "So, my fellow Americans,");
        CHECK(std::string(result->cues[1].transcript) == "ask not what your country can do for you,");
        CHECK(std::string(result->cues[1].translation) == "do not ask what your country can do for you,");
        CHECK(std::string(result->cues[2].transcript) == "ask what you can do for your country.");
        CHECK(std::string(result->cues[2].translation) == "ask what you can do for your country.");
        CHECK(result->cues[0].start_seconds == 0.0);
        CHECK(result->cues[2].end_seconds == 11.0);
        if (repeat)
            CHECK(std::string(result->raw_text) == previous);
        previous = result->raw_text;
    }
}
