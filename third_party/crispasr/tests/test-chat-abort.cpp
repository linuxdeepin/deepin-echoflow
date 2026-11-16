// test-chat-abort.cpp — cancellation over the crispasr_chat_* C ABI.
//
// Gated on CRISPASR_CHAT_TEST_MODEL — a path to a small GGUF chat model
// (e.g. gemma-3-1b-it-Q4_K_M.gguf, qwen2.5-0.5b-instruct, smollm2-360m).
// When unset every case is reported as SKIPPED so unrelated builds stay
// green without a model on disk.
//
// Covers `crispasr_chat_set_abort_callback`:
//   • a callback that aborts after N delivered pieces stops the stream
//     with CRISPASR_CHAT_ERR_ABORTED, short of max_tokens
//   • a callback that never aborts produces the output a session with no
//     callback produces
//   • clearing with NULL restores unaborted behaviour
//   • an abort requested before the call delivers no piece at all
//   • the session is reusable after an abort
//   • a NULL session is a no-op
//   • an abort during a multi-piece prefill returns before the remaining
//     prompt batches are decoded

#include <catch2/catch_test_macros.hpp>

#include "crispasr_chat.h"

#include <cstdlib>
#include <cstring>
#include <string>

namespace {

const char* test_model_path() {
    return std::getenv("CRISPASR_CHAT_TEST_MODEL");
}

// Shared user data for both callbacks: the abort hook records how often
// it was consulted and decides from what the token hook has delivered.
struct recorder {
    int calls = 0;    // abort-callback invocations
    int pieces = 0;   // token pieces delivered
    std::string text; // concatenated pieces

    // Abort once `calls` exceeds this, if non-negative.
    int allow_calls = -1;
    // Abort once `pieces` reaches this, if non-negative.
    int abort_at_pieces = -1;
};

// Returns false to abort — the ABI's convention.
bool abort_hook(void* user) {
    auto* r = static_cast<recorder*>(user);
    r->calls += 1;
    if (r->allow_calls >= 0 && r->calls > r->allow_calls) {
        return false;
    }
    if (r->abort_at_pieces >= 0 && r->pieces >= r->abort_at_pieces) {
        return false;
    }
    return true;
}

void on_token_recorder(const char* chunk, void* user) {
    auto* r = static_cast<recorder*>(user);
    r->pieces += 1;
    r->text.append(chunk);
}

crispasr_chat_open_params short_prompt_open_params() {
    crispasr_chat_open_params op;
    crispasr_chat_open_params_default(&op);
    op.n_gpu_layers = -1;
    op.n_ctx = 1024;
    return op;
}

// Prefill `messages` on a fresh session opened at `n_batch`, with an abort
// hook that always says continue, and return how many times that hook was
// consulted. prefill_only, so no sampled token contributes.
//
// n_ubatch follows n_batch so one prompt piece is one decode is one graph;
// left at its default, a large batch would be split into several ubatches and
// the per-piece cost would stop being a constant.
int prefill_consultations(const char* model, int32_t n_batch, const crispasr_chat_message* messages,
                          size_t n_messages) {
    crispasr_chat_open_params op;
    crispasr_chat_open_params_default(&op);
    op.n_gpu_layers = -1;
    op.n_ctx = 2048;
    op.n_batch = n_batch;
    op.n_ubatch = n_batch;

    crispasr_chat_error err{};
    crispasr_chat_session_t s = crispasr_chat_open(model, &op, &err);
    REQUIRE(s != nullptr);

    crispasr_chat_generate_params gp;
    crispasr_chat_generate_params_default(&gp);
    gp.max_tokens = 8;
    gp.temperature = 0.0f;
    gp.seed = 1;
    gp.prefill_only = true;

    recorder r;
    crispasr_chat_set_abort_callback(s, abort_hook, &r);
    REQUIRE(crispasr_chat_generate_stream(s, messages, n_messages, &gp, on_token_recorder, &r, &err) == 0);
    REQUIRE(r.pieces == 0);
    crispasr_chat_close(s);
    return r.calls;
}

crispasr_chat_generate_params greedy_params(int32_t max_tokens) {
    crispasr_chat_generate_params gp;
    crispasr_chat_generate_params_default(&gp);
    gp.max_tokens = max_tokens;
    gp.temperature = 0.0f; // greedy → the same prompt gives the same text
    gp.seed = 1;
    return gp;
}

const crispasr_chat_message* short_messages() {
    static const crispasr_chat_message msgs[] = {
        {"system", "You are a helpful assistant."},
        {"user", "List the days of the week, one per line."},
    };
    return msgs;
}
constexpr size_t kShortMessages = 2;

// Whitespace-separated words are never fewer tokens than words, so this
// body is at least `kLongPromptWords` prompt tokens — above two 512-token
// prompt batches and below the 2048-token context the prefill case opens
// with.
constexpr int kLongPromptSentences = 120;
constexpr int kLongPromptWords = 1024;

std::string long_user_message() {
    std::string body;
    body.reserve(kLongPromptSentences * 45);
    for (int i = 0; i < kLongPromptSentences; ++i) {
        body += "The quick brown fox jumps over the lazy dog. ";
    }
    body += "\nReply with the single word: fox.";
    return body;
}

size_t word_count(const std::string& text) {
    size_t words = 0;
    bool in_word = false;
    for (const char c : text) {
        const bool space = c == ' ' || c == '\n' || c == '\t' || c == '\r';
        if (!space && !in_word) {
            words += 1;
        }
        in_word = !space;
    }
    return words;
}

} // namespace

TEST_CASE("crispasr_chat abort callback stops a stream after N pieces", "[chat][abort]") {
    const char* model = test_model_path();
    if (!model) {
        SKIP("CRISPASR_CHAT_TEST_MODEL not set; skipping abort-after-N-pieces");
    }

    const crispasr_chat_open_params op = short_prompt_open_params();
    crispasr_chat_error err{};
    crispasr_chat_session_t s = crispasr_chat_open(model, &op, &err);
    REQUIRE(s != nullptr);
    REQUIRE(err.code == 0);

    constexpr int kAbortAtPieces = 3;
    constexpr int32_t kMaxTokens = 32;
    crispasr_chat_generate_params gp = greedy_params(kMaxTokens);

    recorder r;
    r.abort_at_pieces = kAbortAtPieces;
    crispasr_chat_set_abort_callback(s, abort_hook, &r);

    const int32_t rc =
        crispasr_chat_generate_stream(s, short_messages(), kShortMessages, &gp, on_token_recorder, &r, &err);
    REQUIRE(rc == CRISPASR_CHAT_ERR_ABORTED);
    REQUIRE(err.code == CRISPASR_CHAT_ERR_ABORTED);
    // The hook is consulted before each sampled token, so the piece that
    // trips it is the last one delivered.
    REQUIRE(r.pieces == kAbortAtPieces);
    REQUIRE(r.pieces < kMaxTokens);
    REQUIRE_FALSE(r.text.empty());

    crispasr_chat_close(s);
}

TEST_CASE("crispasr_chat a never-aborting callback matches no callback", "[chat][abort]") {
    const char* model = test_model_path();
    if (!model) {
        SKIP("CRISPASR_CHAT_TEST_MODEL not set; skipping never-abort equivalence");
    }

    const crispasr_chat_open_params op = short_prompt_open_params();
    crispasr_chat_error err{};
    crispasr_chat_session_t s = crispasr_chat_open(model, &op, &err);
    REQUIRE(s != nullptr);

    crispasr_chat_generate_params gp = greedy_params(16);

    char* baseline = crispasr_chat_generate(s, short_messages(), kShortMessages, &gp, &err);
    REQUIRE(baseline != nullptr);
    REQUIRE(err.code == 0);
    const std::string without_callback = baseline;
    crispasr_chat_string_free(baseline);
    REQUIRE_FALSE(without_callback.empty());

    REQUIRE(crispasr_chat_reset(s, &err) == 0);

    recorder r; // allow_calls / abort_at_pieces left at -1 → never aborts
    crispasr_chat_set_abort_callback(s, abort_hook, &r);
    char* guarded = crispasr_chat_generate(s, short_messages(), kShortMessages, &gp, &err);
    REQUIRE(guarded != nullptr);
    REQUIRE(err.code == 0);
    const std::string with_callback = guarded;
    crispasr_chat_string_free(guarded);

    REQUIRE(with_callback == without_callback);
    REQUIRE(r.calls > 0); // the hook really was consulted

    crispasr_chat_close(s);
}

TEST_CASE("crispasr_chat clearing the abort callback restores generation", "[chat][abort]") {
    const char* model = test_model_path();
    if (!model) {
        SKIP("CRISPASR_CHAT_TEST_MODEL not set; skipping NULL-clears-callback");
    }

    const crispasr_chat_open_params op = short_prompt_open_params();
    crispasr_chat_error err{};
    crispasr_chat_session_t s = crispasr_chat_open(model, &op, &err);
    REQUIRE(s != nullptr);

    crispasr_chat_generate_params gp = greedy_params(16);

    char* baseline = crispasr_chat_generate(s, short_messages(), kShortMessages, &gp, &err);
    REQUIRE(baseline != nullptr);
    const std::string without_callback = baseline;
    crispasr_chat_string_free(baseline);
    REQUIRE(crispasr_chat_reset(s, &err) == 0);

    recorder r;
    r.allow_calls = 0; // abort on the first consultation
    crispasr_chat_set_abort_callback(s, abort_hook, &r);
    err = crispasr_chat_error{};
    char* aborted = crispasr_chat_generate(s, short_messages(), kShortMessages, &gp, &err);
    REQUIRE(aborted == nullptr); // the one-shot path reports an abort as NULL
    REQUIRE(err.code == CRISPASR_CHAT_ERR_ABORTED);

    crispasr_chat_set_abort_callback(s, nullptr, nullptr);
    REQUIRE(crispasr_chat_reset(s, &err) == 0);
    err = crispasr_chat_error{};
    char* cleared = crispasr_chat_generate(s, short_messages(), kShortMessages, &gp, &err);
    REQUIRE(cleared != nullptr);
    REQUIRE(err.code == 0);
    const std::string after_clear = cleared;
    crispasr_chat_string_free(cleared);
    REQUIRE(after_clear == without_callback);

    crispasr_chat_close(s);
}

TEST_CASE("crispasr_chat an abort requested up front delivers no piece", "[chat][abort]") {
    const char* model = test_model_path();
    if (!model) {
        SKIP("CRISPASR_CHAT_TEST_MODEL not set; skipping abort-before-call");
    }

    const crispasr_chat_open_params op = short_prompt_open_params();
    crispasr_chat_error err{};
    crispasr_chat_session_t s = crispasr_chat_open(model, &op, &err);
    REQUIRE(s != nullptr);

    crispasr_chat_generate_params gp = greedy_params(16);

    recorder r;
    r.allow_calls = 0; // already aborting when the call starts
    crispasr_chat_set_abort_callback(s, abort_hook, &r);

    const int32_t rc =
        crispasr_chat_generate_stream(s, short_messages(), kShortMessages, &gp, on_token_recorder, &r, &err);
    REQUIRE(rc == CRISPASR_CHAT_ERR_ABORTED);
    REQUIRE(err.code == CRISPASR_CHAT_ERR_ABORTED);
    // The first check precedes the first prompt batch, so nothing is
    // sampled and no piece is delivered.
    REQUIRE(r.pieces == 0);
    REQUIRE(r.text.empty());
    // Exactly one consultation: the check that fired is the one ahead of
    // the first prompt batch. Were it missing, the abort would instead
    // land inside the batch — where the CPU backend consults the hook
    // again — and the count would be higher.
    REQUIRE(r.calls == 1);

    crispasr_chat_close(s);
}

TEST_CASE("crispasr_chat a session is reusable after an abort", "[chat][abort]") {
    const char* model = test_model_path();
    if (!model) {
        SKIP("CRISPASR_CHAT_TEST_MODEL not set; skipping reuse-after-abort");
    }

    const crispasr_chat_open_params op = short_prompt_open_params();
    crispasr_chat_error err{};
    crispasr_chat_session_t s = crispasr_chat_open(model, &op, &err);
    REQUIRE(s != nullptr);

    crispasr_chat_generate_params gp = greedy_params(16);

    char* first = crispasr_chat_generate(s, short_messages(), kShortMessages, &gp, &err);
    REQUIRE(first != nullptr);
    const std::string baseline = first;
    crispasr_chat_string_free(first);
    REQUIRE(crispasr_chat_reset(s, &err) == 0);

    recorder r;
    r.abort_at_pieces = 2;
    crispasr_chat_set_abort_callback(s, abort_hook, &r);
    const int32_t rc =
        crispasr_chat_generate_stream(s, short_messages(), kShortMessages, &gp, on_token_recorder, &r, &err);
    REQUIRE(rc == CRISPASR_CHAT_ERR_ABORTED);

    // No reset: the abort itself must leave the session coherent, so the
    // same prompt has to prefill from scratch and reproduce the baseline.
    // A session still holding the aborted turn's history would tokenise
    // the next prompt as a continuation instead.
    crispasr_chat_set_abort_callback(s, nullptr, nullptr);
    err = crispasr_chat_error{};
    char* resumed = crispasr_chat_generate(s, short_messages(), kShortMessages, &gp, &err);
    REQUIRE(resumed != nullptr);
    REQUIRE(err.code == 0);
    const std::string after_abort = resumed;
    crispasr_chat_string_free(resumed);
    REQUIRE(after_abort == baseline);

    REQUIRE(crispasr_chat_reset(s, &err) == 0);
    char* out = crispasr_chat_generate(s, short_messages(), kShortMessages, &gp, &err);
    REQUIRE(out != nullptr);
    REQUIRE(err.code == 0);
    REQUIRE(std::strlen(out) > 0);
    crispasr_chat_string_free(out);

    crispasr_chat_close(s);
}

TEST_CASE("crispasr_chat setting an abort callback on a NULL session is a no-op", "[chat][abort]") {
    const char* model = test_model_path();
    if (!model) {
        // No model is needed here, but the executable's exit code is the
        // gate ctest reads: skipping every case keeps a model-less machine
        // reporting SKIPPED rather than a partial pass.
        SKIP("CRISPASR_CHAT_TEST_MODEL not set; skipping NULL-session no-op");
    }

    recorder r;
    crispasr_chat_set_abort_callback(nullptr, abort_hook, &r);
    crispasr_chat_set_abort_callback(nullptr, nullptr, nullptr);
    REQUIRE(r.calls == 0);
}

TEST_CASE("crispasr_chat an abort during prefill skips the remaining prompt batches", "[chat][abort]") {
    const char* model = test_model_path();
    if (!model) {
        SKIP("CRISPASR_CHAT_TEST_MODEL not set; skipping prefill abort");
    }

    crispasr_chat_open_params op;
    crispasr_chat_open_params_default(&op);
    op.n_gpu_layers = -1;
    op.n_ctx = 2048;
    op.n_batch = 512; // the prompt below needs three of these

    crispasr_chat_error err{};
    crispasr_chat_session_t s = crispasr_chat_open(model, &op, &err);
    REQUIRE(s != nullptr);

    const std::string user = long_user_message();
    REQUIRE(word_count(user) >= (size_t)kLongPromptWords);
    const crispasr_chat_message messages[] = {
        {"system", "You are a terse assistant. Answer in one word."},
        {"user", user.c_str()},
    };

    // How many prompt batches this prompt costs, and therefore how many times
    // the loop's own check between pieces has to run.
    const int32_t n_prompt = crispasr_chat_count_tokens(s, messages, 2, &err);
    REQUIRE(n_prompt > 0);
    const int pieces = (n_prompt + op.n_batch - 1) / op.n_batch;
    REQUIRE(pieces == 3);

    crispasr_chat_generate_params gp = greedy_params(8);

    // The loop's own check between pieces has to have an oracle of its own,
    // because a run that never reaches it still aborts: the hook is also
    // registered with llama, and the CPU backend — which sits in the scheduler
    // even when every layer is on the GPU — reaches it from inside the graph.
    // Counting consultations alone cannot tell the two apart.
    //
    // What separates them is how the count SCALES. Prefilling the same prompt
    // in one piece costs one seam check plus one graph's worth of in-graph
    // ones; in three pieces it costs three of each, because a graph's node
    // count is a property of the model and not of how many tokens are in the
    // batch. So the three-piece count is exactly three times the one-piece
    // count — and if the seam check ran only ahead of the first piece it would
    // fall short by exactly pieces - 1, whatever the backend contributes.
    const int one_piece = prefill_consultations(model, /*n_batch=*/2048, messages, 2);
    const int three_pieces = prefill_consultations(model, /*n_batch=*/512, messages, 2);
    REQUIRE(one_piece > 0);
    REQUIRE(three_pieces == pieces * one_piece);

    recorder full;
    crispasr_chat_set_abort_callback(s, abort_hook, &full);
    err = crispasr_chat_error{};
    const int32_t rc_full = crispasr_chat_generate_stream(s, messages, 2, &gp, on_token_recorder, &full, &err);
    REQUIRE(rc_full == 0);
    REQUIRE(full.pieces > 0);

    REQUIRE(crispasr_chat_reset(s, &err) == 0);

    // Abort on the second consultation: the first prompt batch is decoded,
    // the check between it and the next one stops the call.
    recorder early;
    early.allow_calls = 1;
    crispasr_chat_set_abort_callback(s, abort_hook, &early);
    err = crispasr_chat_error{};
    const int32_t rc_early = crispasr_chat_generate_stream(s, messages, 2, &gp, on_token_recorder, &early, &err);
    REQUIRE(rc_early == CRISPASR_CHAT_ERR_ABORTED);
    REQUIRE(err.code == CRISPASR_CHAT_ERR_ABORTED);
    // Consulted more than once, so the check that fired sits between two
    // prompt batches rather than ahead of the first.
    REQUIRE(early.calls >= 2);
    // Fewer consultations than the unaborted run, which is the same thing
    // as fewer prompt batches decoded.
    REQUIRE(early.calls < full.calls);
    // Generation never started.
    REQUIRE(early.pieces == 0);
    REQUIRE(early.text.empty());

    crispasr_chat_close(s);
}
