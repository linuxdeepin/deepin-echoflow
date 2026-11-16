// test-chat-close.cpp — teardown of a crispasr_chat_* session under load.
//
// Gated on CRISPASR_CHAT_TEST_MODEL — a path to a small GGUF chat model
// (e.g. gemma-3-1b-it-Q4_K_M.gguf, qwen2.5-0.5b-instruct, smollm2-360m).
// When unset every model-backed case is reported as SKIPPED so unrelated
// builds stay green without a model on disk.
//
// `crispasr_chat_close` frees the context, the model and the session itself,
// and a generation holds the session for as long as it decodes. Covers what
// the header promises about closing from another thread:
//   • close waits for a generation already in flight rather than freeing
//     underneath it, and that generation still returns its full output
//   • a call arriving after the retirement is declined with a non-zero code
//     rather than run against memory about to be freed
//   • the accessors that answer a NULL session with "nothing here" answer a
//     retiring session the same way
//   • close on NULL is a no-op
//
// What is NOT covered, because it is not coverable from here: a call
// descheduled after reading its handle and before it reaches the session's
// lifetime lock. Every rejected call below reaches that lock safely only
// because a parked generation is holding the allocation alive for the whole
// case. With no such call in flight, a close frees while that thread is
// suspended and it wakes on a destroyed mutex. crispasr_chat.h states the
// ordering rule that makes this the caller's to prevent; nothing inside the
// session can test for it, because the test would live in the freed memory.
//
// Catch2 assertion macros are not thread-safe, so what the worker threads
// observe travels back in atomics and is asserted on the main thread after
// the joins.

#include <catch2/catch_test_macros.hpp>

#include "crispasr_chat.h"

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>

namespace {

const char* test_model_path() {
    return std::getenv("CRISPASR_CHAT_TEST_MODEL");
}

// A generation parked inside its own token callback.
//
// The callback signals the test that the session is occupied and then blocks
// until the test releases it, which is what makes the overlap deterministic:
// there is no sleep anywhere below, and no window in which the generation
// might have finished before the close was issued. Every piece consults
// `release`, so once it is set the remaining pieces stream straight through.
struct parked_generation {
    std::mutex mu;
    std::condition_variable cv;
    bool inside = false;  // the callback has been reached
    bool release = false; // the test is done and the callback may return

    std::string text;
    std::atomic<bool> generate_returned{false};
    std::atomic<int32_t> rc{-1};
    std::atomic<int32_t> err_code{-1};
};

void on_token_park(const char* chunk, void* user) {
    auto* p = static_cast<parked_generation*>(user);
    std::unique_lock<std::mutex> lk(p->mu);
    p->text.append(chunk);
    if (!p->inside) {
        p->inside = true;
        p->cv.notify_all();
    }
    p->cv.wait(lk, [p] { return p->release; });
}

crispasr_chat_session_t open_chat(const char* model) {
    crispasr_chat_open_params op;
    crispasr_chat_open_params_default(&op);
    op.n_gpu_layers = -1;
    op.n_ctx = 1024;

    crispasr_chat_error err{};
    crispasr_chat_session_t s = crispasr_chat_open(model, &op, &err);
    REQUIRE(s != nullptr);
    REQUIRE(err.code == 0);
    return s;
}

crispasr_chat_generate_params greedy_params(int32_t max_tokens) {
    crispasr_chat_generate_params gp;
    crispasr_chat_generate_params_default(&gp);
    gp.max_tokens = max_tokens;
    gp.temperature = 0.0f;
    gp.seed = 1;
    return gp;
}

// Run one parked streaming generation on `s`, recording its outcome.
void run_parked(crispasr_chat_session_t s, const crispasr_chat_message* msgs,
                const crispasr_chat_generate_params* gp, parked_generation* p) {
    crispasr_chat_error err{};
    const int32_t rc = crispasr_chat_generate_stream(s, msgs, 1, gp, on_token_park, p, &err);
    p->rc.store(rc, std::memory_order_relaxed);
    p->err_code.store(err.code, std::memory_order_relaxed);
    p->generate_returned.store(true, std::memory_order_release);
}

void await_inside(parked_generation& p) {
    std::unique_lock<std::mutex> lk(p.mu);
    p.cv.wait(lk, [&] { return p.inside; });
}

void release(parked_generation& p) {
    {
        std::lock_guard<std::mutex> lk(p.mu);
        p.release = true;
    }
    p.cv.notify_all();
}

// Spin until the session declines a hold, which is the observable edge of
// `crispasr_chat_close` having retired the handle. `_n_ctx` is the cheapest
// probe: it takes no session lock, so it cannot block behind the parked
// generation, and it reports a session it may not enter as 0 — the same
// answer it gives for NULL.
//
// Safe to spin on: the close doing the retiring is itself waiting for the
// parked generation, which only this test releases, so the session is alive
// for the whole loop.
void await_retirement(crispasr_chat_session_t s) {
    while (crispasr_chat_n_ctx(s) != 0) {
        std::this_thread::yield();
    }
}

} // namespace

TEST_CASE("crispasr_chat_close on a null session is a no-op", "[chat][unit]") {
    crispasr_chat_close(nullptr);
    SUCCEED("returned without dereferencing");
}

TEST_CASE("crispasr_chat_close waits for a generation in flight", "[chat][gguf]") {
    const char* model = test_model_path();
    if (!model) {
        SKIP("CRISPASR_CHAT_TEST_MODEL not set; skipping close-under-load");
    }

    crispasr_chat_session_t s = open_chat(model);
    REQUIRE(crispasr_chat_n_ctx(s) > 0);
    const crispasr_chat_message msgs[] = {{"user", "Count from 1 to 12, one number per line."}};
    const crispasr_chat_generate_params gp = greedy_params(48);

    parked_generation p;
    std::thread worker([&] { run_parked(s, msgs, &gp, &p); });
    await_inside(p);
    REQUIRE_FALSE(p.generate_returned.load(std::memory_order_acquire));

    std::atomic<bool> close_returned{false};
    std::thread closer([&] {
        crispasr_chat_close(s);
        close_returned.store(true, std::memory_order_release);
    });

    // The close cannot have completed: the generation still holds the session
    // and nothing has released it.
    await_retirement(s);
    REQUIRE_FALSE(close_returned.load(std::memory_order_acquire));
    REQUIRE_FALSE(p.generate_returned.load(std::memory_order_acquire));

    release(p);
    worker.join();
    closer.join();

    // Deliberately NOT asserted here: that the generation returned before the
    // close did. It is not the contract and it would flake. `in_use` is
    // dropped by session_hold's destructor, which runs as the C function
    // unwinds — before that function returns to run_parked, and long before
    // run_parked sets generate_returned. So the close is legitimately free to
    // wake, free and return first. What the contract says is that the close
    // does not free while the session is IN USE, and the discriminating
    // evidence for that is above: with the generation parked inside, the
    // close had not returned.
    //
    // The generation was not disturbed by the close overlapping it: it
    // succeeded and produced its text, rather than being cut short.
    REQUIRE(p.rc.load(std::memory_order_relaxed) == 0);
    REQUIRE(p.err_code.load(std::memory_order_relaxed) == 0);
    REQUIRE_FALSE(p.text.empty());
}

TEST_CASE("a call arriving during a close is declined, not run", "[chat][gguf]") {
    const char* model = test_model_path();
    if (!model) {
        SKIP("CRISPASR_CHAT_TEST_MODEL not set; skipping decline-during-close");
    }

    crispasr_chat_session_t s = open_chat(model);
    const crispasr_chat_message msgs[] = {{"user", "Count from 1 to 12, one number per line."}};
    const crispasr_chat_generate_params gp = greedy_params(48);

    parked_generation p;
    std::thread worker([&] { run_parked(s, msgs, &gp, &p); });
    await_inside(p);

    std::thread closer([&] { crispasr_chat_close(s); });
    await_retirement(s);

    // Every entry point that would touch the session now refuses. Without the
    // refusal these would take the session mutex and block until the parked
    // generation released it — and then run against a session the close is
    // about to free.
    {
        crispasr_chat_error err{};
        REQUIRE(crispasr_chat_count_tokens(s, msgs, 1, &err) < 0);
        REQUIRE(err.code != 0);
    }
    {
        crispasr_chat_error err{};
        REQUIRE(crispasr_chat_reset(s, &err) != 0);
        REQUIRE(err.code != 0);
    }
    {
        crispasr_chat_error err{};
        REQUIRE(crispasr_chat_generate(s, msgs, 1, &gp, &err) == nullptr);
        REQUIRE(err.code != 0);
        // Not a cancellation: nothing was generated and nothing was aborted.
        REQUIRE(err.code != CRISPASR_CHAT_ERR_ABORTED);
    }
    {
        crispasr_chat_error err{};
        REQUIRE(crispasr_chat_generate_stream(s, msgs, 1, &gp, nullptr, nullptr, &err) != 0);
        REQUIRE(err.code != 0);
    }

    // The value-returning accessors answer a retiring session the way they
    // answer a NULL one.
    REQUIRE(crispasr_chat_n_ctx(s) == 0);
    REQUIRE(crispasr_chat_template_name(s) == nullptr);

    release(p);
    worker.join();
    closer.join();
    REQUIRE(p.generate_returned.load(std::memory_order_acquire));
}
