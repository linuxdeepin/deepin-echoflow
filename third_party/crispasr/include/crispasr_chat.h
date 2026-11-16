// crispasr_chat.h — public C ABI for text → text chat / LLM inference.
//
// Sits next to crispasr.h (the ASR + audio surface). Implemented in
// src/chat.cpp on top of the private `crispasr-llama-core` static lib
// (vendored llama.cpp). Downstream consumers see ONLY POD structs and
// opaque handles — no llama.h types leak into this header.
//
// Threading
// ---------
//   One call at a time per `crispasr_chat_session_t` — the session
//   serialises its own context internally with a mutex. The intended
//   server pattern is one session per worker thread; multiple sessions
//   over one process are fully supported.
//
//   `crispasr_chat_close` is the exception, and the one call meant to be
//   made from another thread: it waits for every call already inside the
//   session to return before it frees anything. See its own note below for
//   what that does and does not promise.
//
// Memory
// ------
//   The KV cache persists across `crispasr_chat_generate` calls inside
//   one session, so multi-turn chats don't re-prefill the full history.
//   Use `crispasr_chat_reset` to flush.
//
//   Reuse depends on the caller passing the WHOLE conversation in
//   `messages` on every call, not just the new turn: the session compares
//   the templated prompt against the tokens it already holds and decodes
//   only what is new. Passing just the latest turn is not wrong — it
//   simply shares no prefix, so every call re-prefills from scratch.
//
// Strings out of this ABI come from malloc — free with
// `crispasr_chat_string_free`.
//
// EU AI Act — read this before shipping a product on top of it
// ------------------------------------------------------------
// Everything else in CrispASR either records what a human said (ASR) or
// generates AUDIO, which the runtime marks for you: every synthesis path
// watermarks by default and cloned voices get an audible AI disclosure. This
// header is the exception. It is open-ended synthetic TEXT generation, and
// there are two duties on it that CrispASR does NOT discharge:
//
//   Art. 50(2) — synthetic text must be marked as artificially generated, in a
//     machine-readable form. There is no watermark-equivalent for short-form
//     text that survives a copy-paste, so nothing here marks the output. The
//     practical option is metadata travelling with the response (the server
//     sends X-Crispasr-Ai-Generated on /v1/chat/completions; do the same).
//
//   Art. 50(1) — a system that interacts directly with natural persons must
//     tell them they are talking to an AI, unless that is obvious to a
//     reasonably well-informed person. A terminal you launched with a `-m
//     model.gguf` flag is obvious. A chat bubble in your app is not.
//     `crispasr_chat_ai_disclosure_text()` is the canonical wording; show it
//     at or before the first turn, and make it visible, not audio-only
//     (Art. 50(5) accessibility).
//
// Neither duty transfers with the model: whichever GGUF you point this at, you
// are the deployer of the system built on it. See docs/eu-ai-act.md §6.6.

#ifndef CRISPASR_CHAT_H
#define CRISPASR_CHAT_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef CRISPASR_SHARED
#ifdef _WIN32
#ifdef CRISPASR_BUILD
#define CRISPASR_CHAT_API __declspec(dllexport)
#else
#define CRISPASR_CHAT_API __declspec(dllimport)
#endif
#else
#define CRISPASR_CHAT_API __attribute__((visibility("default")))
#endif
#else
#define CRISPASR_CHAT_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Error reporting
// ---------------------------------------------------------------------------
// Every entry point that can fail accepts a `crispasr_chat_error*` (may be
// NULL). On success the struct is left untouched. On failure `code` is
// set non-zero and `message` carries a short null-terminated diagnostic.
//
// `CRISPASR_CHAT_ERR_ABORTED` (below) is the one code with a stable,
// documented meaning — a caller has to tell a cancellation apart from a
// fault. Every other non-zero value is a diagnostic aid, not a contract:
// read `message`, don't switch on the number.
typedef struct crispasr_chat_error {
    int32_t code;
    char message[256];
} crispasr_chat_error;

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------
// `role` is one of "system", "user", "assistant", "tool" — matches the
// OpenAI chat schema; the chat-template layer will translate the role
// names into whatever the model's template expects.
typedef struct crispasr_chat_message {
    const char* role;
    const char* content;
} crispasr_chat_message;

// ---------------------------------------------------------------------------
// Open params (per-session, model-level)
// ---------------------------------------------------------------------------
// All fields have well-defined defaults via `crispasr_chat_open_params_default`.
typedef struct crispasr_chat_open_params {
    int32_t n_threads;       // generation threads      (default: physical cores)
    int32_t n_threads_batch; // batch / prefill threads (default: n_threads)
    int32_t n_ctx;           // context window in tokens — 0 = model default
    int32_t n_batch;         // logical batch size      (default: 512)
    int32_t n_ubatch;        // physical micro-batch    (default: 512)
    int32_t n_gpu_layers;    // -1 = all, 0 = CPU only  (default: -1)
    bool use_mmap;           // default: true
    bool use_mlock;          // default: false
    bool embeddings;         // future use; keep false  (default: false)
    // If non-NULL overrides the template baked into the GGUF. NULL =
    // read `tokenizer.chat_template` from the model; if absent fall back
    // to "chatml". The string is copied — caller may free immediately.
    const char* chat_template;
} crispasr_chat_open_params;

// Populate `out` with sensible defaults. Safe to call before _open.
CRISPASR_CHAT_API void crispasr_chat_open_params_default(crispasr_chat_open_params* out);

// ---------------------------------------------------------------------------
// Generate params (per-call, sampler-level)
// ---------------------------------------------------------------------------
// Phase 2 + 3 fields: top_k / top_p / repeat_penalty / seed.
typedef struct crispasr_chat_generate_params {
    int32_t max_tokens;    // hard cap on tokens generated (default: 256)
    float temperature;     // 0.0 = greedy                 (default: 0.8)
    int32_t top_k;         // 0 = disabled                 (default: 40)
    float top_p;           // 1.0 = disabled               (default: 0.95)
    float min_p;           // 0.0 = disabled               (default: 0.05)
    float repeat_penalty;  // 1.0 = disabled               (default: 1.1)
    int32_t repeat_last_n; // -1 = ctx size, 0 = disabled  (default: 64)
    uint32_t seed;         // 0xFFFFFFFF = random          (default: 0, fixed)

    // Stop sequences: NULL = none. Generation halts (output is truncated
    // BEFORE the match) the first time any of these substrings appears
    // in the accumulated decoded output.
    const char* const* stop;
    size_t n_stop;

    // If true, the system / user portion is prefilled but assistant
    // generation is suppressed — useful for measuring prompt cost.
    bool prefill_only;
} crispasr_chat_generate_params;

CRISPASR_CHAT_API void crispasr_chat_generate_params_default(crispasr_chat_generate_params* out);

// ---------------------------------------------------------------------------
// Session lifecycle
// ---------------------------------------------------------------------------
typedef struct crispasr_chat_session crispasr_chat_session;
typedef crispasr_chat_session* crispasr_chat_session_t;

// Open a session from a GGUF chat model on disk. Returns NULL on failure
// and fills `err` when non-NULL. `params` may be NULL — defaults apply.
CRISPASR_CHAT_API crispasr_chat_session_t crispasr_chat_open(const char* model_path,
                                                             const crispasr_chat_open_params* params,
                                                             crispasr_chat_error* err);

// Free the session and its KV cache. Safe to call with NULL.
//
// THE CALLER MUST ORDER THIS AGAINST EVERY OTHER CALL ON THE SAME HANDLE.
// Like `fclose` and `llama_free`, this is not a synchronisation point: it is
// undefined behaviour for a call on `s` to be entered unordered with respect
// to the close, and no amount of checking inside this library can change
// that, because the check itself lives in the memory being freed. Keep the
// handle behind your own lock, or set it aside so nothing can reach it, and
// close it after that. Every binding shipped with CrispASR does exactly this,
// and it is where the real lifetime safety comes from.
//
// What this call adds, on top of that, is the one thing the caller cannot do
// for itself: it waits for calls that are ALREADY RUNNING inside the session.
// A generation holds the session for as long as it decodes, so a caller that
// wants to shut down mid-answer would otherwise have to choose between
// blocking its own shutdown path and freeing the context out from under a
// running `llama_decode`. Instead this retires the handle, waits for every
// admitted call to stop touching session state, and only then frees.
//
// A call that reaches the session while the close is waiting is declined with
// a non-zero `code`; the two accessors that report a NULL session as
// "nothing here" (`_template_name`, `_n_ctx`) answer the same way. Treat that
// as a diagnostic for a caller that has ALREADY broken the ordering rule
// above, not as a guarantee — a call descheduled just before it reaches the
// session can be freed out from under instead, and a second close racing the
// first can end up on destroyed locks. Both are the same use-after-free the
// rule exists to prevent. Close exactly once.
//
// Cancel first if you do not want to wait: register an abort callback, ask it
// to stop, and the generation returns in a token or two.
CRISPASR_CHAT_API void crispasr_chat_close(crispasr_chat_session_t s);

// Clear the KV cache so the next _generate re-prefills from scratch. Call
// when starting a new conversation in a reused session.
CRISPASR_CHAT_API int32_t crispasr_chat_reset(crispasr_chat_session_t s, crispasr_chat_error* err);

// ---------------------------------------------------------------------------
// One-shot generate
// ---------------------------------------------------------------------------
// Applies the model's chat template to `messages`, prefills, runs
// generation to `max_tokens` or a stop sequence, and returns a freshly
// malloc'd UTF-8 string holding the assistant's reply. Free with
// `crispasr_chat_string_free`. Returns NULL on failure (sets `err`).
CRISPASR_CHAT_API char* crispasr_chat_generate(crispasr_chat_session_t s, const crispasr_chat_message* messages,
                                               size_t n_messages, const crispasr_chat_generate_params* params,
                                               crispasr_chat_error* err);

// ---------------------------------------------------------------------------
// Streaming generate
// ---------------------------------------------------------------------------
// Fires `on_token` once per detokenised UTF-8 chunk (typically one piece
// per llama_token). The chunk pointer is valid only during the callback.
// `user` is forwarded verbatim. Returns 0 on clean completion (including
// stop-sequence / EOG termination) and non-zero on failure.
typedef void (*crispasr_chat_on_token)(const char* utf8_chunk, void* user);

CRISPASR_CHAT_API int32_t crispasr_chat_generate_stream(crispasr_chat_session_t s,
                                                        const crispasr_chat_message* messages, size_t n_messages,
                                                        const crispasr_chat_generate_params* params,
                                                        crispasr_chat_on_token on_token, void* user,
                                                        crispasr_chat_error* err);

// ---------------------------------------------------------------------------
// Cancellation
// ---------------------------------------------------------------------------
// Returned by `crispasr_chat_generate_stream`, and written to `err.code` by
// both generate entry points, when a registered abort callback stopped the
// run. Distinct from every failure code so a caller cannot read a
// cancellation as a decode fault — or, on the one-shot path, a NULL return
// as an allocation failure.
#define CRISPASR_CHAT_ERR_ABORTED 40

// Return true to let the generation continue, false to abort it. Matches
// the `whisper_encoder_begin_callback` convention on the ASR surface.
// `user` is the pointer handed to `crispasr_chat_set_abort_callback`.
//
// Called on the generating thread: once before each prompt batch during
// prefill, and once before each sampled token. On the CPU backend it is
// additionally called from inside a running compute graph, so it can be
// invoked many times per batch there and must be cheap and non-blocking.
// On Metal and CUDA an in-flight batch runs to completion, so a cancel
// takes effect at the next prompt-batch or token boundary.
//
// The callback must not call back into the same session — the session
// mutex is held for the whole generation and re-entering deadlocks.
typedef bool (*crispasr_chat_abort_callback)(void* user);

// Register `cb` on the session; `user` is forwarded verbatim on every
// call. Passing cb = NULL clears the callback. A NULL session is a no-op.
//
// Register before starting a generation and have the callback read your
// own flag: this call takes the session lock, so calling it while a
// generation is running blocks until that generation finishes rather than
// cancelling it.
//
// On abort `crispasr_chat_generate` returns NULL and
// `crispasr_chat_generate_stream` returns `CRISPASR_CHAT_ERR_ABORTED`,
// having already delivered the partial text through `on_token`; both set
// `err.code` to `CRISPASR_CHAT_ERR_ABORTED`. An abort flushes the session
// back to its just-opened state — KV cache and history cleared — so the
// next `_generate` prefills from scratch whatever messages it is given
// and no `crispasr_chat_reset` is needed first.
CRISPASR_CHAT_API void crispasr_chat_set_abort_callback(crispasr_chat_session_t s, crispasr_chat_abort_callback cb,
                                                        void* user);

// ---------------------------------------------------------------------------
// Memory + introspection
// ---------------------------------------------------------------------------
// Returns the name of the chat template the session resolved against
// (e.g. "chatml", "llama3", "gemma"). Pointer is owned by the session
// and stays valid until _close.
CRISPASR_CHAT_API const char* crispasr_chat_template_name(crispasr_chat_session_t s);

// Returns the context window in tokens.
CRISPASR_CHAT_API int32_t crispasr_chat_n_ctx(crispasr_chat_session_t s);

// Tokens the model's own tokenizer produces for `messages` once the
// session's chat template has been applied — the prompt length a FRESH
// session prefills for the same `messages`, so it can be compared
// straight against `crispasr_chat_n_ctx` when sizing a context window.
//
// The number covers the whole prompt: the template's control tokens, the
// leading BOS the tokenizer adds to a new sequence, and the trailing
// generation prompt that opens the assistant turn — everything
// `crispasr_chat_generate` decodes before it samples its first token, and
// nothing it generates afterwards. An empty `messages` array therefore
// still counts the template's own opening. A session part-way through a
// conversation re-decodes only the suffix its history does not already
// hold, so there the count is an upper bound.
//
// Pure query: it neither touches the KV cache nor extends the history, so
// it can be called freely between generations. It does take the session
// lock, so it waits for a generation in flight to finish.
//
// Returns a negative value on failure and fills `err`. Mirrors
// `whisper_token_count` on the ASR surface.
CRISPASR_CHAT_API int32_t crispasr_chat_count_tokens(crispasr_chat_session_t s, const crispasr_chat_message* messages,
                                                     size_t n_messages, crispasr_chat_error* err);

// Pre-flight memory estimate for a GGUF chat model on disk. Returns the
// approximate working-set in bytes (weights + KV cache + activations) or
// 0 if it could not be estimated. Mirrors the shape of crispasr's
// existing `crispasr_memory_estimate_*` family; lets CrisperWeaver's
// pre-flight guard short-circuit on low-RAM devices.
CRISPASR_CHAT_API size_t crispasr_chat_memory_estimate(const char* model_path, const crispasr_chat_open_params* params,
                                                       crispasr_chat_error* err);

// Free a malloc'd string returned by `crispasr_chat_generate`.
CRISPASR_CHAT_API void crispasr_chat_string_free(char* s);

// ---------------------------------------------------------------------------
// AI disclosure (EU AI Act Art. 50(1))
// ---------------------------------------------------------------------------
// The canonical wording for "you are talking to an AI", the text counterpart of
// crispasr_session_disclaimer_text() on the audio side. Show it at or before
// the first turn of any conversational product built on this ABI, and show it
// VISIBLY — Art. 50(5) requires disclosures to meet accessibility requirements.
//
// This is a string, not a gate: the ABI cannot know whether your product faces
// a natural person or is a batch summarizer where no disclosure is owed. What
// it can do is stop every downstream integrator inventing their own wording,
// and stop the duty going unnoticed because it lives only in a doc.
//
// Returns a static string; never NULL, never needs freeing.
CRISPASR_CHAT_API const char* crispasr_chat_ai_disclosure_text(void);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CRISPASR_CHAT_H
