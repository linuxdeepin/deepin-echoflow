// crispasr_consent_record.h — what a `[CONSENT]` line has to carry to be worth
// anything, and the one place all four surfaces build it.
//
// The consent GATE (crispasr_voice_clone_policy.h) decides whether cloning is
// allowed. This is the RECORD it writes afterwards, and until now that record
// named the voice — `voice=alice.wav` — without ever saying which bytes those
// were. A name is not evidence: the file can be swapped a minute later and the
// line still reads true. That is also why hash-CHAINING the log would have been
// the wrong first move (see PLAN): a chain protects the sequence of records,
// so chaining unbound assertions yields a perfectly verifiable log that proves
// nothing. Bind each record to the audio first.
//
// WHAT THIS DOES AND DOES NOT CLAIM
// ---------------------------------
// This is TAMPER-EVIDENCE only in the weakest sense: it makes a record specific
// enough to be checked against the audio it authorised. It is NOT
// tamper-proofing, and it deliberately does not try to be. The attestation is
// made BY the operator, who controls this process, its output and any file it
// writes — no in-process mechanism defends against the party it is recording.
// Real tamper-resistance is a STORAGE decision the operator makes: append-only
// permissions, object-lock/WORM, or shipping the sink off-box to something they
// cannot rewrite. `--consent-log` exists to make that routing possible; it does
// not provide it.
//
// DATA PROTECTION
// ---------------
// Deliberately minimal. A SHA-256 of the reference is recorded; the recording
// itself, and any speaker name, are not. Every field stored is a field the
// operator must be able to erase on request, and a hash is far easier to
// justify retaining than the audio it was taken from. The operator is the
// controller here — CrispASR is a tool they run (docs/eu-ai-act.md 6.x).
//
// Header-only and IO-light so the field construction is unit-testable without a
// model, a voice file, or a server, like the other compliance headers.

#pragma once

#include "core/crispasr_sha256.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <mutex>
#include <string>
#include <vector>

#ifndef _WIN32
#include <unistd.h> // getpid
#endif

namespace crispasr_consent {

// ── Reference binding ──────────────────────────────────────────────────────

// Lowercase-hex SHA-256 of a file, streamed so a long reference recording does
// not have to be resident. Returns "" for anything it cannot read — a bank
// entry that names no file, a bare preset name, a missing path. An empty result
// is reported as `ref_sha256=none`, which is an honest "not applicable", never
// a zero hash that would look like a real one.
inline std::string file_sha256(const std::string& path) {
    if (path.empty())
        return {};
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f)
        return {};
    crispasr::sha::Sha256 h;
    h.init();
    std::vector<uint8_t> buf(64 * 1024);
    size_t n;
    while ((n = std::fread(buf.data(), 1, buf.size(), f)) > 0)
        h.update(buf.data(), n);
    std::fclose(f);
    uint8_t out[32];
    h.final(out);
    static const char* kHex = "0123456789abcdef";
    std::string hex;
    hex.reserve(64);
    for (unsigned char c : out) {
        hex += kHex[c >> 4];
        hex += kHex[c & 0xF];
    }
    return hex;
}

// Same, for a buffer already in memory — the voice-upload path holds the
// uploaded bytes and should hash THOSE, not re-read whatever landed on disk.
inline std::string bytes_sha256(const void* data, size_t len) {
    if (!data || len == 0)
        return {};
    auto d = crispasr::sha::sha256((const uint8_t*)data, len);
    static const char* kHex = "0123456789abcdef";
    std::string hex;
    hex.reserve(64);
    for (unsigned char c : d) {
        hex += kHex[c >> 4];
        hex += kHex[c & 0xF];
    }
    return hex;
}

// ── Correlation ────────────────────────────────────────────────────────────

// Per-process id, so a disputed clip can be walked back to the attestation that
// authorised it: the same value appears in the `[CONSENT]` line and in the
// post-synthesis audit line. Derived from the process start, the pid and the
// address of a local, hashed — enough to distinguish concurrent runs, and
// explicitly NOT a security token.
inline const std::string& run_id() {
    static const std::string id = [] {
        uint64_t seed = (uint64_t)std::time(nullptr);
        int local = 0;
        seed ^= (uint64_t)(uintptr_t)&local * 0x9E3779B97F4A7C15ULL;
#ifndef _WIN32
        seed ^= (uint64_t)::getpid() << 32;
#endif
        uint8_t bytes[8];
        for (int i = 0; i < 8; i++)
            bytes[i] = (uint8_t)(seed >> (8 * i));
        auto d = crispasr::sha::sha256(bytes, sizeof(bytes));
        static const char* kHex = "0123456789abcdef";
        std::string hex;
        for (int i = 0; i < 8; i++) { // 64 bits is plenty to correlate a run
            hex += kHex[d[i] >> 4];
            hex += kHex[d[i] & 0xF];
        }
        return hex;
    }();
    return id;
}

// Per-REQUEST correlation id, for surfaces that serve many requests from one
// process. `run_id` alone cannot distinguish them, so on a busy server a
// consent line and the response it authorised are otherwise unlinkable.
// thread_local because httplib serves each request on one thread; call
// new_request_id() at request entry and the same value is available to every
// record emitted while handling it. Empty on surfaces that never set it, and
// omitted from the record in that case.
inline std::string& request_id_storage() {
    static thread_local std::string id;
    return id;
}

inline const std::string& new_request_id() {
    static std::atomic<uint64_t> counter{0};
    const uint64_t n = counter.fetch_add(1, std::memory_order_relaxed);
    uint8_t bytes[16];
    const uint64_t t = (uint64_t)std::time(nullptr);
    for (int i = 0; i < 8; i++) {
        bytes[i] = (uint8_t)(t >> (8 * i));
        bytes[8 + i] = (uint8_t)(n >> (8 * i));
    }
    auto d = crispasr::sha::sha256(bytes, sizeof(bytes));
    static const char* kHex = "0123456789abcdef";
    std::string hex;
    for (int i = 0; i < 8; i++) {
        hex += kHex[d[i] >> 4];
        hex += kHex[d[i] & 0xF];
    }
    request_id_storage() = hex;
    return request_id_storage();
}

inline const std::string& request_correlation_id() {
    return request_id_storage();
}

// ── The record ─────────────────────────────────────────────────────────────

struct Field {
    std::string key;
    std::string value;
    bool quoted = false; // free text (attestation) is quoted on the stderr line
};

// JSON string escape. The values reaching here are already control-char
// screened by the callers' sanitizers (a newline in a voice name forges an
// audit record), but the sink is machine-read, so escape rather than trust.
inline std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
        case '"':
            o += "\\\"";
            break;
        case '\\':
            o += "\\\\";
            break;
        case '\n':
            o += "\\n";
            break;
        case '\r':
            o += "\\r";
            break;
        case '\t':
            o += "\\t";
            break;
        default:
            if ((unsigned char)c < 0x20) {
                static const char* kHex = "0123456789abcdef";
                o += "\\u00";
                o += kHex[(unsigned char)c >> 4];
                o += kHex[(unsigned char)c & 0xF];
            } else {
                o += c;
            }
        }
    }
    return o;
}

// Where `--consent-log` / CRISPASR_CONSENT_LOG routes the machine-readable
// copy. Empty = stderr only, which stays the default: turning on a persistent
// record of who attested what is the operator's decision, not ours.
inline std::string& log_path() {
    static std::string p;
    return p;
}

inline void set_log_path(const std::string& p) {
    log_path() = p;
}

inline void init_log_path_from_env() {
    if (!log_path().empty())
        return;
    const char* e = std::getenv("CRISPASR_CONSENT_LOG");
    if (e && *e)
        log_path() = e;
}

// Emit one record.
//
// The stderr line keeps its historical shape and gains its new fields at the
// END, so anything grepping `key=value` (tests, operators' log rules) keeps
// working. The JSON Lines copy goes to the sink when one is configured — that
// is the form worth feeding to a SIEM, and the reason a separate sink exists at
// all: stderr here is interleaved with model-load noise and progress output,
// which makes it a poor evidential artefact however carefully it is written.
inline void emit(const char* kind, const std::string& ts, const std::vector<Field>& fields) {
    std::string line = std::string("[") + kind + "] ts=" + ts;
    for (const auto& f : fields) {
        // An empty UNQUOTED value means "not applicable on this surface" (the
        // per-request id off the server, say) — omit it rather than emit a bare
        // `key=`, which a log rule would have to special-case. A quoted field
        // stays: `attestation=""` is a real state (Wyoming has no per-request
        // attestation) and must not silently vanish.
        if (f.value.empty() && !f.quoted)
            continue;
        line += " " + f.key + "=";
        line += f.quoted ? ("\"" + f.value + "\"") : f.value;
    }
    line += " run_id=" + run_id();
    std::fprintf(stderr, "%s\n", line.c_str());

    const std::string& path = log_path();
    if (path.empty())
        return;
    // Serialised: the server emits these from request threads, and an
    // interleaved half-line is not a record.
    static std::mutex mtx;
    std::lock_guard<std::mutex> lk(mtx);
    FILE* f = std::fopen(path.c_str(), "ab");
    if (!f)
        return; // never fail synthesis over the audit sink; stderr already has it
    std::string js = "{\"kind\":\"" + json_escape(kind) + "\",\"ts\":\"" + json_escape(ts) + "\"";
    for (const auto& fl : fields) {
        if (fl.value.empty() && !fl.quoted)
            continue;
        js += ",\"" + json_escape(fl.key) + "\":\"" + json_escape(fl.value) + "\"";
    }
    js += ",\"run_id\":\"" + run_id() + "\"}";
    std::fprintf(f, "%s\n", js.c_str());
    std::fclose(f);
}

} // namespace crispasr_consent
