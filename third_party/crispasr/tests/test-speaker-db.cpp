// test-speaker-db.cpp — unit tests for the named-voiceprint profile DB
// (src/speaker_db.{h,cpp}). Drives enroll / load / count / match with
// synthetic L2-normalized embeddings, so the suite is deterministic and
// needs no model load, no audio, and no network.
//
// This DB underpins the deliberately opt-in --speaker-db biometric path
// (1:N identification). The session-scoped *clustering* path that produces
// anonymous (speaker N) labels is covered separately by
// test-speaker-cluster.cpp.

#include "../src/speaker_db.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#include <process.h>
#include <windows.h>
static std::string make_temp_dir() {
    char buf[MAX_PATH];
    GetTempPathA(MAX_PATH, buf);
    std::string base = buf;
    if (!base.empty() && (base.back() == '\\' || base.back() == '/'))
        base.pop_back();
    std::string dir = base + "/crispasr_spkdb_" + std::to_string(_getpid());
    _mkdir(dir.c_str());
    return dir;
}
#else
#include <sys/stat.h>
#include <unistd.h>
static std::string make_temp_dir() {
    const char* env = std::getenv("CRISPASR_SCRATCH_DIR");
    std::string base = (env && *env) ? env : ".scratch";
    mkdir(base.c_str(), 0755);
    std::string pattern = base + "/crispasr_spkdb_XXXXXX";
    std::string writable = pattern;
    char* buf = writable.data();
    return mkdtemp(buf) ? std::string(buf) : base;
}
#endif

namespace {

// L2-normalize a copy so cosine == dot, matching how real embedders feed
// the DB (speaker_db_match assumes normalized inputs).
std::vector<float> norm(std::vector<float> v) {
    double s = 0.0;
    for (float x : v)
        s += (double)x * x;
    if (s > 0.0) {
        const float inv = (float)(1.0 / std::sqrt(s));
        for (float& x : v)
            x *= inv;
    }
    return v;
}

constexpr int D = 8;

} // namespace

TEST_CASE("speaker_db: empty / missing directory yields zero speakers and no match", "[unit]") {
    const std::string dir = make_temp_dir() + "/does_not_exist_yet";
    speaker_db* db = speaker_db_load(dir.c_str());
    REQUIRE(db != nullptr); // missing dir is valid (0 speakers), not an error
    REQUIRE(speaker_db_count(db) == 0);

    const std::vector<float> q = norm({1, 0, 0, 0, 0, 0, 0, 0});
    float score = 123.0f;
    REQUIRE(speaker_db_match(db, q.data(), D, 0.5f, &score) == nullptr);
    speaker_db_free(db);
}

TEST_CASE("speaker_db: enroll then match returns the nearest enrolled name", "[unit]") {
    const std::string dir = make_temp_dir();
    const std::vector<float> alice = norm({1, 0, 0, 0, 0, 0, 0, 0});
    const std::vector<float> bob = norm({0, 1, 0, 0, 0, 0, 0, 0});

    REQUIRE(speaker_db_enroll(dir.c_str(), "alice", alice.data(), D, /*consent=*/true));
    REQUIRE(speaker_db_enroll(dir.c_str(), "bob", bob.data(), D, /*consent=*/true));

    speaker_db* db = speaker_db_load(dir.c_str());
    REQUIRE(db != nullptr);
    REQUIRE(speaker_db_count(db) == 2);

    // A query close to alice (cosine ~0.995) should resolve to "alice".
    const std::vector<float> near_alice = norm({0.9f, 0.1f, 0, 0, 0, 0, 0, 0});
    float score = 0.0f;
    const char* name = speaker_db_match(db, near_alice.data(), D, 0.7f, &score);
    REQUIRE(name != nullptr);
    REQUIRE(std::string(name) == "alice");
    REQUIRE(score > 0.9f);

    // ...and a query close to bob resolves to "bob".
    const std::vector<float> near_bob = norm({0.1f, 0.9f, 0, 0, 0, 0, 0, 0});
    name = speaker_db_match(db, near_bob.data(), D, 0.7f, &score);
    REQUIRE(name != nullptr);
    REQUIRE(std::string(name) == "bob");
    speaker_db_free(db);
}

TEST_CASE("speaker_db: a dissimilar voice is rejected (below threshold => no name)", "[unit]") {
    const std::string dir = make_temp_dir();
    REQUIRE(speaker_db_enroll(dir.c_str(), "alice", norm({1, 0, 0, 0, 0, 0, 0, 0}).data(), D, /*consent=*/true));

    speaker_db* db = speaker_db_load(dir.c_str());
    REQUIRE(db != nullptr);

    // Orthogonal to alice => cosine ~0 => below any sane threshold.
    const std::vector<float> stranger = norm({0, 0, 1, 0, 0, 0, 0, 0});
    float score = 1.0f;
    REQUIRE(speaker_db_match(db, stranger.data(), D, 0.5f, &score) == nullptr);
    REQUIRE(score < 0.5f); // out_score still reports the best (rejected) similarity
    speaker_db_free(db);
}

TEST_CASE("speaker_db: threshold gates an otherwise-close match", "[unit]") {
    const std::string dir = make_temp_dir();
    REQUIRE(speaker_db_enroll(dir.c_str(), "alice", norm({1, 0, 0, 0, 0, 0, 0, 0}).data(), D, /*consent=*/true));
    speaker_db* db = speaker_db_load(dir.c_str());

    // cosine ~0.707 with alice: matches at 0.7 but is rejected at 0.8.
    const std::vector<float> q = norm({1, 1, 0, 0, 0, 0, 0, 0});
    REQUIRE(speaker_db_match(db, q.data(), D, 0.7f, nullptr) != nullptr);
    REQUIRE(speaker_db_match(db, q.data(), D, 0.8f, nullptr) == nullptr);
    speaker_db_free(db);
}

TEST_CASE("speaker_db: dimension mismatch never matches", "[unit]") {
    const std::string dir = make_temp_dir();
    REQUIRE(speaker_db_enroll(dir.c_str(), "alice", norm({1, 0, 0, 0, 0, 0, 0, 0}).data(), D, /*consent=*/true));
    speaker_db* db = speaker_db_load(dir.c_str());

    // Querying with a different dimensionality (e.g. wrong embedder) must
    // not produce a false identification.
    const std::vector<float> q4 = norm({1, 0, 0, 0});
    REQUIRE(speaker_db_match(db, q4.data(), 4, 0.0f, nullptr) == nullptr);
    speaker_db_free(db);
}

// ── Issue #266: consent gate + closed-roster narrowing + legacy format ──────

TEST_CASE("speaker_db: enrollment refuses without a consent attestation", "[unit]") {
    const std::string dir = make_temp_dir();
    const std::vector<float> v = norm({1, 0, 0, 0, 0, 0, 0, 0});
    REQUIRE_FALSE(speaker_db_enroll(dir.c_str(), "alice", v.data(), D, /*consent=*/false));

    // Nothing may have been written.
    speaker_db* db = speaker_db_load(dir.c_str());
    REQUIRE(speaker_db_count(db) == 0);
    speaker_db_free(db);
}

TEST_CASE("speaker_db: retain narrows to the claimed roster (no open 1:N)", "[unit]") {
    const std::string dir = make_temp_dir();
    REQUIRE(speaker_db_enroll(dir.c_str(), "alice", norm({1, 0, 0, 0, 0, 0, 0, 0}).data(), D, /*consent=*/true));
    REQUIRE(speaker_db_enroll(dir.c_str(), "bob", norm({0, 1, 0, 0, 0, 0, 0, 0}).data(), D, /*consent=*/true));
    REQUIRE(speaker_db_enroll(dir.c_str(), "carol", norm({0, 0, 1, 0, 0, 0, 0, 0}).data(), D, /*consent=*/true));

    speaker_db* db = speaker_db_load(dir.c_str());
    REQUIRE(speaker_db_count(db) == 3);

    // Claim two of three (with whitespace + one name that isn't enrolled).
    REQUIRE(speaker_db_retain(db, " alice , carol , mallory ") == 2);
    REQUIRE(speaker_db_count(db) == 2);
    REQUIRE(std::string(speaker_db_name(db, 0)) == "alice");
    REQUIRE(std::string(speaker_db_name(db, 1)) == "carol");

    // The unclaimed profile must be unreachable: a query that would have
    // matched bob now stays unmatched.
    const std::vector<float> near_bob = norm({0.1f, 0.9f, 0, 0, 0, 0, 0, 0});
    REQUIRE(speaker_db_match(db, near_bob.data(), D, 0.7f, nullptr) == nullptr);
    speaker_db_free(db);
}

TEST_CASE("speaker_db: retain with an empty roster keeps nothing", "[unit]") {
    const std::string dir = make_temp_dir();
    REQUIRE(speaker_db_enroll(dir.c_str(), "alice", norm({1, 0, 0, 0, 0, 0, 0, 0}).data(), D, /*consent=*/true));
    speaker_db* db = speaker_db_load(dir.c_str());
    REQUIRE(speaker_db_retain(db, "") == 0);
    REQUIRE(speaker_db_count(db) == 0);
    speaker_db_free(db);
}

TEST_CASE("speaker_db: legacy v1 profiles (no consent trailer) still load", "[unit]") {
    const std::string dir = make_temp_dir();
    const std::vector<float> alice = norm({1, 0, 0, 0, 0, 0, 0, 0});

    // Hand-write a v1 file: magic, version=1, dim, embedding — no trailer.
    {
        FILE* f = fopen((dir + "/alice.spkr").c_str(), "wb");
        REQUIRE(f != nullptr);
        const char magic[4] = {'S', 'P', 'K', 'R'};
        const uint32_t version = 1, dim = D;
        REQUIRE(fwrite(magic, 1, 4, f) == 4);
        REQUIRE(fwrite(&version, 4, 1, f) == 1);
        REQUIRE(fwrite(&dim, 4, 1, f) == 1);
        REQUIRE(fwrite(alice.data(), sizeof(float), D, f) == (size_t)D);
        fclose(f);
    }

    speaker_db* db = speaker_db_load(dir.c_str());
    REQUIRE(speaker_db_count(db) == 1);
    float score = 0.0f;
    const char* name = speaker_db_match(db, alice.data(), D, 0.9f, &score);
    REQUIRE(name != nullptr);
    REQUIRE(std::string(name) == "alice");
    speaker_db_free(db);
}

// ---------------------------------------------------------------------------
// enroll_into — same-handle enrollment (found via CrisperWeaver #35 round 2).
//
// The path-based speaker_db_enroll() writes the profile to disk but an
// already-open handle never learns of it: enroll → match on the SAME handle
// scored -1.0 against an empty roster, and every caller had to know the
// undocumented "close and reopen after enrolling" contract. enroll_into()
// keeps the on-disk write AND updates the handle — without ever widening a
// retained roster (the #266 closed-roster guarantee).
// ---------------------------------------------------------------------------

TEST_CASE("speaker_db: enroll_into makes the name matchable on the same handle", "[unit]") {
    const std::string dir = make_temp_dir();
    struct speaker_db* db = speaker_db_load(dir.c_str());
    REQUIRE(db != nullptr);
    speaker_db_retain(db, "jfk"); // empty dir: roster claimed, none enrolled yet

    std::vector<float> jfk = norm({1, 0, 1, 0, 1, 0, 1, 0});
    REQUIRE(speaker_db_enroll_into(db, "jfk", jfk.data(), D, /*consent=*/true));
    REQUIRE(speaker_db_count(db) == 1);

    float score = -2.0f;
    const char* name = speaker_db_match(db, jfk.data(), D, 0.7f, &score);
    REQUIRE(name != nullptr);
    REQUIRE(std::string(name) == "jfk");
    REQUIRE(score >= 0.99f);
    speaker_db_free(db);

    // The disk write happened too: a fresh handle sees the profile.
    struct speaker_db* db2 = speaker_db_load(dir.c_str());
    REQUIRE(speaker_db_count(db2) == 1);
    speaker_db_free(db2);
}

TEST_CASE("speaker_db: enroll_into never widens a retained roster", "[unit]") {
    const std::string dir = make_temp_dir();
    struct speaker_db* db = speaker_db_load(dir.c_str());
    speaker_db_retain(db, "alice");

    std::vector<float> bob = norm({0, 1, 0, 1, 0, 1, 0, 1});
    // Disk write succeeds — bob is enrolled for FUTURE handles that claim
    // him — but this handle's closed roster must not grow.
    REQUIRE(speaker_db_enroll_into(db, "bob", bob.data(), D, /*consent=*/true));
    REQUIRE(speaker_db_count(db) == 0);
    float score = -2.0f;
    REQUIRE(speaker_db_match(db, bob.data(), D, 0.7f, &score) == nullptr);
    speaker_db_free(db);

    struct speaker_db* db2 = speaker_db_load(dir.c_str());
    REQUIRE(speaker_db_count(db2) == 1); // bob's .spkr is on disk
    speaker_db_free(db2);
}

TEST_CASE("speaker_db: enroll_into replaces an existing profile in place", "[unit]") {
    const std::string dir = make_temp_dir();
    std::vector<float> v1 = norm({1, 0, 0, 0, 0, 0, 0, 0});
    std::vector<float> v2 = norm({0, 0, 0, 0, 0, 0, 0, 1});
    REQUIRE(speaker_db_enroll(dir.c_str(), "alice", v1.data(), D, true));

    struct speaker_db* db = speaker_db_load(dir.c_str());
    speaker_db_retain(db, "alice");
    REQUIRE(speaker_db_enroll_into(db, "alice", v2.data(), D, true));
    REQUIRE(speaker_db_count(db) == 1); // replaced, not duplicated

    float score = -2.0f;
    const char* name = speaker_db_match(db, v2.data(), D, 0.7f, &score);
    REQUIRE(name != nullptr);
    REQUIRE(score >= 0.99f); // matches the NEW embedding
    speaker_db_free(db);
}

TEST_CASE("speaker_db: enroll_into refuses without consent", "[unit]") {
    const std::string dir = make_temp_dir();
    struct speaker_db* db = speaker_db_load(dir.c_str());
    speaker_db_retain(db, "alice");
    std::vector<float> a = norm({1, 1, 0, 0, 0, 0, 0, 0});
    REQUIRE_FALSE(speaker_db_enroll_into(db, "alice", a.data(), D, /*consent=*/false));
    REQUIRE(speaker_db_count(db) == 0);
    speaker_db_free(db);
}
