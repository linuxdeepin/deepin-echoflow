// test-vad-load-failure.cpp — a VAD model that fails to RESOLVE must be
// distinguishable from "no VAD requested" (#311 follow-up).
//
// crispasr_resolve_vad_model() returns an empty string for BOTH cases:
//   (a) the user asked for no VAD at all, and
//   (b) the user explicitly asked for one and the download/resolve failed.
//
// crispasr_compute_audio_slices() only enters its VAD block when the path is
// non-empty, so case (b) silently took the "no VAD requested" branch and left
// `out_vad_load_failed` false. The #311 strict guard in crispasr_run.cpp keys
// off exactly that flag, so --strict-pipeline / --require-vad could not fire
// on a failed download: the run exited 0 having quietly not run VAD at all.
//
// Reproduced on a real machine, not theorised: with ~/.cache/crispasr dangling
// (its backing volume unmounted), `--strict-pipeline --vad -vm marblenet`
// printed "download failed" and still exited 0 with a full-file chunk export.
// A gate whose whole documented purpose is a non-zero exit, that cannot go red.
//
// This test pins the CONTRACT at source level; it needs no network and no
// model, which is the point — the bug only shows up when a download fails.

#include <catch2/catch_test_macros.hpp>

#include <fstream>
#include <sstream>
#include <string>

#ifndef CRISPASR_SOURCE_DIR
#error "CRISPASR_SOURCE_DIR must be defined by the build"
#endif

namespace {
std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    REQUIRE(f.good());
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
} // namespace

TEST_CASE("an explicitly-requested VAD that fails to resolve reports load failure", "[vad-load-failure]") {
    const std::string src = slurp(std::string(CRISPASR_SOURCE_DIR) + "/examples/cli/crispasr_vad_cli.cpp");

    const size_t fn = src.find("crispasr_compute_audio_slices(");
    REQUIRE(fn != std::string::npos);
    const std::string body = src.substr(fn);

    // The empty-path branch must exist AND must set the load-failed flag when
    // the user actually asked for a VAD. Asserting only that the flag is
    // written somewhere would pass on the buggy version, which wrote it solely
    // inside the non-empty branch.
    const size_t empty_branch = body.find("vad_path.empty()");
    REQUIRE(empty_branch != std::string::npos);

    // Assert the token that exists ONLY when the behaviour does.
    //
    // A first draft of this test looked for "out_vad_load_failed" near the
    // empty-path branch and PASSED against the buggy code, because the flag is
    // also written (to false, then from the inner call) inside the non-empty
    // branch a few hundred characters away. Proximity is not behaviour.
    //
    // The flag being set to TRUE is the thing that only the fix introduces:
    // before it, the sole assignments were `= false` at entry and `= load_failed`
    // from crispasr_compute_vad_slices.
    const bool raises_true = body.find("*out_vad_load_failed = true") != std::string::npos;
    REQUIRE(raises_true);

    // ...and it must be guarded by "did the user actually ask for a VAD",
    // so a plain run with no --vad does not report a phantom failure.
    const size_t raise = body.find("*out_vad_load_failed = true");
    const std::string ctx = body.substr(raise > 600 ? raise - 600 : 0, 900);
    INFO(ctx);
    REQUIRE((ctx.find("params.vad") != std::string::npos || ctx.find("vad_model") != std::string::npos));
}

TEST_CASE("the #311 strict guard still keys off the load-failed flag", "[vad-load-failure]") {
    // The other half of the join: the flag is only worth setting because
    // crispasr_run.cpp turns it into a non-zero exit.
    const std::string run = slurp(std::string(CRISPASR_SOURCE_DIR) + "/examples/cli/crispasr_run.cpp");
    REQUIRE(run.find("vad_load_failed") != std::string::npos);
    REQUIRE(run.find("CRISPASR_STRICT_RC_VAD") != std::string::npos);
}
