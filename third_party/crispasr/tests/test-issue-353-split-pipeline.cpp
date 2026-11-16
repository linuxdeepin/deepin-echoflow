// test-issue-353-split-pipeline.cpp — #353 encode∥decode gating guard.
//
// The slice pipeline runs a backend's encode_slice() on a producer thread while
// decode_slice() runs on the caller's. That is safe only while nothing ELSE
// touches the model, and the conditions are the kind you lose by omission:
//
//  - the original CRISPASR_SLICE_PIPELINE override re-derived the decision and
//    dropped the return_logits and worker-pool conditions the default path
//    enforced, so `=1` could turn either hazard back on;
//  - nothing covered the per-slice #89 gap-fill that finish_slice runs when
//    vad_slice_cap_seconds() > 0. That pass calls back into be.transcribe() on
//    the CONSUMER thread, so it encoded concurrently with the producer.
//    Reproduced on parakeet-tdt-0.6b-v3 (non-JA) with
//    CRISPASR_PARAKEET_VAD_SLICE_CAP=12 --vad: SIGABRT on
//    GGML_ASSERT(ggml_are_same_layout(src, dst)) — two threads, one ggml sched.
//
// So the decision is one pure function and every condition gets a case here.
// Pure CPU, no model load.

#include "crispasr_split_pipeline.h"

#include <catch2/catch_test_macros.hpp>

namespace {

// The shape of a run that SHOULD pipeline: several slices, a split-capable
// backend, no worker pool, no logits, no model-reentering post-pass.
crispasr_split::Inputs good() {
    crispasr_split::Inputs in;
    in.multiple_slices = true;
    in.backend_supports_split = true;
    in.worker_pool_requested = false;
    in.return_logits = false;
    in.post_pass_reenters_model = false;
    return in;
}

} // namespace

TEST_CASE("issue #353: a plain multi-slice run pipelines", "[unit][split-pipeline][issue-353]") {
    REQUIRE(crispasr_split::available(good()));
    REQUIRE(crispasr_split::enabled(good(), nullptr));
}

TEST_CASE("issue #353: every hazard refuses the pipeline", "[unit][split-pipeline][issue-353]") {
    SECTION("a single slice has nothing to overlap") {
        auto in = good();
        in.multiple_slices = false;
        REQUIRE_FALSE(crispasr_split::available(in));
    }
    SECTION("a backend that does not expose the split pair") {
        auto in = good();
        in.backend_supports_split = false;
        REQUIRE_FALSE(crispasr_split::available(in));
    }
    SECTION("-p workers already parallelise") {
        auto in = good();
        in.worker_pool_requested = true;
        REQUIRE_FALSE(crispasr_split::available(in));
    }
    SECTION("--return-logits: last_ctc_logits() is per-call backend state") {
        auto in = good();
        in.return_logits = true;
        REQUIRE_FALSE(crispasr_split::available(in));
    }
    SECTION("a post-pass that re-enters the model would encode on two threads") {
        auto in = good();
        in.post_pass_reenters_model = true;
        REQUIRE_FALSE(crispasr_split::available(in));
    }
}

TEST_CASE("issue #353: CRISPASR_SLICE_PIPELINE=0 always turns it off", "[unit][split-pipeline][issue-353]") {
    REQUIRE_FALSE(crispasr_split::enabled(good(), "0"));
}

TEST_CASE("issue #353: CRISPASR_SLICE_PIPELINE=1 cannot override a safety condition",
          "[unit][split-pipeline][issue-353]") {
    // This is the regression: the override used to re-derive the decision from
    // a shorter list, so `=1` re-enabled the hazards the default path refused.
    for (int which = 0; which < 3; which++) {
        auto in = good();
        if (which == 0)
            in.return_logits = true;
        else if (which == 1)
            in.worker_pool_requested = true;
        else
            in.post_pass_reenters_model = true;
        INFO("hazard index " << which);
        REQUIRE_FALSE(crispasr_split::enabled(in, "1"));
    }
    // It may still enable a run that is already safe.
    REQUIRE(crispasr_split::enabled(good(), "1"));
}
