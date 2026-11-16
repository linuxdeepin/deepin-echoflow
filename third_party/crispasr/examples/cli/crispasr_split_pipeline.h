// crispasr_split_pipeline.h — when may the dispatcher overlap encode and decode?
//
// The #353 slice pipeline runs a backend's encode_slice() on a producer thread
// while decode_slice() runs on the caller's. That is only safe while NOTHING
// else touches the model, and the conditions are easy to get wrong by omission:
// the original wiring enforced them in the default expression but dropped two
// of them in the CRISPASR_SLICE_PIPELINE override, and missed the per-slice
// post-pass entirely — which is a crash, not a degradation. Reproduced on
// parakeet-tdt-0.6b-v3 with CRISPASR_PARAKEET_VAD_SLICE_CAP=12:
//
//   GGML_ASSERT(ggml_are_same_layout(src, dst)) failed   (SIGABRT)
//
// because the #89 gap-fill inside finish_slice re-entered be.transcribe() on
// the consumer thread while the producer was encoding, putting two threads on
// one ggml scheduler.
//
// So the decision lives in one pure function, with one place to add a condition
// and a unit test that pins each one.
#pragma once

#include <cstdlib>

namespace crispasr_split {

// Everything the decision depends on. Every field is a REASON TO REFUSE except
// `supports_split`, so a new hazard is added as a new `must be false` field.
struct Inputs {
    bool multiple_slices = false; // nothing to overlap with a single slice
    bool backend_supports_split = false;
    bool worker_pool_requested = false; // -p N: the pool already parallelises
    bool return_logits = false;         // last_ctc_logits() is per-call backend state
    // A per-slice post-pass that calls back into the model (the #89 gap-fill in
    // finish_slice, gated on vad_slice_cap_seconds() > 0). It runs on the
    // consumer thread, so it would encode concurrently with the producer.
    bool post_pass_reenters_model = false;
};

// The env override may only turn the pipeline OFF, or ON when it is already
// safe. It must never be able to switch a safety condition off — that is what
// made the override a crash vector.
inline bool available(const Inputs& in) {
    if (!in.multiple_slices || !in.backend_supports_split)
        return false;
    if (in.worker_pool_requested || in.return_logits || in.post_pass_reenters_model)
        return false;
    return true;
}

// `env` is the raw CRISPASR_SLICE_PIPELINE value (nullptr when unset).
inline bool enabled(const Inputs& in, const char* env) {
    const bool ok = available(in);
    if (env && atoi(env) == 0)
        return false; // explicit off always wins
    return ok;        // explicit on cannot override a safety condition
}

} // namespace crispasr_split
