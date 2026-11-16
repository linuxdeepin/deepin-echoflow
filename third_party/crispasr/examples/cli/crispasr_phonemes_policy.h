// crispasr_phonemes_policy.h — which backends can be driven by phonemes.
//
// `--tts-phonemes` hands a phoneme string straight to the acoustic model,
// skipping the G2P. That seam is what separates "our G2P is wrong" from "our
// model is wrong": feeding another implementation's phonemes through our model
// answers it in one run, which is how #316 was diagnosed.
//
// A backend can only honour it if its runtime exposes a phonemes-in entry
// point. Two do:
//
//   kokoro   kokoro_synthesize_phonemes()
//   piper    piper_tts_synthesize_phonemes()   (the runtime always had this;
//            the adapter only reached it by accident, as a fallback when
//            phonemization failed AND the text happened to contain non-ASCII)
//
// Everything else must REFUSE rather than quietly synthesize the text instead:
// a silent fallback makes an A/B look like the phonemes changed nothing, which
// is the exact wrong conclusion to hand someone debugging a pronunciation.
//
// Weight-free so tests/test-tts-phonemes-policy.cpp can pin it without a model.

#pragma once

#include <string>

namespace crispasr_phonemes_policy {

// Backends whose adapter forwards --tts-phonemes to a phonemes-in runtime call.
inline bool backend_supports(const std::string& backend) {
    return backend == "kokoro" || backend == "piper";
}

// One-line diagnostic for a backend that cannot honour the flag. Names the
// alternatives so the user is not left guessing.
inline std::string unsupported_message(const std::string& backend) {
    return "--tts-phonemes is not supported by backend '" + backend +
           "' (only kokoro and piper expose a phonemes-in entry point); refusing to fall back to "
           "text synthesis so an A/B cannot be misread.";
}

} // namespace crispasr_phonemes_policy
