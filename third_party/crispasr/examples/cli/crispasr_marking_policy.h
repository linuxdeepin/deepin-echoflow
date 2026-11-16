// crispasr_marking_policy.h — who may opt out of the spoken AI-disclaimer.
//
// The rule the server applies to a /v1/audio/speech request that sets
// "spoken_disclaimer": false, in one weight-free place so it can be unit-tested
// without a model, a socket, or a server process (tests/test-marking-policy.cpp).
// It lived inline in the request handler until #312, where a policy change went
// four days before anyone noticed it had broken every released client — a gate
// nobody can test on the tier CI actually runs is a gate that ships wrong.
//
// The policy itself:
//   - Only voice clones get a spoken disclaimer at all, so only they can opt out.
//   - Opting out requires an attestation that the caller accepts the AI-content
//     disclosure duty: the request's own `marking_attestation`, or a server
//     launched with --accept-marking-responsibility (the operator accepted that
//     duty for EVERY response the process serves, which subsumes the per-request
//     field — checking only the body refuses requests the operator has covered).
//   - #312: an unattested opt-out is DENIED, not refused. The caller doesn't get
//     the opt-out; it still gets its audio, with the disclaimer the default
//     prescribes. Serving the STRONGER default cannot emit weaker-than-default
//     output — which is the entire point of the gate — while refusing outright
//     only turns a client one field behind into a client with no TTS at all.
//     The caller is told (headers + audit log); the denial is never silent.

#pragma once

#include <string>

namespace crispasr_marking {

// What actually happens to this response, as opposed to what was asked for.
struct Decision {
    // Prepend the spoken AI-disclosure to the audio. True for a clone that
    // either didn't ask to skip it, or asked without attesting.
    bool apply_spoken_disclaimer = false;
    // The caller asked to skip it and attested — honored.
    bool optout_honored = false;
    // The caller asked to skip it without attesting — denied, served with the
    // disclaimer. The caller learns this from the response headers.
    bool optout_denied = false;
    // The attestation recorded in the audit log; set only when honored.
    std::string attestation;
    // Where that attestation came from: "request" or "server". Empty unless
    // honored.
    std::string scope;
};

// `needs_spoken_disclaimer` is crispasr_voice::requires_spoken_disclosure(),
// i.e. a clone OR a preset voice that belongs to an identifiable person. It
// used to be `is_voice_clone` alone, which silently excluded every real-person
// preset from a duty that Art. 3(60) attaches to the audio, not to the pipeline
// that made it (see crispasr_speaker_identity.h).
//
// The opt-out policy below is unchanged and deliberately does not care WHICH of
// the two put a disclaimer there: once one is owed, dropping it needs the same
// attestation either way.
//
// `server_accepted` / `server_attestation` are the launch-time
// --accept-marking-responsibility state (whisper_params::
// tts_marking_responsibility_accepted / tts_marking_attestation).
inline Decision decide(bool needs_spoken_disclaimer, bool requested_spoken_disclaimer,
                       const std::string& request_attestation, bool server_accepted,
                       const std::string& server_attestation) {
    Decision d;
    // Output that carries no spoken disclaimer in the first place has nothing to
    // opt out of, so "spoken_disclaimer": false is a no-op rather than a policy
    // decision to police.
    if (!needs_spoken_disclaimer)
        return d;
    if (requested_spoken_disclaimer) {
        d.apply_spoken_disclaimer = true;
        return d;
    }
    if (!request_attestation.empty()) {
        d.optout_honored = true;
        d.attestation = request_attestation;
        d.scope = "request";
        return d;
    }
    if (server_accepted) {
        d.optout_honored = true;
        // The flag can be passed without prose; record that rather than an
        // empty string, so the audit line always names a source.
        d.attestation = server_attestation.empty() ? "(unspecified)" : server_attestation;
        d.scope = "server";
        return d;
    }
    d.optout_denied = true;
    d.apply_spoken_disclaimer = true;
    return d;
}

// ---------------------------------------------------------------------------
// Which marks a given output container can actually carry.
//
// The watertight floor rests on this: an audio watermark may be dropped only
// when a C2PA manifest still marks the output. If the container can carry
// neither, the output would be fully unmarked AI audio, so the watermark opt-out
// must not be honored for it.
//
// The CLI enforces that per process (crispasr_enforce_cli_watermark_floor(),
// keyed on the output file extension). The SERVER picks its container per
// request and had no floor at all: `--no-watermark --accept-marking-
// responsibility` plus {"response_format": "mp3"} returned audio with no
// watermark and — because signing was hardcoded to the WAV branch — no manifest
// either, while the CLI in the same configuration forced the watermark back on.
// Same operator, same attestation, two different floors.
// ---------------------------------------------------------------------------

struct ContainerMarking {
    // A C2PA manifest can be embedded in this container.
    bool carries_c2pa = false;
    // MIME to hand crispasr_c2pa_sign_auto(); empty when carries_c2pa is false.
    const char* c2pa_mime = "";
};

// `response_format` as accepted by /v1/audio/speech and
// /v1/audio/speech-to-speech. Unknown values fall back to WAV, mirroring the
// handlers' own trailing `else` branch — the fallback must stay in step with
// them or the floor would be computed for a container that is never produced.
inline ContainerMarking container_marking_for_format(const std::string& response_format) {
    ContainerMarking m;
    if (response_format == "pcm" || response_format == "f32") {
        // Raw samples, no container ⇒ nowhere to put a manifest.
        return m;
    }
    if (response_format == "aac" || response_format == "opus") {
        // Raw ADTS / Ogg carry no manifest. The CLI remuxes these to MP4 when
        // C2PA is active; the server encodes what was asked for, so here the
        // watermark is the only mark available.
        return m;
    }
    if (response_format == "mp3") {
        m.carries_c2pa = true;
        m.c2pa_mime = "audio/mpeg"; // ID3v2.4 GEOB — native signer handles it
        return m;
    }
    m.carries_c2pa = true;
    m.c2pa_mime = "audio/wav"; // RIFF C2PA chunk; also the unknown-format default
    return m;
}

// ---------------------------------------------------------------------------
// Surfaces that emit RAW PCM with no container and no per-request attestation.
//
// The Wyoming protocol server (--wyoming-port, examples/cli/wyoming.cpp) is the
// case this exists for: it streams int16 PCM inside audio-chunk events, so no
// C2PA manifest is possible, and the protocol has no field a client could put a
// consent or marking attestation in.
//
// That surface shipped with NONE of this — no watermark, no clone
// classification, no disclaimer, no consent gate — for four releases, because
// the marking work was organised per-surface from a list in prose that Wyoming
// was never added to. The policy is named and unit-tested here so the next
// container-less surface has something to call instead of re-deriving it, and
// so the rule can go red in CI (tests/test-marking-policy.cpp) rather than only
// in a live server with a model loaded.
//
// The rules, and why they differ from the HTTP surface:
//   - Watermark is FORCED. No container ⇒ it is the only mark obtainable, so
//     --no-watermark must not reach it. Same conclusion the CLI floor reaches
//     for .opus/.aac output, by the same reasoning.
//   - The spoken disclaimer on a clone is NOT opt-out-able. Opting out requires
//     an attestation (#312) and there is no request field to carry one; a
//     server-wide --accept-marking-responsibility covers the *marking* duty but
//     is not a per-clip decision to drop the audible label, and silently
//     honoring it here would make every HA clone undisclosed.
//   - A clone with no operator --i-have-rights is REFUSED, not served. This is
//     the one place a refusal is right: #312's "deny the opt-out, not the
//     request" applies to opt-outs, and the HTTP surface likewise hard-refuses
//     a clone with no consent_attestation. Serving would emit an ungated clone
//     of a real person, which is the outcome the gate exists to prevent.
// ---------------------------------------------------------------------------

struct RawSurfaceDecision {
    // Do not synthesize: a clone was requested with no operator consent.
    bool refuse = false;
    // Prepend the audible AI disclosure (Art. 50(4)). Clones only.
    bool apply_spoken_disclaimer = false;
    // Embed the audio watermark with force=true, ignoring any --no-watermark.
    // Always true — a container-less surface has no other mark to fall back on.
    bool force_watermark = true;
};

// `is_clone`                — crispasr_voice::classify_voice(...).is_clone.
//                             Governs the CONSENT gate: only cloning asks this
//                             operator to attest to a speaker's permission.
// `operator_consent`        — whisper_params::tts_voice_clone_consent, i.e. the
//                             server was launched with --i-have-rights. The only
//                             attestation a protocol with no consent field can
//                             have.
// `needs_spoken_disclaimer` — crispasr_voice::requires_spoken_disclosure(), i.e.
//                             a clone OR a real-person preset. Governs the
//                             Art. 50(4) audible label, which is a different
//                             duty with a different trigger.
//
// The two are separate parameters because a real-person PRESET discloses
// without being gated: whether that voice's donor consented to the model being
// trained is a licensing question settled upstream, which this operator cannot
// attest to. Refusing them would be theatre; not disclosing them would be the
// bug. See crispasr_speaker_identity.h.
inline RawSurfaceDecision decide_raw_surface(bool is_clone, bool operator_consent, bool needs_spoken_disclaimer) {
    RawSurfaceDecision d;
    if (is_clone && !operator_consent) {
        d.refuse = true;
        return d;
    }
    d.apply_spoken_disclaimer = needs_spoken_disclaimer;
    return d;
}

} // namespace crispasr_marking
