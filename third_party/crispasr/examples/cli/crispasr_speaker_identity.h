// crispasr_speaker_identity.h — whose voice is a PRESET voice?
//
// The voice-clone gate (crispasr_voice_clone_policy.h) answers "did this voice
// come from a recording *we* processed". That is the right question for the
// speaker-consent gate, and the wrong one for Art. 50(4).
//
// A deep fake under Art. 3(60) is AI-generated audio "resembling existing
// persons ... which would falsely appear to a person to be authentic". Nothing
// in that turns on whether a WAV passed through one of our bakers. A preset
// voice shipped inside a model can be an identifiable individual — a named
// donor, or a corpus speaker such as VCTK's p225 — and synthesizing with it
// produces exactly the content Art. 50(4) is about.
//
// So `is_clone == false` was doing double duty: correctly meaning "no consent
// attestation needed from this operator", and incorrectly meaning "nothing to
// disclose". docs/eu-ai-act.md said as much in prose — "kokoro and vibevoice
// packs are converted from upstream voicepacks and .pt prompts with no
// recording involved" — which is a true statement about OUR conversion step and
// silent about whose voice it is.
//
// CONSENT AND DISCLOSURE ARE NOT THE SAME DUTY
// --------------------------------------------
// A real_person preset requires the audible disclosure and does NOT require
// --i-have-rights. Those are different questions with different holders:
// whether the voice donor consented to their recordings being used to build the
// model is a licensing matter settled upstream, between the donor and whoever
// trained it. The operator downstream cannot attest to it and demanding that
// they do would be theatre. What the operator owes is the audience knowing the
// audio is synthetic, which is the Art. 50(4) duty and nothing else.
//
// Cloning is where both apply: the operator is the one taking a specific
// person's voice, so they attest, and the output is a deep fake, so it is
// disclosed.
//
// WHY "unknown" IS ITS OWN VALUE
// ------------------------------
// Most upstream models simply do not say. Collapsing that into "synthetic"
// would silently assert the answer that happens to require no work, on exactly
// the models nobody has checked. Collapsing it into "real_person" would prepend
// a spoken sentence to every stock TTS voice in the project and train operators
// to reach for --no-spoken-disclaimer, which is worse than the disease.
//
// So `unknown` warns once per model and does not force a disclosure — a
// warning the operator can act on (they can pass --speaker-identity), attached
// to the model it is actually about. It is a question handed to the deployer,
// not a verdict.
//
// This header is the MECHANISM. The per-backend verdicts are researched
// separately, from each provider's own model card, and land as
// declared_speaker_identity() overrides and GGUF stamps. Until a backend has
// been read, it stays Unknown — an unresearched default that claims nothing.
// Guessing "synthetic" to quiet the warning is the costly error here, because
// it is silent and it is wrong in the direction that removes a disclosure.

#pragma once

#include <mutex>
#include <set>
#include <string>

namespace crispasr_voice {

// Whose voice a fixed-speaker (non-cloning) path produces.
enum class SpeakerIdentity {
    // Not established. Warns once per model; does not force a disclosure.
    Unknown = 0,
    // A designed or blended voice that is not any one person.
    Synthetic,
    // An identifiable individual: a named donor, or a corpus speaker.
    RealPerson,
};

// The GGUF metadata key a voice pack carries to declare its speaker's status.
// Sits alongside crispasr.voice.cloned_from_recording; the two are independent
// (a pack can be a non-clone preset AND a real person).
inline const char* speaker_identity_key() {
    return "crispasr.voice.speaker_identity";
}

// Per-entry variant inside a multi-voice bank, mirroring
// bank_provenance_key_for(): one bundle can hold voices of different status.
inline std::string speaker_identity_key_for(const std::string& voice_name) {
    return "crispasr.voice." + voice_name + ".speaker_identity";
}

inline const char* to_string(SpeakerIdentity id) {
    switch (id) {
    case SpeakerIdentity::RealPerson:
        return "real_person";
    case SpeakerIdentity::Synthetic:
        return "synthetic";
    default:
        return "unknown";
    }
}

// Parse a declared or operator-supplied value. Case- and whitespace-tolerant
// because it arrives from a CLI flag, a JSON field and GGUF metadata alike.
//
// An unrecognised value resolves to Unknown rather than being trusted: a typo
// ("real-person", "human") must not silently become a value with weaker duties
// than the author intended. `out_recognised` lets callers warn about the typo
// instead of swallowing it.
// An EMPTY string means "not supplied" and is recognised; only a non-empty
// value that matches nothing is a typo worth reporting.
inline SpeakerIdentity parse_speaker_identity(const std::string& raw, bool* out_recognised = nullptr) {
    std::string v;
    for (char c : raw) {
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n')
            continue;
        v += (c >= 'A' && c <= 'Z') ? (char)(c - 'A' + 'a') : c;
    }
    if (out_recognised)
        *out_recognised = true;
    if (v == "real_person")
        return SpeakerIdentity::RealPerson;
    if (v == "synthetic")
        return SpeakerIdentity::Synthetic;
    if (v.empty() || v == "unknown")
        return SpeakerIdentity::Unknown;
    if (out_recognised)
        *out_recognised = false;
    return SpeakerIdentity::Unknown;
}

// How strong a duty each value implies. RealPerson > Synthetic > Unknown.
inline int duty_rank(SpeakerIdentity id) {
    switch (id) {
    case SpeakerIdentity::RealPerson:
        return 2;
    case SpeakerIdentity::Synthetic:
        return 1;
    default:
        return 0;
    }
}

// Resolve whose voice this is.
//
//   1. an explicit operator override (--speaker-identity / "speaker_identity")
//      wins outright, in BOTH directions.
//   2. otherwise: the STRONGEST duty any declared source claims — the voice
//      pack or bank entry, the model's own stamp, and the researched backend
//      table.
//
// The override is absolute because it is an explicit human decision by the
// party carrying the duty: someone who knows a "synthetic" label is wrong has
// to be able to say so, and someone who knows a "real_person" label is wrong
// does too. A flag that can only add duties gets ignored for the other half.
//
// The declared sources are combined by STRONGEST rather than by precedence,
// which is the one non-obvious rule here. They are independent machine-recorded
// claims about the same fact, and the failure that costs a disclosure is one of
// them silently overriding another downward — a stale stamp saying "synthetic"
// on a model the table knows is a real person, say. Taking the strongest means
// a stamp can only ever upgrade a weaker answer, and disagreement fails toward
// disclosure. The escape hatch for a genuinely wrong strong claim is rule 1,
// which requires a human to type it.
//
// (Precedence would be the natural design if these were versions of one claim.
// They are not: the pack knows about itself, the stamp knows about the
// checkpoint, and the table is research about the upstream model. Any of the
// three can be the only one that has heard of a given voice.)
inline SpeakerIdentity resolve_speaker_identity(SpeakerIdentity override_value, SpeakerIdentity pack_value,
                                                SpeakerIdentity backend_value,
                                                SpeakerIdentity model_value = SpeakerIdentity::Unknown) {
    if (override_value != SpeakerIdentity::Unknown)
        return override_value;
    SpeakerIdentity best = pack_value;
    if (duty_rank(model_value) > duty_rank(best))
        best = model_value;
    if (duty_rank(backend_value) > duty_rank(best))
        best = backend_value;
    return best;
}

// Does this output need the spoken Art. 50(4) disclosure?
//
// True for a clone (the operator took a specific person's voice) and for a
// preset that IS a specific person (the audio resembles an identifiable
// individual either way — the audience cannot tell which pipeline produced it,
// and Art. 3(60) does not ask them to).
//
// Unknown returns false: see the header comment on why it is not treated as
// real_person. It is the caller's job to surface should_warn_unknown_identity()
// so the question reaches someone who can answer it.
inline bool requires_spoken_disclosure(bool is_clone, SpeakerIdentity identity) {
    return is_clone || identity == SpeakerIdentity::RealPerson;
}

// Does this voice need a speaker-consent attestation (--i-have-rights /
// "consent_attestation")?
//
// Cloning only. Spelled out as its own function, next to the one above, because
// the whole bug this header exists for was these two questions sharing one
// boolean. See "CONSENT AND DISCLOSURE ARE NOT THE SAME DUTY" above.
inline bool requires_consent_attestation(bool is_clone, SpeakerIdentity /*identity*/) {
    return is_clone;
}

// Worth telling the operator that nobody has established whose voice this is.
// Only for non-clones: a clone is disclosed and gated regardless, so the
// question does not change the outcome and the warning would be noise.
inline bool should_warn_unknown_identity(bool is_clone, SpeakerIdentity identity) {
    return !is_clone && identity == SpeakerIdentity::Unknown;
}

// The warning text. One canonical wording so the CLI, the server, Wyoming and
// the ABI cannot drift, and so it names what to DO — a warning an operator
// cannot act on is one they learn to ignore.
inline std::string unknown_identity_warning(const std::string& model_id) {
    return "crispasr: warning: '" + (model_id.empty() ? std::string("<unknown-model>") : model_id) +
           "' does not record whether its preset voice belongs to a real person, so no spoken AI\n"
           "  disclosure was added. If the voice is that of an identifiable individual, the output is a\n"
           "  deep fake under EU AI Act Art. 3(60) and you carry the Art. 50(4) duty to disclose it.\n"
           "  Pass --speaker-identity real_person to have CrispASR prepend the disclosure, or\n"
           "  --speaker-identity synthetic to silence this.";
}

// Once-per-model gate for the warning above.
//
// Keyed by model id, not by process: a server may load several backends over
// its lifetime and each unanswered one is a separate thing the operator needs
// to know. Returns true the first time a given id is seen.
//
// Thread-safe because the server calls it from request threads.
namespace detail {
// One home for the warned-model set, so the reset below actually clears the
// state the claim below reads. Keeping them in separate function-local statics
// is the obvious way to write a reset that silently does nothing.
struct WarnedModels {
    std::mutex mu;
    std::set<std::string> seen;
};
inline WarnedModels& warned_models() {
    static WarnedModels w;
    return w;
}
} // namespace detail

inline bool claim_unknown_identity_warning(const std::string& model_id) {
    detail::WarnedModels& w = detail::warned_models();
    std::lock_guard<std::mutex> lock(w.mu);
    return w.seen.insert(model_id.empty() ? "<unknown-model>" : model_id).second;
}

// Test-only: forget which models have warned, so cases don't depend on order.
inline void reset_unknown_identity_warnings_for_test() {
    detail::WarnedModels& w = detail::warned_models();
    std::lock_guard<std::mutex> lock(w.mu);
    w.seen.clear();
}

} // namespace crispasr_voice
