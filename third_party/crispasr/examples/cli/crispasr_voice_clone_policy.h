// crispasr_voice_clone_policy.h — what counts as a voice CLONE.
//
// The predicate behind two gates that must agree on every surface:
//   * speaker-consent   — CLI --i-have-rights, server "consent_attestation"
//   * Art. 50(4) audible disclosure — the spoken AI-disclosure prefix
//
// Kept weight-free and IO-free so it can be unit-tested without a model, a
// voice file, or a server process (tests/test-voice-clone-policy.cpp), for the
// same reason crispasr_marking_policy.h is: a gate nobody can test on the tier
// CI actually runs is a gate that ships wrong.
//
// WHY THIS EXISTS
// ---------------
// Both gates used to spell "is this a clone?" as `voice ends with .wav`,
// inline, in two places. That predicate was wrong in three ways, and each way
// silently produced an unattested, undisclosed clone of a real person's voice:
//
//   1. The TADA one-command clone flow bakes `victim.wav` into a temp .gguf and
//      REWRITES --voice to point at it (crispasr_run.cpp) before the gate runs.
//      The most explicit cloning command in the CLI scored as "not a clone".
//   2. chatterbox clones ONLY through a baked .gguf voice pack (there is no
//      .wav path for it at all — models/bake-chatterbox-voice-from-wav.py makes
//      the pack). A headline cloning backend could therefore never trip either
//      gate. The same held for --make-ref output and for any hand-baked pack.
//   3. The server accepts a BARE voice name resolved against --voice-dir, so
//      {"voice": "victim"} reached the same .wav file as {"voice": "victim.wav"}
//      while scoring as "not a clone".
//
// A .gguf baked from someone's recording is exactly as much a deepfake as the
// recording it was baked from. The suffix is an implementation detail.
//
// WHAT IS *NOT* A CLONE
// ---------------------
// Not every voice pack is a clone. kokoro / qwen3-tts / miotts / vibevoice ship
// synthetic or upstream-licensed preset voices as .gguf, and `tada-ref-<lang>`
// packs are shipped references. Demanding --i-have-rights for
// `--voice kokoro-voice-af_heart.gguf` would gate a preset behind a speaker-
// consent attestation nobody can meaningfully give, and would break every
// documented example. So packs are presets by default and clones when they SAY
// they are: the bakers stamp `crispasr.voice.cloned_from_recording` into packs
// they derive from a user recording, and the gate reads it back.
//
// The honest limitation: a pack baked by a CrispASR older than that stamp
// carries no provenance and reads as a preset. Re-bake it to gate it. The two
// paths where the runtime KNOWS the provenance without asking the file — an
// inline bake this run, and a recording passed directly as --voice — do not
// depend on the stamp at all.
//
// VOICE BANKS — the bypass a file-shaped predicate cannot see
// -----------------------------------------------------------
// Everything above assumes --voice names a FILE: a recording, or a pack whose
// metadata can be read. cosyvoice3 breaks that assumption. Its voices live
// inside ONE bundle (`voices.gguf`), auto-discovered as a sibling of the model
// or pointed at by CRISPASR_COSYVOICE3_VOICES_PATH, and --voice selects an
// entry inside it BY NAME (crispasr_backend_cosyvoice3.cpp — whose own header
// calls them "baked voice-clone bundles").
//
// So the gate saw a bare name that resolved to no file on disk, read no
// metadata, and returned "preset" — for a zero-shot cloning backend, on every
// surface at once (CLI, server, Wyoming, ABI). `--voice victim.wav` on the same
// backend WAS gated, which is exactly why this looked covered.
//
// A bank name is therefore classified from the bank, not from the name: the
// per-voice stamp first (a bundle can legitimately hold both a preset and a
// clone), then the bank-wide stamp, then the producer architecture for bundles
// baked before the stamp existed. See read_bank_provenance() in
// crispasr_voice_provenance.h.
//
// The general lesson, which the next multi-voice backend will need: this gate
// reads `--voice`. Any OTHER route by which a voice reaches a backend — a
// bundle, an env var, a config file — is invisible to it until it is plumbed
// in. Adding a backend that selects voices by name from a container means
// adding a voice_bank_path() override, not just a new adapter.

#pragma once

#include "crispasr_speaker_identity.h" // SpeakerIdentity, carried on BankFacts

#include <string>

namespace crispasr_voice {

// Case-insensitive suffix test. Local so this header stays dependency-free.
inline bool ends_with_ci(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size())
        return false;
    for (size_t i = 0; i < suffix.size(); ++i) {
        const char a = s[s.size() - suffix.size() + i];
        const char b = suffix[i];
        const char la = (a >= 'A' && a <= 'Z') ? (char)(a - 'A' + 'a') : a;
        const char lb = (b >= 'A' && b <= 'Z') ? (char)(b - 'A' + 'a') : b;
        if (la != lb)
            return false;
    }
    return true;
}

// Does this voice name carry control characters?
//
// Rejected at the network ingress of every surface that accepts a caller-chosen
// voice, because the name is echoed into logs by code far away that has no idea
// it is handling untrusted input: the GGUF loader ("failed to open GGUF file
// '<name>'"), the kokoro adapter ("failed to read voice pack '<name>'"), and
// ggml itself, which this project does not patch. A newline in the name forges
// whole log records — including the [CONSENT] audit lines that are supposed to
// be the evidence that a clone was gated.
//
// Sanitizing at each fprintf is the losing move: there are many, they live in
// several libraries, and the next one added won't know to do it. Rejecting at
// ingress is one check that makes every downstream site safe, including the
// ones in third-party code. No legitimate voice name — a bare name, a preset, a
// path — contains a control character.
//
// This is deliberately NOT a general path guard. Each surface keeps its own
// (the server rejects "..", absolute and home paths; the CLI accepts real paths
// because the operator typed them). This is only about what can reach a log.
inline bool voice_name_has_control_chars(const std::string& voice) {
    for (unsigned char c : voice)
        if (c < 0x20 || c == 0x7f)
            return true;
    return false;
}

// A raw reference recording handed straight to a cloning backend.
//
// NOT just .wav: zonos accepts .mp3 and .flac references
// (crispasr_backend_zonos.cpp), and adding a decoder to another adapter is a
// one-line change that nobody would think to mirror in a consent gate. So this
// covers every container the audio loader can read, and errs toward "clone":
// a false positive costs an attestation the caller can give, a false negative
// ships an ungated clone of a real person.
inline bool is_recording_reference(const std::string& voice) {
    static const char* kAudioExt[] = {".wav", ".mp3",  ".flac", ".ogg",  ".opus", ".m4a", ".mp4",
                                      ".aac", ".aiff", ".aif",  ".webm", ".wma",  ".amr", ".caf"};
    for (const char* ext : kAudioExt)
        if (ends_with_ci(voice, ext))
            return true;
    return false;
}

// A baked voice pack. Says nothing on its own about whether it is a clone —
// see the header comment; that is what the provenance stamp is for.
inline bool is_voice_pack(const std::string& voice) {
    return ends_with_ci(voice, ".gguf");
}

// The GGUF metadata key a baker writes into a pack derived from a real
// recording. Read back by the gate; see crispasr_voice_provenance.h.
//
// In a multi-voice BANK the same key applies bank-wide: true means every entry
// in the bundle came from a recording. Use it when that is actually true; a
// mixed bundle wants the per-voice key below instead.
inline const char* provenance_key() {
    return "crispasr.voice.cloned_from_recording";
}

// Per-entry provenance inside a multi-voice bank:
//   crispasr.voice.<name>.cloned_from_recording
//
// Needed because a bank need not be all-or-nothing. Every entry cosyvoice3's
// converter writes today IS a clone — it bakes each one from a WAV, so the
// baker stamps them all true, including the built-in manifest's upstream
// asset — but a bundle that later mixes in a synthetic preset must be able to
// say so per entry. A bank-wide flag could only gate all of them or none.
inline std::string bank_provenance_key_for(const std::string& voice_name) {
    return "crispasr.voice." + voice_name + ".cloned_from_recording";
}

// Sentinel a current baker writes into every bank it produces: "this bundle
// stamps its entries".
//
// Without it, a missing per-voice key is ambiguous — it could mean "this entry
// is a preset" or "this bundle was baked before the stamp existed", and those
// two need opposite defaults. With it, absence of the sentinel is the honest
// signal that the bundle cannot answer, and only then does the producer
// architecture decide. Applying the architecture fallback unconditionally would
// override an explicit per-entry "false" with a guess.
inline const char* bank_stamped_key() {
    return "crispasr.voice.bank_stamped";
}

// LEGACY PACKS — packs baked before the stamp existed carry no provenance, and
// cannot be retro-stamped once published. For those, fall back to what the pack
// IS: its `general.architecture` names the producer that made it, and some
// producers only ever consume a reference recording.
//
// Listed here are the architectures whose ONLY producer in this repo takes a
// user WAV, and for which no preset pack exists to break:
//
//   chatterbox-voice     models/bake-chatterbox-voice-from-wav.py  (--input WAV)
//   qwen3tts.voicepack   models/bake-qwen3-tts-voice-pack.py       (name:wav:text)
//   cosyvoice3-voices    models/convert-cosyvoice3-voices-to-gguf.py (manifest of WAVs)
//
// cosyvoice3-voices is a BANK, and every entry in it is baked from a WAV —
// including the built-in default manifest, which runs the upstream
// `asset/zero_shot_prompt.wav` through CAMPPlus and the speech tokenizer. That
// is a real human speaker being cloned zero-shot, so it gates. It is NOT the
// tada-ref case: CrispASR ships no cosyvoice3 bank and auto-downloads none, so
// there is no shipped preset to break — the operator bakes the bundle
// themselves, which is the moment to ask for the attestation.
//
// The fallback applies only to bundles carrying NO provenance metadata at all
// (BankFacts::has_stamps below). Once a bank is baked by the current script
// every entry says explicitly what it is, and the per-entry answer wins.
//
// Deliberately NOT listed:
//   kokoro-voice, vibevoice-voice   — converted from upstream voicepacks/.pt,
//                                     no recording involved. Presets.
//   crispasr.reference (tada)       — AMBIGUOUS: both the shipped
//                                     tada-ref-<lang> packs and user --make-ref
//                                     output use it, and gating the shipped ones
//                                     would break `-l de` auto-download. Covered
//                                     going forward by the stamp instead; a
//                                     tada ref baked before it reads as a preset.
//   dots-tts-spk                    — a CAM++ speaker ENCODER (model weights),
//                                     never passed as --voice.
//
// This is classification by producer, not by filename: renaming a pack changes
// nothing. An unknown architecture is left as a preset — every recording-derived
// producer in this repo is enumerated above, so an unrecognised one is most
// likely a third-party or future preset, and defaulting unknown-to-clone would
// gate arbitrary packs on a guess.
inline bool architecture_is_recording_derived(const std::string& arch) {
    return arch == "chatterbox-voice" || arch == "qwen3tts.voicepack" || arch == "cosyvoice3-voices";
}

// The attestation recorded at bake time, when the baker had one. Audit trail
// only — an attestation in the pack does NOT satisfy the synthesis-time gate,
// because consent to bake and consent to publish this particular utterance are
// different questions.
inline const char* provenance_attestation_key() {
    return "crispasr.voice.consent_attestation";
}

struct CloneDecision {
    // Synthesizing with this voice clones an identifiable person: require the
    // consent attestation, prepend the spoken AI disclosure, emit [CONSENT].
    bool is_clone = false;
    // Stable token naming WHY, for the audit line and for tests. One of
    // "recording-reference" / "baked-from-wav" / "pack-provenance" /
    // "pack-architecture" / "bank-provenance" / "bank-architecture" /
    // "" (none).
    const char* reason = "";
    // What the pack or bank entry DECLARED about whose voice it is — not the
    // final answer. Callers resolve it against --speaker-identity and the
    // backend default via resolve_speaker_identity(); it rides along here
    // because classify_voice() already has the metadata open and re-reading it
    // per request on the server's hot path would be the obvious way to make
    // this expensive.
    SpeakerIdentity pack_identity = SpeakerIdentity::Unknown;
};

// What a multi-voice BANK says about the entry being selected. Filled in by
// read_bank_provenance() (crispasr_voice_provenance.h) when the backend selects
// voices by name from a bundle rather than by path; all-default otherwise.
struct BankFacts {
    // Whose voice this entry is (crispasr.voice.<name>.speaker_identity, then
    // the bank-wide key). Independent of declares_clone.
    SpeakerIdentity identity = SpeakerIdentity::Unknown;
    // The per-voice stamp for this entry, falling back to the bank-wide one.
    bool declares_clone = false;
    // The bundle carries provenance metadata at all. False means it predates
    // the stamp, which is the only case where the architecture fallback fires —
    // a current bundle has already answered the question per entry, and an
    // explicit "false" must not be overridden by a producer-level guess.
    bool has_stamps = false;
    // general.architecture of the bundle, for banks baked before the stamp.
    std::string architecture;
};

// `voice`                  — the --voice / "voice" value, resolved to the path
//                            the backend will actually open (bare server names
//                            resolved against --voice-dir first).
// `baked_from_wav_this_run` — this process baked `voice` from a user-supplied
//                            recording during this run (the TADA inline clone),
//                            so its .gguf suffix is an artefact of a WAV clone.
// `pack_declares_clone`     — the pack's provenance_key(), read from the file.
// `pack_architecture`       — the pack's general.architecture, for legacy packs
//                            baked before the stamp existed.
// `bank`                    — what the multi-voice bundle says about this entry,
//                            when the backend selects voices by name from one.
inline CloneDecision classify(const std::string& voice, bool baked_from_wav_this_run, bool pack_declares_clone,
                              const std::string& pack_architecture = std::string(),
                              const BankFacts& bank = BankFacts()) {
    CloneDecision d;
    if (voice.empty())
        return d;
    // Checked before the suffix tests: the inline-bake path rewrites `voice` to
    // a .gguf, and this flag is the only thing that still remembers it came
    // from a recording. Getting this order wrong is the original bug.
    if (baked_from_wav_this_run) {
        d.is_clone = true;
        d.reason = "baked-from-wav";
        return d;
    }
    if (is_recording_reference(voice)) {
        d.is_clone = true;
        d.reason = "recording-reference";
        return d;
    }
    if (pack_declares_clone) {
        d.is_clone = true;
        d.reason = "pack-provenance";
        return d;
    }
    // Legacy fallback: no stamp, but the producer named by the pack's
    // architecture only ever bakes from a recording.
    if (architecture_is_recording_derived(pack_architecture)) {
        d.is_clone = true;
        d.reason = "pack-architecture";
        return d;
    }
    // The voice is an entry inside a multi-voice bundle. Nothing above could
    // see it: `voice` here is a bare name that names no file on disk, so the
    // suffix tests and the pack read both came back empty.
    if (bank.declares_clone) {
        d.is_clone = true;
        d.reason = "bank-provenance";
        return d;
    }
    if (!bank.has_stamps && architecture_is_recording_derived(bank.architecture)) {
        d.is_clone = true;
        d.reason = "bank-architecture";
        return d;
    }
    return d;
}

} // namespace crispasr_voice
