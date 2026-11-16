// crispasr_speaker_identity_models.h — the researched verdicts.
//
// crispasr_speaker_identity.h is the MECHANISM: three values, two duties, a
// resolution order. This file is the DATA — which shipped model produces whose
// voice, with the evidence for each answer beside it.
//
// Kept as one table rather than 50 adapter overrides on purpose. These are
// research results, not code: they come from reading each provider's own model
// card, they get revised when a provider says more, and they need to be
// reviewable in one place by someone checking the reasoning rather than the
// plumbing. A verdict scattered across `crispasr_backend_*.cpp` is a verdict
// nobody re-reads.
//
// HOW A VERDICT IS REACHED
// ------------------------
// From the provider's own documentation, never from the file name, the repo
// name, or what would be convenient. Two rules earned elsewhere in this
// project apply directly:
//
//   * Guessing "synthetic" to quiet the warning is the costly error. It is
//     silent, and it is wrong in the direction that REMOVES a disclosure.
//     Unknown is a question; synthetic is a claim.
//   * A model whose card does not say gets Unknown, and the check is recorded
//     here so the next reader does not re-litigate it.
//
// WHAT MATCHING ON A FILE NAME COSTS
// ----------------------------------
// One CrispASR backend serves many checkpoints — `orpheus` runs Canopy's base
// model AND Kartoffel's German fine-tune, and they have different answers — so
// the verdict cannot key on the backend alone. There is no metadata that
// distinguishes them either: models/convert-orpheus-to-gguf.py writes
// general.name = "orpheus-<variant>" for every one of them.
//
// So the table matches on the model's file name, which this project's own rule
// 3 says not to trust ("classify by provenance, not by filename"). It is used
// here because the alternative is no answer at all, and because the failure is
// SAFE: a renamed file matches nothing, falls through to Unknown, and warns.
// A rename cannot silently turn real_person into synthetic — it can only turn a
// known answer back into a question.
//
// The durable fix EXISTS NOW: a `crispasr.voice.speaker_identity` stamp in the
// GGUF itself, written by `models/convert-*.py --speaker-identity` and read by
// crispasr_voice::read_model_speaker_identity(). A stamped checkpoint answers
// for itself and survives being renamed, re-quantised or moved.
//
// This table is therefore the LEGACY FALLBACK — exactly the role
// architecture_is_recording_derived() plays for unstamped voice packs — and it
// stays until the published cstr/ mirrors are re-converted with the flag. Note
// the two are combined by strongest-duty, not by precedence: a stamp can
// upgrade a table answer but cannot silently cancel one. See
// resolve_speaker_identity().

#pragma once

#include "crispasr_speaker_identity.h"

#include <string>

namespace crispasr_voice {

// Case-insensitive "does `haystack` contain `needle`". Local so this header
// stays pure and unit-testable with no model, no GGUF and no filesystem.
inline bool contains_ci(const std::string& haystack, const std::string& needle) {
    if (needle.empty() || haystack.size() < needle.size())
        return false;
    for (size_t i = 0; i + needle.size() <= haystack.size(); ++i) {
        size_t j = 0;
        for (; j < needle.size(); ++j) {
            char a = haystack[i + j];
            char b = needle[j];
            if (a >= 'A' && a <= 'Z')
                a = (char)(a - 'A' + 'a');
            if (b >= 'A' && b <= 'Z')
                b = (char)(b - 'A' + 'a');
            if (a != b)
                break;
        }
        if (j == needle.size())
            return true;
    }
    return false;
}

// Whose voice does this (backend, model) produce?
//
// `backend` is CrispasrBackend::name(); `model_path` is the loaded checkpoint
// (full path or bare name — only the tail is inspected).
inline SpeakerIdentity identity_for_model(const std::string& backend, const std::string& model_path) {
    // ── piper ────────────────────────────────────────────────────────────
    // rhasspy/piper voices are single-speaker VITS models, each trained on one
    // named donor's recordings. Every voice in this project's registry is an
    // identifiable person: en_US-lessac (the Lessac Technologies corpus),
    // de_DE-thorsten (Thorsten Müller, who released his voice under CC0 —
    // released, which is consent to publish, not a reason to stop disclosing),
    // de_DE-kerstin. Blanket verdict: there is no synthetic piper voice here to
    // wrongly gate.
    if (backend == "piper")
        return SpeakerIdentity::RealPerson;

    // ── fastpitch ────────────────────────────────────────────────────────
    // nvidia/tts_en_fastpitch: "This model is trained on LJSpeech sampled at
    // 22050Hz". LJSpeech is 13,100 clips of ONE narrator — Linda Johnson,
    // recorded 2016-17 for LibriVox — and CrispASR ships only the English
    // single-speaker checkpoint (n_speakers=1).
    //
    // NOTE this is NOT the verdict CrispTTS reached for its German NeMo
    // FastPitch (HUI narrators). Different weights, independently checked,
    // same answer by a different route.
    if (backend == "fastpitch")
        return SpeakerIdentity::RealPerson;

    // ── bananamind-tts ───────────────────────────────────────────────────
    // Banaxi-Tech/BananaMind-TTS-V2.1-Preview, both shipped variants:
    //   en-us  LJSpeech — Linda Johnson again, the same donor as fastpitch.
    //   de-de  "ThorstenVoice Dataset 2022.10", card credits "Voice: Thorsten
    //          Müller" — the same donor as piper-de_DE-thorsten, reached by a
    //          second route.
    // The card is explicit that these are fixed recorded voices: "Fixed voices
    // only; this is not voice cloning", "No speaker embeddings or reference
    // audio conditioning".
    if (backend == "bananamind-tts")
        return SpeakerIdentity::RealPerson;

    // ── parler-tts ───────────────────────────────────────────────────────
    // Trained on LibriTTS-R and MLS — both LibriVox-derived, i.e. real
    // volunteer narrators — and the card states it "was also trained on 34
    // speakers, characterized by name (e.g. Jon, Lea, Gary, Jenna, Mike,
    // Laura)" to keep speaker identity consistent across generations.
    //
    // Whether "Jon" is that reader's real name does not change the analysis:
    // it is a consistent reproduction of one identifiable corpus speaker,
    // pseudonymous exactly like VCTK's p225 or CMU ARCTIC's bdl. Classified by
    // what the checkpoint CAN speak as, not by the safest thing an operator
    // might prompt it for — the rule CrispTTS applied to SauerkrautTTS.
    if (backend == "parler-tts")
        return SpeakerIdentity::RealPerson;

    // ── csm ──────────────────────────────────────────────────────────────
    // sesame/csm-1b: "The model open sourced here is a base generation model.
    // It is capable of producing a variety of voices, but it has not been
    // fine-tuned on any specific voice." No preset persona to disclose.
    //
    // CSM's actual cloning path takes a context recording as --voice, which is
    // a recording-reference and is caught by the CLONE gate, not here.
    if (backend == "csm")
        return SpeakerIdentity::Synthetic;

    // ── kokoro ───────────────────────────────────────────────────────────
    // Deliberately Unknown at the MODEL level, for every checkpoint including
    // the German HUI backbone: a kokoro model is a backbone, not a voice.
    // CrispASR's own card is explicit — "This is a base model, not a voice. It
    // pairs with a German voice pack". The backbone is multispeaker (HUI, 51
    // speakers); which person you hear is decided entirely by the style pack.
    //
    // So the answer lives in identity_for_voice_pack() below. Returning
    // Synthetic here — which an earlier revision did, inheriting hexgrad's
    // English verdict — would have overridden a real-person PACK through
    // strongest-duty resolution's weakest link: a wrong positive claim.
    if (backend == "kokoro")
        return SpeakerIdentity::Unknown;

    // ── orpheus ──────────────────────────────────────────────────────────
    // One backend, several checkpoints, different answers.
    //
    // kartoffel-orpheus-de-natural: the card describes it as fine-tuned
    // "primarily on natural human speech recordings" — permissively licensed
    // podcasts, lectures and OER — and its 19 speakers were EXTRACTED from
    // those recordings ("not all speakers could be reconstructed"). Real people
    // who spoke in public.
    if (backend == "orpheus") {
        if (contains_ci(model_path, "kartoffel-orpheus-de-natural"))
            return SpeakerIdentity::RealPerson;
        // The SYNTHETIC sibling is a genuinely different checkpoint: "trained
        // on synthetic German speech with explicit emotion and outburst
        // control" (its own card). Its four speakers — Martin, Luca, Anne,
        // Emma — are personas over generated audio, not extracted people.
        //
        // Checked rather than inferred from the repo name. "synthetic" in a
        // repo name is a naming convention, not a provenance statement, and
        // this is the direction where being wrong removes a disclosure.
        if (contains_ci(model_path, "kartoffel-orpheus-de-synthetic"))
            return SpeakerIdentity::Synthetic;
        // Everything else on this backend stays Unknown:
        //   orpheus-3b-0.1-ft   Canopy Labs disclose 100k+ h of "permissive /
        //                       non-copyrighted" audio and nothing at all about
        //                       the origin of tara, leah, jess, leo, dan, mia,
        //                       zac, zoe. HF card, GitHub repo and web checked.
        //   Orpheus-3b-German-FT (lex-au)   No training-data documentation.
        return SpeakerIdentity::Unknown;
    }

    // Every other backend: not yet researched. Unknown warns once per model and
    // names the fix; it does not claim the voice is synthetic.
    return SpeakerIdentity::Unknown;
}

// Map a GGUF's `general.architecture` to the backend name identity_for_model()
// keys on.
//
// Needed by --print-speaker-identity, which inspects a FILE with no session and
// therefore no CrispasrBackend to ask for name(). Without it that verb consulted
// only the voice-pack table and reported `unknown` for every piper, fastpitch
// and bananamind MODEL — silently, and the bulk stamper then skipped them all.
// Caught by running it against the real published repos before uploading.
//
// Most architectures already equal the backend name; the ones that differ do so
// for the usual reason (underscore vs hyphen, or a `-tts` suffix), and they are
// listed rather than normalised so a future mismatch is a visible edit here.
inline std::string backend_for_architecture(const std::string& arch) {
    if (arch == "bananamind_tts")
        return "bananamind-tts";
    if (arch == "csm-tts")
        return "csm";
    return arch;
}

// Whose voice is a VOICE PACK?
//
// The companion to identity_for_model(), for backends where the checkpoint is a
// backbone and the pack decides who you hear — kokoro is the case this exists
// for. Same legacy-fallback role: a pack carrying its own
// crispasr.voice.speaker_identity stamp answers for itself and never reaches
// here.
//
// Matched on the pack file name, with the same safe failure as above: an
// unrecognised pack falls through to Unknown and warns.
inline SpeakerIdentity identity_for_voice_pack(const std::string& pack_path) {
    if (pack_path.empty())
        return SpeakerIdentity::Unknown;

    // ── German packs recovered from the deleted Tundragoon/Kokoro-German ──
    // df_eva and dm_bernd are per-speaker style packs extracted from
    // HUI-Audio-Corpus-German, whose narrators are named — and these two carry
    // the narrators' names. HUI is built from librivox.org recordings, so Eva
    // and Bernd are real volunteers who read audiobooks.
    //
    // These are the same donors CrispTTS cites when marking the NeMo FastPitch
    // German model real_person. The "kokoro is synthetic" verdict is right for
    // hexgrad's English packs and wrong for these two; resolving that conflict
    // is what moved kokoro's answer from the model to the pack.
    if (contains_ci(pack_path, "df_eva") || contains_ci(pack_path, "dm_bernd"))
        return SpeakerIdentity::RealPerson;

    // ── kikiri German packs ──────────────────────────────────────────────
    // df_victoria and dm_martin are fine-tunes of
    // kikiri-german-base-51speakers-synthetic, whose card states: "Trained
    // entirely on synthetic (TTS-generated) audio — real human recordings may
    // improve naturalness".
    //
    // Their cards label the speakers "Victoria Asztaller (synthetic)" and
    // "Martin Harbecke (synthetic)". Person-shaped names, explicitly marked
    // synthetic by the provider, over a base with no human recordings in it.
    if (contains_ci(pack_path, "df_victoria") || contains_ci(pack_path, "dm_martin"))
        return SpeakerIdentity::Synthetic;

    // ── hexgrad's own packs ──────────────────────────────────────────────
    // af_heart, ef_dora, ff_siwis: Kokoro-82M's shipped style vectors,
    // documented upstream as designed/blended rather than any one person.
    if (contains_ci(pack_path, "af_heart") || contains_ci(pack_path, "ef_dora") || contains_ci(pack_path, "ff_siwis"))
        return SpeakerIdentity::Synthetic;

    return SpeakerIdentity::Unknown;
}

// ─────────────────────────────────────────────────────────────────────────
// OPEN QUESTIONS — models whose card has NOT been read, or where the evidence
// points somewhere the current verdict does not.
//
// EMPTY as of 2026-08-03: every TTS backend CrispASR ships has been checked
// against its provider's own documentation. Kept as a section, not deleted,
// because the next backend added starts here.
//
// The seven that resolved to Unknown did so from evidence of ABSENCE, and the
// check is recorded beside each so it is not re-litigated:
//
//   orpheus base (Canopy)   100k+ h of "permissive / non-copyrighted" audio
//                           disclosed, nothing about tara/leah/jess/leo/dan/
//                           mia/zac/zoe. HF card, GitHub and web checked.
//   orpheus lex-au German   No training-data documentation.
//   bark                    Checked, 2026-08-03: the HF card, the GitHub
//                           README, the repo's own model-card.md and the
//                           linked Notion prompt library. None of the four
//                           says where the 100+ presets came from. The only
//                           adjacent sentence is about CLONING ("not
//                           straightforward to voice clone known people"),
//                           which is a different question.
//                           NOTE: third-party write-ups describe the presets
//                           as "fully synthetic"; that phrasing is in none of
//                           Suno's own documents, and a summary is not a
//                           provider statement. Held at Unknown for that
//                           reason, not for lack of looking.
//   melotts                 Checked, 2026-08-03: the HF card, the GitHub
//                           README and docs/training.md. The training guide
//                           explains how to train YOUR OWN model and discloses
//                           nothing about the shipped EN-US / EN-BR /
//                           EN-Default / EN_INDIA / ES / FR / ZH / JP / KR
//                           speakers.
//   speecht5                Structurally unanswerable per model — the voice is
//                           a 512-d x-vector the OPERATOR supplies via --voice.
//                           Answer it per run with --speaker-identity.
//   (kokoro checkpoints)    Not unknown-by-ignorance: a kokoro model is a
//                           backbone, and the answer belongs to the pack.
//
// WHEN ADDING A BACKEND: read the provider's card before touching this file.
// Unknown is the correct default and costs only a warning. Synthetic is a
// claim, and the one that silently removes a disclosure.
// ─────────────────────────────────────────────────────────────────────────

} // namespace crispasr_voice
