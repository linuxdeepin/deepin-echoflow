// test-voice-clone-policy.cpp — what counts as a voice clone, and which
// containers can carry a C2PA manifest.
//
// Two gates hang off the first question (speaker consent: --i-have-rights /
// "consent_attestation"; and the Art. 50(4) audible AI disclosure), and the
// watertight marking floor hangs off the second. Both predicates are pure, so
// they are guarded here — on the unit tier CI actually runs — rather than only
// through a live server with a model loaded. Same reasoning as
// test-marking-policy.cpp, and the same lesson behind it (#312).
//
// Every "BYPASS" case below is a real one: it passed the old
// `voice ends with .wav` predicate as "not a clone", and each one produced an
// unattested, undisclosed clone of a real person's voice. They are the cases
// that prove this gate can go red.

#include "crispasr_marking_policy.h"
#include "crispasr_voice_clone_policy.h"
#include "crispasr_consent_record.h"
#include "crispasr_watermark_stats.h"

#include <catch2/catch_test_macros.hpp>


using crispasr_voice::classify;
using crispasr_voice::CloneDecision;

// Convenience: the common case where nothing was baked this run and the pack
// carries no provenance stamp — i.e. exactly what the old predicate saw.
static CloneDecision plain(const std::string& voice) {
    return classify(voice, /*baked_from_wav_this_run=*/false, /*pack_declares_clone=*/false);
}

TEST_CASE("no voice is not a clone", "[unit][compliance]") {
    REQUIRE_FALSE(plain("").is_clone);
}

TEST_CASE("a recording reference is a clone", "[unit][compliance]") {
    REQUIRE(plain("speaker.wav").is_clone);
    REQUIRE(std::string(plain("speaker.wav").reason) == "recording-reference");
    // Case-insensitive: the old predicate special-cased .WAV by hand, and any
    // other casing (.Wav) slipped through it.
    REQUIRE(plain("speaker.WAV").is_clone);
    REQUIRE(plain("speaker.Wav").is_clone);
    REQUIRE(plain("/abs/path/to/victim.wav").is_clone);
}

TEST_CASE("BYPASS 1: a voice baked from a wav this run is a clone", "[unit][compliance]") {
    // The TADA one-command clone bakes victim.wav into a temp .gguf and
    // REWRITES --voice to point at it before the gate runs. Suffix-only, the
    // most explicit cloning command in the CLI scored as "not a clone": no
    // --i-have-rights demanded, no [CONSENT] line, no spoken AI disclosure.
    const CloneDecision d = classify("/cache/tada-inline-voice.gguf",
                                     /*baked_from_wav_this_run=*/true, /*pack_declares_clone=*/false);
    REQUIRE(d.is_clone);
    REQUIRE(std::string(d.reason) == "baked-from-wav");
    // ... and the same path without that knowledge is what used to happen.
    REQUIRE_FALSE(plain("/cache/tada-inline-voice.gguf").is_clone);
}

TEST_CASE("BYPASS 2: a pack that declares it was baked from a recording is a clone", "[unit][compliance]") {
    // chatterbox clones ONLY through a baked .gguf — it has no .wav cloning
    // path at all — so a headline cloning backend could never trip either gate.
    // Same for --make-ref output and any hand-baked pack.
    const CloneDecision d = classify("my_voice.gguf",
                                     /*baked_from_wav_this_run=*/false, /*pack_declares_clone=*/true);
    REQUIRE(d.is_clone);
    REQUIRE(std::string(d.reason) == "pack-provenance");
}

TEST_CASE("an unstamped pack is a preset, not a clone", "[unit][compliance]") {
    // Deliberate: kokoro / qwen3-tts / miotts / vibevoice ship synthetic or
    // upstream-licensed preset voices as .gguf, and tada-ref-<lang> packs are
    // shipped references. Gating those behind a speaker-consent attestation
    // nobody can meaningfully give would break every documented example.
    // The honest cost is that a pack baked before the stamp existed reads as a
    // preset — re-bake it to gate it.
    REQUIRE_FALSE(plain("kokoro-voice-af_heart.gguf").is_clone);
    REQUIRE_FALSE(plain("tada-ref-de.gguf").is_clone);
    REQUIRE(std::string(plain("kokoro-voice-af_heart.gguf").reason) == "");
}

TEST_CASE("baked-from-wav outranks everything the file says", "[unit][compliance]") {
    // Runtime knowledge of provenance beats file metadata: a pack the runtime
    // just baked from a recording is a clone even if the pack forgot to say so.
    REQUIRE(classify("x.gguf", true, false).is_clone);
    REQUIRE(classify("x.wav", true, false).is_clone);
    REQUIRE(std::string(classify("x.wav", true, false).reason) == "baked-from-wav");
}

TEST_CASE("BYPASS 4: non-wav reference recordings are clones too", "[unit][compliance]") {
    // zonos accepts .mp3 and .flac references (crispasr_backend_zonos.cpp), so a
    // .wav-only predicate left `--voice victim.mp3` ungated on that backend.
    for (const char* v : {"victim.mp3", "victim.flac", "victim.m4a", "victim.ogg", "victim.OPUS"}) {
        INFO("voice=" << v);
        REQUIRE(plain(v).is_clone);
        REQUIRE(std::string(plain(v).reason) == "recording-reference");
    }
}

TEST_CASE("BYPASS 5: legacy packs are classified by producer architecture", "[unit][compliance]") {
    // A pack baked before the provenance stamp existed cannot be retro-stamped
    // once published. Fall back to what the pack IS: these architectures have
    // exactly one producer in-repo and it takes a user WAV.
    for (const char* arch : {"chatterbox-voice", "qwen3tts.voicepack"}) {
        INFO("arch=" << arch);
        const CloneDecision d = classify("legacy.gguf", false, /*pack_declares_clone=*/false, arch);
        REQUIRE(d.is_clone);
        REQUIRE(std::string(d.reason) == "pack-architecture");
    }
}

TEST_CASE("preset architectures stay ungated", "[unit][compliance]") {
    // Converted from upstream voicepacks / .pt files — no recording involved.
    // Gating these would demand a speaker-consent attestation nobody can give
    // and break every documented preset example.
    for (const char* arch : {"kokoro-voice", "vibevoice-voice", "", "some-future-arch"}) {
        INFO("arch=" << arch);
        REQUIRE_FALSE(classify("preset.gguf", false, false, arch).is_clone);
    }
    // tada refs are ambiguous (shipped tada-ref-<lang> AND user --make-ref share
    // the architecture), so they are covered by the stamp, not by the arch list.
    REQUIRE_FALSE(classify("tada-ref-de.gguf", false, false, "crispasr.reference").is_clone);
    REQUIRE(classify("myref.gguf", false, /*stamped=*/true, "crispasr.reference").is_clone);
}

// ---------------------------------------------------------------------------
// BYPASS 6: voices selected by name from a multi-voice BANK.
//
// cosyvoice3 keeps every voice inside one voices.gguf, discovered as a sibling
// of the model (or CRISPASR_COSYVOICE3_VOICES_PATH) and selected by name. So
// --voice named no file, resolve_voice_path() had nothing to resolve, no
// metadata was read, and a zero-shot voice clone scored as a preset — on the
// CLI, the server, Wyoming and the ABI at once. `--voice victim.wav` on the
// same backend WAS gated, which is why this looked covered.
// ---------------------------------------------------------------------------

using crispasr_voice::BankFacts;

// A bank entry: a bare name, no file behind it, plus what the bundle says.
static CloneDecision bank_entry(const std::string& name, const BankFacts& facts) {
    return classify(name, /*baked_from_wav_this_run=*/false, /*pack_declares_clone=*/false,
                    /*pack_architecture=*/std::string(), facts);
}

TEST_CASE("BYPASS 6: a bare bank name was invisible to the gate", "[unit][compliance]") {
    // This is the bug, pinned: with nothing known about the bundle, the name
    // "fleurs-en" is exactly what the old gate saw and it returned "preset".
    REQUIRE_FALSE(bank_entry("fleurs-en", BankFacts()).is_clone);
    // Handing the bank's facts in is the whole fix.
    BankFacts f;
    f.has_stamps = true;
    f.declares_clone = true;
    f.architecture = "cosyvoice3-voices";
    const CloneDecision d = bank_entry("fleurs-en", f);
    REQUIRE(d.is_clone);
    REQUIRE(std::string(d.reason) == "bank-provenance");
}

TEST_CASE("a bank baked before the stamp is classified by its producer", "[unit][compliance]") {
    // No provenance metadata at all: the bundle cannot say. Its architecture
    // can — convert-cosyvoice3-voices-to-gguf.py bakes every entry from a WAV,
    // and CrispASR ships no cosyvoice3 bank, so there is no preset to break.
    BankFacts legacy;
    legacy.has_stamps = false;
    legacy.architecture = "cosyvoice3-voices";
    const CloneDecision d = bank_entry("zero_shot", legacy);
    REQUIRE(d.is_clone);
    REQUIRE(std::string(d.reason) == "bank-architecture");
}

TEST_CASE("a stamped bank's explicit 'not a clone' is not overridden", "[unit][compliance]") {
    // The reason has_stamps exists. Once a bundle stamps its entries, an absent
    // per-voice key MEANS preset — falling back to the producer architecture
    // there would override an explicit answer with a guess, and re-gate every
    // preset entry in a mixed bundle.
    BankFacts stamped_preset;
    stamped_preset.has_stamps = true;
    stamped_preset.declares_clone = false;
    stamped_preset.architecture = "cosyvoice3-voices";
    REQUIRE_FALSE(bank_entry("some-preset", stamped_preset).is_clone);
}

TEST_CASE("a bank can hold a clone and a preset at once", "[unit][compliance]") {
    // The mixed-bundle case the per-voice key exists for: the upstream default
    // manifest entry and a user's own recording land in the SAME voices.gguf.
    BankFacts as_clone;
    as_clone.has_stamps = true;
    as_clone.declares_clone = true;
    BankFacts as_preset;
    as_preset.has_stamps = true;
    as_preset.declares_clone = false;
    REQUIRE(bank_entry("my-recording", as_clone).is_clone);
    REQUIRE_FALSE(bank_entry("shipped-demo", as_preset).is_clone);
}

TEST_CASE("an unknown bank architecture stays a preset", "[unit][compliance]") {
    // Same defaulting rule as packs: an unrecognised producer is most likely a
    // third-party or future preset, and defaulting unknown-to-clone would gate
    // arbitrary bundles on a guess.
    BankFacts unknown;
    unknown.has_stamps = false;
    unknown.architecture = "some-future-bank";
    REQUIRE_FALSE(bank_entry("whatever", unknown).is_clone);
}

TEST_CASE("bank facts never weaken a decision the file already earned", "[unit][compliance]") {
    // A recording reference stays a clone no matter what an unrelated bundle
    // says, and runtime knowledge still outranks everything.
    BankFacts inert;
    inert.has_stamps = true;
    inert.declares_clone = false;
    REQUIRE(classify("victim.wav", false, false, std::string(), inert).is_clone);
    REQUIRE(std::string(classify("victim.wav", false, false, std::string(), inert).reason) == "recording-reference");
    REQUIRE(classify("baked.gguf", true, false, std::string(), inert).is_clone);
    REQUIRE(classify("stamped.gguf", false, true, std::string(), inert).is_clone);
}

TEST_CASE("cosyvoice3-voices is on the recording-derived producer list", "[unit][compliance]") {
    using crispasr_voice::architecture_is_recording_derived;
    REQUIRE(architecture_is_recording_derived("cosyvoice3-voices"));
    REQUIRE(architecture_is_recording_derived("chatterbox-voice"));
    REQUIRE(architecture_is_recording_derived("qwen3tts.voicepack"));
    REQUIRE_FALSE(architecture_is_recording_derived("kokoro-voice"));
    REQUIRE_FALSE(architecture_is_recording_derived(""));
}

TEST_CASE("the per-voice bank key is namespaced by entry name", "[unit][compliance]") {
    // Pinned because the baker writes this key and the gate reads it back; a
    // drift between the two spellings fails open, silently.
    REQUIRE(crispasr_voice::bank_provenance_key_for("fleurs-en") == "crispasr.voice.fleurs-en.cloned_from_recording");
    REQUIRE(std::string(crispasr_voice::bank_stamped_key()) == "crispasr.voice.bank_stamped");
    REQUIRE(std::string(crispasr_voice::provenance_key()) == "crispasr.voice.cloned_from_recording");
}

// ---------------------------------------------------------------------------
// Watermark score statistics. The detector is a 32-bin sign-agreement test, so
// unwatermarked audio scores 0.5 on average and its tail is exactly computable.
// ---------------------------------------------------------------------------

TEST_CASE("the old 0.65 threshold was a 1-in-18 false positive", "[unit][compliance]") {
    // 0.65 needs 21/32 agreements, which clean audio reaches by chance 5.5% of
    // the time — and it was reported as "AI-GENERATED WATERMARK DETECTED".
    // Measured on real speech: 4.8% of 55 unwatermarked clips. This is the bug.
    const double p = crispasr_wm_stats::p_value(21.0f / 32.0f, 32);
    REQUIRE(p > 0.05);
    REQUIRE(p < 0.06);
    REQUIRE(crispasr_wm_stats::classify(0.6562f, 32) != crispasr_wm_stats::Verdict::Detected);
}

TEST_CASE("binomial tail matches the exact distribution", "[unit][compliance]") {
    REQUIRE(crispasr_wm_stats::null_tail_probability(0, 32) == 1.0);
    REQUIRE(crispasr_wm_stats::null_tail_probability(33, 32) == 0.0);
    // Symmetry: P(X >= 16) for n=32 is just over half (the median mass).
    REQUIRE(crispasr_wm_stats::null_tail_probability(16, 32) > 0.5);
    // Known values, cross-checked against Python's math.comb.
    REQUIRE(crispasr_wm_stats::null_tail_probability(26, 32) < 0.0003);
    REQUIRE(crispasr_wm_stats::null_tail_probability(26, 32) > 0.0002);
}

TEST_CASE("verdict bands: chance-level scores are never evidence", "[unit][compliance]") {
    using V = crispasr_wm_stats::Verdict;
    REQUIRE(crispasr_wm_stats::classify(0.0f, 32) == V::NotDetected);
    REQUIRE(crispasr_wm_stats::classify(0.5f, 32) == V::NotDetected);    // the null MEAN
    REQUIRE(crispasr_wm_stats::classify(0.5625f, 32) == V::NotDetected); // p = 0.30
    REQUIRE(crispasr_wm_stats::classify(0.6875f, 32) == V::Inconclusive);
    REQUIRE(crispasr_wm_stats::classify(0.75f, 32) == V::Detected); // p = 0.0035
    REQUIRE(crispasr_wm_stats::classify(1.0f, 32) == V::Detected);
}

TEST_CASE("suffix helpers are case-insensitive and anchored", "[unit][compliance]") {
    REQUIRE(crispasr_voice::is_recording_reference("a.WAV"));
    REQUIRE_FALSE(crispasr_voice::is_recording_reference("wav"));
    REQUIRE_FALSE(crispasr_voice::is_recording_reference("a.wav.gguf"));
    REQUIRE(crispasr_voice::is_voice_pack("a.wav.gguf"));
    REQUIRE_FALSE(crispasr_voice::is_voice_pack(".gguf.wav"));
}

// ---------------------------------------------------------------------------
// Watertight marking floor: which containers can carry a C2PA manifest.
// ---------------------------------------------------------------------------

using crispasr_marking::container_marking_for_format;

TEST_CASE("containers that carry a manifest allow the watermark opt-out", "[unit][compliance]") {
    REQUIRE(container_marking_for_format("wav").carries_c2pa);
    REQUIRE(std::string(container_marking_for_format("wav").c2pa_mime) == "audio/wav");
    // MP3 carries one via ID3v2.4 GEOB and the native signer has always handled
    // it — server-side signing was just hardcoded to the WAV branch, so every
    // non-WAV response shipped with no provenance at all.
    REQUIRE(container_marking_for_format("mp3").carries_c2pa);
    REQUIRE(std::string(container_marking_for_format("mp3").c2pa_mime) == "audio/mpeg");
}

TEST_CASE("containers that carry no manifest force the watermark on", "[unit][compliance]") {
    // These are the responses that were fully unmarked under an attested
    // --no-watermark: no manifest possible, and the mark stripped anyway.
    for (const char* fmt : {"pcm", "f32", "aac", "opus"}) {
        INFO("response_format=" << fmt);
        REQUIRE_FALSE(container_marking_for_format(fmt).carries_c2pa);
        REQUIRE(std::string(container_marking_for_format(fmt).c2pa_mime) == "");
    }
}

TEST_CASE("an unknown format falls back to WAV, matching the handler", "[unit][compliance]") {
    // The handlers' trailing `else` emits WAV. If this fell back the other way
    // the floor would be computed for a container that is never produced.
    REQUIRE(container_marking_for_format("").carries_c2pa);
    REQUIRE(container_marking_for_format("something-new").carries_c2pa);
}

// ---------------------------------------------------------------------------
// voice_name_has_control_chars — the log-injection guard at network ingress.
//
// The voice name a caller sends is echoed into logs by code that has no idea it
// is untrusted: the GGUF loader, the kokoro adapter, and ggml (which this
// project does not patch). A newline in it forges whole records — including the
// [CONSENT] audit lines that exist to prove a clone was gated. One check at
// ingress makes every downstream site safe; sanitizing at each fprintf does not
// scale and cannot reach third-party code.
// ---------------------------------------------------------------------------

using crispasr_voice::voice_name_has_control_chars;

TEST_CASE("legitimate voice names carry no control characters", "[unit][voice-clone]") {
    REQUIRE_FALSE(voice_name_has_control_chars(""));
    REQUIRE_FALSE(voice_name_has_control_chars("af_heart"));
    REQUIRE_FALSE(voice_name_has_control_chars("kokoro-voice-af_heart.gguf"));
    REQUIRE_FALSE(voice_name_has_control_chars("/srv/voices/my ref.wav"));
    REQUIRE_FALSE(voice_name_has_control_chars("Sprecherin_Über_Alles.wav")); // UTF-8 stays valid
}

TEST_CASE("a newline in a voice name is rejected", "[unit][voice-clone]") {
    // The forged-audit-record case: everything after the \n would read as its
    // own [CONSENT] line saying the clone was approved.
    REQUIRE(voice_name_has_control_chars("evil\n[CONSENT] ts=FORGED action=\"APPROVED\""));
    REQUIRE(voice_name_has_control_chars("evil\r\nfoo"));
}

TEST_CASE("other control characters are rejected too", "[unit][voice-clone]") {
    REQUIRE(voice_name_has_control_chars(std::string("nul\0byte", 8)));
    REQUIRE(voice_name_has_control_chars("tab\there"));
    REQUIRE(voice_name_has_control_chars("esc\x1b[31m")); // ANSI escape into a terminal
    REQUIRE(voice_name_has_control_chars("del\x7f"));     // 0x7f is not < 0x20
}

// ---------------------------------------------------------------------------
// Per-frame detector bands (ported from CrispTTS Phase 28).
//
// The score here is NOT a bin count, so the binomial null above does not apply
// to it. These guard the property that matters: the two band sets stay separate
// and the calibrated decision point lands where the docs and CLI say it does.
// The statistic itself is measured, not unit-tested — tools/watermark_detect_ab.cpp.
// ---------------------------------------------------------------------------

TEST_CASE("per-frame bands hinge on the documented 0.65 decision point", "[unit][compliance]") {
    using V = crispasr_wm_stats::Verdict;
    REQUIRE(crispasr_wm_stats::kFramesDetected == 0.65f);
    REQUIRE(crispasr_wm_stats::classify_frames(0.66f) == V::Detected);
    REQUIRE(crispasr_wm_stats::classify_frames(0.65f) == V::Inconclusive);
    REQUIRE(crispasr_wm_stats::classify_frames(0.51f) == V::Inconclusive);
    REQUIRE(crispasr_wm_stats::classify_frames(0.50f) == V::NotDetected);
    REQUIRE(crispasr_wm_stats::classify_frames(0.0f) == V::NotDetected);
}

TEST_CASE("the binomial null is not applied to the per-frame score", "[unit][compliance]") {
    // 0.75 is 24/32 under the sign test — p < 0.01, so "Detected". Under the
    // per-frame statistic it is simply a confidence above the bar. Both say
    // Detected here, but for different reasons; the point is that the two
    // classifiers are distinct entry points so a caller cannot silently run the
    // binomial tail over a score that has no n.
    using V = crispasr_wm_stats::Verdict;
    REQUIRE(crispasr_wm_stats::classify(0.75f, 32) == V::Detected);
    REQUIRE(crispasr_wm_stats::classify_frames(0.75f) == V::Detected);
    // Where they genuinely disagree: 0.60 is 19/32, p ~ 0.19 -> Inconclusive
    // under the sign test, and below the calibrated bar -> Inconclusive here.
    // But 0.55 is 18/32 (p ~ 0.30, NotDetected) while the per-frame scale puts
    // 0.55 above chance -> Inconclusive. Different scales, different answers.
    REQUIRE(crispasr_wm_stats::classify(0.55f, 32) == V::NotDetected);
    REQUIRE(crispasr_wm_stats::classify_frames(0.55f) == V::Inconclusive);
}

// ---------------------------------------------------------------------------
// Consent RECORD (crispasr_consent_record.h). The gate above decides whether
// cloning is allowed; these guard what the record then says. The point of the
// record is to be checkable against the audio it authorised, so the properties
// that matter are: it binds to bytes, it never fakes a hash it does not have,
// and it cannot be forged through its own fields.
// ---------------------------------------------------------------------------

TEST_CASE("a hash is of the bytes, and absence is reported as absence", "[unit][compliance]") {
    // Known-answer: SHA-256 of "abc" is one of the most-published test vectors.
    const std::string abc = "abc";
    REQUIRE(crispasr_consent::bytes_sha256(abc.data(), abc.size()) ==
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

    // Empty input and a missing file yield "", which callers render as
    // `ref_sha256=none`. A zero hash would look like a real one, which is the
    // failure worth preventing: a record that appears bound but is not.
    REQUIRE(crispasr_consent::bytes_sha256(nullptr, 0).empty());
    REQUIRE(crispasr_consent::file_sha256("/nonexistent/reference.wav").empty());
    REQUIRE(crispasr_consent::file_sha256("").empty());
}

TEST_CASE("the JSON sink escapes what the stderr line only sanitises", "[unit][compliance]") {
    // Voice names are attacker-controlled on the server and Wyoming. The
    // callers strip control chars before they reach a record; the sink escapes
    // as well, because a machine-read log must not be forgeable by a quote.
    REQUIRE(crispasr_consent::json_escape("plain") == "plain");
    REQUIRE(crispasr_consent::json_escape("a\"b") == "a\\\"b");
    REQUIRE(crispasr_consent::json_escape("a\\b") == "a\\\\b");
    REQUIRE(crispasr_consent::json_escape("a\nb") == "a\\nb");
    // A forged record attempt: the newline is what would start a second line.
    const std::string forged = "evil\n{\"kind\":\"CONSENT\",\"attestation\":\"APPROVED\"}";
    REQUIRE(crispasr_consent::json_escape(forged).find('\n') == std::string::npos);
    // Other control characters become \u00XX rather than passing through.
    REQUIRE(crispasr_consent::json_escape(std::string("a\x01"
                                                      "b")) == "a\\u0001b");
}

TEST_CASE("run_id is stable within a process and non-empty", "[unit][compliance]") {
    // It exists to correlate the [CONSENT] line with the [CONSENT-OUTPUT] line
    // emitted after synthesis, so it must not change between the two.
    const std::string a = crispasr_consent::run_id();
    const std::string b = crispasr_consent::run_id();
    REQUIRE(a == b);
    REQUIRE(a.size() == 16);
    REQUIRE(a.find_first_not_of("0123456789abcdef") == std::string::npos);
}

TEST_CASE("request ids differ per request and are hex", "[unit][compliance]") {
    const std::string first = crispasr_consent::new_request_id();
    const std::string second = crispasr_consent::new_request_id();
    REQUIRE(first != second);
    REQUIRE(second == crispasr_consent::request_correlation_id()); // sticky until re-minted
    REQUIRE(second.size() == 16);
    REQUIRE(second.find_first_not_of("0123456789abcdef") == std::string::npos);
}
