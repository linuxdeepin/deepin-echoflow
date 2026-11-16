// test-speaker-identity.cpp — whose voice is a PRESET voice, and which duty
// that triggers.
//
// The gap this guards: `is_clone == false` was doing two jobs at once. It
// correctly meant "this operator owes no consent attestation", and it
// incorrectly meant "there is nothing to disclose". A preset voice shipped
// inside a model can be an identifiable individual — a named donor, a corpus
// speaker such as VCTK's p225 — and Art. 3(60) attaches to the audio resembling
// that person, not to which pipeline produced it.
//
// Pure predicate, so it is guarded on the tier CI actually runs, for the same
// reason as test-marking-policy.cpp and test-voice-clone-policy.cpp.
//
// TWO KINDS OF TEST LIVE HERE, and the difference matters when one fails:
//
//   * The MECHANISM (first half) — three values, two duties, a resolution
//     order. Changing one of these is a design decision.
//   * The RESEARCHED VERDICTS (second half) — claims about specific shipped
//     models, each backed by the provider's own documentation. Changing one of
//     these is a research decision, and it should require someone to justify a
//     failing test rather than quietly editing a table.
//
// A model nobody has read stays Unknown in both halves. Pinning a verdict here
// that nobody researched would be the same mistake as guessing "synthetic" to
// quiet a warning: silent, and wrong in the direction that removes a duty.

#include "crispasr_speaker_identity.h"
#include "crispasr_speaker_identity_models.h"

#include <catch2/catch_test_macros.hpp>

using crispasr_voice::parse_speaker_identity;
using crispasr_voice::requires_consent_attestation;
using crispasr_voice::requires_spoken_disclosure;
using crispasr_voice::resolve_speaker_identity;
using crispasr_voice::should_warn_unknown_identity;
using crispasr_voice::SpeakerIdentity;

// ---------------------------------------------------------------------------
// The split: two duties, two triggers.
// ---------------------------------------------------------------------------

TEST_CASE("a real-person preset is disclosed", "[unit][compliance]") {
    // The bug, pinned. Not a clone — nothing went through a baker — but the
    // voice is a specific person, so the output is a deep fake and owes the
    // Art. 50(4) audible label.
    REQUIRE(requires_spoken_disclosure(/*is_clone=*/false, SpeakerIdentity::RealPerson));
}

TEST_CASE("a real-person preset is NOT consent-gated", "[unit][compliance]") {
    // Whether that donor agreed to the model being trained is a licensing
    // question settled upstream between them and whoever trained it. The
    // operator downstream cannot attest to it, so demanding --i-have-rights
    // would be theatre — and would break every documented preset example.
    REQUIRE_FALSE(requires_consent_attestation(/*is_clone=*/false, SpeakerIdentity::RealPerson));
}

TEST_CASE("cloning triggers both duties", "[unit][compliance]") {
    // The operator IS the one taking a specific person's voice here, so they
    // attest; and the output is a deep fake, so it is disclosed.
    REQUIRE(requires_consent_attestation(true, SpeakerIdentity::Unknown));
    REQUIRE(requires_spoken_disclosure(true, SpeakerIdentity::Unknown));
    // ...and the identity of a cloned voice cannot weaken either. A pack that
    // claims "synthetic" while being handed a real recording must not escape.
    REQUIRE(requires_spoken_disclosure(true, SpeakerIdentity::Synthetic));
    REQUIRE(requires_consent_attestation(true, SpeakerIdentity::Synthetic));
}

TEST_CASE("a synthetic preset owes neither", "[unit][compliance]") {
    // Nobody's voice, no deep fake. Art. 50(2) marking still applies and is
    // handled elsewhere — synthesis always watermarks.
    REQUIRE_FALSE(requires_spoken_disclosure(false, SpeakerIdentity::Synthetic));
    REQUIRE_FALSE(requires_consent_attestation(false, SpeakerIdentity::Synthetic));
}

TEST_CASE("unknown does not force a disclosure, but does warn", "[unit][compliance]") {
    // Treating unknown as real_person would prepend a spoken sentence to every
    // stock TTS voice in the project and teach operators to reach for
    // --no-spoken-disclaimer, which is worse than the disease. Treating it as
    // synthetic would silently assert the convenient answer on exactly the
    // models nobody has checked. So: no disclosure, but say so.
    REQUIRE_FALSE(requires_spoken_disclosure(false, SpeakerIdentity::Unknown));
    REQUIRE(should_warn_unknown_identity(false, SpeakerIdentity::Unknown));
}

TEST_CASE("a clone never warns about unknown identity", "[unit][compliance]") {
    // It is disclosed and gated regardless, so the question cannot change the
    // outcome and the warning would be pure noise on the one path that is
    // already fully handled.
    REQUIRE_FALSE(should_warn_unknown_identity(true, SpeakerIdentity::Unknown));
    // Nor does an answered question warn.
    REQUIRE_FALSE(should_warn_unknown_identity(false, SpeakerIdentity::Synthetic));
    REQUIRE_FALSE(should_warn_unknown_identity(false, SpeakerIdentity::RealPerson));
}

// ---------------------------------------------------------------------------
// Parsing — the values arrive from a CLI flag, a JSON field and GGUF metadata.
// ---------------------------------------------------------------------------

TEST_CASE("the three values parse, case- and space-tolerantly", "[unit][compliance]") {
    REQUIRE(parse_speaker_identity("real_person") == SpeakerIdentity::RealPerson);
    REQUIRE(parse_speaker_identity("REAL_PERSON") == SpeakerIdentity::RealPerson);
    REQUIRE(parse_speaker_identity("  Real_Person  ") == SpeakerIdentity::RealPerson);
    REQUIRE(parse_speaker_identity("synthetic") == SpeakerIdentity::Synthetic);
    REQUIRE(parse_speaker_identity("unknown") == SpeakerIdentity::Unknown);
}

TEST_CASE("an unset value is unknown and is NOT a typo", "[unit][compliance]") {
    // Absence is the normal case — most packs declare nothing — and must not
    // produce a warning telling the operator they mistyped something.
    bool recognised = false;
    REQUIRE(parse_speaker_identity("", &recognised) == SpeakerIdentity::Unknown);
    REQUIRE(recognised);
}

TEST_CASE("a typo is reported, not silently downgraded", "[unit][compliance]") {
    // "real-person" resolving quietly to unknown would drop the disclosure the
    // operator explicitly asked for — a wrong answer in the direction that
    // removes a duty, which is the direction that must never be silent.
    for (const char* bad : {"real-person", "human", "person", "REALPERSON", "true"}) {
        INFO("value=" << bad);
        bool recognised = true;
        REQUIRE(parse_speaker_identity(bad, &recognised) == SpeakerIdentity::Unknown);
        REQUIRE_FALSE(recognised);
    }
}

TEST_CASE("the round trip through to_string is stable", "[unit][compliance]") {
    // These strings are written into GGUF metadata by bakers and read back by
    // the gate; a drift between writer and reader fails open.
    for (auto id : {SpeakerIdentity::RealPerson, SpeakerIdentity::Synthetic, SpeakerIdentity::Unknown}) {
        REQUIRE(parse_speaker_identity(crispasr_voice::to_string(id)) == id);
    }
    REQUIRE(std::string(crispasr_voice::to_string(SpeakerIdentity::RealPerson)) == "real_person");
    REQUIRE(std::string(crispasr_voice::speaker_identity_key()) == "crispasr.voice.speaker_identity");
    REQUIRE(crispasr_voice::speaker_identity_key_for("af_heart") == "crispasr.voice.af_heart.speaker_identity");
}

// ---------------------------------------------------------------------------
// Resolution precedence: override > pack > backend > unknown.
// ---------------------------------------------------------------------------

TEST_CASE("the operator override outranks everything", "[unit][compliance]") {
    // They may have read the model card the pack was built before anyone wrote.
    REQUIRE(resolve_speaker_identity(SpeakerIdentity::RealPerson, SpeakerIdentity::Synthetic,
                                     SpeakerIdentity::Synthetic) == SpeakerIdentity::RealPerson);
    // ...and it must move the answer in BOTH directions. Someone who knows a
    // "real_person" label is wrong has to be able to say so, or the flag is
    // only usable for adding duties and gets ignored for the other half.
    REQUIRE(resolve_speaker_identity(SpeakerIdentity::Synthetic, SpeakerIdentity::RealPerson,
                                     SpeakerIdentity::RealPerson) == SpeakerIdentity::Synthetic);
}

TEST_CASE("the pack outranks the backend default", "[unit][compliance]") {
    // The backend default describes its built-in voices; a pack knows about
    // itself, and is the more specific claim.
    REQUIRE(resolve_speaker_identity(SpeakerIdentity::Unknown, SpeakerIdentity::RealPerson,
                                     SpeakerIdentity::Synthetic) == SpeakerIdentity::RealPerson);
}

TEST_CASE("the backend default fills in when nothing else speaks", "[unit][compliance]") {
    REQUIRE(resolve_speaker_identity(SpeakerIdentity::Unknown, SpeakerIdentity::Unknown, SpeakerIdentity::RealPerson) ==
            SpeakerIdentity::RealPerson);
}

TEST_CASE("nothing declared resolves to unknown", "[unit][compliance]") {
    // The default state of the project today: the mechanism ships before the
    // research does, and claims nothing until a model card has been read.
    REQUIRE(resolve_speaker_identity(SpeakerIdentity::Unknown, SpeakerIdentity::Unknown, SpeakerIdentity::Unknown) ==
            SpeakerIdentity::Unknown);
}

// ---------------------------------------------------------------------------
// The once-per-model warning.
// ---------------------------------------------------------------------------

TEST_CASE("the unknown-identity warning fires once per model", "[unit][compliance]") {
    crispasr_voice::reset_unknown_identity_warnings_for_test();
    REQUIRE(crispasr_voice::claim_unknown_identity_warning("kokoro"));
    REQUIRE_FALSE(crispasr_voice::claim_unknown_identity_warning("kokoro"));
    // Keyed per model, not per process: a server may load several backends and
    // each unanswered one is a separate thing the operator needs to know.
    REQUIRE(crispasr_voice::claim_unknown_identity_warning("piper"));
    REQUIRE_FALSE(crispasr_voice::claim_unknown_identity_warning("piper"));
}

TEST_CASE("the test reset actually clears the state", "[unit][compliance]") {
    // Guarding the guard: an earlier draft kept the set and the reset in
    // separate function-local statics, so the reset compiled, ran, and did
    // nothing — which would have made every case above order-dependent.
    crispasr_voice::reset_unknown_identity_warnings_for_test();
    REQUIRE(crispasr_voice::claim_unknown_identity_warning("same-model"));
    crispasr_voice::reset_unknown_identity_warnings_for_test();
    REQUIRE(crispasr_voice::claim_unknown_identity_warning("same-model"));
}

TEST_CASE("the warning names what to do about it", "[unit][compliance]") {
    // A warning an operator cannot act on is one they learn to ignore, and this
    // one fires on every unresearched preset backend in the project.
    const std::string w = crispasr_voice::unknown_identity_warning("kokoro");
    REQUIRE(w.find("kokoro") != std::string::npos);
    REQUIRE(w.find("--speaker-identity real_person") != std::string::npos);
    REQUIRE(w.find("--speaker-identity synthetic") != std::string::npos);
    REQUIRE(w.find("Art. 50(4)") != std::string::npos);
    // An unnamed model still produces a usable line rather than a dangling quote.
    REQUIRE(crispasr_voice::unknown_identity_warning("").find("<unknown-model>") != std::string::npos);
}

// ---------------------------------------------------------------------------
// The researched verdicts (crispasr_speaker_identity_models.h).
//
// Unlike the mechanism above, these ARE claims about specific shipped models,
// each backed by the provider's own documentation. They are pinned here so a
// verdict cannot be changed silently: flipping one is a research decision and
// should show up as a failing test that someone has to justify.
// ---------------------------------------------------------------------------

using crispasr_voice::identity_for_model;

TEST_CASE("piper voices are named human donors", "[unit][compliance]") {
    // Every piper voice in the registry is one identifiable person: the Lessac
    // corpus, Thorsten Müller, Kerstin. Releasing your voice under CC0 is
    // consent to publish — it is not a reason to stop telling listeners the
    // audio is synthetic.
    REQUIRE(identity_for_model("piper", "piper-en_US-lessac-medium-f16.gguf") == SpeakerIdentity::RealPerson);
    REQUIRE(identity_for_model("piper", "piper-de_DE-thorsten-medium-f16.gguf") == SpeakerIdentity::RealPerson);
    REQUIRE(identity_for_model("piper", "/models/piper-de_DE-kerstin-low-f16.gguf") == SpeakerIdentity::RealPerson);
}

TEST_CASE("a kokoro CHECKPOINT declares nothing about whose voice it is", "[unit][compliance]") {
    // A kokoro model is a backbone, not a voice — CrispASR's own card says so:
    // "This is a base model, not a voice. It pairs with a German voice pack."
    // Which person you hear is decided entirely by the style pack, so the model
    // must claim nothing. An earlier revision returned Synthetic here, which
    // under strongest-duty resolution would have OVERRIDDEN a real-person pack.
    REQUIRE(identity_for_model("kokoro", "kokoro-82m-q8_0.gguf") == SpeakerIdentity::Unknown);
    REQUIRE(identity_for_model("kokoro", "kokoro-de-hui-base-q8_0.gguf") == SpeakerIdentity::Unknown);
}

TEST_CASE("the kokoro German HUI backbone is NOT inherited as synthetic", "[unit][compliance]") {
    // The conflict this project found while porting CrispTTS's research.
    // CrispTTS marks kokoro synthetic, which is right for the English packs.
    // But CrispASR's German backbone is trained on HUI-Audio-Corpus-German,
    // whose narrators are NAMED — the same corpus CrispTTS itself cites when
    // marking the NeMo FastPitch German model real_person. Inheriting
    // "synthetic" here would assume the answer on the one variant there is a
    // reason to doubt. Held at Unknown until someone reads it.
    REQUIRE(identity_for_model("kokoro", "kokoro-de-hui-base-q8_0.gguf") == SpeakerIdentity::Unknown);
}

TEST_CASE("orpheus is decided per checkpoint, not per backend", "[unit][compliance]") {
    // One backend, several models, different answers — which is why the table
    // is keyed on (backend, checkpoint) and not on the backend alone.
    REQUIRE(identity_for_model("orpheus", "kartoffel-orpheus-de-natural-q8_0.gguf") == SpeakerIdentity::RealPerson);
    // Canopy Labs disclose 100k+ h of "permissive" audio and nothing about the
    // origin of tara/leah/jess/leo/dan/mia/zac/zoe.
    REQUIRE(identity_for_model("orpheus", "orpheus-3b-0.1-ft-q8_0.gguf") == SpeakerIdentity::Unknown);
    // lex-au's German fine-tune: no training-data documentation.
    REQUIRE(identity_for_model("orpheus", "Orpheus-3b-German-FT-Q8_0.gguf") == SpeakerIdentity::Unknown);
    // The kartoffel SYNTHETIC variant: a genuinely different checkpoint,
    // "trained on synthetic German speech with explicit emotion and outburst
    // control". Read, not inferred from the repo name — a name is a convention,
    // not a provenance statement, and this is the direction where being wrong
    // removes a disclosure.
    REQUIRE(identity_for_model("orpheus", "kartoffel-orpheus-de-synthetic-q8_0.gguf") == SpeakerIdentity::Synthetic);
}

TEST_CASE("matching is case-insensitive and path-tolerant", "[unit][compliance]") {
    REQUIRE(identity_for_model("orpheus", "/models/KARTOFFEL-ORPHEUS-DE-NATURAL-q8_0.gguf") ==
            SpeakerIdentity::RealPerson);
    REQUIRE(identity_for_model("piper", "/a/b/c/PIPER-de_DE-thorsten-high-f16.gguf") == SpeakerIdentity::RealPerson);
}

TEST_CASE("a renamed checkpoint fails SAFE, to unknown", "[unit][compliance]") {
    // The cost of matching on a file name, stated. A rename cannot turn
    // real_person into synthetic — it can only turn a known answer back into a
    // question, which warns. The durable fix is a GGUF stamp.
    REQUIRE(identity_for_model("orpheus", "my-model.gguf") == SpeakerIdentity::Unknown);
    REQUIRE(identity_for_model("orpheus", "") == SpeakerIdentity::Unknown);
}

TEST_CASE("single-donor backends are real people", "[unit][compliance]") {
    // fastpitch  nvidia/tts_en_fastpitch, "trained on LJSpeech" — 13,100 clips
    //            of ONE LibriVox narrator, Linda Johnson.
    // bananamind Banaxi-Tech card: en-us on LJSpeech (Linda Johnson again),
    //            de-de on ThorstenVoice, credited "Voice: Thorsten Müller" —
    //            the same donor as piper-de_DE-thorsten, reached twice.
    // parler-tts LibriTTS-R + MLS, both LibriVox-derived, with 34 speakers
    //            "characterized by name" — pseudonymous corpus readers, the
    //            VCTK p225 case.
    REQUIRE(identity_for_model("fastpitch", "fastpitch-en-q8_0.gguf") == SpeakerIdentity::RealPerson);
    REQUIRE(identity_for_model("bananamind-tts", "bananamind-tts-en-q8_0.gguf") == SpeakerIdentity::RealPerson);
    REQUIRE(identity_for_model("bananamind-tts", "bananamind-tts-de-q8_0.gguf") == SpeakerIdentity::RealPerson);
    REQUIRE(identity_for_model("parler-tts", "parler-tts-mini-v1.1-q8_0.gguf") == SpeakerIdentity::RealPerson);
}

TEST_CASE("csm has no preset persona", "[unit][compliance]") {
    // sesame/csm-1b: "a base generation model ... it has not been fine-tuned on
    // any specific voice". Its cloning path takes a context recording as
    // --voice, which the CLONE gate catches, not this table.
    REQUIRE(identity_for_model("csm", "csm-1b-q4_k.gguf") == SpeakerIdentity::Synthetic);
}

TEST_CASE("genuinely undocumented backends stay unknown", "[unit][compliance]") {
    // Not from lack of looking — these providers do not say. Recorded so the
    // check is not repeated, and specifically NOT downgraded to "synthetic",
    // which would silently remove a disclosure.
    //
    // bark     Suno's README documents 100+ presets and a prompt library and
    //          says nothing about their provenance. The "fully synthetic"
    //          phrasing circulating for it is third-party summary, not Suno's.
    // melotts  MyShell's card carries no training-data statement at all.
    // speecht5 Structurally unanswerable per model: the voice is a 512-d
    //          x-vector the OPERATOR supplies via --voice.
    for (const char* b : {"bark", "melotts", "speecht5"}) {
        INFO("backend=" << b);
        REQUIRE(identity_for_model(b, "whatever-q8_0.gguf") == SpeakerIdentity::Unknown);
    }
}

TEST_CASE("an unknown backend name is unknown, not a crash", "[unit][compliance]") {
    REQUIRE(identity_for_model("", "") == SpeakerIdentity::Unknown);
    REQUIRE(identity_for_model("some-future-backend", "x.gguf") == SpeakerIdentity::Unknown);
}

// ---------------------------------------------------------------------------
// Combining declared sources: STRONGEST duty wins, the override is absolute.
//
// The non-obvious rule, and the one worth pinning. The pack, the model stamp
// and the researched table are independent claims about the same fact, not
// versions of one claim — any of the three can be the only one that has heard
// of a given voice. Precedence between them would let a stale "synthetic" stamp
// silently cancel a researched real_person verdict, which is the exact failure
// that costs a disclosure.
// ---------------------------------------------------------------------------

TEST_CASE("a model stamp can upgrade an unresearched backend", "[unit][compliance]") {
    // The point of stamping: a checkpoint carries its own answer and stops
    // depending on file-name matching.
    REQUIRE(resolve_speaker_identity(SpeakerIdentity::Unknown, SpeakerIdentity::Unknown, SpeakerIdentity::Unknown,
                                     /*model=*/SpeakerIdentity::RealPerson) == SpeakerIdentity::RealPerson);
}

TEST_CASE("no declared source can silently downgrade another", "[unit][compliance]") {
    // A stamp saying "synthetic" on a checkpoint the table knows is a real
    // person must not remove the disclosure. Disagreement fails toward
    // disclosing.
    REQUIRE(resolve_speaker_identity(SpeakerIdentity::Unknown, SpeakerIdentity::Unknown,
                                     /*backend=*/SpeakerIdentity::RealPerson,
                                     /*model=*/SpeakerIdentity::Synthetic) == SpeakerIdentity::RealPerson);
    // ...in every direction between the three declared sources.
    REQUIRE(resolve_speaker_identity(SpeakerIdentity::Unknown, /*pack=*/SpeakerIdentity::Synthetic,
                                     /*backend=*/SpeakerIdentity::Unknown,
                                     /*model=*/SpeakerIdentity::RealPerson) == SpeakerIdentity::RealPerson);
    REQUIRE(resolve_speaker_identity(SpeakerIdentity::Unknown, /*pack=*/SpeakerIdentity::RealPerson,
                                     /*backend=*/SpeakerIdentity::Synthetic,
                                     /*model=*/SpeakerIdentity::Synthetic) == SpeakerIdentity::RealPerson);
}

TEST_CASE("synthetic still beats unknown", "[unit][compliance]") {
    // "Strongest duty" is not "always real_person": a source that positively
    // says synthetic resolves the question and silences the warning.
    REQUIRE(resolve_speaker_identity(SpeakerIdentity::Unknown, SpeakerIdentity::Unknown, SpeakerIdentity::Unknown,
                                     /*model=*/SpeakerIdentity::Synthetic) == SpeakerIdentity::Synthetic);
    REQUIRE_FALSE(should_warn_unknown_identity(
        false, resolve_speaker_identity(SpeakerIdentity::Unknown, SpeakerIdentity::Unknown, SpeakerIdentity::Unknown,
                                        SpeakerIdentity::Synthetic)));
}

TEST_CASE("the operator override still overrules every declared source", "[unit][compliance]") {
    // Rule 1 is the escape hatch for a wrong strong claim, and it has to keep
    // working downward or a mis-stamped model becomes unfixable without
    // re-converting it.
    REQUIRE(resolve_speaker_identity(/*override=*/SpeakerIdentity::Synthetic, SpeakerIdentity::RealPerson,
                                     SpeakerIdentity::RealPerson,
                                     SpeakerIdentity::RealPerson) == SpeakerIdentity::Synthetic);
    REQUIRE(resolve_speaker_identity(/*override=*/SpeakerIdentity::RealPerson, SpeakerIdentity::Synthetic,
                                     SpeakerIdentity::Synthetic,
                                     SpeakerIdentity::Synthetic) == SpeakerIdentity::RealPerson);
}

TEST_CASE("duty_rank orders the values", "[unit][compliance]") {
    REQUIRE(crispasr_voice::duty_rank(SpeakerIdentity::RealPerson) >
            crispasr_voice::duty_rank(SpeakerIdentity::Synthetic));
    REQUIRE(crispasr_voice::duty_rank(SpeakerIdentity::Synthetic) >
            crispasr_voice::duty_rank(SpeakerIdentity::Unknown));
}

// ---------------------------------------------------------------------------
// Voice-pack verdicts. kokoro's answer lives here, not on the checkpoint.
// ---------------------------------------------------------------------------

using crispasr_voice::identity_for_voice_pack;

TEST_CASE("the HUI-narrator kokoro packs are real people", "[unit][compliance]") {
    // The resolution of the conflict that held kokoro-de at unknown. df_eva and
    // dm_bernd are per-speaker style packs from HUI-Audio-Corpus-German — a
    // librivox.org-derived corpus — and they carry the narrators' own names.
    // Eva and Bernd are the same donors CrispTTS cites for its real_person NeMo
    // FastPitch German verdict.
    REQUIRE(identity_for_voice_pack("kokoro-voice-df_eva.gguf") == SpeakerIdentity::RealPerson);
    REQUIRE(identity_for_voice_pack("kokoro-voice-dm_bernd.gguf") == SpeakerIdentity::RealPerson);
}

TEST_CASE("the kikiri kokoro packs are synthetic, on their base's evidence", "[unit][compliance]") {
    // Person-shaped names ("Victoria Asztaller (synthetic)", "Martin Harbecke
    // (synthetic)") over kikiri-german-base-51speakers-synthetic, whose card
    // states: "Trained entirely on synthetic (TTS-generated) audio". The name
    // is not the evidence; the base's training data is.
    REQUIRE(identity_for_voice_pack("kokoro-voice-df_victoria.gguf") == SpeakerIdentity::Synthetic);
    REQUIRE(identity_for_voice_pack("kokoro-voice-dm_martin.gguf") == SpeakerIdentity::Synthetic);
}

TEST_CASE("hexgrad's own kokoro packs are designed voices", "[unit][compliance]") {
    REQUIRE(identity_for_voice_pack("kokoro-voice-af_heart.gguf") == SpeakerIdentity::Synthetic);
    REQUIRE(identity_for_voice_pack("kokoro-voice-ef_dora.gguf") == SpeakerIdentity::Synthetic);
    REQUIRE(identity_for_voice_pack("kokoro-voice-ff_siwis.gguf") == SpeakerIdentity::Synthetic);
}

TEST_CASE("an unrecognised pack falls through to unknown", "[unit][compliance]") {
    REQUIRE(identity_for_voice_pack("") == SpeakerIdentity::Unknown);
    REQUIRE(identity_for_voice_pack("somebody-elses-pack.gguf") == SpeakerIdentity::Unknown);
}

TEST_CASE("the German cascade crosses an identity boundary", "[unit][compliance]") {
    // Worth pinning as one case because it is the practical consequence: the
    // documented German fallback order is df_victoria -> df_eva -> ff_siwis, so
    // a missing default silently moves the user from a synthetic voice to a
    // real HUI narrator. The disclosure has to follow the pack, not the run.
    REQUIRE(identity_for_voice_pack("kokoro-voice-df_victoria.gguf") == SpeakerIdentity::Synthetic);
    REQUIRE(identity_for_voice_pack("kokoro-voice-df_eva.gguf") == SpeakerIdentity::RealPerson);
    REQUIRE(requires_spoken_disclosure(/*is_clone=*/false, identity_for_voice_pack("kokoro-voice-df_eva.gguf")));
    REQUIRE_FALSE(
        requires_spoken_disclosure(/*is_clone=*/false, identity_for_voice_pack("kokoro-voice-df_victoria.gguf")));
}


// ---------------------------------------------------------------------------
// Architecture -> backend name, for callers with a file and no session.
//
// --print-speaker-identity inspects a FILE. It has no CrispasrBackend to ask
// for name(), so it derives one from what the file declares. Without this it
// consulted only the voice-pack table and reported `unknown` for every piper,
// fastpitch and bananamind MODEL — silently, and the bulk stamper then skipped
// all of them. Caught by running the verb against the real published repos
// before uploading, not by reading the code.
// ---------------------------------------------------------------------------

using crispasr_voice::backend_for_architecture;

TEST_CASE("architectures that already match the backend name pass through", "[unit][compliance]") {
    REQUIRE(backend_for_architecture("piper") == "piper");
    REQUIRE(backend_for_architecture("fastpitch") == "fastpitch");
    REQUIRE(backend_for_architecture("orpheus") == "orpheus");
    REQUIRE(backend_for_architecture("kokoro-voice") == "kokoro-voice");
}

TEST_CASE("the architectures that differ are mapped", "[unit][compliance]") {
    // Underscore vs hyphen, and a -tts suffix. Listed explicitly rather than
    // normalised with a rule, so a future mismatch is a visible edit.
    REQUIRE(backend_for_architecture("bananamind_tts") == "bananamind-tts");
    REQUIRE(backend_for_architecture("csm-tts") == "csm");
}

TEST_CASE("a mapped architecture reaches its verdict", "[unit][compliance]") {
    // The join that was broken: arch -> backend -> verdict. Asserting the map
    // alone would not have caught it, because the map was not there at all.
    REQUIRE(identity_for_model(backend_for_architecture("piper"), "piper-de_DE-eva_k-x_low-f16.gguf") ==
            SpeakerIdentity::RealPerson);
    REQUIRE(identity_for_model(backend_for_architecture("bananamind_tts"), "bananamind-tts-de-q8_0.gguf") ==
            SpeakerIdentity::RealPerson);
    REQUIRE(identity_for_model(backend_for_architecture("csm-tts"), "csm-1b-q4_k.gguf") == SpeakerIdentity::Synthetic);
    // ...and an unmapped architecture stays unknown rather than crashing.
    REQUIRE(identity_for_model(backend_for_architecture("some-future-arch"), "x.gguf") == SpeakerIdentity::Unknown);
    REQUIRE(backend_for_architecture("").empty());
}
