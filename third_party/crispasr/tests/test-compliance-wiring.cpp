// test-compliance-wiring.cpp — the EU AI Act gates are WIRED UP on every
// surface that can reach them.
//
// The pure predicates are guarded elsewhere: test-voice-clone-policy.cpp knows
// what a clone is, test-marking-policy.cpp knows what a container-less surface
// must do. Both were green throughout every failure this file exists for,
// because none of those failures were in the predicate. They were in the
// plumbing — a surface that never called the gate, a baker that never stamped
// the pack the gate reads, an upload endpoint nobody had added a gate to.
//
// The pattern, four times now:
//
//   #172  Wyoming shipped for four releases marking nothing at all — it was
//         never added to the surface list the compliance work walked.
//   9a91e4 chatterbox clones only through a baked .gguf, so a `.wav` suffix
//         test could never see the headline cloning backend.
//   a66fc2 --make-ref built a reusable voiceprint with no attestation asked
//         anywhere, because it returned before the TTS block's gate.
//   (here) cosyvoice3 keeps every voice inside one bundle and selects by name,
//         so --voice named no file, no metadata was read, and a zero-shot
//         voice clone scored as a preset on CLI, server, Wyoming and ABI.
//
// So this is a SOURCE-level guard, deliberately, and it tests the joins rather
// than the logic. The behavioural version needs models and sockets, which puts
// it in the live tier CI does not run — and a compliance gate nobody can run on
// the tier CI actually runs is a gate that ships wrong (#312's lesson).
//
// It is coarse by design: it cannot prove the gate is called correctly, only
// that the call is still there. That is exactly the failure mode above.

#include <catch2/catch_test_macros.hpp>

#include <fstream>
#include <sstream>
#include <string>

#ifndef CRISPASR_SOURCE_DIR
#error "CRISPASR_SOURCE_DIR must be defined by the build"
#endif

namespace {

std::string read_file(const std::string& rel) {
    const std::string path = std::string(CRISPASR_SOURCE_DIR) + "/" + rel;
    std::ifstream f(path);
    INFO("reading " << rel);
    REQUIRE(f.good());
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

bool contains(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

} // namespace

// ---------------------------------------------------------------------------
// Voice BANKS reach the clone gate on every surface.
//
// cosyvoice3's voices live in one voices.gguf, discovered as a sibling of the
// model, and --voice selects an entry by name. Only the backend knows which
// bundle it resolved, so it has to hand the path to the gate; without that the
// gate sees an unresolvable bare name and returns "preset" for a voice clone.
// ---------------------------------------------------------------------------

TEST_CASE("the backend base class offers a voice-bank hook", "[unit][compliance]") {
    const std::string src = read_file("examples/cli/crispasr_backend.h");
    REQUIRE(contains(src, "virtual std::string voice_bank_path() const"));
}

TEST_CASE("cosyvoice3 hands its voice bundle to the gate", "[unit][compliance]") {
    const std::string src = read_file("examples/cli/crispasr_backend_cosyvoice3.cpp");
    // Overrides the hook...
    REQUIRE(contains(src, "std::string voice_bank_path() const override"));
    // ...and actually records the bundle it resolved, rather than returning a
    // member nothing ever assigns. That would compile, pass a naive test, and
    // leave the gate exactly as blind as before.
    REQUIRE(contains(src, "voices_path_ = voices_path;"));
}

TEST_CASE("every synthesis surface passes the bank to classify_voice", "[unit][compliance]") {
    // Four surfaces, one gate. The Wyoming lesson is that a surface inherits no
    // compliance behaviour — it has to be wired up, and only a check like this
    // will say whether it was.
    SECTION("CLI") {
        REQUIRE(contains(read_file("examples/cli/crispasr_run.cpp"), "backend->voice_bank_path()"));
    }
    SECTION("HTTP server") {
        REQUIRE(contains(read_file("examples/cli/crispasr_server.cpp"), "backend->voice_bank_path()"));
    }
    SECTION("Wyoming") {
        REQUIRE(contains(read_file("examples/cli/wyoming.cpp"), "g_backend->voice_bank_path()"));
    }
    SECTION("C ABI") {
        // No CrispasrBackend here — the session resolves the bundle itself.
        const std::string src = read_file("src/crispasr_c_api.cpp");
        REQUIRE(contains(src, "s->cosyvoice3_voices_path = std::move(cv3_voices);"));
        REQUIRE(contains(src, "s->cosyvoice3_voices_path)"));
    }
}

// ---------------------------------------------------------------------------
// Whose voice a PRESET voice is reaches the disclosure decision, on every
// surface.
//
// `is_clone == false` used to mean both "no consent owed" (right) and "nothing
// to disclose" (wrong). A preset shipped inside a model can be an identifiable
// person, and Art. 3(60) does not care which pipeline produced the audio. The
// mechanism is guarded in test-speaker-identity.cpp; this is the plumbing.
// ---------------------------------------------------------------------------

TEST_CASE("the backend base class resolves identity per checkpoint", "[unit][compliance]") {
    const std::string src = read_file("examples/cli/crispasr_backend.h");
    // Takes the model path, because one backend serves several checkpoints with
    // different answers — `orpheus` runs both Canopy's base model and
    // Kartoffel's German fine-tune. A no-argument hook cannot tell them apart
    // and would have to pick one verdict for both.
    REQUIRE(contains(src, "virtual crispasr_voice::SpeakerIdentity declared_speaker_identity(const std::string& "
                          "model_path) const"));
    REQUIRE(contains(src, "crispasr_voice::identity_for_model(name(), model_path)"));
}

TEST_CASE("every surface passes the actual checkpoint to the lookup", "[unit][compliance]") {
    // Passing an empty string here would compile, resolve every model to
    // Unknown, and look exactly like "nothing is researched yet" — a fix that
    // ships inert. Pin the argument.
    REQUIRE(contains(read_file("examples/cli/crispasr_run.cpp"), "declared_speaker_identity(params.model)"));
    REQUIRE(contains(read_file("examples/cli/crispasr_server.cpp"), "declared_speaker_identity(params.model)"));
    REQUIRE(contains(read_file("examples/cli/wyoming.cpp"), "declared_speaker_identity(g_params.model)"));
}

TEST_CASE("the model stamp reaches the resolver on every surface", "[unit][compliance]") {
    // The durable half of the identity answer. A reader nothing consults is the
    // inert-fix failure mode (#324), so pin the call at each join.
    REQUIRE(contains(read_file("examples/cli/crispasr_run.cpp"),
                     "crispasr_voice::read_model_speaker_identity(params.model)"));
    REQUIRE(contains(read_file("examples/cli/crispasr_server.cpp"),
                     "crispasr_voice::read_model_speaker_identity(params.model)"));
    REQUIRE(
        contains(read_file("examples/cli/wyoming.cpp"), "crispasr_voice::read_model_speaker_identity(g_params.model)"));
}

TEST_CASE("a converter can write the stamp the runtime reads", "[unit][compliance]") {
    // A read path with no writer is a feature nobody can use. The key spelling
    // has to match on both sides or the whole thing fails open, silently.
    const std::string conv = read_file("models/convert-orpheus-to-gguf.py");
    REQUIRE(contains(conv, "--speaker-identity"));
    // The WRITE CALL, not the key name: the flag's own help text quotes the key
    // too, so matching the bare string stayed green with the add_string()
    // deleted. Fourth time the red-proof has caught a guard of mine matching
    // prose instead of code — the pattern is "assert the token that only exists
    // when the behaviour does".
    REQUIRE(contains(conv, "w.add_string(\"crispasr.voice.speaker_identity\", args.speaker_identity)"));
    // Never written as a guess: an omitted or "unknown" value writes no key.
    REQUIRE(contains(conv, "if args.speaker_identity and args.speaker_identity != \"unknown\":"));
}

TEST_CASE("the diff harness marks the audio it writes", "[unit][compliance]") {
    // Found by re-enumerating emitters from code rather than from the surface
    // table (rule 4). crispasr-diff is an INSTALLED binary, and
    // CRISPASR_CSM_WAV_OUT makes it write a real, playable WAV of synthesized
    // speech straight from csm_tts_diag_synth_wav() — a hand-rolled RIFF writer
    // that never touched the marking path.
    //
    // It is a developer diagnostic, which is exactly the reasoning that left
    // the Wyoming server marking nothing for four releases. The file does not
    // know what you meant to use it for.
    const std::string src = read_file("src/csm_tts.cpp");
    REQUIRE(contains(src, "crispasr_watermark_embed_impl(pcm, n);"));
    // Before the peak-normalise, so the mark scales with the signal instead of
    // being applied on top of a rescaled one.
    const size_t mark = src.find("crispasr_watermark_embed_impl(pcm, n);");
    const size_t norm = src.find("Peak-normalise to 0.95");
    REQUIRE(mark != std::string::npos);
    REQUIRE(norm != std::string::npos);
    REQUIRE(mark < norm);
}

TEST_CASE("every binding can answer the speaker-identity question", "[unit][compliance]") {
    // The mechanism shipped reachable only from the C ABI: all ten bindings
    // could HEAR the Art. 50(4) reminder for a real-person preset and none of
    // them could answer it. Same shape as the earlier gap where a binding could
    // opt out of marking (synthesize_raw) but had no way to mark the result —
    // a duty you can be told about and cannot discharge is not a mechanism.
    struct Binding {
        const char* file;
        const char* needle;
    };
    static const Binding kBindings[] = {
        {"python/crispasr/_binding.py", "def set_speaker_identity(self, identity: str)"},
        {"bindings/go/crispasr_session.go", "func (s *CrispasrSession) SetSpeakerIdentity(identity string) error"},
        {"crispasr/src/lib.rs", "pub fn set_speaker_identity(&self, identity: &str)"},
        {"crispasr-sys/src/lib.rs", "pub fn crispasr_session_set_speaker_identity("},
        {"bindings/ruby/ext/ruby_crispasr_session.c", "\"set_speaker_identity\", rb_session_set_speaker_identity"},
        {"bindings/java/src/main/java/io/github/ggerganov/whispercpp/CrispasrSession.java",
         "public void setSpeakerIdentity(String identity)"},
        {"bindings/csharp/CrispASR/Session.cs", "public void SetSpeakerIdentity(string identity)"},
        {"flutter/crispasr/lib/src/crispasr.dart", "void setSpeakerIdentity(String identity)"},
        {"bindings/javascript/emscripten.cpp", "\"ttsSetSpeakerIdentity\""},
    };
    for (const auto& b : kBindings) {
        INFO("binding=" << b.file);
        REQUIRE(contains(read_file(b.file), b.needle));
    }
}

TEST_CASE("bindings surface a bad value instead of swallowing it", "[unit][compliance]") {
    // The ABI returns -2 for an unrecognised value precisely so a binding can
    // report it. A binding that ignores the code turns "real-person" (a typo
    // for real_person) into "unknown" — silently dropping the disclosure the
    // integrator was trying to declare.
    //
    // Matched on each binding's ERROR MESSAGE, not on "-2": that literal
    // already appears 5x in _binding.py and 4x in the Go file for unrelated
    // reasons, so a `contains("if rc == -2:")` guard stayed green with this
    // check gutted. The red-proof caught it. Assert the token that exists only
    // when the behaviour does.
    // Python and Go assert the CONDITION PAIRED WITH the message. Asserting the
    // message alone was still too weak: flipping `-2` to another code leaves the
    // message present but unreachable, and the guard stayed green. Two rounds of
    // red-proof to get this one honest.
    REQUIRE(contains(read_file("python/crispasr/_binding.py"),
                     "if rc == -2:\n            raise ValueError(\n"
                     "                f\"unrecognised speaker_identity {identity!r}; \""));
    REQUIRE(contains(read_file("bindings/go/crispasr_session.go"),
                     "case -2:\n\t\treturn fmt.Errorf(\"unrecognised speaker_identity %q "
                     "(expected real_person, synthetic or unknown)\", identity)"));
    REQUIRE(contains(read_file("crispasr/src/lib.rs"),
                     "unrecognised speaker_identity {identity:?} (expected real_person, synthetic or unknown)"));
    REQUIRE(contains(read_file("bindings/ruby/ext/ruby_crispasr_session.c"),
                     "unrecognised speaker_identity (expected real_person, synthetic or unknown)"));
    REQUIRE(contains(read_file("bindings/java/src/main/java/io/github/ggerganov/whispercpp/CrispasrSession.java"),
                     "(expected real_person, synthetic or unknown)"));
    REQUIRE(contains(read_file("bindings/csharp/CrispASR/Session.cs"), "(expected real_person, synthetic or unknown)"));
    REQUIRE(contains(read_file("flutter/crispasr/lib/src/crispasr.dart"), "'real_person', 'synthetic' or 'unknown'"));
}

TEST_CASE("the verdict has one source, queryable from the binary", "[unit][compliance]") {
    // --print-speaker-identity resolves a file with the SAME code the
    // disclosure gate uses, so a packaging script never has to restate a
    // verdict. That matters more than it looks: the verdicts already live in
    // C++, in GGUF stamps and in the docs, and a fourth copy in a shell script
    // is the one that silently drifts.
    const std::string run = read_file("examples/cli/crispasr_run.cpp");
    REQUIRE(contains(run, "params.print_speaker_identity_file"));
    REQUIRE(contains(run, "crispasr_voice::read_model_speaker_identity(path)"));
    REQUIRE(contains(run, "crispasr_voice::identity_for_voice_pack(path)"));

    // It has to be ROUTED before the "no input files" guard — it inspects a
    // file, not a session. Missing that made the verb exit 2 on first run.
    REQUIRE(
        contains(read_file("examples/cli/cli.cpp"),
                 "if (!params.print_speaker_identity_file.empty()) {\n        return crispasr_run_backend(params);"));

    // The bulk stamper asks the binary rather than deciding for itself, and
    // skips unknowns instead of guessing — writing a guess is the one error
    // that silently removes a disclosure.
    const std::string driver = read_file("models/stamp-published-voices.sh");
    REQUIRE(contains(driver, "--print-speaker-identity"));
    REQUIRE(contains(driver, "SKIP (unknown"));
    REQUIRE_FALSE(contains(driver, "real_person)")); // no restated verdict table
}

TEST_CASE("the converters share one definition of the stamp", "[unit][compliance]") {
    // Seven converters can write this key. Seven hand-written copies would be
    // seven chances for one to drift from crispasr_voice::speaker_identity_key()
    // — and a drift fails OPEN: the stamp is simply never found, and nothing
    // errors. One module, one spelling.
    const std::string helper = read_file("models/_speaker_identity_arg.py");
    REQUIRE(contains(helper, "IDENTITY_KEY = \"crispasr.voice.speaker_identity\""));
    REQUIRE(contains(helper, "WRITABLE = (\"real_person\", \"synthetic\")"));
    // Writes nothing when no value was given: absence means "not established",
    // and a converter must never assert a verdict nobody made.
    REQUIRE(contains(helper, "if not value:\n        return False"));

    for (const char* conv : {
             "models/convert-kokoro-voice-to-gguf.py",
             "models/convert-piper-to-gguf.py",
             "models/convert-fastpitch-to-gguf.py",
             "models/convert-bananamind-tts-to-gguf.py",
             "models/convert-parler-to-gguf.py",
             "models/convert-csm-to-gguf.py",
         }) {
        INFO("converter=" << conv);
        const std::string src = read_file(conv);
        REQUIRE(contains(src, "add_speaker_identity_arg("));
        REQUIRE(contains(src, "stamp_speaker_identity("));
    }
}

TEST_CASE("marking primitives live at the core layer", "[unit][compliance]") {
    // crispasr_watermark.h used to sit under examples/cli/, so src/ reached
    // into the examples tree to find it (src/crispasr_c_api.cpp, and then
    // src/csm_tts.cpp when the diff harness needed marking — the second
    // instance is what made it worth fixing). It now lives next to
    // crispasr_c2pa.h, which is where marking primitives belong.
    REQUIRE(read_file("src/core/crispasr_watermark.h").find("crispasr_watermark_embed_impl") != std::string::npos);
    for (const char* f : {"src/crispasr_c_api.cpp", "src/csm_tts.cpp", "examples/cli/crispasr_run.cpp",
                          "examples/cli/crispasr_server.cpp", "examples/cli/crispasr_watermark_dispatch.h"}) {
        INFO("includer=" << f);
        const std::string src = read_file(f);
        REQUIRE(contains(src, "#include \"core/crispasr_watermark.h\""));
        // No path back into the examples tree from anywhere.
        REQUIRE_FALSE(contains(src, "../examples/cli/crispasr_watermark.h"));
    }
}

TEST_CASE("published GGUFs can be stamped without re-converting", "[unit][compliance]") {
    // The stamp is only real if it can reach files that are ALREADY published.
    // Re-converting a 3.5 GB checkpoint to add one string is a price that means
    // it does not get done, so the answer would stay in the file-name table for
    // good.
    const std::string tool = read_file("models/stamp-speaker-identity.py");
    REQUIRE(contains(tool, "IDENTITY_KEY = \"crispasr.voice.speaker_identity\""));
    // "unknown" must not be writable: absence of the key IS unknown, and
    // writing it would turn "nobody established this" into a claim the file
    // makes about itself.
    REQUIRE(contains(tool, "WRITABLE = (\"real_person\", \"synthetic\")"));
    // GGUFReader exposes the file header as pseudo-fields; copying them emits
    // duplicate keys. Caught by round-tripping a real file, not by reading the
    // code — the output still loads, so nothing fails loudly.
    REQUIRE(contains(tool, "HEADER_PSEUDO_FIELDS"));
    REQUIRE(contains(tool, "skip = HEADER_PSEUDO_FIELDS |"));
}

TEST_CASE("declared sources combine by strongest duty", "[unit][compliance]") {
    // The rule that keeps a stale stamp from cancelling a researched verdict.
    // Guarded here as well as in the pure test because it is the kind of thing
    // a later "simplification" turns back into precedence.
    const std::string src = read_file("examples/cli/crispasr_speaker_identity.h");
    REQUIRE(contains(src, "duty_rank"));
    REQUIRE(contains(src, "if (duty_rank(model_value) > duty_rank(best))"));
    REQUIRE(contains(src, "if (duty_rank(backend_value) > duty_rank(best))"));
}

TEST_CASE("the verdict table records its evidence", "[unit][compliance]") {
    // These are research results, not code. The value of the table is the
    // reasoning beside each entry — a bare `return RealPerson;` is unreviewable
    // and gets flipped by whoever finds it inconvenient.
    const std::string src = read_file("examples/cli/crispasr_speaker_identity_models.h");
    // The SECTION BANNER, not a bare mention: "OPEN QUESTIONS" also appears as
    // a cross-reference inside the orpheus branch, so matching the short string
    // stayed green with the backlog itself deleted. The red-proof caught that.
    REQUIRE(contains(src, "OPEN QUESTIONS — models whose card has NOT been read"));
    // The conflict found while porting CrispTTS's research, and its resolution:
    // kokoro is synthetic upstream, but CrispASR reaches HUI's named narrators
    // through df_eva / dm_bernd. The evidence has to stay next to the verdict.
    REQUIRE(contains(src, "HUI-Audio-Corpus-German"));
    REQUIRE(contains(src, "identity_for_voice_pack"));
    // ...and it has to be CALLED. Asserting the name against the models header
    // (where it is defined) proved nothing: deleting the call site left the
    // definition, and the guard stayed green. Pin the call, in the file that
    // makes it.
    REQUIRE(contains(read_file("examples/cli/crispasr_voice_provenance.h"),
                     "d.pack_identity = identity_for_voice_pack(resolved);"));
    // Each Unknown must record that the provider was CHECKED, not skipped —
    // otherwise "unknown" stops meaning anything and gets cleared as backlog.
    // Matched on the REASONING, not on a prose fragment: an earlier version
    // pinned an exact sentence and went red the moment the evidence note was
    // reworded, which trains people to edit the test instead of reading it.
    // Single-line tokens: comment text wraps, so a phrase that reads as one
    // sentence in the source may not exist as one contiguous string.
    REQUIRE(contains(src, "provider statement"));
    REQUIRE(contains(src, "Checked, "));
    // And the cost of matching on a file name has to be stated, not hidden.
    REQUIRE(contains(src, "WHAT MATCHING ON A FILE NAME COSTS"));
}

TEST_CASE("every synthesis surface resolves the speaker identity", "[unit][compliance]") {
    SECTION("CLI") {
        const std::string src = read_file("examples/cli/crispasr_run.cpp");
        REQUIRE(contains(src, "resolve_speaker_identity"));
        REQUIRE(contains(src, "needs_spoken_disclosure"));
        // ...and USES it: the disclaimer must key on the resolved answer, not
        // on is_voice_clone. A resolve whose result is never read is the exact
        // shape of a fix that ships inert.
        REQUIRE(contains(src, "if (needs_spoken_disclosure && !params.tts_no_spoken_disclaimer)"));
        REQUIRE_FALSE(contains(src, "if (is_voice_clone && !params.tts_no_spoken_disclaimer)"));
    }
    SECTION("HTTP server") {
        const std::string src = read_file("examples/cli/crispasr_server.cpp");
        REQUIRE(contains(src, "resolve_speaker_identity"));
        REQUIRE(contains(src, "crispasr_marking::decide(needs_spoken_disclosure,"));
    }
    SECTION("Wyoming") {
        const std::string src = read_file("examples/cli/wyoming.cpp");
        REQUIRE(contains(src, "resolve_speaker_identity"));
        REQUIRE(contains(src, "clone_decision.is_clone, rp.tts_voice_clone_consent, needs_spoken_disclosure"));
    }
    SECTION("C ABI") {
        const std::string src = read_file("src/crispasr_c_api.cpp");
        REQUIRE(contains(src, "crispasr_session_set_speaker_identity"));
        REQUIRE(contains(src, "s->voice_pack_identity = voice_decision.pack_identity;"));
        REQUIRE(contains(src, "requires_spoken_disclosure(s->voice_is_clone, identity)"));
    }
}

TEST_CASE("the identity is readable from a pack and a bank entry", "[unit][compliance]") {
    const std::string src = read_file("examples/cli/crispasr_voice_provenance.h");
    REQUIRE(contains(src, "speaker_identity_key()"));
    REQUIRE(contains(src, "speaker_identity_key_for(voice_name)"));
    // Carried out of classify_voice so the server's hot path does not reopen
    // the GGUF per request just to ask a second question about it.
    REQUIRE(contains(src, "d.pack_identity = is_bank_entry ? bank.identity : p.identity;"));
}

TEST_CASE("the operator can answer the question on every surface", "[unit][compliance]") {
    // A warning nobody can act on is one they learn to ignore, and this one
    // fires on every unresearched preset backend in the project.
    // Matched WITH the closing quote: a bare "--speaker-identity" substring
    // survives renaming the flag to "--speaker-identity-disabled", which the
    // red-proof duly demonstrated on an earlier draft of this line.
    REQUIRE(contains(read_file("examples/cli/cli.cpp"), "arg == \"--speaker-identity\""));
    REQUIRE(contains(read_file("examples/cli/crispasr_server.cpp"), "body.value(\"speaker_identity\""));
    REQUIRE(contains(read_file("include/crispasr_session.h"), "crispasr_session_set_speaker_identity"));
    REQUIRE(contains(read_file("include/crispasr.h"), "crispasr_session_set_speaker_identity"));
}

TEST_CASE("consent stays keyed on cloning alone", "[unit][compliance]") {
    // The other half of the split, and the one that is easy to get wrong in the
    // "safe" direction: making a real-person preset demand --i-have-rights
    // would gate every documented preset example behind an attestation the
    // operator cannot truthfully give.
    const std::string policy = read_file("examples/cli/crispasr_speaker_identity.h");
    REQUIRE(contains(policy, "inline bool requires_consent_attestation(bool is_clone, SpeakerIdentity /*identity*/) {\n"
                             "    return is_clone;\n"
                             "}"));
}

// ---------------------------------------------------------------------------
// Every producer that consumes a recording gates and stamps.
//
// Baking IS the cloning step: everything downstream just replays the pack. A
// baker that skips the stamp produces a clone the runtime reads as a preset —
// no consent asked, no [CONSENT] line, no Art. 50(4) audible disclosure.
// ---------------------------------------------------------------------------

TEST_CASE("the cosyvoice3 bank baker gates and stamps", "[unit][compliance]") {
    const std::string src = read_file("models/convert-cosyvoice3-voices-to-gguf.py");
    // Assert the REFUSAL, not the flag. `--i-have-rights` appears in the
    // argparse help whether or not anything checks it, and `raise SystemExit(`
    // appears in any script that validates anything — a guard built from those
    // two stayed green with the gate gutted, which is how this test earned its
    // own red-proof.
    REQUIRE(contains(src, "if not args.i_have_rights:"));
    REQUIRE(contains(src, "baking a CosyVoice3 voice bundle requires --i-have-rights."));
    // Per-entry stamp: a bundle can hold a preset and a clone at once, and a
    // bank-wide flag would have to gate both or free both.
    REQUIRE(contains(src, "crispasr.voice.{name}.cloned_from_recording"));
    // The sentinel that makes an ABSENT per-voice key mean "preset" rather than
    // "baked before the stamp existed". Without it every entry falls back to
    // the producer architecture and re-gates the presets.
    REQUIRE(contains(src, "crispasr.voice.bank_stamped"));
}

TEST_CASE("the kugelaudio voice baker gates the --audio path only", "[unit][compliance]") {
    const std::string src = read_file("models/convert-kugelaudio-voice-to-gguf.py");
    REQUIRE(contains(src, "if not args.i_have_rights:"));
    REQUIRE(contains(src, "encoding a voice from --audio requires --i-have-rights."));
    REQUIRE(contains(src, "crispasr.voice.cloned_from_recording"));
    // --voice-pt converts an upstream pre-encoded voice: no recording of a
    // natural person, so no attestation and no stamp. That dual mode is also
    // why `kugelaudio-voice` must NOT go on the architecture list — it cannot
    // tell the two apart, and the stamp is the only predicate that can.
    REQUIRE(contains(src, "cloned_from_recording=True"));
    REQUIRE_FALSE(contains(read_file("examples/cli/crispasr_voice_clone_policy.h"), "arch == \"kugelaudio-voice\""));
}

TEST_CASE("the recording-derived producer list matches the bakers", "[unit][compliance]") {
    // Classification by producer, for packs baked before the stamp existed.
    // The header claims "every recording-derived producer in this repo is
    // enumerated above" — that claim was false for a year, which is what let
    // cosyvoice3 bundles through. Pin it.
    const std::string policy = read_file("examples/cli/crispasr_voice_clone_policy.h");
    REQUIRE(contains(policy, "\"chatterbox-voice\""));
    REQUIRE(contains(policy, "\"qwen3tts.voicepack\""));
    REQUIRE(contains(policy, "\"cosyvoice3-voices\""));
}

// ---------------------------------------------------------------------------
// Voice ENROLLMENT over the network is gated where the recording enters.
// ---------------------------------------------------------------------------

TEST_CASE("uploading a voice reference requires a consent attestation", "[unit][compliance]") {
    const std::string src = read_file("examples/cli/crispasr_server.cpp");
    // POST /v1/voices stores an arbitrary uploaded recording as a reusable
    // voiceprint — the network equivalent of --make-ref, which demands
    // --i-have-rights. It accepted uploads from anyone who could reach the
    // endpoint and logged only a byte count.
    // The record is built from structured fields now (crispasr_consent_record.h)
    // rather than one format string, so the assertion follows the field the
    // emitted line still carries — `scope=voice-upload` — in its new shape.
    // Deleting the record still fails this, which is the property that matters.
    REQUIRE(contains(src, "\"scope\", \"voice-upload\""));
    // And the record must be bound to the bytes that were uploaded, not merely
    // to a byte count: hashing the buffer is what makes the attestation
    // checkable against the recording later.
    REQUIRE(contains(src, "bytes_sha256(voice_file.content.data()"));
    REQUIRE(contains(src, "uploading a voice reference requires a 'consent_attestation' form field"));
}

// ---------------------------------------------------------------------------
// The bindings' watermark helper marks at a DETECTABLE strength.
//
// This is the call the docs point synthesize_raw() callers at: opting out of
// automatic marking makes marking the result their duty, and this discharges
// it. Three bindings defaulted it to 0.005 — the strength the C ABI itself
// documents as "too faint to reliably detect on real speech, so it did not
// robustly satisfy EU AI Act Art. 50 'detectable' marking". The ABI fixed that
// by treating alpha <= 0 as the robust default; an explicit positive 0.005
// walks straight past the fallback. Marking you cannot detect is not marking.
// ---------------------------------------------------------------------------

TEST_CASE("bindings request the robust default watermark strength", "[unit][compliance]") {
    SECTION("python") {
        const std::string src = read_file("python/crispasr/_binding.py");
        REQUIRE(contains(src, "def watermark_embed(pcm: \"numpy.ndarray\", alpha: float = -1.0)"));
        REQUIRE_FALSE(contains(src, "alpha: float = 0.005"));
    }
    SECTION("dart / flutter") {
        const std::string src = read_file("flutter/crispasr/lib/src/crispasr.dart");
        REQUIRE(contains(src, "{double alpha = -1.0,"));
        REQUIRE_FALSE(contains(src, "double alpha = 0.005"));
    }
    SECTION("go") {
        const std::string src = read_file("bindings/go/crispasr_session.go");
        REQUIRE(contains(src, "C.int(len(pcm)), -1.0)"));
        REQUIRE_FALSE(contains(src, "C.int(len(pcm)), 0.005)"));
    }
}

TEST_CASE("every binding that can opt out of marking can also mark", "[unit][compliance]") {
    // Rust, Ruby, Java and C# exposed synthesize_raw (UNMARKED audio, behind an
    // attestation) while exposing no watermark call at all — so the documented
    // remediation, "mark the result yourself", was unreachable from the binding
    // that handed you the unmarked buffer.
    SECTION("rust") {
        REQUIRE(contains(read_file("crispasr/src/lib.rs"), "pub fn watermark_embed(pcm: &mut [f32])"));
    }
    SECTION("ruby") {
        REQUIRE(contains(read_file("bindings/ruby/ext/ruby_crispasr_session.c"),
                         "\"watermark_embed\", rb_watermark_embed"));
    }
    SECTION("java") {
        REQUIRE(contains(read_file("bindings/java/src/main/java/io/github/ggerganov/whispercpp/CrispasrSession.java"),
                         "public static void watermarkEmbed(float[] pcm)"));
    }
    SECTION("csharp") {
        REQUIRE(contains(read_file("bindings/csharp/CrispASR/Session.cs"),
                         "public static void WatermarkEmbed(float[] pcm)"));
    }
}

// ---------------------------------------------------------------------------
// Synthetic TEXT surfaces disclose (Art. 50(1)) and carry marking metadata
// (Art. 50(2)).
//
// CrispASR marks audio; nothing marks short-form text, and the docs said so —
// for ONE of the four surfaces that generate it. The chat ABI, the installed
// crispasr-chat binary and the Flutter chat binding were all unlisted, and the
// Flutter one is exactly where §6.3's "a CLI is obvious to a reasonably
// well-informed person" stops being true.
// ---------------------------------------------------------------------------

TEST_CASE("the chat ABI publishes a canonical AI disclosure", "[unit][compliance]") {
    const std::string hdr = read_file("include/crispasr_chat.h");
    REQUIRE(contains(hdr, "crispasr_chat_ai_disclosure_text(void)"));
    // The header is where an integrator learns the duty exists. It carried no
    // AI Act text at all, unlike crispasr.h and crispasr_session.h.
    REQUIRE(contains(hdr, "Art. 50(1)"));
    REQUIRE(contains(hdr, "Art. 50(2)"));

    const std::string impl = read_file("src/chat.cpp");
    REQUIRE(contains(impl, "crispasr_chat_ai_disclosure_text(void)"));
    // Pinned wording: four surfaces read this string, and a drift between them
    // is a disclosure that differs depending on which one you called.
    REQUIRE(contains(impl, "You are interacting with an AI system. "
                           "Responses are generated by artificial intelligence."));
}

TEST_CASE("the chat CLI discloses at startup", "[unit][compliance]") {
    const std::string src = read_file("examples/cli/crispasr_chat_main.cpp");
    REQUIRE(contains(src, "crispasr_chat_ai_disclosure_text()"));
}

TEST_CASE("the chat endpoint marks its responses as AI-generated", "[unit][compliance]") {
    // No watermark-equivalent survives a copy-paste of a sentence, so the
    // Commission's guidance points at metadata travelling with the response.
    // For an HTTP API that is a header, and it must be set on the buffered and
    // the SSE branch alike — set_header after a body write is too late.
    const std::string src = read_file("examples/cli/crispasr_server.cpp");
    REQUIRE(contains(src, "X-Crispasr-Ai-Generated"));
    REQUIRE(contains(src, "X-Crispasr-Ai-Disclosure"));
}

TEST_CASE("the flutter chat binding surfaces the disclosure", "[unit][compliance]") {
    const std::string src = read_file("flutter/crispasr/lib/src/chat.dart");
    REQUIRE(contains(src, "aiDisclosureText"));
    REQUIRE(contains(src, "crispasr_chat_ai_disclosure_text"));
}
