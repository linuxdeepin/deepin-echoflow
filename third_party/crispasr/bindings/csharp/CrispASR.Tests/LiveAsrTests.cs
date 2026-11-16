using System;
using System.Linq;
using CrispASR;
using Xunit;

namespace CrispASR.Tests
{
    /// <summary>
    /// End-to-end tests against a REAL GGUF model and a REAL audio file —
    /// the coverage issue #291 asked for ("Which tests use a real world gguf
    /// model?"). Everything above these ran without ever opening a model, so
    /// "the xunit tests pass" said nothing about whether transcription worked.
    /// <para>
    /// Provisioning (CI does all three; locally, set them by hand):
    /// <list type="bullet">
    /// <item><c>CRISPASR_MODEL_ASR</c> — path to an ASR GGUF (CI uses moonshine-tiny,
    ///   ~20 MB, with its companion <c>tokenizer.bin</c> in the same directory)</item>
    /// <item><c>CRISPASR_AUDIO_SAMPLE</c> — path to an audio file (CI uses samples/jfk.wav)</item>
    /// <item><c>CRISPASR_ASR_BACKEND</c> — the backend name auto-detection must return</item>
    /// </list>
    /// With <c>CRISPASR_CS_REQUIRE_LIVE=1</c> a missing model is a failure, so
    /// these cannot quietly stop running the way the old live tests did.
    /// </para>
    /// </summary>
    public class LiveAsrTests
    {
        private static string? Model => Environment.GetEnvironmentVariable("CRISPASR_MODEL_ASR");
        private static string? Sample => Environment.GetEnvironmentVariable("CRISPASR_AUDIO_SAMPLE");
        private static string? ExpectedBackend => Environment.GetEnvironmentVariable("CRISPASR_ASR_BACKEND");

        private static bool Ready() =>
            Live.LibraryLoadable()
            && Live.Model(Model, "CRISPASR_MODEL_ASR", required: true)
            && Live.Model(Sample, "CRISPASR_AUDIO_SAMPLE", required: true);

        // The C ABI has its OWN architecture→backend table; the CLI additionally
        // has a filename heuristic that short-circuits it. So "the CLI opens my
        // model" proves nothing about Session.Open — this asserts the binding's
        // path directly, which is the one issue #291 was reporting on.
        [Fact]
        public void AutoDetectionResolvesTheBackendThroughTheCAbi()
        {
            if (!Ready()) return;
            var backend = Session.DetectBackendFromGguf(Model!);
            Assert.False(string.IsNullOrEmpty(backend),
                $"crispasr_detect_backend_from_gguf returned nothing for {Model} — " +
                "the architecture is missing from the C-ABI table (src/core/arch_backend_map.h)");
            if (!string.IsNullOrEmpty(ExpectedBackend))
                Assert.Equal(ExpectedBackend, backend);
        }

        // The .NET BCL has no audio decoder, so before Audio.Load a C# caller
        // had to hand-roll a WAV reader — the usual reason a model that worked
        // under crispasr.exe produced nothing under libcrispasr.
        [Fact]
        public void AudioLoadDecodesToSixteenKilohertzMono()
        {
            if (!Ready()) return;
            var pcm = Audio.Load(Sample!);
            Assert.NotEmpty(pcm);
            Assert.All(pcm, v => Assert.InRange(v, -1.5f, 1.5f));   // normalised float, not int16
            Assert.True(pcm.Length > Audio.AsrSampleRate / 2,
                $"only {pcm.Length} samples decoded from {Sample} — under half a second");
        }

        [Fact]
        public void TranscribeFileProducesText()
        {
            if (!Ready()) return;
            using var s = Session.Open(Model!, 2);
            var segs = s.TranscribeFile(Sample!);
            Assert.NotEmpty(segs);
            var text = string.Concat(segs.Select(x => x.Text)).Trim();
            Assert.False(string.IsNullOrWhiteSpace(text),
                $"{segs.Length} segment(s) came back but every one was empty");
        }

        // Issue #291's original complaint was a units bug (VAD spans in
        // centiseconds behind a doc-comment promising seconds). Segment/Word
        // T0/T1 had the SAME defect and were cited in the answer as the
        // reference for "everything is seconds". They are seconds now; this is
        // the guard. A centisecond regression overshoots by 100x, so the
        // duration bound catches it on any clip longer than ~10 ms.
        [Fact]
        public void SegmentAndWordTimesAreSecondsNotCentiseconds()
        {
            if (!Ready()) return;
            var pcm = Audio.Load(Sample!);
            double durationSeconds = (double)pcm.Length / Audio.AsrSampleRate;
            double ceiling = durationSeconds + 1.0;   // slack for a trailing pad

            using var s = Session.Open(Model!, 2);
            var segs = s.Transcribe(pcm);
            Assert.NotEmpty(segs);

            foreach (var seg in segs)
            {
                InSecondsOrUntimed(seg.T0, ceiling, "segment T0");
                InSecondsOrUntimed(seg.T1, ceiling, "segment T1");
                if (seg.T0 >= 0 && seg.T1 >= 0)
                    Assert.True(seg.T1 >= seg.T0, $"segment ends ({seg.T1}) before it starts ({seg.T0})");
                foreach (var w in seg.Words)
                {
                    InSecondsOrUntimed(w.T0, ceiling, "word T0");
                    InSecondsOrUntimed(w.T1, ceiling, "word T1");
                }
            }
        }

        // -1 is the C ABI's "this backend emits no timing here" sentinel
        // (moonshine sets every token t0/t1 to -1), and the binding passes it
        // through unscaled rather than reporting -0.01 s. Anything else must
        // land inside the clip: a centisecond regression overshoots by 100x.
        private static void InSecondsOrUntimed(double value, double ceiling, string what)
        {
            if (value == -1.0) return;
            Assert.True(value >= 0.0 && value <= ceiling,
                $"{what}={value} is outside [0, {ceiling:F2}] for this clip — centiseconds leaking through?");
        }

        // Question 3 of issue #291: load the model once, reuse it. The answer
        // was "that is what a Session is" — with nothing asserting it.
        [Fact]
        public void OneSessionTranscribesRepeatedlyWithoutReopening()
        {
            if (!Ready()) return;
            var pcm = Audio.Load(Sample!);
            using var s = Session.Open(Model!, 2);

            var first = string.Concat(s.Transcribe(pcm).Select(x => x.Text)).Trim();
            var second = string.Concat(s.Transcribe(pcm).Select(x => x.Text)).Trim();
            var third = string.Concat(s.Transcribe(pcm).Select(x => x.Text)).Trim();

            Assert.False(string.IsNullOrWhiteSpace(first));
            Assert.Equal(first, second);
            Assert.Equal(first, third);
        }

        // The resolver must report where it actually loaded from — the first
        // thing to ask when the CLI works and the binding does not.
        [Fact]
        public void ResolverReportsTheLibraryItLoaded()
        {
            if (!Live.LibraryLoadable()) return;
            var path = NativeLibraryResolver.ResolvedPath;
            if (path is null) return;   // found by the runtime's own probing
            Assert.True(System.IO.File.Exists(path), $"resolver reported {path}, which does not exist");
            Assert.False(NativeLibraryResolver.IsManagedAssembly(path),
                $"resolver loaded a MANAGED assembly ({path}) — issue #291's collision is back");
        }
    }
}
