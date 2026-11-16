using System;
using Xunit;
using CrispASR;

namespace CrispASR.Tests
{
    /// <summary>
    /// Tests for the readonly-struct data types — these run without
    /// the native library and validate the public API surface.
    /// </summary>
    public class DataTypeTests
    {
        // ---- Word ----

        [Fact]
        public void Word_DefaultAlts_IsEmptyArray()
        {
            var w = new Word("hello", 0, 100, 0.95f);
            Assert.NotNull(w.Alts);
            Assert.Empty(w.Alts);
        }

        [Fact]
        public void Word_WithAlts_PreservesAll()
        {
            var alts = new[] { new AltToken("helo", 0.03f), new AltToken("hullo", 0.02f) };
            var w = new Word("hello", 10, 50, 0.95f, alts);
            Assert.Equal("hello", w.Text);
            Assert.Equal(10, w.T0);
            Assert.Equal(50, w.T1);
            Assert.Equal(0.95f, w.P);
            Assert.Equal(2, w.Alts.Length);
            Assert.Equal("helo", w.Alts[0].Text);
            Assert.Equal(0.03f, w.Alts[0].P);
        }

        // Times are SECONDS (issue #291), formatted culture-invariantly so the
        // string is the same on a machine whose decimal separator is a comma.
        [Fact]
        public void Word_ToString_ShowsRangeInSeconds()
        {
            var w = new Word("test", 1.0, 2.0, 0.9f);
            Assert.Equal("1.00-2.00 test", w.ToString());
        }

        [Fact]
        public void Word_UntimedSentinelSurvivesRoundTrip()
        {
            var w = new Word("test", -1.0, -1.0, 0.9f);
            Assert.Equal(-1.0, w.T0);
            Assert.Equal(-1.0, w.T1);
        }

        // ---- AltToken ----

        [Fact]
        public void AltToken_ToString_ShowsPercentage()
        {
            var a = new AltToken("foo", 0.123f);
            Assert.Contains("foo", a.ToString());
            Assert.Contains("12.3%", a.ToString());
        }

        [Fact]
        public void AltToken_ZeroProbability()
        {
            var a = new AltToken("x", 0f);
            Assert.Equal(0f, a.P);
        }

        // ---- Segment ----

        [Fact]
        public void Segment_Properties()
        {
            var words = new[] { new Word("a", 0, 10, 1.0f), new Word("b", 10, 20, 0.9f) };
            var seg = new Segment("a b", 0, 20, words);
            Assert.Equal("a b", seg.Text);
            Assert.Equal(0, seg.T0);
            Assert.Equal(20, seg.T1);
            Assert.Equal(2, seg.Words.Length);
        }

        [Fact]
        public void Segment_ToString_ShowsBracketedRangeInSeconds()
        {
            var seg = new Segment("hello world", 0.5, 3.0, Array.Empty<Word>());
            Assert.Equal("[0.50-3.00] hello world", seg.ToString());
        }

        [Fact]
        public void Segment_EmptyWords()
        {
            var seg = new Segment("text", 0, 100, Array.Empty<Word>());
            Assert.Empty(seg.Words);
        }

        // ---- AlignedWord ----

        [Fact]
        public void AlignedWord_Properties()
        {
            var aw = new AlignedWord("hello", 100, 250);
            Assert.Equal("hello", aw.Text);
            Assert.Equal(100, aw.T0);
            Assert.Equal(250, aw.T1);
        }

        // ---- VadSpan ----

        [Fact]
        public void VadSpan_Properties()
        {
            var span = new VadSpan(1.5, 3.7);
            Assert.Equal(1.5, span.T0);
            Assert.Equal(3.7, span.T1);
        }

        [Fact]
        public void VadSpan_ZeroDuration()
        {
            var span = new VadSpan(2.0, 2.0);
            Assert.Equal(span.T0, span.T1);
        }

        // ---- LanguageDetection ----

        [Fact]
        public void LanguageDetection_Ok_WhenValid()
        {
            var ld = new LanguageDetection("en", 0.98f);
            Assert.True(ld.Ok);
            Assert.Equal("en", ld.Code);
            Assert.Equal(0.98f, ld.Probability);
        }

        [Fact]
        public void LanguageDetection_NotOk_WhenEmpty()
        {
            var ld = new LanguageDetection("", 0.5f);
            Assert.False(ld.Ok);
        }

        [Fact]
        public void LanguageDetection_NotOk_WhenNegativeProb()
        {
            var ld = new LanguageDetection("de", -1.0f);
            Assert.False(ld.Ok);
        }

        [Fact]
        public void LanguageDetection_ToString()
        {
            var ld = new LanguageDetection("fr", 0.876f);
            Assert.Contains("fr", ld.ToString());
            Assert.Contains("87.6%", ld.ToString());
        }

        // ---- KokoroResolved ----

        [Fact]
        public void KokoroResolved_WithVoice()
        {
            var r = new KokoroResolved("/models/kokoro.gguf", "/voices/de.gguf", "df_victoria", true);
            Assert.Equal("/models/kokoro.gguf", r.ModelPath);
            Assert.Equal("/voices/de.gguf", r.VoicePath);
            Assert.Equal("df_victoria", r.VoiceName);
            Assert.True(r.BackboneSwapped);
        }

        [Fact]
        public void KokoroResolved_WithoutVoice()
        {
            var r = new KokoroResolved("/models/kokoro.gguf", null, null, false);
            Assert.Null(r.VoicePath);
            Assert.Null(r.VoiceName);
            Assert.False(r.BackboneSwapped);
        }

        // ---- StreamingUpdate ----

        [Fact]
        public void StreamingUpdate_Properties()
        {
            var su = new StreamingUpdate("hello world", 1.5, 3.0, 42);
            Assert.Equal("hello world", su.Text);
            Assert.Equal(1.5, su.T0);
            Assert.Equal(3.0, su.T1);
            Assert.Equal(42, su.Counter);
        }

        [Fact]
        public void StreamingUpdate_EmptyText()
        {
            var su = new StreamingUpdate("", 0, 0, 0);
            Assert.Equal("", su.Text);
            Assert.Equal(0, su.Counter);
        }

        // ---- Music / task-backend result types (newly bound; issue #291 follow-up) ----

        [Fact]
        public void BeatEvent_And_Result()
        {
            var b = new BeatEvent(1.25, true);
            Assert.Equal(1.25, b.TimeSeconds);
            Assert.True(b.IsDownbeat);
            var r = new BeatResult(new[] { b }, 120.0);
            Assert.Single(r.Beats);
            Assert.Equal(120.0, r.TempoBpm);
        }

        [Fact]
        public void ChordSpan_Properties()
        {
            var c = new ChordSpan(0.5, 2.0, 3, 0.9f, "Am");
            Assert.Equal(0.5, c.StartSeconds);
            Assert.Equal(2.0, c.EndSeconds);
            Assert.Equal(3, c.Label);
            Assert.Equal("Am", c.Name);
        }

        [Fact]
        public void PianoNote_Properties()
        {
            var n = new PianoNote(0.1, 0.6, 60, 100);
            Assert.Equal(0.1, n.OnsetSeconds);
            Assert.Equal(60, n.MidiNote);
            Assert.Equal(100, n.Velocity);
        }

        [Fact]
        public void PitchFrame_Properties()
        {
            var p = new PitchFrame(0.02, 220.0f, 0.95f);
            Assert.Equal(0.02, p.TimeSeconds);
            Assert.Equal(220.0f, p.F0Hz);
            Assert.Equal(0.95f, p.VoicedProb);
        }

        [Fact]
        public void Stem_Properties()
        {
            var s = new Stem("vocals", new float[] { 0f, 0f, 1f, 1f }, 2);
            Assert.Equal("vocals", s.Name);
            Assert.Equal(2, s.PerChannelSamples);
            Assert.Equal(4, s.Interleaved.Length);
        }

        [Fact]
        public void TabEmissions_Indexing()
        {
            var t = new TabEmissions(2, 6, 21, new float[2 * 6 * 21], 20, 0.0232f, new[] { 40, 45, 50, 55, 59, 64 });
            Assert.Equal(6, t.Strings);
            Assert.Equal(20, t.SilentClass);
            Assert.Equal(6, t.StringOpenMidi.Length);
            Assert.Equal(2 * 6 * 21, t.LogProbs.Length);
        }
    }
}
