using System;
using Xunit;
using CrispASR;

namespace CrispASR.Tests
{
    /// <summary>
    /// Tests for TitaNet and SpeakerDb types. Live tests are gated
    /// by the availability of the native library and model files.
    /// </summary>
    public class TitaNetTests
    {
        // Routed through Live so CRISPASR_CS_REQUIRE_LIVE can turn a silent
        // skip into a failure (issue #291 — nothing in CI ever loaded the
        // library, so the suite was green either way).
        private static bool CanLoadLibrary() => Live.LibraryLoadable();

        [Fact]
        public void TitaNet_ImplementsIDisposable()
        {
            Assert.True(typeof(IDisposable).IsAssignableFrom(typeof(TitaNet)));
        }

        [Fact]
        public void TitaNet_Open_ThrowsOnBadPath()
        {
            if (!CanLoadLibrary()) return;
            Assert.Throws<InvalidOperationException>(() =>
                TitaNet.Open("/nonexistent/titanet.gguf"));
        }

        [Fact]
        public void CosineSim_IdenticalVectors_ReturnsOne()
        {
            if (!CanLoadLibrary()) return;
            var v = new float[] { 1, 0, 0, 0 };
            float sim = TitaNet.CosineSim(v, v);
            Assert.InRange(sim, 0.999f, 1.001f);
        }

        [Fact]
        public void CosineSim_OrthogonalVectors_ReturnsZero()
        {
            if (!CanLoadLibrary()) return;
            var a = new float[] { 1, 0, 0, 0 };
            var b = new float[] { 0, 1, 0, 0 };
            float sim = TitaNet.CosineSim(a, b);
            Assert.InRange(sim, -0.001f, 0.001f);
        }

        [Fact]
        public void CosineSim_OppositeVectors_ReturnsNegOne()
        {
            if (!CanLoadLibrary()) return;
            var a = new float[] { 1, 0, 0, 0 };
            var b = new float[] { -1, 0, 0, 0 };
            float sim = TitaNet.CosineSim(a, b);
            Assert.InRange(sim, -1.001f, -0.999f);
        }

        [Fact]
        public void CosineSim_MismatchedDimensions_Throws()
        {
            var a = new float[] { 1, 0 };
            var b = new float[] { 1, 0, 0 };
            Assert.Throws<ArgumentException>(() => TitaNet.CosineSim(a, b));
        }
    }

    public class SpeakerDbTests
    {
        // Routed through Live so CRISPASR_CS_REQUIRE_LIVE can turn a silent
        // skip into a failure (issue #291 — nothing in CI ever loaded the
        // library, so the suite was green either way).
        private static bool CanLoadLibrary() => Live.LibraryLoadable();

        [Fact]
        public void SpeakerDb_ImplementsIDisposable()
        {
            Assert.True(typeof(IDisposable).IsAssignableFrom(typeof(SpeakerDb)));
        }

        [Fact]
        public void SpeakerDb_Open_ThrowsWithoutConsent()
        {
            // Consent gate is enforced managed-side; no native library needed.
            Assert.Throws<InvalidOperationException>(() =>
                SpeakerDb.Open("/nonexistent/speakers", "Alice", consentAttested: false));
        }

        [Fact]
        public void SpeakerDb_Open_ThrowsOnEmptyRoster()
        {
            // No open 1:N mode: an empty claimed roster is refused natively.
            if (!CanLoadLibrary()) return;
            Assert.Throws<InvalidOperationException>(() =>
                SpeakerDb.Open("/nonexistent/speakers", "", consentAttested: true));
        }

        [Fact]
        public void SpeakerDb_Enroll_ThrowsWithoutConsent()
        {
            Assert.Throws<InvalidOperationException>(() =>
                SpeakerDb.Enroll("/nonexistent/speakers", "alice", new float[] { 1, 0 }, consentAttested: false));
        }
    }
}
