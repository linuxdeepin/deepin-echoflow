using System;
using Xunit;
using CrispASR;

namespace CrispASR.Tests
{
    /// <summary>
    /// Smoke tests for the AI-content marking-attestation gate on the C-ABI
    /// (crispasr_session_synthesize_raw + crispasr_session_accept_marking_responsibility).
    ///
    /// The C# binding cannot be compiled on the author's machine, so these tests
    /// exist to make CI catch two failure modes:
    ///   1. P/Invoke signature drift — the whole test project fails to
    ///      <c>dotnet build</c> if NativeMethods / Session signatures no longer
    ///      match the C ABI (this is compile-time, so it fires even without a model).
    ///   2. Entry-point / marshalling errors at runtime — the null-handle calls
    ///      below exercise the real exported symbols against a freshly built
    ///      libcrispasr; the native layer is null-safe, so no model is needed.
    /// A live behavioral test (refuse → attest) is gated by a TTS model env var and
    /// skips cleanly when unset, like the other live tests in this project.
    /// </summary>
    public class MarkingAttestationTests
    {
        // Routed through Live so CRISPASR_CS_REQUIRE_LIVE can turn a silent
        // skip into a failure (issue #291 — nothing in CI ever loaded the
        // library, so the suite was green either way).
        private static bool CanLoadLibrary() => Live.LibraryLoadable();

        [Fact]
        public void AcceptMarkingResponsibility_NativeEntryPoint_ResolvesAndIsNullSafe()
        {
            if (!CanLoadLibrary()) return; // skip when the native lib isn't loadable
            // Null session → -1 (native impl is null-safe). Reaching this return
            // value proves the P/Invoke signature matches and the symbol resolves.
            int rc = NativeMethods.crispasr_session_accept_marking_responsibility(IntPtr.Zero, "csharp ci smoke");
            Assert.Equal(-1, rc);
        }

        [Fact]
        public void SynthesizeRaw_NativeEntryPoint_ResolvesAndIsNullSafe()
        {
            if (!CanLoadLibrary()) return;
            IntPtr ptr = NativeMethods.crispasr_session_synthesize_raw(IntPtr.Zero, "hello", out int n);
            Assert.Equal(IntPtr.Zero, ptr);
            Assert.Equal(0, n);
        }

        [Fact]
        public void Session_ExposesMarkingApi()
        {
            // Pure reflection (no native lib needed). nameof also gives a
            // compile-time guarantee the public methods exist.
            Assert.NotNull(typeof(Session).GetMethod(nameof(Session.AcceptMarkingResponsibility)));
            Assert.NotNull(typeof(Session).GetMethod(nameof(Session.SynthesizeRaw)));
        }

        /// <summary>
        /// Live behavioral check: unmarked SynthesizeRaw is refused until the
        /// caller attests. Gated by a TTS model on disk; skips in CI when unset.
        /// </summary>
        [Fact]
        public void SynthesizeRaw_RefusedWithoutAttestation_ThenGateOpens()
        {
            if (!CanLoadLibrary()) return;
            var modelPath = Environment.GetEnvironmentVariable("CRISPASR_MODEL_TTS");
            if (!Live.Model(modelPath, "CRISPASR_MODEL_TTS")) return;

            using var s = Session.Open(modelPath);
            // Without an attestation the unmarked path must be hard-refused.
            Assert.Throws<InvalidOperationException>(() => s.SynthesizeRaw("hello world"));
            // Attesting opens the gate; the attestation call itself must succeed.
            s.AcceptMarkingResponsibility("csharp ci: I accept marking responsibility");
        }
    }
}
