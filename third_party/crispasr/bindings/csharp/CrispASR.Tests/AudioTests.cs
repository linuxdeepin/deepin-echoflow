using System;
using System.IO;
using CrispASR;
using Xunit;

namespace CrispASR.Tests
{
    /// <summary>
    /// Round-trips through the native audio helpers added for issue #291.
    /// These need only the library, no model, so they run in every job — and
    /// they are the coverage for a mismatched allocator pairing, which is the
    /// kind of bug that corrupts the heap long after the call that caused it
    /// (<c>crispasr_pcm_to_wav</c> mallocs; the buffer is released with
    /// <c>crispasr_c2pa_free</c>, which is the matching <c>free</c>).
    /// </summary>
    public class AudioTests
    {
        private static float[] Tone(int sampleRate, double seconds, double hz = 440.0)
        {
            var pcm = new float[(int)(sampleRate * seconds)];
            for (int i = 0; i < pcm.Length; i++)
                pcm[i] = 0.25f * MathF.Sin(2f * MathF.PI * (float)hz * i / sampleRate);
            return pcm;
        }

        [Fact]
        public void PcmToWav_ProducesAReadableRiffContainer()
        {
            if (!Live.LibraryLoadable()) return;
            var wav = Audio.PcmToWav(Tone(16000, 0.5), 16000);

            Assert.True(wav.Length > 44, $"only {wav.Length} bytes — smaller than a WAV header");
            Assert.Equal((byte)'R', wav[0]);
            Assert.Equal((byte)'I', wav[1]);
            Assert.Equal((byte)'F', wav[2]);
            Assert.Equal((byte)'F', wav[3]);
            Assert.Equal((byte)'W', wav[8]);
            Assert.Equal((byte)'A', wav[9]);
            Assert.Equal((byte)'V', wav[10]);
            Assert.Equal((byte)'E', wav[11]);
        }

        // Write with the binding, read back with the binding: the two halves of
        // the file surface added for #291 have to agree with each other before
        // it is worth asking whether they agree with anything else.
        [Fact]
        public void WriteWav_ThenLoad_PreservesDurationAndSignal()
        {
            if (!Live.LibraryLoadable()) return;
            const int sr = 16000;
            var original = Tone(sr, 1.0);
            var path = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".wav");
            try
            {
                Audio.WriteWav(path, original, sr);
                var decoded = Audio.Load(path);

                // 16-bit round-trip, so exact equality is not the contract;
                // the length and the energy are.
                Assert.InRange(decoded.Length, original.Length - sr / 100, original.Length + sr / 100);

                double rmsIn = Rms(original), rmsOut = Rms(decoded);
                Assert.True(Math.Abs(rmsIn - rmsOut) < 0.01,
                    $"RMS changed across the round trip: {rmsIn:F4} -> {rmsOut:F4}");
            }
            finally { if (File.Exists(path)) File.Delete(path); }
        }

        [Fact]
        public void Load_ResamplesToTheRequestedRate()
        {
            if (!Live.LibraryLoadable()) return;
            const int srcRate = 24000;
            var path = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".wav");
            try
            {
                Audio.WriteWav(path, Tone(srcRate, 1.0), srcRate);

                var at16k = Audio.Load(path, 16000, out int got16k);
                Assert.Equal(16000, got16k);
                Assert.InRange(at16k.Length, 15800, 16200);

                var native = Audio.Load(path, srcRate, out int gotNative);
                Assert.Equal(srcRate, gotNative);
                Assert.InRange(native.Length, srcRate - 200, srcRate + 200);
            }
            finally { if (File.Exists(path)) File.Delete(path); }
        }

        [Fact]
        public void Load_MissingFile_ThrowsFileNotFound()
        {
            Assert.Throws<FileNotFoundException>(
                () => Audio.Load(Path.Combine(Path.GetTempPath(), "definitely-not-here-291.wav")));
        }

        [Fact]
        public void Load_RejectsANonPositiveRate()
        {
            if (!Live.LibraryLoadable()) return;
            var path = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".wav");
            try
            {
                Audio.WriteWav(path, Tone(16000, 0.1), 16000);
                Assert.Throws<ArgumentOutOfRangeException>(() => Audio.Load(path, 0, out _));
            }
            finally { if (File.Exists(path)) File.Delete(path); }
        }

        // Allocate and release many buffers in a row: a wrong free would be far
        // more likely to surface here than on a single call.
        [Fact]
        public void PcmToWav_RepeatedAllocationAndRelease_IsStable()
        {
            if (!Live.LibraryLoadable()) return;
            var pcm = Tone(16000, 0.05);
            int lastLength = 0;
            for (int i = 0; i < 200; i++)
            {
                var wav = Audio.PcmToWav(pcm, 16000);
                if (i > 0) Assert.Equal(lastLength, wav.Length);
                lastLength = wav.Length;
            }
            Assert.True(lastLength > 44);
        }

        private static double Rms(float[] x)
        {
            if (x.Length == 0) return 0.0;
            double sum = 0.0;
            foreach (var v in x) sum += (double)v * v;
            return Math.Sqrt(sum / x.Length);
        }
    }
}
