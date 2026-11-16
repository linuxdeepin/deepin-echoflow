using System;
using System.Runtime.InteropServices;

namespace CrispASR
{
    /// <summary>
    /// Audio file decoding, using the same decoder the <c>crispasr</c> CLI
    /// uses — WAV/MP3/FLAC/OGG/Opus/AAC/M4A/WebM/AMR, any sample rate, any
    /// channel count, resampled to 16 kHz mono float32.
    /// <para>
    /// Issue #291: every ASR entry point in this binding takes
    /// <c>float[]</c> PCM, and the .NET base class library has no audio
    /// decoder at all — so "it works with crispasr.exe but not with
    /// libcrispasr" was usually a hand-rolled WAV reader feeding the session
    /// 44.1 kHz int16 or a buffer that still had the RIFF header in it. Use
    /// <see cref="Load(string)"/> and the two paths decode identically.
    /// </para>
    /// </summary>
    public static class Audio
    {
        /// <summary>The sample rate every ASR backend expects.</summary>
        public const int AsrSampleRate = 16000;

        /// <summary>
        /// Decode an audio file to 16 kHz mono float32 PCM, ready for
        /// <see cref="Session.Transcribe(float[])"/>.
        /// </summary>
        /// <exception cref="System.IO.FileNotFoundException">the file does not exist</exception>
        /// <exception cref="InvalidOperationException">the file could not be decoded</exception>
        public static float[] Load(string path)
            => Load(path, AsrSampleRate, out _);

        /// <summary>
        /// Decode an audio file, resampling to <paramref name="targetRate"/>.
        /// When the source already has that rate no resampling happens, so a
        /// non-16 kHz backend avoids a lossy down-then-up trip.
        /// <paramref name="sampleRate"/> receives the rate of the returned PCM.
        /// </summary>
        public static float[] Load(string path, int targetRate, out int sampleRate)
        {
            if (path is null) throw new ArgumentNullException(nameof(path));
            if (!System.IO.File.Exists(path))
                throw new System.IO.FileNotFoundException("Audio file not found", path);
            if (targetRate <= 0)
                throw new ArgumentOutOfRangeException(nameof(targetRate), targetRate, "must be positive");

            int rc = NativeMethods.crispasr_audio_load_at_rate(
                path, targetRate, out IntPtr pcm, out int nSamples, out sampleRate);
            if (rc != 0 || pcm == IntPtr.Zero)
                throw new InvalidOperationException(
                    $"Failed to decode audio file '{path}' (rc={rc}). " +
                    "Supported: WAV/MP3/FLAC/OGG/Opus/AAC/M4A/WebM/AMR.");
            try
            {
                if (nSamples <= 0) return Array.Empty<float>();
                var managed = new float[nSamples];
                Marshal.Copy(pcm, managed, 0, nSamples);
                return managed;
            }
            finally
            {
                NativeMethods.crispasr_audio_free(pcm);
            }
        }

        /// <summary>
        /// Wrap float32 mono PCM into 16-bit WAV bytes, tagged with the
        /// AI-generated provenance marker in a standard LIST/INFO chunk. This
        /// is the counterpart of <see cref="Load(string)"/> for
        /// <see cref="Session.Synthesize(string)"/> output.
        /// </summary>
        public static byte[] PcmToWav(float[] pcm, int sampleRate = AsrSampleRate)
        {
            if (pcm is null) throw new ArgumentNullException(nameof(pcm));
            if (sampleRate <= 0)
                throw new ArgumentOutOfRangeException(nameof(sampleRate), sampleRate, "must be positive");

            IntPtr wav = NativeMethods.crispasr_pcm_to_wav(pcm, pcm.Length, sampleRate, out UIntPtr len);
            if (wav == IntPtr.Zero)
                throw new InvalidOperationException("crispasr_pcm_to_wav failed");
            try
            {
                var bytes = new byte[(int)len];
                Marshal.Copy(wav, bytes, 0, bytes.Length);
                return bytes;
            }
            finally
            {
                NativeMethods.crispasr_c2pa_free(wav);
            }
        }

        /// <summary>Write float32 mono PCM to a 16-bit WAV file.</summary>
        public static void WriteWav(string path, float[] pcm, int sampleRate = AsrSampleRate)
            => System.IO.File.WriteAllBytes(path, PcmToWav(pcm, sampleRate));
    }
}
