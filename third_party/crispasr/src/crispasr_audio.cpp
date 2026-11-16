// Minimal audio file decoder for the language wrappers.
//
// libwhisper callers (Dart, Python, Rust wrappers) need cross-platform
// decoding of common audio formats so they can hand
// `crispasr_session_transcribe` a clean 16-kHz mono float32 buffer
// regardless of the original input format — without an ffmpeg dependency.
//
// miniaudio (MIT-0) handles WAV/MP3/FLAC (+ AIFF/W64/RF64 via its dr_wav) out
// of the box and does resampling + channel down-mix internally via its
// `ma_decoder` stream. Ogg Vorbis is handled by stb_vorbis (include it
// header-only before miniaudio so MA_HAS_VORBIS is auto-defined). Opus is added
// as a miniaudio custom backend (libopus/opusfile, CRISPASR_HAVE_OPUS; see
// below). Raw ADTS AAC-LC (.aac) is decoded by the in-tree glint clean-room
// decoder — cross-platform and always available, no runtime library (gated by
// CRISPASR_AAC_DECODER, glint primary); container AAC (M4A/ALAC/CAF) still
// falls back to fdk-aac (dlopen) or AudioToolbox on Apple. All
// permissive-licensed; ffmpeg stays an optional dynamic fallback only.

// On Windows, include Media Foundation headers BEFORE miniaudio /
// stb_vorbis. These headers define GUIDs with aggregate initializers
// that break if any macro (from miniaudio or stb) clobbers tokens used
// in the initializer (mfapi.h line 1779 → C2059 'constant' on MSVC).
// Including them first, before any user-defined macros, avoids this.
#if defined(_WIN32) && !defined(__MINGW32__)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#endif

// cstdint before anything else — MinGW GCC doesn't pull in uint32_t
// via windows.h, and stb_vorbis / miniaudio use fixed-width types.
#include <cstdint>

// stb_vorbis lives in examples/ — use relative path from src/
#define STB_VORBIS_HEADER_ONLY
#include "../examples/stb_vorbis.c"

#define MINIAUDIO_IMPLEMENTATION
// Device IO (capture mode) is needed for `crispasr_mic_*` (PLAN #62d);
// MA_NO_DEVICE_IO would strip the ma_device_* symbols. Threading
// follows from device IO. MA_NO_GENERATION (no oscillators / synth
// helpers) is still safe to keep.
#define MA_NO_GENERATION

// On iOS / tvOS / watchOS / visionOS, miniaudio's CoreAudio backend
// pulls in AVFoundation Objective-C headers from a .cpp TU, which the
// C++ front-end can't parse (NSString, etc., need Objective-C++ →
// .mm), and visionOS additionally has no AVAudioSession at all. The
// static-lib build artifact for those platforms is consumed by host
// apps that handle mic capture themselves, so we drop device IO here.
// The crispasr_mic_* C ABI is stubbed out to return failure on the
// same platforms (see crispasr_mic.cpp). macOS keeps device IO.
//
// TARGET_OS_IPHONE is the catch-all for the iOS-family platforms
// (iOS, tvOS, watchOS, visionOS); macOS leaves it as 0.
#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#define MA_NO_DEVICE_IO
#endif
#endif

#include "miniaudio.h"

#undef STB_VORBIS_HEADER_ONLY
#include "../examples/stb_vorbis.c"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// In-tree glint clean-room codec suite — used here for always-available,
// cross-platform AAC-LC (ADTS) decode of raw .aac input (public C ABI only).
#include <glint/glint.h>

// Optional .opus (Ogg/Opus) support via libopus + opusfile (BSD-3-Clause),
// wired in as a miniaudio custom decoding backend so .opus flows through the
// same ma_decoder resample-to-16k + downmix + chunked-read path as WAV / MP3 /
// FLAC / OGG-Vorbis — no separate decode path, and no ffmpeg. The backend
// vtable (`ma_decoding_backend_libopus`) is defined in the vendored
// examples/miniaudio_libopus.c, compiled as a separate C TU and linked against
// this TU's MINIAUDIO_IMPLEMENTATION (see src/CMakeLists.txt). Gated on
// CRISPASR_HAVE_OPUS, which CMake sets when opusfile is found (pkg-config) or
// built statically (FetchContent fallback for platforms without system libs).
#if defined(CRISPASR_HAVE_OPUS)
#include "miniaudio_libopus.h"
namespace {
// ma_decoding_backend_libopus is already a `ma_decoding_backend_vtable*`.
ma_decoding_backend_vtable* g_crispasr_opus_backends[] = {ma_decoding_backend_libopus};
} // namespace
#define CRISPASR_OPUS_DECODER_CONFIG(cfg)                                                                              \
    do {                                                                                                               \
        (cfg).ppCustomBackendVTables = g_crispasr_opus_backends;                                                       \
        (cfg).customBackendCount = 1;                                                                                  \
    } while (0)
#else
#define CRISPASR_OPUS_DECODER_CONFIG(cfg)                                                                              \
    do {                                                                                                               \
    } while (0)
#endif

#ifdef _WIN32
#define CA_EXPORT extern "C" __declspec(dllexport)
#else
#define CA_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// Forward declarations — satisfies -Wmissing-declarations without
// pulling in the full crispasr.h header (which conflicts with
// miniaudio's implementation-mode defines in this TU).
CA_EXPORT int crispasr_audio_load(const char*, float**, int*, int*);
CA_EXPORT int crispasr_audio_load_at_rate(const char*, int, float**, int*, int*);
CA_EXPORT int crispasr_audio_load_stereo(const char*, float**, float**, int*, int*, int*);
CA_EXPORT void crispasr_audio_free(float*);

namespace {
// Thread-local so crispasr_audio_load_at_rate can set the target rate for
// the current decode without affecting other threads.  Default is 16 kHz
// (Whisper-compatible) — overridden by crispasr_audio_load_at_rate().
static thread_local int kTargetSampleRate = 16000;
constexpr int kTargetChannels = 1;

// Ceiling on how much PCM one decode may produce.
//
// Found by CI's seeded fuzzer on 2026-09-04 (run 33840954769): a 344 KB WAV
// declaring `sampleRate = 1` is a structurally valid RIFF file, and miniaudio
// dutifully resamples it to the 16 kHz target — a 16000x expansion — turning
// 176 000 stored samples into 2.816e9 output frames, 11.3 GB, from an input
// small enough to arrive as an ordinary HTTP upload. The three chunked-decode
// loops below grow their buffer geometrically and had no ceiling at all, so
// they simply kept doubling until the allocator or the OOM killer intervened.
// Every surface that accepts a user-supplied file reaches this path.
//
// THE BOUND IS AGAINST INPUT SIZE, because the defect is AMPLIFICATION, not
// length: a genuinely long recording carries proportionally more input bytes,
// so a ratio bound never penalises it, while an absolute duration limit would
// have to choose between rejecting real 8-hour recordings and permitting this.
//
// kMaxFramesPerInputByte = 256 against the worst ratios real encoders produce
// at 16 kHz mono: AMR-NB at 4.75 kbit/s is ~27 output frames per input byte,
// Opus at 6 kbit/s ~21, MPEG-2.5 mp3 at 8 kbit/s ~16, 8-bit 8 kHz PCM ~2. So
// ~9.5x headroom over anything an encoder emits in practice. The one case that
// reaches the bound is libopus driven to its 500 bit/s API floor (exactly 256),
// which no encoder produces for speech; such a caller raises the ceiling with
// CRISPASR_MAX_DECODED_FRAMES rather than being silently truncated.
//
// A second, absolute backstop covers large inputs, where 256x is still a lot:
// 24 h at the target rate, which at 16 kHz mono f32 is 5.5 GB — past anything
// that loads into RAM today, so it constrains only the pathological case.
//
// CRISPASR_MAX_DECODED_FRAMES REPLACES the effective ceiling (0 = no limit)
// rather than raising just one of the two bounds. The first draft here had it
// lift only the absolute backstop while the ratio bound still applied — so the
// rejection message named a knob that could not lift the bound that had
// actually fired, which is worse than no knob.
constexpr size_t kMaxFramesPerInputByte = 256;

size_t crispasr_max_decoded_frames(const char* path) {
    // Every caller computes `ceiling + one chunk` when clamping its allocation,
    // so a ceiling near SIZE_MAX would wrap that sum to a tiny number and spin
    // the grow branch forever. Halving the representable range costs nothing
    // real (SIZE_MAX/2 frames is 4 exabytes of f32) and removes the wrap.
    const size_t kCeilingMax = (size_t)-1 / 2;
    auto clamp = [&](size_t v) { return v > kCeilingMax ? kCeilingMax : v; };

    if (const char* env = std::getenv("CRISPASR_MAX_DECODED_FRAMES")) {
        char* end = nullptr;
        const unsigned long long v = std::strtoull(env, &end, 10);
        if (end != env && *end == '\0')
            return v == 0 ? 0 : clamp((v > (unsigned long long)(size_t)-1) ? (size_t)-1 : (size_t)v); // 0 = unlimited
        // An unparseable value falls through to the defaults rather than
        // silently meaning "unlimited".
    }

    // uint64 math then clamp: at a 96 kHz target rate 24 h does not fit in a
    // 32-bit size_t, and a wrapped backstop would read as "almost no limit".
    const uint64_t abs64 = (uint64_t)(kTargetSampleRate > 0 ? kTargetSampleRate : 16000) * 60u * 60u * 24u;
    const size_t abs_cap = (abs64 > (uint64_t)(size_t)-1) ? (size_t)-1 : (size_t)abs64;

    size_t rel_cap = 0;
    if (FILE* f = std::fopen(path, "rb")) {
        if (std::fseek(f, 0, SEEK_END) == 0) {
            const long n = std::ftell(f);
            if (n > 0) {
                // Saturate rather than wrap on 32-bit, where size_t is 4 bytes
                // and a 500 MB input times 256 does not fit.
                rel_cap =
                    ((size_t)n > (size_t)-1 / kMaxFramesPerInputByte) ? (size_t)-1 : (size_t)n * kMaxFramesPerInputByte;
            }
        }
        std::fclose(f);
    }

    if (rel_cap == 0)
        return clamp(abs_cap); // unstattable: only the backstop applies
    return clamp(rel_cap < abs_cap ? rel_cap : abs_cap);
}

// Shared by the three loops: report once, with the override, so a caller who
// hits the ceiling legitimately can tell it apart from a corrupt file.
void crispasr_report_decode_ceiling(const char* path, size_t max_frames) {
    std::fprintf(stderr,
                 "[crispasr-audio] '%s' decodes to more than %zu frames at %d Hz and was rejected.\n"
                 "                 This usually means a malformed header (e.g. a WAV declaring a 1 Hz\n"
                 "                 sample rate) asking for an enormous resample. If the file is real,\n"
                 "                 raise CRISPASR_MAX_DECODED_FRAMES.\n",
                 path ? path : "(null)", max_frames, kTargetSampleRate);
}
} // namespace

// Apple-platform fallback for formats the permissive miniaudio path can't
// decode — notably AAC / M4A / ALAC / CAF. AudioToolbox's ExtAudioFile is a
// system framework (no ffmpeg, no GPL) that decodes + resamples + remixes to a
// requested client format in one pass. On non-Apple platforms the equivalent
// is Media Foundation (Windows) / MediaCodec (Android), wired separately; Linux
// has no system AAC decoder (fdk-aac or the optional ffmpeg fallback). Decodes
// to interleaved f32 @ 16 kHz with `want_channels` channels (1..2; 0 = native,
// capped at 2). malloc-owned buffer; 0 on success.
#if defined(__APPLE__)
#include <AudioToolbox/AudioToolbox.h>
#include <CoreFoundation/CoreFoundation.h>
namespace {
int crispasr_at_decode(const char* path, int want_channels, float** out_interleaved, int* out_frames,
                       int* out_channels) {
    CFStringRef cfpath = CFStringCreateWithCString(nullptr, path, kCFStringEncodingUTF8);
    if (!cfpath)
        return -2;
    CFURLRef url = CFURLCreateWithFileSystemPath(nullptr, cfpath, kCFURLPOSIXPathStyle, false);
    CFRelease(cfpath);
    if (!url)
        return -2;
    ExtAudioFileRef af = nullptr;
    OSStatus st = ExtAudioFileOpenURL(url, &af);
    CFRelease(url);
    if (st != noErr || !af)
        return -2;

    AudioStreamBasicDescription fileFmt;
    UInt32 sz = sizeof(fileFmt);
    if (ExtAudioFileGetProperty(af, kExtAudioFileProperty_FileDataFormat, &sz, &fileFmt) != noErr) {
        ExtAudioFileDispose(af);
        return -2;
    }
    int native_ch = (int)fileFmt.mChannelsPerFrame;
    if (native_ch < 1)
        native_ch = 1;
    int ch = want_channels > 0 ? want_channels : (native_ch >= 2 ? 2 : 1);
    if (ch < 1)
        ch = 1;
    if (ch > 2)
        ch = 2;

    AudioStreamBasicDescription cli;
    std::memset(&cli, 0, sizeof(cli));
    cli.mSampleRate = kTargetSampleRate;
    cli.mFormatID = kAudioFormatLinearPCM;
    cli.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked; // interleaved f32
    cli.mBitsPerChannel = 32;
    cli.mChannelsPerFrame = (UInt32)ch;
    cli.mFramesPerPacket = 1;
    cli.mBytesPerFrame = (UInt32)(sizeof(float) * ch);
    cli.mBytesPerPacket = cli.mBytesPerFrame;
    if (ExtAudioFileSetProperty(af, kExtAudioFileProperty_ClientDataFormat, sizeof(cli), &cli) != noErr) {
        ExtAudioFileDispose(af);
        return -2;
    }

    const UInt32 chunkFrames = (UInt32)kTargetSampleRate; // 1 s
    const size_t max_frames = crispasr_max_decoded_frames(path);
    float* buf = nullptr;
    size_t cap = 0, used = 0; // frames
    for (;;) {
        if (max_frames && used >= max_frames) {
            crispasr_report_decode_ceiling(path, max_frames);
            std::free(buf);
            ExtAudioFileDispose(af);
            return -3;
        }
        if (cap - used < chunkFrames) {
            size_t newcap = cap ? cap * 2 : (size_t)chunkFrames * 8;
            // Never allocate past the ceiling. One chunk of slack keeps
            // `cap - used >= chunkFrames` true for every `used < max_frames`,
            // so the clamp cannot spin re-allocating the same size forever.
            if (max_frames && newcap > max_frames + chunkFrames)
                newcap = max_frames + chunkFrames;
            float* nb = (float*)std::realloc(buf, newcap * (size_t)ch * sizeof(float));
            if (!nb) {
                std::free(buf);
                ExtAudioFileDispose(af);
                return -3;
            }
            buf = nb;
            cap = newcap;
        }
        AudioBufferList abl;
        abl.mNumberBuffers = 1;
        abl.mBuffers[0].mNumberChannels = (UInt32)ch;
        abl.mBuffers[0].mDataByteSize = (UInt32)(chunkFrames * (UInt32)ch * sizeof(float));
        abl.mBuffers[0].mData = buf + used * (size_t)ch;
        UInt32 n = chunkFrames;
        if (ExtAudioFileRead(af, &n, &abl) != noErr) {
            std::free(buf);
            ExtAudioFileDispose(af);
            return -4;
        }
        if (n == 0)
            break;
        used += n;
    }
    ExtAudioFileDispose(af);
    if (used == 0) {
        std::free(buf);
        return -2;
    }
    *out_interleaved = buf;
    *out_frames = (int)used;
    *out_channels = ch;
    return 0;
}
} // namespace
#endif // __APPLE__

// ── Sun AU / .snd fallback (inline, no deps) ────────────────────────────────
// AU header: magic ".snd" (0x2e736e64 big-endian), data_offset, data_size,
// encoding, sample_rate, channels — all 32-bit big-endian.
// Encodings: 1=µ-law 8-bit, 2=linear PCM 8-bit, 3=linear 16-bit, 5=linear 32-bit,
// 6=float32, 27=A-law 8-bit. Decodes to f32 mono @ 16 kHz via ma_resampler.
namespace {
inline uint32_t au_be32(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
}

int crispasr_au_decode(const char* path, int want_channels, float** out_buf, int* out_frames, int* out_channels) {
    FILE* f = std::fopen(path, "rb");
    if (!f)
        return -2;

    uint8_t hdr[24];
    if (std::fread(hdr, 1, 24, f) != 24) {
        std::fclose(f);
        return -2;
    }
    uint32_t magic = au_be32(hdr);
    if (magic != 0x2e736e64u) { // ".snd"
        std::fclose(f);
        return -2;
    }
    uint32_t data_offset = au_be32(hdr + 4);
    uint32_t data_size = au_be32(hdr + 8);
    uint32_t encoding = au_be32(hdr + 12);
    uint32_t file_rate = au_be32(hdr + 16);
    uint32_t file_ch = au_be32(hdr + 20);

    if (file_rate == 0 || file_ch == 0) {
        std::fclose(f);
        return -2;
    }

    // Determine bytes per sample and decoder
    int bytes_per_sample = 0;
    bool is_ulaw = false, is_alaw = false, is_float = false;
    switch (encoding) {
    case 1:
        bytes_per_sample = 1;
        is_ulaw = true;
        break; // µ-law
    case 27:
        bytes_per_sample = 1;
        is_alaw = true;
        break; // A-law
    case 2:
        bytes_per_sample = 1;
        break; // linear PCM 8-bit signed
    case 3:
        bytes_per_sample = 2;
        break; // linear 16-bit
    case 5:
        bytes_per_sample = 4;
        break; // linear 32-bit
    case 6:
        bytes_per_sample = 4;
        is_float = true;
        break; // float32
    default:
        std::fclose(f);
        return -2; // unsupported encoding
    }

    // Determine the real file size up front so an untrusted data_offset /
    // data_size can never drive an out-of-bounds seek or an unbounded
    // allocation. Without this, a crafted .snd with data_size ≈ 4 GB in a
    // tiny file forces a ~4 GB `raw` allocation (memory-exhaustion DoS /
    // unhandled bad_alloc), and a data_offset past EOF makes the old
    // `end - cur` underflow into a huge size_t.
    if (std::fseek(f, 0, SEEK_END) != 0) {
        std::fclose(f);
        return -2;
    }
    long file_end = std::ftell(f);
    if (file_end < 0 || (uint64_t)data_offset > (uint64_t)file_end) {
        std::fclose(f);
        return -2;
    }

    // Seek to data
    if (std::fseek(f, (long)data_offset, SEEK_SET) != 0) {
        std::fclose(f);
        return -2;
    }

    // Determine actual data size — always clamped to the bytes the file
    // really holds past data_offset, whatever the header claims.
    size_t avail = (size_t)(file_end - (long)data_offset);
    size_t actual_size;
    if (data_size != 0xFFFFFFFFu && data_size != 0) {
        actual_size = ((size_t)data_size < avail) ? (size_t)data_size : avail;
    } else {
        actual_size = avail;
    }

    size_t frame_size = (size_t)bytes_per_sample * file_ch;
    if (frame_size == 0) {
        std::fclose(f);
        return -2;
    }
    size_t n_frames = actual_size / frame_size;
    if (n_frames == 0) {
        std::fclose(f);
        return -2;
    }

    // Read raw data
    std::vector<uint8_t> raw(actual_size);
    size_t got = std::fread(raw.data(), 1, actual_size, f);
    std::fclose(f);
    if (got < frame_size)
        return -2;
    n_frames = got / frame_size;

    auto ulaw_to_f32 = [](uint8_t u) -> float {
        u = ~u;
        int sign = (u & 0x80) ? -1 : 1;
        int exponent = (u >> 4) & 0x07;
        int mantissa = u & 0x0F;
        int sample = ((mantissa << 1) + 33) << exponent;
        sample -= 33;
        return sign * sample / 32768.0f;
    };

    auto alaw_to_f32 = [](uint8_t a) -> float {
        a ^= 0x55;
        int sign = (a & 0x80) ? -1 : 1;
        int exponent = (a >> 4) & 0x07;
        int mantissa = a & 0x0F;
        int sample;
        if (exponent == 0) {
            sample = (mantissa << 1) | 1;
        } else {
            sample = ((mantissa << 1) | 0x21) << (exponent - 1);
        }
        return sign * sample / 32768.0f;
    };

    // Decode to float32 interleaved at file_rate
    std::vector<float> decoded(n_frames * file_ch);
    const uint8_t* p = raw.data();
    for (size_t i = 0; i < n_frames * file_ch; ++i) {
        if (is_ulaw) {
            decoded[i] = ulaw_to_f32(p[i]);
        } else if (is_alaw) {
            decoded[i] = alaw_to_f32(p[i]);
        } else if (bytes_per_sample == 1) {
            // 8-bit signed linear
            decoded[i] = (float)(int8_t)p[i] / 128.0f;
        } else if (bytes_per_sample == 2) {
            // 16-bit big-endian signed
            int16_t s = (int16_t)(((uint16_t)p[i * 2] << 8) | p[i * 2 + 1]);
            decoded[i] = s / 32768.0f;
        } else if (is_float) {
            // 32-bit big-endian float
            uint32_t bits = au_be32(p + i * 4);
            float val;
            std::memcpy(&val, &bits, 4);
            decoded[i] = val;
        } else {
            // 32-bit big-endian signed int
            int32_t s = (int32_t)au_be32(p + i * 4);
            decoded[i] = s / 2147483648.0f;
        }
    }

    // Downmix to target channel count
    int out_ch = want_channels > 0 ? want_channels : (file_ch >= 2 ? 2 : 1);
    if (out_ch > 2)
        out_ch = 2;

    std::vector<float> mixed;
    if (out_ch == 1 && file_ch > 1) {
        mixed.resize(n_frames);
        for (size_t i = 0; i < n_frames; ++i) {
            float sum = 0;
            for (uint32_t c = 0; c < file_ch; ++c)
                sum += decoded[i * file_ch + c];
            mixed[i] = sum / (float)file_ch;
        }
    } else if (out_ch == 2 && file_ch == 1) {
        mixed.resize(n_frames * 2);
        for (size_t i = 0; i < n_frames; ++i) {
            mixed[i * 2] = decoded[i];
            mixed[i * 2 + 1] = decoded[i];
        }
    } else if (out_ch <= (int)file_ch) {
        // Take first out_ch channels
        mixed.resize(n_frames * (size_t)out_ch);
        for (size_t i = 0; i < n_frames; ++i)
            for (int c = 0; c < out_ch; ++c)
                mixed[i * (size_t)out_ch + c] = decoded[i * file_ch + c];
    } else {
        mixed = std::move(decoded);
    }

    // Resample to kTargetSampleRate if needed
    if (file_rate != (uint32_t)kTargetSampleRate) {
        ma_resampler_config rcfg = ma_resampler_config_init(ma_format_f32, (ma_uint32)out_ch, file_rate,
                                                            kTargetSampleRate, ma_resample_algorithm_linear);
        ma_resampler resampler;
        if (ma_resampler_init(&rcfg, nullptr, &resampler) != MA_SUCCESS)
            return -2;

        // Estimate output size
        ma_uint64 in_len = n_frames;
        ma_uint64 out_len = 0;
        ma_resampler_get_expected_output_frame_count(&resampler, in_len, &out_len);
        out_len += 256; // margin

        float* resampled = (float*)std::malloc(out_len * (size_t)out_ch * sizeof(float));
        if (!resampled) {
            ma_resampler_uninit(&resampler, nullptr);
            return -3;
        }

        ma_uint64 in_consumed = in_len;
        ma_uint64 out_produced = out_len;
        ma_resampler_process_pcm_frames(&resampler, mixed.data(), &in_consumed, resampled, &out_produced);
        ma_resampler_uninit(&resampler, nullptr);

        *out_buf = resampled;
        *out_frames = (int)out_produced;
        *out_channels = out_ch;
        return 0;
    }

    // No resampling needed
    float* result = (float*)std::malloc(mixed.size() * sizeof(float));
    if (!result)
        return -3;
    std::memcpy(result, mixed.data(), mixed.size() * sizeof(float));
    *out_buf = result;
    *out_frames = (int)n_frames;
    *out_channels = out_ch;
    return 0;
}
} // namespace

// ── AMR-NB/WB fallback via opencore-amr (Apache-2.0) ────────────────────────
// Telephony / voicemail speech at 8 kHz (NB) or 16 kHz (WB). Framing is a
// simple "#!AMR\n" or "#!AMR-WB\n" header followed by packed frames with a
// 1-byte TOC (frame type in bits 3..6). Decoded PCM is resampled to 16 kHz.
#if defined(CRISPASR_HAVE_AMR)
#include <opencore-amrnb/interf_dec.h>
#include <opencore-amrwb/dec_if.h>
namespace {

// AMR-NB frame sizes by mode (index 0–8 = modes, 9–14 = comfort noise / bad, 15 = no data)
static const int amrnb_frame_size[16] = {12, 13, 15, 17, 19, 20, 26, 31, 5, 6, 5, 5, 0, 0, 0, 0};
// AMR-WB frame sizes by mode (index 0–8 = modes, 9 = SID, 14 = speech lost, 15 = no data)
static const int amrwb_frame_size[16] = {17, 23, 32, 36, 40, 46, 50, 58, 60, 5, 0, 0, 0, 0, 0, 0};

int crispasr_amr_decode(const char* path, int want_channels, float** out_buf, int* out_frames, int* out_channels) {
    FILE* f = std::fopen(path, "rb");
    if (!f)
        return -2;

    // Read header to detect NB vs WB
    char hdr[12];
    size_t hdr_read = std::fread(hdr, 1, 9, f);
    if (hdr_read < 6) {
        std::fclose(f);
        return -2;
    }

    bool is_wb = false;
    if (hdr_read >= 9 && std::memcmp(hdr, "#!AMR-WB\n", 9) == 0) {
        is_wb = true;
    } else if (hdr_read >= 6 && std::memcmp(hdr, "#!AMR\n", 6) == 0) {
        is_wb = false;
        // Seek back since we read 3 extra bytes
        std::fseek(f, 6, SEEK_SET);
    } else {
        std::fclose(f);
        return -2;
    }

    int pcm_rate = is_wb ? 16000 : 8000;
    int frame_samples = is_wb ? 320 : 160; // 20 ms frames
    const int* frame_sizes = is_wb ? amrwb_frame_size : amrnb_frame_size;

    void* dec = is_wb ? (void*)D_IF_init() : (void*)Decoder_Interface_init();
    if (!dec) {
        std::fclose(f);
        return -2;
    }

    std::vector<int16_t> pcm_all;
    uint8_t toc;
    uint8_t frame_buf[64]; // max AMR-WB frame is 60 bytes + toc

    while (std::fread(&toc, 1, 1, f) == 1) {
        int mode = (toc >> 3) & 0x0F;
        int fsize = frame_sizes[mode];
        if (fsize == 0 && mode == 15) {
            // No data frame
            continue;
        }
        if (fsize <= 0 || fsize > 60) {
            break; // corrupt or unsupported
        }

        frame_buf[0] = toc;
        if ((int)std::fread(frame_buf + 1, 1, (size_t)fsize, f) != fsize) {
            break;
        }

        int16_t pcm_frame[320]; // WB max
        if (is_wb) {
            D_IF_decode(dec, frame_buf, pcm_frame, 0);
        } else {
            Decoder_Interface_Decode(dec, frame_buf, pcm_frame, 0);
        }
        pcm_all.insert(pcm_all.end(), pcm_frame, pcm_frame + frame_samples);
    }
    std::fclose(f);

    if (is_wb)
        D_IF_exit(dec);
    else
        Decoder_Interface_exit(dec);

    if (pcm_all.empty())
        return -2;

    // Convert s16 → f32
    size_t n_samples = pcm_all.size();
    std::vector<float> f32(n_samples);
    for (size_t i = 0; i < n_samples; ++i)
        f32[i] = pcm_all[i] / 32768.0f;

    int out_ch = want_channels > 0 ? want_channels : 1;
    if (out_ch > 2)
        out_ch = 2;

    // Resample to 16 kHz if NB (8 kHz)
    if (pcm_rate != kTargetSampleRate) {
        ma_resampler_config rcfg = ma_resampler_config_init(ma_format_f32, 1, (ma_uint32)pcm_rate, kTargetSampleRate,
                                                            ma_resample_algorithm_linear);
        ma_resampler resampler;
        if (ma_resampler_init(&rcfg, nullptr, &resampler) != MA_SUCCESS)
            return -2;

        ma_uint64 in_len = n_samples;
        ma_uint64 out_len = 0;
        ma_resampler_get_expected_output_frame_count(&resampler, in_len, &out_len);
        out_len += 256;

        float* resampled = (float*)std::malloc(out_len * sizeof(float));
        if (!resampled) {
            ma_resampler_uninit(&resampler, nullptr);
            return -3;
        }

        ma_uint64 in_consumed = in_len;
        ma_uint64 out_produced = out_len;
        ma_resampler_process_pcm_frames(&resampler, f32.data(), &in_consumed, resampled, &out_produced);
        ma_resampler_uninit(&resampler, nullptr);

        // If stereo requested from mono source, duplicate
        if (out_ch == 2) {
            float* stereo = (float*)std::malloc(out_produced * 2 * sizeof(float));
            if (!stereo) {
                std::free(resampled);
                return -3;
            }
            for (ma_uint64 i = 0; i < out_produced; ++i) {
                stereo[i * 2] = resampled[i];
                stereo[i * 2 + 1] = resampled[i];
            }
            std::free(resampled);
            *out_buf = stereo;
        } else {
            *out_buf = resampled;
        }
        *out_frames = (int)out_produced;
        *out_channels = out_ch;
        return 0;
    }

    // Already 16 kHz (WB)
    size_t out_size = n_samples * (size_t)out_ch;
    float* result = (float*)std::malloc(out_size * sizeof(float));
    if (!result)
        return -3;
    if (out_ch == 2) {
        for (size_t i = 0; i < n_samples; ++i) {
            result[i * 2] = f32[i];
            result[i * 2 + 1] = f32[i];
        }
    } else {
        std::memcpy(result, f32.data(), n_samples * sizeof(float));
    }
    *out_buf = result;
    *out_frames = (int)n_samples;
    *out_channels = out_ch;
    return 0;
}
} // namespace
#endif // CRISPASR_HAVE_AMR

// ── WebM/Matroska Opus|Vorbis demux (inline EBML parser, no deps) ───────────
// Minimal EBML parser extracts Opus or Vorbis codec data from WebM/Matroska
// containers, then feeds the extracted Ogg stream to the existing
// in-tree glint Opus decoder (or libopus when pinned) and stb_vorbis. This
// covers browser-recorded audio (MediaRecorder → WebM/Opus) and Matroska files
// with Opus/Vorbis tracks. No external dependency — glint + stb_vorbis are
// in-tree; libopus is an optional fallback (built with CRISPASR_OPUS, selected
// via CRISPASR_OPUS_DECODER=libopus).
#if defined(CRISPASR_HAVE_OPUS)
#include <opusfile.h> // only for the optional libopus fallback path
#endif

namespace {

// Minimal EBML IDs we care about
static constexpr uint32_t EBML_SEGMENT = 0x18538067;
static constexpr uint32_t EBML_TRACKS = 0x1654AE6B;
static constexpr uint32_t EBML_TRACK_ENTRY = 0xAE;
static constexpr uint32_t EBML_TRACK_TYPE = 0x83;
static constexpr uint32_t EBML_CODEC_ID = 0x86;
static constexpr uint32_t EBML_CODEC_PRIVATE = 0x63A2;
static constexpr uint32_t EBML_CLUSTER = 0x1F43B675;
static constexpr uint32_t EBML_SIMPLE_BLOCK = 0xA3;
static constexpr uint32_t EBML_BLOCK_GROUP = 0xA0;
static constexpr uint32_t EBML_BLOCK = 0xA1;
static constexpr uint32_t EBML_AUDIO = 0xE1;
static constexpr uint32_t EBML_SAMPLING_FREQ = 0xB5;
static constexpr uint32_t EBML_CHANNELS = 0x9F;
static constexpr uint32_t EBML_TRACK_NUMBER = 0xD7;

struct EBMLReader {
    const uint8_t* data;
    size_t size;
    size_t pos;

    EBMLReader(const uint8_t* d, size_t s) : data(d), size(s), pos(0) {}

    bool eof() const { return pos >= size; }
    size_t remaining() const { return pos < size ? size - pos : 0; }

    // Read EBML variable-length integer (VINT). Returns the value and advances pos.
    // On failure returns UINT64_MAX.
    //
    // A size VINT whose data bits are ALL ONES means "unknown size" — the
    // element runs until the next element that cannot be its child. That is
    // what live/streaming muxers (Chrome MediaRecorder via libwebm, ffmpeg
    // `-live 1`) write for the Segment and for every Cluster. It must be
    // distinguished from a parse failure, and from a genuine size: the
    // all-ones pattern is length-dependent (0xFF is 127, not "huge"), so it
    // cannot be recognised from the returned value alone. Callers that care
    // pass `out_unknown`; the value returned in that case is the all-ones
    // number and must not be used as a length.
    uint64_t read_vint(bool* out_unknown = nullptr) {
        if (out_unknown)
            *out_unknown = false;
        if (eof())
            return UINT64_MAX;
        uint8_t first = data[pos];
        if (first == 0)
            return UINT64_MAX;

        int len = 0;
        uint8_t mask = 0x80;
        while (len < 8 && !(first & mask)) {
            mask >>= 1;
            ++len;
        }
        ++len; // len is now 1..8

        if (pos + (size_t)len > size)
            return UINT64_MAX;

        uint64_t val = first & (mask - 1);
        for (int i = 1; i < len; ++i)
            val = (val << 8) | data[pos + i];
        pos += len;

        if (out_unknown) {
            // All 7*len data bits set == "unknown size" marker.
            const uint64_t all_ones = (len >= 9) ? UINT64_MAX : ((1ULL << (7 * len)) - 1);
            *out_unknown = (val == all_ones);
        }
        return val;
    }

    // Read EBML element ID. IDs use the same VINT encoding but include the
    // length-indicator bit(s) in the returned value.
    uint32_t read_id() {
        if (eof())
            return 0;
        uint8_t first = data[pos];
        if (first == 0)
            return 0;

        int len = 0;
        uint8_t mask = 0x80;
        while (len < 4 && !(first & mask)) {
            mask >>= 1;
            ++len;
        }
        ++len;

        if (pos + (size_t)len > size)
            return 0;

        uint32_t id = first;
        for (int i = 1; i < len; ++i)
            id = (id << 8) | data[pos + i];
        pos += len;
        return id;
    }

    // Read a fixed number of bytes as a big-endian unsigned integer
    uint64_t read_uint(size_t len) {
        if (pos + len > size)
            return 0;
        uint64_t val = 0;
        for (size_t i = 0; i < len; ++i)
            val = (val << 8) | data[pos + i];
        pos += len;
        return val;
    }

    // Read a float (4 or 8 bytes, big-endian)
    double read_float(size_t len) {
        if (len == 4) {
            uint32_t bits = (uint32_t)read_uint(4);
            float val;
            std::memcpy(&val, &bits, 4);
            return val;
        } else if (len == 8) {
            uint64_t bits = read_uint(8);
            double val;
            std::memcpy(&val, &bits, 8);
            return val;
        }
        pos += len;
        return 0;
    }

    // Skip n bytes
    void skip(size_t n) { pos = (pos + n > size) ? size : pos + n; }

    // Read string of given length
    std::string read_string(size_t len) {
        if (pos + len > size)
            len = size - pos;
        std::string s((const char*)data + pos, len);
        pos += len;
        return s;
    }

    const uint8_t* ptr() const { return data + pos; }
};

// Valid direct children of a Cluster (Matroska spec). Used to find where an
// unknown-size Cluster ends: it ends at the first element that is not one of
// these — in practice the next Cluster, or Cues/Tags at the end of the file.
static bool is_cluster_child_id(uint32_t id) {
    switch (id) {
    case 0xE7:   // Timestamp (Timecode)
    case 0x5854: // SilentTracks
    case 0xA7:   // Position
    case 0xAB:   // PrevSize
    case 0xA3:   // SimpleBlock
    case 0xA0:   // BlockGroup
    case 0xAF:   // EncryptedBlock
        return true;
    default:
        return false;
    }
}

// Resolve the end offset of an unknown-size Cluster whose body starts at
// `body_pos`. Walks the child headers only (no payload) until an element that
// is not a Cluster child, and returns that element's start offset — that is
// where the Cluster ends and the next top-level element begins. Bounded by
// `hard_end`.
static size_t resolve_unknown_cluster_end(const uint8_t* data, size_t body_pos, size_t hard_end) {
    EBMLReader r(data, hard_end);
    r.pos = body_pos;
    while (r.pos < hard_end) {
        const size_t id_pos = r.pos;
        uint32_t id = r.read_id();
        bool child_unknown = false;
        uint64_t sz = r.read_vint(&child_unknown);
        // A malformed header, or a nested unknown size we cannot resolve,
        // terminates the cluster here rather than swallowing the rest of the
        // file — the caller still keeps every packet found so far.
        if (id == 0 || sz == UINT64_MAX || child_unknown)
            return id_pos;
        if (!is_cluster_child_id(id))
            return id_pos;
        const size_t child_end = r.pos + (size_t)sz;
        if (child_end > hard_end || child_end < r.pos)
            return hard_end;
        r.pos = child_end;
    }
    return hard_end;
}

struct WebMTrack {
    uint64_t track_number = 0;
    int track_type = 0; // 1 = video, 2 = audio
    std::string codec_id;
    std::vector<uint8_t> codec_private;
    int channels = 1;
    double sample_rate = 48000;
};

// Parse a single track entry
static WebMTrack parse_track_entry(EBMLReader& r, size_t end_pos) {
    WebMTrack t;
    while (r.pos < end_pos && !r.eof()) {
        uint32_t id = r.read_id();
        uint64_t sz = r.read_vint();
        if (sz == UINT64_MAX || id == 0)
            break;
        size_t elem_end = r.pos + (size_t)sz;
        if (elem_end > end_pos)
            break;

        switch (id) {
        case EBML_TRACK_NUMBER:
            t.track_number = r.read_uint((size_t)sz);
            break;
        case EBML_TRACK_TYPE:
            t.track_type = (int)r.read_uint((size_t)sz);
            break;
        case EBML_CODEC_ID:
            t.codec_id = r.read_string((size_t)sz);
            break;
        case EBML_CODEC_PRIVATE:
            t.codec_private.assign(r.ptr(), r.ptr() + (size_t)sz);
            r.skip((size_t)sz);
            break;
        case EBML_AUDIO: {
            // Parse Audio sub-element
            EBMLReader ar(r.data, elem_end);
            ar.pos = r.pos;
            while (ar.pos < elem_end) {
                uint32_t aid = ar.read_id();
                uint64_t asz = ar.read_vint();
                if (asz == UINT64_MAX || aid == 0)
                    break;
                size_t aend = ar.pos + (size_t)asz;
                if (aend > elem_end)
                    break;
                if (aid == EBML_SAMPLING_FREQ) {
                    t.sample_rate = ar.read_float((size_t)asz);
                } else if (aid == EBML_CHANNELS) {
                    t.channels = (int)ar.read_uint((size_t)asz);
                } else {
                    ar.skip((size_t)asz);
                }
            }
            r.pos = elem_end;
            break;
        }
        default:
            r.skip((size_t)sz);
            break;
        }
        r.pos = elem_end;
    }
    return t;
}

// Extract Opus packets from a SimpleBlock/Block
// Returns vector of raw Opus packet data
struct OpusPacket {
    const uint8_t* data;
    size_t size;
};

static std::vector<OpusPacket> parse_block_packets(const uint8_t* block_data, size_t block_size, uint64_t audio_track) {
    std::vector<OpusPacket> packets;
    if (block_size < 4)
        return packets;

    // Block header: track number (vint) + timecode (int16) + flags (uint8)
    EBMLReader br(block_data, block_size);
    uint64_t track = br.read_vint();
    if (track != audio_track)
        return packets;
    if (br.remaining() < 3)
        return packets;
    br.skip(2); // timecode
    uint8_t flags = block_data[br.pos];
    br.skip(1);

    int lacing = (flags >> 1) & 0x03;

    if (lacing == 0) {
        // No lacing — single frame
        if (br.remaining() > 0)
            packets.push_back({br.ptr(), br.remaining()});
    } else if (lacing == 2) {
        // Fixed-size lacing
        if (br.remaining() < 1)
            return packets;
        int n_frames = (int)block_data[br.pos] + 1;
        br.skip(1);
        if (n_frames <= 0 || br.remaining() == 0)
            return packets;
        size_t frame_size = br.remaining() / (size_t)n_frames;
        // A zero-size fixed frame is meaningless and makes the loop below emit
        // n_frames (up to 256) empty packets while br.skip(0) never advances —
        // a crafted cluster of tiny blocks amplifies that into billions of
        // empty vectors (OOM DoS). Reject it.
        if (frame_size == 0)
            return packets;
        for (int i = 0; i < n_frames && br.remaining() >= frame_size; ++i) {
            packets.push_back({br.ptr(), frame_size});
            br.skip(frame_size);
        }
    } else if (lacing == 1) {
        // Xiph lacing
        if (br.remaining() < 1)
            return packets;
        int n_frames = (int)block_data[br.pos] + 1;
        br.skip(1);
        std::vector<size_t> sizes;
        size_t total = 0;
        for (int i = 0; i < n_frames - 1; ++i) {
            size_t sz = 0;
            while (br.remaining() > 0) {
                uint8_t b = block_data[br.pos];
                br.skip(1);
                sz += b;
                if (b < 255)
                    break;
            }
            sizes.push_back(sz);
            total += sz;
        }
        for (int i = 0; i < n_frames - 1; ++i) {
            if (br.remaining() < sizes[i])
                break;
            packets.push_back({br.ptr(), sizes[i]});
            br.skip(sizes[i]);
        }
        // Last frame gets the rest
        if (br.remaining() > 0)
            packets.push_back({br.ptr(), br.remaining()});
    } else {
        // EBML lacing
        if (br.remaining() < 1)
            return packets;
        int n_frames = (int)block_data[br.pos] + 1;
        br.skip(1);
        std::vector<size_t> sizes;
        size_t total = 0;
        // First size is a normal vint
        uint64_t first_sz = br.read_vint();
        if (first_sz == UINT64_MAX)
            return packets;
        sizes.push_back((size_t)first_sz);
        total += (size_t)first_sz;
        // Subsequent sizes are signed deltas from previous
        for (int i = 1; i < n_frames - 1; ++i) {
            uint64_t raw = br.read_vint();
            if (raw == UINT64_MAX)
                break;
            // The value is encoded as unsigned but represents a signed offset
            // from the previous size. Subtract the bias based on vint length.
            int64_t delta = (int64_t)raw - (int64_t)((1ULL << (7 * 1)) / 2 - 1); // approximate
            int64_t sz = (int64_t)sizes.back() + delta;
            if (sz < 0)
                sz = 0;
            sizes.push_back((size_t)sz);
            total += (size_t)sz;
        }
        for (size_t i = 0; i < sizes.size(); ++i) {
            if (br.remaining() < sizes[i])
                break;
            packets.push_back({br.ptr(), sizes[i]});
            br.skip(sizes[i]);
        }
        if (br.remaining() > 0)
            packets.push_back({br.ptr(), br.remaining()});
    }
    return packets;
}

int crispasr_webm_decode(const char* path, int want_channels, float** out_buf, int* out_frames, int* out_channels) {
    // Read entire file
    FILE* f = std::fopen(path, "rb");
    if (!f)
        return -2;
    std::fseek(f, 0, SEEK_END);
    long fsize = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (fsize <= 0 || fsize > 500 * 1024 * 1024) { // 500 MB sanity limit
        std::fclose(f);
        return -2;
    }
    std::vector<uint8_t> file_data((size_t)fsize);
    if (std::fread(file_data.data(), 1, (size_t)fsize, f) != (size_t)fsize) {
        std::fclose(f);
        return -2;
    }
    std::fclose(f);

    EBMLReader r(file_data.data(), file_data.size());

    // Verify EBML header
    uint32_t ebml_id = r.read_id();
    if (ebml_id != 0x1A45DFA3) // EBML header
        return -2;
    uint64_t ebml_sz = r.read_vint();
    if (ebml_sz == UINT64_MAX)
        return -2;
    r.skip((size_t)ebml_sz); // skip EBML header body

    // Find Segment
    uint32_t seg_id = r.read_id();
    bool seg_unknown = false;
    uint64_t seg_sz = r.read_vint(&seg_unknown);
    if (seg_id != EBML_SEGMENT || seg_sz == UINT64_MAX)
        return -2;

    // A live/streaming muxer writes the Segment with unknown size (it cannot
    // seek back to patch it once recording ends), so the Segment runs to EOF.
    size_t seg_end = seg_unknown ? r.size : r.pos + (size_t)seg_sz;
    if (seg_end > r.size || seg_end < r.pos)
        seg_end = r.size;

    // First pass: find Tracks element and parse audio track info
    WebMTrack audio_track;
    bool found_audio = false;
    size_t saved_pos = r.pos;

    while (r.pos < seg_end && !r.eof()) {
        uint32_t id = r.read_id();
        bool unknown_size = false;
        uint64_t sz = r.read_vint(&unknown_size);
        if (sz == UINT64_MAX || id == 0)
            break;
        size_t elem_end;
        if (unknown_size) {
            // Only a Cluster can legitimately carry an unknown size here, and
            // Tracks always precedes the first one — but resolve it properly
            // rather than giving up, so a stray unknown-size element before
            // Tracks does not hide the track list.
            elem_end = (id == EBML_CLUSTER) ? resolve_unknown_cluster_end(r.data, r.pos, seg_end) : seg_end;
        } else {
            elem_end = r.pos + (size_t)sz;
            if (elem_end > seg_end || elem_end < r.pos)
                elem_end = seg_end;
        }

        if (id == EBML_TRACKS) {
            // Parse track entries
            while (r.pos < elem_end) {
                uint32_t tid = r.read_id();
                uint64_t tsz = r.read_vint();
                if (tsz == UINT64_MAX || tid == 0)
                    break;
                size_t te_end = r.pos + (size_t)tsz;
                if (te_end > elem_end)
                    break;
                if (tid == EBML_TRACK_ENTRY) {
                    WebMTrack t = parse_track_entry(r, te_end);
                    if (t.track_type == 2 && !found_audio) { // audio
                        if (t.codec_id == "A_OPUS" || t.codec_id == "A_VORBIS") {
                            audio_track = std::move(t);
                            found_audio = true;
                        }
                    }
                }
                r.pos = te_end;
            }
            break; // done with tracks
        }
        r.pos = elem_end;
    }

    if (!found_audio)
        return -2;

    bool is_opus = (audio_track.codec_id == "A_OPUS");

    // Second pass: collect audio packets from Clusters
    r.pos = saved_pos;
    std::vector<std::vector<uint8_t>> opus_packets;
    std::vector<uint8_t> vorbis_data; // for Vorbis, concatenate raw packets

    size_t n_clusters = 0;
    while (r.pos < seg_end && !r.eof()) {
        uint32_t id = r.read_id();
        bool unknown_size = false;
        uint64_t sz = r.read_vint(&unknown_size);
        if (sz == UINT64_MAX || id == 0)
            break;
        size_t elem_end;
        if (unknown_size) {
            elem_end = (id == EBML_CLUSTER) ? resolve_unknown_cluster_end(r.data, r.pos, seg_end) : seg_end;
        } else {
            elem_end = r.pos + (size_t)sz;
            if (elem_end > seg_end || elem_end < r.pos)
                elem_end = seg_end;
        }

        if (id == EBML_CLUSTER) {
            ++n_clusters;
            // Parse blocks within cluster
            while (r.pos < elem_end && !r.eof()) {
                uint32_t bid = r.read_id();
                uint64_t bsz = r.read_vint();
                if (bsz == UINT64_MAX || bid == 0)
                    break;
                size_t block_end = r.pos + (size_t)bsz;
                if (block_end > elem_end)
                    break;

                if (bid == EBML_SIMPLE_BLOCK) {
                    auto pkts = parse_block_packets(r.ptr(), (size_t)bsz, audio_track.track_number);
                    for (auto& p : pkts) {
                        opus_packets.emplace_back(p.data, p.data + p.size);
                    }
                    // Defense-in-depth cap: a valid Opus packet occupies >=1 byte
                    // of the file, so the packet count can never legitimately
                    // exceed the file size — bail rather than accumulate forever.
                    if (opus_packets.size() > file_data.size())
                        return -2;
                } else if (bid == EBML_BLOCK_GROUP) {
                    // Find Block element inside BlockGroup
                    EBMLReader bg(r.data, block_end);
                    bg.pos = r.pos;
                    while (bg.pos < block_end) {
                        uint32_t bgid = bg.read_id();
                        uint64_t bgsz = bg.read_vint();
                        if (bgsz == UINT64_MAX || bgid == 0)
                            break;
                        size_t bg_elem_end = bg.pos + (size_t)bgsz;
                        if (bg_elem_end > block_end)
                            break;
                        if (bgid == EBML_BLOCK) {
                            auto pkts = parse_block_packets(bg.ptr(), (size_t)bgsz, audio_track.track_number);
                            for (auto& p : pkts) {
                                opus_packets.emplace_back(p.data, p.data + p.size);
                            }
                            if (opus_packets.size() > file_data.size())
                                return -2;
                        }
                        bg.pos = bg_elem_end;
                    }
                }
                r.pos = block_end;
            }
        }
        r.pos = elem_end;
    }

    if (opus_packets.empty())
        return -2;

    // Decode the extracted packets
    if (is_opus) {
        int ch = audio_track.channels > 0 ? audio_track.channels : 2;
        if (ch > 2)
            ch = 2;

        // Parse OpusHead from codec_private to get channel count and pre-skip
        int pre_skip = 0;
        if (audio_track.codec_private.size() >= 19 &&
            std::memcmp(audio_track.codec_private.data(), "OpusHead", 8) == 0) {
            ch = audio_track.codec_private[9];
            pre_skip = (int)((uint16_t)audio_track.codec_private[10] | ((uint16_t)audio_track.codec_private[11] << 8));
            if (ch < 1)
                ch = 1;
            if (ch > 2)
                ch = 2;
        }

        std::vector<float> pcm_all;
        // Opus decodes at 48 kHz natively
        const int max_frame = 5760 * ch; // max 120ms at 48kHz
        std::vector<float> frame_buf((size_t)max_frame);

        // Decode the packets. Default: the in-tree glint packet decoder
        // (RFC-conformant, no libopus). CRISPASR_OPUS_DECODER=libopus selects
        // libopus (only when built with CRISPASR_OPUS).
        bool use_libopus = false;
#if defined(CRISPASR_HAVE_OPUS)
        if (const char* pref = std::getenv("CRISPASR_OPUS_DECODER"))
            use_libopus = pref[0] && std::strcmp(pref, "glint") != 0 && std::strcmp(pref, "auto") != 0;
#endif
        if (use_libopus) {
#if defined(CRISPASR_HAVE_OPUS)
            int opus_err = 0;
            OpusDecoder* opus_dec = opus_decoder_create(48000, ch, &opus_err);
            if (!opus_dec || opus_err != OPUS_OK)
                return -2;
            for (auto& pkt : opus_packets) {
                int n = opus_decode_float(opus_dec, pkt.data(), (opus_int32)pkt.size(), frame_buf.data(), 5760, 0);
                if (n > 0)
                    pcm_all.insert(pcm_all.end(), frame_buf.data(), frame_buf.data() + n * ch);
            }
            opus_decoder_destroy(opus_dec);
#endif
        } else {
            if (const char* e = std::getenv("CRISPASR_OPUS_DEBUG"); e && e[0] && e[0] != '0')
                std::fprintf(stderr, "[glint-webm-opus] %zu packets from %zu cluster(s), %d ch @ 48000\n",
                             opus_packets.size(), n_clusters, ch);
            glint_opus_dec_t gdec = glint_opus_dec_create(ch, 48000);
            if (!gdec)
                return -2;
            for (auto& pkt : opus_packets) {
                int n = glint_opus_decode(gdec, pkt.data(), (int)pkt.size(), frame_buf.data(), 5760);
                if (n > 0)
                    pcm_all.insert(pcm_all.end(), frame_buf.data(), frame_buf.data() + n * ch);
            }
            glint_opus_dec_destroy(gdec);
        }

        if (pcm_all.empty())
            return -2;

        // Skip pre-skip samples
        size_t total_samples = pcm_all.size() / (size_t)ch;
        size_t skip = (size_t)pre_skip < total_samples ? (size_t)pre_skip : 0;
        float* trimmed = pcm_all.data() + skip * ch;
        size_t trimmed_frames = total_samples - skip;

        // Downmix to requested channel count and resample 48k → 16k
        int out_ch = want_channels > 0 ? want_channels : 1;
        if (out_ch > 2)
            out_ch = 2;

        // First downmix if needed
        std::vector<float> mono;
        float* src = trimmed;
        int src_ch = ch;
        if (out_ch == 1 && ch > 1) {
            mono.resize(trimmed_frames);
            for (size_t i = 0; i < trimmed_frames; ++i) {
                float sum = 0;
                for (int c = 0; c < ch; ++c)
                    sum += trimmed[i * ch + c];
                mono[i] = sum / (float)ch;
            }
            src = mono.data();
            src_ch = 1;
        }

        // Resample 48000 → 16000
        ma_resampler_config rcfg = ma_resampler_config_init(ma_format_f32, (ma_uint32)src_ch, 48000, kTargetSampleRate,
                                                            ma_resample_algorithm_linear);
        ma_resampler resampler;
        if (ma_resampler_init(&rcfg, nullptr, &resampler) != MA_SUCCESS)
            return -2;

        ma_uint64 in_len = trimmed_frames;
        ma_uint64 out_len = 0;
        ma_resampler_get_expected_output_frame_count(&resampler, in_len, &out_len);
        out_len += 256;

        float* result = (float*)std::malloc(out_len * (size_t)src_ch * sizeof(float));
        if (!result) {
            ma_resampler_uninit(&resampler, nullptr);
            return -3;
        }

        ma_uint64 in_consumed = in_len;
        ma_uint64 out_produced = out_len;
        ma_resampler_process_pcm_frames(&resampler, src, &in_consumed, result, &out_produced);
        ma_resampler_uninit(&resampler, nullptr);

        *out_buf = result;
        *out_frames = (int)out_produced;
        *out_channels = src_ch;
        return 0;
    } else {
        // Vorbis: reconstruct a valid Ogg/Vorbis byte stream from the Matroska
        // codec_private (3 header packets) + data packets, then feed to stb_vorbis.
        //
        // Matroska codec_private for Vorbis:
        //   byte 0: N-1 where N = number of headers (always 2 → 3 headers)
        //   then Xiph-laced sizes of first N-1 headers
        //   then concatenated header packet data
        if (audio_track.codec_private.empty())
            return -2;

        const uint8_t* cp = audio_track.codec_private.data();
        size_t cp_size = audio_track.codec_private.size();
        if (cp_size < 3 || cp[0] != 2)
            return -2; // must have exactly 3 headers

        // Parse Xiph lacing for first two header sizes
        size_t pos = 1;
        size_t hdr_sizes[3] = {};
        for (int h = 0; h < 2; ++h) {
            while (pos < cp_size) {
                uint8_t b = cp[pos++];
                hdr_sizes[h] += b;
                if (b < 255)
                    break;
            }
        }
        size_t hdr_data_start = pos;
        size_t used = hdr_sizes[0] + hdr_sizes[1];
        if (hdr_data_start + used > cp_size)
            return -2;
        hdr_sizes[2] = cp_size - hdr_data_start - used;

        const uint8_t* hdr_ptrs[3];
        hdr_ptrs[0] = cp + hdr_data_start;
        hdr_ptrs[1] = hdr_ptrs[0] + hdr_sizes[0];
        hdr_ptrs[2] = hdr_ptrs[1] + hdr_sizes[1];

        // Ogg page builder helper
        // Each page: capture pattern(4) + version(1) + header_type(1) +
        //   granule_position(8) + serial(4) + page_sequence(4) + checksum(4) +
        //   page_segments(1) + segment_table(page_segments) + data
        struct OggBuilder {
            std::vector<uint8_t> data;
            uint32_t serial;
            uint32_t page_seq;
            int64_t granule;

            OggBuilder() : serial(0x43415352), page_seq(0), granule(0) {} // "CASR"

            // CRC-32 lookup table for Ogg (polynomial 0x04C11DB7)
            static uint32_t ogg_crc(const uint8_t* data, size_t len) {
                // Standard Ogg CRC table
                static const uint32_t crc_table[256] = {
                    0x00000000, 0x04C11DB7, 0x09823B6E, 0x0D4326D9, 0x130476DC, 0x17C56B6B, 0x1A864DB2, 0x1E475005,
                    0x2608EDB8, 0x22C9F00F, 0x2F8AD6D6, 0x2B4BCB61, 0x350C9B64, 0x31CD86D3, 0x3C8EA00A, 0x384FBDBD,
                    0x4C11DB70, 0x48D0C6C7, 0x4593E01E, 0x4152FDA9, 0x5F15ADAC, 0x5BD4B01B, 0x569796C2, 0x52568B75,
                    0x6A1936C8, 0x6ED82B7F, 0x639B0DA6, 0x675A1011, 0x791D4014, 0x7DDC5DA3, 0x709F7B7A, 0x745E66CD,
                    0x9823B6E0, 0x9CE2AB57, 0x91A18D8E, 0x95609039, 0x8B27C03C, 0x8FE6DD8B, 0x82A5FB52, 0x8664E6E5,
                    0xBE2B5B58, 0xBAEA46EF, 0xB7A96036, 0xB3687D81, 0xAD2F2D84, 0xA9EE3033, 0xA4AD16EA, 0xA06C0B5D,
                    0xD4326D90, 0xD0F37027, 0xDDB056FE, 0xD9714B49, 0xC7361B4C, 0xC3F706FB, 0xCEB42022, 0xCA753D95,
                    0xF23A8028, 0xF6FB9D9F, 0xFBB8BB46, 0xFF79A6F1, 0xE13EF6F4, 0xE5FFEB43, 0xE8BCCD9A, 0xEC7DD02D,
                    0x34867077, 0x30476DC0, 0x3D044B19, 0x39C556AE, 0x278206AB, 0x23431B1C, 0x2E003DC5, 0x2AC12072,
                    0x128E9DCF, 0x164F8078, 0x1B0CA6A1, 0x1FCDBB16, 0x018AEB13, 0x054BF6A4, 0x0808D07D, 0x0CC9CDCA,
                    0x7897AB07, 0x7C56B6B0, 0x71159069, 0x75D48DDE, 0x6B93DDDB, 0x6F52C06C, 0x6211E6B5, 0x66D0FB02,
                    0x5E9F46BF, 0x5A5E5B08, 0x571D7DD1, 0x53DC6066, 0x4D9B3063, 0x495A2DD4, 0x44190B0D, 0x40D816BA,
                    0xACA5C697, 0xA864DB20, 0xA527FDF9, 0xA1E6E04E, 0xBFA1B04B, 0xBB60ADFC, 0xB6238B25, 0xB2E29692,
                    0x8AAD2B2F, 0x8E6C3698, 0x832F1041, 0x87EE0DF6, 0x99A95DF3, 0x9D684044, 0x902B669D, 0x94EA7B2A,
                    0xE0B41DE7, 0xE4750050, 0xE9362689, 0xEDF73B3E, 0xF3B06B3B, 0xF771768C, 0xFA325055, 0xFEF34DE2,
                    0xC6BCF05F, 0xC27DEDE8, 0xCF3ECB31, 0xCBFFD686, 0xD5B88683, 0xD1799B34, 0xDC3ABDED, 0xD8FBA05A,
                    0x690CE0EE, 0x6DCDFD59, 0x608EDB80, 0x644FC637, 0x7A089632, 0x7EC98B85, 0x738AAD5C, 0x774BB0EB,
                    0x4F040D56, 0x4BC510E1, 0x46863638, 0x42472B8F, 0x5C007B8A, 0x58C1663D, 0x558240E4, 0x51435D53,
                    0x251D3B9E, 0x21DC2629, 0x2C9F00F0, 0x285E1D47, 0x36194D42, 0x32D850F5, 0x3F9B762C, 0x3B5A6B9B,
                    0x0315D626, 0x07D4CB91, 0x0A97ED48, 0x0E56F0FF, 0x1011A0FA, 0x14D0BD4D, 0x19939B94, 0x1D528623,
                    0xF12F560E, 0xF5EE4BB9, 0xF8AD6D60, 0xFC6C70D7, 0xE22B20D2, 0xE6EA3D65, 0xEBA91BBC, 0xEF68060B,
                    0xD727BBB6, 0xD3E6A601, 0xDEA580D8, 0xDA649D6F, 0xC423CD6A, 0xC0E2D0DD, 0xCDA1F604, 0xC960EBB3,
                    0xBD3E8D7E, 0xB9FF90C9, 0xB4BCB610, 0xB07DABA7, 0xAE3AFBA2, 0xAAFBE615, 0xA7B8C0CC, 0xA379DD7B,
                    0x9B3660C6, 0x9FF77D71, 0x92B45BA8, 0x9675461F, 0x8832161A, 0x8CF30BAD, 0x81B02D74, 0x857130C3,
                    0x5D8A9099, 0x594B8D2E, 0x5408ABF7, 0x50C9B640, 0x4E8EE645, 0x4A4FFBF2, 0x470CDD2B, 0x43CDC09C,
                    0x7B827D21, 0x7F436096, 0x7200464F, 0x76C15BF8, 0x68860BFD, 0x6C47164A, 0x61043093, 0x65C52D24,
                    0x119B4BE9, 0x155A565E, 0x18197087, 0x1CD86D30, 0x029F3D35, 0x065E2082, 0x0B1D065B, 0x0FDC1BEC,
                    0x3793A651, 0x3352BBE6, 0x3E119D3F, 0x3AD08088, 0x2497D08D, 0x2056CD3A, 0x2D15EBE3, 0x29D4F654,
                    0xC5A92679, 0xC1683BCE, 0xCC2B1D17, 0xC8EA00A0, 0xD6AD50A5, 0xD26C4D12, 0xDF2F6BCB, 0xDBEE767C,
                    0xE3A1CBC1, 0xE760D676, 0xEA23F0AF, 0xEEE2ED18, 0xF0A5BD1D, 0xF464A0AA, 0xF9278673, 0xFDE69BC4,
                    0x89B8FD09, 0x8D79E0BE, 0x803AC667, 0x84FBDBD0, 0x9ABC8BD5, 0x9E7D9662, 0x933EB0BB, 0x97FFAD0C,
                    0xAFB010B1, 0xAB710D06, 0xA6322BDF, 0xA2F33668, 0xBCB4666D, 0xB8757BDA, 0xB5365D03, 0xB1F740B4,
                };
                uint32_t crc = 0;
                for (size_t i = 0; i < len; ++i)
                    crc = (crc << 8) ^ crc_table[((crc >> 24) ^ data[i]) & 0xFF];
                return crc;
            }

            void write_page(const uint8_t* pkt_data, size_t pkt_size, uint8_t header_type, int64_t gran) {
                // Split packet into 255-byte segments + remainder
                size_t n_segs = pkt_size / 255 + 1;
                size_t hdr_size = 27 + n_segs;
                size_t page_start = data.size();
                data.resize(page_start + hdr_size + pkt_size);

                uint8_t* h = data.data() + page_start;
                // Capture pattern
                h[0] = 'O';
                h[1] = 'g';
                h[2] = 'g';
                h[3] = 'S';
                h[4] = 0; // version
                h[5] = header_type;
                // Granule position (little-endian int64)
                for (int i = 0; i < 8; ++i)
                    h[6 + i] = (uint8_t)((gran >> (i * 8)) & 0xFF);
                // Serial number (little-endian)
                for (int i = 0; i < 4; ++i)
                    h[14 + i] = (uint8_t)((serial >> (i * 8)) & 0xFF);
                // Page sequence number (little-endian)
                for (int i = 0; i < 4; ++i)
                    h[18 + i] = (uint8_t)((page_seq >> (i * 8)) & 0xFF);
                // Checksum placeholder (filled after)
                h[22] = h[23] = h[24] = h[25] = 0;
                // Number of segments
                h[26] = (uint8_t)n_segs;
                // Segment table
                size_t remaining = pkt_size;
                for (size_t s = 0; s < n_segs; ++s) {
                    if (remaining >= 255) {
                        h[27 + s] = 255;
                        remaining -= 255;
                    } else {
                        h[27 + s] = (uint8_t)remaining;
                        remaining = 0;
                    }
                }
                // Packet data (the EOS page is empty: pkt_data=nullptr, pkt_size=0
                // — memcpy(dst, nullptr, 0) is UB, so guard it).
                if (pkt_data && pkt_size > 0)
                    std::memcpy(data.data() + page_start + hdr_size, pkt_data, pkt_size);

                // Compute and fill CRC
                uint32_t crc = ogg_crc(data.data() + page_start, hdr_size + pkt_size);
                h[22] = (uint8_t)(crc & 0xFF);
                h[23] = (uint8_t)((crc >> 8) & 0xFF);
                h[24] = (uint8_t)((crc >> 16) & 0xFF);
                h[25] = (uint8_t)((crc >> 24) & 0xFF);

                ++page_seq;
            }
        };

        OggBuilder ogg;

        // Write header pages: identification (BOS), comment, setup
        ogg.write_page(hdr_ptrs[0], hdr_sizes[0], 0x02, 0); // BOS flag
        ogg.write_page(hdr_ptrs[1], hdr_sizes[1], 0x00, 0); // comment
        ogg.write_page(hdr_ptrs[2], hdr_sizes[2], 0x00, 0); // setup

        // Parse Vorbis identification header to get sample rate and channels
        int vorbis_ch = 1;
        uint32_t vorbis_rate = 44100;
        if (hdr_sizes[0] >= 30 && std::memcmp(hdr_ptrs[0] + 1, "vorbis", 6) == 0) {
            vorbis_ch = hdr_ptrs[0][11];
            vorbis_rate = (uint32_t)hdr_ptrs[0][12] | ((uint32_t)hdr_ptrs[0][13] << 8) |
                          ((uint32_t)hdr_ptrs[0][14] << 16) | ((uint32_t)hdr_ptrs[0][15] << 24);
        }

        // Write audio data pages. stb_vorbis can handle multiple packets
        // per page but for simplicity we write one packet per page.
        // Each Vorbis audio packet typically decodes to blocksize/2 samples
        // but we don't know the exact count; use a running granule estimate.
        // stb_vorbis is tolerant of approximate granule positions.
        int64_t granule = 0;
        int samples_per_packet = 1024;   // approximate; Vorbis block size varies
        for (auto& pkt : opus_packets) { // reusing the vector name
            granule += samples_per_packet;
            ogg.write_page(pkt.data(), pkt.size(), 0x00, granule);
        }

        // Write EOS page (empty, with EOS flag)
        ogg.write_page(nullptr, 0, 0x04, granule);

        // Decode via stb_vorbis
        int stb_ch = 0, stb_rate = 0;
        short* stb_pcm = nullptr;
        int stb_samples = stb_vorbis_decode_memory(ogg.data.data(), (int)ogg.data.size(), &stb_ch, &stb_rate, &stb_pcm);
        if (stb_samples <= 0 || !stb_pcm)
            return -2;

        // Convert s16 → f32 and downmix
        int out_ch = want_channels > 0 ? want_channels : 1;
        if (out_ch > 2)
            out_ch = 2;

        size_t n_frames = (size_t)stb_samples;
        std::vector<float> f32(n_frames);
        if (stb_ch >= 2 && out_ch == 1) {
            for (size_t i = 0; i < n_frames; ++i)
                f32[i] = ((float)stb_pcm[i * stb_ch] + (float)stb_pcm[i * stb_ch + 1]) / (2.0f * 32768.0f);
        } else {
            for (size_t i = 0; i < n_frames; ++i)
                f32[i] = (float)stb_pcm[i * stb_ch] / 32768.0f;
        }
        free(stb_pcm);

        // Resample to 16 kHz
        if ((uint32_t)stb_rate != (uint32_t)kTargetSampleRate) {
            ma_resampler_config rcfg = ma_resampler_config_init(ma_format_f32, 1, (ma_uint32)stb_rate,
                                                                kTargetSampleRate, ma_resample_algorithm_linear);
            ma_resampler resampler;
            if (ma_resampler_init(&rcfg, nullptr, &resampler) != MA_SUCCESS)
                return -2;

            ma_uint64 in_len = n_frames;
            ma_uint64 out_len_est = 0;
            ma_resampler_get_expected_output_frame_count(&resampler, in_len, &out_len_est);
            out_len_est += 256;

            float* result = (float*)std::malloc(out_len_est * sizeof(float));
            if (!result) {
                ma_resampler_uninit(&resampler, nullptr);
                return -3;
            }

            ma_uint64 in_consumed = in_len;
            ma_uint64 out_produced = out_len_est;
            ma_resampler_process_pcm_frames(&resampler, f32.data(), &in_consumed, result, &out_produced);
            ma_resampler_uninit(&resampler, nullptr);

            *out_buf = result;
            *out_frames = (int)out_produced;
            *out_channels = 1;
            return 0;
        }

        float* result = (float*)std::malloc(n_frames * sizeof(float));
        if (!result)
            return -3;
        std::memcpy(result, f32.data(), n_frames * sizeof(float));
        *out_buf = result;
        *out_frames = (int)n_frames;
        *out_channels = 1;
        return 0;
    }
}
} // namespace

// ── M4A/AAC decode via fdk-aac (runtime dlopen, MIT-clean) ──────────────────
// Minimal ISOBMFF (MP4/M4A) parser extracts AudioSpecificConfig + raw AAC
// frames. If libfdk-aac.so is installed on the system, it is loaded at
// runtime via dlopen and used to decode the AAC frames. The project binary
// itself never links against fdk-aac — the license responsibility lies with
// the user who installs the library. On Apple, AudioToolbox is preferred;
// on Windows, Media Foundation; on Android, NDK MediaCodec.
//
// Not compiled on Apple (AudioToolbox covers it) or when targeting WASM.
#if !defined(__APPLE__) && !defined(__EMSCRIPTEN__)

namespace {

// ── fdk-aac dlopen wrapper (no compile-time dependency) ──────────────────────
// Mirror just the types and function signatures we need from aacdecoder_lib.h.
// Loaded at runtime via dlopen("libfdk-aac.so.2") — if the library isn't
// installed, crispasr_m4a_decode returns -2 and the caller falls through to
// the next fallback or gives up.
#if !defined(_WIN32) // dlopen is POSIX; Windows uses LoadLibrary below
#include <dlfcn.h>
#endif

typedef void* FDK_HANDLE; // opaque HANDLE_AACDECODER
typedef int FDK_ERR;      // AAC_DECODER_ERROR
typedef short FDK_PCM;    // INT_PCM = SHORT
typedef unsigned char FDK_UCHAR;
typedef unsigned int FDK_UINT;
typedef int FDK_INT;

// CStreamInfo — we only access the first 3 int fields.
struct FDK_StreamInfo {
    FDK_INT sampleRate;
    FDK_INT frameSize;
    FDK_INT numChannels;
    // ... more fields follow that we don't use
};

// Transport type constants
static constexpr int FDK_TT_MP4_RAW = 0;
static constexpr int FDK_TT_MP4_ADTS = 2;
// Error codes
static constexpr FDK_ERR FDK_AAC_DEC_OK = 0x0000;
static constexpr FDK_ERR FDK_AAC_DEC_NOT_ENOUGH_BITS = 0x1002;

// Function pointer types
using fn_aacDecoder_Open = FDK_HANDLE (*)(int transport_type, FDK_UINT nr_of_layers);
using fn_aacDecoder_ConfigRaw = FDK_ERR (*)(FDK_HANDLE, FDK_UCHAR** conf, FDK_UINT* length);
using fn_aacDecoder_Fill = FDK_ERR (*)(FDK_HANDLE, FDK_UCHAR** pBuffer, FDK_UINT* bufferSize, FDK_UINT* bytesValid);
using fn_aacDecoder_DecodeFrame = FDK_ERR (*)(FDK_HANDLE, FDK_PCM* pTimeData, int timeDataSize, FDK_UINT flags);
using fn_aacDecoder_GetStreamInfo = FDK_StreamInfo* (*)(FDK_HANDLE);
using fn_aacDecoder_Close = void (*)(FDK_HANDLE);

struct FdkAacLib {
    void* handle = nullptr;
    fn_aacDecoder_Open Open = nullptr;
    fn_aacDecoder_ConfigRaw ConfigRaw = nullptr;
    fn_aacDecoder_Fill Fill = nullptr;
    fn_aacDecoder_DecodeFrame DecodeFrame = nullptr;
    fn_aacDecoder_GetStreamInfo GetStreamInfo = nullptr;
    fn_aacDecoder_Close Close = nullptr;
    bool tried = false;
    bool ok = false;

    bool load() {
        if (tried)
            return ok;
        tried = true;
#if defined(_WIN32)
        handle = (void*)LoadLibraryA("libfdk-aac-2.dll");
        if (!handle)
            handle = (void*)LoadLibraryA("fdk-aac.dll");
#define DLSYM(h, name) (void*)GetProcAddress((HMODULE)(h), (name))
#else
        handle = dlopen("libfdk-aac.so.2", RTLD_NOW | RTLD_LOCAL);
        if (!handle)
            handle = dlopen("libfdk-aac.so", RTLD_NOW | RTLD_LOCAL);
#define DLSYM(h, name) dlsym((h), (name))
#endif
        if (!handle)
            return false;
        Open = (fn_aacDecoder_Open)DLSYM(handle, "aacDecoder_Open");
        ConfigRaw = (fn_aacDecoder_ConfigRaw)DLSYM(handle, "aacDecoder_ConfigRaw");
        Fill = (fn_aacDecoder_Fill)DLSYM(handle, "aacDecoder_Fill");
        DecodeFrame = (fn_aacDecoder_DecodeFrame)DLSYM(handle, "aacDecoder_DecodeFrame");
        GetStreamInfo = (fn_aacDecoder_GetStreamInfo)DLSYM(handle, "aacDecoder_GetStreamInfo");
        Close = (fn_aacDecoder_Close)DLSYM(handle, "aacDecoder_Close");
#undef DLSYM
        ok = Open && ConfigRaw && Fill && DecodeFrame && GetStreamInfo && Close;
        return ok;
    }
};

static FdkAacLib& fdk_aac() {
    static FdkAacLib lib;
    return lib;
}

// Read 32-bit big-endian from buffer
static inline uint32_t mp4_be32(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
}
static inline uint64_t mp4_be64(const uint8_t* p) {
    return ((uint64_t)mp4_be32(p) << 32) | mp4_be32(p + 4);
}
static inline uint16_t mp4_be16(const uint8_t* p) {
    return (uint16_t)((p[0] << 8) | p[1]);
}

struct MP4Box {
    uint32_t type;
    uint64_t size;     // total box size including header
    uint64_t data_off; // offset of payload after header
};

// Read box header at position `pos` in buffer; returns false at EOF.
static bool mp4_read_box(const uint8_t* data, size_t data_size, size_t pos, MP4Box& box) {
    if (pos + 8 > data_size)
        return false;
    uint32_t sz32 = mp4_be32(data + pos);
    box.type = mp4_be32(data + pos + 4);
    if (sz32 == 1) {
        // 64-bit extended size
        if (pos + 16 > data_size)
            return false;
        box.size = mp4_be64(data + pos + 8);
        box.data_off = pos + 16;
    } else if (sz32 == 0) {
        // Box extends to EOF
        box.size = data_size - pos;
        box.data_off = pos + 8;
    } else {
        box.size = sz32;
        box.data_off = pos + 8;
    }
    return box.size >= 8 && pos + box.size <= data_size;
}

// 4-char code as uint32
static constexpr uint32_t FOURCC(char a, char b, char c, char d) {
    return ((uint32_t)(uint8_t)a << 24) | ((uint32_t)(uint8_t)b << 16) | ((uint32_t)(uint8_t)c << 8) | (uint8_t)d;
}

struct MP4AudioInfo {
    std::vector<uint8_t> asc; // AudioSpecificConfig
    uint32_t sample_rate = 0;
    uint32_t channels = 0;
    std::vector<uint32_t> sample_sizes;
    std::vector<uint64_t> chunk_offsets;
    // samples_per_chunk table entries
    struct StscEntry {
        uint32_t first_chunk;
        uint32_t samples_per_chunk;
    };
    std::vector<StscEntry> stsc;
    // Encoder priming: number of PCM samples to skip from the decoded
    // output (from the elst edit list media_time field). AAC encoders
    // emit 1024–2048 priming samples that should be trimmed.
    int64_t priming_samples = 0;
};

// Parse `esds` box to extract AudioSpecificConfig
static bool parse_esds(const uint8_t* data, size_t size, std::vector<uint8_t>& asc) {
    // Skip version(4) then parse ES_Descriptor tags
    if (size < 4)
        return false;
    size_t pos = 4; // skip version+flags

    auto read_desc_tag = [&](uint8_t& tag, uint32_t& len) -> bool {
        if (pos >= size)
            return false;
        tag = data[pos++];
        len = 0;
        for (int i = 0; i < 4 && pos < size; ++i) {
            uint8_t b = data[pos++];
            len = (len << 7) | (b & 0x7F);
            if (!(b & 0x80))
                break;
        }
        return true;
    };

    // ES_Descriptor (tag 3)
    uint8_t tag;
    uint32_t len;
    if (!read_desc_tag(tag, len) || tag != 3)
        return false;
    if (pos + 2 > size)
        return false;
    pos += 2; // ES_ID
    uint8_t flags = (pos < size) ? data[pos++] : 0;
    if (flags & 0x80)
        pos += 2; // streamDependenceFlag
    if (flags & 0x40) {
        // URL_Flag
        if (pos < size) {
            uint8_t url_len = data[pos++];
            pos += url_len;
        }
    }
    if (flags & 0x20)
        pos += 2; // OCR_ES_Id

    // DecoderConfigDescriptor (tag 4)
    if (!read_desc_tag(tag, len) || tag != 4)
        return false;
    if (pos + 13 > size)
        return false;
    pos += 13; // skip DecoderConfigDescriptor fixed fields

    // DecoderSpecificInfo (tag 5) — this IS the AudioSpecificConfig
    if (!read_desc_tag(tag, len) || tag != 5)
        return false;
    if (pos + len > size)
        return false;
    asc.assign(data + pos, data + pos + len);
    return !asc.empty();
}

// Find first audio track and extract info
static bool mp4_parse_audio(const uint8_t* data, size_t data_size, MP4AudioInfo& info) {
    // Walk top-level boxes to find moov
    size_t pos = 0;
    MP4Box box;
    size_t moov_start = 0, moov_end = 0;
    while (mp4_read_box(data, data_size, pos, box)) {
        if (box.type == FOURCC('m', 'o', 'o', 'v')) {
            moov_start = box.data_off;
            moov_end = pos + box.size;
            break;
        }
        pos += box.size;
    }
    if (moov_end == 0)
        return false;

    // Find first audio trak in moov
    pos = moov_start;
    while (pos < moov_end && mp4_read_box(data, data_size, pos, box)) {
        if (box.type == FOURCC('t', 'r', 'a', 'k')) {
            size_t trak_end = pos + box.size;
            // Check if this trak has an audio mdia/hdlr
            bool is_audio = false;
            size_t mdia_start = 0, mdia_end = 0;
            size_t p2 = box.data_off;
            while (p2 < trak_end) {
                MP4Box b2;
                if (!mp4_read_box(data, data_size, p2, b2))
                    break;
                if (b2.type == FOURCC('m', 'd', 'i', 'a')) {
                    mdia_start = b2.data_off;
                    mdia_end = p2 + b2.size;
                    // Look for hdlr
                    size_t p3 = mdia_start;
                    while (p3 < mdia_end) {
                        MP4Box b3;
                        if (!mp4_read_box(data, data_size, p3, b3))
                            break;
                        if (b3.type == FOURCC('h', 'd', 'l', 'r') && b3.data_off + 12 <= p3 + b3.size) {
                            uint32_t handler = mp4_be32(data + b3.data_off + 8);
                            if (handler == FOURCC('s', 'o', 'u', 'n'))
                                is_audio = true;
                        }
                        p3 += b3.size;
                    }
                    break;
                }
                p2 += b2.size;
            }
            if (!is_audio || mdia_end == 0) {
                pos += box.size;
                continue;
            }

            // Parse edts/elst for encoder priming (sibling of mdia in trak)
            p2 = box.data_off;
            while (p2 < trak_end) {
                MP4Box b2e;
                if (!mp4_read_box(data, data_size, p2, b2e))
                    break;
                if (b2e.type == FOURCC('e', 'd', 't', 's')) {
                    size_t edts_end = p2 + b2e.size;
                    size_t pe = b2e.data_off;
                    while (pe < edts_end) {
                        MP4Box elst;
                        if (!mp4_read_box(data, data_size, pe, elst))
                            break;
                        if (elst.type == FOURCC('e', 'l', 's', 't')) {
                            size_t ep = elst.data_off;
                            size_t elst_end = pe + elst.size;
                            if (ep + 8 <= elst_end) {
                                uint8_t ver = data[ep];
                                uint32_t count = mp4_be32(data + ep + 4);
                                ep += 8;
                                // First entry with media_time >= 0 gives the priming skip
                                for (uint32_t ei = 0; ei < count; ++ei) {
                                    int64_t media_time;
                                    if (ver == 1) {
                                        if (ep + 20 > elst_end)
                                            break;
                                        ep += 8; // skip segment_duration
                                        media_time = (int64_t)mp4_be64(data + ep);
                                        ep += 8 + 4; // media_time + rate
                                    } else {
                                        if (ep + 12 > elst_end)
                                            break;
                                        ep += 4; // skip segment_duration
                                        media_time = (int32_t)mp4_be32(data + ep);
                                        ep += 4 + 4; // media_time + rate
                                    }
                                    if (media_time > 0) {
                                        info.priming_samples = media_time;
                                        break;
                                    }
                                }
                            }
                        }
                        pe += elst.size;
                    }
                    break;
                }
                p2 += b2e.size;
            }

            // Found audio mdia — find minf/stbl
            size_t minf_start = 0, minf_end = 0;
            size_t p3 = mdia_start;
            while (p3 < mdia_end) {
                MP4Box b3;
                if (!mp4_read_box(data, data_size, p3, b3))
                    break;
                if (b3.type == FOURCC('m', 'i', 'n', 'f')) {
                    minf_start = b3.data_off;
                    minf_end = p3 + b3.size;
                    break;
                }
                p3 += b3.size;
            }
            if (minf_end == 0) {
                pos += box.size;
                continue;
            }

            // Find stbl inside minf
            size_t stbl_start = 0, stbl_end = 0;
            size_t p4 = minf_start;
            while (p4 < minf_end) {
                MP4Box b4;
                if (!mp4_read_box(data, data_size, p4, b4))
                    break;
                if (b4.type == FOURCC('s', 't', 'b', 'l')) {
                    stbl_start = b4.data_off;
                    stbl_end = p4 + b4.size;
                    break;
                }
                p4 += b4.size;
            }
            if (stbl_end == 0) {
                pos += box.size;
                continue;
            }

            // Parse stbl children: stsd (AudioSpecificConfig), stsz, stco/co64, stsc
            size_t p5 = stbl_start;
            while (p5 < stbl_end) {
                MP4Box b5;
                if (!mp4_read_box(data, data_size, p5, b5))
                    break;
                size_t bstart = b5.data_off;
                size_t bend = p5 + b5.size;

                if (b5.type == FOURCC('s', 't', 's', 'd')) {
                    // Sample Description — find mp4a/esds
                    // stsd: version(4) + entry_count(4) + entries
                    if (bstart + 8 <= bend) {
                        size_t entry_pos = bstart + 8;
                        MP4Box ent;
                        if (mp4_read_box(data, data_size, entry_pos, ent) &&
                            (ent.type == FOURCC('m', 'p', '4', 'a') || ent.type == FOURCC('.', 'm', 'p', '3'))) {
                            // mp4a box: 6 reserved + 2 data_ref_index + 8 reserved + 2 channels + 2 sample_size +
                            //   2 compression_id + 2 packet_size + 4 sample_rate (16.16)
                            size_t mp4a_data = ent.data_off;
                            if (mp4a_data + 28 <= entry_pos + ent.size) {
                                info.channels = mp4_be16(data + mp4a_data + 16);
                                // sample_rate is 16.16 fixed point at offset 24
                                info.sample_rate = mp4_be16(data + mp4a_data + 24);
                            }
                            // Find esds inside mp4a
                            size_t p6 = mp4a_data + 28;
                            size_t mp4a_end = entry_pos + ent.size;
                            while (p6 < mp4a_end) {
                                MP4Box b6;
                                if (!mp4_read_box(data, data_size, p6, b6))
                                    break;
                                if (b6.type == FOURCC('e', 's', 'd', 's')) {
                                    parse_esds(data + b6.data_off, (p6 + b6.size) - b6.data_off, info.asc);
                                }
                                p6 += b6.size;
                            }
                        }
                    }
                } else if (b5.type == FOURCC('s', 't', 's', 'z')) {
                    // Sample size box: version(4) + sample_size(4) + sample_count(4) + [sizes]
                    if (bstart + 12 <= bend) {
                        uint32_t uniform_size = mp4_be32(data + bstart + 4);
                        uint32_t count = mp4_be32(data + bstart + 8);
                        // Clamp the untrusted count so a crafted box can't drive a
                        // multi-GB resize(): the non-uniform table needs 4 bytes per
                        // entry (bounded by the box), and even a uniform table can't
                        // describe more samples than the file has bytes.
                        size_t max_count = (uniform_size != 0) ? data_size : (bend - (bstart + 12)) / 4;
                        if (count > max_count)
                            count = (uint32_t)max_count;
                        info.sample_sizes.resize(count);
                        if (uniform_size != 0) {
                            for (uint32_t i = 0; i < count; ++i)
                                info.sample_sizes[i] = uniform_size;
                        } else {
                            for (uint32_t i = 0; i < count && bstart + 12 + (i + 1) * 4 <= bend; ++i)
                                info.sample_sizes[i] = mp4_be32(data + bstart + 12 + i * 4);
                        }
                    }
                } else if (b5.type == FOURCC('s', 't', 'c', 'o')) {
                    // Chunk offset box (32-bit): version(4) + count(4) + [offsets]
                    if (bstart + 8 <= bend) {
                        uint32_t count = mp4_be32(data + bstart + 4);
                        // Clamp: 4 bytes per offset, bounded by the box's remaining bytes.
                        size_t max_count = (bend - (bstart + 8)) / 4;
                        if (count > max_count)
                            count = (uint32_t)max_count;
                        info.chunk_offsets.resize(count);
                        for (uint32_t i = 0; i < count && bstart + 8 + (i + 1) * 4 <= bend; ++i)
                            info.chunk_offsets[i] = mp4_be32(data + bstart + 8 + i * 4);
                    }
                } else if (b5.type == FOURCC('c', 'o', '6', '4')) {
                    // Chunk offset box (64-bit): version(4) + count(4) + [offsets]
                    if (bstart + 8 <= bend) {
                        uint32_t count = mp4_be32(data + bstart + 4);
                        // Clamp: 8 bytes per offset, bounded by the box's remaining bytes.
                        size_t max_count = (bend - (bstart + 8)) / 8;
                        if (count > max_count)
                            count = (uint32_t)max_count;
                        info.chunk_offsets.resize(count);
                        for (uint32_t i = 0; i < count && bstart + 8 + (i + 1) * 8 <= bend; ++i)
                            info.chunk_offsets[i] = mp4_be64(data + bstart + 8 + i * 8);
                    }
                } else if (b5.type == FOURCC('s', 't', 's', 'c')) {
                    // Sample-to-chunk box: version(4) + count(4) + [entries(12 each)]
                    if (bstart + 8 <= bend) {
                        uint32_t count = mp4_be32(data + bstart + 4);
                        for (uint32_t i = 0; i < count && bstart + 8 + (i + 1) * 12 <= bend; ++i) {
                            const uint8_t* e = data + bstart + 8 + i * 12;
                            info.stsc.push_back({mp4_be32(e), mp4_be32(e + 4)});
                        }
                    }
                }
                p5 += b5.size;
            }
            return !info.asc.empty() && !info.sample_sizes.empty();
        }
        pos += box.size;
    }
    return false;
}

// Determine the file offset of each AAC sample using stco + stsc + stsz
static std::vector<uint64_t> mp4_sample_offsets(const MP4AudioInfo& info) {
    std::vector<uint64_t> offsets(info.sample_sizes.size());
    if (info.chunk_offsets.empty() || info.stsc.empty())
        return offsets;

    size_t sample_idx = 0;
    for (size_t chunk_idx = 0; chunk_idx < info.chunk_offsets.size() && sample_idx < info.sample_sizes.size();
         ++chunk_idx) {
        // Find how many samples in this chunk
        uint32_t samples_in_chunk = 0;
        for (size_t s = 0; s < info.stsc.size(); ++s) {
            if (info.stsc[s].first_chunk - 1 <= (uint32_t)chunk_idx) {
                samples_in_chunk = info.stsc[s].samples_per_chunk;
            }
        }
        if (samples_in_chunk == 0)
            samples_in_chunk = 1;

        uint64_t off = info.chunk_offsets[chunk_idx];
        for (uint32_t s = 0; s < samples_in_chunk && sample_idx < info.sample_sizes.size(); ++s) {
            offsets[sample_idx] = off;
            off += info.sample_sizes[sample_idx];
            ++sample_idx;
        }
    }
    return offsets;
}

int crispasr_m4a_decode(const char* path, int want_channels, float** out_buf, int* out_frames, int* out_channels) {
    // Check for ADTS AAC first (raw .aac files without MP4 container)
    FILE* f = std::fopen(path, "rb");
    if (!f)
        return -2;
    std::fseek(f, 0, SEEK_END);
    long fsize = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (fsize <= 0 || fsize > 500 * 1024 * 1024) {
        std::fclose(f);
        return -2;
    }
    std::vector<uint8_t> file_data((size_t)fsize);
    if (std::fread(file_data.data(), 1, (size_t)fsize, f) != (size_t)fsize) {
        std::fclose(f);
        return -2;
    }
    std::fclose(f);

    bool is_adts = (fsize >= 2 && file_data[0] == 0xFF && (file_data[1] & 0xF0) == 0xF0);
    bool is_mp4 = false;
    if (!is_adts && fsize >= 8) {
        uint32_t box_type = mp4_be32(file_data.data() + 4);
        is_mp4 = (box_type == FOURCC('f', 't', 'y', 'p') || box_type == FOURCC('m', 'o', 'o', 'v') ||
                  box_type == FOURCC('m', 'd', 'a', 't') || box_type == FOURCC('f', 'r', 'e', 'e') ||
                  box_type == FOURCC('s', 'k', 'i', 'p') || box_type == FOURCC('w', 'i', 'd', 'e'));
    }
    if (!is_adts && !is_mp4)
        return -2;

    // Try to load fdk-aac at runtime
    FdkAacLib& fdk = fdk_aac();
    if (!fdk.load())
        return -2; // libfdk-aac not installed — caller falls through

    FDK_HANDLE dec = fdk.Open(is_adts ? FDK_TT_MP4_ADTS : FDK_TT_MP4_RAW, 1);
    if (!dec)
        return -2;

    // For MP4/M4A, parse container and configure decoder with AudioSpecificConfig
    MP4AudioInfo mp4info;
    std::vector<uint64_t> sample_offsets;
    uint32_t dec_sample_rate = 0;
    int dec_channels = 0;

    if (is_mp4) {
        if (!mp4_parse_audio(file_data.data(), file_data.size(), mp4info)) {
            fdk.Close(dec);
            return -2;
        }
        // Configure decoder with ASC
        FDK_UCHAR* asc_buf = mp4info.asc.data();
        FDK_UINT asc_size = (FDK_UINT)mp4info.asc.size();
        if (fdk.ConfigRaw(dec, &asc_buf, &asc_size) != FDK_AAC_DEC_OK) {
            fdk.Close(dec);
            return -2;
        }
        sample_offsets = mp4_sample_offsets(mp4info);
        dec_sample_rate = mp4info.sample_rate;
        dec_channels = (int)mp4info.channels;
    }

    // Decode
    std::vector<int16_t> pcm_all;
    FDK_PCM decode_buf[8 * 2048]; // max frame size

    if (is_adts) {
        // Feed entire ADTS stream in chunks
        FDK_UCHAR* input = file_data.data();
        FDK_UINT bytes_remaining = (FDK_UINT)file_data.size();
        FDK_UINT bytes_valid = bytes_remaining;

        while (bytes_valid > 0) {
            FDK_UCHAR* in_ptr = input + (bytes_remaining - bytes_valid);
            FDK_UINT in_size = bytes_valid;
            FDK_ERR err = fdk.Fill(dec, &in_ptr, &in_size, &bytes_valid);
            if (err != FDK_AAC_DEC_OK)
                break;

            while (true) {
                err = fdk.DecodeFrame(dec, decode_buf, sizeof(decode_buf) / sizeof(decode_buf[0]), 0);
                if (err == FDK_AAC_DEC_NOT_ENOUGH_BITS)
                    break;
                if (err != FDK_AAC_DEC_OK)
                    break;

                FDK_StreamInfo* si = fdk.GetStreamInfo(dec);
                if (si && si->numChannels > 0 && si->frameSize > 0) {
                    dec_sample_rate = (uint32_t)si->sampleRate;
                    dec_channels = si->numChannels;
                    for (int i = 0; i < si->frameSize; ++i) {
                        int32_t sum = 0;
                        for (int c = 0; c < si->numChannels; ++c)
                            sum += decode_buf[i * si->numChannels + c];
                        pcm_all.push_back((int16_t)(sum / si->numChannels));
                    }
                }
            }
        }
    } else {
        // MP4: feed each sample individually
        for (size_t i = 0; i < mp4info.sample_sizes.size(); ++i) {
            uint64_t off = (i < sample_offsets.size()) ? sample_offsets[i] : 0;
            uint32_t sz = mp4info.sample_sizes[i];
            // Overflow-safe bounds check: `off + sz` can wrap in uint64 for a
            // co64 offset near UINT64_MAX, defeating the guard and yielding a
            // wild `file_data.data() + off` pointer. Compare subtractively.
            if (off > file_data.size() || sz > file_data.size() - off)
                continue;

            FDK_UCHAR* in_ptr = file_data.data() + off;
            FDK_UINT in_size = sz;
            FDK_UINT bytes_valid = sz;
            FDK_ERR err = fdk.Fill(dec, &in_ptr, &in_size, &bytes_valid);
            if (err != FDK_AAC_DEC_OK)
                continue;

            err = fdk.DecodeFrame(dec, decode_buf, sizeof(decode_buf) / sizeof(decode_buf[0]), 0);
            if (err != FDK_AAC_DEC_OK)
                continue;

            FDK_StreamInfo* si = fdk.GetStreamInfo(dec);
            if (si && si->numChannels > 0 && si->frameSize > 0) {
                dec_sample_rate = (uint32_t)si->sampleRate;
                dec_channels = si->numChannels;
                for (int j = 0; j < si->frameSize; ++j) {
                    int32_t sum = 0;
                    for (int c = 0; c < si->numChannels; ++c)
                        sum += decode_buf[j * si->numChannels + c];
                    pcm_all.push_back((int16_t)(sum / si->numChannels));
                }
            }
        }
    }
    fdk.Close(dec);

    if (pcm_all.empty() || dec_sample_rate == 0)
        return -2;

    // Strip encoder priming samples (from elst media_time). AAC encoders
    // emit 1024–2048 priming samples; the edit list tells us how many
    // to skip so the output aligns with the original pre-encode audio.
    size_t skip = 0;
    if (is_mp4 && mp4info.priming_samples > 0 && (size_t)mp4info.priming_samples < pcm_all.size()) {
        skip = (size_t)mp4info.priming_samples;
    }

    // Convert s16 → f32, skipping priming samples
    size_t n_samples = pcm_all.size() - skip;
    std::vector<float> f32(n_samples);
    for (size_t i = 0; i < n_samples; ++i)
        f32[i] = pcm_all[i + skip] / 32768.0f;

    int out_ch = want_channels > 0 ? want_channels : 1;
    if (out_ch > 2)
        out_ch = 2;

    // Resample to 16 kHz
    if (dec_sample_rate != (uint32_t)kTargetSampleRate) {
        ma_resampler_config rcfg = ma_resampler_config_init(ma_format_f32, 1, dec_sample_rate, kTargetSampleRate,
                                                            ma_resample_algorithm_linear);
        ma_resampler resampler;
        if (ma_resampler_init(&rcfg, nullptr, &resampler) != MA_SUCCESS)
            return -2;

        ma_uint64 in_len = n_samples;
        ma_uint64 out_len = 0;
        ma_resampler_get_expected_output_frame_count(&resampler, in_len, &out_len);
        out_len += 256;

        float* result = (float*)std::malloc(out_len * sizeof(float));
        if (!result) {
            ma_resampler_uninit(&resampler, nullptr);
            return -3;
        }

        ma_uint64 in_consumed = in_len;
        ma_uint64 out_produced = out_len;
        ma_resampler_process_pcm_frames(&resampler, f32.data(), &in_consumed, result, &out_produced);
        ma_resampler_uninit(&resampler, nullptr);

        *out_buf = result;
        *out_frames = (int)out_produced;
        *out_channels = 1;
        return 0;
    }

    float* result = (float*)std::malloc(n_samples * sizeof(float));
    if (!result)
        return -3;
    std::memcpy(result, f32.data(), n_samples * sizeof(float));
    *out_buf = result;
    *out_frames = (int)n_samples;
    *out_channels = 1;
    return 0;
}
} // namespace
#endif // !__APPLE__ && !__EMSCRIPTEN__

// ── Windows Media Foundation AAC decoder ─────────────────────────────────────
#if defined(_WIN32) && !defined(__APPLE__)
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")

namespace {
int crispasr_mf_decode(const char* path, int want_channels, float** out_buf, int* out_frames, int* out_channels) {
    // Convert path to wide string
    int wlen = MultiByteToWideChar(CP_UTF8, 0, path, -1, nullptr, 0);
    if (wlen <= 0)
        return -2;
    std::vector<wchar_t> wpath((size_t)wlen);
    MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath.data(), wlen);

    HRESULT hr = MFStartup(MF_VERSION);
    if (FAILED(hr))
        return -2;

    IMFSourceReader* reader = nullptr;
    hr = MFCreateSourceReaderFromURL(wpath.data(), nullptr, &reader);
    if (FAILED(hr) || !reader) {
        MFShutdown();
        return -2;
    }

    // Configure output as PCM float 16kHz mono
    IMFMediaType* outType = nullptr;
    MFCreateMediaType(&outType);
    outType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio);
    outType->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_Float);
    outType->SetUINT32(MF_MT_AUDIO_NUM_CHANNELS, 1);
    outType->SetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, kTargetSampleRate);
    outType->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 32);
    outType->SetUINT32(MF_MT_AUDIO_BLOCK_ALIGNMENT, 4);
    outType->SetUINT32(MF_MT_AUDIO_AVG_BYTES_PER_SECOND, kTargetSampleRate * 4);
    reader->SetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, nullptr, outType);
    outType->Release();

    std::vector<float> pcm;
    for (;;) {
        DWORD flags = 0;
        IMFSample* sample = nullptr;
        hr = reader->ReadSample((DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, 0, nullptr, &flags, nullptr, &sample);
        if (FAILED(hr) || (flags & MF_SOURCE_READERF_ENDOFSTREAM))
            break;
        if (!sample)
            continue;

        IMFMediaBuffer* buf = nullptr;
        sample->ConvertToContiguousBuffer(&buf);
        if (buf) {
            BYTE* raw = nullptr;
            DWORD len = 0;
            buf->Lock(&raw, nullptr, &len);
            size_t n = len / sizeof(float);
            const float* fp = reinterpret_cast<const float*>(raw);
            pcm.insert(pcm.end(), fp, fp + n);
            buf->Unlock();
            buf->Release();
        }
        sample->Release();
    }
    reader->Release();
    MFShutdown();

    if (pcm.empty())
        return -2;

    float* result = (float*)std::malloc(pcm.size() * sizeof(float));
    if (!result)
        return -3;
    std::memcpy(result, pcm.data(), pcm.size() * sizeof(float));
    *out_buf = result;
    *out_frames = (int)pcm.size();
    *out_channels = 1;
    return 0;
}
} // namespace
#endif // _WIN32

// ── Android NDK AAC decoder via MediaExtractor + MediaCodec ──────────────────
// Uses the NDK AMediaExtractor / AMediaCodec C API (available since API 21)
// to decode AAC/M4A/MP4 audio without ffmpeg. The OS provides hardware-
// accelerated AAC decode for free. Gated on __ANDROID__ and the absence of
// fdk-aac (if fdk-aac is linked, prefer it for consistency across platforms).
#if defined(CRISPASR_HAVE_MEDIA_NDK)
#include <media/NdkMediaCodec.h>
#include <media/NdkMediaExtractor.h>
#include <media/NdkMediaFormat.h>

namespace {
int crispasr_ndk_decode(const char* path, int want_channels, float** out_buf, int* out_frames, int* out_channels) {
    AMediaExtractor* ex = AMediaExtractor_new();
    if (!ex)
        return -2;

    media_status_t st = AMediaExtractor_setDataSource(ex, path);
    if (st != AMEDIA_OK) {
        AMediaExtractor_delete(ex);
        return -2;
    }

    // Find the first audio track
    int audio_track = -1;
    size_t num_tracks = AMediaExtractor_getTrackCount(ex);
    AMediaFormat* fmt = nullptr;
    for (size_t i = 0; i < num_tracks; ++i) {
        AMediaFormat* tf = AMediaExtractor_getTrackFormat(ex, i);
        const char* mime = nullptr;
        if (AMediaFormat_getString(tf, AMEDIAFORMAT_KEY_MIME, &mime) && mime) {
            if (std::strncmp(mime, "audio/", 6) == 0) {
                audio_track = (int)i;
                fmt = tf;
                break;
            }
        }
        AMediaFormat_delete(tf);
    }
    if (audio_track < 0 || !fmt) {
        AMediaExtractor_delete(ex);
        return -2;
    }

    AMediaExtractor_selectTrack(ex, (size_t)audio_track);

    // Get sample rate and channel count from the format
    int32_t file_rate = 44100, file_ch = 1;
    AMediaFormat_getInt32(fmt, AMEDIAFORMAT_KEY_SAMPLE_RATE, &file_rate);
    AMediaFormat_getInt32(fmt, AMEDIAFORMAT_KEY_CHANNEL_COUNT, &file_ch);

    // Create the codec from the track's MIME type
    const char* mime = nullptr;
    AMediaFormat_getString(fmt, AMEDIAFORMAT_KEY_MIME, &mime);
    AMediaCodec* codec = AMediaCodec_createDecoderByType(mime);
    if (!codec) {
        AMediaFormat_delete(fmt);
        AMediaExtractor_delete(ex);
        return -2;
    }

    st = AMediaCodec_configure(codec, fmt, nullptr, nullptr, 0);
    AMediaFormat_delete(fmt);
    if (st != AMEDIA_OK) {
        AMediaCodec_delete(codec);
        AMediaExtractor_delete(ex);
        return -2;
    }

    AMediaCodec_start(codec);

    std::vector<int16_t> pcm_all;
    bool eos_input = false;
    bool eos_output = false;

    while (!eos_output) {
        // Feed input
        if (!eos_input) {
            ssize_t buf_idx = AMediaCodec_dequeueInputBuffer(codec, 2000);
            if (buf_idx >= 0) {
                size_t buf_size = 0;
                uint8_t* buf = AMediaCodec_getInputBuffer(codec, (size_t)buf_idx, &buf_size);
                ssize_t sample_size = AMediaExtractor_readSampleData(ex, buf, buf_size);
                if (sample_size < 0) {
                    AMediaCodec_queueInputBuffer(codec, (size_t)buf_idx, 0, 0, 0,
                                                 AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM);
                    eos_input = true;
                } else {
                    int64_t pts = AMediaExtractor_getSampleTime(ex);
                    AMediaCodec_queueInputBuffer(codec, (size_t)buf_idx, 0, (size_t)sample_size, (uint64_t)pts, 0);
                    AMediaExtractor_advance(ex);
                }
            }
        }

        // Read output
        AMediaCodecBufferInfo info;
        ssize_t out_idx = AMediaCodec_dequeueOutputBuffer(codec, &info, 2000);
        if (out_idx >= 0) {
            if (info.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM) {
                eos_output = true;
            }
            if (info.size > 0) {
                size_t out_size = 0;
                uint8_t* out_data = AMediaCodec_getOutputBuffer(codec, (size_t)out_idx, &out_size);
                // Output is interleaved s16 PCM
                int n_samples = info.size / (int)(sizeof(int16_t) * file_ch);
                const int16_t* samples = reinterpret_cast<const int16_t*>(out_data + info.offset);
                // Downmix to mono
                for (int i = 0; i < n_samples; ++i) {
                    int32_t sum = 0;
                    for (int c = 0; c < file_ch; ++c)
                        sum += samples[i * file_ch + c];
                    pcm_all.push_back((int16_t)(sum / file_ch));
                }
            }
            AMediaCodec_releaseOutputBuffer(codec, (size_t)out_idx, false);
        }
    }

    AMediaCodec_stop(codec);
    AMediaCodec_delete(codec);
    AMediaExtractor_delete(ex);

    if (pcm_all.empty())
        return -2;

    // Convert s16 → f32
    size_t n = pcm_all.size();
    std::vector<float> f32(n);
    for (size_t i = 0; i < n; ++i)
        f32[i] = pcm_all[i] / 32768.0f;

    // Resample to 16 kHz if needed
    if (file_rate != kTargetSampleRate) {
        ma_resampler_config rcfg = ma_resampler_config_init(ma_format_f32, 1, (ma_uint32)file_rate, kTargetSampleRate,
                                                            ma_resample_algorithm_linear);
        ma_resampler resampler;
        if (ma_resampler_init(&rcfg, nullptr, &resampler) != MA_SUCCESS)
            return -2;

        ma_uint64 in_len = n;
        ma_uint64 out_len = 0;
        ma_resampler_get_expected_output_frame_count(&resampler, in_len, &out_len);
        out_len += 256;

        float* result = (float*)std::malloc(out_len * sizeof(float));
        if (!result) {
            ma_resampler_uninit(&resampler, nullptr);
            return -3;
        }

        ma_uint64 in_consumed = in_len;
        ma_uint64 out_produced = out_len;
        ma_resampler_process_pcm_frames(&resampler, f32.data(), &in_consumed, result, &out_produced);
        ma_resampler_uninit(&resampler, nullptr);

        *out_buf = result;
        *out_frames = (int)out_produced;
        *out_channels = 1;
        return 0;
    }

    float* result = (float*)std::malloc(n * sizeof(float));
    if (!result)
        return -3;
    std::memcpy(result, f32.data(), n * sizeof(float));
    *out_buf = result;
    *out_frames = (int)n;
    *out_channels = 1;
    return 0;
}
} // namespace
#endif // CRISPASR_HAVE_MEDIA_NDK

/// Decode an audio file into float32 mono PCM at 16 kHz. Supports WAV / MP3 /
/// FLAC / AIFF / W64 / RF64 (miniaudio), OGG Vorbis (stb_vorbis), .opus
/// (libopus/opusfile, when CRISPASR_HAVE_OPUS), Sun AU / .snd (inline, µ-law /
/// A-law / PCM), AMR-NB/WB (opencore-amr, when CRISPASR_HAVE_AMR),
/// WebM/Matroska Opus|Vorbis (EBML demux, when CRISPASR_HAVE_OPUS),
/// M4A/AAC/ADTS (fdk-aac via dlopen if installed; Media Foundation on
/// Windows; NDK MediaCodec on Android), and AAC / M4A / ALAC / CAF on Apple
/// (AudioToolbox fallback). The returned buffer is malloc-owned and must be
/// released with
/// `crispasr_audio_free`.
///
/// Returns 0 on success and writes:
///   *out_pcm         → float * of `*out_samples` elements (mono)
///   *out_samples     → number of samples written
///   *out_sample_rate → 16000 (we always resample to this)
///
/// Negative return codes:
///   -1 bad args
///   -2 decoder init failed (unsupported format or read error)
///   -3 allocation failed
///   -4 decode of a chunk failed mid-stream
namespace {

// ── glint ADTS AAC-LC decoder (cross-platform, no runtime dependency) ────────
// Always-available AAC-LC decode for raw ADTS (.aac) via the in-tree glint
// clean-room decoder (public C ABI). This is the PRIMARY path for raw AAC on
// every platform, replacing the previously platform-specific chain
// (fdk-aac dlopen on Linux/Windows, AudioToolbox on Apple) which required an
// optional runtime library. glint's AAC-LC decoder matches ffmpeg and Apple
// CoreAudio at 86–135 dB.
//
// Gated per the A/B convention: set CRISPASR_AAC_DECODER=fdk|coreaudio|os to
// pin the old platform decoder (glint returns -2, caller falls through);
// =glint|auto|unset keeps glint primary. On any glint failure we also return
// -2 so the existing fallback chain still runs — the working path is never
// removed. CRISPASR_AAC_DEBUG=1 prints a one-line decode summary.
//
// Signature matches the loader's other fallbacks (and the stereo
// split_fallback lambda): returns 0 with a malloc-owned 16 kHz mono f32 buffer,
// or a negative error. MP4/M4A/ALAC/CAF are NOT ADTS and are left to the
// container-aware decoders.
int crispasr_adts_decode_glint(const char* path, int want_channels, float** out_buf, int* out_frames,
                               int* out_channels) {
    (void)want_channels; // 16 kHz ASR path is mono; downmix here

    if (const char* pref = std::getenv("CRISPASR_AAC_DECODER")) {
        if (pref[0] && std::strcmp(pref, "glint") != 0 && std::strcmp(pref, "auto") != 0)
            return -2; // user pinned a non-glint decoder
    }
    const bool dbg = [] {
        const char* e = std::getenv("CRISPASR_AAC_DEBUG");
        return e && e[0] && e[0] != '0';
    }();

    FILE* f = std::fopen(path, "rb");
    if (!f)
        return -2;
    std::fseek(f, 0, SEEK_END);
    long fsize = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (fsize <= 2 || fsize > 500 * 1024 * 1024) {
        std::fclose(f);
        return -2;
    }
    std::vector<uint8_t> buf((size_t)fsize);
    if (std::fread(buf.data(), 1, (size_t)fsize, f) != (size_t)fsize) {
        std::fclose(f);
        return -2;
    }
    std::fclose(f);

    // Skip a leading ID3v2 tag if present (our own AAC writer prepends one).
    size_t pos = 0;
    if (buf.size() > 10 && buf[0] == 'I' && buf[1] == 'D' && buf[2] == '3') {
        size_t tag = ((size_t)(buf[6] & 0x7f) << 21) | ((size_t)(buf[7] & 0x7f) << 14) |
                     ((size_t)(buf[8] & 0x7f) << 7) | (size_t)(buf[9] & 0x7f);
        pos = 10 + tag;
    }
    // ADTS syncword is 0xFFF (12 bits): byte0 == 0xFF, byte1 high nibble == 0xF.
    if (pos + 2 > buf.size() || buf[pos] != 0xFF || (buf[pos + 1] & 0xF0) != 0xF0)
        return -2; // not ADTS — let the container decoders handle it

    glint_aac_dec_t dec = glint_aac_dec_create();
    if (!dec)
        return -2;

    std::vector<float> mono; // native-rate mono
    std::vector<float> frame_pcm(1024 * 8);
    int sr = 0, decoded_frames = 0;
    while (pos + 2 <= buf.size()) {
        if (buf[pos] != 0xFF || (buf[pos + 1] & 0xF0) != 0xF0) {
            // Resync to the next ADTS syncword (tolerate stray bytes).
            size_t scan = pos + 1;
            while (scan + 1 < buf.size() && !(buf[scan] == 0xFF && (buf[scan + 1] & 0xF0) == 0xF0))
                ++scan;
            if (scan + 1 >= buf.size())
                break;
            pos = scan;
        }
        glint_dec_frame_info fi;
        int n = glint_aac_decode(dec, buf.data() + pos, (int)(buf.size() - pos), frame_pcm.data(), &fi);
        if (n <= 0 || fi.frame_bytes <= 0)
            break;
        sr = fi.sample_rate;
        const int ch = fi.channels > 0 ? fi.channels : 1;
        mono.reserve(mono.size() + (size_t)n);
        for (int i = 0; i < n; ++i) {
            float s = 0.0f;
            for (int c = 0; c < ch; ++c)
                s += frame_pcm[(size_t)i * ch + c];
            mono.push_back(s / (float)ch);
        }
        pos += (size_t)fi.frame_bytes;
        ++decoded_frames;
    }
    glint_aac_dec_destroy(dec);

    if (mono.empty() || sr <= 0)
        return -2;
    if (dbg)
        std::fprintf(stderr, "[glint-aac] decoded %d frames, %zu samples @ %d Hz -> 16 kHz mono\n", decoded_frames,
                     mono.size(), sr);

    // Resample native → 16 kHz mono (linear; matches the fdk/M4A path).
    if (sr != kTargetSampleRate) {
        ma_resampler_config rcfg =
            ma_resampler_config_init(ma_format_f32, 1, (ma_uint32)sr, kTargetSampleRate, ma_resample_algorithm_linear);
        ma_resampler rs;
        if (ma_resampler_init(&rcfg, nullptr, &rs) != MA_SUCCESS)
            return -2;
        ma_uint64 in_len = mono.size();
        ma_uint64 out_len = 0;
        ma_resampler_get_expected_output_frame_count(&rs, in_len, &out_len);
        out_len += 256;
        float* result = (float*)std::malloc((size_t)out_len * sizeof(float));
        if (!result) {
            ma_resampler_uninit(&rs, nullptr);
            return -3;
        }
        ma_uint64 in_consumed = in_len;
        ma_uint64 out_produced = out_len;
        ma_resampler_process_pcm_frames(&rs, mono.data(), &in_consumed, result, &out_produced);
        ma_resampler_uninit(&rs, nullptr);
        *out_buf = result;
        *out_frames = (int)out_produced;
        *out_channels = 1;
        return 0;
    }

    float* result = (float*)std::malloc(mono.size() * sizeof(float));
    if (!result)
        return -3;
    std::memcpy(result, mono.data(), mono.size() * sizeof(float));
    *out_buf = result;
    *out_frames = (int)mono.size();
    *out_channels = 1;
    return 0;
}

// ── glint Ogg Opus (.opus) decoder (RFC-conformant, no libopus) ──────────────
// Decodes Ogg Opus (.opus, and Ogg-Opus-in-.ogg) via the in-tree glint decoder
// (glint_decode_audio; SILK/CELT/hybrid, all 12 RFC 6716/8251 vectors pass,
// SILK byte-identical to libopus). PRIMARY for Ogg Opus unless
// CRISPASR_OPUS_DECODER pins libopus — so .opus reads even in builds without
// libopus (CRISPASR_OPUS=OFF). Returns -2 for non-Opus Ogg (e.g. Vorbis) so
// those fall through to stb_vorbis, and for non-Ogg input so WAV/etc. take the
// ma_decoder path. On glint failure the caller still reaches the libopus path.
// NOTE: only the Ogg container — WebM/Matroska Opus stays on libopus.
//
// Core: decode to INTERLEAVED 16 kHz f32 with the native channel count
// preserved (malloc-owned, capped at 2 ch). The mono + stereo loaders wrap it.
int crispasr_ogg_opus_decode_glint_16k(const char* path, float** out_itl, int* out_frames, int* out_channels) {
    if (const char* pref = std::getenv("CRISPASR_OPUS_DECODER")) {
        if (pref[0] && std::strcmp(pref, "glint") != 0 && std::strcmp(pref, "auto") != 0)
            return -2; // user pinned libopus
    }
    const bool dbg = [] {
        const char* e = std::getenv("CRISPASR_OPUS_DEBUG");
        return e && e[0] && e[0] != '0';
    }();

    FILE* f = std::fopen(path, "rb");
    if (!f)
        return -2;
    std::fseek(f, 0, SEEK_END);
    long fsize = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (fsize <= 4 || fsize > 500 * 1024 * 1024) {
        std::fclose(f);
        return -2;
    }
    std::vector<uint8_t> buf((size_t)fsize);
    if (std::fread(buf.data(), 1, (size_t)fsize, f) != (size_t)fsize) {
        std::fclose(f);
        return -2;
    }
    std::fclose(f);

    // Must be an Ogg stream ("OggS"); glint_decode_audio further checks it is
    // Opus (returns NULL for Ogg Vorbis, which then falls through to stb_vorbis).
    if (buf[0] != 'O' || buf[1] != 'g' || buf[2] != 'g' || buf[3] != 'S')
        return -2;

    int sr = 0, ch = 0, frames = 0;
    float* pcm = glint_decode_audio(buf.data(), (int)buf.size(), &sr, &ch, &frames);
    if (!pcm)
        return -2; // not Ogg Opus (e.g. Vorbis) or a decode error → fall through
    if (frames <= 0 || ch < 1 || sr <= 0) {
        glint_free(pcm);
        return -2;
    }
    if (ch > 2) {
        // Downmix >2 ch to stereo up front (rare; the loaders cap at 2).
        for (int i = 0; i < frames; i++) {
            pcm[(size_t)i * 2] = pcm[(size_t)i * ch];
            pcm[(size_t)i * 2 + 1] = pcm[(size_t)i * ch + 1];
        }
        ch = 2;
    }
    if (dbg)
        std::fprintf(stderr, "[glint-opus] decoded %d frames, %d ch @ %d -> 16 kHz\n", frames, ch, sr);

    // Resample native -> 16 kHz, preserving channels (interleaved, linear).
    if (sr != kTargetSampleRate) {
        ma_resampler_config rcfg = ma_resampler_config_init(ma_format_f32, (ma_uint32)ch, (ma_uint32)sr,
                                                            kTargetSampleRate, ma_resample_algorithm_linear);
        ma_resampler rs;
        if (ma_resampler_init(&rcfg, nullptr, &rs) != MA_SUCCESS) {
            glint_free(pcm);
            return -2;
        }
        ma_uint64 in_len = (ma_uint64)frames;
        ma_uint64 out_len = 0;
        ma_resampler_get_expected_output_frame_count(&rs, in_len, &out_len);
        out_len += 256;
        float* result = (float*)std::malloc((size_t)out_len * (size_t)ch * sizeof(float));
        if (!result) {
            ma_resampler_uninit(&rs, nullptr);
            glint_free(pcm);
            return -3;
        }
        ma_uint64 in_consumed = in_len;
        ma_uint64 out_produced = out_len;
        ma_resampler_process_pcm_frames(&rs, pcm, &in_consumed, result, &out_produced);
        ma_resampler_uninit(&rs, nullptr);
        glint_free(pcm);
        *out_itl = result;
        *out_frames = (int)out_produced;
        *out_channels = ch;
        return 0;
    }

    float* result = (float*)std::malloc((size_t)frames * (size_t)ch * sizeof(float));
    if (!result) {
        glint_free(pcm);
        return -3;
    }
    std::memcpy(result, pcm, (size_t)frames * (size_t)ch * sizeof(float));
    glint_free(pcm);
    *out_itl = result;
    *out_frames = frames;
    *out_channels = ch;
    return 0;
}

// Mono wrapper — signature matches the loader fallbacks: 0 + malloc-owned
// 16 kHz mono f32. Downmixes the core's interleaved output.
int crispasr_ogg_opus_decode_glint(const char* path, int want_channels, float** out_buf, int* out_frames,
                                   int* out_channels) {
    (void)want_channels; // 16 kHz ASR path is mono; downmix here
    float* itl = nullptr;
    int frames = 0, ch = 0;
    int rc = crispasr_ogg_opus_decode_glint_16k(path, &itl, &frames, &ch);
    if (rc != 0)
        return rc;
    float* mono = (float*)std::malloc((size_t)frames * sizeof(float));
    if (!mono) {
        std::free(itl);
        return -3;
    }
    for (int i = 0; i < frames; i++) {
        float s = 0.0f;
        for (int c = 0; c < ch; c++)
            s += itl[(size_t)i * ch + c];
        mono[(size_t)i] = s / (float)ch;
    }
    std::free(itl);
    *out_buf = mono;
    *out_frames = frames;
    *out_channels = 1;
    return 0;
}

} // namespace

CA_EXPORT int crispasr_audio_load(const char* path, float** out_pcm, int* out_samples, int* out_sample_rate) {
    if (!path || !out_pcm || !out_samples)
        return -1;
    *out_pcm = nullptr;
    *out_samples = 0;
    if (out_sample_rate)
        *out_sample_rate = 0;

    // Ogg Opus (.opus) via the in-tree glint decoder — RFC-conformant, no
    // libopus required, so .opus reads even in builds without CRISPASR_OPUS.
    // Runs BEFORE ma_decoder (which would otherwise take it via libopus).
    // Non-Opus Ogg (Vorbis) and non-Ogg inputs return -2 → normal path below.
    {
        float* g_buf = nullptr;
        int g_fr = 0, g_ch = 0;
        if (crispasr_ogg_opus_decode_glint(path, 1, &g_buf, &g_fr, &g_ch) == 0) {
            *out_pcm = g_buf;
            *out_samples = g_fr;
            if (out_sample_rate)
                *out_sample_rate = kTargetSampleRate;
            return 0;
        }
    }

    ma_decoder_config cfg = ma_decoder_config_init(ma_format_f32, kTargetChannels, kTargetSampleRate);
    CRISPASR_OPUS_DECODER_CONFIG(cfg);
    ma_decoder decoder;
    if (ma_decoder_init_file(path, &cfg, &decoder) != MA_SUCCESS) {
        // Raw ADTS AAC-LC (.aac) via the in-tree glint decoder — cross-platform,
        // no runtime dependency, PRIMARY unless CRISPASR_AAC_DECODER pins the
        // platform decoder. Returns -2 for non-ADTS (MP4/M4A) so the
        // container-aware fallbacks below still run.
        {
            float* g_buf = nullptr;
            int g_fr = 0, g_ch = 0;
            if (crispasr_adts_decode_glint(path, 1, &g_buf, &g_fr, &g_ch) == 0) {
                *out_pcm = g_buf;
                *out_samples = g_fr;
                if (out_sample_rate)
                    *out_sample_rate = kTargetSampleRate;
                return 0;
            }
        }
#if defined(__APPLE__)
        // Format miniaudio can't decode (AAC / M4A / ALAC / CAF …) — try the
        // OS-native AudioToolbox decoder. Requesting 1 channel gives mono
        // directly (AudioConverter downmixes).
        float* itl = nullptr;
        int fr = 0, ch = 0;
        if (crispasr_at_decode(path, 1, &itl, &fr, &ch) == 0) {
            *out_pcm = itl;
            *out_samples = fr;
            if (out_sample_rate)
                *out_sample_rate = kTargetSampleRate;
            return 0;
        }
#endif
        // AU / .snd fallback (inline, no external deps)
        {
            float* au_buf = nullptr;
            int au_fr = 0, au_ch = 0;
            if (crispasr_au_decode(path, 1, &au_buf, &au_fr, &au_ch) == 0) {
                *out_pcm = au_buf;
                *out_samples = au_fr;
                if (out_sample_rate)
                    *out_sample_rate = kTargetSampleRate;
                return 0;
            }
        }
#if defined(CRISPASR_HAVE_AMR)
        // AMR-NB/WB fallback (opencore-amr, Apache-2.0)
        {
            float* amr_buf = nullptr;
            int amr_fr = 0, amr_ch = 0;
            if (crispasr_amr_decode(path, 1, &amr_buf, &amr_fr, &amr_ch) == 0) {
                *out_pcm = amr_buf;
                *out_samples = amr_fr;
                if (out_sample_rate)
                    *out_sample_rate = kTargetSampleRate;
                return 0;
            }
        }
#endif
        // WebM/Matroska fallback (EBML demux → glint Opus, or libopus when
        // CRISPASR_OPUS_DECODER=libopus; Vorbis via stb_vorbis). Now always
        // available — no libopus required.
        {
            float* webm_buf = nullptr;
            int webm_fr = 0, webm_ch = 0;
            if (crispasr_webm_decode(path, 1, &webm_buf, &webm_fr, &webm_ch) == 0) {
                *out_pcm = webm_buf;
                *out_samples = webm_fr;
                if (out_sample_rate)
                    *out_sample_rate = kTargetSampleRate;
                return 0;
            }
        }
#if !defined(__APPLE__) && !defined(__EMSCRIPTEN__)
        // M4A / AAC / ADTS fallback (fdk-aac via dlopen, if installed)
        {
            float* aac_buf = nullptr;
            int aac_fr = 0, aac_ch = 0;
            if (crispasr_m4a_decode(path, 1, &aac_buf, &aac_fr, &aac_ch) == 0) {
                *out_pcm = aac_buf;
                *out_samples = aac_fr;
                if (out_sample_rate)
                    *out_sample_rate = kTargetSampleRate;
                return 0;
            }
        }
#endif
#if defined(_WIN32) && !defined(__APPLE__)
        // Windows Media Foundation fallback for AAC/M4A
        {
            float* mf_buf = nullptr;
            int mf_fr = 0, mf_ch = 0;
            if (crispasr_mf_decode(path, 1, &mf_buf, &mf_fr, &mf_ch) == 0) {
                *out_pcm = mf_buf;
                *out_samples = mf_fr;
                if (out_sample_rate)
                    *out_sample_rate = kTargetSampleRate;
                return 0;
            }
        }
#endif
#if defined(CRISPASR_HAVE_MEDIA_NDK)
        // Android NDK MediaCodec fallback for AAC/M4A
        {
            float* ndk_buf = nullptr;
            int ndk_fr = 0, ndk_ch = 0;
            if (crispasr_ndk_decode(path, 1, &ndk_buf, &ndk_fr, &ndk_ch) == 0) {
                *out_pcm = ndk_buf;
                *out_samples = ndk_fr;
                if (out_sample_rate)
                    *out_sample_rate = kTargetSampleRate;
                return 0;
            }
        }
#endif
        return -2;
    }

    // Decode in 1-second chunks. ma_decoder_get_length_in_pcm_frames can
    // fail on MP3 / streaming sources; chunked-read is what the CLI uses
    // and sidesteps that. The total allocation grows geometrically so we
    // don't re-alloc every chunk.
    const ma_uint64 kChunkFrames = (ma_uint64)kTargetSampleRate; // 1 s
    const size_t max_frames = crispasr_max_decoded_frames(path);
    float* buf = nullptr;
    size_t capacity = 0;
    size_t used = 0;

    for (;;) {
        if (max_frames && used >= max_frames) {
            crispasr_report_decode_ceiling(path, max_frames);
            std::free(buf);
            ma_decoder_uninit(&decoder);
            return -3;
        }
        if (capacity - used < kChunkFrames) {
            size_t new_cap = capacity ? capacity * 2 : (size_t)kChunkFrames * 8;
            // See the Apple loop: clamping to the ceiling plus one chunk keeps
            // the grow branch unreachable once the ceiling is allocated.
            if (max_frames && new_cap > max_frames + (size_t)kChunkFrames)
                new_cap = max_frames + (size_t)kChunkFrames;
            float* nb = (float*)std::realloc(buf, new_cap * sizeof(float));
            if (!nb) {
                if (buf)
                    std::free(buf);
                ma_decoder_uninit(&decoder);
                return -3;
            }
            buf = nb;
            capacity = new_cap;
        }

        ma_uint64 frames_read = 0;
        const ma_result rc = ma_decoder_read_pcm_frames(&decoder, buf + used, kChunkFrames, &frames_read);
        used += (size_t)frames_read;

        if (rc == MA_AT_END || frames_read == 0)
            break;
        if (rc != MA_SUCCESS) {
            std::free(buf);
            ma_decoder_uninit(&decoder);
            return -4;
        }
    }
    ma_decoder_uninit(&decoder);

    // Trim trailing capacity we didn't fill — keeps the allocation tight.
    if (used < capacity) {
        float* tb = (float*)std::realloc(buf, used * sizeof(float));
        if (tb)
            buf = tb;
    }

    *out_pcm = buf;
    *out_samples = (int)used;
    if (out_sample_rate)
        *out_sample_rate = kTargetSampleRate;
    return 0;
}

/// Like crispasr_audio_load but resamples to `target_rate` instead of 16 kHz.
/// When the source already matches `target_rate`, no resampling is performed.
/// Avoids the quality-degrading down-then-up path that occurs when a backend
/// needs non-16 kHz audio (e.g. 24 kHz for Kyutai / IndexTTS / Mimi).
CA_EXPORT int crispasr_audio_load_at_rate(const char* path, int target_rate, float** out_pcm, int* out_samples,
                                          int* out_sample_rate) {
    const int prev = kTargetSampleRate;
    kTargetSampleRate = (target_rate > 0) ? target_rate : 16000;
    const int rc = crispasr_audio_load(path, out_pcm, out_samples, out_sample_rate);
    kTargetSampleRate = prev;
    return rc;
}

/// Decode an audio file into stereo (2-channel) float32 PCM at 16 kHz.
/// If the source is mono, both left and right receive the same data and
/// `*out_channels` is set to 1. If stereo, the interleaved samples are
/// deinterleaved into separate left and right buffers and `*out_channels`
/// is set to 2. Each output buffer is malloc-owned and must be released
/// with `crispasr_audio_free`.
///
/// Returns 0 on success, negative on error (same codes as
/// `crispasr_audio_load`).
CA_EXPORT int crispasr_audio_load_stereo(const char* path, float** out_left, float** out_right, int* out_samples,
                                         int* out_sample_rate, int* out_channels) {
    if (!path || !out_left || !out_right || !out_samples || !out_channels)
        return -1;
    *out_left = nullptr;
    *out_right = nullptr;
    *out_samples = 0;
    *out_channels = 0;
    if (out_sample_rate)
        *out_sample_rate = 0;

    // Ogg Opus (.opus) via glint — RFC-conformant, no libopus; stereo-preserving.
    // Runs BEFORE ma_decoder (which would otherwise take it via libopus). Gated
    // CRISPASR_OPUS_DECODER=libopus; non-Opus Ogg / non-Ogg fall through.
    {
        float* itl = nullptr;
        int fr = 0, ch = 0;
        if (crispasr_ogg_opus_decode_glint_16k(path, &itl, &fr, &ch) == 0) {
            float* left = (float*)std::malloc((size_t)fr * sizeof(float));
            float* right = (float*)std::malloc((size_t)fr * sizeof(float));
            if (!left || !right) {
                std::free(itl);
                std::free(left);
                std::free(right);
                return -3;
            }
            for (int i = 0; i < fr; i++) {
                left[i] = itl[(size_t)i * ch];
                right[i] = (ch >= 2) ? itl[(size_t)i * ch + 1] : itl[(size_t)i * ch];
            }
            std::free(itl);
            *out_left = left;
            *out_right = right;
            *out_samples = fr;
            *out_channels = (ch >= 2) ? 2 : 1;
            if (out_sample_rate)
                *out_sample_rate = kTargetSampleRate;
            return 0;
        }
    }

    // Detect native channel count (channels = 0 → native).
    ma_decoder_config probe_cfg = ma_decoder_config_init(ma_format_f32, 0, kTargetSampleRate);
    CRISPASR_OPUS_DECODER_CONFIG(probe_cfg);
    ma_decoder probe;
    if (ma_decoder_init_file(path, &probe_cfg, &probe) != MA_SUCCESS) {
#if defined(__APPLE__)
        // AAC / M4A / ALAC / CAF … via AudioToolbox. Decode native (≤2 ch)
        // interleaved, then split into the per-channel L/R outputs.
        float* itl = nullptr;
        int fr = 0, ch = 0;
        if (crispasr_at_decode(path, 0, &itl, &fr, &ch) == 0) {
            float* left = (float*)std::malloc((size_t)fr * sizeof(float));
            float* right = (float*)std::malloc((size_t)fr * sizeof(float));
            if (!left || !right) {
                std::free(itl);
                std::free(left);
                std::free(right);
                return -3;
            }
            if (ch >= 2) {
                for (int i = 0; i < fr; ++i) {
                    left[i] = itl[(size_t)i * 2];
                    right[i] = itl[(size_t)i * 2 + 1];
                }
                *out_channels = 2;
            } else {
                std::memcpy(left, itl, (size_t)fr * sizeof(float));
                std::memcpy(right, itl, (size_t)fr * sizeof(float));
                *out_channels = 1;
            }
            std::free(itl);
            *out_left = left;
            *out_right = right;
            *out_samples = fr;
            if (out_sample_rate)
                *out_sample_rate = kTargetSampleRate;
            return 0;
        }
#endif
        // Helper: split interleaved fallback decode into L/R outputs.
        auto split_fallback = [&](int (*decode_fn)(const char*, int, float**, int*, int*)) -> int {
            float* itl = nullptr;
            int fr = 0, ch = 0;
            if (decode_fn(path, 0, &itl, &fr, &ch) != 0)
                return -2;
            float* left = (float*)std::malloc((size_t)fr * sizeof(float));
            float* right = (float*)std::malloc((size_t)fr * sizeof(float));
            if (!left || !right) {
                std::free(itl);
                std::free(left);
                std::free(right);
                return -3;
            }
            if (ch >= 2) {
                for (int i = 0; i < fr; ++i) {
                    left[i] = itl[(size_t)i * 2];
                    right[i] = itl[(size_t)i * 2 + 1];
                }
                *out_channels = 2;
            } else {
                std::memcpy(left, itl, (size_t)fr * sizeof(float));
                std::memcpy(right, itl, (size_t)fr * sizeof(float));
                *out_channels = 1;
            }
            std::free(itl);
            *out_left = left;
            *out_right = right;
            *out_samples = fr;
            if (out_sample_rate)
                *out_sample_rate = kTargetSampleRate;
            return 0;
        };
        // Raw ADTS AAC-LC (.aac) via glint — cross-platform primary (see the
        // mono loader). Non-ADTS returns -2 and falls through.
        if (split_fallback(crispasr_adts_decode_glint) == 0)
            return 0;
        // AU / .snd fallback
        if (split_fallback(crispasr_au_decode) == 0)
            return 0;
#if defined(CRISPASR_HAVE_AMR)
        if (split_fallback(crispasr_amr_decode) == 0)
            return 0;
#endif
        // WebM/Matroska (glint Opus / stb_vorbis) — always available, no libopus.
        if (split_fallback(crispasr_webm_decode) == 0)
            return 0;
#if !defined(__APPLE__) && !defined(__EMSCRIPTEN__)
        if (split_fallback(crispasr_m4a_decode) == 0)
            return 0;
#endif
#if defined(CRISPASR_HAVE_MEDIA_NDK)
        if (split_fallback(crispasr_ndk_decode) == 0)
            return 0;
#endif
        return -2;
    }
    const int native_channels = (int)probe.outputChannels;
    ma_decoder_uninit(&probe);

    // Re-open with the target channel count (1 or 2) and 16 kHz.
    const int decode_channels = (native_channels >= 2) ? 2 : 1;
    ma_decoder_config cfg = ma_decoder_config_init(ma_format_f32, (ma_uint32)decode_channels, kTargetSampleRate);
    CRISPASR_OPUS_DECODER_CONFIG(cfg);
    ma_decoder decoder;
    if (ma_decoder_init_file(path, &cfg, &decoder) != MA_SUCCESS)
        return -2;

    const ma_uint64 kChunkFrames = (ma_uint64)kTargetSampleRate; // 1 s
    const size_t max_frames = crispasr_max_decoded_frames(path);
    float* buf = nullptr;
    size_t capacity = 0; // in frames
    size_t used = 0;     // in frames

    for (;;) {
        if (max_frames && used >= max_frames) {
            crispasr_report_decode_ceiling(path, max_frames);
            std::free(buf);
            ma_decoder_uninit(&decoder);
            return -3;
        }
        if (capacity - used < kChunkFrames) {
            size_t new_cap = capacity ? capacity * 2 : (size_t)kChunkFrames * 8;
            // See the Apple loop: clamping to the ceiling plus one chunk keeps
            // the grow branch unreachable once the ceiling is allocated.
            if (max_frames && new_cap > max_frames + (size_t)kChunkFrames)
                new_cap = max_frames + (size_t)kChunkFrames;
            float* nb = (float*)std::realloc(buf, new_cap * (size_t)decode_channels * sizeof(float));
            if (!nb) {
                if (buf)
                    std::free(buf);
                ma_decoder_uninit(&decoder);
                return -3;
            }
            buf = nb;
            capacity = new_cap;
        }

        ma_uint64 frames_read = 0;
        const ma_result rc =
            ma_decoder_read_pcm_frames(&decoder, buf + used * (size_t)decode_channels, kChunkFrames, &frames_read);
        used += (size_t)frames_read;

        if (rc == MA_AT_END || frames_read == 0)
            break;
        if (rc != MA_SUCCESS) {
            std::free(buf);
            ma_decoder_uninit(&decoder);
            return -4;
        }
    }
    ma_decoder_uninit(&decoder);

    // Allocate per-channel output buffers.
    float* left = (float*)std::malloc(used * sizeof(float));
    float* right = (float*)std::malloc(used * sizeof(float));
    if (!left || !right) {
        std::free(buf);
        std::free(left);
        std::free(right);
        return -3;
    }

    if (decode_channels == 1) {
        // Mono: copy same data to both channels.
        std::memcpy(left, buf, used * sizeof(float));
        std::memcpy(right, buf, used * sizeof(float));
        *out_channels = 1;
    } else {
        // Stereo: deinterleave [L0 R0 L1 R1 ...] into separate buffers.
        for (size_t i = 0; i < used; ++i) {
            left[i] = buf[i * 2];
            right[i] = buf[i * 2 + 1];
        }
        *out_channels = 2;
    }
    std::free(buf);

    *out_left = left;
    *out_right = right;
    *out_samples = (int)used;
    if (out_sample_rate)
        *out_sample_rate = kTargetSampleRate;
    return 0;
}

/// Release a buffer allocated by `crispasr_audio_load`.
CA_EXPORT void crispasr_audio_free(float* pcm) {
    if (pcm)
        std::free(pcm);
}
