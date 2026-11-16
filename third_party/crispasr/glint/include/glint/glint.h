// glint - Fixed-point MPEG-1 Layer III encoder
// MIT License - Clean-room implementation

#ifndef GLINT_H
#define GLINT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct glint_context* glint_t;

enum glint_mode {
    GLINT_MONO   = 0,
    GLINT_DUAL   = 1,
    GLINT_JOINT  = 2,
    GLINT_STEREO = 3,
};

enum glint_path {
    GLINT_PATH_DEFAULT = 0,  // use compile-time default
    GLINT_PATH_DOUBLE  = 1,  // force double-precision
    GLINT_PATH_FIXED   = 2,  // force fixed-point Q31
};

enum glint_simd {
    GLINT_SIMD_AUTO   = 0,  // detect best available at runtime
    GLINT_SIMD_AVX    = 1,  // force AVX (crashes if unsupported!)
    GLINT_SIMD_SSE2   = 2,  // force SSE2
    GLINT_SIMD_NONE   = 3,  // scalar only
    GLINT_SIMD_NEON   = 4,  // force AArch64 NEON (crashes if unsupported!)
};

enum glint_quality {
    GLINT_QUALITY_SPEED  = 0,  // no masking, fastest
    GLINT_QUALITY_NORMAL = 1,  // gain correction + headroom SF (default)
    GLINT_QUALITY_BEST   = 2,  // multi-factor search + psychoacoustic masking
};

enum glint_vbr {
    GLINT_VBR_OFF = 0,  // CBR (default)
    GLINT_VBR_ON  = 1,  // VBR with quality target
};

struct glint_config {
    int sample_rate;
    int num_channels;
    enum glint_mode mode;
    int bitrate;
    enum glint_path path;  // signal path selection (0 = default)
    enum glint_simd simd;  // SIMD selection (0 = auto-detect)
    enum glint_quality quality;  // quality mode (0 = speed, 1 = normal)
    enum glint_vbr vbr;           // VBR mode (0 = CBR, 1 = VBR)
    int vbr_quality;               // VBR quality 0-9 (0=best, 9=worst), only used when vbr=1
};

// Callback: called with each encoded MP3 frame
typedef void (*glint_write_cb)(const uint8_t* data, int size, void* user_data);

// Set the worker-thread count for the per-granule scale-factor search,
// process-wide. 1 (the default) runs single-threaded. The output bitstream is
// byte-identical regardless of thread count — candidates are reduced in a
// fixed order — so threading is a pure throughput knob with no quality effect.
// Call once before encoding (it (re)creates the shared pool).
void           glint_set_threads(int num_threads);

int            glint_check_config(int sample_rate, int bitrate);
glint_t        glint_create(const struct glint_config* cfg);
glint_t        glint_create_streaming(const struct glint_config* cfg,
                                      glint_write_cb callback, void* user_data);
int            glint_samples_per_frame(glint_t enc);

// Encode one frame. channel_data[ch] points to samples_per_frame samples.
const uint8_t* glint_encode(glint_t enc, const int16_t** channel_data, int* out_size);
const uint8_t* glint_encode_float(glint_t enc, const float** channel_data, int* out_size);
const uint8_t* glint_encode_int32(glint_t enc, const int32_t** channel_data, int* out_size);

const uint8_t* glint_flush(glint_t enc, int* out_size);

// VBR only: fill `buf` with the finalized Xing header frame (frame count,
// byte count, 100-point seek TOC). Call AFTER glint_flush, then overwrite
// the beginning of the output file with it — frame 0 of a VBR stream is a
// silent placeholder of exactly this size. Returns the frame size, or 0
// when not applicable (CBR, no frames emitted, buf too small). Streaming
// consumers that cannot seek may skip this; the placeholder decodes as
// ~26 ms of leading silence.
int            glint_vbr_header(glint_t enc, uint8_t* buf, int buf_capacity);

void           glint_destroy(glint_t enc);

// ---------------------------------------------------------------------------
// AAC-LC encoder (phase 1: long blocks, CBR-average, ADTS output).
// Independent of the MP3 encoder above. One encode call consumes exactly
// glint_aac_samples_per_frame() samples per channel (1024) and returns one
// ADTS frame. Call glint_aac_flush once at end of stream — the MDCT looks
// back one block, so the final 1024 samples are emitted by the flush frame.
// Sample rates: 8000..96000 (the 12 standard AAC rates); 1-2 channels;
// bitrate in kbps.
// ---------------------------------------------------------------------------

typedef struct glint_aac_context* glint_aac_t;

// ZERO-INITIALIZE this struct before filling it (memset or `= {0}`): the
// reserved tail lets future releases add options without breaking the ABI,
// and zeroed reserved fields select defaults.
struct glint_aac_config {
    int sample_rate;
    int num_channels;
    int bitrate;        // kbps; under VBR this is only the per-frame CAP hint
    enum glint_quality quality;  // SPEED = no noise shaping; NORMAL/BEST = psy
                                 // NMR shaping loop (BEST iterates further)
    int vbr;            // 0 = CBR (default), 1 = constant-quality VBR
    int vbr_quality;    // 0 (best/largest) .. 9 (worst/smallest), when vbr=1
    int reserved[4];    // must be zero
};

// Library version as (major << 16) | (minor << 8) | patch.
int            glint_version(void);

glint_aac_t    glint_aac_create(const struct glint_aac_config* cfg);
int            glint_aac_samples_per_frame(glint_aac_t enc);
const uint8_t* glint_aac_encode(glint_aac_t enc, const int16_t** channel_data, int* out_size);
const uint8_t* glint_aac_encode_float(glint_aac_t enc, const float** channel_data, int* out_size);
const uint8_t* glint_aac_flush(glint_aac_t enc, int* out_size);
void           glint_aac_destroy(glint_aac_t enc);

// ---------------------------------------------------------------------------
// Opus (RFC 6716/7845). Decoder: any Opus stream (SILK/CELT/hybrid),
// packet-loss concealment, SILK in-band FEC, output rates 8/12/16/24/48 kHz.
// Encoder: CELT-only at 48 kHz (frame sizes 120/240/480/960 samples),
// CBR or unconstrained VBR; every packet carries its TOC byte and is
// playable as-is (mux with an Ogg layer for .opus files).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// MP3 + AAC-LC DECODERS. Feed whole frames (locate them with the
// frame_info helpers, which read one header). Output is interleaved
// float PCM (±1.0). The MP3 decoder keeps a bit reservoir across calls,
// so the first frame(s) of a stream may return 0 samples until enough
// history has accumulated. Both are stateful — one context per stream.
// ---------------------------------------------------------------------------

typedef struct glint_mp3_dec_context* glint_mp3_dec_t;
typedef struct glint_aac_dec_context* glint_aac_dec_t;

struct glint_dec_frame_info {
    int sample_rate;
    int channels;
    int samples;      // per channel this frame (0 while MP3 reservoir fills)
    int frame_bytes;  // whole-frame length incl. header
};

// Parse one frame header; 0 + fills info, or -1 if data is not a valid
// frame sync. Use frame_bytes to advance to the next frame.
int glint_mp3_frame_info(const uint8_t* data, int len,
                         struct glint_dec_frame_info* info);
int glint_aac_frame_info(const uint8_t* data, int len,
                         struct glint_dec_frame_info* info);

// Resample interleaved float PCM (±1.0) from sr_in to sr_out with a
// Kaiser-windowed sinc kernel (anti-aliased, unity passband). Returns a
// malloc'd interleaved buffer of *out_frames*channels floats — free it
// with glint_free — or NULL on bad arguments.
float* glint_resample(const float* in, int in_frames, int channels,
                      int sr_in, int sr_out, int* out_frames);
void   glint_free(void* p);

// Output codec selector for glint_encode_audio.
enum glint_enc_format {
    GLINT_ENC_MP3  = 0,
    GLINT_ENC_AAC  = 1,
    GLINT_ENC_OPUS = 2,
};

// One-call encode: interleaved float PCM (±1.0, `frames` per channel, 1-2
// channels, at `sample_rate`) -> a complete MP3 / AAC-LC / Ogg-Opus
// stream. The input is auto-resampled to a codec-valid rate (Opus -> 48k;
// MP3/AAC -> nearest supported rate). bitrate_kbps is the CBR/target rate;
// vbr_quality 0..9 selects VBR (-1 = CBR); quality is GLINT_QUALITY_*.
// Returns a malloc'd buffer of *out_size bytes — free with glint_free —
// or NULL on error.
uint8_t* glint_encode_audio(const float* pcm, int frames, int channels,
                            int sample_rate, int format, int bitrate_kbps,
                            int vbr_quality, int quality, int* out_size);

// Read a WAV file (PCM 8/16/24/32, IEEE float 32/64, A-law, mu-law,
// WAVE_FORMAT_EXTENSIBLE) into interleaved float PCM (±1.0). Returns a
// malloc'd buffer of *out_frames*out_ch floats — free with glint_free —
// or NULL on malformed / unsupported input.
float* glint_wav_read(const uint8_t* data, int len, int* out_sr,
                      int* out_ch, int* out_frames);
// Encode interleaved float PCM (±1.0) to a WAV buffer. bits: 8/16/24/32
// integer PCM, or 32/64 with is_float!=0 for IEEE float; invalid combos
// fall back to 16-bit. Returns a malloc'd buffer of *out_size bytes —
// free with glint_free — or NULL on error.
uint8_t* glint_wav_write(const float* pcm, int frames, int channels,
                         int sample_rate, int bits, int is_float,
                         int* out_size);

// Decode a whole encoded stream (MP3 / AAC-LC / Ogg-Opus / Ogg-Vorbis / FLAC,
// auto-detected
// from the header) to interleaved float PCM (±1.0). Returns a malloc'd
// buffer of *out_frames*out_ch floats — free with glint_free — and writes
// the sample rate, channel count and per-channel frame count. NULL on
// error or unrecognized input.
float* glint_decode_audio(const uint8_t* data, int len, int* out_sr,
                          int* out_ch, int* out_frames);

// Like glint_decode_audio with two optional knobs: out_rate resamples the
// decoded PCM (0 = keep native rate), and want_int16 chooses the output
// sample format — the returned buffer is int16_t* when want_int16!=0, else
// float*. Opus surround (mapping family 1, up to 8 channels) is decoded.
// Returns a malloc'd buffer (free with glint_free) or NULL on error.
void* glint_decode_audio_ex(const uint8_t* data, int len, int out_rate,
                            int want_int16, int* out_sr, int* out_ch,
                            int* out_frames);

// ---------------------------------------------------------------------------
// Ogg-Vorbis I decoder. Decodes a COMPLETE in-memory Ogg-Vorbis logical
// stream (identification + comment + setup headers followed by audio
// packets) to interleaved PCM. The whole-buffer form fits .sf3, where each
// sample is its own short Ogg-Vorbis stream. glint_decode_audio / _ex also
// auto-detect Vorbis (both Vorbis and Opus use the OggS container; they are
// distinguished by the first packet's codec-id header).
// ---------------------------------------------------------------------------

// Decode to a malloc'd buffer of out_frames*out_ch interleaved floats (+-1.0)
// — free with glint_free. Writes sample rate, channel count and per-channel
// frame count. NULL on error / not a Vorbis stream.
float* glint_vorbis_decode(const uint8_t* ogg, int len, int* out_sr,
                           int* out_ch, int* out_frames);

// Like glint_vorbis_decode with an optional output rate (0 = native) and an
// int16 output option: the returned buffer is int16_t* when want_int16!=0,
// else float*. Free with glint_free. NULL on error.
void* glint_vorbis_decode_ex(const uint8_t* ogg, int len, int out_rate,
                             int want_int16, int* out_sr, int* out_ch,
                             int* out_frames);

// ---------------------------------------------------------------------------
// FLAC decoder. Decodes a COMPLETE in-memory native FLAC stream ("fLaC"
// marker, metadata blocks, frames) to interleaved PCM. Supports the standard
// lossless subframe types: constant, verbatim, fixed prediction and LPC, with
// Rice residuals and stereo decorrelation. glint_decode_audio / _ex also
// auto-detect FLAC by its stream marker.
// ---------------------------------------------------------------------------

float* glint_flac_decode(const uint8_t* flac, int len, int* out_sr,
                         int* out_ch, int* out_frames);
void* glint_flac_decode_ex(const uint8_t* flac, int len, int out_rate,
                           int want_int16, int* out_sr, int* out_ch,
                           int* out_frames);

glint_mp3_dec_t glint_mp3_dec_create(void);
// Decode ONE frame at data[0]. pcm must hold samples*channels floats
// (1152*2 is always enough for MP3, 1024*2 for AAC). Returns samples per
// channel written (0 or a frame's worth), or a negative error.
int  glint_mp3_decode(glint_mp3_dec_t dec, const uint8_t* data, int len,
                      float* pcm, struct glint_dec_frame_info* info);
void glint_mp3_dec_destroy(glint_mp3_dec_t dec);

glint_aac_dec_t glint_aac_dec_create(void);
int  glint_aac_decode(glint_aac_dec_t dec, const uint8_t* data, int len,
                      float* pcm, struct glint_dec_frame_info* info);
void glint_aac_dec_destroy(glint_aac_dec_t dec);

typedef struct glint_opus_dec_context* glint_opus_dec_t;
typedef struct glint_opus_ms_dec_context* glint_opus_ms_dec_t;
typedef struct glint_opus_enc_context* glint_opus_enc_t;

// channels 1 or 2; sample_rate 48000/24000/16000/12000/8000.
glint_opus_dec_t glint_opus_dec_create(int channels, int sample_rate);
// Decode one packet to interleaved float PCM (+-1.0). Returns samples
// per channel at sample_rate, or negative on error. max_samples guards
// pcm (per channel; 5760*48000ths covers any packet).
int      glint_opus_decode(glint_opus_dec_t dec, const uint8_t* packet,
                           int len, float* pcm, int max_samples);
// Conceal a LOST packet of frame_size samples per channel. packet (the
// one FOLLOWING the loss) may carry SILK in-band FEC; pass NULL for
// plain concealment. Returns frame_size or negative.
int      glint_opus_decode_fec(glint_opus_dec_t dec, const uint8_t* packet,
                               int len, float* pcm, int frame_size);
// Range-coder state after the last decode (Opus conformance identity;
// equals the encoder's final range for a correctly decoded packet).
uint32_t glint_opus_dec_final_range(glint_opus_dec_t dec);
void     glint_opus_dec_destroy(glint_opus_dec_t dec);

// Multistream/surround (RFC 7845 mapping family 1, up to 8 channels).
// streams elementary streams, the first `coupled` of them stereo;
// mapping[channels] assigns stream channels to outputs (255 = silent).
glint_opus_ms_dec_t glint_opus_ms_dec_create(int channels, int streams,
                                             int coupled,
                                             const uint8_t* mapping,
                                             int sample_rate);
int      glint_opus_ms_decode(glint_opus_ms_dec_t dec,
                              const uint8_t* packet, int len, float* pcm,
                              int max_samples);
void     glint_opus_ms_dec_destroy(glint_opus_ms_dec_t dec);

// CELT-only encoder: 48 kHz interleaved float input, channels 1 or 2.
// vbr = 0 for CBR at bitrate_bps, 1 for unconstrained VBR targeting it.
glint_opus_enc_t glint_opus_enc_create(int channels, int bitrate_bps,
                                       int vbr);
// Encode one frame (frame_size = 120, 240, 480 or 960 samples/channel).
// Writes a complete Opus packet (TOC + payload) into out; returns its
// size in bytes, or negative on error. max_bytes >= 1276 always works.
int      glint_opus_encode(glint_opus_enc_t enc, const float* pcm,
                           int frame_size, uint8_t* out, int max_bytes);
uint32_t glint_opus_enc_final_range(glint_opus_enc_t enc);
void     glint_opus_enc_destroy(glint_opus_enc_t enc);

// One-shot: encode interleaved 48 kHz float PCM (±1.0, `frames` per
// channel, 1-2 channels) to a complete Ogg-Opus file (CELT-only, 20 ms
// frames, pre-skip 120). vbr!=0 selects unconstrained VBR at bitrate_bps.
// Returns a malloc'd buffer of *out_size bytes — free with glint_free —
// or NULL on error.
uint8_t* glint_opus_encode_file(const float* pcm, int frames, int channels,
                                int bitrate_bps, int vbr, int* out_size);

#ifdef __cplusplus
}
#endif

#endif // GLINT_H
