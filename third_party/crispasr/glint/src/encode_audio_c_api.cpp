// glint - one-call encode C ABI: interleaved float PCM (any rate/channels)
// -> MP3 / AAC-LC / Ogg-Opus, auto-resampling to a codec-valid rate.
// Centralizes what the CLI's encode path does so every wrapper gets a
// single "encode this PCM to that codec" call.
// MIT License - Clean-room implementation.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "glint/glint.h"
#include "resample.hpp"

namespace {

// Nearest supported sample rate for a codec (resample the input if it is
// not already valid). MP3 spans MPEG-1/2/2.5; AAC the standard set.
const int kMp3Rates[] = {8000, 11025, 12000, 16000, 22050, 24000,
                         32000, 44100, 48000};
const int kAacRates[] = {8000, 11025, 12000, 16000, 22050, 24000, 32000,
                         44100, 48000, 64000, 88200, 96000};

int nearest_rate(const int* rates, int n, int sr) {
    int best = rates[0];
    long bestd = -1;
    for (int i = 0; i < n; i++) {
        long d = rates[i] > sr ? rates[i] - sr : sr - rates[i];
        if (bestd < 0 || d < bestd) {
            bestd = d;
            best = rates[i];
        }
    }
    return best;
}

// De-interleave `frames` of `channels`-interleaved float into per-channel
// planar buffers for the encoder's channel-pointer API.
void deinterleave(const float* pcm, long frames, int channels,
                  std::vector<std::vector<float>>& planar) {
    planar.assign(channels, std::vector<float>(frames));
    for (long i = 0; i < frames; i++)
        for (int c = 0; c < channels; c++)
            planar[c][i] = pcm[i * channels + c];
}

uint8_t* dup(const std::vector<uint8_t>& v, int* out_size) {
    uint8_t* buf = static_cast<uint8_t*>(std::malloc(v.empty() ? 1 : v.size()));
    if (!buf) return nullptr;
    if (!v.empty()) std::memcpy(buf, v.data(), v.size());
    if (out_size) *out_size = static_cast<int>(v.size());
    return buf;
}

std::vector<uint8_t> encode_mp3_planar(const float* pcm, long frames,
                                       int channels, int sr, int bitrate,
                                       int vbr_q, int quality) {
    struct glint_config cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.sample_rate = sr;
    cfg.num_channels = channels;
    cfg.mode = channels == 1 ? GLINT_MONO : GLINT_JOINT;
    cfg.bitrate = bitrate;
    cfg.quality = static_cast<enum glint_quality>(quality);
    if (vbr_q >= 0) {
        cfg.vbr = GLINT_VBR_ON;
        cfg.vbr_quality = vbr_q;
    }
    glint_t enc = glint_create(&cfg);
    std::vector<uint8_t> out;
    if (!enc) return out;
    int spf = glint_samples_per_frame(enc);

    std::vector<std::vector<float>> planar;
    deinterleave(pcm, frames, channels, planar);
    std::vector<const float*> chp(channels);
    long off = 0;
    while (off < frames) {
        int got = static_cast<int>(frames - off < spf ? frames - off : spf);
        std::vector<std::vector<float>> block(channels,
                                              std::vector<float>(spf, 0.0f));
        for (int c = 0; c < channels; c++) {
            std::memcpy(block[c].data(), planar[c].data() + off,
                        sizeof(float) * got);
            chp[c] = block[c].data();
        }
        int n = 0;
        const uint8_t* p = glint_encode_float(enc, chp.data(), &n);
        if (p && n > 0) out.insert(out.end(), p, p + n);
        off += spf;
    }
    int n = 0;
    const uint8_t* p = glint_flush(enc, &n);
    if (p && n > 0) out.insert(out.end(), p, p + n);
    glint_destroy(enc);
    return out;
}

std::vector<uint8_t> encode_aac_planar(const float* pcm, long frames,
                                       int channels, int sr, int bitrate,
                                       int vbr_q, int quality) {
    struct glint_aac_config cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.sample_rate = sr;
    cfg.num_channels = channels;
    cfg.bitrate = bitrate;
    cfg.quality = static_cast<enum glint_quality>(quality);
    if (vbr_q >= 0) {
        cfg.vbr = 1;
        cfg.vbr_quality = vbr_q;
    }
    glint_aac_t enc = glint_aac_create(&cfg);
    std::vector<uint8_t> out;
    if (!enc) return out;
    int spf = glint_aac_samples_per_frame(enc);

    std::vector<std::vector<float>> planar;
    deinterleave(pcm, frames, channels, planar);
    std::vector<const float*> chp(channels);
    long off = 0;
    while (off < frames) {
        int got = static_cast<int>(frames - off < spf ? frames - off : spf);
        std::vector<std::vector<float>> block(channels,
                                              std::vector<float>(spf, 0.0f));
        for (int c = 0; c < channels; c++) {
            std::memcpy(block[c].data(), planar[c].data() + off,
                        sizeof(float) * got);
            chp[c] = block[c].data();
        }
        int n = 0;
        const uint8_t* p = glint_aac_encode_float(enc, chp.data(), &n);
        if (p && n > 0) out.insert(out.end(), p, p + n);
        off += spf;
    }
    int n = 0;
    const uint8_t* p = glint_aac_flush(enc, &n);
    if (p && n > 0) out.insert(out.end(), p, p + n);
    glint_aac_destroy(enc);
    return out;
}

}  // namespace

extern "C" {

uint8_t* glint_encode_audio(const float* pcm, int frames, int channels,
                            int sample_rate, int format, int bitrate_kbps,
                            int vbr_quality, int quality, int* out_size) {
    if (out_size) *out_size = 0;
    if (!pcm || frames <= 0 || channels < 1 || channels > 2 ||
        sample_rate < 1)
        return nullptr;

    // Pick a codec-valid rate; resample if the input isn't already there.
    int target = sample_rate;
    if (format == GLINT_ENC_OPUS) {
        target = 48000;
    } else if (format == GLINT_ENC_MP3) {
        target = nearest_rate(kMp3Rates,
                              (int)(sizeof(kMp3Rates) / sizeof(int)),
                              sample_rate);
    } else if (format == GLINT_ENC_AAC) {
        target = nearest_rate(kAacRates,
                              (int)(sizeof(kAacRates) / sizeof(int)),
                              sample_rate);
    } else {
        return nullptr;
    }

    std::vector<float> resampled;
    const float* use = pcm;
    long use_frames = frames;
    if (target != sample_rate) {
        int nf = 0;
        resampled = glint::resample(pcm, frames, channels, sample_rate,
                                    target, &nf);
        use = resampled.data();
        use_frames = nf;
    }

    std::vector<uint8_t> out;
    if (format == GLINT_ENC_MP3) {
        out = encode_mp3_planar(use, use_frames, channels, target,
                                bitrate_kbps, vbr_quality, quality);
    } else if (format == GLINT_ENC_AAC) {
        out = encode_aac_planar(use, use_frames, channels, target,
                                bitrate_kbps, vbr_quality, quality);
    } else {  // Opus — reuse the muxing whole-file encoder.
        int n = 0;
        uint8_t* p = glint_opus_encode_file(use, static_cast<int>(use_frames),
                                            channels, bitrate_kbps * 1000,
                                            vbr_quality >= 0 ? 1 : 0, &n);
        if (!p) return nullptr;
        out.assign(p, p + n);
        glint_free(p);
    }
    if (out.empty()) return nullptr;
    return dup(out, out_size);
}

}  // extern "C"
