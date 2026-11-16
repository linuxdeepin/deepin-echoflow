// Opus packet decoding — RFC 6716 section 3
// MIT License - Clean-room implementation

#include "opus_decoder.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "opus_mdct.hpp"

namespace glint {
namespace opus {

void pcm_soft_clip(float* x_, int n, int channels, float* declip_mem) {
    if (channels < 1 || n < 1 || !x_ || !declip_mem) return;

    // The nonlinearity's derivative hits zero at ±2; saturate there first.
    for (int i = 0; i < n * channels; i++) {
        float v = x_[i];
        x_[i] = v < -2.f ? -2.f : (v > 2.f ? 2.f : v);
    }
    for (int c = 0; c < channels; c++) {
        float* x = x_ + c;
        float a = declip_mem[c];
        // Keep applying the previous frame's curve until a sign change so
        // the nonlinearity is continuous across the frame boundary.
        for (int i = 0; i < n; i++) {
            if (x[i * channels] * a >= 0) break;
            x[i * channels] += a * x[i * channels] * x[i * channels];
        }

        int curr = 0;
        float x0 = x[0];
        for (;;) {
            int i;
            for (i = curr; i < n; i++) {
                if (x[i * channels] > 1 || x[i * channels] < -1) break;
            }
            if (i == n) {
                a = 0;
                break;
            }
            int peak_pos = i;
            int start = i, end = i;
            float maxval = std::fabs(x[i * channels]);
            // Expand to the zero crossings around the excursion, tracking
            // the biggest peak inside it.
            while (start > 0 &&
                   x[i * channels] * x[(start - 1) * channels] >= 0)
                start--;
            while (end < n && x[i * channels] * x[end * channels] >= 0) {
                if (std::fabs(x[end * channels]) > maxval) {
                    maxval = std::fabs(x[end * channels]);
                    peak_pos = end;
                }
                end++;
            }
            int special = start == 0 && x[i * channels] * x[0] >= 0;

            // a such that maxval + a*maxval^2 == 1 (tiny boost keeps
            // -ffast-math builds strictly inside ±1).
            a = (maxval - 1) / (maxval * maxval);
            a += a * 2.4e-7f;
            if (x[i * channels] > 0) a = -a;
            for (i = start; i < end; i++)
                x[i * channels] += a * x[i * channels] * x[i * channels];

            if (special && peak_pos >= 2) {
                // Ramp from the frame's first sample to the peak to avoid a
                // discontinuity at the frame boundary.
                float offset = x0 - x[0];
                float delta = offset / peak_pos;
                for (i = curr; i < peak_pos; i++) {
                    offset -= delta;
                    x[i * channels] += offset;
                    float v = x[i * channels];
                    x[i * channels] = v < -1.f ? -1.f : (v > 1.f ? 1.f : v);
                }
            }
            curr = end;
            if (curr == n) break;
        }
        declip_mem[c] = a;
    }
}

namespace {

// One- or two-byte frame length (RFC 3.2.1). Returns bytes consumed, -1 on
// truncation; *size gets the frame length (0..1275).
int parse_size(const uint8_t* data, int32_t len, int16_t* size) {
    if (len < 1) {
        *size = -1;
        return -1;
    }
    if (data[0] < 252) {
        *size = data[0];
        return 1;
    }
    if (len < 2) {
        *size = -1;
        return -1;
    }
    *size = static_cast<int16_t>(4 * data[1] + data[0]);
    return 2;
}

}  // namespace

int opus_packet_parse_ext(const uint8_t* data, int32_t len,
                          bool self_delimited, OpusPacket* pkt,
                          int32_t* packet_offset) {
    const uint8_t* data0 = data;
    if (len < 1) return -1;
    const uint8_t toc = data[0];
    pkt->config = toc >> 3;
    pkt->stereo = (toc >> 2) & 1;
    data++;
    len--;

    // Audio size from config: CELT-only (16..31) uses 2.5/5/10/20 ms.
    if (pkt->config >= 16) {
        pkt->frame_size = 120 << (pkt->config & 3);
    } else if (pkt->config < 12) {
        // SILK-only: 10/20/40/60 ms.
        static const int kSilkSizes[4] = { 480, 960, 1920, 2880 };
        pkt->frame_size = kSilkSizes[pkt->config & 3];
    } else {
        // Hybrid: 10/20 ms.
        pkt->frame_size = 480 << (pkt->config & 1);
    }

    int count;
    int16_t size0;
    const int code = toc & 3;
    int32_t pad = 0;
    bool cbr = true;
    switch (code) {
    case 0:
        count = 1;
        pkt->sizes[0] = static_cast<int16_t>(len);
        break;
    case 1:
        count = 2;
        if (!self_delimited) {
            if (len & 1) return -4;
            pkt->sizes[0] = pkt->sizes[1] = static_cast<int16_t>(len / 2);
        }
        break;
    case 2: {
        count = 2;
        int used = parse_size(data, len, &size0);
        if (used < 0 || size0 > len - used) return -4;
        pkt->sizes[0] = size0;
        pkt->sizes[1] = static_cast<int16_t>(len - used - size0);
        data += used;
        len -= used;
        cbr = false;
        break;
    }
    default: {  // code 3: arbitrary count, optional padding, CBR or VBR
        if (len < 1) return -4;
        const uint8_t ch = data[0];
        count = ch & 0x3F;
        if (count <= 0 || count > 48 ||
            pkt->frame_size * count > 5760 /* 120 ms cap */)
            return -4;
        const bool vbr = (ch & 0x80) != 0;
        const bool padded = (ch & 0x40) != 0;
        data++;
        len--;
        if (padded) {
            // Padding length: 255 bytes chain as 254 + continue.
            for (;;) {
                if (len <= 0) return -4;
                int p = data[0];
                data++;
                len--;
                pad += p == 255 ? 254 : p;
                if (p != 255) break;
            }
        }
        if (pad > len) return -4;
        len -= pad;
        if (vbr) {
            cbr = false;
            int32_t total = 0;
            for (int i = 0; i < count - 1; i++) {
                int used = parse_size(data, len, &pkt->sizes[i]);
                if (used < 0) return -4;
                data += used;
                len -= used;
                if (pkt->sizes[i] > len - total) return -4;
                total += pkt->sizes[i];
            }
            pkt->sizes[count - 1] = static_cast<int16_t>(len - total);
        }
        if (cbr && !self_delimited) {
            if (len % count) return -4;
            for (int i = 0; i < count; i++)
                pkt->sizes[i] = static_cast<int16_t>(len / count);
        }
        break;
    }
    }

    // Self-delimited framing: an explicit size for the LAST frame (for
    // CBR it applies to every frame); the packet may end before `len`.
    if (self_delimited) {
        int16_t last;
        int used = parse_size(data, len, &last);
        if (used < 0 || last > len - used) return -4;
        data += used;
        len -= used;
        if (cbr) {
            if (static_cast<int32_t>(last) * count > len) return -4;
            for (int i = 0; i < count; i++) pkt->sizes[i] = last;
        } else {
            // VBR: sizes[0..count-2] already parsed; the last frame's
            // explicit size must fit in what remains.
            int32_t rest = len;
            for (int i = 0; i < count - 1; i++) rest -= pkt->sizes[i];
            if (last > rest) return -4;
            pkt->sizes[count - 1] = last;
        }
    }

    // A frame must not exceed the 1275-byte cap.
    int32_t offset = 0;
    for (int i = 0; i < count; i++) {
        if (pkt->sizes[i] < 0 || pkt->sizes[i] > 1275) return -4;
        pkt->frames[i] = data + offset;
        offset += pkt->sizes[i];
    }
    if (offset > len) return -4;
    pkt->frame_count = count;
    if (packet_offset)
        *packet_offset =
            pad + static_cast<int32_t>(data - data0) + offset;
    return 0;
}

int opus_packet_parse(const uint8_t* data, int32_t len, OpusPacket* pkt) {
    return opus_packet_parse_ext(data, len, false, pkt, nullptr);
}

namespace {
// The CELT overlap window doubles as the transition fade (2.5 ms @48k).
const double* celt_window() {
    static double w[120];
    static bool once = [] {
        mdct_window_fill(w, 120);
        return true;
    }();
    (void)once;
    return w;
}

// Crossfade in1 -> in2 with the squared CELT overlap window (2.5 ms).
void smooth_fade(const float* in1, const float* in2, float* out,
                 int overlap, int channels, const double* window,
                 int inc = 1) {
    // inc = 48000/Fs: the 48k CELT window sampled at the output rate.
    for (int c = 0; c < channels; c++) {
        for (int i = 0; i < overlap; i++) {
            float w = static_cast<float>(window[i * inc] * window[i * inc]);
            out[i * channels + c] = w * in2[i * channels + c] +
                                    (1.0f - w) * in1[i * channels + c];
        }
    }
}
}  // namespace

int OpusDecoder::decode_frame_impl(const uint8_t* data, int32_t size,
                                   float* pcm, int frame_size, int config,
                                   int stereo_flag, int fec) {
    const int stream_channels = stereo_flag ? 2 : 1;
    int mode, audiosize, end_band = 21;
    constexpr int kF20 = 960, kF10 = 480, kF5 = 240, kF2_5 = 120;
    // Control flow runs in 48 kHz units; pcm positions in output units.
    const int ds = 48000 / fs_;
    const int f2_5 = kF2_5 / ds, f5 = kF5 / ds;

    if (data != nullptr) {
        audiosize = frame_size;
        mode = config < 12 ? 1 : (config < 16 ? 2 : 3);
        if (mode == 1)
            end_band = (config >> 2) == 0 ? 13 : 17;  // NB / MB+WB
        else if (mode == 2)
            end_band = (config & 2) ? 21 : 19;  // SWB 19, FB 21
        else
            end_band = ((config >> 2) & 3) == 0
                           ? 13
                           : (((config >> 2) & 3) == 1
                                  ? 17
                                  : (((config >> 2) & 3) == 2 ? 19 : 21));
    } else {
        // Concealment: continue in the last mode (CELT if it was primed
        // by redundancy).
        audiosize = frame_size;
        mode = prev_redundancy_ ? 3 : prev_mode_;
        end_band = 21;
        if (mode == 0) {
            // Nothing decoded yet: silence.
            std::memset(pcm, 0, static_cast<size_t>(audiosize / ds) *
                                    channels_ * sizeof(float));
            return audiosize;
        }
        if (audiosize > kF20) {
            do {
                int ret = decode_frame_impl(
                    nullptr, 0, pcm, audiosize < kF20 ? audiosize : kF20,
                    config, stereo_flag);
                if (ret < 0) return ret;
                pcm += (ret / ds) * channels_;
                audiosize -= ret;
            } while (audiosize > 0);
            return frame_size;
        }
        if (audiosize < kF20) {
            if (audiosize > kF10)
                audiosize = kF10;
            else if (mode != 1 && audiosize > kF5 && audiosize < kF10)
                audiosize = kF5;
        }
        frame_size = audiosize;
    }

    // Mode-transition fade source: a 5 ms concealment frame in the OLD
    // mode, when the switch has no redundancy to cover it.
    int transition = 0;
    float pcm_transition[2 * 240];
    if (data != nullptr && prev_mode_ > 0 &&
        ((mode == 3 && prev_mode_ != 3 && !prev_redundancy_) ||
         (mode != 3 && prev_mode_ == 3)))
        transition = 1;
    if (transition && mode == 3) {
        int ret = decode_frame_impl(nullptr, 0, pcm_transition,
                                    audiosize < kF5 ? audiosize : kF5,
                                    config, stereo_flag);
        if (ret < 0) return ret;
    }

    RangeDecoder dec;
    if (data != nullptr) dec.init(data, static_cast<uint32_t>(size));

    int32_t len = size;
    int16_t pcm_silk[2 * 2880];
    int n_silk = 0;
    if (mode != 3) {
        if (prev_mode_ == 3) silk_ = silk::SilkDecoder();
        // The SILK PLC cannot conceal less than 10 ms.
        int payload_ms = audiosize / 48 > 10 ? audiosize / 48 : 10;
        int internal_khz = 16;
        if (data != nullptr && mode == 1)
            internal_khz = 8 + 4 * ((config >> 2) & 3);  // NB/MB/WB
        // n_silk counts OUTPUT-rate samples (SILK resamples to fs_
        // directly — at a matching internal rate that's no resampling
        // at all).
        while (n_silk < frame_size / ds) {
            int ret;
            if (data != nullptr)
                ret = silk_.decode(dec, pcm_silk + n_silk * channels_,
                                   channels_, stream_channels,
                                   internal_khz, fs_, payload_ms,
                                   n_silk == 0, fec != 0);
            else
                ret = silk_.decode_lost(pcm_silk + n_silk * channels_,
                                        channels_, payload_ms,
                                        n_silk == 0);
            if (ret < 0) return ret;
            n_silk += ret;
        }
    }

    int redundancy = 0, celt_to_silk = 0;
    int32_t redundancy_bytes = 0;
    if (data != nullptr && !fec && mode != 3 &&
        static_cast<int32_t>(dec.tell()) + 17 + 20 * (mode == 2) <=
            8 * len) {
        redundancy = mode == 2 ? dec.dec_bit_logp(12) : 1;
        if (redundancy) {
            celt_to_silk = dec.dec_bit_logp(1);
            redundancy_bytes =
                mode == 2
                    ? static_cast<int32_t>(dec.dec_uint(256)) + 2
                    : len - ((static_cast<int32_t>(dec.tell()) + 7) >> 3);
            len -= redundancy_bytes;
            if (len * 8 < static_cast<int32_t>(dec.tell())) {
                len = 0;
                redundancy_bytes = 0;
                redundancy = 0;
            }
            dec.shrink(static_cast<uint32_t>(redundancy_bytes));
        }
    }

    // A redundant frame covers the switch better than a concealment fade.
    if (redundancy) transition = 0;

    if (transition && mode != 3) {
        int ret = decode_frame_impl(nullptr, 0, pcm_transition,
                                    audiosize < kF5 ? audiosize : kF5,
                                    config, stereo_flag);
        if (ret < 0) return ret;
    }

    float redundant_audio[2 * 240];
    uint32_t redundant_rng = 0;
    if (redundancy && celt_to_silk) {
        // Always decoded (the final range needs it) even when the audio
        // is stale and unused (gated application below).
        RangeDecoder rdec;
        rdec.init(data + len, static_cast<uint32_t>(redundancy_bytes));
        int ret = celt_.decode_frame(
            rdec, static_cast<uint32_t>(redundancy_bytes), redundant_audio,
            kF5, stream_channels, end_band, 0);
        if (ret < 0) return ret;
        redundant_rng = rdec.range();
    }

    if (mode != 1) {
        // CELT part (hybrid: same range decoder, bands from 17).
        int start_band = mode == 2 ? 17 : 0;
        int celt_frame = frame_size < kF20 ? frame_size : kF20;
        if (mode != prev_mode_ && prev_mode_ > 0 && !prev_redundancy_)
            celt_.init(channels_, fs_);
        int ret;
        if (data != nullptr && !fec)
            ret = celt_.decode_frame(dec, static_cast<uint32_t>(len), pcm,
                                     celt_frame, stream_channels, end_band,
                                     start_band);
        else
            ret = celt_.decode_lost(pcm, celt_frame, start_band, end_band);
        if (ret < 0) return ret;
    } else {
        std::memset(pcm, 0, static_cast<size_t>(frame_size / ds) *
                                channels_ * sizeof(float));
        // Hybrid -> SILK: decode a silence frame so the CELT MDCT fades
        // itself out.
        if (prev_mode_ == 2 &&
            !(redundancy && celt_to_silk && prev_redundancy_)) {
            static const uint8_t kSilence[2] = { 0xFF, 0xFF };
            RangeDecoder sdec;
            sdec.init(kSilence, 2);
            celt_.decode_frame(sdec, 2, pcm, kF2_5, stream_channels, 21,
                               0);
        }
    }

    if (mode != 3) {
        for (int i = 0; i < (frame_size / ds) * channels_; i++)
            pcm[i] += pcm_silk[i] * (1.0f / 32768.0f);
    }


    if (redundancy && !celt_to_silk) {
        // SILK->CELT: prime a fresh CELT state, fade the tail into it.
        celt_.init(channels_, fs_);
        RangeDecoder rdec;
        rdec.init(data + len, static_cast<uint32_t>(redundancy_bytes));
        int ret = celt_.decode_frame(
            rdec, static_cast<uint32_t>(redundancy_bytes), redundant_audio,
            kF5, stream_channels, end_band, 0);
        if (ret < 0) return ret;
        redundant_rng = rdec.range();
        smooth_fade(pcm + channels_ * (frame_size / ds - f2_5),
                    redundant_audio + channels_ * f2_5,
                    pcm + channels_ * (frame_size / ds - f2_5), f2_5,
                    channels_, celt_window(), ds);
    }
    if (redundancy && celt_to_silk &&
        (prev_mode_ != 1 || prev_redundancy_)) {
        // Apply only if the previous frame actually used CELT (its first
        // redundancy frame may have been lost).
        for (int c = 0; c < channels_; c++)
            for (int i = 0; i < f2_5; i++)
                pcm[channels_ * i + c] = redundant_audio[channels_ * i + c];
        smooth_fade(redundant_audio + channels_ * f2_5,
                    pcm + channels_ * f2_5, pcm + channels_ * f2_5,
                    f2_5, channels_, celt_window(), ds);
    }
    if (transition) {
        if (audiosize >= kF5) {
            for (int i = 0; i < channels_ * f2_5; i++)
                pcm[i] = pcm_transition[i];
            smooth_fade(pcm_transition + channels_ * f2_5,
                        pcm + channels_ * f2_5, pcm + channels_ * f2_5,
                        f2_5, channels_, celt_window(), ds);
        } else {
            smooth_fade(pcm_transition, pcm, pcm, f2_5, channels_,
                        celt_window(), ds);
        }
    }
    (void)f5;

    prev_mode_ = mode;
    prev_redundancy_ = redundancy && !celt_to_silk;
    final_range_ = data != nullptr ? dec.range() ^ redundant_rng : 0;
    return frame_size;
}

int OpusDecoder::decode_fec(const uint8_t* data, int32_t len, float* pcm,
                            int frame_size) {
    // frame_size is in OUTPUT-rate samples (like the reference API).
    const int ds = 48000 / fs_;
    const int frame48 = frame_size * ds;
    OpusPacket pkt;
    bool have = data != nullptr && len > 0 &&
                opus_packet_parse(data, len, &pkt) == 0;
    int packet_mode =
        have ? (pkt.config < 12 ? 1 : (pkt.config < 16 ? 2 : 3)) : 0;
    if (!have || frame48 < pkt.frame_size || packet_mode == 3 ||
        prev_mode_ == 3) {
        // No usable FEC: plain concealment for the whole gap.
        int ret = decode_frame_impl(nullptr, 0, pcm, frame48, 0,
                                    channels_ == 2);
        return ret < 0 ? ret : ret / ds;
    }
    // Conceal everything the redundant copy doesn't cover, then decode
    // the FIRST frame's LBRR data for the tail (reference
    // opus_decode_native decode_fec path).
    int plc48 = frame48 - pkt.frame_size;
    if (plc48 > 0) {
        int ret = decode_frame_impl(nullptr, 0, pcm, plc48, pkt.config,
                                    pkt.stereo);
        if (ret < 0) return ret;
    }
    int ret = decode_frame_impl(pkt.frames[0], pkt.sizes[0],
                                pcm + (plc48 / ds) * channels_,
                                pkt.frame_size, pkt.config, pkt.stereo,
                                /*fec=*/1);
    if (ret < 0) return ret;
    return frame_size;
}

int OpusDecoder::decode(const uint8_t* data, int32_t len, float* pcm,
                        int max_samples) {
    OpusPacket pkt;
    int err = opus_packet_parse(data, len, &pkt);
    if (err) return err;
    return decode_parsed(pkt, pcm, max_samples);
}

int OpusDecoder::decode_parsed(const OpusPacket& pkt, float* pcm,
                               int max_samples) {
    const int ds = 48000 / fs_;
    if (pkt.frame_count * pkt.frame_size / ds > max_samples) return -2;

    int total = 0;  // output-rate samples
    for (int f = 0; f < pkt.frame_count; f++) {
        int ret;
        if (pkt.sizes[f] < 1)  // DTX: conceal in the previous mode
            ret = decode_frame_impl(nullptr, 0, pcm + total * channels_,
                                    pkt.frame_size, pkt.config,
                                    pkt.stereo);
        else
            ret = decode_frame_impl(pkt.frames[f], pkt.sizes[f],
                                    pcm + total * channels_,
                                    pkt.frame_size, pkt.config,
                                    pkt.stereo);
        if (ret < 0) return ret;
        total += ret / ds;
    }
    return total;
}

}  // namespace opus
}  // namespace glint
