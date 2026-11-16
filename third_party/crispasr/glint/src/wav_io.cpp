// glint - WAV / raw-PCM I/O implementation. See wav_io.hpp.
// MIT License - Clean-room implementation.

#include "wav_io.hpp"

#include <cstring>

namespace glint {

namespace {

constexpr uint16_t WAVE_FORMAT_PCM        = 0x0001;
constexpr uint16_t WAVE_FORMAT_IEEE_FLOAT = 0x0003;
constexpr uint16_t WAVE_FORMAT_ALAW       = 0x0006;
constexpr uint16_t WAVE_FORMAT_MULAW      = 0x0007;
constexpr uint16_t WAVE_FORMAT_EXTENSIBLE = 0xFFFE;

// ITU-T G.711 A-law decode table (256 entries).
constexpr int16_t alaw_table[256] = {
    -5504, -5248, -6016, -5760, -4480, -4224, -4992, -4736,
    -7552, -7296, -8064, -7808, -6528, -6272, -7040, -6784,
    -2752, -2624, -3008, -2880, -2240, -2112, -2496, -2368,
    -3776, -3648, -4032, -3904, -3264, -3136, -3520, -3392,
    -22016,-20992,-24064,-23040,-17920,-16896,-19968,-18944,
    -30208,-29184,-32256,-31232,-26112,-25088,-28160,-27136,
    -11008,-10496,-12032,-11520, -8960, -8448, -9984, -9472,
    -15104,-14592,-16128,-15616,-13056,-12544,-14080,-13568,
      -344,  -328,  -376,  -360,  -280,  -264,  -312,  -296,
      -472,  -456,  -504,  -488,  -408,  -392,  -440,  -424,
      -88,   -72,  -120,  -104,   -24,    -8,   -56,   -40,
     -216,  -200,  -248,  -232,  -152,  -136,  -184,  -168,
    -1376, -1312, -1504, -1440, -1120, -1056, -1248, -1184,
    -1888, -1824, -2016, -1952, -1632, -1568, -1760, -1696,
     -688,  -656,  -752,  -720,  -560,  -528,  -624,  -592,
     -944,  -912, -1008,  -976,  -816,  -784,  -880,  -848,
     5504,  5248,  6016,  5760,  4480,  4224,  4992,  4736,
     7552,  7296,  8064,  7808,  6528,  6272,  7040,  6784,
     2752,  2624,  3008,  2880,  2240,  2112,  2496,  2368,
     3776,  3648,  4032,  3904,  3264,  3136,  3520,  3392,
    22016, 20992, 24064, 23040, 17920, 16896, 19968, 18944,
    30208, 29184, 32256, 31232, 26112, 25088, 28160, 27136,
    11008, 10496, 12032, 11520,  8960,  8448,  9984,  9472,
    15104, 14592, 16128, 15616, 13056, 12544, 14080, 13568,
      344,   328,   376,   360,   280,   264,   312,   296,
      472,   456,   504,   488,   408,   392,   440,   424,
       88,    72,   120,   104,    24,     8,    56,    40,
      216,   200,   248,   232,   152,   136,   184,   168,
     1376,  1312,  1504,  1440,  1120,  1056,  1248,  1184,
     1888,  1824,  2016,  1952,  1632,  1568,  1760,  1696,
      688,   656,   752,   720,   560,   528,   624,   592,
      944,   912,  1008,   976,   816,   784,   880,   848,
};

// ITU-T G.711 mu-law decode table (256 entries).
constexpr int16_t ulaw_table[256] = {
    -32124,-31100,-30076,-29052,-28028,-27004,-25980,-24956,
    -23932,-22908,-21884,-20860,-19836,-18812,-17788,-16764,
    -15996,-15484,-14972,-14460,-13948,-13436,-12924,-12412,
    -11900,-11388,-10876,-10364, -9852, -9340, -8828, -8316,
     -7932, -7676, -7420, -7164, -6908, -6652, -6396, -6140,
     -5884, -5628, -5372, -5116, -4860, -4604, -4348, -4092,
     -3900, -3772, -3644, -3516, -3388, -3260, -3132, -3004,
     -2876, -2748, -2620, -2492, -2364, -2236, -2108, -1980,
     -1884, -1820, -1756, -1692, -1628, -1564, -1500, -1436,
     -1372, -1308, -1244, -1180, -1116, -1052,  -988,  -924,
      -876,  -844,  -812,  -780,  -748,  -716,  -684,  -652,
      -620,  -588,  -556,  -524,  -492,  -460,  -428,  -396,
      -372,  -356,  -340,  -324,  -308,  -292,  -276,  -260,
      -244,  -228,  -212,  -196,  -180,  -164,  -148,  -132,
      -120,  -112,  -104,   -96,   -88,   -80,   -72,   -64,
       -56,   -48,   -40,   -32,   -24,   -16,    -8,     0,
     32124, 31100, 30076, 29052, 28028, 27004, 25980, 24956,
     23932, 22908, 21884, 20860, 19836, 18812, 17788, 16764,
     15996, 15484, 14972, 14460, 13948, 13436, 12924, 12412,
     11900, 11388, 10876, 10364,  9852,  9340,  8828,  8316,
      7932,  7676,  7420,  7164,  6908,  6652,  6396,  6140,
      5884,  5628,  5372,  5116,  4860,  4604,  4348,  4092,
      3900,  3772,  3644,  3516,  3388,  3260,  3132,  3004,
      2876,  2748,  2620,  2492,  2364,  2236,  2108,  1980,
      1884,  1820,  1756,  1692,  1628,  1564,  1500,  1436,
      1372,  1308,  1244,  1180,  1116,  1052,   988,   924,
       876,   844,   812,   780,   748,   716,   684,   652,
       620,   588,   556,   524,   492,   460,   428,   396,
       372,   356,   340,   324,   308,   292,   276,   260,
       244,   228,   212,   196,   180,   164,   148,   132,
       120,   112,   104,    96,    88,    80,    72,    64,
        56,    48,    40,    32,    24,    16,     8,     0,
};

const uint8_t GUID_PCM[16] = {
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00,
    0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71};
const uint8_t GUID_FLOAT[16] = {
    0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00,
    0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71};
const uint8_t GUID_ALAW[16] = {
    0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00,
    0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71};
const uint8_t GUID_MULAW[16] = {
    0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00,
    0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71};

uint32_t rd32(const uint8_t* p) {
    return p[0] | (p[1] << 8) | (p[2] << 16) | (uint32_t)p[3] << 24;
}
uint16_t rd16(const uint8_t* p) { return p[0] | (p[1] << 8); }

float sample_to_float(const uint8_t* d, uint16_t fmt, int bps) {
    switch (fmt) {
    case WAVE_FORMAT_PCM:
        switch (bps) {
        case 8:
            return (static_cast<int>(d[0]) - 128) / 128.0f;
        case 16: {
            int16_t s;
            std::memcpy(&s, d, 2);
            return s / 32768.0f;
        }
        case 24: {
            int32_t v = d[0] | (d[1] << 8) | (d[2] << 16);
            if (v & 0x800000) v |= 0xFF000000;
            return v / 8388608.0f;
        }
        case 32: {
            int32_t v;
            std::memcpy(&v, d, 4);
            return static_cast<float>(v / 2147483648.0);
        }
        }
        return 0;
    case WAVE_FORMAT_IEEE_FLOAT:
        if (bps == 32) {
            float f;
            std::memcpy(&f, d, 4);
            return f;
        } else if (bps == 64) {
            double g;
            std::memcpy(&g, d, 8);
            return static_cast<float>(g);
        }
        return 0;
    case WAVE_FORMAT_ALAW:
        return alaw_table[d[0]] / 32768.0f;
    case WAVE_FORMAT_MULAW:
        return ulaw_table[d[0]] / 32768.0f;
    }
    return 0;
}

void put32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back(x); v.push_back(x >> 8);
    v.push_back(x >> 16); v.push_back(x >> 24);
}
void put16(std::vector<uint8_t>& v, uint16_t x) {
    v.push_back(x); v.push_back(x >> 8);
}

}  // namespace

bool wav_read(const uint8_t* d, size_t n, std::vector<float>& pcm,
              int& sample_rate, int& channels) {
    if (!d || n < 12 || std::memcmp(d, "RIFF", 4) ||
        std::memcmp(d + 8, "WAVE", 4))
        return false;
    size_t off = 12;
    uint16_t fmt = 0;
    int bps = 0, ch = 0, sr = 0;
    const uint8_t* data = nullptr;
    size_t data_len = 0;
    bool have_fmt = false;
    while (off + 8 <= n) {
        uint32_t csz = rd32(d + off + 4);
        const uint8_t* body = d + off + 8;
        if (!std::memcmp(d + off, "fmt ", 4) && off + 8 + 16 <= n) {
            fmt = rd16(body);
            ch = rd16(body + 2);
            sr = rd32(body + 4);
            bps = rd16(body + 14);
            // The EXTENSIBLE extension reads valid_bits (body+18) and the
            // 16-byte SubFormat GUID (body+24..39). csz is the CLAIMED
            // size; require the bytes to actually be present or a truncated
            // file reads past the buffer.
            if (fmt == WAVE_FORMAT_EXTENSIBLE && csz >= 40 &&
                off + 8 + 40 <= n) {
                const uint8_t* sub = body + 24;  // SubFormat GUID
                if (!std::memcmp(sub, GUID_PCM, 16))
                    fmt = WAVE_FORMAT_PCM;
                else if (!std::memcmp(sub, GUID_FLOAT, 16))
                    fmt = WAVE_FORMAT_IEEE_FLOAT;
                else if (!std::memcmp(sub, GUID_ALAW, 16))
                    fmt = WAVE_FORMAT_ALAW;
                else if (!std::memcmp(sub, GUID_MULAW, 16))
                    fmt = WAVE_FORMAT_MULAW;
                int vbits = rd16(body + 18);
                if (vbits > 0) bps = vbits;
            }
            have_fmt = true;
        } else if (!std::memcmp(d + off, "data", 4)) {
            data = body;
            data_len = csz;
            if (off + 8 + data_len > n) data_len = n - (off + 8);
        }
        off += 8 + csz + (csz & 1);  // chunks are word-aligned
    }
    if (!have_fmt || !data || ch < 1 || sr < 1 || bps < 8) return false;
    int bytes = bps / 8;
    int stride = bytes * ch;
    if (stride <= 0) return false;
    long frames = static_cast<long>(data_len / stride);
    sample_rate = sr;
    channels = ch;
    pcm.assign(static_cast<size_t>(frames) * ch, 0.0f);
    for (long i = 0; i < frames; i++)
        for (int c = 0; c < ch; c++)
            pcm[i * ch + c] =
                sample_to_float(data + i * stride + c * bytes, fmt, bps);
    return true;
}

bool pcm_read(const uint8_t* d, size_t n, int sr, int ch, int bits,
              std::vector<float>& pcm) {
    if (!d || ch < 1 || sr < 1 || bits < 8) return false;
    int bytes = bits / 8, stride = bytes * ch;
    if (stride <= 0) return false;
    long frames = static_cast<long>(n / stride);
    pcm.assign(static_cast<size_t>(frames) * ch, 0.0f);
    for (long i = 0; i < frames; i++)
        for (int c = 0; c < ch; c++)
            pcm[i * ch + c] =
                sample_to_float(d + i * stride + c * bytes,
                                WAVE_FORMAT_PCM, bits);
    return true;
}

std::vector<uint8_t> wav_write(const float* pcm, long frames, int channels,
                               int sample_rate, int bits, bool is_float) {
    // Validate: integer 8/16/24/32, or float 32/64. Fall back to 16-bit.
    if (is_float) {
        if (bits != 32 && bits != 64) bits = 32;
    } else if (bits != 8 && bits != 16 && bits != 24 && bits != 32) {
        bits = 16;
    }
    const int bytes = bits / 8;
    const long total = frames * channels;
    const uint16_t tag = is_float ? WAVE_FORMAT_IEEE_FLOAT : WAVE_FORMAT_PCM;

    std::vector<uint8_t> v;
    uint32_t data_bytes = static_cast<uint32_t>(total) * bytes;
    v.insert(v.end(), {'R', 'I', 'F', 'F'});
    put32(v, 36 + data_bytes);
    v.insert(v.end(), {'W', 'A', 'V', 'E', 'f', 'm', 't', ' '});
    put32(v, 16);
    put16(v, tag);
    put16(v, static_cast<uint16_t>(channels));
    put32(v, static_cast<uint32_t>(sample_rate));
    put32(v, static_cast<uint32_t>(sample_rate) * channels * bytes);
    put16(v, static_cast<uint16_t>(channels * bytes));
    put16(v, static_cast<uint16_t>(bits));
    v.insert(v.end(), {'d', 'a', 't', 'a'});
    put32(v, data_bytes);

    for (long i = 0; i < total; i++) {
        float f = pcm[i];
        if (is_float) {
            if (bits == 32) {
                uint8_t b[4];
                std::memcpy(b, &f, 4);
                v.insert(v.end(), b, b + 4);
            } else {
                double g = f;
                uint8_t b[8];
                std::memcpy(b, &g, 8);
                v.insert(v.end(), b, b + 8);
            }
            continue;
        }
        // Clamp then round to the target integer depth.
        double s = f;
        if (s > 1.0) s = 1.0;
        if (s < -1.0) s = -1.0;
        switch (bits) {
        case 8: {
            int iv = static_cast<int>(s * 127.0 + (s >= 0 ? 0.5 : -0.5));
            if (iv > 127) iv = 127;
            if (iv < -128) iv = -128;
            v.push_back(static_cast<uint8_t>(iv + 128));  // unsigned
            break;
        }
        case 16: {
            int iv = static_cast<int>(s * 32767.0 + (s >= 0 ? 0.5 : -0.5));
            if (iv > 32767) iv = 32767;
            if (iv < -32768) iv = -32768;
            put16(v, static_cast<uint16_t>(static_cast<int16_t>(iv)));
            break;
        }
        case 24: {
            long iv = static_cast<long>(s * 8388607.0 + (s >= 0 ? 0.5 : -0.5));
            if (iv > 8388607) iv = 8388607;
            if (iv < -8388608) iv = -8388608;
            uint32_t u = static_cast<uint32_t>(iv);
            v.push_back(u & 0xFF);
            v.push_back((u >> 8) & 0xFF);
            v.push_back((u >> 16) & 0xFF);
            break;
        }
        case 32: {
            double sc = s * 2147483647.0 + (s >= 0 ? 0.5 : -0.5);
            long long iv = static_cast<long long>(sc);
            if (iv > 2147483647LL) iv = 2147483647LL;
            if (iv < -2147483648LL) iv = -2147483648LL;
            put32(v, static_cast<uint32_t>(static_cast<int32_t>(iv)));
            break;
        }
        }
    }
    return v;
}

}  // namespace glint
