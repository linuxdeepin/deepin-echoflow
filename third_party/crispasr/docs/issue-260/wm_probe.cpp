// wm_probe.cpp — isolate the CrispASR spread-spectrum watermark (#260).
// Feeds a clean synthetic-speech signal through the exact watermark header
// and writes clean / alpha=0.005 / alpha=0.08 WAVs plus the injected delta.
#include "core/crispasr_watermark.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

static void write_wav(const char* path, const std::vector<float>& x, int sr) {
    std::ofstream f(path, std::ios::binary);
    uint32_t n = (uint32_t)x.size();
    uint32_t data_bytes = n * 2;
    uint32_t chunk = 36 + data_bytes;
    uint16_t fmt = 1, ch = 1, bits = 16;
    uint32_t byte_rate = sr * ch * bits / 8;
    uint16_t block = ch * bits / 8;
    f.write("RIFF", 4); f.write((char*)&chunk, 4); f.write("WAVE", 4);
    f.write("fmt ", 4); uint32_t sub1 = 16; f.write((char*)&sub1, 4);
    f.write((char*)&fmt, 2); f.write((char*)&ch, 2); f.write((char*)&sr, 4);
    f.write((char*)&byte_rate, 4); f.write((char*)&block, 2); f.write((char*)&bits, 2);
    f.write("data", 4); f.write((char*)&data_bytes, 4);
    for (float v : x) { int s = (int)std::lround(v * 32767.0f); if (s > 32767) s = 32767; if (s < -32768) s = -32768; int16_t o = (int16_t)s; f.write((char*)&o, 2); }
}

int main() {
    const int sr = 24000;      // qwen3-tts output rate
    const int N = sr * 3;      // 3 seconds
    std::vector<float> clean(N);
    // Synthetic voiced speech: 130 Hz fundamental + harmonics rolling off,
    // amplitude-modulated at 4 Hz (syllable rate). Band-limited ~<4 kHz like
    // real speech, so anything the watermark injects above that is obvious.
    for (int i = 0; i < N; i++) {
        double t = (double)i / sr;
        double env = 0.4 * (1.0 + 0.9 * std::sin(2 * M_PI * 4.0 * t));
        double s = 0.0;
        for (int h = 1; h <= 25; h++) {
            double f = 130.0 * h;
            if (f > 4000.0) break;               // speech energy dies off by ~4 kHz
            s += (1.0 / h) * std::sin(2 * M_PI * f * t);
        }
        clean[i] = (float)(env * s * 0.15);
    }

    auto wm005 = clean;
    auto wm08 = clean;
    crispasr_watermark_embed_impl(wm005.data(), N, 0.005f);  // legacy
    crispasr_watermark_embed_impl(wm08.data(), N, 0.08f);    // shipped default

    write_wav("clean.wav", clean, sr);
    write_wav("wm005.wav", wm005, sr);
    write_wav("wm08.wav", wm08, sr);

    std::vector<float> delta(N);
    for (int i = 0; i < N; i++) delta[i] = (wm08[i] - clean[i]) * 8.0f; // amplified for audibility
    write_wav("delta08_x8.wav", delta, sr);

    // Report: which frequencies are being nudged (the "comb"), and SNR.
    auto bins = crispasr_wm::generate_bin_pattern(CRISPASR_WATERMARK_KEY, 1024, CRISPASR_WATERMARK_NBINS);
    printf("Watermark modulates %d FIXED bins (n_fft=1024, sr=%d):\n", (int)bins.size(), sr);
    printf("  bin  freq_Hz  sign\n");
    for (auto& b : bins) printf("  %4d  %7.0f   %+d\n", b.index, (double)b.index * sr / 1024.0, b.sign);

    auto rms = [&](const std::vector<float>& a, const std::vector<float>& b) {
        double sig = 0, noi = 0;
        for (int i = 0; i < N; i++) { sig += (double)b[i] * b[i]; noi += (double)(a[i] - b[i]) * (a[i] - b[i]); }
        double snr = 10.0 * std::log10(sig / (noi + 1e-30));
        return snr;
    };
    printf("\nSNR clean vs wm alpha=0.005 : %6.2f dB\n", rms(wm005, clean));
    printf("SNR clean vs wm alpha=0.08  : %6.2f dB\n", rms(wm08, clean));
    return 0;
}
