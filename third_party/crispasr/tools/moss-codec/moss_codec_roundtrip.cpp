// Acceptance test for moss_tts_local_codec::encode() — HARD RULE 3: the decoded
// output is the only acceptance test, so this reconstructs REAL speech and hands
// it back for ASR rather than scoring codes against codes.
//
// Two modes:
//   <codec.gguf> --codes            cheap smoke test: random codes -> decode ->
//                                   encode, report per-quantizer agreement.
//                                   Above chance (0.1%) means the encoder
//                                   inverts at all; it is NOT acceptance, since
//                                   random codes decode to audio off the
//                                   manifold the encoder was trained on and RVQ
//                                   is lossy in any case.
//   <codec.gguf> <in.wav> <out.wav> the real one: speech -> encode -> decode ->
//                                   wav, for the caller to ASR.
#include "moss_tts_local_codec.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
using namespace moss_tts_local_codec;

extern "C" int crispasr_audio_load(const char*, float**, int*, int*);
extern "C" void crispasr_audio_free(float*);

static bool write_wav(const char* path, const std::vector<float>& mono, int sr) {
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    const uint32_t n = (uint32_t)mono.size();
    const uint32_t data_bytes = n * 2, chunk = 36 + data_bytes;
    const uint16_t ch = 1, bits = 16;
    const uint32_t byte_rate = (uint32_t)sr * ch * bits / 8;
    const uint16_t align = ch * bits / 8, fmt = 1, pcm_hdr = 16;
    std::fwrite("RIFF", 1, 4, f); std::fwrite(&chunk, 4, 1, f); std::fwrite("WAVE", 1, 4, f);
    std::fwrite("fmt ", 1, 4, f); std::fwrite(&pcm_hdr, 4, 1, f); std::fwrite(&fmt, 2, 1, f);
    std::fwrite(&ch, 2, 1, f); std::fwrite(&sr, 4, 1, f); std::fwrite(&byte_rate, 4, 1, f);
    std::fwrite(&align, 2, 1, f); std::fwrite(&bits, 2, 1, f);
    std::fwrite("data", 1, 4, f); std::fwrite(&data_bytes, 4, 1, f);
    for (float v : mono) {
        if (v > 1.f) v = 1.f; if (v < -1.f) v = -1.f;
        int16_t s = (int16_t)(v * 32767.f);
        std::fwrite(&s, 2, 1, f);
    }
    std::fclose(f);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <codec.gguf> (--codes | <in.wav> <out.wav>)\n", argv[0]); return 2; }
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_t bes[1] = {be};
    ggml_backend_sched_t sched = ggml_backend_sched_new(bes, nullptr, 1, 8192, false, false);
    Codec* c = load(argv[1], be, sched, 1);
    if (!c) { std::fprintf(stderr, "load failed\n"); return 1; }
    std::printf("encoder_ready = %s\n", encoder_ready(c) ? "true" : "false");
    if (!encoder_ready(c)) return 1;
    const int NVQ = 12;

    if (std::string(argv[2]) == "--codes") {
        const int T = 24;
        std::mt19937 rng(1234);
        std::vector<int32_t> codes((size_t)NVQ * T);
        for (auto& v : codes) v = (int32_t)(rng() % 1024);
        std::vector<float> wav = decode(c, codes.data(), NVQ, T);
        int nvq_out = 0, t_out = 0;
        std::vector<int32_t> rec = encode(c, wav.data(), (int64_t)wav.size(), nvq_out, t_out);
        if (rec.empty()) return 1;
        const int t_cmp = t_out < T ? t_out : T;
        for (int q = 0; q < nvq_out; q++) {
            int a = 0;
            for (int t = 0; t < t_cmp; t++)
                if (rec[(size_t)q * t_out + t] == codes[(size_t)q * T + t]) a++;
            std::printf("  q%-2d agreement %3d/%-3d = %5.1f%%\n", q, a, t_cmp, 100.0 * a / t_cmp);
        }
        std::printf("(chance %.2f%%)\n", 100.0 / 1024.0);
        free(c);
        return 0;
    }

    // Real speech round-trip. crispasr_audio_load yields 16 kHz mono; the codec
    // wants 48 kHz channel-interleaved, so upsample 3x (linear) and duplicate to
    // stereo. Resampling quality is not the thing under test — intelligibility
    // after encode->decode is.
    float* pcm = nullptr; int n = 0, sr = 0;
    if (crispasr_audio_load(argv[2], &pcm, &n, &sr) != 0 || n <= 0) {
        std::fprintf(stderr, "failed to load %s\n", argv[2]); return 1;
    }
    std::printf("in: %d samples @ %d Hz\n", n, sr);
    std::vector<float> inter((size_t)n * 3 * 2);
    for (int i = 0; i < n * 3; i++) {
        const float pos = (float)i / 3.0f;
        const int i0 = (int)pos, i1 = i0 + 1 < n ? i0 + 1 : n - 1;
        const float fr = pos - (float)i0;
        const float v = pcm[i0] * (1.f - fr) + pcm[i1] * fr;
        inter[(size_t)i * 2 + 0] = v;
        inter[(size_t)i * 2 + 1] = v;
    }
    crispasr_audio_free(pcm);

    int nvq_out = 0, t_out = 0;
    std::vector<int32_t> codes = encode(c, inter.data(), (int64_t)inter.size(), nvq_out, t_out);
    std::printf("encode -> n_vq=%d t_audio=%d\n", nvq_out, t_out);
    if (codes.empty()) return 1;
    std::vector<float> rec = decode(c, codes.data(), nvq_out, t_out);
    std::printf("decode -> %zu interleaved samples\n", rec.size());
    if (rec.empty()) return 1;

    // De-interleave to mono and drop back to 16 kHz for ASR.
    std::vector<float> mono16;
    mono16.reserve(rec.size() / 6 + 1);
    for (size_t i = 0; i + 5 < rec.size(); i += 6)
        mono16.push_back(0.5f * (rec[i] + rec[i + 1]));
    if (!write_wav(argv[3], mono16, 16000)) { std::fprintf(stderr, "write failed\n"); return 1; }
    std::printf("wrote %s (%zu samples @ 16000 Hz)\n", argv[3], mono16.size());
    free(c);
    return 0;
}
