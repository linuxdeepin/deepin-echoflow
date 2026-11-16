// moss_tts_local_smoke.cpp — P5-lite smoke test for the 4B MossTTSLocal runtime.
//
// Loads the moss-tts-local GGUF, runs generate_codes on a short text, and prints
// the (n_vq, T) code grid shape + a few frames. Confirms the runtime actually
// RUNS end-to-end on the real 4B weights (backbone + local/depth transformer +
// depth-first generate loop) without crashing — before the codec-v2 (P3) and the
// full integration (P4). Not a parity test; just "does it run + produce a plausible
// grid". Run on Kaggle (4B F16 ~9 GB fits a P100).
//
// Usage: moss-tts-local-smoke <model.gguf> ["text"] [max_frames]

#include "moss_tts_local.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "core/crispasr_env.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [text] [max_frames]\n", argv[0]);
        return 2;
    }
    const char* path = argv[1];
    const char* text = argc >= 3 ? argv[2] : "The quick brown fox jumps over the lazy dog.";
    const int max_frames = argc >= 4 ? atoi(argv[3]) : 40;

    moss_tts_local_context_params cp = moss_tts_local_context_default_params();
    const char* no_gpu = crispasr_env::get("CRISPASR_MOSS_TTS_LOCAL_NO_GPU");
    if (no_gpu && *no_gpu && *no_gpu != '0')
        cp.use_gpu = false;

    moss_tts_local_context* ctx = moss_tts_local_init_from_file(path, cp);
    if (!ctx) {
        fprintf(stderr, "FAIL: init_from_file returned null\n");
        return 1;
    }
    printf("loaded: n_vq=%d hidden=%d audio_vocab=%d sr=%d\n", moss_tts_local_n_vq(ctx),
           moss_tts_local_hidden_size(ctx), moss_tts_local_audio_vocab_size(ctx), moss_tts_local_sampling_rate(ctx));

    moss_tts_local_synth_params sp = moss_tts_local_synth_default_params();
    sp.max_new_frames = max_frames;
    sp.max_audio_frames = max_frames;
    sp.text_temperature = 0.0f;  // greedy stop head
    sp.audio_temperature = 0.0f; // greedy codes — deterministic smoke
    sp.seed = 1;

    int n_vq = 0, t_audio = 0;
    int32_t* codes = moss_tts_local_generate_codes(ctx, text, &sp, &n_vq, &t_audio);
    if (!codes || n_vq <= 0 || t_audio <= 0) {
        fprintf(stderr, "FAIL: generate_codes empty (n_vq=%d T=%d)\n", n_vq, t_audio);
        free(codes);
        moss_tts_local_free(ctx);
        return 1;
    }
    printf("generated grid: n_vq=%d T=%d\n", n_vq, t_audio);

    // sanity: all codes in [0, audio_vocab_size)
    const int av = moss_tts_local_audio_vocab_size(ctx);
    int bad = 0, mn = av, mx = -1;
    for (int i = 0; i < n_vq * t_audio; i++) {
        const int c = codes[i];
        if (c < 0 || c >= av)
            bad++;
        if (c < mn)
            mn = c;
        if (c > mx)
            mx = c;
    }
    printf("code range [%d, %d], out-of-range=%d\n", mn, mx, bad);
    const int show = t_audio < 4 ? t_audio : 4;
    for (int t = 0; t < show; t++) {
        printf("  frame %d:", t);
        for (int k = 0; k < n_vq; k++)
            printf(" %d", codes[(size_t)k * t_audio + t]);
        printf("\n");
    }
    free(codes);
    moss_tts_local_free(ctx);
    printf("%s\n", bad == 0 ? "SMOKE PASS" : "SMOKE FAIL (out-of-range codes)");
    return bad == 0 ? 0 : 1;
}
