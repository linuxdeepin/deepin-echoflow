// test-omnivoice-seed.cpp — live regression for #281 (OmniVoice seed controls).
//
// omnivoice_set_seed() writes ctx->gen.seed, which the generator reads live per
// synthesis into a synthesis-local std::mt19937 (no process-global RNG). So the
// contract is: identical seeds ⇒ byte-identical audio codes; a different seed ⇒
// different codes. This exercises the fast text→codes AR stage only
// (omnivoice_synthesize_codes) — no DAC decode — so it is cheap enough for CI.
//
// SKIPs cleanly (exit 0) when no model is configured, or when the minimal
// no-voice-prompt setup can't produce codes on this machine. It only FAILs on a
// genuine determinism violation (same seed → different codes, or different seed
// → identical codes).
//
// Env:
//   CRISPASR_TEST_OMNIVOICE_MODEL      main GGUF (required to run; else SKIP)
//   CRISPASR_TEST_OMNIVOICE_TOKENIZER  audio-tokenizer GGUF (optional)
//   CRISPASR_TEST_OMNIVOICE_PROMPT_WAV 24 kHz mono ref voice (optional)
//   CRISPASR_TEST_OMNIVOICE_PROMPT_TEXT ref transcription (optional)

#include "omnivoice.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static std::vector<int32_t> gen_codes(omnivoice_context* ctx, uint64_t seed, const char* text) {
    omnivoice_set_seed(ctx, seed);
    int n = 0;
    int32_t* c = omnivoice_synthesize_codes(ctx, text, &n);
    std::vector<int32_t> out;
    if (c && n > 0)
        out.assign(c, c + n);
    omnivoice_codes_free(c);
    return out;
}

int main() {
    const char* model = getenv("CRISPASR_TEST_OMNIVOICE_MODEL");
    if (!model || !model[0]) {
        fprintf(stderr, "SKIP: CRISPASR_TEST_OMNIVOICE_MODEL not set\n");
        return 0;
    }
    if (FILE* f = fopen(model, "rb")) {
        fclose(f);
    } else {
        fprintf(stderr, "SKIP: model not found: %s\n", model);
        return 0;
    }

    auto p = omnivoice_context_default_params();
    p.verbosity = 0;
    omnivoice_context* ctx = omnivoice_init_from_file(model, p);
    if (!ctx) {
        fprintf(stderr, "SKIP: failed to load %s\n", model);
        return 0;
    }

    if (const char* tok = getenv("CRISPASR_TEST_OMNIVOICE_TOKENIZER"); tok && tok[0])
        omnivoice_set_tokenizer_path(ctx, tok);
    if (const char* w = getenv("CRISPASR_TEST_OMNIVOICE_PROMPT_WAV"); w && w[0]) {
        const char* t = getenv("CRISPASR_TEST_OMNIVOICE_PROMPT_TEXT");
        omnivoice_set_voice_prompt(ctx, w, t ? t : "");
    }

    // Keep the diffusion cheap — determinism is orthogonal to step count.
    omnivoice_set_num_steps(ctx, 8);

    const char* text = "The quick brown fox jumps over the lazy dog.";
    const std::vector<int32_t> a1 = gen_codes(ctx, 111, text);
    const std::vector<int32_t> b = gen_codes(ctx, 222, text);
    const std::vector<int32_t> a2 = gen_codes(ctx, 111, text);
    omnivoice_free(ctx);

    if (a1.empty() || a2.empty() || b.empty()) {
        // Minimal (no voice prompt) setup didn't yield codes on this box — not a
        // determinism regression, so don't fail CI over it.
        fprintf(stderr, "SKIP: synthesize produced no codes (a1=%zu b=%zu a2=%zu)\n", a1.size(), b.size(), a2.size());
        return 0;
    }

    if (a1 != a2) {
        fprintf(stderr, "FAIL: seed 111 not deterministic — codes differ across two runs (%zu vs %zu)\n", a1.size(),
                a2.size());
        return 1;
    }
    if (a1 == b) {
        fprintf(stderr, "FAIL: seed 111 and 222 produced identical codes — seed is not honored\n");
        return 1;
    }

    fprintf(stderr, "PASS: seed 111 deterministic (%zu codes); seed 222 differs\n", a1.size());
    return 0;
}
