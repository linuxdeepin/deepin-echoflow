// crispasr_tab_cli.cpp — see crispasr_tab_cli.h.

#include "crispasr_tab_cli.h"

#include "common-crispasr.h" // read_audio_data
#include "crispasr_model_mgr_cli.h"
#include "whisper_params.h"

#include "core/gguf_loader.h" // core_gguf::open_metadata / kv_str
#include "tabcnn.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

// TabCNN is trained at 22.05 kHz; read_audio_data resamples for us. (The
// runtime would resample too, but decoding straight to the model's rate keeps
// one resampler out of the chain.)
constexpr int kTabSampleRate = TABCNN_SAMPLE_RATE;

// Argmax per string, for display only — see the warning in the header.
void argmax_row(const float* logp, int n_classes, int& best, float& best_logp) {
    best = 0;
    best_logp = logp[0];
    for (int c = 1; c < n_classes; c++) {
        if (logp[c] > best_logp) {
            best_logp = logp[c];
            best = c;
        }
    }
}

// One line per frame: "time_sec<TAB>f0 f1 f2 f3 f4 f5", low string first, with
// "-" for a string that is not played. Tab-separated and greppable, matching
// the shape of the --beats and --chords output.
void print_frames_text(const std::vector<float>& logp, int n_frames, int n_strings, int n_classes, int silent,
                       float period) {
    for (int t = 0; t < n_frames; t++) {
        printf("%.3f", (double)t * period);
        for (int s = 0; s < n_strings; s++) {
            int fret = 0;
            float lp = 0.0f;
            argmax_row(logp.data() + ((size_t)t * n_strings + s) * n_classes, n_classes, fret, lp);
            if (fret == silent)
                printf("\t-");
            else
                printf("\t%d", fret);
        }
        printf("\n");
    }
}

void print_frames_json(const std::vector<float>& logp, int n_frames, int n_strings, int n_classes, int silent,
                       float period, const std::string& fname) {
    printf("{\n");
    printf("  \"file\": \"%s\",\n", fname.c_str());
    printf("  \"frame_period_sec\": %.6f,\n", period);
    printf("  \"n_strings\": %d,\n", n_strings);
    printf("  \"n_classes\": %d,\n", n_classes);
    printf("  \"silent_class\": %d,\n", silent);
    printf("  \"frames\": [\n");
    for (int t = 0; t < n_frames; t++) {
        printf("    {\"time\": %.3f, \"frets\": [", (double)t * period);
        for (int s = 0; s < n_strings; s++) {
            int fret = 0;
            float lp = 0.0f;
            argmax_row(logp.data() + ((size_t)t * n_strings + s) * n_classes, n_classes, fret, lp);
            // Emit the confidence too: a decoder consumer wants to know a
            // displayed fret was a near-tie, and the text form cannot show it.
            printf("%s{\"fret\": %d, \"logp\": %.4f}", s ? ", " : "", fret == silent ? -1 : fret, lp);
        }
        printf("]}%s\n", t + 1 < n_frames ? "," : "");
    }
    printf("  ]\n}\n");
}

} // namespace

int crispasr_run_tab(const whisper_params& params) {
    if (params.fname_inp.empty()) {
        fprintf(stderr, "crispasr: --tab needs an input file (-f)\n");
        return 2;
    }

    const bool json = params.tab_format == "json";
    if (!json && !params.tab_format.empty() && params.tab_format != "text") {
        fprintf(stderr, "crispasr: --tab-format: unknown format '%s' (expected text or json)\n",
                params.tab_format.c_str());
        return 2;
    }

    const std::string backend_key = params.backend.empty() ? "tabcnn" : params.backend;
    const std::string model = crispasr_resolve_model_cli(params.model, backend_key, params.no_prints, params.cache_dir,
                                                         params.auto_download, "", params.accept_license);
    if (model.empty()) {
        fprintf(stderr, "crispasr: --tab: could not resolve a model.\n");
        return 2;
    }

    gguf_context* meta = core_gguf::open_metadata(model.c_str());
    if (!meta) {
        fprintf(stderr, "crispasr: --tab: cannot open '%s'\n", model.c_str());
        return 2;
    }
    const std::string arch = core_gguf::kv_str(meta, "general.architecture", "");
    core_gguf::free_metadata(meta);
    if (arch != "tabcnn") {
        fprintf(stderr, "crispasr: --tab: '%s' is not a tablature model (arch='%s'). Expected tabcnn.\n", model.c_str(),
                arch.c_str());
        return 2;
    }

    tabcnn_context* ctx = tabcnn_init(model.c_str(), params.n_threads);
    if (!ctx) {
        fprintf(stderr, "crispasr: --tab: failed to load '%s'\n", model.c_str());
        return 2;
    }
    const int silent = tabcnn_silent_class(ctx);
    const float period = tabcnn_frame_period(ctx);

    int rc = 0;
    for (const auto& fname : params.fname_inp) {
        std::vector<float> mono;
        std::vector<std::vector<float>> stereo;
        if (!read_audio_data(fname, mono, stereo, /*stereo=*/false, /*target_rate=*/kTabSampleRate)) {
            fprintf(stderr, "crispasr: error: cannot read '%s'\n", fname.c_str());
            rc = 20;
            continue;
        }

        const int n_frames = tabcnn_n_frames(ctx, (int)mono.size(), kTabSampleRate);
        if (n_frames <= 0) {
            fprintf(stderr, "crispasr: --tab: '%s' is too short\n", fname.c_str());
            rc = 1;
            continue;
        }
        std::vector<float> logp((size_t)n_frames * TABCNN_NUM_STRINGS * TABCNN_NUM_CLASSES);
        const int got = tabcnn_compute(ctx, mono.data(), (int)mono.size(), kTabSampleRate, logp.data(), n_frames);
        if (got <= 0) {
            fprintf(stderr, "crispasr: --tab failed on '%s'\n", fname.c_str());
            rc = 1;
            continue;
        }

        if (json) {
            print_frames_json(logp, got, TABCNN_NUM_STRINGS, TABCNN_NUM_CLASSES, silent, period, fname);
        } else {
            if (!params.no_prints && params.fname_inp.size() > 1)
                printf("# %s\n", fname.c_str());
            print_frames_text(logp, got, TABCNN_NUM_STRINGS, TABCNN_NUM_CLASSES, silent, period);
        }
        if (!params.no_prints) {
            int active = 0;
            for (int t = 0; t < got; t++)
                for (int s = 0; s < TABCNN_NUM_STRINGS; s++) {
                    int fret = 0;
                    float lp = 0.0f;
                    argmax_row(logp.data() + ((size_t)t * TABCNN_NUM_STRINGS + s) * TABCNN_NUM_CLASSES,
                               TABCNN_NUM_CLASSES, fret, lp);
                    active += (fret != silent);
                }
            fprintf(stderr,
                    "crispasr: %s: %d frames (%.3f s hop), %d fretted string-frames\n"
                    "  note: displayed frets are a plain argmax — no playability constraints are\n"
                    "  applied. Consume the emission scores via the C ABI and run your own decoder.\n",
                    fname.c_str(), got, period, active);
        }
    }

    tabcnn_free(ctx);
    return rc;
}
