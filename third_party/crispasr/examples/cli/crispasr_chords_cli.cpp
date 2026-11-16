// crispasr_chords_cli.cpp — see crispasr_chords_cli.h.

#include "crispasr_chords_cli.h"

#include "common-crispasr.h" // read_audio_data
#include "crispasr_model_mgr_cli.h"
#include "whisper_params.h"

#include "btc_chords.h"
#include "core/gguf_loader.h" // core_gguf::open_metadata / kv_str

#include <cstdio>
#include <string>
#include <vector>

namespace {

// BTC is trained at 22050 Hz; read_audio_data resamples for us.
constexpr int kBtcSampleRate = 22050;

const char* span_name(btc_chords_context* ctx, const btc_chord_span& s) {
    return btc_chords_label_name(ctx, s.label);
}

// One line per span, greppable and lab-file shaped:
// "start_sec end_sec chord" matches the .lab convention the chord datasets
// (Isophonics et al.) use, so the output drops straight into mir_eval.
void print_spans_text(btc_chords_context* ctx, const btc_chords_result* r) {
    for (int i = 0; i < r->n_spans; i++) {
        const btc_chord_span& s = r->spans[i];
        printf("%.3f\t%.3f\t%s\n", s.start_ms / 1000.0, s.end_ms / 1000.0, span_name(ctx, s));
    }
}

void print_spans_json(btc_chords_context* ctx, const btc_chords_result* r, const std::string& fname) {
    printf("{\n");
    printf("  \"file\": \"%s\",\n", fname.c_str());
    printf("  \"vocabulary\": %d,\n", r->n_chords);
    printf("  \"n_spans\": %d,\n", r->n_spans);
    printf("  \"chords\": [\n");
    for (int i = 0; i < r->n_spans; i++) {
        const btc_chord_span& s = r->spans[i];
        printf("    {\"start_ms\": %.1f, \"end_ms\": %.1f, \"chord\": \"%s\", \"confidence\": %.4f}%s\n", s.start_ms,
               s.end_ms, span_name(ctx, s), s.confidence, i + 1 < r->n_spans ? "," : "");
    }
    printf("  ]\n}\n");
}

} // namespace

int crispasr_run_chords(const whisper_params& params) {
    if (params.fname_inp.empty()) {
        fprintf(stderr, "crispasr: --chords needs an input file (-f)\n");
        return 2;
    }

    const bool json = params.chords_format == "json";
    if (!json && !params.chords_format.empty() && params.chords_format != "text") {
        fprintf(stderr, "crispasr: --chords-format: unknown format '%s' (expected text or json)\n",
                params.chords_format.c_str());
        return 2;
    }

    // Defaults to the `btc-chords` registry entry, which is the 170-class
    // model: it can be reduced to maj/min with CRISPASR_BTC_MAJ_MIN=1, whereas
    // a 25-class model could never be expanded.
    const std::string backend_key = params.backend.empty() ? "btc-chords" : params.backend;
    const std::string model = crispasr_resolve_model_cli(params.model, backend_key, params.no_prints, params.cache_dir,
                                                         params.auto_download, "", params.accept_license);
    if (model.empty()) {
        fprintf(stderr, "crispasr: --chords: could not resolve a model.\n"
                        "  The BTC weights are CC-BY-NC-SA (non-commercial); pass\n"
                        "  --accept-license cc-by-nc-sa-4.0 along with --auto-download.\n");
        return 2;
    }

    gguf_context* meta = core_gguf::open_metadata(model.c_str());
    if (!meta) {
        fprintf(stderr, "crispasr: --chords: cannot open '%s'\n", model.c_str());
        return 2;
    }
    const std::string arch = core_gguf::kv_str(meta, "general.architecture", "");
    core_gguf::free_metadata(meta);
    if (arch != "btc") {
        fprintf(stderr, "crispasr: --chords: '%s' is not a chord model (arch='%s'). Expected btc.\n", model.c_str(),
                arch.c_str());
        return 2;
    }

    btc_chords_params bp = btc_chords_default_params();
    bp.n_threads = params.n_threads;
    bp.use_gpu = params.use_gpu;
    bp.gpu_device = params.gpu_device;
    btc_chords_context* ctx = btc_chords_init_from_file(model.c_str(), bp);
    if (!ctx) {
        fprintf(stderr, "crispasr: --chords: failed to load '%s'\n", model.c_str());
        return 2;
    }

    int rc = 0;
    for (const auto& fname : params.fname_inp) {
        std::vector<float> mono;
        std::vector<std::vector<float>> stereo;
        if (!read_audio_data(fname, mono, stereo, /*stereo=*/false, /*target_rate=*/kBtcSampleRate)) {
            fprintf(stderr, "crispasr: error: cannot read '%s'\n", fname.c_str());
            rc = 20;
            continue;
        }

        btc_chords_result* r = btc_chords_recognize(ctx, mono.data(), (int)mono.size(), kBtcSampleRate);
        if (!r) {
            fprintf(stderr, "crispasr: --chords failed on '%s'\n", fname.c_str());
            rc = 1;
            continue;
        }

        if (json) {
            print_spans_json(ctx, r, fname);
        } else {
            if (!params.no_prints && params.fname_inp.size() > 1)
                printf("# %s\n", fname.c_str());
            print_spans_text(ctx, r);
        }
        if (!params.no_prints)
            fprintf(stderr, "crispasr: %s: %d chord spans (%d-class vocabulary)\n", fname.c_str(), r->n_spans,
                    r->n_chords);
        btc_chords_result_free(r);
    }

    btc_chords_free(ctx);
    return rc;
}
