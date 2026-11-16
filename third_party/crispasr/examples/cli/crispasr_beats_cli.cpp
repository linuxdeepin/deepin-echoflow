// crispasr_beats_cli.cpp — see crispasr_beats_cli.h.

#include "crispasr_beats_cli.h"

#include "common-crispasr.h" // read_audio_data
#include "crispasr_model_mgr_cli.h"
#include "whisper_params.h"

#include "beat_this.h"
#include "core/gguf_loader.h" // core_gguf::open_metadata / kv_str

#include <cstdio>
#include <string>
#include <vector>

namespace {

// Beat This! is trained at 22050 Hz; read_audio_data resamples for us.
constexpr int kBeatSampleRate = BEAT_THIS_SAMPLE_RATE;

// One line per beat: "time_sec<TAB>beat|downbeat". This is the .beats
// convention used by the MIR beat-tracking datasets (Ballroom, GTZAN-Rhythm),
// so the output drops straight into mir_eval.beat without reformatting.
//
// EVERY DOWNBEAT IS ALSO EMITTED AS A BEAT, which matches the model's own
// contract: the postprocessor snaps each downbeat onto its nearest beat, so a
// downbeat that is not a beat cannot occur and callers never have to merge two
// lists to get the full grid.
void print_events_text(const beat_this_event* ev, int n) {
    for (int i = 0; i < n; i++)
        printf("%.3f\t%s\n", ev[i].time_s, ev[i].is_downbeat ? "downbeat" : "beat");
}

void print_events_json(const beat_this_event* ev, int n, const std::string& fname) {
    printf("{\n");
    printf("  \"file\": \"%s\",\n", fname.c_str());
    printf("  \"n_beats\": %d,\n", n);
    printf("  \"tempo_bpm\": %.2f,\n", beat_this_tempo_bpm(ev, n));
    printf("  \"beats\": [\n");
    for (int i = 0; i < n; i++)
        printf("    {\"time\": %.3f, \"downbeat\": %s}%s\n", ev[i].time_s, ev[i].is_downbeat ? "true" : "false",
               i + 1 < n ? "," : "");
    printf("  ]\n}\n");
}

} // namespace

int crispasr_run_beats(const whisper_params& params) {
    if (params.fname_inp.empty()) {
        fprintf(stderr, "crispasr: --beats needs an input file (-f)\n");
        return 2;
    }

    const bool json = params.beats_format == "json";
    if (!json && !params.beats_format.empty() && params.beats_format != "text") {
        fprintf(stderr, "crispasr: --beats-format: unknown format '%s' (expected text or json)\n",
                params.beats_format.c_str());
        return 2;
    }

    const std::string backend_key = params.backend.empty() ? "beat-this" : params.backend;
    const std::string model = crispasr_resolve_model_cli(params.model, backend_key, params.no_prints, params.cache_dir,
                                                         params.auto_download, "", params.accept_license);
    if (model.empty()) {
        fprintf(stderr, "crispasr: --beats: could not resolve a model (try --auto-download).\n");
        return 2;
    }

    gguf_context* meta = core_gguf::open_metadata(model.c_str());
    if (!meta) {
        fprintf(stderr, "crispasr: --beats: cannot open '%s'\n", model.c_str());
        return 2;
    }
    const std::string arch = core_gguf::kv_str(meta, "general.architecture", "");
    core_gguf::free_metadata(meta);
    if (arch != "beat-this") {
        fprintf(stderr, "crispasr: --beats: '%s' is not a beat model (arch='%s'). Expected beat-this.\n", model.c_str(),
                arch.c_str());
        return 2;
    }

    beat_this_context* ctx = beat_this_init(model.c_str(), params.n_threads);
    if (!ctx) {
        fprintf(stderr, "crispasr: --beats: failed to load '%s'\n", model.c_str());
        return 2;
    }

    int rc = 0;
    for (const auto& fname : params.fname_inp) {
        std::vector<float> mono;
        std::vector<std::vector<float>> stereo;
        if (!read_audio_data(fname, mono, stereo, /*stereo=*/false, /*target_rate=*/kBeatSampleRate)) {
            fprintf(stderr, "crispasr: error: cannot read '%s'\n", fname.c_str());
            rc = 20;
            continue;
        }

        // One event per frame is the hard ceiling the peak-picker can produce,
        // so this can never truncate a real result.
        const int max_events = beat_this_n_frames((int)mono.size());
        std::vector<beat_this_event> ev((size_t)(max_events > 0 ? max_events : 1));
        const int n = beat_this_track(ctx, mono.data(), (int)mono.size(), ev.data(), (int)ev.size());
        if (n < 0) {
            fprintf(stderr, "crispasr: --beats failed on '%s'\n", fname.c_str());
            rc = 1;
            continue;
        }

        if (json) {
            print_events_json(ev.data(), n, fname);
        } else {
            if (!params.no_prints && params.fname_inp.size() > 1)
                printf("# %s\n", fname.c_str());
            print_events_text(ev.data(), n);
        }
        if (!params.no_prints) {
            int n_db = 0;
            for (int i = 0; i < n; i++)
                n_db += ev[(size_t)i].is_downbeat;
            fprintf(stderr, "crispasr: %s: %d beats (%d downbeats), %.1f BPM\n", fname.c_str(), n, n_db,
                    beat_this_tempo_bpm(ev.data(), n));
        }
    }

    beat_this_free(ctx);
    return rc;
}
