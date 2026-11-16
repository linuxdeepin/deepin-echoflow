// test_beat_this_track.cpp — windowing + postprocessing parity (§251b-1).
//
//   ./test-beat-this-track <gguf> <track_logmel.bin> <ref_logits.bin> <out_prefix>
//
// Writes three artefacts for tools/cmp_beat_this_track.py:
//   <p>logits.bin   beat then downbeat logits, T floats each, from OUR chunking
//   <p>postp.bin    events peak-picked from the REFERENCE logits
//   <p>events.bin   events from our own logits (end-to-end)
//
// The middle one is the point of this test. Feeding the reference's own logits
// through our peak-picker isolates the postprocessing completely: if it and the
// logits both match, an end-to-end difference can only come from the front end,
// and if postp matches while events do not, the difference is numerical rather
// than algorithmic. Testing only end-to-end conflates all three.
//
// Event records are (float time_s, int32 is_downbeat) pairs after an int32
// count, matching what the comparison script unpacks.
#include "beat_this.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static std::vector<float> read_f32(const char* p) {
    std::vector<float> v;
    FILE* f = fopen(p, "rb");
    if (!f)
        return v;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    v.resize((size_t)(n / 4));
    if (fread(v.data(), 4, v.size(), f) != v.size())
        v.clear();
    fclose(f);
    return v;
}

static bool write_events(const std::string& path, const std::vector<beat_this_event>& ev, int n) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f)
        return false;
    const int32_t cnt = n;
    fwrite(&cnt, sizeof(int32_t), 1, f);
    for (int i = 0; i < n; i++) {
        fwrite(&ev[(size_t)i].time_s, sizeof(float), 1, f);
        const int32_t d = ev[(size_t)i].is_downbeat;
        fwrite(&d, sizeof(int32_t), 1, f);
    }
    fclose(f);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <gguf> <track_logmel.bin> <ref_logits.bin> <out_prefix>\n", argv[0]);
        return 2;
    }
    auto lm = read_f32(argv[2]);
    auto reflog = read_f32(argv[3]);
    if (lm.empty() || reflog.empty()) {
        fprintf(stderr, "cannot read inputs\n");
        return 1;
    }
    const int T = (int)(lm.size() / BEAT_THIS_MEL_BINS);
    if ((int)reflog.size() != 2 * T) {
        fprintf(stderr, "ref logits have %d floats, expected 2*%d\n", (int)reflog.size(), T);
        return 1;
    }

    beat_this_context* ctx = beat_this_init(argv[1], 4);
    if (!ctx) {
        fprintf(stderr, "init failed\n");
        return 1;
    }
    const std::string p = argv[4];

    std::vector<float> lb((size_t)T), ld((size_t)T);
    if (beat_this_logits(ctx, lm.data(), T, lb.data(), ld.data()) != T) {
        fprintf(stderr, "logits failed\n");
        beat_this_free(ctx);
        return 1;
    }
    FILE* f = fopen((p + "logits.bin").c_str(), "wb");
    if (f) {
        fwrite(lb.data(), sizeof(float), (size_t)T, f);
        fwrite(ld.data(), sizeof(float), (size_t)T, f);
        fclose(f);
    }

    const int max_ev = T; // at most one beat per frame
    std::vector<beat_this_event> ev((size_t)max_ev);

    const int n_ref = beat_this_events_from_logits(reflog.data(), reflog.data() + T, T, ev.data(), max_ev);
    if (n_ref < 0) {
        beat_this_free(ctx);
        return 1;
    }
    write_events(p + "postp.bin", ev, n_ref);

    const int n_own = beat_this_events_from_logits(lb.data(), ld.data(), T, ev.data(), max_ev);
    write_events(p + "events.bin", ev, n_own);

    printf("T=%d  events from ref logits: %d   from our logits: %d\n", T, n_ref, n_own);
    printf("tempo (our logits): %.2f BPM\n", beat_this_tempo_bpm(ev.data(), n_own));
    beat_this_free(ctx);
    return 0;
}
