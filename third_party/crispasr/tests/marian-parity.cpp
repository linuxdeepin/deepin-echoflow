// marian-parity.cpp — in-process driver behind tools/marian_parity.py.
//
// Loads a translation GGUF once (Marian / Opus-MT, or M2M-100 for the speed
// comparison) and, for every line of a UTF-8 text file, prints
//
//     <index> TAB <encoder ids, space separated> TAB <median ms> TAB <translation>
//
// so a script can hold the ids and the text against the Hugging Face
// reference, and so timings are warm (model loaded, one untimed call first)
// rather than one process start per sentence.
//
//   marian-parity <model.gguf> <sentences.txt> [--src de] [--tgt en]
//                 [--beam 1] [--reps 3] [--tok-only]
//
// There is no thread option: the m2m100 runtime does not forward n_threads to
// the ggml CPU backend, so every run uses ggml's default (4).
//
// Not a test: it needs a model. Built on demand (`--target marian-parity`).

#include "m2m100.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s <model.gguf> <sentences.txt> [--src de] [--tgt en] [--beam 1] [--reps 3] [--tok-only]\n",
                argv[0]);
        return 2;
    }
    std::string src = "de", tgt = "en";
    int beam = 1, reps = 3;
    bool tok_only = false;
    for (int i = 3; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--src")
            src = next();
        else if (a == "--tgt")
            tgt = next();
        else if (a == "--beam")
            beam = std::atoi(next());
        else if (a == "--reps")
            reps = std::max(1, std::atoi(next()));
        else if (a == "--tok-only")
            tok_only = true;
        else {
            fprintf(stderr, "unknown argument '%s'\n", a.c_str());
            return 2;
        }
    }

    std::vector<std::string> lines;
    {
        std::ifstream in(argv[2]);
        if (!in) {
            fprintf(stderr, "cannot read '%s'\n", argv[2]);
            return 2;
        }
        std::string line;
        while (std::getline(in, line))
            lines.push_back(line);
    }

    m2m100_context_params p = m2m100_context_default_params();
    p.verbosity = 0;
    m2m100_context* ctx = m2m100_init_from_file(argv[1], p);
    if (!ctx) {
        fprintf(stderr, "failed to load '%s'\n", argv[1]);
        return 1;
    }
    m2m100_set_beam_size(ctx, beam);

    if (!tok_only && !lines.empty()) {
        // warm-up, untimed
        char* w = m2m100_translate(ctx, lines[0].c_str(), src.c_str(), tgt.c_str(), 0);
        free(w);
    }

    std::vector<int32_t> ids(4096);
    int rc = 0;
    for (size_t i = 0; i < lines.size(); i++) {
        const int n = m2m100_tokenize(ctx, lines[i].c_str(), src.c_str(), tgt.c_str(), ids.data(), (int)ids.size());
        printf("%zu\t", i);
        for (int k = 0; k < n && k < (int)ids.size(); k++)
            printf(k ? " %d" : "%d", ids[k]);
        if (tok_only) {
            printf("\t0\t\n");
            continue;
        }
        std::vector<double> ms;
        std::string text;
        for (int r = 0; r < reps; r++) {
            const auto t0 = std::chrono::steady_clock::now();
            char* out = m2m100_translate(ctx, lines[i].c_str(), src.c_str(), tgt.c_str(), 0);
            const auto t1 = std::chrono::steady_clock::now();
            ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            if (!out) {
                rc = 1;
                text = "<FAILED>";
            } else {
                text = out;
            }
            free(out);
        }
        std::sort(ms.begin(), ms.end());
        printf("\t%.2f\t%s\n", ms[ms.size() / 2], text.c_str());
    }
    m2m100_free(ctx);
    return rc;
}
