// crispasr_tab_cli.h — CLI dispatcher for the `--tab` task.
//
// Guitar tablature is its own task: audio in, a per-frame per-string grid of
// fret SCORES out — not crispasr_segments — so per
// docs/source-separation-surface.md it gets its own early dispatcher rather
// than being layered onto transcribe(), mirroring --pitch, --chords and
// --beats. This resolves the model, detects its architecture from the GGUF
// ("tabcnn"), decodes the audio and prints one line per frame (or JSON).
//
// ⚠️ The text/JSON output argmaxes the emission grid purely so the CLI has
// something to show. That is NOT the intended production path: it ignores every
// playability constraint (one note per string, fret range, capo, hand span).
// The real consumer takes the log-probabilities through the C ABI and runs its
// own constrained Viterbi/DP.

#pragma once

struct whisper_params;

// Run the --tab task for every input file in `params.fname_inp`. Returns a
// process exit code (0 = success). Called from crispasr_run_backend() before
// any transcribe backend is constructed.
int crispasr_run_tab(const whisper_params& params);
