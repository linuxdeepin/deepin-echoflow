// crispasr_chords_cli.h — CLI dispatcher for the `--chords` task.
//
// Chord recognition is its own task: audio in, a chord timeline out — not
// crispasr_segments — so per docs/source-separation-surface.md it gets its own
// early dispatcher rather than being layered onto transcribe(), mirroring
// --pitch and --separate. This resolves the model, detects its architecture
// from the GGUF ("btc"), decodes the audio and prints one span per line
// (or JSON).

#pragma once

struct whisper_params;

// Run the --chords task for every input file in `params.fname_inp`. Returns a
// process exit code (0 = success). Called from crispasr_run_backend() before
// any transcribe backend is constructed.
int crispasr_run_chords(const whisper_params& params);
