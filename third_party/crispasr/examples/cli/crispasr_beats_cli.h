// crispasr_beats_cli.h — CLI dispatcher for the `--beats` task.
//
// Beat tracking is its own task: audio in, a beat/downbeat grid out — not
// crispasr_segments — so per docs/source-separation-surface.md it gets its own
// early dispatcher rather than being layered onto transcribe(), mirroring
// --chords, --pitch and --separate. This resolves the model, detects its
// architecture from the GGUF ("beat-this"), decodes the audio to 22.05 kHz mono
// and prints one event per line (or JSON).
//
// Unlike --chords, the weights here are MIT for code AND weights, and the model
// is madmom-free, so there is no acceptance gate: no DBN is involved at any
// point. See docs/music-transcription/PLAN.md §251b.

#pragma once

struct whisper_params;

// Run the --beats task for every input file in `params.fname_inp`. Returns a
// process exit code (0 = success). Called from crispasr_run_backend() before
// any transcribe backend is constructed.
int crispasr_run_beats(const whisper_params& params);
