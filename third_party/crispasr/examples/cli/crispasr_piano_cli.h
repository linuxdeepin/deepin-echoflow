// crispasr_piano_cli.h — CLI dispatcher for the `--piano` task.
//
// Piano transcription is its own task: audio in, NOTE EVENTS out (onset,
// offset, MIDI pitch, velocity) — not crispasr_segments — so per
// docs/source-separation-surface.md it gets its own early dispatcher rather
// than being layered onto transcribe(), mirroring --pitch / --chords /
// --separate.
//
// Before this existed, piano was reachable only as `--backend
// piano-transcription` through transcribe(), which rendered each note into a
// segment whose text read like "C4 v=80". Parsing that string back into a note
// is lossy — velocity and exact offsets survive only as formatted text — and it
// was never the intended seam. The structured path (crispasr_session_piano*)
// already existed for bindings; this gives the CLI the same fidelity.

#pragma once

struct whisper_params;

// Run the --piano task for every input file in `params.fname_inp`. Returns a
// process exit code (0 = success). Called from crispasr_run_backend() before
// any transcribe backend is constructed.
int crispasr_run_piano(const whisper_params& params);
