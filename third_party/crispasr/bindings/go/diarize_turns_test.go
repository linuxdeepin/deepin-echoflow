package whisper_test

// FoxNose speaker turns over the Go binding (#395).
//
// The C ABI grew crispasr_diarize_segments_turns_abi so a caller can split one
// of its OWN segments that spans a speaker change: labelling alone can never
// resolve finer than the grid it was handed, because apply_foxnose awards each
// segment wholly to the turn it overlaps most. Rust bound it; Go, Java,
// JavaScript and Ruby did not, so a Go caller could not reach the turns at all.
//
// The contract cases below need no model. The live case is gated on
// FOXNOSE_EMBEDDER (a WeSpeaker GGUF) plus FOXNOSE_WAV, so a checkout without
// weights stays green.

import (
	"encoding/binary"
	"fmt"
	"os"
	"testing"

	whisper "github.com/CrispStrobe/CrispASR/bindings/go"
	assert "github.com/stretchr/testify/assert"
	require "github.com/stretchr/testify/require"
)

// Minimal 16-bit mono PCM WAV reader — enough for the gated live case, and
// deliberately strict so a wrong fixture fails loudly instead of diarizing
// noise. Chunks are walked rather than assumed at fixed offsets: a WAV written
// by ffmpeg carries a LIST chunk ahead of "data".
func readWav16kMono(path string) ([]float32, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(raw) < 12 || string(raw[0:4]) != "RIFF" || string(raw[8:12]) != "WAVE" {
		return nil, fmt.Errorf("%s is not a RIFF/WAVE file", path)
	}
	var channels, bits int
	var rate int
	for off := 12; off+8 <= len(raw); {
		id := string(raw[off : off+4])
		size := int(binary.LittleEndian.Uint32(raw[off+4 : off+8]))
		body := off + 8
		if body+size > len(raw) {
			size = len(raw) - body
		}
		switch id {
		case "fmt ":
			if size < 16 {
				return nil, fmt.Errorf("%s: short fmt chunk", path)
			}
			channels = int(binary.LittleEndian.Uint16(raw[body+2 : body+4]))
			rate = int(binary.LittleEndian.Uint32(raw[body+4 : body+8]))
			bits = int(binary.LittleEndian.Uint16(raw[body+14 : body+16]))
		case "data":
			if channels != 1 || bits != 16 || rate != 16000 {
				return nil, fmt.Errorf("%s: need 16 kHz mono 16-bit, got %d Hz %dch %dbit",
					path, rate, channels, bits)
			}
			out := make([]float32, size/2)
			for i := range out {
				out[i] = float32(int16(binary.LittleEndian.Uint16(raw[body+2*i:body+2*i+2]))) / 32768.0
			}
			return out, nil
		}
		off = body + size
		if size%2 == 1 { // chunks are word-aligned
			off++
		}
	}
	return nil, fmt.Errorf("%s: no data chunk", path)
}

// A method that derives no turns must report zero of them, not an error — only
// FoxNose derives turns; the other four label the caller's grid directly.
func TestDiarizeTurnsNonFoxNoseReportsNone(t *testing.T) {
	pcm := make([]float32, 16000*4)
	for i := range pcm {
		pcm[i] = 0.01
	}
	segs := []whisper.DiarizeSeg{{T0: 0, T1: 100, Speaker: -1}, {T0: 200, T1: 300, Speaker: -1}}

	turns, err := whisper.DiarizeSegmentsWithTurns(pcm, nil, false, segs,
		whisper.DiarizeVADTurns, 2, "", nil)
	require.NoError(t, err)
	assert.Empty(t, turns, "only FoxNose derives turns")
}

// Asking for turns must not change the labels. Both entry points share one
// implementation in the C ABI, so this pins the older symbol's behaviour
// against a future edit to the shared body.
func TestDiarizeTurnsDoesNotChangeLabels(t *testing.T) {
	pcm := make([]float32, 16000*4)
	for i := range pcm {
		pcm[i] = 0.01
	}
	oldSegs := []whisper.DiarizeSeg{{T0: 0, T1: 100, Speaker: -1}, {T0: 200, T1: 300, Speaker: -1}}
	newSegs := []whisper.DiarizeSeg{{T0: 0, T1: 100, Speaker: -1}, {T0: 200, T1: 300, Speaker: -1}}

	require.NoError(t, whisper.DiarizeSegments(pcm, nil, false, oldSegs, whisper.DiarizeVADTurns, 2, ""))
	_, err := whisper.DiarizeSegmentsWithTurns(pcm, nil, false, newSegs, whisper.DiarizeVADTurns, 2, "", nil)
	require.NoError(t, err)

	for i := range oldSegs {
		assert.Equal(t, oldSegs[i].Speaker, newSegs[i].Speaker, "seg %d label changed", i)
	}
}

// A missing embedder must surface as an error rather than a crash.
func TestDiarizeTurnsFoxNoseMissingEmbedderErrors(t *testing.T) {
	pcm := make([]float32, 16000)
	segs := []whisper.DiarizeSeg{{T0: 0, T1: 100, Speaker: -1}}

	_, err := whisper.DiarizeSegmentsWithTurns(pcm, nil, false, segs,
		whisper.DiarizeMethodFoxNose, 2, "",
		&whisper.FoxNoseOpts{EmbedderPath: "/nonexistent/wespeaker.gguf"})
	assert.Error(t, err, "a missing embedder must fail, not crash")
}

// Real turns, real model. Also exercises the size-and-retry path: the buffer is
// sized from the audio length (one slot per 0.5 s, above FoxNose's 0.6 s
// embedding hop), and rc 2 retries once with the capacity the ABI reports.
func TestDiarizeTurnsFoxNoseLive(t *testing.T) {
	embedder := os.Getenv("FOXNOSE_EMBEDDER")
	wav := os.Getenv("FOXNOSE_WAV")
	if embedder == "" || wav == "" {
		t.Skip("set FOXNOSE_EMBEDDER + FOXNOSE_WAV (16 kHz mono) to run")
	}
	pcm, err := readWav16kMono(wav)
	require.NoError(t, err, "could not read %s as 16 kHz mono", wav)
	require.Greater(t, len(pcm), 16000, "fixture too short")

	// A 3 s grid: FoxNose skips spans under kMinSegmentSeconds (0.4 s), so a
	// per-word grid would starve the embedder.
	var segs []whisper.DiarizeSeg
	for t0 := 0; t0 < len(pcm)/160; t0 += 300 {
		t1 := t0 + 300
		if t1 > len(pcm)/160 {
			t1 = len(pcm) / 160
		}
		segs = append(segs, whisper.DiarizeSeg{T0: int64(t0), T1: int64(t1), Speaker: -1})
	}
	require.GreaterOrEqual(t, len(segs), 2, "fixture too short for a 3 s grid")

	turns, err := whisper.DiarizeSegmentsWithTurns(pcm, nil, false, segs,
		whisper.DiarizeMethodFoxNose, 4, "", &whisper.FoxNoseOpts{EmbedderPath: embedder})
	require.NoError(t, err)
	require.NotEmpty(t, turns, "FoxNose derived no turns")

	for i, tn := range turns {
		assert.Less(t, tn.T0, tn.T1, "turn %d is not forward in time", i)
		assert.GreaterOrEqual(t, tn.Speaker, int32(0), "turn %d speaker is dense, never -1", i)
		if i > 0 {
			assert.GreaterOrEqual(t, tn.T0, turns[i-1].T0, "turns must be ordered")
		}
	}
}
