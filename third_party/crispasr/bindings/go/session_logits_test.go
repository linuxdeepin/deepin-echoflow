package whisper_test

import (
	"math"
	"os"
	"strings"
	"testing"

	// Packages
	whisper "github.com/CrispStrobe/CrispASR/bindings/go"
	wav "github.com/go-audio/wav"
	assert "github.com/stretchr/testify/assert"
)

// loadJFKClip reads samples/jfk.wav and returns the first ~4 s of 16 kHz mono
// float32 PCM. The 300M Omni CTC model has a ~5 s positional-encoding limit
// (per its HF card), so the ~11 s clip is truncated.
func loadJFKClip(t *testing.T) []float32 {
	t.Helper()
	fh, err := os.Open(SamplePath)
	if err != nil {
		t.Skip("Skipping test, sample not found:", SamplePath)
	}
	defer fh.Close()
	buf, err := wav.NewDecoder(fh).FullPCMBuffer()
	assert.NoError(t, err)
	data := buf.AsFloat32Buffer().Data
	if len(data) > 16000*4 {
		data = data[:16000*4]
	}
	return data
}

// Test_OmniCtcLogits exercises the raw per-frame CTC logits accessor on the
// Omni CTC backend. Mirrors the Rust session_omni_ctc_logits integration test
// and the Python TestOmniCtcLogits case. Self-skips unless OMNI_CTC_MODEL points
// at an Omni CTC GGUF.
func Test_OmniCtcLogits(t *testing.T) {
	assert := assert.New(t)
	modelPath := os.Getenv("OMNI_CTC_MODEL")
	if modelPath == "" {
		t.Skip("Skipping test, OMNI_CTC_MODEL not set")
	}
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, Omni CTC model not found:", modelPath)
	}

	// Auto-detect doesn't recognise every Omni GGUF on this pinned release;
	// the explicit "omniasr" backend routes all CTC/LLM variants.
	sess, err := whisper.SessionOpenExplicit(modelPath, "omniasr", 4)
	assert.NoError(err)
	assert.NotNil(sess)
	defer sess.Close()

	pcm := loadJFKClip(t)
	res, logits, err := sess.TranscribeWithLogits(pcm)
	assert.NoError(err)
	assert.NotNil(res)

	var text string
	if res != nil {
		parts := make([]string, 0, len(res.Segments))
		for _, s := range res.Segments {
			parts = append(parts, s.Text)
		}
		text = strings.TrimSpace(strings.Join(parts, " "))
	}
	assert.NotEmpty(text, "expected a transcript")

	// Accessor contract: a dense (NFrames, NVocab) grid, shaped + finite.
	assert.NotNil(logits, "CTC backend should return a logits grid")
	if logits == nil {
		return
	}
	assert.Greater(logits.NFrames, 0)
	assert.Greater(logits.NVocab, 0)
	assert.Equal(logits.NFrames*logits.NVocab, len(logits.Data))
	for _, v := range logits.Data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("logits must be finite, got %v", v)
		}
	}

	// Greedy CTC over the exposed logits (argmax per frame, collapse repeats,
	// drop blank id 0) must yield a non-degenerate token stream — evidence the
	// grid is the real decode input, not zeros/garbage.
	nTokens := 0
	prev := -1
	for t := 0; t < logits.NFrames; t++ {
		best, bestV := 0, float32(math.Inf(-1))
		base := t * logits.NVocab
		for v := 0; v < logits.NVocab; v++ {
			if logits.Data[base+v] > bestV {
				bestV = logits.Data[base+v]
				best = v
			}
		}
		if best != 0 && best != prev {
			nTokens++
		}
		prev = best
	}
	assert.Greater(nTokens, 0)
	assert.Less(nTokens, logits.NFrames)
}

// Test_OmniCtcVocab exercises the CTC vocabulary accessor: a greedy decode over
// the exposed logits, detokenized via the exposed vocab, must reproduce the
// built-in transcript — proving the vocab shares the logits' id space. Mirrors
// the Rust session_omni_ctc_vocab case. Self-skips unless OMNI_CTC_MODEL is set.
func Test_OmniCtcVocab(t *testing.T) {
	assert := assert.New(t)
	modelPath := os.Getenv("OMNI_CTC_MODEL")
	if modelPath == "" {
		t.Skip("Skipping test, OMNI_CTC_MODEL not set")
	}
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, Omni CTC model not found:", modelPath)
	}

	sess, err := whisper.SessionOpenExplicit(modelPath, "omniasr", 4)
	assert.NoError(err)
	defer sess.Close()

	vocab := sess.CtcVocab()
	assert.NotNil(vocab, "CTC backend should expose a vocab")
	assert.Greater(len(vocab), 1000)
	hasBoundary := false
	for _, p := range vocab {
		if p == " " || strings.Contains(p, "▁") {
			hasBoundary = true
			break
		}
	}
	assert.True(hasBoundary, "no word-boundary token in vocab")

	pcm := loadJFKClip(t)
	res, logits, err := sess.TranscribeWithLogits(pcm)
	assert.NoError(err)
	assert.NotNil(res)
	assert.NotNil(logits)
	if res == nil || logits == nil {
		return
	}
	parts := make([]string, 0, len(res.Segments))
	for _, s := range res.Segments {
		parts = append(parts, s.Text)
	}
	text := strings.TrimSpace(strings.Join(parts, " "))
	assert.NotEmpty(text)
	assert.Equal(len(vocab), logits.NVocab, "logit vocab dim != vocab len")

	// Greedy CTC: argmax per frame, collapse repeats, drop blank (id 0);
	// detokenize with U+2581 → space, then trim.
	var decoded strings.Builder
	prev := -1
	for tf := 0; tf < logits.NFrames; tf++ {
		best, bestV := 0, float32(math.Inf(-1))
		base := tf * logits.NVocab
		for v := 0; v < logits.NVocab; v++ {
			if logits.Data[base+v] > bestV {
				bestV = logits.Data[base+v]
				best = v
			}
		}
		if best != 0 && best != prev {
			decoded.WriteString(strings.ReplaceAll(vocab[best], "▁", " "))
		}
		prev = best
	}
	assert.Equal(text, strings.TrimSpace(decoded.String()),
		"vocab-detokenized greedy decode != built-in transcript")
}

// Test_OmniCtcLogits_CapturePreservesTranscript verifies that turning logit
// capture on and back off leaves the plain transcript unchanged. Mirrors the
// Python test_logits_capture_preserves_transcript case.
func Test_OmniCtcLogits_CapturePreservesTranscript(t *testing.T) {
	assert := assert.New(t)
	modelPath := os.Getenv("OMNI_CTC_MODEL")
	if modelPath == "" {
		t.Skip("Skipping test, OMNI_CTC_MODEL not set")
	}
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, Omni CTC model not found:", modelPath)
	}

	sess, err := whisper.SessionOpenExplicit(modelPath, "omniasr", 4)
	assert.NoError(err)
	defer sess.Close()

	pcm := loadJFKClip(t)
	res, _, err := sess.TranscribeWithLogits(pcm)
	assert.NoError(err)
	joined := func(r *whisper.TranscribeResult) string {
		parts := make([]string, 0, len(r.Segments))
		for _, s := range r.Segments {
			parts = append(parts, s.Text)
		}
		return strings.TrimSpace(strings.Join(parts, " "))
	}
	withLogits := joined(res)

	// Turn capture back off; the plain transcript must be unchanged.
	assert.NoError(sess.SetReturnLogits(false))
	plain, err := sess.Transcribe(pcm)
	assert.NoError(err)
	assert.Equal(withLogits, joined(plain), "logits capture changed the transcript")
}
