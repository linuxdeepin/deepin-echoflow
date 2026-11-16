package io.github.ggerganov.whispercpp;

import static org.junit.jupiter.api.Assertions.*;

import com.sun.jna.Pointer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import java.io.File;

/**
 * Self-skipping accessor test for the raw CTC logits binding. Mirrors the
 * Rust {@code session_omni_ctc_logits} integration test.
 *
 * <p>Gated behind {@code OMNI_CTC_MODEL} (an absolute path to an Omni CTC
 * GGUF, e.g. {@code omniasr-ctc-300m-v2-q4_k.gguf}); the test self-skips when
 * the variable is unset so a fresh checkout has nothing to download. Auto-
 * detect doesn't recognise every Omni GGUF on the pinned release, so the
 * generic {@code "omniasr"} backend is selected explicitly.
 */
class CrispasrLogitsTest {

    @Test
    @EnabledIfEnvironmentVariable(named = "OMNI_CTC_MODEL", matches = ".+")
    void sessionOmniCtcLogits() throws Exception {
        String modelPath = System.getenv("OMNI_CTC_MODEL");

        // The 300M CTC model has a ~5 s positional limit (per its HF card),
        // so decode only the first ~4 s of the ~11 s JFK clip.
        float[] full = readWavPcm(new File(System.getProperty("user.dir"), "../../samples/jfk.wav"));
        int n = Math.min(full.length, 16000 * 4);
        float[] pcm = new float[n];
        System.arraycopy(full, 0, pcm, 0, n);

        Pointer session = CrispasrSession.Lib.INSTANCE.crispasr_session_open_explicit(modelPath, "omniasr", 4);
        assertNotNull(session, "session open omniasr");
        try {
            assertEquals(0,
                    CrispasrSession.Lib.INSTANCE.crispasr_session_set_return_logits(session, 1),
                    "set_return_logits");
            Pointer r = CrispasrSession.Lib.INSTANCE.crispasr_session_transcribe_lang(session, pcm, pcm.length, null);
            assertNotNull(r, "transcribe_lang");
            try {
                int nFrames = CrispasrSession.Lib.INSTANCE.crispasr_session_result_n_logit_frames(r);
                int nVocab = CrispasrSession.Lib.INSTANCE.crispasr_session_result_n_logit_vocab(r);
                Pointer ptr = CrispasrSession.Lib.INSTANCE.crispasr_session_result_logits(r);
                assertTrue(nFrames > 0 && nVocab > 0, "expected a dense CTC grid");
                assertNotNull(ptr, "CTC backend should return a logits buffer");

                float[] data = ptr.getFloatArray(0, nFrames * nVocab);
                assertEquals(nFrames * nVocab, data.length);
                for (float x : data) {
                    assertTrue(Float.isFinite(x), "logits must be finite");
                }

                // Greedy CTC over the exposed logits (argmax per frame, collapse
                // repeats, drop blank id 0) must yield a non-degenerate token
                // stream — evidence the grid is the real decode input.
                int prev = -1;
                int nTokens = 0;
                for (int t = 0; t < nFrames; t++) {
                    int base = t * nVocab;
                    int best = 0;
                    for (int v = 1; v < nVocab; v++) {
                        if (data[base + v] > data[base + best]) best = v;
                    }
                    if (best != 0 && best != prev) nTokens++;
                    prev = best;
                }
                assertTrue(nTokens > 0 && nTokens < nFrames,
                        "degenerate greedy decode: " + nTokens + " tokens over " + nFrames + " frames");
            } finally {
                CrispasrSession.Lib.INSTANCE.crispasr_session_result_free(r);
            }
        } finally {
            CrispasrSession.Lib.INSTANCE.crispasr_session_close(session);
        }
    }

    /**
     * CTC vocabulary accessor: a greedy decode over the exposed logits,
     * detokenized via the exposed vocab, must reproduce the built-in
     * transcript — proving the vocab shares the logits' id space. Mirrors the
     * Rust {@code session_omni_ctc_vocab} case. Uses the low-level {@code Lib}
     * API for the omniasr open, like {@link #sessionOmniCtcLogits()}.
     */
    @Test
    @EnabledIfEnvironmentVariable(named = "OMNI_CTC_MODEL", matches = ".+")
    void sessionOmniCtcVocab() throws Exception {
        String modelPath = System.getenv("OMNI_CTC_MODEL");
        float[] full = readWavPcm(new File(System.getProperty("user.dir"), "../../samples/jfk.wav"));
        int n = Math.min(full.length, 16000 * 4);
        float[] pcm = new float[n];
        System.arraycopy(full, 0, pcm, 0, n);

        Pointer session = CrispasrSession.Lib.INSTANCE.crispasr_session_open_explicit(modelPath, "omniasr", 4);
        assertNotNull(session, "session open omniasr");
        try {
            int nVocab = CrispasrSession.Lib.INSTANCE.crispasr_session_n_vocab(session);
            assertTrue(nVocab > 1000, "unexpectedly small vocab: " + nVocab);
            String[] vocab = new String[nVocab];
            boolean boundary = false;
            for (int i = 0; i < nVocab; i++) {
                String p = CrispasrSession.Lib.INSTANCE.crispasr_session_token_text(session, i);
                vocab[i] = p != null ? p : "";
                if (vocab[i].equals(" ") || vocab[i].contains("▁")) boundary = true;
            }
            assertTrue(boundary, "no word-boundary token in vocab");

            // Greedy CTC decode over the logits, detokenized via the vocab, must
            // reproduce the built-in transcript.
            assertEquals(0,
                    CrispasrSession.Lib.INSTANCE.crispasr_session_set_return_logits(session, 1), "set_return_logits");
            Pointer r = CrispasrSession.Lib.INSTANCE.crispasr_session_transcribe_lang(session, pcm, pcm.length, null);
            assertNotNull(r, "transcribe_lang");
            try {
                int nSeg = CrispasrSession.Lib.INSTANCE.crispasr_session_result_n_segments(r);
                StringBuilder textB = new StringBuilder();
                for (int i = 0; i < nSeg; i++) {
                    String t = CrispasrSession.Lib.INSTANCE.crispasr_session_result_segment_text(r, i);
                    if (t != null && !t.isEmpty()) {
                        if (textB.length() > 0) textB.append(' ');
                        textB.append(t);
                    }
                }
                String text = textB.toString().trim();
                assertFalse(text.isEmpty(), "expected a transcript");

                int nFrames = CrispasrSession.Lib.INSTANCE.crispasr_session_result_n_logit_frames(r);
                int nv = CrispasrSession.Lib.INSTANCE.crispasr_session_result_n_logit_vocab(r);
                Pointer ptr = CrispasrSession.Lib.INSTANCE.crispasr_session_result_logits(r);
                assertEquals(nVocab, nv, "logit vocab dim != vocab len");
                float[] data = ptr.getFloatArray(0, nFrames * nv);

                // Argmax per frame, collapse repeats, drop blank (id 0);
                // detokenize with U+2581 → space, then trim.
                StringBuilder decoded = new StringBuilder();
                int prev = -1;
                for (int t = 0; t < nFrames; t++) {
                    int base = t * nv;
                    int best = 0;
                    for (int v = 1; v < nv; v++) {
                        if (data[base + v] > data[base + best]) best = v;
                    }
                    if (best != 0 && best != prev) decoded.append(vocab[best].replace("▁", " "));
                    prev = best;
                }
                assertEquals(text, decoded.toString().trim(),
                        "vocab-detokenized greedy decode != built-in transcript");
            } finally {
                CrispasrSession.Lib.INSTANCE.crispasr_session_result_free(r);
            }
        } finally {
            CrispasrSession.Lib.INSTANCE.crispasr_session_close(session);
        }
    }

    /** Read a 16 kHz mono 16-bit PCM WAV into float32 (same idiom as WhisperCppTest). */
    private static float[] readWavPcm(File file) throws Exception {
        try (AudioInputStream ais = AudioSystem.getAudioInputStream(file)) {
            byte[] b = new byte[ais.available()];
            int off = 0, r;
            while (off < b.length && (r = ais.read(b, off, b.length - off)) > 0) off += r;
            float[] floats = new float[off / 2];
            for (int i = 0, j = 0; i + 1 < off; i += 2, j++) {
                int intSample = (int) (b[i + 1]) << 8 | (int) (b[i]) & 0xFF;
                floats[j] = intSample / 32767.0f;
            }
            return floats;
        }
    }
}
