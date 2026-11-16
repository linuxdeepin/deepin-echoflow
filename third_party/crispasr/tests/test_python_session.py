#!/usr/bin/env python3
"""Integration tests for the CrispASR Python Session API.

Requires:
  - Built libwhisper.so/dylib (cmake --build build)
  - whisper-tiny model: models/ggml-tiny.en.bin
  - parakeet model: set PARAKEET_MODEL env var or skip
  - samples/jfk.wav (11s JFK speech)

Run:
  python tests/test_python_session.py
  # or with pytest:
  pytest tests/test_python_session.py -v
"""

import os
import sys
import unittest
import wave

import numpy as np

# Add the python dir to path so we can import crispasr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
JFK_WAV = os.path.join(REPO_ROOT, "samples", "jfk.wav")
CRISPASR_TINY = os.path.join(REPO_ROOT, "models", "ggml-tiny.en.bin")
PARAKEET_MODEL = os.environ.get(
    "PARAKEET_MODEL",
    os.path.join(os.path.dirname(__file__), "..", "..", "test_cohere", "parakeet-tdt-0.6b-v3.gguf"),
)
# Omni CTC model for the raw-logits accessor test. Skips when unset/missing.
OMNI_CTC_MODEL = os.environ.get(
    "OMNI_CTC_MODEL",
    os.path.join(os.path.dirname(__file__), "..", "models", "omniasr-ctc.gguf"),
)

# Find the built shared library
LIB_PATH = os.environ.get("CRISPASR_LIB")
if not LIB_PATH:
    for candidate in [
        "/tmp/build-shared/src/libwhisper.so",
        os.path.join(REPO_ROOT, "build", "src", "libwhisper.so"),
        os.path.join(REPO_ROOT, "build", "src", "libwhisper.dylib"),
        os.path.join(REPO_ROOT, "build", "src", "Release", "whisper.dll"),
        os.path.join(REPO_ROOT, "build-shared", "src", "libwhisper.so"),
    ]:
        if os.path.exists(candidate):
            LIB_PATH = candidate
            break


def load_jfk_pcm():
    """Load jfk.wav as 16kHz mono float32 numpy array."""
    with wave.open(JFK_WAV, "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
        frames = wf.readframes(wf.getnframes())
        pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm


@unittest.skipUnless(LIB_PATH, "libwhisper not built")
class TestWhisperSession(unittest.TestCase):
    """Test the Session API with whisper-tiny.en."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(CRISPASR_TINY):
            raise unittest.SkipTest(f"Model not found: {CRISPASR_TINY}")
        from crispasr import Session
        # Whisper GGML files are auto-detected via magic bytes fallback
        cls.session = Session(CRISPASR_TINY, lib_path=LIB_PATH, n_threads=2)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "session"):
            cls.session.close()

    def test_backend_name(self):
        self.assertEqual(self.session.backend, "whisper")

    def test_transcribe_jfk(self):
        pcm = load_jfk_pcm()
        segs = self.session.transcribe(pcm)
        self.assertGreater(len(segs), 0)
        full_text = " ".join(s.text for s in segs).lower()
        self.assertIn("fellow americans", full_text)
        self.assertIn("country", full_text)

    def test_segment_timestamps(self):
        pcm = load_jfk_pcm()
        segs = self.session.transcribe(pcm)
        for seg in segs:
            self.assertGreaterEqual(seg.start, 0.0)
            self.assertGreater(seg.end, seg.start)
            self.assertLess(seg.end, 15.0)  # audio is ~11s

    def test_empty_audio(self):
        """Empty or near-silent audio should not crash."""
        silence = np.zeros(16000, dtype=np.float32)  # 1s silence
        segs = self.session.transcribe(silence)
        # May produce empty or very short result — just shouldn't crash
        self.assertIsInstance(segs, list)

    def test_very_short_audio(self):
        """Very short audio (0.1s) should not crash."""
        short = np.zeros(1600, dtype=np.float32)
        segs = self.session.transcribe(short)
        self.assertIsInstance(segs, list)


@unittest.skipUnless(LIB_PATH, "libwhisper not built")
@unittest.skipUnless(os.path.exists(PARAKEET_MODEL), f"Parakeet model not found at {PARAKEET_MODEL}")
class TestParakeetSession(unittest.TestCase):
    """Test the Session API with parakeet."""

    @classmethod
    def setUpClass(cls):
        from crispasr import Session
        cls.session = Session(PARAKEET_MODEL, lib_path=LIB_PATH, n_threads=2)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "session"):
            cls.session.close()

    def test_backend_name(self):
        self.assertEqual(self.session.backend, "parakeet")

    def test_transcribe_jfk(self):
        pcm = load_jfk_pcm()
        segs = self.session.transcribe(pcm)
        self.assertGreater(len(segs), 0)
        full_text = " ".join(s.text for s in segs).lower()
        self.assertIn("fellow americans", full_text)
        self.assertIn("country", full_text)

    def test_word_timestamps(self):
        """Parakeet should produce word-level timestamps."""
        pcm = load_jfk_pcm()
        segs = self.session.transcribe(pcm)
        self.assertGreater(len(segs), 0)
        # Parakeet produces words natively
        words = segs[0].words
        self.assertGreater(len(words), 0)
        for w in words:
            self.assertGreaterEqual(w.start, 0.0)
            self.assertGreaterEqual(w.end, w.start)
            self.assertTrue(len(w.text) > 0)

    def test_timestamps_monotonic(self):
        """Word timestamps should be non-decreasing."""
        pcm = load_jfk_pcm()
        segs = self.session.transcribe(pcm)
        for seg in segs:
            prev_end = 0.0
            for w in seg.words:
                self.assertGreaterEqual(w.start, prev_end - 0.01,
                    f"Word '{w.text}' starts at {w.start} before prev end {prev_end}")
                prev_end = w.end


@unittest.skipUnless(LIB_PATH, "libwhisper not built")
@unittest.skipUnless(os.path.exists(OMNI_CTC_MODEL), f"Omni CTC model not found at {OMNI_CTC_MODEL}")
class TestOmniCtcLogits(unittest.TestCase):
    """Raw per-frame CTC logits accessor (Omni CTC backend).

    Mirrors the Rust ``session_omni_ctc_logits`` integration test.
    """

    @classmethod
    def setUpClass(cls):
        from crispasr import Session
        # Auto-detect doesn't recognise every Omni GGUF on this pinned release;
        # the generic "omniasr" backend routes all CTC/LLM variants.
        cls.session = Session(OMNI_CTC_MODEL, lib_path=LIB_PATH, n_threads=4, backend="omniasr")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "session"):
            cls.session.close()

    def _clip(self):
        # The 300M CTC model has a ~5 s positional-encoding limit (per its HF
        # card), so decode only the first ~4 s of the ~11 s clip.
        return load_jfk_pcm()[: 16000 * 4]

    def test_transcribe_with_logits(self):
        pcm = self._clip()
        segs, logits = self.session.transcribe_with_logits(pcm)
        text = " ".join(s.text for s in segs).strip()
        self.assertTrue(text, "expected a transcript")

        # Accessor contract: a dense (n_frames, n_vocab) grid, shaped + finite.
        self.assertIsNotNone(logits, "CTC backend should return a logits grid")
        self.assertEqual(logits.ndim, 2)
        n_frames, n_vocab = logits.shape
        self.assertGreater(n_frames, 0)
        self.assertGreater(n_vocab, 0)
        self.assertEqual(logits.size, n_frames * n_vocab)
        self.assertTrue(np.isfinite(logits).all(), "logits must be finite")

        # Greedy CTC over the exposed logits (argmax per frame, collapse
        # repeats, drop blank id 0) must yield a non-degenerate token stream —
        # evidence the grid is the real decode input, not zeros/garbage.
        best = logits.argmax(axis=1)
        keep = best != 0
        keep[1:] &= best[1:] != best[:-1]
        n_tokens = int(keep.sum())
        self.assertGreater(n_tokens, 0)
        self.assertLess(n_tokens, n_frames)

    def test_logits_capture_preserves_transcript(self):
        pcm = self._clip()
        segs, _ = self.session.transcribe_with_logits(pcm)
        with_logits = " ".join(s.text for s in segs).strip()
        # Turn capture back off; the plain transcript must be unchanged.
        self.session.set_return_logits(False)
        plain = " ".join(s.text for s in self.session.transcribe(pcm)).strip()
        self.assertEqual(plain, with_logits, "logits capture changed the transcript")

    def test_ctc_vocab(self):
        vocab = self.session.ctc_vocab()
        self.assertIsNotNone(vocab, "CTC backend should expose a vocab")
        self.assertGreater(len(vocab), 1000)
        # A word-boundary token exists — literal space (v2) or U+2581 (v1).
        self.assertTrue(any(p == " " or "▁" in p for p in vocab),
                        "no word-boundary token in vocab")

        # End-to-end: a greedy CTC decode over the exposed logits, detokenized
        # via the exposed vocab, must reproduce the built-in transcript. Proves
        # the vocab shares the logits' id space.
        pcm = self._clip()
        segs, logits = self.session.transcribe_with_logits(pcm)
        text = " ".join(s.text for s in segs).strip()
        self.assertTrue(text)
        self.assertEqual(logits.shape[1], len(vocab))
        best = logits.argmax(axis=1)
        keep = best != 0
        keep[1:] &= best[1:] != best[:-1]
        decoded = "".join(vocab[i].replace("▁", " ") for i in best[keep]).strip()
        self.assertEqual(decoded, text,
                         "vocab-detokenized greedy decode != built-in transcript")


@unittest.skipUnless(LIB_PATH, "libwhisper not built")
class TestAvailableBackends(unittest.TestCase):
    """Test backend discovery."""

    def test_available_backends(self):
        from crispasr import Session
        backends = Session.available_backends(lib_path=LIB_PATH)
        self.assertIsInstance(backends, list)
        self.assertIn("whisper", backends)
        self.assertIn("parakeet", backends)

    def test_backend_autodetect(self):
        """Session should auto-detect backend from GGUF metadata."""
        if not os.path.exists(PARAKEET_MODEL):
            self.skipTest("Parakeet model not available")
        from crispasr import Session
        with Session(PARAKEET_MODEL, lib_path=LIB_PATH) as s:
            self.assertEqual(s.backend, "parakeet")


@unittest.skipUnless(LIB_PATH, "libwhisper not built")
class TestRegistryAndCache(unittest.TestCase):
    """Test model registry and cache helpers."""

    def test_cache_dir(self):
        from crispasr import cache_dir
        d = cache_dir(lib_path=LIB_PATH)
        self.assertIsInstance(d, str)
        self.assertGreater(len(d), 0)

    def test_registry_lookup(self):
        from crispasr import registry_lookup
        entry = registry_lookup("parakeet", lib_path=LIB_PATH)
        # May return None if registry not compiled in, or a RegistryEntry
        if entry is not None:
            self.assertIsInstance(entry.filename, str)
            self.assertGreater(len(entry.filename), 0)

    def test_registry_default_bundle(self):
        from crispasr import registry_default_bundle
        bundle = registry_default_bundle("omnivoice", lib_path=LIB_PATH)
        self.assertIsNotNone(bundle)
        self.assertEqual(bundle.backend, "omnivoice")
        self.assertEqual(
            [artifact.kind for artifact in bundle.artifacts],
            ["primary", "companion"],
        )
        self.assertEqual(bundle.artifacts[0].filename, "omnivoice-f16.gguf")
        self.assertEqual(
            bundle.artifacts[1].filename,
            "omnivoice-tokenizer-f16.gguf",
        )

    def test_registry_default_bundle_unknown(self):
        from crispasr import registry_default_bundle
        self.assertIsNone(
            registry_default_bundle("nonexistent-backend-xyz", lib_path=LIB_PATH)
        )


@unittest.skipUnless(LIB_PATH, "libwhisper not built")
class TestKokoroPhonemeCacheClear(unittest.TestCase):
    """PLAN #56 #5: kokoro_phoneme_cache_clear C ABI is exposed and callable.

    Doesn't require a kokoro model — only verifies the symbol export and
    the null-handle return path.
    """

    def test_symbol_exported(self):
        import ctypes
        lib = ctypes.CDLL(LIB_PATH)
        self.assertTrue(hasattr(lib, "crispasr_session_kokoro_clear_phoneme_cache"))

    def test_null_handle_returns_neg_one(self):
        import ctypes
        lib = ctypes.CDLL(LIB_PATH)
        fn = lib.crispasr_session_kokoro_clear_phoneme_cache
        fn.argtypes = [ctypes.c_void_p]
        fn.restype = ctypes.c_int
        self.assertEqual(fn(ctypes.c_void_p(0)), -1)


@unittest.skipUnless(LIB_PATH, "libwhisper not built")
class TestMicAPI(unittest.TestCase):
    """PLAN #62d: mic capture API exists, default device probe works."""

    def test_symbols_exported(self):
        import ctypes
        lib = ctypes.CDLL(LIB_PATH)
        for name in (
            "crispasr_mic_open",
            "crispasr_mic_start",
            "crispasr_mic_stop",
            "crispasr_mic_close",
            "crispasr_mic_default_device_name",
        ):
            self.assertTrue(hasattr(lib, name), f"missing symbol: {name}")

    def test_default_device_name_callable(self):
        from crispasr import mic_default_device_name
        # On CI/headless the device list may be empty; not asserting non-empty.
        name = mic_default_device_name(lib_path=LIB_PATH)
        self.assertIsInstance(name, str)


@unittest.skipUnless(LIB_PATH, "libwhisper not built")
class TestStreamingAPI(unittest.TestCase):
    """PLAN #62a/b: session-level streaming API symbols are exported and
    null-handle paths return cleanly."""

    def test_symbols_exported(self):
        import ctypes
        lib = ctypes.CDLL(LIB_PATH)
        for name in (
            "crispasr_session_stream_open",
            "crispasr_stream_feed",
            "crispasr_stream_get_text",
            "crispasr_stream_flush",
            "crispasr_stream_close",
        ):
            self.assertTrue(hasattr(lib, name), f"missing symbol: {name}")

    def test_session_stream_open_null_handle_returns_null(self):
        import ctypes
        lib = ctypes.CDLL(LIB_PATH)
        fn = lib.crispasr_session_stream_open
        fn.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                       ctypes.c_char_p, ctypes.c_int]
        fn.restype = ctypes.c_void_p
        self.assertIsNone(fn(ctypes.c_void_p(0), 4, 3000, 10000, 200, b"", 0))


@unittest.skipUnless(LIB_PATH, "libwhisper not built")
class TestRegistryEnumeration(unittest.TestCase):
    """Registry enumeration via list_known_models() returns a non-empty
    list of backend names; each looks up to a full RegistryEntry."""

    def test_list_known_models(self):
        from crispasr import list_known_models, registry_lookup
        backends = list_known_models(lib_path=LIB_PATH)
        self.assertIsInstance(backends, list)
        self.assertGreater(len(backends), 10, "registry should have >10 entries")
        self.assertIn("whisper", backends)
        self.assertIn("parakeet", backends)
        # First entry should be looked-up-able
        entry = registry_lookup(backends[0], lib_path=LIB_PATH)
        self.assertIsNotNone(entry)
        self.assertGreater(len(entry.filename), 0)
        self.assertTrue(entry.url.startswith("http"))


@unittest.skipUnless(LIB_PATH, "libwhisper not built")
class TestSessionStateSetters(unittest.TestCase):
    """PLAN #59 partial unblock: session-level setters for source_language,
    target_language, punctuation, translate, temperature, detect_language.

    Symbol-level + null-handle return path checks. No model needed.
    """

    def test_all_symbols_exported(self):
        import ctypes
        lib = ctypes.CDLL(LIB_PATH)
        for name in (
            "crispasr_session_set_source_language",
            "crispasr_session_set_target_language",
            "crispasr_session_set_punctuation",
            "crispasr_session_set_translate",
            "crispasr_session_set_temperature",
            "crispasr_session_detect_language",
        ):
            self.assertTrue(hasattr(lib, name), f"missing symbol: {name}")

    def test_null_handle_returns_neg_one(self):
        import ctypes
        lib = ctypes.CDLL(LIB_PATH)
        for name, argtypes, args in [
            ("crispasr_session_set_source_language", [ctypes.c_void_p, ctypes.c_char_p], [None, b"en"]),
            ("crispasr_session_set_target_language", [ctypes.c_void_p, ctypes.c_char_p], [None, b"de"]),
            ("crispasr_session_set_punctuation", [ctypes.c_void_p, ctypes.c_int], [None, 1]),
            ("crispasr_session_set_translate", [ctypes.c_void_p, ctypes.c_int], [None, 1]),
            ("crispasr_session_set_temperature", [ctypes.c_void_p, ctypes.c_float, ctypes.c_uint64], [None, 0.5, 0]),
        ]:
            fn = getattr(lib, name)
            fn.argtypes = argtypes
            fn.restype = ctypes.c_int
            self.assertEqual(fn(*args), -1, f"{name}(null, ...) should return -1")


@unittest.skipUnless(LIB_PATH, "libwhisper not built")
class TestKokoroPhonemizerParityHarness(unittest.TestCase):
    """PLAN #56 #4: phonemizer-diff harness ABI is exposed.

    Both kokoro_phonemize_text_lib and kokoro_phonemize_text_popen
    must be present for `tools/check_kokoro_phonemizer_parity.py` to
    work. Doesn't require espeak-ng to be installed — only checks the
    symbols are exported.
    """

    def test_symbols_exported(self):
        import ctypes
        lib = ctypes.CDLL(LIB_PATH)
        self.assertTrue(hasattr(lib, "kokoro_phonemize_text_lib"))
        self.assertTrue(hasattr(lib, "kokoro_phonemize_text_popen"))

    def test_null_args_return_nullptr(self):
        import ctypes
        lib = ctypes.CDLL(LIB_PATH)
        for name in ("kokoro_phonemize_text_lib", "kokoro_phonemize_text_popen"):
            fn = getattr(lib, name)
            fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
            fn.restype = ctypes.c_void_p
            self.assertIsNone(fn(None, None), f"{name}(None, None) should return nullptr")
            self.assertIsNone(fn(b"en-us", None), f"{name}('en-us', None) should return nullptr")
            self.assertIsNone(fn(None, b"hello"), f"{name}(None, 'hello') should return nullptr")


VOXTRAL4B_MODEL = os.environ.get(
    "VOXTRAL4B_MODEL", "/Volumes/backups/ai/crispasr-models/voxtral-mini-4b-realtime-q4_k.gguf"
)


@unittest.skipUnless(LIB_PATH, "libwhisper not built")
class TestVoxtral4bStreamingABI(unittest.TestCase):
    """PLAN #7: voxtral4b native streaming ABI is exposed and the
    null-handle paths return cleanly. Doesn't require the model — pure
    symbol + null-arg lifecycle checks."""

    def test_symbols_exported(self):
        import ctypes
        lib = ctypes.CDLL(LIB_PATH)
        for name in (
            "voxtral4b_stream_open",
            "voxtral4b_stream_feed",
            "voxtral4b_stream_get_text",
            "voxtral4b_stream_flush",
            "voxtral4b_stream_close",
        ):
            self.assertTrue(hasattr(lib, name), f"missing symbol: {name}")

    def test_null_handle_returns_null(self):
        import ctypes
        lib = ctypes.CDLL(LIB_PATH)
        fn = lib.voxtral4b_stream_open
        fn.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        fn.restype = ctypes.c_void_p
        self.assertIsNone(fn(ctypes.c_void_p(0), 0, 0))


@unittest.skipUnless(LIB_PATH and os.path.exists(VOXTRAL4B_MODEL),
                     "voxtral4b model not configured (set VOXTRAL4B_MODEL)")
class TestVoxtral4bStreamingBitExact(unittest.TestCase):
    """PLAN #7 phase 1.5+2+3 regression test: streaming output on JFK
    must match the CLI batch transcribe byte-for-byte.

    Skipped in CI (model not present); runs locally where the
    voxtral4b GGUF is available. Validates the full pipeline:
    incremental encoder + speculative prefill + combined-chunk flush
    + decode + tokenize. Catches regressions in any of the phase 1+2+3
    perf changes.
    """

    EXPECTED_TRANSCRIPT = (
        "And so, my fellow Americans, ask not what your country can do "
        "for you. Ask what you can do for your country."
    )

    def test_streaming_jfk_matches_batch(self):
        # Use the bench harness directly. It opens a session, streams
        # the audio in 80ms chunks, flushes, and reads the transcript.
        import ctypes
        # Set up the lib path so the python binding can find it.
        os.environ.setdefault("CRISPASR_LIB_PATH", LIB_PATH)
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
        import crispasr  # noqa: F401 — import side-effect
        sess = crispasr.Session(backend="voxtral4b", model_path=VOXTRAL4B_MODEL)
        try:
            stream = sess.stream_open(step_ms=80, length_ms=15000)
            try:
                pcm = load_jfk_pcm()
                # Feed in 80ms chunks (1280 samples at 16kHz)
                chunk_n = 1280
                for i in range(0, len(pcm), chunk_n):
                    stream.feed(pcm[i:i + chunk_n])
                stream.flush()
                got = stream.get_text()["text"].strip()
            finally:
                stream.close()
        finally:
            sess.close()
        self.assertEqual(got, self.EXPECTED_TRANSCRIPT,
                         f"streaming transcript diverged from batch:\n  got:    {got!r}\n  want:   {self.EXPECTED_TRANSCRIPT!r}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
