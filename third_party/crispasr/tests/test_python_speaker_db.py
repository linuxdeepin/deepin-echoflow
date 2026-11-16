#!/usr/bin/env python3
"""Runtime tests for the Python SpeakerDB wrapper (issue #266, PLAN F4).

`crispasr.SpeakerDB` (python/crispasr/_binding.py) wraps the closed-roster
speaker profile C-ABI: consent-gated construction, `crispasr_speaker_db_open`
/ `_enroll2` (the gated v2 entry points), and refuses the legacy ungated
`crispasr_speaker_db_load` symbol at runtime. This had never been executed
end-to-end before this test.

Requires:
  - Built shared lib (cmake -DBUILD_SHARED_LIBS=ON -B build-shared &&
    cmake --build build-shared --target crispasr-lib)
  - numpy (the default Homebrew python3 on this box lacks it; miniconda at
    /Users/christianstrobele/miniconda3/bin/python has it)

Run:
  /Users/christianstrobele/miniconda3/bin/python tests/test_python_speaker_db.py
  # or with pytest:
  pytest tests/test_python_speaker_db.py -v
"""

import ctypes
import math
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")

LIB_PATH = os.environ.get("CRISPASR_LIB_PATH")
if not LIB_PATH:
    for candidate in [
        os.path.join(REPO_ROOT, "build-shared", "src", "libcrispasr.dylib"),
        os.path.join(REPO_ROOT, "build-shared", "src", "libcrispasr.so"),
        os.path.join(REPO_ROOT, "build", "src", "libcrispasr.dylib"),
        os.path.join(REPO_ROOT, "build", "src", "libcrispasr.so"),
    ]:
        if os.path.exists(candidate):
            LIB_PATH = candidate
            break

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


def l2_normalize(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


@unittest.skipUnless(LIB_PATH, "libcrispasr not built (set CRISPASR_LIB_PATH or build build-shared)")
@unittest.skipUnless(HAVE_NUMPY, "numpy not available (try /Users/christianstrobele/miniconda3/bin/python)")
class TestSpeakerDB(unittest.TestCase):
    """Issue #266: consent-gated, closed-roster speaker profile wrapper."""

    def setUp(self):
        from crispasr import SpeakerDB
        self.SpeakerDB = SpeakerDB
        self.tmpdir = tempfile.mkdtemp(prefix="crispasr-speaker-db-test-")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # (1) construction without consent raises BEFORE any native call.
    def test_no_consent_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.SpeakerDB(self.tmpdir, lib_path=LIB_PATH)

    def test_no_consent_raises_even_with_expected_names(self):
        # consent=False must refuse regardless of expected_names — the
        # ValueError is raised before crispasr_speaker_db_open is called.
        with self.assertRaises(ValueError):
            self.SpeakerDB(self.tmpdir, expected_names="Alice", lib_path=LIB_PATH)

    # (2) consent=True, no expected_names: enroll() still writes profiles.
    def test_enroll_with_consent_no_roster(self):
        db = self.SpeakerDB(self.tmpdir, consent=True, lib_path=LIB_PATH)
        try:
            # No roster given at construction -> db handle stays unopened
            # (no crispasr_speaker_db_open call at all); enroll() doesn't
            # need it, it writes straight to disk via enroll2.
            self.assertEqual(db.count, 0)

            emb = np.array(l2_normalize([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dtype=np.float32)
            ok = db.enroll("Alice", emb)
            self.assertTrue(ok, "enroll() should return True with consent")

            profile_path = os.path.join(self.tmpdir, "Alice.spkr")
            self.assertTrue(os.path.exists(profile_path), f"expected {profile_path} to exist")

            # v2 .spkr format: magic(4) + version(4) + dim(4) + dim*float32
            # + consent(1) + timestamp(8).
            with open(profile_path, "rb") as f:
                data = f.read()
            self.assertEqual(data[0:4], b"SPKR")
            version = int.from_bytes(data[4:8], "little")
            dim = int.from_bytes(data[8:12], "little")
            self.assertEqual(version, 2)
            self.assertEqual(dim, 8)
            consent_byte = data[12 + dim * 4]
            self.assertEqual(consent_byte, 1, "v2 profile should record consent=1")
        finally:
            db.close()

    # (3) reopening with expected_names narrows the roster; match() works.
    def test_reopen_with_roster_narrows_and_matches(self):
        # Enroll two speakers first (consent=True, no roster — same as
        # test_enroll_with_consent_no_roster).
        db0 = self.SpeakerDB(self.tmpdir, consent=True, lib_path=LIB_PATH)
        emb_a = np.array(l2_normalize([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dtype=np.float32)
        emb_b = np.array(l2_normalize([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dtype=np.float32)
        self.assertTrue(db0.enroll("a", emb_a))
        self.assertTrue(db0.enroll("b", emb_b))
        db0.close()

        # Reopen narrowed to just "a".
        db = self.SpeakerDB(self.tmpdir, expected_names="a", consent=True, lib_path=LIB_PATH)
        try:
            self.assertEqual(db.count, 1, "roster should narrow the loaded db to exactly 1 profile")

            # Same embedding as enrolled 'a' -> should match with cos ~= 1.0.
            name, score = db.match(emb_a, threshold=0.7)
            self.assertEqual(name, "a")
            self.assertGreater(score, 0.99)

            # Orthogonal embedding -> below threshold -> no match (None).
            orth = np.array(l2_normalize([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dtype=np.float32)
            name2, score2 = db.match(orth, threshold=0.7)
            self.assertIsNone(name2)
            self.assertLess(score2, 0.7)
        finally:
            db.close()

    # (4) the legacy ungated ABI symbol refuses at runtime (returns NULL).
    def test_legacy_speaker_db_load_refuses(self):
        lib = ctypes.CDLL(LIB_PATH)
        lib.crispasr_speaker_db_load.argtypes = [ctypes.c_char_p]
        lib.crispasr_speaker_db_load.restype = ctypes.c_void_p
        # Even against a directory with real enrolled profiles, the
        # pre-#266 open-1:N entry point must refuse (return NULL) —
        # it is kept only as a linkable symbol for old callers.
        db0 = self.SpeakerDB(self.tmpdir, consent=True, lib_path=LIB_PATH)
        emb = np.array(l2_normalize([1.0] + [0.0] * 7), dtype=np.float32)
        db0.enroll("Someone", emb)
        db0.close()

        handle = lib.crispasr_speaker_db_load(self.tmpdir.encode())
        self.assertIsNone(handle, "legacy crispasr_speaker_db_load must refuse (return NULL)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
