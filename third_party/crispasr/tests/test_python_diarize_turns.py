import ctypes
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import crispasr._binding as binding  # noqa: E402


class _Callable:
    def __init__(self, fn):
        self.fn = fn
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.fn(*args)


class PythonDiarizeTurnsTest(unittest.TestCase):
    def test_abi_layout_and_capacity_retry(self):
        self.assertEqual(ctypes.sizeof(binding._DiarizeSegAbi), 24)
        self.assertEqual(ctypes.sizeof(binding._DiarizeTurnAbi), 24)
        if ctypes.sizeof(ctypes.c_void_p) == 8:
            self.assertEqual(ctypes.sizeof(binding._DiarizeOptsAbi), 48)
            self.assertEqual(binding._DiarizeOptsAbi.foxnose_embedder_path.offset, 24)

        calls = 0

        def old_call(*_args):
            self.fail("the turn-returning wrapper called the legacy symbol")

        def turns_call(_left, _right, _n, _stereo, segs_ptr, n_segs,
                       opts_ptr, turns_ptr, turn_cap, out_n_ptr):
            nonlocal calls
            calls += 1
            opts = ctypes.cast(
                opts_ptr, ctypes.POINTER(binding._DiarizeOptsAbi)
            ).contents
            self.assertEqual(opts.method, binding.DiarizeMethod.FOXNOSE)
            self.assertEqual(opts.foxnose_embedder_path, b"wespeaker.gguf")
            self.assertEqual((opts.min_speakers, opts.max_speakers,
                              opts.num_speakers), (2, 4, 3))

            out_n = ctypes.cast(out_n_ptr, ctypes.POINTER(ctypes.c_int32))
            if calls == 1:
                # Force the wrapper through the ABI's size-and-retry protocol.
                self.assertGreater(turn_cap, 0)
                out_n.contents.value = turn_cap + 2
                return 2

            out_n.contents.value = 2
            segs = ctypes.cast(
                segs_ptr, ctypes.POINTER(binding._DiarizeSegAbi)
            )
            for i in range(n_segs):
                segs[i].speaker = i + 1
            turns = ctypes.cast(
                turns_ptr, ctypes.POINTER(binding._DiarizeTurnAbi)
            )
            turns[0].t0_cs, turns[0].t1_cs, turns[0].speaker = 125, 250, 1
            turns[1].t0_cs, turns[1].t1_cs, turns[1].speaker = 250, 375, 2
            return 0

        fake_lib = type("FakeLib", (), {})()
        fake_lib.crispasr_diarize_segments_abi = _Callable(old_call)
        fake_lib.crispasr_diarize_segments_turns_abi = _Callable(turns_call)

        segs = [
            binding.DiarizeSegment(1.0, 2.5),
            binding.DiarizeSegment(2.5, 4.0),
        ]
        with mock.patch.object(binding.ctypes, "CDLL", return_value=fake_lib):
            ok, turns = binding.diarize_segments_with_turns(
                segs,
                np.zeros(16000, dtype=np.float32),
                method=binding.DiarizeMethod.FOXNOSE,
                foxnose_embedder_path="wespeaker.gguf",
                min_speakers=2,
                max_speakers=4,
                num_speakers=3,
                lib_path="/fake/libcrispasr.so",
            )

        self.assertTrue(ok)
        self.assertEqual(calls, 2)
        self.assertEqual([s.speaker for s in segs], [1, 2])
        self.assertEqual(
            [(t.t0, t.t1, t.speaker) for t in turns],
            [(1.25, 2.5, 1), (2.5, 3.75, 2)],
        )


if __name__ == "__main__":
    unittest.main()
