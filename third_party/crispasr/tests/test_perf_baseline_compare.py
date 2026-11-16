#!/usr/bin/env python3
"""Unit tests for tools/perf_baseline_compare.py (docs/perf-sweep/PLAN.md TODO-4).

Model-free, no network — pure logic checks on the compare() function. Runs under
both pytest and `python -m unittest` (the CI unit-tests job invokes it either way).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import perf_baseline_compare as pbc  # noqa: E402


def _payload(rows):
    return {"results": rows}


BASE = _payload([
    {"engine": "whisper", "quant": "q8_0", "mode": "whole", "audio": "short",
     "realtime_factor": 10.0, "transcript_sample": "hello world"},
    {"engine": "parakeet", "quant": "q4_k", "mode": "whole", "audio": "long",
     "realtime_factor": 50.0, "transcript_sample": "the quick brown fox"},
])


class TestPerfBaselineCompare(unittest.TestCase):
    def test_identical_is_clean(self):
        hard, soft = pbc.compare(BASE, BASE)
        self.assertEqual(hard, [])
        self.assertEqual(soft, [])

    def test_dropped_engine_is_hard(self):
        cur = _payload(BASE["results"][:1])  # parakeet dropped
        hard, soft = pbc.compare(BASE, cur)
        self.assertTrue(any("MISSING" in m and "parakeet" in m for m in hard), hard)

    def test_zero_rtf_is_hard(self):
        # a wrong-backend load-failure exits ~0.5s and mints a bogus/zero RTF
        cur = _payload([dict(BASE["results"][0], realtime_factor=0.0)] + BASE["results"][1:])
        hard, _ = pbc.compare(BASE, cur)
        self.assertTrue(any("NO-WORK" in m for m in hard), hard)

    def test_empty_transcript_is_hard(self):
        cur = _payload([dict(BASE["results"][0], transcript_sample="")] + BASE["results"][1:])
        hard, _ = pbc.compare(BASE, cur)
        self.assertTrue(any("EMPTY" in m for m in hard), hard)

    def test_big_slowdown_is_soft_not_hard(self):
        # 10x -> 3x is a >2x slowdown: SOFT (informational), never HARD
        cur = _payload([dict(BASE["results"][0], realtime_factor=3.0)] + BASE["results"][1:])
        hard, soft = pbc.compare(BASE, cur)
        self.assertEqual(hard, [])
        self.assertTrue(any("SLOWER" in m for m in soft), soft)

    def test_small_slowdown_is_tolerated(self):
        # 10x -> 6.6x is only 1.5x slower: below the 2x factor, no warning (runner noise)
        cur = _payload([dict(BASE["results"][0], realtime_factor=6.6)] + BASE["results"][1:])
        hard, soft = pbc.compare(BASE, cur)
        self.assertEqual(hard, [])
        self.assertFalse(any("SLOWER" in m for m in soft), soft)

    def test_factor_is_configurable(self):
        # 10x -> 6.0x is 1.67x slower: warns at factor=1.5, not at factor=2.0
        cur = _payload([dict(BASE["results"][0], realtime_factor=6.0)] + BASE["results"][1:])
        _, soft_15 = pbc.compare(BASE, cur, factor=1.5)
        _, soft_20 = pbc.compare(BASE, cur, factor=2.0)
        self.assertTrue(any("SLOWER" in m for m in soft_15), soft_15)
        self.assertFalse(any("SLOWER" in m for m in soft_20), soft_20)

    def test_new_engine_is_informational_soft(self):
        cur = _payload(BASE["results"] + [
            {"engine": "canary", "quant": "q4_k", "mode": "whole", "audio": "short",
             "realtime_factor": 20.0, "transcript_sample": "new"}])
        hard, soft = pbc.compare(BASE, cur)
        self.assertEqual(hard, [])
        self.assertTrue(any("NEW" in m and "canary" in m for m in soft), soft)

    def test_strict_exit_code(self):
        # main() returns 1 only with --strict AND a hard issue
        import json
        import tempfile
        d = tempfile.mkdtemp()
        bp = os.path.join(d, "b.json")
        cp = os.path.join(d, "c.json")
        with open(bp, "w") as f:
            json.dump(BASE, f)
        with open(cp, "w") as f:
            json.dump(_payload(BASE["results"][:1]), f)  # parakeet dropped -> hard
        self.assertEqual(pbc.main([bp, cp]), 0)             # informational: exit 0
        self.assertEqual(pbc.main([bp, cp, "--strict"]), 1)  # strict: exit 1


if __name__ == "__main__":
    unittest.main()
