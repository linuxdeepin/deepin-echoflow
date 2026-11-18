# SPDX-FileCopyrightText: 2026 Hualet Wang
# SPDX-License-Identifier: GPL-3.0-or-later

import copy
import unittest

from scripts import performance_gate as gate


IDENTITY = {
    "cpu_model": "AMD Ryzen 7 6800H",
    "architecture": "x86_64",
    "cpu_flags": ["avx", "avx2", "bmi2", "f16c", "fma", "sse4_2"],
    "build_profile": "x86-64-v3",
    "benchmark_schema": 1,
    "model_sha256": "1" * 64,
    "manifest_sha256": "2" * 64,
    "config_sha256": "3" * 64,
    "threads": 6,
    "openblas_threads": 4,
}


def result(latencies=(100.0, 100.0), *, cer=0.0, transcript="测试", ids=("normal",)):
    samples = []
    for sample_id in ids:
        samples.append({
            "id": sample_id,
            "sha256": (sample_id[0] * 64)[:64],
            "required": True,
            "latency_ms": list(latencies),
            "transcript": transcript,
            "cer": cer,
        })
    return {"identity": copy.deepcopy(IDENTITY), "samples": samples, "aggregate_cer": cer}


def with_live(*, preview=(500.0, 500.0, 500.0), stop=(100.0, 100.0, 100.0),
              cer=0.0, transcript="测试", ids=("normal",)):
    offline = result(ids=ids)
    live = result(stop, cer=cer, transcript=transcript, ids=ids)
    for sample in live["samples"]:
        sample["first_preview_ms"] = list(preview)
    offline["live"] = live
    return offline


class PerformanceGateTest(unittest.TestCase):
    def test_reuses_matching_identity(self):
        self.assertTrue(gate.identities_match(IDENTITY, copy.deepcopy(IDENTITY)))

    def test_cpu_mismatch_requires_rebuild(self):
        other = copy.deepcopy(IDENTITY)
        other["cpu_model"] = "Different CPU"
        self.assertFalse(gate.identities_match(IDENTITY, other))

    def test_cpu_flag_order_does_not_change_identity(self):
        other = copy.deepcopy(IDENTITY)
        other["cpu_flags"].reverse()
        self.assertTrue(gate.identities_match(IDENTITY, other))

    def test_aggregate_latency_at_ten_percent_passes(self):
        comparison = gate.compare_results(result(), result((110.0, 110.0)))
        self.assertTrue(comparison["passed"], comparison["failures"])

    def test_aggregate_latency_above_ten_percent_fails(self):
        comparison = gate.compare_results(result(), result((110.1, 110.1)))
        self.assertFalse(comparison["passed"])
        self.assertIn("aggregate_latency", comparison["failures"])

    def test_per_sample_latency_above_twenty_percent_fails(self):
        baseline = result(ids=("normal", "quiet"))
        candidate = result(ids=("normal", "quiet"))
        candidate["samples"][0]["latency_ms"] = [121.0, 121.0]
        candidate["samples"][1]["latency_ms"] = [80.0, 80.0]
        comparison = gate.compare_results(baseline, candidate)
        self.assertIn("sample_latency:normal", comparison["failures"])

    def test_cer_regression_fails(self):
        comparison = gate.compare_results(result(cer=0.0), result(cer=0.01))
        self.assertIn("aggregate_cer", comparison["failures"])

    def test_required_empty_transcript_fails(self):
        comparison = gate.compare_results(result(), result(transcript="  "))
        self.assertIn("empty_transcript:normal", comparison["failures"])

    def test_missing_sample_fails_without_crashing(self):
        comparison = gate.compare_results(
            result(ids=("normal", "quiet")), result(ids=("normal",)))
        self.assertIn("sample_set", comparison["failures"])

    def test_sample_hash_mismatch_fails(self):
        candidate = result()
        candidate["samples"][0]["sha256"] = "f" * 64
        comparison = gate.compare_results(result(), candidate)
        self.assertIn("sample_hash:normal", comparison["failures"])

    def test_first_live_release_establishes_a_baseline(self):
        comparison = gate.compare_release_results(result(), with_live(), live_candidate=True)
        self.assertTrue(comparison["passed"])
        self.assertEqual(comparison["metrics"]["live"]["mode"], "establish_baseline")

    def test_live_release_cannot_skip_required_measurements(self):
        with self.assertRaisesRegex(ValueError, "missing live replay"):
            gate.compare_release_results(result(), result(), live_candidate=True)
        with self.assertRaisesRegex(ValueError, "missing live replay"):
            gate.compare_release_results(result(), with_live(), live_candidate=True, live_baseline=True)

    def test_live_latency_and_quality_regressions_are_independent_of_offline(self):
        comparison = gate.compare_release_results(with_live(),
            with_live(preview=(650.0,) * 3, stop=(130.0,) * 3, cer=0.1),
            live_candidate=True, live_baseline=True)
        self.assertIn("live:sample_first_preview:normal", comparison["failures"])
        self.assertIn("live:aggregate_first_preview", comparison["failures"])
        self.assertIn("live:sample_latency:normal", comparison["failures"])
        self.assertIn("live:aggregate_cer", comparison["failures"])

    def test_short_clip_without_preview_is_valid_but_lost_preview_is_not(self):
        baseline = with_live(ids=("normal", "short"))
        baseline["live"]["samples"][1]["first_preview_ms"] = [None] * 3
        candidate = copy.deepcopy(baseline)
        comparison = gate.compare_release_results(baseline, candidate, live_candidate=True, live_baseline=True)
        self.assertTrue(comparison["passed"])
        candidate["live"]["samples"][0]["first_preview_ms"] = [500.0, None, None]
        comparison = gate.compare_release_results(baseline, candidate, live_candidate=True, live_baseline=True)
        self.assertIn("live:missing_preview:normal", comparison["failures"])


if __name__ == "__main__":
    unittest.main()
