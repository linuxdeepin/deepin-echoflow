#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Hualet Wang
# SPDX-License-Identifier: GPL-3.0-or-later

"""Deterministic comparison logic for deepin-echoflow release benchmarks."""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path


IDENTITY_FIELDS = (
    "cpu_model",
    "architecture",
    "cpu_flags",
    "build_profile",
    "benchmark_schema",
    "model_sha256",
    "manifest_sha256",
    "config_sha256",
    "threads",
    "openblas_threads",
)

AGGREGATE_LATENCY_LIMIT = 1.10
SAMPLE_LATENCY_LIMIT = 1.20
LIVE_BENCHMARK_SOURCE = "tests/benchmarks/live_replay_benchmark.cpp"
LIVE_BENCHMARK_SCHEMA = 2
LIVE_CHUNK_MS = 20.0


def _identity_value(identity, field):
    if field not in identity:
        return None
    value = identity[field]
    if field == "cpu_flags":
        return sorted(value)
    return value


def identities_match(left, right):
    """Return true only when every field needed for reuse is present and equal."""
    return all(
        field in left
        and field in right
        and _identity_value(left, field) == _identity_value(right, field)
        for field in IDENTITY_FIELDS
    )


def _sample_map(result):
    samples = result.get("samples", [])
    mapped = {sample.get("id"): sample for sample in samples}
    if None in mapped or len(mapped) != len(samples):
        raise ValueError("sample IDs must be present and unique")
    return mapped


def _median(sample):
    values = sample.get("latency_ms", [])
    if not values:
        raise ValueError(f"sample has no latency iterations: {sample['id']}")
    return statistics.median(float(value) for value in values)


def compare_results(baseline, candidate):
    """Compare two normalized result objects and return auditable gate metrics."""
    baseline_by_id = _sample_map(baseline)
    candidate_by_id = _sample_map(candidate)
    failures = []
    if baseline_by_id.keys() != candidate_by_id.keys():
        failures.append("sample_set")

    ratios = {}
    baseline_medians = []
    candidate_medians = []
    shared_ids = sorted(baseline_by_id.keys() & candidate_by_id.keys())
    for sample_id in shared_ids:
        before_sample = baseline_by_id[sample_id]
        after_sample = candidate_by_id[sample_id]
        if before_sample.get("sha256") != after_sample.get("sha256"):
            failures.append(f"sample_hash:{sample_id}")
        before = _median(before_sample)
        after = _median(after_sample)
        if before <= 0:
            raise ValueError(f"baseline latency must be positive: {sample_id}")
        ratio = after / before
        ratios[sample_id] = ratio
        baseline_medians.append(before)
        candidate_medians.append(after)
        if ratio > SAMPLE_LATENCY_LIMIT:
            failures.append(f"sample_latency:{sample_id}")
        if after_sample.get("required", True) and not after_sample.get("transcript", "").strip():
            failures.append(f"empty_transcript:{sample_id}")

    if not shared_ids:
        raise ValueError("results have no shared samples")
    baseline_aggregate = statistics.median(baseline_medians)
    candidate_aggregate = statistics.median(candidate_medians)
    aggregate_ratio = candidate_aggregate / baseline_aggregate
    if aggregate_ratio > AGGREGATE_LATENCY_LIMIT:
        failures.append("aggregate_latency")

    baseline_cer = float(baseline.get("aggregate_cer", 0.0))
    candidate_cer = float(candidate.get("aggregate_cer", 0.0))
    if candidate_cer > baseline_cer:
        failures.append("aggregate_cer")

    return {
        "passed": not failures,
        "failures": failures,
        "metrics": {
            "aggregate_latency_limit": AGGREGATE_LATENCY_LIMIT,
            "sample_latency_limit": SAMPLE_LATENCY_LIMIT,
            "baseline_median_ms": baseline_aggregate,
            "candidate_median_ms": candidate_aggregate,
            "aggregate_ratio": aggregate_ratio,
            "sample_ratios": ratios,
            "baseline_cer": baseline_cer,
            "candidate_cer": candidate_cer,
        },
    }


def _validate_live(live, offline):
    if not isinstance(live, dict):
        raise ValueError("missing live replay benchmark results")
    samples = _sample_map(live)
    offline_samples = _sample_map(offline)
    if not samples or samples.keys() != offline_samples.keys():
        raise ValueError("live replay sample set does not match offline benchmark")
    for sample_id, sample in samples.items():
        reference = offline_samples[sample_id]
        if any(sample.get(key) != reference.get(key) for key in ("sha256", "required", "reference")):
            raise ValueError(f"live replay sample metadata mismatch: {sample_id}")
        stop = sample.get("latency_ms", [])
        preview = sample.get("first_preview_ms", [])
        if (len(stop) < 3 or len(preview) != len(stop)
                or any(not math.isfinite(value) or value <= 0 for value in stop)
                or any(value is not None and (not math.isfinite(value) or value <= 0)
                       for value in preview)):
            raise ValueError(f"invalid live replay timings: {sample_id}")
    cer = live.get("aggregate_cer", -1)
    if not math.isfinite(cer) or cer < 0:
        raise ValueError("invalid live replay CER")
    return samples


def compare_release_results(baseline, candidate, *, live_candidate=False, live_baseline=False):
    comparison = compare_results(baseline, candidate)
    if not live_candidate:
        return comparison

    after = _validate_live(candidate.get("live"), candidate)
    failures = [f"live:empty_transcript:{sample_id}" for sample_id, sample in sorted(after.items())
                if sample.get("required", True) and not sample.get("transcript", "").strip()]
    if not any(value is not None for sample in after.values() for value in sample["first_preview_ms"]):
        failures.append("live:no_preview_observed")
    live_metrics = {"mode": "establish_baseline", "candidate_cer": candidate["live"]["aggregate_cer"]}
    if live_baseline:
        before = _validate_live(baseline.get("live"), baseline)
        # Differences smaller than one audio delivery interval are not useful
        # release blockers, especially when STOP reuses an earlier decode.
        def with_stop_floor(live):
            return {**live, "samples": [
                {**sample, "latency_ms": [max(LIVE_CHUNK_MS, ms) for ms in sample["latency_ms"]]}
                for sample in live["samples"]]}

        stop = compare_results(with_stop_floor(baseline["live"]), with_stop_floor(candidate["live"]))
        failures.extend(f"live:{failure}" for failure in stop["failures"]
                        if not failure.startswith("empty_transcript:"))
        first_before = []
        first_after = []
        preview_ratios = {}
        for sample_id in sorted(before.keys() & after.keys()):
            old = [ms for ms in before[sample_id]["first_preview_ms"] if ms is not None]
            new = [ms for ms in after[sample_id]["first_preview_ms"] if ms is not None]
            if not old or len(old) * 2 < len(before[sample_id]["first_preview_ms"]):
                continue  # Short clips can finalize without any preview.
            if not new or len(new) * 2 < len(after[sample_id]["first_preview_ms"]):
                failures.append(f"live:missing_preview:{sample_id}")
                continue
            old_ms, new_ms = statistics.median(old), statistics.median(new)
            first_before.append(old_ms)
            first_after.append(new_ms)
            preview_ratios[sample_id] = new_ms / old_ms
            if preview_ratios[sample_id] > SAMPLE_LATENCY_LIMIT:
                failures.append(f"live:sample_first_preview:{sample_id}")
        first_ratio = statistics.median(first_after) / statistics.median(first_before) if first_before else None
        if first_ratio is not None and first_ratio > AGGREGATE_LATENCY_LIMIT:
            failures.append("live:aggregate_first_preview")
        live_metrics.update({"mode": "compare", "stop_to_final": stop["metrics"],
                             "first_preview_aggregate_ratio": first_ratio,
                             "first_preview_sample_ratios": preview_ratios})
    comparison["metrics"]["live"] = live_metrics
    comparison["failures"].extend(failures)
    comparison["passed"] = not comparison["failures"]
    return comparison


def _load(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("baseline")
    compare.add_argument("candidate")
    compare.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    comparison = compare_results(_load(args.baseline), _load(args.candidate))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not comparison["passed"]:
        print("performance gate failed: " + ", ".join(comparison["failures"]), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
