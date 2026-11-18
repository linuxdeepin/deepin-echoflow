#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Hualet Wang
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compare raw/RNNoise ASR on identical, labelled, local 16 kHz mono WAVs.

Does not modify the user's config, download models, or enable transcript polish.
The reported recognition latency includes loading the model for each cold run;
use the live service diagnostics for capture queue/preview latency.
"""

import argparse
import configparser
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import unicodedata
import wave


def normalized(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold()
                   if not c.isspace() and not unicodedata.category(c).startswith("P"))


def edits(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, 1):
        current = [row]
        for column, actual in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (expected != actual)))
        previous = current
    return previous[-1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(rows):
    return {"character_errors": sum(row["character_errors"] for row in rows),
            "reference_characters": sum(row["reference_characters"] for row in rows),
            "cer": sum(row["character_errors"] for row in rows)
                   / sum(row["reference_characters"] for row in rows),
            "median_cold_recognition_ms": statistics.median(row["recognition_ms"] for row in rows)}


def compare(args):
    model = args.model.resolve(strict=True)
    binary = args.binary.resolve(strict=True)
    manifest = args.manifest.resolve(strict=True)
    samples = json.loads(manifest.read_text())
    if not isinstance(samples, list) or not samples:
        raise ValueError("manifest must contain a nonempty list of labelled samples")
    prepared = []
    ids = set()
    for sample in samples:
        sample_id = sample["id"]
        if not isinstance(sample_id, str) or sample_id in ids:
            raise ValueError("sample ids must be unique strings")
        ids.add(sample_id)
        reference = normalized(sample["reference"])
        if not reference:
            raise ValueError(f"{sample_id}: reference has no characters after normalization")
        audio = (manifest.parent / sample["audio"]).resolve(strict=True)
        with wave.open(str(audio), "rb") as reader:
            if (reader.getnchannels(), reader.getframerate(), reader.getsampwidth(), reader.getcomptype()) != (1, 16000, 2, "NONE"):
                raise ValueError(f"{audio}: expected PCM16, 16 kHz mono WAV")
            duration = reader.getnframes() / reader.getframerate()
            if duration <= 0:
                raise ValueError(f"{audio}: empty recording")
        digest = sha256(audio)
        if sample.get("sha256") and digest != sample["sha256"]:
            raise ValueError(f"{audio}: sha256 does not match manifest")
        prepared.append((sample, reference, audio, duration, digest))

    rows = []
    with tempfile.TemporaryDirectory(prefix="deepin-echoflow-denoise-compare-") as directory:
        config_path = Path(directory) / "compare.conf"
        for sample, reference, audio, duration, digest in prepared:
            for enabled in (False, True):
                config = configparser.ConfigParser(interpolation=None)
                for section, value in {
                    "advanced.crisp.model": str(model),
                    "advanced.crisp.threads": str(args.threads),
                    "basic.recognition.language": args.language,
                    "basic.recognition.noise_suppression": str(enabled).lower(),
                    "basic.polish.enabled": "false",
                    "advanced.storage.log_transcripts": "false",
                    "advanced.storage.save_live_debug_audio": "false",
                    "advanced.storage.recordings_dir": directory,
                }.items():
                    config[section] = {"value": value}
                with config_path.open("w") as output:
                    config.write(output)
                for run in range(args.repeat):
                    command = [str(binary), "--config", str(config_path),
                               "--transcribe-file", str(audio), "--transcribe-json"]
                    process = subprocess.run(command, capture_output=True, text=True,
                                             timeout=args.timeout,
                                             env=dict(os.environ, OPENBLAS_NUM_THREADS=str(args.threads)))
                    result = None
                    for line in reversed(process.stdout.splitlines()):
                        try:
                            candidate = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(candidate, dict) and "transcript" in candidate:
                            result = candidate
                            break
                    if process.returncode not in (0, 1) or result is None:
                        raise RuntimeError(f"{sample['id']}: ASR failed (exit={process.returncode}): "
                                           + process.stderr[-2000:])
                    if result.get("noise_suppression") is not enabled:
                        raise RuntimeError("ASR did not use the requested noise suppression setting")
                    rows.append({"id": sample["id"], "condition": sample.get("condition", "unspecified"),
                                 "audio": str(audio), "sha256": digest, "audio_seconds": duration,
                                 "noise_suppression": enabled, "run": run + 1,
                                 "reference": sample["reference"], "transcript": result["transcript"],
                                 "reference_characters": len(reference),
                                 "character_errors": edits(reference, normalized(result["transcript"])),
                                 "recognition_ms": result["recognition_ms"]})
    groups = {}
    for condition in sorted({row["condition"] for row in rows}):
        groups[condition] = {mode: summarize([row for row in rows
            if row["condition"] == condition and row["noise_suppression"] == enabled])
            for mode, enabled in (("raw", False), ("rnnoise", True))}
    return {"schema_version": 1, "model": str(model), "model_sha256": sha256(model),
            "binary": str(binary), "binary_sha256": sha256(binary),
            "cpu": platform.processor(), "machine": platform.machine(),
            "language": args.language, "threads": args.threads, "polish_enabled": False,
            "latency_includes_model_loading": True,
            "normalization": "NFKC, casefold, remove Unicode punctuation and whitespace",
            "conditions": groups, "samples": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.threads < 1 or args.repeat < 1 or args.timeout <= 0:
        parser.error("threads, repeat, and timeout must be positive")
    try:
        result = compare(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired, wave.Error) as error:
        parser.exit(1, f"compare-denoise: {error}\n")
    print(json.dumps(result["conditions"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
