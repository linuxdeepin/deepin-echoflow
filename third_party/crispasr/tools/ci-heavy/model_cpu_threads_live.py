#!/usr/bin/env python3
"""#486: real CLI/C ABI ASR and seeded Dia roundtrips at 1/4/8 threads."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import wave

REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.environ["HEAVY_OUT"])
SCRATCH = Path(os.environ["HEAVY_SCRATCH"])
EXPECTED = "And so my fellow Americans ask not what your country can do for you ask what you can do for your country"
# Preserve a fixed prompt for old/new decoded-output comparison. Dia's
# existing 200-step limit truncates it; this is a parity check, not a claim
# that the existing runtime passes full-prompt intelligibility.
EXTENDED_PHRASE = ("The quick brown fox jumps over the lazy dog. Please listen carefully, because this is "
                   "a test of speech synthesis with a chosen number of CPU threads.")


def words(text):
    text = re.sub(r"<[^>]*>", "", text)
    # Nemotron emits a locale marker; the CLI subtitle formatter removes its
    # brackets. Neither representation is a spoken word in the JFK fixture.
    text = re.sub(r"\ben-US\b", "", text, flags=re.IGNORECASE)
    return re.findall(r"[a-z]+", text.lower())


def wer(reference, actual):
    row = list(range(len(actual) + 1))
    for i, r in enumerate(reference, 1):
        next_row = [i]
        for j, a in enumerate(actual, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (r != a)))
        row = next_row
    return row[-1] / len(reference)


parser = argparse.ArgumentParser()
parser.add_argument("--case", choices=("nemotron", "paraformer", "dia"))
parser.add_argument("--threads", type=int)
parser.add_argument("--model")
parser.add_argument("--asr-model")
parser.add_argument("--lib")
parser.add_argument("--baseline-ref")
parser.add_argument("--phrase", default="The quick brown fox jumps over the lazy dog.")
parser.add_argument("--diagnostic", action="store_true")
parser.add_argument("--tag")
args = parser.parse_args()
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)

if args.case:
    import numpy as np
    sys.path.insert(0, str(REPO / "python"))
    from crispasr import Session
    result = {"backend": args.case, "threads": args.threads, "outputs": [], "seed": 123 if args.case == "dia" else None}
    with Session(args.model, lib_path=args.lib, backend=args.case, n_threads=args.threads) as session:
        if args.case == "dia":
            session.set_temperature(1.2, seed=123)
            start = time.perf_counter()
            pcm = session.synthesize("[S1] " + args.phrase)
            result["seconds"] = time.perf_counter() - start
            result["sample_rate"] = session.output_sample_rate()
            assert result["sample_rate"] == 44100
            assert np.isfinite(pcm).all() and np.sqrt(np.mean(pcm.astype(np.float64) ** 2)) > 1e-4
            result["audio_seconds"] = len(pcm) / result["sample_rate"]
            assert 1 < result["audio_seconds"] < 45, result
        else:
            with wave.open(str(REPO / "samples/jfk.wav")) as source:
                assert source.getframerate() == 16000 and source.getnchannels() == 1
                pcm = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768
            for _ in range(2):
                start = time.perf_counter()
                text = " ".join(s.text for s in session.transcribe(pcm))
                result["outputs"].append({"text": text, "seconds": time.perf_counter() - start})
                assert words(text) == words(EXPECTED), text
    if args.case == "dia":
        with Session(args.asr_model, lib_path=args.lib, backend="nemotron", n_threads=4) as recognizer:
            text = " ".join(s.text for s in recognizer.transcribe(pcm, sample_rate=44100))
        result["outputs"].append({"text": text})
        result["phrase"] = args.phrase
        result["wer"] = wer(words(args.phrase), words(text))
        # Same intelligibility gate as the committed Dia roundtrip manifest.
    (OUT / f"{args.tag or str(args.case) + '-' + str(args.threads)}.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.case == "dia" and not args.diagnostic:
        assert result["wer"] <= .2, result
    print("LIVE_THREADS_RESULT", json.dumps(result), flush=True)
    sys.exit(0)

from huggingface_hub import hf_hub_download


def run(cmd, name, timeout=1800, env=None):
    with (OUT / f"{name}.log").open("w") as log:
        result = subprocess.run([str(c) for c in cmd], cwd=REPO, stdout=log,
                                stderr=subprocess.STDOUT, timeout=timeout, env=env)
    text = (OUT / f"{name}.log").read_text()
    print(name, "rc=", result.returncode, text[-1800:], flush=True)
    assert result.returncode == 0, f"{name} failed"
    return text


subprocess.run(["uptime"], check=True)
subprocess.run(["free", "-h"], check=True)
build = SCRATCH / "live-build"
run(["cmake", "-S", REPO, "-B", build, "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON",
     "-DGGML_NATIVE=OFF", "-DGGML_CUDA=OFF", "-DGGML_VULKAN=OFF", "-DGGML_BLAS=OFF",
     "-DCRISPASR_BUILD_TESTS=OFF", "-DCRISPASR_BUILD_SERVER=OFF",
     "-DCRISPASR_OPUS=OFF", "-DCRISPASR_AMR=OFF"], "live-configure")
run(["cmake", "--build", build, "--target", "crispasr-cli", "crispasr-lib", "-j4"], "live-build")
lib = next(build.rglob("libcrispasr.so"))
cli = build / "bin/crispasr"
manifest = json.loads((REPO / "tests/regression/manifest.json").read_text())
entries = {entry["backend_id"]: entry for entry in manifest["backends"]
           if entry["backend_id"] in ("nemotron", "paraformer")}
entries["dia"] = next(e for e in manifest["tts_backends"] if e["backend_id"] == "dia")
receipt = {"sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
           "models": {}, "cases": {}, "timing_note": "Functional proof, not a performance benchmark.",
           "dia_quality_scope": "Decoded-output parity with the unfixed default; full-prompt quality remains failing.",
           "dia_existing_default_step_cap": 200}
models = {}
for backend in ("nemotron", "paraformer", "dia"):
    spec = entries[backend]["gguf"]
    models[backend] = hf_hub_download(spec["repo"], spec["file"], revision=spec["revision"])
    receipt["models"][backend] = spec
    if backend == "dia":
        hf_hub_download(spec["repo"], "dac-44khz.gguf", revision=spec["revision"])
    if backend == "dia" and args.baseline_ref:
        subprocess.run(["git", "fetch", "--depth", "1", "origin", args.baseline_ref], check=True)
        runtime_files = ("src/nemotron.cpp", "src/paraformer.cpp", "src/dia_tts.cpp")
        candidate = {name: (REPO / name).read_bytes() for name in runtime_files}
        for name in runtime_files:
            (REPO / name).write_bytes(subprocess.check_output(
                ["git", "show", f"{args.baseline_ref}:{name}"]))
        run(["cmake", "--build", build, "--target", "crispasr-cli", "crispasr-lib", "-j4"], "baseline-rebuild")
        for revision in ("baseline", "candidate"):
            if revision == "candidate":
                for name, data in candidate.items():
                    (REPO / name).write_bytes(data)
                run(["cmake", "--build", build, "--target", "crispasr-cli", "crispasr-lib", "-j4"], "candidate-rebuild")
            tag = f"dia-extended-{revision}-4"
            run([sys.executable, __file__, "--case", "dia", "--threads", "4",
                 "--model", models["dia"], "--asr-model", models["nemotron"], "--lib", lib,
                 "--phrase", EXTENDED_PHRASE, "--diagnostic", "--tag", tag], tag)
            receipt["cases"][tag] = json.loads((OUT / f"{tag}.json").read_text())
        old = receipt["cases"]["dia-extended-baseline-4"]
        new = receipt["cases"]["dia-extended-candidate-4"]
        # Compare the unchanged default at four workers. Full-prompt WER
        # remains recorded, without calling the existing truncation a pass.
        assert words(old["outputs"][0]["text"]) == words(new["outputs"][0]["text"]), (old, new)
        assert old["audio_seconds"] == new["audio_seconds"], (old, new)
        receipt["baseline_runtime_sha"] = args.baseline_ref
        (OUT / "live-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for threads in (1, 4, 8):
        tag = f"{backend}-{threads}"
        cmd = [sys.executable, __file__, "--case", backend, "--threads", threads,
               "--model", models[backend], "--asr-model", models["nemotron"], "--lib", lib,
               "--phrase", EXTENDED_PHRASE]
        if backend == "dia":
            assert args.baseline_ref, "Dia parity requires an explicit unfixed control"
            cmd.append("--diagnostic")
        run(cmd, tag)
        receipt["cases"][tag] = json.loads((OUT / f"{tag}.json").read_text())
        if backend == "dia":
            baseline = receipt["cases"]["dia-extended-baseline-4"]
            actual = receipt["cases"][tag]
            assert words(actual["outputs"][0]["text"]) == words(baseline["outputs"][0]["text"]), (baseline, actual)
            assert actual["audio_seconds"] == baseline["audio_seconds"], (baseline, actual)
        (OUT / "live-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if backend != "dia":
        prefix = OUT / f"{backend}-cli-8"
        run([cli, "--backend", backend, "-m", models[backend], "-f", REPO / "samples/jfk.wav",
             "-t", "8", "-l", "en", "--no-gpu", "--no-punctuation", "-osrt", "-of", prefix], f"{backend}-cli")
        srt = prefix.with_suffix(".srt").read_text()
        text = " ".join(line for line in srt.splitlines() if line.strip() and not line.strip().isdigit() and "-->" not in line)
        assert words(text) == words(EXPECTED), text
        receipt["cases"][f"{backend}-cli-8"] = {"text": text}
        (OUT / "live-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print("MODEL_CPU_THREADS_LIVE_PASS", receipt["sha"], flush=True)
