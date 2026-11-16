#!/usr/bin/env python3
"""#486: run actual model CPU worker probes in linked and module builds."""
import argparse
import collections
import json
import os
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--expect-broken", action="store_true")
parser.add_argument("--baseline-ref")
args = parser.parse_args()
repo = Path(__file__).resolve().parents[2]
out = Path(os.environ["HEAVY_OUT"])
scratch = Path(os.environ["HEAVY_SCRATCH"])
out.mkdir(parents=True, exist_ok=True)
scratch.mkdir(parents=True, exist_ok=True)
receipt = {"sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
           "expect_broken": args.expect_broken, "configurations": {}}
receipt["runtime_sha"] = receipt["sha"]
if args.expect_broken:
    assert args.baseline_ref, "Pass the unfixed runtime revision explicitly"
    subprocess.run(["git", "fetch", "--depth", "1", "origin", args.baseline_ref], check=True)
    receipt["runtime_sha"] = subprocess.check_output(
        ["git", "rev-parse", args.baseline_ref], text=True).strip()
    # Replay the unfixed runtime with the same current worker probe. This only
    # modifies the disposable hosted checkout; no baseline is committed/pushed.
    for filename in ("src/nemotron.cpp", "src/paraformer.cpp", "src/dia_tts.cpp"):
        (repo / filename).write_bytes(subprocess.check_output(
            ["git", "show", f"{receipt['runtime_sha']}:{filename}"]))


def run(cmd, name, cwd=repo, env=None):
    with (out / f"{name}.log").open("w") as log:
        result = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env,
                                stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    text = (out / f"{name}.log").read_text()
    print(name, "rc=", result.returncode, text[-1800:], flush=True)
    return result.returncode, text


subprocess.run(["uptime"], check=True)
subprocess.run(["free", "-h"], check=True)
for mode in ("linked", "module"):
    build = scratch / mode
    rc, _ = run(["cmake", "-S", repo, "-B", build, "-DCMAKE_BUILD_TYPE=Release",
                 f"-DBUILD_SHARED_LIBS={'ON' if mode == 'module' else 'OFF'}",
                 f"-DGGML_BACKEND_DL={'ON' if mode == 'module' else 'OFF'}",
                 "-DGGML_NATIVE=OFF", "-DGGML_CUDA=OFF", "-DGGML_VULKAN=OFF", "-DGGML_BLAS=OFF",
                 "-DCRISPASR_BUILD_TESTS=ON", "-DCRISPASR_BUILD_SERVER=OFF",
                 "-DCRISPASR_OPUS=OFF", "-DCRISPASR_AMR=OFF"], f"{mode}-configure")
    assert rc == 0, "Configure failed"
    rc, _ = run(["cmake", "--build", build, "--target", "test-model-cpu-threads", "ggml-cpu",
                 "-j4"], f"{mode}-build")
    assert rc == 0, "Build failed"
    binary = next(p for p in build.rglob("test-model-cpu-threads") if p.is_file())
    env = dict(os.environ, GGML_BACKEND_PATH=str(build / "bin"),
               CRISPASR_PARAFORMER_GPU="0", CRISPASR_DIA_TTS_GPU="0")
    spec = "[model-threads]" if args.expect_broken else "[issue-486]"
    rc, text = run([binary, spec], f"{mode}-workers", cwd=build, env=env)
    if args.expect_broken:
        assert rc != 0, "The unfixed build must fail the worker-count guard"
        assert "observed 4 and 4 calls" in text, "Missing reproduction of the ignored count"
        for name in ("nemotron CPU", "paraformer CPU", "dia CPU", "dia runtime setter"):
            assert name in text, f"Missing failing case: {name}"
        failures = re.findall(r"expected (\d+) workers, observed (\d+) and (\d+) calls", text)
        assert collections.Counter(failures) == {("1", "4", "4"): 4, ("3", "4", "4"): 3,
                                                 ("8", "4", "4"): 4}, failures
        assert re.search(r"test cases:\s+4\s*\|.*\b4 failed", text), text[-1000:]
    else:
        assert rc == 0 and "All tests passed" in text, "Worker-count regression failed"
    receipt["configurations"][mode] = {"returncode": rc, "tail": text[-1800:]}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print("MODEL_CPU_THREADS_PASS", receipt["sha"], flush=True)
