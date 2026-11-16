#!/usr/bin/env python3
"""
ggml v0.17 sync — the TTS gate.

The sync is green everywhere mechanical: full test-backend-ops on CPU + Metal,
six-job fork CI (including Vulkan executed on lavapipe), and 12863 ops on a real
P100. None of that covers the one thing that can still be silently wrong.

Upstream's ggml_col2im_1d has the SAME name and signature as the op we carried but
crops p0 from BOTH sides instead of treating it as a left offset. We adopted
upstream's and rewrote core_convt::convt1d_decomp to pass p0=0 and crop with a
view. Every TTS decoder's ConvTranspose1d goes through that one function — 20
callsites, plus convt1d_decomp_tf which delegates to it. A mistake there changes
decoder output LENGTH, which compiles, links, passes op-level tests, and produces
subtly wrong audio.

test_ggml_audio_ops_backend already proves convt1d_decomp is bit-exact against a
host ConvTranspose1d in both crop patterns. This closes the loop at the other
end: synthesise real speech through the real decoders and assert the words come
back, via tests/regression/run_one.py's TTS->ASR roundtrip (WER-gated against
parakeet).

Backends chosen to cover BOTH crop patterns and the specific families the sync
notes flag, not just to be numerous:
  symmetric crop (pad, pad)          kokoro, tada-codec, melotts, piper
  causal crop (0, K-stride)          pocket-tts, csm
  named in the sync as must-validate f5-tts, cosyvoice3

Follows the kaggle_harness regime (clone in-kernel, heartbeat around long ops,
HF token from the dataset).
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp")
TMP.mkdir(parents=True, exist_ok=True)
RESULTS = WORK / "tts_gate_results.json"

CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
BRANCH = os.environ.get("CRISPASR_BRANCH", "main")
REPO = TMP / "CrispASR"
BUILD = TMP / "build-regression"

# Order matters: cheapest / most load-bearing first, so a quota or timeout kill
# still leaves a usable verdict instead of nothing.
_env = os.environ.get("TTS_GATE_BACKENDS", "").strip()

# --recurse-submodules is the whole point: it is what pins the ggml under test.
if not REPO.exists():
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", BRANCH,
         "--recurse-submodules", "--shallow-submodules", CRISPASR_URL, str(REPO)],
        check=True, timeout=2400)

# Prefer the harness from the clone; fall back to the copy bundled beside this
# script (kaggle_usage.md, "MUST follow"): a CPU-only worker has no internet and
# the clone above can fail outright.
_h = REPO / "tools" / "kaggle"
if (_h / "kaggle_harness.py").exists():
    sys.path.insert(0, str(_h))
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
results: dict = {"branch": BRANCH, "backends": {}}


def sh(cmd, cwd=None, timeout=3600, env=None):
    print(f"$ {cmd}", flush=True)
    p = subprocess.run(cmd, shell=True, cwd=cwd, timeout=timeout, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


kh.step("clone.done")

# Full sweep: every tts_backends entry in the manifest, read from the CLONE (so a
# new entry needs no kernel edit). Must come after the clone — reading it at module
# scope raced ahead of the checkout. Cheapest-first by approx_size_mb, so a quota
# or timeout kill still leaves a usable partial verdict.
if _env:
    BACKENDS = [b.strip() for b in _env.split(",") if b.strip()]
else:
    with (REPO / "tests" / "regression" / "manifest.json").open() as _f:
        _m = json.load(_f)
    BACKENDS = [e["name"] for e in sorted(
        _m["tts_backends"], key=lambda e: e["gguf"].get("approx_size_mb", 1 << 30))]
print(f"sweeping {len(BACKENDS)} TTS backends: {', '.join(BACKENDS)}", flush=True)
rc, sha = sh("git rev-parse HEAD", cwd=REPO)
rc, gsha = sh("git -C ggml rev-parse HEAD", cwd=REPO)
results["crispasr_sha"] = sha.strip()
results["ggml_sha"] = gsha.strip()
print(f"CrispASR {results['crispasr_sha'][:8]}  ggml {results['ggml_sha'][:8]}", flush=True)

# Guard against the classic own-goal: a shallow clone whose submodule silently
# stayed at the old pin would make this whole run meaningless.
if not (REPO / "ggml" / "CMakeLists.txt").exists():
    raise SystemExit("ggml submodule not populated — the gate would test nothing")

rc, out = sh("nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader")
results["gpu"] = out.strip()
gpu_ok = rc == 0 and bool(out.strip())
print(f"GPU: {results['gpu'] or '(none)'}", flush=True)

kh.step("toolchain")
kh.install_build_toolchain()

token = kh.resolve_hf_token()
if token:
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token

# Use the harness's flag helpers rather than hand-rolling: they already carry the
# accumulated Kaggle lessons — the c2pa submodule that breaks cmake generate on a
# shallow clone, ccache/mold, the CUDA stubs on LIBRARY_PATH, NO_VMM, and a
# single pinned arch (Kaggle gives a T4 or a P100 with no say, and a hardcoded
# arch fails at RUN time after the entire build).
flags = [
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCRISPASR_BUILD_TESTS=OFF",
    "-DCRISPASR_BUILD_EXAMPLES=ON",
    "-DCRISPASR_BUILD_SERVER=OFF",
] + kh.cache_and_link_flags()
arch = ""
if gpu_ok:
    arch = kh.detect_cuda_arch()
    flags += kh.cuda_build_flags(arch)
results["cuda_arch"] = arch

kh.step("cmake.configure", arch=arch or "cpu")
with kh.build_heartbeat("cmake.configure"):
    rc, out = sh(f"cmake -S {REPO} -B {BUILD} -G Ninja " + " ".join(flags))
if rc != 0:
    print(out[-8000:], flush=True)
    raise SystemExit(f"configure failed rc={rc}")

# Target is crispasr-cli, NOT crispasr: the latter builds only the library and
# leaves bin/crispasr absent (examples/cli/CMakeLists.txt:12). That has burned
# both the GH and Kaggle regression runners before.
kh.step("cmake.build")
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli "
        f"-j{kh.safe_build_jobs(gpu=gpu_ok)}")

CRISPASR_BIN = BUILD / "bin" / "crispasr"
if not CRISPASR_BIN.exists():
    raise SystemExit(f"{CRISPASR_BIN} missing after build")
results["build_ok"] = True

# Preflight every artifact in one shot before burning GPU hours on synthesis.
kh.step("dry_run")
env = dict(os.environ, CRISPASR_BIN=str(CRISPASR_BIN), BUILD_DIR=str(BUILD))
rc, out = sh(f"{sys.executable} tests/regression/run_one.py --dry-run",
             cwd=REPO, timeout=1800, env=env)
results["dry_run_rc"] = rc
print(out[-3000:], flush=True)

kh.step("tts.roundtrip", n=len(BACKENDS))
for name in BACKENDS:
    kh.step(f"tts.{name}")
    t0 = time.time()
    with kh.build_heartbeat(f"tts.{name}", interval_s=30.0):
        rc, out = sh(f"{sys.executable} tests/regression/run_one.py {name}",
                     cwd=REPO, timeout=5400, env=env)
    clean = re.sub(r"\x1b\[[0-9;]*m", "", out)
    wer = None
    m = re.findall(r"wer\s*[=:]\s*([0-9.]+)", clean, re.I)
    if m:
        wer = float(m[-1])
    advisory = "ADVISORY" in clean
    results["backends"][name] = {
        "rc": rc, "wer": wer, "advisory": advisory,
        "secs": round(time.time() - t0, 1),
        "tail": clean.strip()[-2500:],
    }
    (WORK / f"tts_{name}.log").write_text(clean)
    print(f"[{name}] rc={rc} wer={wer} advisory={advisory} "
          f"({results['backends'][name]['secs']}s)", flush=True)
    if rc != 0:
        print(clean[-4000:], flush=True)
    RESULTS.write_text(json.dumps(results, indent=2))

gating = {k: v for k, v in results["backends"].items() if not v["advisory"]}
results["n_pass"] = sum(1 for v in gating.values() if v["rc"] == 0)
results["n_fail"] = sum(1 for v in gating.values() if v["rc"] != 0)
results["verdict"] = results["n_fail"] == 0 and results["n_pass"] > 0
RESULTS.write_text(json.dumps(results, indent=2))

kh.step("done", verdict=results["verdict"])
print(json.dumps({k: v for k, v in results.items() if k != "backends"}, indent=2), flush=True)
for k, v in results["backends"].items():
    print(f"  {k:28} rc={v['rc']} wer={v['wer']} "
          f"{'ADVISORY' if v['advisory'] else ''}", flush=True)
print("VERDICT:", "PASS" if results["verdict"] else "FAIL", flush=True)
if not results["verdict"]:
    raise SystemExit(1)
