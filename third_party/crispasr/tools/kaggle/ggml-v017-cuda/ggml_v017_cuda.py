#!/usr/bin/env python3
"""
ggml v0.17 sync — CUDA EXECUTION validation on a real GPU.

The v0.10.2 -> v0.17.0 merge of CrispStrobe/ggml is validated on CPU and Metal
(full test-backend-ops green on an M1), and the new fork CI compiles CUDA — but
GitHub-hosted runners have no GPU, so no CUDA kernel has actually *run*. That is
the last gap before CrispASR's submodule pin can move, and it matters here
specifically because the merge touched CUDA col2im_1d, conv_transpose_1d and the
mul_mat/cuBLAS path:

  - upstream's ggml_col2im_1d replaced ours (different length rule + it is
    type-preserving), and src/core/conv.h was rewritten against it;
  - conv-transpose-1d.cu needed a ggml_cuda_cast for our F16 weight template;
  - ggml-cuda.cu was rebased onto upstream after the merge resurrected ~590
    lines upstream had deleted, and our #38/#125 F16-accumulation fix was
    re-applied at upstream's new compute_type decision point.

This kernel builds ggml standalone (not CrispASR — nothing here needs models or
weights) with CUDA for the T4, then runs test-backend-ops on the GPU. It reports
the conv/col2im families explicitly, since those are what the merge changed.

Follows the kaggle_harness regime (clone in-kernel, heartbeat around long ops).
"""
import os, re, sys, json, subprocess
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp"); TMP.mkdir(parents=True, exist_ok=True)
RESULTS = WORK / "ggml_v017_cuda_results.json"

GGML_URL = "https://github.com/CrispStrobe/ggml.git"
BRANCH = "sync/upstream-v0.17"
GGML = TMP / "ggml"

# The harness lives in CrispASR, so clone that too (shallow) just for the import.
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
CLONE = TMP / "CrispASR"
if not CLONE.exists():
    subprocess.run(["git", "clone", "--depth", "1", CRISPASR_URL, str(CLONE)],
                   check=False, timeout=1200)
# Prefer the harness from the clone; fall back to the copy bundled beside this
# script (kaggle_usage.md, "MUST follow") — a CPU-only worker has no internet and
# the clone can fail outright.
_h = CLONE / "tools" / "kaggle"
if (_h / "kaggle_harness.py").exists():
    sys.path.insert(0, str(_h))
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
results: dict = {"branch": BRANCH}


def sh(cmd, cwd=None, timeout=3600):
    print(f"$ {cmd}", flush=True)
    p = subprocess.run(cmd, shell=True, cwd=cwd, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


kh.step("clone.ggml", branch=BRANCH)
if not GGML.exists():
    rc, out = sh(f"git clone --depth 1 --branch {BRANCH} {GGML_URL} {GGML}")
    if rc != 0:
        print(out[-4000:], flush=True)
        raise SystemExit(f"ggml clone failed rc={rc}")
rc, sha = sh("git rev-parse HEAD", cwd=GGML)
results["ggml_sha"] = sha.strip()
kh.step("clone.done", sha=results["ggml_sha"])

rc, out = sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader")
results["gpu"] = out.strip()
print(f"GPU: {results['gpu']}", flush=True)

kh.step("toolchain")
kh.install_build_toolchain()

# The carried-patch guard runs here too: it is cheap and this is a fresh clone,
# so it also proves the branch on GitHub (not just the local tree) is intact.
kh.step("patch.guard")
rc, out = sh("./ci/check-crispasr-patches.sh", cwd=GGML)
results["patch_guard"] = {"rc": rc, "out": out.strip()[-500:]}
print(out, flush=True)

kh.step("cmake.configure")
# Kaggle hands out either a T4 (sm_75) or a P100 (sm_60) and does not let you
# choose, so hardcoding an arch gets "no kernel image is available for execution
# on the device" at RUN time — after a 20-minute build. Ask the device.
# (Do not blind-add sm_60 on CUDA 13: nvcc 13 dropped it. Kaggle is CUDA 12.x.)
rc, cap = sh("nvidia-smi --query-gpu=compute_cap --format=csv,noheader")
arch = cap.strip().splitlines()[0].replace(".", "") if rc == 0 and cap.strip() else "75"
results["compute_cap"] = cap.strip()
results["cuda_arch"] = arch
print(f"building for sm_{arch}", flush=True)
rc, out = sh(
    "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON "
    f"-DCMAKE_CUDA_ARCHITECTURES={arch} -DGGML_NATIVE=OFF",
    cwd=GGML)
if rc != 0:
    print(out[-6000:], flush=True)
    raise SystemExit(f"configure failed rc={rc}")

kh.step("build")
with kh.build_heartbeat("cuda.build", interval_s=30.0):
    rc, out = sh(f"cmake --build build -j{os.cpu_count()}", cwd=GGML, timeout=7200)
results["build_rc"] = rc
if rc != 0:
    print(out[-12000:], flush=True)
    results["build_tail"] = out[-4000:]
    RESULTS.write_text(json.dumps(results, indent=2))
    raise SystemExit(f"build failed rc={rc}")
print("build OK", flush=True)


ANSI = re.compile(r"\x1b\[[0-9;]*m")


def run_ops(filt=None, label="all"):
    """Run test-backend-ops, optionally filtered to one op."""
    cmd = "./build/bin/test-backend-ops"
    if filt:
        cmd += f" -o {filt}"
    with kh.build_heartbeat(f"ops.{label}", interval_s=30.0):
        rc, out = sh(cmd, cwd=GGML, timeout=5400)

    # Strip ANSI first: test-backend-ops colourises the verdict, so a naive
    # r"\bOK\b" never matches ("...[1;32mOK" — the preceding `m` is a word char)
    # and every run reports 0/0, which reads like success. Count on clean text.
    clean = ANSI.sub("", out)
    ok = len(re.findall(r": OK$", clean, re.M))
    fail_lines = [l.strip() for l in clean.splitlines() if re.search(r": FAIL", l)]
    passed = rc == 0 and "backends passed" in clean and "Backend CUDA0: FAIL" not in clean

    # Always keep the whole log as a kernel output — truncating to the tail hid
    # the actual failing case last time; the summary sits at the end but the
    # failures do not.
    (WORK / f"ops_{label}.log").write_text(clean)

    print(f"[{label}] rc={rc} OK={ok} FAIL={len(fail_lines)} passed={passed}", flush=True)
    for l in fail_lines[:40]:
        print("   FAILED:", l[:200], flush=True)
    return {"rc": rc, "ok": ok, "fail": len(fail_lines), "passed": passed,
            "failures": fail_lines[:40], "tail": clean.strip()[-1500:]}


# The families the merge actually changed, reported separately so a regression
# is attributable instead of buried in a 40k-line full run.
kh.step("ops.targeted")
results["ops"] = {}
for op in ["COL2IM_1D", "CONV_TRANSPOSE_1D", "CONV_2D", "CONV_3D", "MUL_MAT", "NORM"]:
    results["ops"][op] = run_ops(op, op)
    RESULTS.write_text(json.dumps(results, indent=2))

kh.step("ops.full")
results["ops"]["FULL"] = run_ops(None, "full")

results["verdict"] = (
    results["build_rc"] == 0
    and results["patch_guard"]["rc"] == 0
    and all(v["passed"] for v in results["ops"].values())
)
RESULTS.write_text(json.dumps(results, indent=2))
kh.step("done", verdict=results["verdict"])
print(json.dumps({k: v for k, v in results.items() if k != "ops"}, indent=2), flush=True)
print("VERDICT:", "PASS" if results["verdict"] else "FAIL", flush=True)
if not results["verdict"]:
    raise SystemExit(1)
