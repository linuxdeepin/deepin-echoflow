#!/usr/bin/env python3
"""Verify the VAD LSTM `ldb >= k` fix BEFORE it is pushed.

THE BUG. `test-vad` aborts in Debug on gcc AND clang (rc=134):

    ggml/src/ggml-cpu/llamafile/sgemm.cpp:3700:
    Assertion `ldb >= k' failed.        (from whisper_vad_detect_speech)

whisper_vad_build_lstm_layer does:

    ggml_tensor* x_t = ggml_transpose(ctx0, cur);            // NON-CONTIGUOUS view
    ggml_tensor* inp_gate = ggml_mul_mat(ctx0, model.lstm_ih_weight, x_t);

ggml_transpose swaps nb[0]/nb[1], so the row stride of x_t becomes sizeof(float)
and `ldb` collapses to 1 while k is the LSTM hidden size (128) — hence the
precondition failure. The codebase's own idiom elsewhere (src/audioseal.cpp, five
sites) is ggml_cont(ctx, ggml_transpose(ctx, x)); the VAD is the outlier.

WHY IT MATTERED AND STAYED HIDDEN. Release defines NDEBUG, so the assert vanishes
and the matmul proceeds with a violated precondition instead of stopping. The test
"passed" everywhere for that reason, and CI ran 1 of 162 unit tests until
2026-07-29, so the Debug leg that does catch it had never run.

WHAT THIS PROVES. Two claims, neither assumed:
  1. the fix removes the abort   — Debug rc 134 -> 0
  2. the fix changes NOTHING     — Release segment output byte-identical
     before and after, so a real precondition fix is not smuggling in a
     behaviour change.

Both compilers, because the abort reproduced on both.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
RESULTS = WORK / "vad_sgemm_fix.json"
CLONE = Path("/kaggle/temp/CrispASR")

if not CLONE.exists():
    try:
        subprocess.run(["git", "clone", "--recurse-submodules",
                        "https://github.com/CrispStrobe/CrispASR.git", str(CLONE)],
                       check=True, timeout=1800)
    except Exception as e:  # noqa: BLE001
        print(f"clone failed: {e}", flush=True)

sys.path.insert(0, str(CLONE / "tools" / "kaggle") if CLONE.exists()
                else str(Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()


def sh(cmd, cwd=None, timeout=None):
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)


if not CLONE.exists():
    RESULTS.write_text(json.dumps({"verdict": "ERROR", "stage": "clone"}))
    raise SystemExit("clone missing")

kh.resolve_hf_token()
kh.install_build_toolchain()
sh("apt-get install -y --no-install-recommends clang || true", timeout=900)

SRC = CLONE / "src" / "crispasr.cpp"
BEFORE = "    struct ggml_tensor* x_t = ggml_transpose(ctx0, cur);"
AFTER = ("    // ggml_transpose returns a NON-CONTIGUOUS view (nb[0]/nb[1] swapped), and a\n"
         "    // transposed src1 makes llamafile_sgemm's ldb collapse to 1 while k is the\n"
         "    // hidden size -> `ldb >= k` fails. Release compiles that assert out and runs\n"
         "    // the matmul with the violated precondition instead of stopping.\n"
         "    struct ggml_tensor* x_t = ggml_cont(ctx0, ggml_transpose(ctx0, cur));")

results = {}
SEG_RE = re.compile(r"VAD segment (\d+): start = ([\d.]+), end = ([\d.]+)")


def build_and_run(tag, cc, cxx, build_type):
    bdir = f"build-{tag}"
    flags = (f"-DCMAKE_BUILD_TYPE={build_type} -DGGML_NATIVE=OFF -DCRISPASR_BUILD_TESTS=ON "
             f"-DCMAKE_C_COMPILER={cc} -DCMAKE_CXX_COMPILER={cxx} " + " ".join(kh.cache_and_link_flags()))
    with kh.build_heartbeat(f"cmake.{tag}", 30):
        r = sh(f"cmake -S . -B {bdir} -G Ninja {flags}", cwd=str(CLONE), timeout=1800)
    if r.returncode != 0:
        return {"configure_failed": r.stderr[-1500:]}
    with kh.build_heartbeat(f"ninja.{tag}", 30):
        try:
            kh.sh_with_progress(f"cmake --build {bdir} -j{kh.safe_build_jobs(gpu=True)} --target test-vad",
                                cwd=str(CLONE))
        except Exception as e:  # noqa: BLE001
            return {"build_failed": str(e)[-1500:]}
    r = sh(f"./{bdir}/bin/test-vad", cwd=str(CLONE), timeout=900)
    segs = SEG_RE.findall(r.stdout)
    assertion = ""
    m = re.search(r"Assertion `([^']+)' failed", r.stderr)
    if m:
        assertion = m.group(1)
    out = {"rc": r.returncode, "segments": segs, "assertion": assertion,
           "stderr_tail": r.stderr[-500:]}
    kh.step(f"run.{tag}", rc=r.returncode, n_segments=len(segs), assertion=assertion)
    print(f"\n##### {tag}: rc={r.returncode} segments={len(segs)} {('ASSERT: ' + assertion) if assertion else ''}",
          flush=True)
    for s in segs:
        print(f"    segment {s[0]}: {s[1]} -> {s[2]}", flush=True)
    return out


COMBOS = (("gcc", "gcc", "g++"), ("clang", "clang", "clang++"))

for phase in ("before", "after"):
    if phase == "after":
        text = SRC.read_text()
        if BEFORE not in text:
            RESULTS.write_text(json.dumps({"verdict": "ERROR", "stage": "patch",
                                           "err": "anchor not found — source moved"}, indent=2))
            raise SystemExit("patch anchor missing")
        SRC.write_text(text.replace(BEFORE, AFTER, 1))
        kh.step("patch.applied")
        print("\n=== applied ggml_cont(ggml_transpose(...)) to whisper_vad_build_lstm_layer ===", flush=True)
    for cc_tag, cc, cxx in COMBOS:
        if not sh(f"which {cc}").stdout.strip():
            results[f"{phase}.{cc_tag}.debug"] = {"skipped": f"{cc} absent"}
            continue
        for bt in ("Debug", "Release"):
            results[f"{phase}.{cc_tag}.{bt.lower()}"] = build_and_run(f"{phase}-{cc_tag}-{bt.lower()}", cc, cxx, bt)


def get(k, field, default=None):
    v = results.get(k) or {}
    return v.get(field, default)


checks = {}
for cc_tag, _, _ in COMBOS:
    b_dbg, a_dbg = f"before.{cc_tag}.debug", f"after.{cc_tag}.debug"
    b_rel, a_rel = f"before.{cc_tag}.release", f"after.{cc_tag}.release"
    if "skipped" in (results.get(b_dbg) or {}):
        continue
    checks[f"{cc_tag}:debug_aborted_before"] = get(b_dbg, "rc") == 134
    checks[f"{cc_tag}:debug_fixed_after"] = get(a_dbg, "rc") == 0
    checks[f"{cc_tag}:release_output_unchanged"] = get(b_rel, "segments") == get(a_rel, "segments")
    checks[f"{cc_tag}:release_still_ok"] = get(a_rel, "rc") == 0

results["checks"] = checks
results["verdict"] = "FIXED" if checks and all(checks.values()) else "FAIL"
RESULTS.write_text(json.dumps(results, indent=2))

print("\n===== VERDICT:", results["verdict"], "=====", flush=True)
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)
kh.step("verdict", result=results["verdict"], **checks)
try:
    kh.export_ccache_tar()
except Exception:  # noqa: BLE001
    pass
