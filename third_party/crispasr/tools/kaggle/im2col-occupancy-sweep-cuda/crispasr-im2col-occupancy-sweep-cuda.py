"""
CrispASR — CUDA im2col occupancy WIDER SWEEP on P100.

Context: on Metal, ggml's im2col sized thread-dim0 from batch N, so at inference
N=1 every threadgroup ran only KH*KW (3-11) threads → ~40x below bandwidth → 58%
of the melotts HiFi-GAN decode. Fixed for Metal (OW-blocking, feat branch). The
CUDA im2col is DIFFERENT: it launches
    <<<(ceil(IC*KH*KW/256), OW, N*OH), MIN(IC*KH*KW, 256)>>>
i.e. threads parallelize over IC*KH*KW (hundreds), NOT batch N — so CUDA should
NOT have the batch-1 occupancy pathology. This kernel VERIFIES that empirically
across a spread of conv-heavy models on a clean P100, and baselines each model's
conv/im2col GPU cost.

Question answered per model: what fraction of GPU kernel time is IM2COL vs MUL_MAT
vs the rest? If im2col is a small/proportionate slice for all → CUDA occupancy is
healthy → the Metal fix has no CUDA analog (Metal-specific). If any model shows
im2col-dominant on CUDA → a CUDA-side fix is warranted (new finding).

Per-kernel GPU breakdown via nsys (hardened detection: PATH, common dirs, find /,
apt install). No runtime code change; clones stock `main`.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
MODELS = WORK / "models"
CRISPASR = BUILD / "bin" / "crispasr"
SAMPLE = WORK / "jfk.wav"

CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")

# (backend, HF repo, kind). gguf filename resolved from the repo at runtime.
SWEEP = [
    ("parakeet",   "cstr/parakeet-ctc-0.6b-GGUF", "asr"),   # FastConformer, depthwise-conv heavy
    ("paraformer", "cstr/paraformer-zh-GGUF",     "asr"),   # SANM conv
    ("sensevoice", "cstr/sensevoice-small-GGUF",  "asr"),   # SANM conv
    ("moonshine",  "cstr/moonshine-tiny-GGUF",    "asr"),   # conv preprocessor
    ("melotts",    "cstr/melotts-en-v2-GGUF",     "tts"),   # HiFi-GAN vocoder (conv-heaviest)
]
TTS_TEXT = "Hello there, this is a vocoder op profile measurement."


def sh(cmd, **kw):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, **kw)


# ── Pre-clone + harness ───────────────────────────────────────────────────
print(f"[pre-clone] CrispASR @ {CRISPASR_REF}", flush=True)
if not REPO.exists():
    sh(f"git clone --depth 1 --branch {CRISPASR_REF} --recursive "
       f"https://github.com/CrispStrobe/CrispASR {REPO}", check=True)
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
if kh.resolve_hf_token():
    print("[auth] HF token resolved", flush=True)
sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
kh.step("clone.done", sha=sha, ref=CRISPASR_REF)

# ── Build (CUDA) ──────────────────────────────────────────────────────────
kh.step("build.begin")
kh.install_build_toolchain()
cmake_cmd = (f"cmake {REPO} -B{BUILD} -GNinja -DCMAKE_BUILD_TYPE=Release "
             + " ".join(kh.cuda_build_flags()) + " " + " ".join(kh.cache_and_link_flags()))
with kh.build_heartbeat("cmake-configure"):
    kh.sh_with_progress(cmake_cmd)
with kh.build_heartbeat("cmake-build"):
    kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli -- -j{kh.safe_build_jobs(gpu=True)}")
assert CRISPASR.is_file(), "crispasr binary missing after build"
subprocess.run(["cp", f"{REPO}/samples/jfk.wav", str(SAMPLE)], check=False)
kh.step("build.done")

# ── Locate nsys (hardened) ────────────────────────────────────────────────
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
sh("pip install -q huggingface_hub hf_transfer", check=False)


def find_nsys():
    for c in ["nsys", "/usr/local/cuda/bin/nsys",
              "/opt/nvidia/nsight-systems/*/bin/nsys",
              "/opt/nvidia/nsight-systems-cli/*/bin/nsys"]:
        hit = subprocess.run(f"ls {c} 2>/dev/null | head -1", shell=True, capture_output=True, text=True).stdout.strip()
        if hit:
            return hit
    hit = subprocess.run("find / -name nsys -type f 2>/dev/null | head -1", shell=True, capture_output=True, text=True).stdout.strip()
    if hit:
        return hit
    subprocess.run("apt-get install -y -q nsight-systems-cli 2>/dev/null || true", shell=True)
    hit = subprocess.run("find / -name nsys -type f 2>/dev/null | head -1", shell=True, capture_output=True, text=True).stdout.strip()
    return hit or None


NSYS = find_nsys()
kh.step("nsys.detect", nsys=NSYS or "NOT-FOUND")

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402
api = HfApi()


def resolve_gguf(repo):
    files = [f for f in api.list_repo_files(repo) if f.endswith(".gguf")]
    # prefer q4_k, then q8, then f16, then anything; skip bert companion here
    main = [f for f in files if "bert" not in f.lower()]
    for pat in ("q4_k", "q8", "f16"):
        for f in main:
            if pat in f.lower():
                return f
    return main[0] if main else files[0]


def profile_model(backend, repo, kind):
    try:
        fname = resolve_gguf(repo)
        model = hf_hub_download(repo, fname, local_dir=str(MODELS), local_dir_use_symlinks=False)
    except Exception as e:
        kh.step(f"{backend}.download.fail", err=str(e)[:200])
        return None
    extra = ""
    if kind == "tts":
        # melotts needs a bert companion
        try:
            bert = next(f for f in api.list_repo_files(repo) if "bert" in f.lower() and f.endswith(".gguf"))
            bpath = hf_hub_download(repo, bert, local_dir=str(MODELS), local_dir_use_symlinks=False)
            extra = f"--codec-model {bpath}"
        except Exception:
            pass
        task = f'--tts "{TTS_TEXT}" --tts-output {WORK}/out.wav'
    else:
        task = f"-l en -f {SAMPLE} -np"
    rep = WORK / f"prof_{backend}"
    cmd = (f"{NSYS} profile --force-overwrite true -o {rep} --stats=false "
           f"{CRISPASR} --backend {backend} -m {model} {extra} {task}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1800)
    if not (rep.with_suffix(".nsys-rep")).exists():
        kh.step(f"{backend}.nsys.fail", tail=(r.stderr or r.stdout or "")[-400:])
        return None
    csv = subprocess.run(f"{NSYS} stats --report cuda_gpu_kern_sum --format csv {rep}.nsys-rep",
                         shell=True, capture_output=True, text=True).stdout or ""
    agg, total = {}, 0.0
    for line in csv.splitlines():
        p = [c.strip().strip('"') for c in line.split(",")]
        if len(p) >= 2 and re.match(r"[\d.]+$", p[0]):
            pct = float(p[0]); name = p[-1].lower()
            key = ("im2col" if "im2col" in name else
                   "mul_mat" if any(k in name for k in ("mul_mat", "gemm", "cublas", "mmq", "mmv")) else
                   "conv" if "conv" in name else
                   "cpy_cast" if any(k in name for k in ("cpy", "cast", "convert", "dup")) else "other")
            agg[key] = agg.get(key, 0.0) + pct; total += pct
    row = {"backend": backend, "im2col_pct": round(agg.get("im2col", 0.0), 1),
           "mul_mat_pct": round(agg.get("mul_mat", 0.0), 1),
           "conv_pct": round(agg.get("conv", 0.0), 1),
           "other_pct": round(agg.get("other", 0.0), 1)}
    kh.step(f"{backend}.profiled", **row)
    return row


# ── Sweep ─────────────────────────────────────────────────────────────────
if not NSYS:
    kh.step("abort", reason="nsys unavailable — cannot get per-kernel breakdown")
    print("VERDICT: INCONCLUSIVE — nsys not found on the Kaggle image.")
    sys.exit(0)

kh.step("sweep.begin", models=len(SWEEP))
rows = [r for r in (profile_model(*m) for m in SWEEP) if r]

print("\n" + "=" * 72)
print(f"CUDA im2col occupancy sweep on P100 ({sha[:8]})")
print("=" * 72)
print(f"  {'backend':12} {'im2col%':>8} {'mul_mat%':>9} {'conv%':>7} {'other%':>7}")
worst = 0.0
for r in rows:
    print(f"  {r['backend']:12} {r['im2col_pct']:>8} {r['mul_mat_pct']:>9} {r['conv_pct']:>7} {r['other_pct']:>7}")
    worst = max(worst, r["im2col_pct"])
if not rows:
    verdict = "INCONCLUSIVE — no models profiled (check .download/.nsys.fail steps)."
elif worst >= 40:
    verdict = (f"CUDA im2col IS a bottleneck (max {worst}%) — unlike the code prediction; "
               f"a CUDA-side occupancy fix is warranted. Investigate the outlier model.")
else:
    verdict = (f"CUDA im2col HEALTHY (max {worst}% across {len(rows)} models) — confirms the "
               f"batch-1 occupancy bug is Metal-specific (CUDA parallelizes threads over "
               f"IC*KH*KW, not batch N). No CUDA port of the Metal fix needed.")
print(f"  VERDICT: {verdict}")
kh.step("verdict", text=verdict, worst_im2col_pct=worst, n_models=len(rows))
