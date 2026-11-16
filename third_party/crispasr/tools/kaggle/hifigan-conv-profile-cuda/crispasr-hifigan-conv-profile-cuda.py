"""
CrispASR — melotts HiFi-GAN conv per-op profile on a CLEAN CUDA P100.

Question this kernel answers: on M1 Metal, the melotts HiFi-GAN decode graph runs
~37x slower than its own FLOP/bandwidth roofline (measured gpu_us ~4.0e6 unloaded
for ~2.6s audio, ideal ~110-140ms), and a per-op profile there showed IM2COL
dominating (~67% of real GPU op time) with MUL_MAT only ~10%. BUT those M1 numbers
were taken on a chronically-loaded box (load 46-136, parallel GPU session), so the
im2col dominance could be contention, not a real kernel problem. This kernel settles
it on a clean P100 (dev-guide's <1%-variance box):

  1. Does the SAME ggml conv graph (fork ggml_conv_1d = im2col -> F16->F32 cast ->
     mul_mat, K in {3,7,11}) run pathologically slow vs roofline on clean CUDA too?
  2. If so, does IM2COL dominate the CUDA per-kernel breakdown as well?

Decision:
  - hifigan_decode near roofline (<3x off) AND im2col small  -> M1 dominance was
    contention; the ideal roofline "fused conv not worth it" verdict STANDS.
  - hifigan_decode >> roofline AND im2col dominates on CUDA too -> genuine
    cross-backend im2col-materialization problem -> a fused direct-conv kernel
    (handover Section 4) is worth building. Cross-backend => shared value.

No runtime code changes: clones stock `main`, times hifigan_decode via
MELOTTS_BENCH, and gets the per-CUDA-kernel GPU breakdown via `nsys` (graceful
skip if nsys is absent — the wall-vs-roofline number alone answers Q1).
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

CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
REPS = int(os.environ.get("REPS", "5"))
TTS_TEXT = "Hello there, this is a vocoder op profile measurement."

# melotts registry model (cstr/melotts-en-v2-GGUF): decoder GGUF + BERT companion.
MELO_REPO = "cstr/melotts-en-v2-GGUF"
MELO_FILE = "melotts-en-v2-f16.gguf"
BERT_FILE = "bert-base-uncased-q4k.gguf"

# M1 GPU peak was 2.6 TF32 / 200 GB/s. P100: ~9.3 TFLOP FP32, ~732 GB/s HBM2.
P100_FLOPS = 9.3e12
P100_BW = 732e9


def roofline_ms(t_latent: int, flops: float, bw: float) -> dict:
    """HiFi-GAN ideal conv-FLOP + im2col-traffic ms for melotts config at t_latent."""
    ch = [512, 256, 128, 64, 32, 16]
    ups = [8, 8, 2, 2, 2]
    rk = [3, 7, 11]

    def t_after(i):
        t = t_latent
        for k in range(i + 1):
            t *= ups[k]
        return t

    flop = 2.0 * 192 * 512 * 7 * t_latent  # conv_pre
    im2col = 2 * (t_latent * 192 * 7 * 4)
    for s in range(5):
        cs, ts = ch[s + 1], t_after(s)
        for k in rk:
            for _ in range(6):
                flop += 2.0 * cs * cs * k * ts
                im2col += 2 * (ts * cs * k * 4)
    t5 = t_after(4)
    flop += 2.0 * 16 * 1 * 7 * t5
    im2col += 2 * (t5 * 16 * 7 * 4)
    return {
        "flop_ms": flop / flops * 1e3,
        "im2col_ms": im2col / bw * 1e3,
        "sum_ms": flop / flops * 1e3 + im2col / bw * 1e3,
    }


def sh(cmd, **kw):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, **kw)


# ── Pre-clone for shared harness ──────────────────────────────────────────
print(f"[pre-clone] cloning CrispASR @ {CRISPASR_REF}", flush=True)
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
cmake_cmd = (
    f"cmake {REPO} -B{BUILD} -GNinja -DCMAKE_BUILD_TYPE=Release "
    + " ".join(kh.cuda_build_flags()) + " " + " ".join(kh.cache_and_link_flags())
)
njobs = kh.safe_build_jobs(gpu=True)
with kh.build_heartbeat("cmake-configure"):
    kh.sh_with_progress(cmake_cmd)
with kh.build_heartbeat("cmake-build"):
    kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli -- -j{njobs}")
assert CRISPASR.is_file(), "crispasr binary missing after build"
kh.step("build.done", binary=str(CRISPASR))

# ── Download melotts + BERT ───────────────────────────────────────────────
kh.step("download.begin")
MODELS.mkdir(exist_ok=True)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
sh("pip install -q huggingface_hub hf_transfer", check=False)
from huggingface_hub import hf_hub_download  # noqa: E402

melo = hf_hub_download(MELO_REPO, MELO_FILE, local_dir=str(MODELS), local_dir_use_symlinks=False)
bert = hf_hub_download(MELO_REPO, BERT_FILE, local_dir=str(MODELS), local_dir_use_symlinks=False)
kh.step("download.done", melo=melo, bert=bert)


def run_once(text, env_extra=None, wrap=""):
    out = WORK / "melo.wav"
    cmd = (f"{wrap} {CRISPASR} --backend melotts -m {melo} --codec-model {bert} "
           f'--tts "{text}" --tts-output {out}').strip()
    env = {**os.environ, "MELOTTS_BENCH": "1", **(env_extra or {})}
    r = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True, timeout=1200)
    return (r.stdout or "") + (r.stderr or "")


# ── T-SCALING: is the off-roofline gap launch-bound (fixed 1198-node dispatch,
#    shrinks as T grows) or im2col-bandwidth-bound (persists/grows with T)? ──
# Force increasing audio length by repeating the sentence. If off-ideal DROPS
# toward ~2-3x at large T -> launch-bound (fix = fewer nodes, modest). If it STAYS
# high (~8x+) -> the conv/im2col kernel is the wall -> fused conv worth building.
BASE = "Hello there, this is a vocoder op profile measurement."
TEXTS = [("T1", BASE),
         ("T3", " ".join([BASE] * 3)),
         ("T6", " ".join([BASE] * 6)),
         ("T10", " ".join([BASE] * 10))]

kh.step("scaling.begin", reps=REPS, texts=len(TEXTS))
scaling = []
for label, text in TEXTS:
    hifi_ms, t_latent = [], None
    for rep in range(REPS):
        log = run_once(text)
        m = re.search(r"hifigan_decode\s+([\d.]+)\s*ms", log)
        if m:
            hifi_ms.append(float(m.group(1)))
        d = re.search(r"synthesized\s+([\d.]+)\s*s\s*\((\d+)\s*samples\s*@\s*(\d+)", log)
        if d:
            t_latent = int(round(int(d.group(2)) / 512))
        if rep == 0 and not hifi_ms:
            print(f"--- {label} rep0 stderr tail (no hifigan_decode parsed) ---", flush=True)
            for line in log.splitlines()[-30:]:
                print(line, flush=True)
    med = sorted(hifi_ms)[len(hifi_ms) // 2] if hifi_ms else None
    rl = roofline_ms(t_latent or 1, P100_FLOPS, P100_BW)
    off = round(med / rl["sum_ms"], 1) if (med and rl["sum_ms"]) else None
    row = dict(label=label, t_latent=t_latent, hifigan_ms=med,
               ideal_ms=round(rl["sum_ms"], 1), off_ideal_x=off)
    scaling.append(row)
    kh.step(f"scaling.{label}", **row)

# Use the LONGEST successful T for the roofline headline + nsys breakdown.
valid = [r for r in scaling if r["off_ideal_x"] is not None]
tail = valid[-1] if valid else None
hifi_med = tail["hifigan_ms"] if tail else None
t_latent = tail["t_latent"] if tail else None
off = tail["off_ideal_x"] if tail else None
long_text = TEXTS[[r["label"] for r in scaling].index(tail["label"])][1] if tail else BASE
off_trend = None
if len(valid) >= 2:
    off_trend = "DROPS(launch-bound)" if valid[-1]["off_ideal_x"] < 0.6 * valid[0]["off_ideal_x"] \
        else "PERSISTS(bandwidth/kernel-bound)"
kh.step("scaling.done", trend=off_trend,
        off_first=valid[0]["off_ideal_x"] if valid else None,
        off_last=valid[-1]["off_ideal_x"] if valid else None)

# ── Per-CUDA-kernel breakdown via nsys (hardened path search) ─────────────
im2col_pct = mul_mat_pct = None
nsys_bin = None
for cand in ["nsys",
             "/usr/local/cuda/bin/nsys",
             "/opt/nvidia/nsight-systems/*/bin/nsys",
             "/opt/nvidia/nsight-systems-cli/*/bin/nsys"]:
    hit = subprocess.run(f"ls {cand} 2>/dev/null | head -1", shell=True,
                         capture_output=True, text=True).stdout.strip()
    if hit:
        nsys_bin = hit
        break
if not nsys_bin:
    subprocess.run("apt-get install -y -q nsight-systems-cli 2>/dev/null || true", shell=True)
    hit = subprocess.run("ls /opt/nvidia/nsight-systems*/*/bin/nsys /usr/local/cuda/bin/nsys 2>/dev/null | head -1",
                         shell=True, capture_output=True, text=True).stdout.strip()
    nsys_bin = hit or None
have_nsys = nsys_bin is not None
if have_nsys:
    kh.step("nsys.begin", nsys=nsys_bin)
    rep = WORK / "hifi_prof"
    # Profile the LONGEST T (im2col-bandwidth regime, where the fused-conv lever matters).
    run_once(long_text, wrap=f"{nsys_bin} profile --force-overwrite true -o {rep} --stats=false")
    csv = subprocess.run(
        f"{nsys_bin} stats --report cuda_gpu_kern_sum --format csv {rep}.nsys-rep",
        shell=True, capture_output=True, text=True)
    text = csv.stdout or ""
    print(text[:4000], flush=True)
    # CSV cols: Time(%),Total Time(ns),Instances,Avg,...,Name
    rows = []
    for line in text.splitlines():
        parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts) >= 2 and re.match(r"[\d.]+$", parts[0]):
            try:
                pct = float(parts[0])
                name = parts[-1]
                rows.append((pct, name))
            except ValueError:
                pass
    agg = {}
    for pct, name in rows:
        n = name.lower()
        key = ("im2col" if "im2col" in n else
               "mul_mat" if ("mul_mat" in n or "gemm" in n or "mmv" in n or "cublas" in n) else
               "cpy/cast" if ("cpy" in n or "cast" in n or "dup" in n or "convert" in n) else
               "other")
        agg[key] = agg.get(key, 0.0) + pct
    im2col_pct = round(agg.get("im2col", 0.0), 1)
    mul_mat_pct = round(agg.get("mul_mat", 0.0), 1)
    kh.step("nsys.done", im2col_pct=im2col_pct, mul_mat_pct=mul_mat_pct,
            cpy_cast_pct=round(agg.get("cpy/cast", 0.0), 1), other_pct=round(agg.get("other", 0.0), 1))
else:
    kh.step("nsys.skip", reason="nsys not on PATH; wall-vs-roofline still answers Q1")

# ── Verdict ───────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print(f"SUMMARY — melotts HiFi-GAN conv on clean P100 ({sha[:8]})")
print("=" * 72)
print(f"  {'label':6} {'t_latent':>9} {'hifigan_ms':>11} {'ideal_ms':>9} {'off_x':>6}")
for r in scaling:
    print(f"  {r['label']:6} {str(r['t_latent']):>9} {str(r['hifigan_ms']):>11} "
          f"{str(r['ideal_ms']):>9} {str(r['off_ideal_x']):>6}")
print(f"  off-ideal trend       : {off_trend}")
if im2col_pct is not None:
    print(f"  per-kernel (longest T): IM2COL {im2col_pct}%  MUL_MAT {mul_mat_pct}%")

if not valid:
    verdict = "INCONCLUSIVE — hifigan_decode not parsed at any T (check rep0 tails)."
elif off_trend and off_trend.startswith("DROPS"):
    verdict = (f"LAUNCH-BOUND — off-ideal shrinks with T ({valid[0]['off_ideal_x']}x@T={valid[0]['t_latent']} "
               f"-> {valid[-1]['off_ideal_x']}x@T={valid[-1]['t_latent']}). The 1198-node dispatch, not im2col "
               f"bandwidth, dominates. Fused conv is the WRONG lever; node-count reduction is the modest one. "
               f"At realistic T the graph is closer to roofline than the loaded M1 (37x) implied.")
elif im2col_pct is not None and im2col_pct >= 40:
    verdict = (f"BUILD FUSED CONV — off-ideal PERSISTS at large T ({off}x@T={t_latent}) AND im2col={im2col_pct}% "
               f"of CUDA kernel time: genuine cross-backend im2col-materialization wall.")
elif im2col_pct is not None:
    verdict = (f"MIXED — off-ideal persists ({off}x) but im2col only {im2col_pct}% (mul_mat {mul_mat_pct}%). "
               f"Bottleneck is not im2col; investigate before a fused conv.")
else:
    verdict = (f"PERSISTS, NO BREAKDOWN — off-ideal stays {off}x at large T (bandwidth/kernel-bound likely) "
               f"but nsys unavailable. Port the CUDA per-op profiler to confirm im2col share before building.")
print(f"  VERDICT: {verdict}")
kh.step("verdict", text=verdict, off_ideal_x=off, im2col_pct=im2col_pct, trend=off_trend)
