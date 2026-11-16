# ─────────────────────────── cell 0 (markdown) ───────────────────────────
# # CrispASR — Voxtral-4B-TTS FM perf-gate A/B on a clean CUDA GPU
#
# Settles, on an UNCONTENDED CUDA GPU (T4/P100), whether the two opt-in
# flow-matching perf gates are wins — the M1 measurements were confounded
# by machine load (quiet-CPU vs loaded-GPU fabricated a false 2x).
#
# Matrix (backend voxtral-tts, Q4_K, seed 42, fixed text), each config run
# 1 warmup + 3 measured, median FM ms/frame reported:
#   A  GPU  8-step   (default / shipped)
#   B  FM_CPU 8-step (CRISPASR_VOXTRAL_TTS_FM_CPU=1)   — FM graph on CPU
#   C  GPU  7-step   (CRISPASR_VOXTRAL_TTS_FM_STEPS=7)
#   D  GPU  5-step   (CRISPASR_VOXTRAL_TTS_FM_STEPS=5)
#   E  FM_CPU 5-step (both)
#
# For each: median LLM/FM ms/frame, frame count, [END] reached, wav duration,
# whisper roundtrip. Question answered: on CUDA, is FM-on-CPU faster or slower
# than FM-on-GPU, and does reducing ODE steps perturb the frame trajectory?
#
# Requirements: Kaggle GPU (T4x1 or P100), Internet ON, ~8 GB disk.
# Secrets/datasets: ${KAGGLE_ACCOUNT}/crispasr-hf-token (public GGUF, token optional),
# ${KAGGLE_ACCOUNT}/crispasr-ccache (build cache).

# ─────────────────────────── cell 1 (code) — config ──────────────────────
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import wave
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
MODELS = WORK / "models"
RESULTS = WORK / "results"
for d in (MODELS, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

CRISPASR_REPO = os.environ.get("CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git")
CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")

HF_TTS_REPO = "cstr/voxtral-4b-tts-GGUF"
HF_TTS_FILE = "voxtral-4b-tts-q4_k.gguf"

TEXT = os.environ.get("VTTS_TEXT", "The quick brown fox jumps over the lazy dog.")
VOICE = "neutral_female"
SEED = 42
N_WARMUP = 1
N_MEASURED = 3


def step(name, **kv):
    print(f"[{time.strftime('%H:%M:%S')}] STEP {name} " + json.dumps(kv), flush=True)


def run(cmd, check=True, env=None, cwd=None, timeout=None):
    e = {**os.environ, **(env or {})}
    r = subprocess.run(cmd, env=e, cwd=cwd, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.stdout:
        print(r.stdout, flush=True)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(map(str, cmd))}")
    return r


# ─────────────────────────── cell 2 (code) — clone + build ───────────────
step("start", ref=CRISPASR_REF)
if REPO.exists():
    shutil.rmtree(REPO)
run(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF, "--recursive", CRISPASR_REPO, str(REPO)])

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
step("cloned", sha=subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip())

run(["nvidia-smi", "-L"])
gpu_name = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
step("gpu", gpu=gpu_name)

kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
step("cuda_arch", arch=arch)

BUILD.mkdir(parents=True, exist_ok=True)
cmake_args = [
    "cmake", "-S", str(REPO), "-B", str(BUILD),
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=ON",
] + kh.cuda_build_flags(arch) + kh.cache_and_link_flags()
run(cmake_args)
step("cmake_done")

with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli "
        f"-j{kh.safe_build_jobs(gpu=True)}")
step("build_done")

CLI = BUILD / "examples" / "cli" / "crispasr"
if not CLI.exists():
    cands = [c for c in BUILD.rglob("crispasr") if c.is_file() and os.access(c, os.X_OK)]
    if not cands:
        raise SystemExit("crispasr binary not found after build")
    CLI = cands[0]
os.environ["LD_LIBRARY_PATH"] = f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
step("cli", path=str(CLI))


# ─────────────────────────── cell 3 (code) — model + helpers ──────────────
def fetch_model():
    from huggingface_hub import hf_hub_download
    token = kh.resolve_hf_token()
    p = hf_hub_download(repo_id=HF_TTS_REPO, filename=HF_TTS_FILE, local_dir=str(MODELS),
                        token=token or None)
    step("model", path=p, size_mb=round(os.path.getsize(p) / 1e6, 1))
    return Path(p)


MODEL = fetch_model()

_TIMING_RE = re.compile(r"LLM-decode\s+([\d.]+)\s+ms/frame,\s+FM\s+([\d.]+)\s+ms/frame")
_FRAMES_RE = re.compile(r"generated\s+(\d+)\s+frames.*?\[(END|max_frames)\]")


def wav_dur(path: Path) -> float:
    if not path.exists():
        return 0.0
    try:
        with wave.open(str(path), "rb") as w:
            return round(w.getnframes() / w.getframerate(), 3)
    except Exception:
        return 0.0


def synth(tag: str, gate_env: dict, out_wav: Path, timeout=900) -> dict:
    env = {"CRISPASR_VOXTRAL_TTS_TIMING": "1", **gate_env}
    cmd = [str(CLI), "--backend", "voxtral-tts", "-m", str(MODEL), "--seed", str(SEED),
           "--tts", TEXT, "--voice", VOICE, "--tts-output", str(out_wav)]
    t0 = time.time()
    r = subprocess.run(cmd, env={**os.environ, **env}, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    dt = round(time.time() - t0, 2)
    out = r.stdout or ""
    (RESULTS / f"{tag}.log").write_text(out)
    tm = _TIMING_RE.search(out)
    fr = _FRAMES_RE.search(out)
    return {
        "rc": r.returncode,
        "wall_s": dt,
        "llm_ms": float(tm.group(1)) if tm else None,
        "fm_ms": float(tm.group(2)) if tm else None,
        "frames": int(fr.group(1)) if fr else None,
        "end": fr.group(2) if fr else None,
        "dur_s": wav_dur(out_wav),
    }


def asr_roundtrip(wav: Path, timeout=600) -> str:
    if not wav.exists():
        return ""
    cmd = [str(CLI), "--backend", "whisper", "-m", "auto", "--auto-download", "-f", str(wav), "--no-prints"]
    try:
        r = subprocess.run(cmd, timeout=timeout, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
        return " ".join(ln for ln in lines if not ln.startswith(("[", "whisper", "ggml", "load")))[-300:]
    except Exception as ex:  # noqa: BLE001
        return f"<asr-error: {type(ex).__name__}>"


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else round((xs[n // 2 - 1] + xs[n // 2]) / 2, 2)


# ─────────────────────────── cell 4 (code) — A/B matrix ───────────────────
CONFIGS = [
    ("A_gpu8", {}),
    # B_cpu8 / E_cpu5 removed: CRISPASR_VOXTRAL_TTS_FM_CPU was dropped from
    # the C++ side after this A/B concluded (current voxtral_tts.cpp reads
    # only TIMING/FM_STEPS/DEBUG/...), so those arms would silently run the
    # same GPU path as A/D and report a fabricated "CPU" result.
    ("C_gpu7", {"CRISPASR_VOXTRAL_TTS_FM_STEPS": "7"}),
    ("D_gpu5", {"CRISPASR_VOXTRAL_TTS_FM_STEPS": "5"}),
]

summary = {"gpu": gpu_name, "cuda_arch": arch, "text": TEXT, "seed": SEED, "configs": {}}
for tag, gate in CONFIGS:
    step("config_start", tag=tag, gate=gate)
    # warmup(s) — first run pays cuBLAS/kernel init; discard.
    for _ in range(N_WARMUP):
        synth(tag, gate, RESULTS / f"{tag}_warm.wav")
    runs = [synth(tag, gate, RESULTS / f"{tag}_{i}.wav") for i in range(N_MEASURED)]
    rt = asr_roundtrip(RESULTS / f"{tag}_0.wav")
    ok = [r for r in runs if r["rc"] == 0 and r["fm_ms"] is not None]
    summary["configs"][tag] = {
        "gate": gate,
        "fm_ms_median": median([r["fm_ms"] for r in ok]),
        "llm_ms_median": median([r["llm_ms"] for r in ok]),
        "fm_ms_runs": [r["fm_ms"] for r in ok],
        "frames": ok[0]["frames"] if ok else None,
        "end": ok[0]["end"] if ok else None,
        "dur_s": ok[0]["dur_s"] if ok else None,
        "wall_s_median": median([r["wall_s"] for r in ok]),
        "n_ok": len(ok),
        "roundtrip": rt,
    }
    step("config_done", tag=tag, **summary["configs"][tag])
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))


# ─────────────────────────── cell 5 (code) — table ────────────────────────
print("\n" + "=" * 78, flush=True)
print(f"VOXTRAL-TTS FM PERF GATES on {gpu_name} (arch {arch}) — median of {N_MEASURED} runs", flush=True)
print(f"text: {TEXT!r}  seed {SEED}", flush=True)
print("=" * 78, flush=True)
hdr = f"{'config':10} {'FM ms/fr':>9} {'LLM ms/fr':>10} {'frames':>7} {'end':>5} {'dur s':>6} {'roundtrip'}"
print(hdr, flush=True)
print("-" * 78, flush=True)
base = summary["configs"].get("A_gpu8", {}).get("fm_ms_median")
for tag, _ in CONFIGS:
    c = summary["configs"][tag]
    dl = ""
    if base and c["fm_ms_median"]:
        dl = f" ({(c['fm_ms_median'] / base - 1) * 100:+.0f}% FM)"
    print(f"{tag:10} {str(c['fm_ms_median']):>9} {str(c['llm_ms_median']):>10} "
          f"{str(c['frames']):>7} {str(c['end']):>5} {str(c['dur_s']):>6}  "
          f"{(c['roundtrip'] or '')[:40]}{dl}", flush=True)
print("=" * 78, flush=True)
print(json.dumps(summary, indent=2), flush=True)
(RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
step("DONE")
