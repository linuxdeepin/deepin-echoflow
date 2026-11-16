# ─────────────────────────── cell 0 (markdown) ───────────────────────────
# # CrispASR — PR #406 SIMDCONV quiet-box CPU A/B + roundtrips
#
# Settles, on an UNCONTENDED Kaggle box, whether the opt-in packed-SIMD
# Conv1d path (merged from PR #406, default OFF) is a perf win with equal
# output — the prerequisite for a default flip per the A/B rule. The VPS
# could prove correctness (1-LSB waveforms, exact roundtrips) but not perf
# (shared load voided wall clocks). Also closes the CosyVoice3 coverage gap:
# that arm had only hermetic unit tests (no local model on the VPS).
#
# Matrix, per backend, seed 42, fixed text, arms INTERLEAVED back-to-back
# (1 warmup each, then OFF/ON alternating x N_MEASURED):
#   chatterbox-turbo  gate CRISPASR_S3GEN_SIMDCONV        metric cb_s3gen synthesize_total
#   cosyvoice3-tts    gate CRISPASR_COSYVOICE3_SIMDCONV   metric cosyvoice3 hift_vocoder
#
# Per backend: median stage ms per arm, speedup, engagement line
# (packed 72/72 + ISA), OFF-vs-ON waveform max|diff| (seeded), and a
# whisper-tiny ASR roundtrip of BOTH arms (proof-of-work: rc=0, non-trivial
# wav, word overlap — a crash or no-op must never mint a fake win, #81).
#
# CPU-only build (the gates are CPU paths; they self-disable on GPU) inside
# a GPU kernel — CPU-only workers get no internet (gotcha #3).
# Datasets: ${KAGGLE_ACCOUNT}/crispasr-hf-token, ${KAGGLE_ACCOUNT}/crispasr-ccache.

# ─────────────────────────── cell 1 (code) — config ──────────────────────
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import wave
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/tmp/simdconv-ab")  # models live on the big /tmp layer (~70 GB)
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
MODELS = TMP / "models"
RESULTS = WORK / "results"
for d in (MODELS, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

CRISPASR_REPO = os.environ.get("CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git")
CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")

TEXT = "The quick brown fox jumps over the lazy dog near the riverbank."
SEED = 42
N_MEASURED = int(os.environ.get("SIMDCONV_AB_N", "3"))

CV3_HF_REPO = "cstr/cosyvoice3-0.5b-2512-GGUF"
# gotcha #20: cosyvoice3 needs SIBLING files beside the LLM; -m auto only
# fetches llm+flow, the adapter then requires hift + voices by exact name.
CV3_FILES = [
    "cosyvoice3-llm-q4_k.gguf",
    "cosyvoice3-flow-q8_0.gguf",
    "cosyvoice3-hift-f16.gguf",
    "cosyvoice3-voices.gguf",
]


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


# ─────────────────────────── cell 2 (code) — clone + CPU build ────────────
step("start", ref=CRISPASR_REF)
if REPO.exists():
    shutil.rmtree(REPO)
run(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF, "--recursive", CRISPASR_REPO, str(REPO)])

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
step("cloned", sha=subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip())

kh.install_build_toolchain()
BUILD.mkdir(parents=True, exist_ok=True)
# CPU-only on purpose: SIMDCONV engages only when the vocoder is CPU-resident,
# and skipping nvcc keeps the build minutes-scale even on a ccache miss.
cmake_args = [
    "cmake", "-S", str(REPO), "-B", str(BUILD),
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=ON",
    "-DGGML_CUDA=OFF",
] + kh.crispasr_cmake_flags() + kh.cache_and_link_flags()
run(cmake_args)
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli -j{kh.safe_build_jobs(gpu=False)}")
step("build_done")

CLI = BUILD / "examples" / "cli" / "crispasr"
if not CLI.exists():
    cands = [c for c in BUILD.rglob("crispasr") if c.is_file() and os.access(c, os.X_OK)]
    if not cands:
        raise SystemExit("crispasr binary not found after build")
    CLI = cands[0]
os.environ["LD_LIBRARY_PATH"] = f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
os.environ["CRISPASR_CACHE_DIR"] = str(MODELS)
step("cli", path=str(CLI))

# ─────────────────────────── cell 3 (code) — models ───────────────────────
from huggingface_hub import hf_hub_download  # noqa: E402

token = kh.resolve_hf_token()
for f in CV3_FILES:
    p = hf_hub_download(repo_id=CV3_HF_REPO, filename=f, local_dir=str(MODELS), token=token or None)
    step("cv3_file", file=f, size_mb=round(os.path.getsize(p) / 1e6, 1))

# whisper tiny.en for the roundtrips
run(["bash", str(REPO / "models" / "download-ggml-model.sh"), "tiny.en", str(MODELS)])
WHISPER = MODELS / "ggml-tiny.en.bin"
if not WHISPER.exists():
    raise SystemExit("whisper tiny.en download failed")


def wav_dur(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as w:
            return round(w.getnframes() / w.getframerate(), 3)
    except Exception:
        return 0.0


def wav_samples(path: Path):
    import numpy as np
    with wave.open(str(path), "rb") as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return a.astype(np.float32) / 32768.0


def roundtrip(wav: Path) -> str:
    r = run([str(CLI), "-m", str(WHISPER), "-nt", "--no-prints", str(wav)], check=False)
    return " ".join(r.stdout.split())


def word_overlap(text: str, hyp: str) -> float:
    ref = {w.strip(".,!?;:").lower() for w in text.split()}
    hyp_w = {w.strip(".,!?;:").lower() for w in hyp.split()}
    return round(len(ref & hyp_w) / max(1, len(ref)), 3)


# ─────────────────────────── cell 4 (code) — the A/B ──────────────────────
BACKENDS = [
    {
        "name": "chatterbox-turbo",
        "cli": ["--backend", "chatterbox-turbo", "-m", "auto", "--auto-download"],
        "gate": "CRISPASR_S3GEN_SIMDCONV",
        "debug": "CRISPASR_S3GEN_SIMDCONV_DEBUG",
        "bench_env": "CRISPASR_CB_S3GEN_BENCH",
        "stage_re": re.compile(r"cb_s3gen_bench:\s+synthesize_total\s+([\d.]+)\s+ms"),
        "engage_re": re.compile(r"HiFT SIMDCONV ON: packed (\d+)/(\d+).*isa=([a-z0-9_]+)"),
        "stage_label": "s3gen synthesize_total",
    },
    {
        "name": "cosyvoice3-tts",
        "cli": ["--backend", "cosyvoice3-tts", "-m", str(MODELS / "cosyvoice3-llm-q4_k.gguf")],
        "gate": "CRISPASR_COSYVOICE3_SIMDCONV",
        "debug": "CRISPASR_COSYVOICE3_SIMDCONV_DEBUG",
        "bench_env": "CRISPASR_COSYVOICE3_BENCH",
        "stage_re": re.compile(r"cosyvoice3_bench:\s+hift_vocoder\s+([\d.]+)\s+ms"),
        "engage_re": re.compile(r"SIMDCONV ON: packed (\d+)/(\d+).*isa=([a-z0-9_]+)"),
        "stage_label": "hift_vocoder",
    },
]


def synth(be, arm: str, out_wav: Path, timeout=1800):
    env = {
        be["gate"]: "1" if arm == "on" else "0",
        be["debug"]: "1",
        be["bench_env"]: "1",
    }
    cmd = [str(CLI)] + be["cli"] + ["--seed", str(SEED), "--tts", TEXT,
                                    "--tts-output", str(out_wav), "-t", "4"]
    t0 = time.time()
    r = subprocess.run(cmd, env={**os.environ, **env}, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    wall = round(time.time() - t0, 2)
    m = be["stage_re"].search(r.stdout)
    eng = be["engage_re"].search(r.stdout)
    if r.returncode != 0 or wav_dur(out_wav) < 1.0:
        print(r.stdout[-4000:], flush=True)
        raise SystemExit(f"{be['name']} arm={arm}: rc={r.returncode} dur={wav_dur(out_wav)} — FAIL, not timed")
    return {
        "wall_s": wall,
        "stage_ms": float(m.group(1)) if m else None,
        "engaged": bool(eng),
        "packed": (eng.group(1) + "/" + eng.group(2)) if eng else None,
        "isa": eng.group(3) if eng else None,
    }


report = {"host_cpu": Path("/proc/cpuinfo").read_text().split("model name")[1].split("\n")[0].strip(": \t"),
          "n_measured": N_MEASURED, "backends": {}}

import numpy as np  # noqa: E402

for be in BACKENDS:
    name = be["name"]
    step("backend_start", backend=name)
    wav_off = RESULTS / f"{name}-off.wav"
    wav_on = RESULTS / f"{name}-on.wav"

    # warmup both arms (also produces the waveform-compare + roundtrip wavs)
    warm_off = synth(be, "off", wav_off)
    warm_on = synth(be, "on", wav_on)
    if not warm_on["engaged"]:
        raise SystemExit(f"{name}: SIMDCONV did not engage in the ON arm — nothing to measure")
    step("warmup", backend=name, off=warm_off, on=warm_on)

    # interleaved measured runs (identical machine state for both arms)
    offs, ons = [], []
    for i in range(N_MEASURED):
        offs.append(synth(be, "off", RESULTS / f"{name}-off-m{i}.wav"))
        ons.append(synth(be, "on", RESULTS / f"{name}-on-m{i}.wav"))
        step("measured_pair", backend=name, i=i, off_ms=offs[-1]["stage_ms"], on_ms=ons[-1]["stage_ms"])

    a = wav_samples(wav_off)
    b = wav_samples(wav_on)
    n = min(len(a), len(b))
    maxdiff = float(np.abs(a[:n] - b[:n]).max()) if n else None

    rt_off = roundtrip(wav_off)
    rt_on = roundtrip(wav_on)
    ov_off = word_overlap(TEXT, rt_off)
    ov_on = word_overlap(TEXT, rt_on)

    med_off = statistics.median([x["stage_ms"] for x in offs if x["stage_ms"]])
    med_on = statistics.median([x["stage_ms"] for x in ons if x["stage_ms"]])
    report["backends"][name] = {
        "stage": be["stage_label"],
        "median_off_ms": med_off,
        "median_on_ms": med_on,
        "speedup": round(med_off / med_on, 3) if med_on else None,
        "engagement": {"packed": warm_on["packed"], "isa": warm_on["isa"]},
        "wav_len_match": len(a) == len(b),
        "max_abs_diff": maxdiff,
        "roundtrip_overlap_off": ov_off,
        "roundtrip_overlap_on": ov_on,
        "roundtrip_on_text": rt_on[:300],
        "all_off_ms": [x["stage_ms"] for x in offs],
        "all_on_ms": [x["stage_ms"] for x in ons],
        "verdict_output_equal": bool(len(a) == len(b) and maxdiff is not None and maxdiff <= 2.0 / 32768.0),
        "verdict_roundtrip": bool(ov_off >= 0.8 and ov_on >= 0.8),
    }
    step("backend_done", backend=name, **{k: v for k, v in report["backends"][name].items()
                                          if not k.startswith("all_")})

(RESULTS / "simdconv-ab.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
step("done")
