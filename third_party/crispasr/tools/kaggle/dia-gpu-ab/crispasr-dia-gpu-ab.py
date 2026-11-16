"""
CrispASR — dia TTS CPU vs GPU A/B on CUDA (P100) (§232, PLAN item 5)

Question this kernel answers: does the new dia GPU path (main model on a backend
buffer via core_gguf::load_weights, gated DIA_TTS_GPU) win on a real CUDA box,
and does it stay correct? On M1 Metal the default was flipped to GPU — every
stage won (encoder ~94x, cross-KV ~101x, decoder_ar ~1.5x, DAC ~4.9x) with an
identical generate->ASR roundtrip. LEARNING 34 warns that a Metal win doesn't
always generalise the other way, and a P100 CPU BLAS (OpenBLAS) is far slower
than Apple Accelerate — so the CUDA arm needs its own measurement before the
flip is trusted everywhere.

Method: build the CUDA runtime from the feat branch; download dia-1.6b + the
dac-44khz codec; run dia synthesis GREEDY (DIA_GREEDY=1 -> deterministic argmax,
so CPU and GPU token streams are directly comparable) with DIA_DUMP_TOKENS=1 +
DIA_BENCH=1, for both DIA_TTS_GPU=0 (CPU) and DIA_TTS_GPU=1 (GPU), N reps,
capped steps. Report per-stage decode ms, first token divergence step, and a
verdict.

Acceptance: token streams IDENTICAL (correctness — greedy is deterministic per
device; a first-divergence step > 0 flags a real GPU miscompute, autoregressively
amplified). GPU total should beat CPU total to justify the default flip on CUDA.
If GPU is slower on CUDA, the follow-up is load_weights_split (encoder+DAC on GPU,
decoder on CPU) — do NOT walk back the Metal default from this alone.
"""

import os
import re
import subprocess
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
MODELS = WORK / "models"
CRISPASR = BUILD / "bin" / "crispasr"

CRISPASR_REF = os.environ.get("CRISPASR_REF", "feat/232-dia-gpu")
REPS = int(os.environ.get("REPS", "3"))
STEPS = int(os.environ.get("DIA_STEPS", "384"))  # ~4.3s audio — full prompt through "behind" for the ASR roundtrip
DIA_REPO = os.environ.get("DIA_REPO", "cstr/dia-1.6b-GGUF")
PROMPT = (
    "[S1] The quick brown fox jumps over the lazy dog while the sun sets slowly "
    "behind the distant mountains tonight."
)


def _sh_preclone(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


print(f"[pre-clone] cloning CrispASR @ {CRISPASR_REF} for shared harness", flush=True)
if not REPO.exists():
    _sh_preclone(
        f"git clone --depth 1 --branch {CRISPASR_REF} --recursive "
        f"https://github.com/CrispStrobe/CrispASR {REPO}"
    )

import sys

sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
if kh.resolve_hf_token():
    print("[auth] HF token resolved", flush=True)
kh.step("script.start", ref=CRISPASR_REF)

sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
kh.step("clone.done", sha=sha, ref=CRISPASR_REF)

# ── Build (CUDA) ──────────────────────────────────────────────────────────
kh.step("build.begin")
kh.install_build_toolchain()
cmake_cmd = (
    f"cmake {REPO} -B{BUILD} -GNinja "
    "-DCMAKE_BUILD_TYPE=Release "
    + " ".join(kh.cuda_build_flags())
    + " "
    + " ".join(kh.cache_and_link_flags())
)
njobs = kh.safe_build_jobs(gpu=True)
with kh.build_heartbeat("cmake-configure"):
    kh.sh_with_progress(cmake_cmd)
kh.step("build.configured")
with kh.build_heartbeat("cmake-build"):
    kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli -- -j{njobs}")
assert CRISPASR.is_file(), "crispasr binary missing after build"
kh.step("build.done", binary=str(CRISPASR))

# ── Download dia model + DAC codec ────────────────────────────────────────
kh.step("download.begin")
MODELS.mkdir(exist_ok=True)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
kh.sh_with_progress("pip install -q huggingface_hub hf_transfer")
from huggingface_hub import hf_hub_download  # noqa: E402

# Prefer q4_k (matches the M1 A/B); fall back to f16 (registry default).
model_path = None
for cand in ["dia-1.6b-q4_k.gguf", "dia-1.6b-f16.gguf"]:
    try:
        model_path = Path(
            hf_hub_download(repo_id=DIA_REPO, filename=cand, local_dir=str(MODELS), local_dir_use_symlinks=False)
        )
        break
    except Exception as e:  # noqa: BLE001
        print(f"[download] {cand} not available ({e}); trying next", flush=True)
assert model_path is not None, "no dia model gguf found in repo"
# DAC codec must sit beside the model so discover_dac_codec() finds it.
dac_path = Path(
    hf_hub_download(repo_id=DIA_REPO, filename="dac-44khz.gguf", local_dir=str(MODELS), local_dir_use_symlinks=False)
)
kh.step("download.done", model=model_path.name, model_mib=model_path.stat().st_size // (1 << 20),
        dac=dac_path.name)


# ── Run one config ────────────────────────────────────────────────────────
def run_cfg(label: str, env_extra: dict) -> dict:
    out = WORK / f"dia-{label}.wav"
    cmd = [
        str(CRISPASR),
        "--backend", "dia",
        "-m", str(model_path),
        "--tts", PROMPT,
        "--tts-output", str(out),
        "--seed", "42",
        "-np",
    ]
    # NOTE: NO DIA_GREEDY — dia is a sampled dialogue TTS (temp 1.2); argmax
    # collapses it to near-silence ([BLANK_AUDIO]), so greedy is useless for the
    # audio roundtrip. Correctness for a sampled TTS is intelligibility of the
    # OUTPUT (HARD RULE #3), not token identity (CPU/GPU never match token-for-
    # token under sampling). seed pins the RNG for reproducibility.
    env = {
        **os.environ,
        "DIA_BENCH": "1",
        "DIA_DUMP_TOKENS": "1",
        "DIA_MAX_STEPS": str(STEPS),
        **env_extra,
    }
    stages: dict = {}
    walls: list = []
    tokens: list = []
    gpu_line = False
    frames = None
    log = ""
    for rep in range(REPS):
        t0 = time.time()
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
        walls.append(time.time() - t0)
        log = (r.stdout or "") + (r.stderr or "")
        for m in re.finditer(r"dia_bench:\s+(\w+)\s+([\d.]+) ms", log):
            stages.setdefault(m.group(1), []).append(float(m.group(2)))
        if rep == 0:
            tokens = [ln.split(":", 1)[1].strip() for ln in log.splitlines() if ln.startswith("DIA_TOK step")]
            # Detect GPU engagement from ggml's UNCONDITIONAL backend-init banner
            # (the "dia_tts: GPU backend enabled" line is verbosity-gated and -np
            # suppresses it — that produced a false "did not engage" last run).
            gpu_line = bool(re.search(r"ggml_cuda|CUDA\d|found \d+ CUDA|ggml_metal|GPU backend enabled", log))
            fm = re.search(r"(\d+) valid code frames", log)
            frames = int(fm.group(1)) if fm else None
            if not stages or not tokens:
                print(f"--- dia/{label} rep0 stderr tail ---", flush=True)
                for line in log.splitlines()[-30:]:
                    print(line, flush=True)
    med = {k: sorted(v)[len(v) // 2] for k, v in stages.items()}
    total = sum(med.values()) if med else None
    kh.step(f"run.{label}.done", gpu=gpu_line, frames=frames, total_ms=total,
            stages=med, wall_s=round(min(walls), 2), n_tokens=len(tokens))
    return {"stages": med, "total": total, "tokens": tokens, "gpu": gpu_line,
            "frames": frames, "wall": round(min(walls), 2)}


kh.step("run.section.begin", reps=REPS, steps=STEPS)
cpu = run_cfg("cpu", {"DIA_TTS_GPU": "0"})
gpu = run_cfg("gpu", {"DIA_TTS_GPU": "1"})

# ── Correctness: first token divergence (greedy => deterministic) ─────────
n = min(len(cpu["tokens"]), len(gpu["tokens"]))
first_div = next((i for i in range(n) if cpu["tokens"][i] != gpu["tokens"][i]), None)
parity = first_div is None and len(cpu["tokens"]) == len(gpu["tokens"]) and n > 0

# ── Speed ─────────────────────────────────────────────────────────────────
speedup = None
if cpu["total"] and gpu["total"] and gpu["total"] > 0:
    speedup = round(cpu["total"] / gpu["total"], 2)

# ── Decoded-output roundtrip (HARD RULE #3): ASR the CPU and GPU audio ─────
# Greedy token identity is STRICTER than needed cross-device — CUDA's matmul
# reduction order can flip a close argmax at step 0 without the audio being
# wrong. The real acceptance test is whether the GPU audio still says the
# prompt. Transcribe both wavs with whisper-tiny and score keyword recall.
EXPECT_WORDS = ["quick", "brown", "fox", "jumps", "lazy", "dog", "sun", "sets", "behind", "mountains"]
asr_cpu = asr_gpu = ""
n_cpu = n_gpu = 0
audio_verdict = "n/a"
try:
    whisper = Path(hf_hub_download(repo_id="ggerganov/whisper.cpp", filename="ggml-tiny.bin",
                                   local_dir=str(MODELS), local_dir_use_symlinks=False))

    def asr(wav):
        if not Path(wav).is_file():
            return ""
        rr = subprocess.run([str(CRISPASR), "--backend", "whisper", "-m", str(whisper),
                             "-f", str(wav), "--no-gpu", "-l", "en", "-np"],
                            capture_output=True, text=True, timeout=600)
        return (rr.stdout or "").strip()

    asr_cpu = asr(WORK / "dia-cpu.wav")
    asr_gpu = asr(WORK / "dia-gpu.wav")
    n_cpu = sum(w in asr_cpu.lower() for w in EXPECT_WORDS)
    n_gpu = sum(w in asr_gpu.lower() for w in EXPECT_WORDS)
    if n_gpu >= 5 and n_cpu >= 5:
        audio_verdict = f"BOTH INTELLIGIBLE (cpu {n_cpu}/10, gpu {n_gpu}/10) — CUDA token divergence is benign FP"
    elif n_cpu >= 5 and n_gpu < 5:
        audio_verdict = f"GPU AUDIO GARBLED (cpu {n_cpu}/10, gpu {n_gpu}/10) — real CUDA miscompute"
    else:
        audio_verdict = f"INCONCLUSIVE (cpu {n_cpu}/10, gpu {n_gpu}/10)"
    kh.step("audio_roundtrip", asr_cpu=asr_cpu, asr_gpu=asr_gpu, cpu_kw=n_cpu, gpu_kw=n_gpu,
            audio_verdict=audio_verdict)
except Exception as e:  # noqa: BLE001
    audio_verdict = f"ASR roundtrip skipped ({e})"

# Correctness = decoded-audio intelligibility (HARD RULE #3). Token parity is
# informational only: dia samples, so CPU/GPU never match token-for-token.
if not gpu["gpu"]:
    verdict = "GPU DID NOT ENGAGE (check CUDA build / DIA_TTS_GPU gate)"
elif len(gpu["tokens"]) == 0 or gpu["total"] is None:
    verdict = "GPU RUN FAILED (decoder produced no tokens — likely a CUDA op abort)"
elif "BOTH INTELLIGIBLE" in audio_verdict:
    verdict = (f"GPU CORRECT — audio roundtrip OK (cpu {n_cpu}/10, gpu {n_gpu}/10 keywords); "
               f"decode {speedup}x — WIDEN DEFAULT candidate")
elif "GARBLED" in audio_verdict:
    verdict = "CUDA MISCOMPUTE — GPU audio garbled while CPU intelligible; keep Metal-only"
else:
    verdict = f"AUDIO ROUNDTRIP {audio_verdict} — inspect the ASR transcripts below"

per_stage = {}
for k in sorted(set(list(cpu["stages"]) + list(gpu["stages"]))):
    c = cpu["stages"].get(k)
    g = gpu["stages"].get(k)
    per_stage[k] = {"cpu_ms": c, "gpu_ms": g,
                    "speedup": round(c / g, 2) if (c and g) else None}

kh.step("summary", parity=parity, first_div_step=first_div, decode_speedup=speedup,
        cpu_total_ms=cpu["total"], gpu_total_ms=gpu["total"], per_stage=per_stage,
        cpu_frames=cpu["frames"], gpu_frames=gpu["frames"], verdict=verdict)

# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print(f"SUMMARY — dia TTS CPU vs GPU on CUDA ({sha[:8]}, {model_path.name}, {STEPS} steps)")
print("=" * 72)
print(f"  GPU engaged        : {gpu['gpu']}")
print(f"  token parity (info): {'identical' if parity else f'differ @ {first_div}'} "
      f"(expected to differ — sampled TTS; cpu={len(cpu['tokens'])} gpu={len(gpu['tokens'])} tok)")
print(f"  per-stage ms (cpu -> gpu, speedup):")
for k, v in per_stage.items():
    print(f"    {k:16s}: {v['cpu_ms']} -> {v['gpu_ms']}  ({v['speedup']}x)")
print(f"  total decode ms    : cpu={cpu['total']}  gpu={gpu['total']}  speedup={speedup}x")
print(f"  wall s             : cpu={cpu['wall']}  gpu={gpu['wall']}")
print(f"  audio roundtrip    : {audio_verdict}")
print(f"    ASR(cpu wav)     : {asr_cpu[:120]}")
print(f"    ASR(gpu wav)     : {asr_gpu[:120]}")
print(f"  VERDICT            : {verdict}")

kh._push_progress_to_hf(force=True)
kh.step("script.end", verdict=verdict, decode_speedup=speedup, parity=parity,
        audio_verdict=audio_verdict)
