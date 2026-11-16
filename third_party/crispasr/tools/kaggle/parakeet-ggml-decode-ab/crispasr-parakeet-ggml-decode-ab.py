"""
CrispASR — Transducer GPU decode: cblas vs ggml-graph A/B on P100 (§232)

Question this kernel answers (P100, the ONLY hardware where the win is
measurable — see LEARNINGS 29-30): does running the transducer decode's LSTM
predictor + joint head as ggml graphs on the GPU (shared core_rnnt_ggml, gated
per backend) beat the default CPU cblas_sgemv path on a real CUDA box?

Context: §232 measured CA decode at ~955 ms (parakeet) / ~2900 ms (nemotron) on
P100 (host-side cblas, GPU idle) vs transcribe.cpp's GPU decode. On M1 the gap
is invisible because Apple Accelerate's cblas is fast (decode ~60 ms) — the A/B
must run where the CPU BLAS is slow relative to the GPU. This kernel builds the
CUDA runtime from the feat branch and, for BOTH transducers, runs JFK N reps ×
{cblas default, ggml (<BACKEND>_GGML_DECODE=1)} with <BACKEND>_DECODE_TIMING=1,
reporting per-config decode ms + transcript parity + wall RTF + a verdict.

Acceptance: transcripts MUST be identical (correctness); ggml decode ms should be
< cblas decode ms to be worth flipping. If ggml is launch-bound (not faster) the
follow-up is a persistent-graph / in-graph-argmax variant — do NOT flip from
this result alone. parakeet + nemotron share core_rnnt_ggml, so the verdict
transfers between them.
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
SAMPLE = WORK / "jfk.wav"
CRISPASR = BUILD / "bin" / "crispasr"

CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
REPS = int(os.environ.get("REPS", "5"))
EXPECTED = "ask not what your country can do for you"


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

# ── Download models + JFK ─────────────────────────────────────────────────
kh.step("download.begin")
MODELS.mkdir(exist_ok=True)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
kh.sh_with_progress("pip install -q huggingface_hub hf_transfer")
from huggingface_hub import hf_hub_download  # noqa: E402

# (backend, HF repo, filename, GGML gate env, DECODE_TIMING env, decode-log regex)
BACKENDS = [
    ("parakeet", "cstr/parakeet-tdt-0.6b-v3-GGUF", "parakeet-tdt-0.6b-v3-q4_k.gguf",
     "PARAKEET_GGML_DECODE", "PARAKEET_DECODE_TIMING", r"parakeet: tdt_decode ([\d.]+) ms"),
    ("nemotron", "cstr/nemotron-3.5-asr-streaming-0.6b-GGUF", "nemotron-3.5-asr-streaming-0.6b-q4_k.gguf",
     "NEMOTRON_GGML_DECODE", "NEMOTRON_DECODE_TIMING", r"nemotron: rnnt_decode ([\d.]+) ms"),
]
model_paths = {}
for backend, repo, fname, _gate, _timing, _rx in BACKENDS:
    p = hf_hub_download(repo_id=repo, filename=fname, local_dir=str(MODELS), local_dir_use_symlinks=False)
    model_paths[backend] = Path(p)
    kh.step(f"download.{backend}.done", path=p, size_mib=Path(p).stat().st_size // (1 << 20))

subprocess.run(["cp", f"{REPO}/samples/jfk.wav", str(SAMPLE)], check=False)
assert SAMPLE.is_file(), "jfk.wav missing"
kh.step("download.done")


# ── Run one (backend, config) ─────────────────────────────────────────────
def run_cfg(backend: str, model_path: Path, timing_env: str, rx: str, label: str, env_extra: dict) -> dict:
    out_stem = WORK / f"{backend}-jfk-{label}"
    for ext in [".txt", ".srt"]:
        f = out_stem.with_suffix(ext)
        if f.exists():
            f.unlink()
    cmd = [
        str(CRISPASR),
        "-m", str(model_path),
        "--backend", backend,
        "-l", "en",
        "-f", str(SAMPLE),
        "-of", str(out_stem),
        "-otxt",
        "-np",
    ]
    env = {**os.environ, timing_env: "1", **env_extra}
    decs, wall_rtfs, text = [], [], ""
    for rep in range(REPS):
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
        log = (r.stdout or "") + (r.stderr or "")
        m = re.search(rx, log)
        if m:
            decs.append(float(m.group(1)))
        rtf = re.search(r"\(([\d.]+)x realtime\)", log)
        if rtf:
            wall_rtfs.append(float(rtf.group(1)))
        txt_path = out_stem.with_suffix(".txt")
        if txt_path.exists() and txt_path.stat().st_size > 0:
            text = txt_path.read_text().strip()
        if rep == 0 and (not decs or not text):
            print(f"--- {backend}/{label} rep0 stderr tail ---", flush=True)
            for line in log.splitlines()[-25:]:
                print(line, flush=True)
    best_dec = min(decs) if decs else None
    med_dec = sorted(decs)[len(decs) // 2] if decs else None
    best_rtf = max(wall_rtfs) if wall_rtfs else None
    ok = EXPECTED in text.lower()
    kh.step(f"run.{backend}.{label}.done", best_decode_ms=best_dec, median_decode_ms=med_dec,
            best_wall_rtf=best_rtf, text_ok=ok, n_reps=len(decs))
    return {"best_dec": best_dec, "med_dec": med_dec, "best_rtf": best_rtf, "text": text, "ok": ok}


kh.step("run.section.begin", reps=REPS)
results = {}
for backend, _repo, _fname, gate, timing_env, rx in BACKENDS:
    mp = model_paths[backend]
    cblas = run_cfg(backend, mp, timing_env, rx, "cblas", {})
    ggml = run_cfg(backend, mp, timing_env, rx, "ggml", {gate: "1"})
    parity = cblas["text"] == ggml["text"]
    speedup = None
    if cblas["best_dec"] and ggml["best_dec"] and ggml["best_dec"] > 0:
        speedup = round(cblas["best_dec"] / ggml["best_dec"], 2)
    verdict = "FLIP CANDIDATE" if (parity and speedup and speedup > 1.2) else (
        "NO WIN (keep cblas; try persistent/CUDA-graph)" if parity else "CORRECTNESS FAIL")
    results[backend] = dict(parity=parity, speedup=speedup, cblas=cblas, ggml=ggml, verdict=verdict)
    kh.step(f"summary.{backend}", parity=parity, cblas_decode_ms=cblas["best_dec"],
            ggml_decode_ms=ggml["best_dec"], decode_speedup=speedup,
            cblas_rtf=cblas["best_rtf"], ggml_rtf=ggml["best_rtf"], verdict=verdict)

# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print(f"SUMMARY — transducer decode cblas vs ggml GPU decode ({sha[:8]})")
print("=" * 72)
for backend, r in results.items():
    print(f"\n[{backend}]")
    print(f"  transcript parity : {'IDENTICAL' if r['parity'] else 'DIFFERENT (BUG!)'}")
    print(f"  cblas decode  (ms): best={r['cblas']['best_dec']}  median={r['cblas']['med_dec']}")
    print(f"  ggml  decode  (ms): best={r['ggml']['best_dec']}  median={r['ggml']['med_dec']}")
    print(f"  decode speedup    : {r['speedup']}x  (>1 = ggml faster)")
    print(f"  wall RTF          : cblas={r['cblas']['best_rtf']}x  ggml={r['ggml']['best_rtf']}x")
    print(f"  VERDICT           : {r['verdict']}")

kh._push_progress_to_hf(force=True)
kh.step("script.end", verdicts={b: r["verdict"] for b, r in results.items()})
