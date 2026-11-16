"""
CrispASR — CPU-vs-GPU A/B for the newly GPU-enabled CPU-pinned backends on CUDA
(§232 audit follow-up to dia).

Backends covered — each was CPU-pinned and got an opt-in GPU path
(core_gguf::load_weights already put weights on ctx->backend; the fix pointed
that at a GPU backend + made the sched 2-backend):

  * paraformer  (non-AR ASR, SAN-M encoder + CIF)   gate CRISPASR_PARAFORMER_GPU
  * m2m100      (418M encoder-decoder MT)            gate CRISPASR_M2M100_GPU
  * madlad/t5   (3B T5 encoder-decoder MT)           gate CRISPASR_T5_GPU

Why CUDA: on M1 (Apple Accelerate cblas) these were roughly neutral — small
models / short sequences are launch-bound (LEARNING 34) and Accelerate is fast
(LEARNING 30). The question is whether a slow-CPU-BLAS CUDA box (P100/OpenBLAS)
shows the encoder win that the M1 hides — the same story as the parakeet decode
(LEARNING 30). Correctness must hold regardless.

Method: build CUDA; for each backend run its task CPU (gate unset) vs GPU (gate=1),
N reps, capture output text (parity) + wall time. Report parity + speedup + verdict.

Acceptance: output IDENTICAL cpu vs gpu (correctness — greedy/deterministic MT +
ASR). speedup>1.1 => a real CUDA win worth considering a default flip; otherwise
keep opt-in (still correct, just not faster here).
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
MT_TEXT = "The quick brown fox jumps over the lazy dog while the sun sets behind the mountains."


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
kh.step("clone.done", sha=sha)

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
kh.step("build.configured")
with kh.build_heartbeat("cmake-build"):
    kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli -- -j{njobs}")
assert CRISPASR.is_file(), "crispasr binary missing after build"
kh.step("build.done")

# ── Models ────────────────────────────────────────────────────────────────
kh.step("download.begin")
MODELS.mkdir(exist_ok=True)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
kh.sh_with_progress("pip install -q huggingface_hub hf_transfer")
from huggingface_hub import hf_hub_download  # noqa: E402


def fetch(repo: str, *cands: str) -> Path:
    for c in cands:
        try:
            return Path(hf_hub_download(repo_id=repo, filename=c, local_dir=str(MODELS),
                                        local_dir_use_symlinks=False))
        except Exception as e:  # noqa: BLE001
            print(f"[dl] {c} not in {repo} ({e})", flush=True)
    raise SystemExit(f"no model found in {repo}")


SAMPLE_ZH = REPO / "samples" / "paraformer_zh.wav"

# (name, gpu_env, [cmd args after the binary], needs_model_repo/cands)
CONFIGS = [
    dict(name="paraformer", gate="CRISPASR_PARAFORMER_GPU",
         repo="cstr/paraformer-zh-GGUF", cands=["paraformer-zh-q4_k.gguf", "paraformer-zh-q8_0.gguf"],
         args=lambda m: ["--backend", "paraformer", "-m", str(m), "-f", str(SAMPLE_ZH), "-np"],
         gpu_marker="paraformer: GPU backend enabled"),
    dict(name="m2m100", gate="CRISPASR_M2M100_GPU",
         repo="cstr/m2m100-418m-GGUF", cands=["m2m100-418m-q4_k.gguf", "m2m100-418m-q8_0.gguf"],
         args=lambda m: ["--backend", "m2m100", "-m", str(m), "--text", MT_TEXT, "-sl", "en", "-tl", "de", "--no-prints"],
         gpu_marker="m2m100: GPU backend enabled"),
    dict(name="madlad", gate="CRISPASR_T5_GPU",
         repo="cstr/madlad400-3b-mt-GGUF", cands=["madlad400-3b-mt-q4_k.gguf"],
         args=lambda m: ["--backend", "madlad", "-m", str(m), "--text", MT_TEXT, "-sl", "en", "-tl", "de", "--no-prints"],
         gpu_marker="t5: GPU backend enabled"),
]
for cfg in CONFIGS:
    cfg["model"] = fetch(cfg["repo"], *cfg["cands"])
    kh.step(f"download.{cfg['name']}.done", model=cfg["model"].name)
kh.step("download.done")


def run(cfg: dict, gpu: bool) -> dict:
    cmd = [str(CRISPASR)] + cfg["args"](cfg["model"])
    env = {**os.environ}
    if gpu:
        env[cfg["gate"]] = "1"
    else:
        env.pop(cfg["gate"], None)
    walls, out, engaged = [], "", False
    for rep in range(REPS):
        t0 = time.time()
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
        walls.append(time.time() - t0)
        log = (r.stdout or "") + (r.stderr or "")
        if rep == 0:
            out = (r.stdout or "").strip()
            # ggml's backend-init banner is unconditional; our per-backend
            # "GPU backend enabled" line is verbosity-gated (-np/--no-prints hide
            # it), so key off the banner to avoid a false "did not engage".
            engaged = bool(re.search(r"ggml_cuda|CUDA\d|found \d+ CUDA|ggml_metal|GPU backend enabled", log)) if gpu \
                else False
            if not out:
                print(f"--- {cfg['name']} gpu={gpu} rep0 tail ---", flush=True)
                for ln in log.splitlines()[-25:]:
                    print(ln, flush=True)
    return {"wall": round(min(walls), 3), "out": out, "engaged": engaged}


kh.step("run.section.begin", reps=REPS)
results = {}
for cfg in CONFIGS:
    cpu = run(cfg, gpu=False)
    gpu = run(cfg, gpu=True)
    parity = cpu["out"] == gpu["out"] and len(cpu["out"]) > 0
    speedup = round(cpu["wall"] / gpu["wall"], 2) if gpu["wall"] > 0 else None
    if not gpu["engaged"]:
        verdict = "GPU DID NOT ENGAGE"
    elif not parity:
        verdict = "CORRECTNESS FAIL (output differs)"
    elif speedup and speedup > 1.1:
        verdict = "GPU WINS ON CUDA — flip candidate"
    else:
        verdict = "correct but not faster — keep opt-in"
    results[cfg["name"]] = dict(parity=parity, speedup=speedup, cpu=cpu, gpu=gpu, verdict=verdict)
    kh.step(f"summary.{cfg['name']}", parity=parity, engaged=gpu["engaged"], speedup=speedup,
            cpu_wall=cpu["wall"], gpu_wall=gpu["wall"], verdict=verdict)

print("\n" + "=" * 72)
print(f"SUMMARY — CPU-pinned backends CPU vs GPU on CUDA ({sha[:8]})")
print("=" * 72)
for name, r in results.items():
    print(f"\n[{name}]")
    print(f"  gpu engaged  : {r['gpu']['engaged']}")
    print(f"  output parity: {'IDENTICAL' if r['parity'] else 'DIFFERENT (BUG!)'}")
    print(f"  wall s       : cpu={r['cpu']['wall']}  gpu={r['gpu']['wall']}  speedup={r['speedup']}x")
    print(f"  VERDICT      : {r['verdict']}")

kh._push_progress_to_hf(force=True)
kh.step("script.end", verdicts={n: r["verdict"] for n, r in results.items()})
