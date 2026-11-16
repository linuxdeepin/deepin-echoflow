#!/usr/bin/env python3
"""Kaggle CUDA A/B: FastConformer manual-attn + bucketed persistent graph (#81).

Branch feat/fc-unified-graph. Configs on parakeet-ctc-0.6b q8_0 (requantized,
native q8 conv pw), JFK 11 s + JFK×5 55 s, in-process Session, warm medians:

  base       — flash_attn_ext (falls back to CPU on CUDA: per-head mask)
  manual     — CRISPASR_FC_GPU_MANUAL_ATTN=1 (all-GPU attention)
  man+bucket — + CRISPASR_FC_BUCKET=500 (persistent graph / CUDA-graph replay)
  bucket     — CRISPASR_FC_BUCKET=500 alone

Acceptance: transcripts must equal base per clip. Timing: per-call series
(reveals cached-graph warm-up on call 2+).

Push ($KAGGLE_ACCOUNT): tools/kaggle/fc-unified-graph-ab/push.sh
"""

import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
BRANCH = "feat/fc-unified-graph"

print("=== Phase 0: clone + build ===", flush=True)
if not REPO.exists():
    subprocess.check_call(["git", "clone", "--depth", "1", "-b", BRANCH,
                           "https://github.com/CrispStrobe/CrispASR", str(REPO)])
if (REPO / "ggml").is_dir() and not (REPO / "ggml" / "CMakeLists.txt").exists():
    subprocess.check_call(["git", "submodule", "update", "--init", "ggml"], cwd=str(REPO))
sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh.init_progress(hf_progress_repo="cstr/crispasr-kaggle-progress")
step = kh.step
step("script.start", branch=BRANCH)
TOKEN = kh.resolve_hf_token("HF_TOKEN")

subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                       "huggingface_hub", "hf_transfer", "soundfile"])
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download  # noqa: E402

BUILD = TEMP / "build"
BUILD.mkdir(parents=True, exist_ok=True)
kh.install_build_toolchain()
import shutil  # noqa: E402

_ccache_run = TEMP / ".ccache"
_warmed = Path("/kaggle/working/.ccache")
if _warmed.exists():
    if _ccache_run.exists():
        shutil.rmtree(_ccache_run, ignore_errors=True)
    shutil.move(str(_warmed), str(_ccache_run))
else:
    _ccache_run.mkdir(parents=True, exist_ok=True)
os.environ["CCACHE_DIR"] = str(_ccache_run)

has_cuda = Path("/usr/local/cuda/bin/nvcc").exists()
step("build.begin", cuda=has_cuda)
flags = (kh.cuda_build_flags(kh.detect_cuda_arch()) if has_cuda else []) + kh.cache_and_link_flags()
subprocess.check_call(f"cmake -G Ninja -B {BUILD} -S {REPO} -DCMAKE_BUILD_TYPE=Release "
                      + " ".join(flags), shell=True)
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} -j {kh.safe_build_jobs(has_cuda)} "
                        f"--target crispasr-lib")
LIB = BUILD / "src" / "libcrispasr.so"
assert LIB.is_file()
step("build.done")

MODELS = TEMP / "models"
MODELS.mkdir(parents=True, exist_ok=True)
MODEL = hf_hub_download("cstr/parakeet-ctc-0.6b-GGUF", "parakeet-ctc-0.6b-q8_0.gguf",
                        local_dir=str(MODELS), token=TOKEN)
step("model.downloaded")

CHILD = r"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.join(sys.argv[1], "python"))
os.environ["CRISPASR_LIB_PATH"] = sys.argv[2]
import soundfile as sf
from crispasr import Session
pcm, sr = sf.read(os.path.join(sys.argv[1], "samples/jfk.wav"), dtype="float32")
clips = {"jfk11": pcm, "jfk55": np.concatenate([pcm] * 5)}
s = Session(sys.argv[3], n_threads=4)
out = {}
for name, audio in clips.items():
    times = []
    text = None
    for i in range(7):
        t0 = time.perf_counter()
        segs = s.transcribe(audio.copy(), language="en")
        times.append(round(time.perf_counter() - t0, 4))
        text = " ".join(seg.text for seg in segs)
    out[name] = {"times": times, "text": text}
print("RESULT::" + json.dumps(out))
"""

CONFIGS = [
    ("base", {}),
    ("manual", {"CRISPASR_FC_GPU_MANUAL_ATTN": "1"}),
    ("man+bucket", {"CRISPASR_FC_GPU_MANUAL_ATTN": "1", "CRISPASR_FC_BUCKET": "500"}),
    ("bucket", {"CRISPASR_FC_BUCKET": "500"}),
]

results = {}
for name, env_extra in CONFIGS:
    env = dict(os.environ, **env_extra)
    r = subprocess.run([sys.executable, "-c", CHILD, str(REPO), str(LIB), MODEL],
                       capture_output=True, text=True, timeout=3600, env=env)
    res = None
    for ln in r.stdout.splitlines():
        if ln.startswith("RESULT::"):
            res = json.loads(ln[8:])
    if res is None:
        step("config.FAIL", config=name, err=r.stderr[-400:].replace("\n", " / "))
        results[name] = {"error": r.stderr[-400:]}
        continue
    results[name] = res
    for clip, d in res.items():
        warm = d["times"][2:]
        step("config.done", config=name, clip=clip, median=statistics.median(warm),
             times=d["times"], text_ok=(results.get("base", {}).get(clip, {}).get("text") == d["text"]
                                        if name != "base" else True))

# transcripts must match base
verdict = {}
for name, _ in CONFIGS[1:]:
    if "error" in results.get(name, {"error": 1}):
        verdict[name] = "ERROR"
        continue
    ok = all(results[name][c]["text"] == results["base"][c]["text"] for c in ("jfk11", "jfk55"))
    verdict[name] = "TRANSCRIPTS-MATCH" if ok else "TRANSCRIPT-MISMATCH"
step("script.done", verdict=verdict)
(WORK / "results.json").write_text(json.dumps({"results": results, "verdict": verdict}, indent=1))
print("DONE", json.dumps(verdict), flush=True)
