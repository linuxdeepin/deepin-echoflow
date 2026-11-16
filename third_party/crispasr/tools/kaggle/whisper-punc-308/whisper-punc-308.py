"""
CrispASR — #308 verify: --backend whisper must not double-punctuate.

Bug (bilo1967): `--backend whisper --output-srt` produced subtitles with the
first word double-capitalised ("Hello" -> "HEllo") and a spurious full stop
appended. Cause: the whisper dispatch adapter lacked CAP_PUNCTUATION_NATIVE, so
the auto-punctuation policy ran FireRedPunc a SECOND time over whisper's already
cased+punctuated text. Fix cfbc082e adds the flag + hardens the FireRedPunc
capitaliser.

Method (single build of current main, which contains the fix):
  1. Build crispasr-cli from main.
  2. Download ggml-base.en.bin (whisper).
  3. Run jfk.wav two ways:
       A) dispatch path:   crispasr --backend whisper ...
       B) historical path: crispasr ...            (no --backend; never
          post-punctuated — the reference for correct output)
  4. Assert: (a) both non-empty, (b) NEITHER has a word starting with two
     uppercase letters (the "HEllo" signature), (c) A's text == B's text
     (the dispatch path no longer adds a second punctuation/caps pass).

Expected ~12-18 min on T4/P100 (build dominates).
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

WORK = Path("/kaggle/working")   # keep tiny (only progress.jsonl) — gotcha #22
TMP = Path("/kaggle/temp")       # repo clone + build + models live here
REPO = TMP / "CrispASR"
BUILD = TMP / "build"
CRISPASR_REPO = "https://github.com/CrispStrobe/CrispASR.git"
CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
# whisper base.en is public (no token) and small; the #308 bug is model-independent.
WHISPER_REPO = "ggerganov/whisper.cpp"
WHISPER_FILE = "ggml-base.en.bin"

_T0 = time.time()
_PROGRESS_PATH = WORK / "progress.jsonl"


def step(name, **kw):
    entry = {"t": round(time.time() - _T0, 1), "step": name, **kw}
    print(json.dumps(entry), flush=True)
    with open(_PROGRESS_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run(cmd, check=True, timeout=1800, cwd=None, env=None):
    r = subprocess.run(cmd, env={**os.environ, **(env or {})}, cwd=cwd, timeout=timeout)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed (rc={r.returncode}): {cmd}")
    return r


# ── Clone + build ──────────────────────────────────────────────────────────
step("start", ref=CRISPASR_REF)
if REPO.exists():
    import shutil
    shutil.rmtree(REPO)
run(["git", "clone", "--depth", "5", "--branch", CRISPASR_REF, "--recursive", CRISPASR_REPO, str(REPO)])
sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
# cfbc082e is the #308 fix; confirm it is in the built tree.
has_fix = subprocess.run(
    ["git", "-C", str(REPO), "merge-base", "--is-ancestor", "cfbc082e", "HEAD"]
).returncode == 0
step("cloned", sha=sha, has_308_fix=has_fix)
if not has_fix:
    print("WARNING: cfbc082e not in clone history (depth too shallow?) — result still valid for tip", flush=True)

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh

kh.init_progress()
gpu_name = ""
try:
    gpu_name = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip()
except Exception:
    pass
step("gpu", gpu_name=gpu_name)

kh.install_build_toolchain()
BUILD.mkdir(parents=True, exist_ok=True)
# CPU-only build: the #308 bug is text post-processing, and whisper base.en on an
# 11 s clip is trivial on CPU. The GPU worker is requested only for internet
# (Kaggle CPU workers have none). Skipping CUDA cuts build time substantially.
cmake_args = (
    ["cmake", "-S", str(REPO), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release",
     "-DBUILD_SHARED_LIBS=ON", "-DGGML_CUDA=OFF", "-DCRISPASR_OPUS_FETCH=ON"]
    + kh.cache_and_link_flags()
)
run(cmake_args)
step("cmake_done")
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli -j{kh.safe_build_jobs(gpu=True)}"
    )
step("build_done")

CLI = BUILD / "bin" / "crispasr"
if not CLI.exists():
    cands = [c for c in BUILD.rglob("crispasr") if c.is_file() and os.access(c, os.X_OK)]
    if not cands:
        raise SystemExit("crispasr binary not found after build")
    CLI = cands[0]
os.environ["LD_LIBRARY_PATH"] = f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
step("cli_found", path=str(CLI))

# ── Download whisper model ────────────────────────────────────────────────
MODEL_DIR = TMP / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
model_path = MODEL_DIR / WHISPER_FILE
if not model_path.exists():
    from huggingface_hub import hf_hub_download
    try:
        token = kh.resolve_hf_token()
    except Exception:
        token = None
    hf_hub_download(repo_id=WHISPER_REPO, filename=WHISPER_FILE, local_dir=str(MODEL_DIR), token=token)
step("model_downloaded", size_mb=round(model_path.stat().st_size / 1e6, 1))

AUDIO = REPO / "samples" / "jfk.wav"
if not AUDIO.exists():
    raise SystemExit(f"test audio not found: {AUDIO}")

# ── Run helper: capture the plain transcript from stdout ──────────────────
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def transcribe(label, extra_args):
    step(f"{label}_start", args=extra_args)
    cmd = [str(CLI), "-m", str(model_path), "-f", str(AUDIO), "-l", "en", "-t", "2", "--output-srt"] + extra_args
    r = subprocess.run(cmd, timeout=300, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=os.environ)
    if r.returncode != 0:
        print(f"  {label}: FAILED rc={r.returncode}\n{r.stderr[-2048:]}", flush=True)
    # whisper prints each segment as "[00:00:00.000 --> ...]   <text>" — take the
    # text AFTER the timestamp bracket (the previous version wrongly dropped these).
    segs = []
    for l in r.stdout.splitlines():
        s = ANSI.sub("", l).strip()
        if s.startswith("[") and "]" in s:
            segs.append(s.split("]", 1)[1].strip())
    text = " ".join(x for x in segs if x).strip()
    if not text:
        print(f"  {label}: EMPTY — raw stdout tail:\n{r.stdout[-1200:]}", flush=True)
    print(f"  {label}: {text[:160]!r}", flush=True)
    step(f"{label}_done", rc=r.returncode, nseg=len(segs), text=text)
    return {"label": label, "rc": r.returncode, "text": text}


# A) dispatch path (the one #308 reported); B) historical reference path.
res_dispatch = transcribe("dispatch_backend_whisper", ["--backend", "whisper"])
res_hist = transcribe("historical_no_backend", [])

# ── Assertions ────────────────────────────────────────────────────────────
DOUBLE_CAP = re.compile(r"\b[A-Z][A-Z][a-z]")  # "HEllo" signature
failures = []
for r in (res_dispatch, res_hist):
    if r["rc"] != 0:
        failures.append(f"{r['label']}: non-zero exit {r['rc']}")
    if not r["text"]:
        failures.append(f"{r['label']}: empty transcript")
    m = DOUBLE_CAP.search(r["text"])
    if m:
        failures.append(f"{r['label']}: double-capitalised word present ({m.group(0)!r}) — #308 signature")

if res_dispatch["text"] and res_hist["text"] and res_dispatch["text"] != res_hist["text"]:
    failures.append(
        "dispatch path text != historical path text (dispatch is still post-processing):\n"
        f"  dispatch:   {res_dispatch['text']!r}\n"
        f"  historical: {res_hist['text']!r}"
    )

print("\n=== #308 VERDICT ===", flush=True)
print(f"  dispatch  : {res_dispatch['text']!r}", flush=True)
print(f"  historical: {res_hist['text']!r}", flush=True)
if failures:
    for f in failures:
        print("FAIL: " + f, flush=True)
    step("FAILED", failures=failures)
    raise SystemExit(f"{len(failures)} assertion(s) failed")
print("\nPASS — no double-cap, dispatch path matches historical (no spurious post-punctuation).", flush=True)
step("PASSED", dispatch=res_dispatch["text"], historical=res_hist["text"])
