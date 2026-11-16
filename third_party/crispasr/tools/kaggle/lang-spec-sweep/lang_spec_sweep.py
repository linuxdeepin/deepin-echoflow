#!/usr/bin/env python3
# lang_spec_sweep.py — language-specification sweep for CrispASR ASR backends (Kaggle)
#
# Tests each ASR backend with correct and incorrect language flags to verify:
#   - rc=0 (no crash) with any valid language flag
#   - Non-empty output for correct language + matching audio
#   - Graceful handling of wrong-language flags (no crash, no silent hang)
#
# Pattern (disk-safe):
#   download model → run N test cases → record results → DELETE model → next model
#
# Resume:    LANG_SWEEP_START_FROM=backend_name  (skip everything before that name)
# CUDA/CPU:  LANG_SWEEP_BUILD=cuda  (default: cuda if GPU present, else cpu)
# HF upload: LANG_SWEEP_UPLOAD=1  (uploads results.json to cstr/crispasr-lang-sweep-results)
#
# Disk budget: 20 GB Kaggle scratch. Largest model pair: mimo-asr (4.3 GB) +
# tokenizer (1.2 GB) = 5.5 GB + 3 GB build = 8.5 GB peak. Safe.
#
# Time budget: ~2 h wall. 18 backends × (download 2 min + tests 1 min) = 54 min.

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Unbuffered I/O ───────────────────────────────────────────────────────────
os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

# ── Workspace ────────────────────────────────────────────────────────────────
WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
MODEL_TMP = WORK / "model_tmp"
RESULTS_PATH = WORK / "lang_results.json"
PROGRESS_PATH = WORK / "progress.jsonl"
AUDIO_DIR = WORK / "audio"

for d in (MODEL_TMP, AUDIO_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Config ───────────────────────────────────────────────────────────────────
START_FROM = os.environ.get("LANG_SWEEP_START_FROM", "").strip()
UPLOAD = os.environ.get("LANG_SWEEP_UPLOAD", "0") == "1"
CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")

# Auto-detect GPU
_gpu_present = bool(shutil.which("nvidia-smi") and
                    subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0)
BUILD_FLAVOUR = os.environ.get("LANG_SWEEP_BUILD", "cuda" if _gpu_present else "cpu")

# ── Progress + HF live push ──────────────────────────────────────────────────
_HF_PROGRESS_REPO = "cstr/crispasr-kaggle-progress"
_HF_PROGRESS_PATH = (
    f"runs/{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    f"-lang-sweep"
    f"-{os.environ.get('KAGGLE_KERNEL_REF','unknown').split('/')[-1] or 'unknown'}"
    f".jsonl"
)
_HF_PUSH_INTERVAL_S = 30.0
_HF_LAST_PUSH = 0.0
_T0 = time.time()


def _push_progress_to_hf(force: bool = False) -> None:
    global _HF_LAST_PUSH
    now = time.time()
    if not force and (now - _HF_LAST_PUSH) < _HF_PUSH_INTERVAL_S:
        return
    if not os.environ.get("HF_TOKEN"):
        return
    if not PROGRESS_PATH.exists():
        return
    try:
        from huggingface_hub import HfApi
        HfApi(token=os.environ["HF_TOKEN"]).upload_file(
            path_or_fileobj=str(PROGRESS_PATH),
            path_in_repo=_HF_PROGRESS_PATH,
            repo_id=_HF_PROGRESS_REPO,
            repo_type="dataset",
            commit_message=f"lang-sweep @ {now - _T0:.0f}s",
        )
        _HF_LAST_PUSH = now
    except Exception:
        pass


def step(name: str, **extra) -> None:
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - _T0, 2),
        "step": name,
        **extra,
    }
    try:
        PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PROGRESS_PATH.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass
    print(f"[{rec['elapsed_s']:>7.1f}s] {name}" +
          (f"  {extra}" if extra else ""), flush=True)
    _push_progress_to_hf()


# ── Build progress streamer ──────────────────────────────────────────────────
_BUILD_PROGRESS: dict = {"last_ninja": None, "last_tu": None, "lines": 0}
_NINJA_RE = re.compile(r"^\[(\d+)/(\d+)\]")
_TU_RE = re.compile(r"(\S+\.(?:cpp|cc|cxx|c|cu))(?::|\s|$)")


def sh_stream(cmd: str, cwd: Path | None = None) -> None:
    print(f"$ {cmd}", flush=True)
    proc = subprocess.Popen(
        cmd, shell=True, cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, text=True,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            _BUILD_PROGRESS["lines"] += 1
            m = _NINJA_RE.match(line)
            if m:
                _BUILD_PROGRESS["last_ninja"] = f"{m.group(1)}/{m.group(2)}"
            m = _TU_RE.search(line)
            if m:
                _BUILD_PROGRESS["last_tu"] = m.group(1).rsplit("/", 1)[-1]
    finally:
        rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


@contextlib.contextmanager
def heartbeat(label: str, interval_s: float = 30.0):
    t0 = time.time()
    stop_ev = threading.Event()
    def _tick():
        while not stop_ev.wait(interval_s):
            extra: dict = {"elapsed_in_block_s": round(time.time() - t0, 1)}
            if _BUILD_PROGRESS["last_ninja"]:
                extra["ninja"] = _BUILD_PROGRESS["last_ninja"]
                extra["tu"] = _BUILD_PROGRESS["last_tu"]
            step(f"{label}.heartbeat", **extra)
    t = threading.Thread(target=_tick, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop_ev.set()
        t.join(timeout=1.0)


# ── HF auth ──────────────────────────────────────────────────────────────────
def _token_from_dataset() -> str | None:
    candidates = [Path("/kaggle/input/crispasr-hf-token/hf_token.txt")]
    # Scan multiple roots — newer Kaggle environments mount datasets at
    # /kaggle/input/datasets/<owner>/<slug>/ instead of /kaggle/input/<slug>/
    roots = [
        Path("/kaggle/input"),
        Path("/kaggle/input/datasets"),
        Path("/kaggle/input/datasets/$KAGGLE_ACCOUNT"),
    ]
    for root in roots:
        if not root.exists():
            continue
        for sub in root.iterdir():
            if not sub.is_dir():
                continue
            if "hf-token" in sub.name or "hf_token" in sub.name:
                p = sub / "hf_token.txt"
                if p not in candidates:
                    candidates.append(p)
    for p in candidates:
        if p.exists():
            tok = p.read_text().strip()
            if tok and len(tok) > 8:
                print(f"HF_TOKEN from {p}", flush=True)
                return tok
    # Debug: dump /kaggle/input for diagnosis
    root = Path("/kaggle/input")
    if root.exists():
        dirs = sorted(root.iterdir())
        print(f"HF auth: /kaggle/input → {[d.name for d in dirs[:10]]}", flush=True)
    return None


step("script.start", mode="lang-spec-sweep", build=BUILD_FLAVOUR, gpu=_gpu_present,
     start_from=START_FROM or "(first)")

hf_token = os.environ.get("HF_TOKEN") or _token_from_dataset()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    print("HF auth: OK", flush=True)
else:
    print("HF auth: no token — downloads may fail for private repos", flush=True)

# ── Clone CrispASR ──────────────────────────────────────────────────────────
step("clone")
if not REPO.exists():
    subprocess.check_call(
        ["git", "clone", "--depth", "1", "--branch", CRISPASR_REF,
         "https://github.com/CrispStrobe/CrispASR.git", str(REPO)]
    )
else:
    subprocess.run(["git", "-C", str(REPO), "pull", "--ff-only"], check=False)

# ── Install build toolchain (ninja + ccache) ─────────────────────────────────
step("toolchain.install")
subprocess.run(
    "apt-get update -qq && apt-get install -y --no-install-recommends "
    "cmake ninja-build ccache 2>&1 | tail -3",
    shell=True, check=False,
)
_HAS_CCACHE = shutil.which("ccache") is not None
_HAS_NINJA  = shutil.which("ninja")  is not None
step("toolchain.done", ccache=_HAS_CCACHE, ninja=_HAS_NINJA)

# ── Build ────────────────────────────────────────────────────────────────────
step("build.configure", flavour=BUILD_FLAVOUR)
BUILD.mkdir(exist_ok=True)

# Warm ccache from dataset if present
_ccache_dir = WORK / ".ccache"
_ccache_dir.mkdir(parents=True, exist_ok=True)
for candidate in [
    Path("/kaggle/input/crispasr-ccache/ccache.tar"),
    Path("/kaggle/input/datasets/${KAGGLE_ACCOUNT}/crispasr-ccache/ccache.tar"),
]:
    if candidate.exists():
        step("ccache.warm", source=str(candidate))
        subprocess.run(["tar", "xf", str(candidate), "-C", str(WORK)], check=False)
        break

os.environ["CCACHE_DIR"] = str(_ccache_dir)
os.environ["CCACHE_MAXSIZE"] = "5G"

if BUILD_FLAVOUR == "cuda":
    # Add CUDA stubs to LIBRARY_PATH so FindCUDAToolkit can find libcuda.so.
    # Without this, cmake --generate fails with "CUDA::cuda_driver not found".
    _stubs = "/usr/local/cuda/lib64/stubs"
    if os.path.isdir(_stubs):
        os.environ["LIBRARY_PATH"] = f"{_stubs}:{os.environ.get('LIBRARY_PATH', '')}"
    # Detect CUDA arch; fall back to P100=60 / T4=75 if nvidia-smi fails.
    try:
        _arch_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True).strip().split("\n")[0].replace(".", "")
        _cuda_arch = _arch_out if _arch_out else "75"
    except Exception:
        _cuda_arch = "75"
    step("cuda_arch", arch=_cuda_arch)
    _nvcc = "/usr/local/cuda/bin/nvcc"
    cmake_extra = (
        f"-DGGML_CUDA=ON -DGGML_CUDA_NO_VMM=ON "
        f"-DCMAKE_CUDA_ARCHITECTURES={_cuda_arch}"
        + (f" -DCMAKE_CUDA_COMPILER={_nvcc}" if os.path.isfile(_nvcc) else "")
    )
else:
    cmake_extra = ""

# Only use ccache as launcher if it was actually installed.
_ccache_flags = (
    "-DCMAKE_C_COMPILER_LAUNCHER=ccache "
    "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
    "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache "
) if _HAS_CCACHE else ""

with heartbeat("build.configure"):
    sh_stream(
        f"cmake -G {'Ninja' if _HAS_NINJA else 'Unix Makefiles'} "
        f"-B {BUILD} -S {REPO} "
        f"-DCMAKE_BUILD_TYPE=Release -DCRISPASR_NO_C2PA_NATIVE=ON "
        f"{_ccache_flags}"
        f"{cmake_extra}"
    )

step("build.compile")
# Cap CUDA builds at -j2 to avoid OOM (nvcc TUs are ~2 GB RAM each on Kaggle).
_nproc = int(subprocess.check_output(["nproc"]).strip())
_build_j = "2" if BUILD_FLAVOUR == "cuda" else str(_nproc)
with heartbeat("build.compile", interval_s=30.0):
    sh_stream(f"cmake --build {BUILD} -j{_build_j}", cwd=BUILD)

CRISPASR_BIN = BUILD / "bin" / "crispasr"
assert CRISPASR_BIN.exists(), "crispasr binary not found after build"
step("build.done")

# ── Audio fixtures ───────────────────────────────────────────────────────────
# jfk.wav is shipped with the CrispASR repo itself (samples/jfk.wav).
# The JA audio is in the fixtures HF model repo (pinned full SHA).
step("fixtures.prepare")
from huggingface_hub import hf_hub_download

JFK_WAV = REPO / "samples" / "jfk.wav"
assert JFK_WAV.exists(), f"jfk.wav missing from cloned repo at {JFK_WAV}"

FIXTURES_REPO = "cstr/crispasr-regression-fixtures"
FIXTURES_REV  = "b61b03014bc99ecce18ac8f99988d5110c83f2d2"  # full SHA, repo_type=model

JA_WAV = Path(hf_hub_download(
    repo_id=FIXTURES_REPO,
    repo_type="model",
    filename="parakeet-tdt-0.6b-ja/reazon_baseball_14s/audio.wav",
    revision=FIXTURES_REV,
    local_dir=str(AUDIO_DIR),
    token=hf_token,
))
step("fixtures.done", jfk=str(JFK_WAV), ja=str(JA_WAV))

# ── Test plan ────────────────────────────────────────────────────────────────
# Each entry:
#   name       — backend entry name (used for resume + result key)
#   backend    — --backend flag value
#   gguf_repo  — HF repo
#   gguf_file  — filename in repo
#   gguf_rev   — pinned revision
#   size_mb    — approximate download size
#   extra_dl   — optional second file to download {repo,file,rev,flag} for --codec-model etc.
#   tests      — list of {audio, lang_flags, note, expect_nonempty, timeout_s}
#
# audio: "jfk" | "ja"
# lang_flags: list of CLI args, e.g. ["-l", "en"] or ["--source-lang", "en"]
# expect_nonempty: True = transcript must be non-empty; False = we just check no crash

LANG_TESTS = [
    # ── tiny/fast models first ──────────────────────────────────────────────
    {
        "name": "nemotron-3.5-asr-streaming-0.6b",
        "backend": "nemotron",
        "gguf_repo": "cstr/nemotron-3.5-asr-streaming-GGUF",
        "gguf_file": "nemotron-3.5-asr-streaming-0.6b-q4_k.gguf",
        "gguf_rev":  "bbd95a9ca5fa",
        "size_mb": 458,
        "tests": [
            {"audio": "jfk", "lang_flags": ["-l", "en"],
             "note": "nemotron: 39-lang, -l en on EN audio → correct EN",
             "expect_nonempty": True},
            {"audio": "jfk", "lang_flags": ["-l", "de"],
             "note": "nemotron: -l de on EN audio → some output, no crash",
             "expect_nonempty": False},
        ],
    },
    {
        "name": "qwen3-asr-0.6b",
        "backend": "qwen3",
        "gguf_repo": "cstr/qwen3-asr-0.6b-GGUF",
        "gguf_file": "qwen3-asr-0.6b-q4_k.gguf",
        "gguf_rev":  "ad086c22597e",
        "size_mb": 400,
        "tests": [
            {"audio": "jfk", "lang_flags": ["-l", "en"],
             "note": "qwen3-asr: -l en injects 'Transcribe in English.' → EN",
             "expect_nonempty": True},
            {"audio": "jfk", "lang_flags": ["-l", "de"],
             "note": "qwen3-asr: -l de injects 'Transcribe in German.' — ASR fine-tune "
                     "tends to transcribe source language anyway (EN here); no crash",
             "expect_nonempty": True},
        ],
    },
    {
        "name": "sensevoice-small",
        "backend": "sensevoice",
        "gguf_repo": "cstr/sensevoice-small-GGUF",
        "gguf_file": "sensevoice-small-q4_k.gguf",
        "gguf_rev":  "abbeaf54fdae",
        "size_mb": 470,
        "tests": [
            {"audio": "jfk", "lang_flags": ["-l", "en"],
             "note": "sensevoice: multilingual, -l en on EN audio",
             "expect_nonempty": True},
            {"audio": "ja",  "lang_flags": ["-l", "ja"],
             "note": "sensevoice: -l ja on JA audio → JA transcript",
             "expect_nonempty": True},
            {"audio": "jfk", "lang_flags": [],
             "note": "sensevoice: no -l (auto) on EN audio",
             "expect_nonempty": True},
        ],
    },
    {
        "name": "parakeet-tdt-0.6b-en",
        "backend": "parakeet",
        "gguf_repo": "cstr/parakeet-tdt-0.6b-v3-GGUF",
        "gguf_file": "parakeet-tdt-0.6b-v3-q4_k.gguf",
        "gguf_rev":  "815cc1bcb5cf",
        "size_mb": 466,
        "tests": [
            {"audio": "jfk", "lang_flags": ["-l", "en"],
             "note": "parakeet-en: EN-only, -l en → correct EN",
             "expect_nonempty": True},
            {"audio": "jfk", "lang_flags": ["-l", "de"],
             "note": "parakeet-en: EN-only, -l de → should warn but not crash, still produces EN",
             "expect_nonempty": True},
        ],
    },
    {
        "name": "parakeet-tdt-0.6b-ja",
        "backend": "parakeet",
        "gguf_repo": "cstr/parakeet-tdt-0.6b-ja-GGUF",
        "gguf_file": "parakeet-tdt-0.6b-ja.gguf",
        "gguf_rev":  "fd1963a153f0",
        "size_mb": 530,
        "tests": [
            {"audio": "ja",  "lang_flags": ["-l", "ja"],
             "note": "parakeet-ja: JA-only, -l ja on JA audio → JA transcript",
             "expect_nonempty": True},
            {"audio": "jfk", "lang_flags": ["-l", "en"],
             "note": "parakeet-ja: JA-only, -l en on EN audio → should not crash (may produce garbage)",
             "expect_nonempty": False},
        ],
    },
    {
        "name": "paraformer-zh",
        "backend": "paraformer",
        "gguf_repo": "cstr/paraformer-zh-GGUF",
        "gguf_file": "paraformer-zh-q8_0.gguf",
        "gguf_rev":  "86ea220d4110",
        "size_mb": 226,
        "tests": [
            {"audio": "jfk", "lang_flags": ["-l", "zh"],
             "note": "paraformer-zh: ZH-only, -l zh on EN audio → likely empty/garbage, no crash",
             "expect_nonempty": False},
            {"audio": "jfk", "lang_flags": ["-l", "en"],
             "note": "paraformer-zh: ZH-only, -l en (unsupported) → no crash",
             "expect_nonempty": False},
        ],
    },
    {
        "name": "kyutai-stt-1b",
        "backend": "kyutai-stt",
        "gguf_repo": "cstr/kyutai-stt-1b-GGUF",
        "gguf_file": "kyutai-stt-1b-q4_k.gguf",
        "gguf_rev":  "e58ef4fec3fa",
        "size_mb": 700,
        "tests": [
            {"audio": "jfk", "lang_flags": ["-l", "en"],
             "note": "kyutai-stt: EN-only, -l en → correct EN",
             "expect_nonempty": True},
            {"audio": "jfk", "lang_flags": ["-l", "de"],
             "note": "kyutai-stt: EN-only, -l de (ignored in adapter) → still EN, no crash",
             "expect_nonempty": True},
        ],
    },
    {
        "name": "omniasr-llm-300m",
        "backend": "omniasr-llm",
        "gguf_repo": "cstr/omniasr-llm-300m-v2-GGUF",
        "gguf_file": "omniasr-llm-300m-v2-q4_k.gguf",
        "gguf_rev":  "9c3d29db4175",
        "size_mb": 1100,
        "tests": [
            {"audio": "jfk", "lang_flags": ["-l", "en"],
             "note": "omniasr-llm: multilingual, -l en",
             "expect_nonempty": True},
            {"audio": "jfk", "lang_flags": ["-l", "de"],
             "note": "omniasr-llm: multilingual, -l de on EN audio → no crash",
             "expect_nonempty": False},
        ],
    },
    {
        "name": "funasr-mlt-nano",
        "backend": "funasr",
        "gguf_repo": "cstr/funasr-mlt-nano-GGUF",
        "gguf_file": "funasr-mlt-nano-2512-f16.gguf",
        "gguf_rev":  "5ba2e21029b7",
        "size_mb": 1980,
        "tests": [
            {"audio": "jfk", "lang_flags": ["-l", "en"],
             "note": "funasr-mlt: EN+ZH, -l en on EN audio",
             "expect_nonempty": True},
            {"audio": "jfk", "lang_flags": ["-l", "zh"],
             "note": "funasr-mlt: EN+ZH, -l zh on EN audio → forced ZH decode, no crash",
             "expect_nonempty": False},
            {"audio": "jfk", "lang_flags": [],
             "note": "funasr-mlt: no -l (auto) on EN audio",
             "expect_nonempty": True},
        ],
    },
    {
        "name": "cohere-transcribe",
        "backend": "cohere",
        "gguf_repo": "cstr/cohere-transcribe-03-2026-GGUF",
        "gguf_file": "cohere-transcribe-q4_k.gguf",
        "gguf_rev":  "2242638d5dfecc6f1dbe6c3a8713b97deb2e150f",
        "size_mb": 1500,
        "tests": [
            {"audio": "jfk", "lang_flags": ["-l", "en"],
             "note": "cohere: multilingual, -l en → correct EN",
             "expect_nonempty": True},
            {"audio": "jfk", "lang_flags": ["-l", "de"],
             "note": "cohere: multilingual, -l de on EN audio → no crash",
             "expect_nonempty": False},
        ],
    },
    {
        "name": "canary-1b-v2",
        "backend": "canary",
        "gguf_repo": "cstr/canary-1b-v2-GGUF",
        "gguf_file": "canary-1b-v2.gguf",
        "gguf_rev":  "b3715a517928",
        "size_mb": 1900,
        "tests": [
            {"audio": "jfk", "lang_flags": ["-l", "en"],
             "note": "canary: EN/DE/FR/ES, -l en on EN audio",
             "expect_nonempty": True},
            {"audio": "jfk", "lang_flags": ["-l", "de"],
             "note": "canary: -l de on EN audio → forces DE decode, likely garbled, no crash",
             "expect_nonempty": False},
            {"audio": "jfk", "lang_flags": ["-l", "fr"],
             "note": "canary: -l fr on EN audio → forces FR decode, no crash",
             "expect_nonempty": False},
        ],
    },
    {
        "name": "gemma4-e2b-it",
        "backend": "gemma4-e2b",
        "gguf_repo": "cstr/gemma4-e2b-it-GGUF",
        "gguf_file": "gemma4-e2b-it-q4_k.gguf",
        "gguf_rev":  "0afc0beb5bff",
        "size_mb": 2500,
        "tests": [
            {"audio": "jfk", "lang_flags": ["--source-lang", "en"],
             "note": "gemma4-e2b: uses --source-lang, not -l; test EN",
             "expect_nonempty": True},
            {"audio": "jfk", "lang_flags": ["-l", "en"],
             "note": "gemma4-e2b: -l en (falls through to params.language as src fallback)",
             "expect_nonempty": True},
            {"audio": "jfk", "lang_flags": ["--source-lang", "de"],
             "note": "gemma4-e2b: --source-lang de on EN audio → no crash",
             "expect_nonempty": False, "timeout_s": 90},
        ],
    },
    {
        "name": "voxtral-mini-3b-2507",
        "backend": "voxtral",
        "gguf_repo": "cstr/voxtral-mini-3b-2507-GGUF",
        "gguf_file": "voxtral-mini-3b-2507-q4_k.gguf",
        "gguf_rev":  "7a6ffdc7ff9e",
        "size_mb": 2529,
        "tests": [
            {"audio": "jfk", "lang_flags": ["-l", "en"],
             "note": "voxtral: multilingual, -l en → correct EN",
             "expect_nonempty": True},
            {"audio": "jfk", "lang_flags": ["-l", "de"],
             "note": "voxtral: multilingual, -l de on EN audio → no crash",
             "expect_nonempty": False},
        ],
    },
    {
        "name": "granite-speech-4.1-2b",
        "backend": "granite",
        "gguf_repo": "cstr/granite-speech-4.1-2b-GGUF",
        "gguf_file": "granite-speech-4.1-2b-q4_k.gguf",
        "gguf_rev":  "7ee888f67b68",
        "size_mb": 2806,
        "tests": [
            {"audio": "jfk", "lang_flags": ["-l", "en"],
             "note": "granite: -l en injects 'transcribe in English' → EN transcript",
             "expect_nonempty": True},
            {"audio": "jfk", "lang_flags": ["-l", "de"],
             "note": "granite: -l de injects 'transcribe in German' → HONORS it, "
                     "emits German translation (validated v6)",
             "expect_nonempty": True},
            {"audio": "jfk", "lang_flags": [],
             "note": "granite: no -l → baseline EN",
             "expect_nonempty": True},
        ],
    },
    {
        "name": "glm-asr-nano",
        "backend": "glm-asr",
        "gguf_repo": "cstr/glm-asr-nano-GGUF",
        "gguf_file": "glm-asr-nano.gguf",
        "gguf_rev":  "0eed3da3582c",
        "size_mb": 4300,
        "tests": [
            {"audio": "jfk", "lang_flags": ["-l", "en"],
             "note": "glm-asr: ZH+EN, -l en injects 'Please transcribe in English.' → EN",
             "expect_nonempty": True},
            {"audio": "jfk", "lang_flags": ["-l", "zh"],
             "note": "glm-asr: -l zh injects 'Please transcribe in Chinese.' — on EN audio "
                     "the model transcribes the source (EN); no crash",
             "expect_nonempty": True},
            {"audio": "jfk", "lang_flags": [],
             "note": "glm-asr: no -l → baseline",
             "expect_nonempty": True},
        ],
    },
    {
        "name": "moss-audio-4b-instruct",
        "backend": "moss-audio",
        "gguf_repo": "cstr/MOSS-Audio-4B-Instruct-GGUF",
        "gguf_file": "moss-audio-4b-instruct-q4_k.gguf",
        "gguf_rev":  "dbc2e72233d4",
        "size_mb": 3920,
        "tests": [
            {"audio": "jfk", "lang_flags": ["-l", "en"],
             "note": "moss-audio: multilingual, -l en injects 'Transcribe this audio in English.'",
             "expect_nonempty": True},
            {"audio": "jfk", "lang_flags": ["-l", "de"],
             "note": "moss-audio: -l de injects 'Transcribe this audio in German.' → HONORS it, "
                     "emits fluent German (validated v6)",
             "expect_nonempty": True},
        ],
    },
    {
        "name": "mimo-asr",
        "backend": "mimo-asr",
        "gguf_repo": "cstr/mimo-asr-GGUF",
        "gguf_file": "mimo-asr-q4_k.gguf",
        "gguf_rev":  "4c6a0ede0874",
        "size_mb": 4308,
        "extra_dl": {
            "repo": "cstr/mimo-audio-tokenizer-GGUF",
            "file": "mimo-tokenizer.gguf",
            "rev":  "e2e4a8dec8fd",
            "cli_flag": "--codec-model",
        },
        "size_mb_extra": 1240,
        "tests": [
            {"audio": "jfk", "lang_flags": ["-l", "en"],
             "note": "mimo-asr: Qwen2 LLM, -l en injects 'Please transcribe this audio in English.' → EN",
             "expect_nonempty": True},
            {"audio": "jfk", "lang_flags": ["-l", "zh"],
             "note": "mimo-asr: -l zh injects 'Please transcribe this audio in Chinese.' — on EN audio "
                     "the model transcribes the source (EN); no crash",
             "expect_nonempty": True},
        ],
    },
]

# ── Result storage ────────────────────────────────────────────────────────────
all_results: list[dict] = []

# ── Per-backend test runner ──────────────────────────────────────────────────
def run_one_backend(entry: dict) -> dict:
    name = entry["name"]
    step(f"{name}.start", size_mb=entry["size_mb"])

    model_dir = MODEL_TMP / name
    shutil.rmtree(model_dir, ignore_errors=True)
    model_dir.mkdir(parents=True)

    # ── Download main GGUF ──────────────────────────────────────────────────
    try:
        step(f"{name}.download")
        model_path = Path(hf_hub_download(
            repo_id=entry["gguf_repo"],
            filename=entry["gguf_file"],
            revision=entry["gguf_rev"],
            local_dir=str(model_dir),
            token=hf_token,
        ))
        step(f"{name}.download.done", path=model_path.name)
    except Exception as exc:
        step(f"{name}.download.fail", error=str(exc))
        shutil.rmtree(model_dir, ignore_errors=True)
        return {"name": name, "error": f"download failed: {exc}", "tests": []}

    # ── Download extra GGUF (codec model etc.) ──────────────────────────────
    extra_dl = entry.get("extra_dl")
    extra_path: Path | None = None
    extra_flag: str | None = None
    if extra_dl:
        try:
            step(f"{name}.download.extra")
            extra_path = Path(hf_hub_download(
                repo_id=extra_dl["repo"],
                filename=extra_dl["file"],
                revision=extra_dl["rev"],
                local_dir=str(model_dir),
                token=hf_token,
            ))
            extra_flag = extra_dl["cli_flag"]
            step(f"{name}.download.extra.done", path=extra_path.name)
        except Exception as exc:
            step(f"{name}.download.extra.fail", error=str(exc))
            # Continue — test may still be useful to check crash behaviour

    # ── Run each test case ──────────────────────────────────────────────────
    test_results = []
    audios = {"jfk": JFK_WAV, "ja": JA_WAV}

    for ti, test in enumerate(entry["tests"]):
        audio_key = test.get("audio", "jfk")
        audio_path = audios[audio_key]
        lang_flags = test.get("lang_flags", [])
        note = test.get("note", "")
        expect_nonempty = test.get("expect_nonempty", True)
        timeout_s = test.get("timeout_s", 120)

        tag = f"{name}.test{ti}"
        step(f"{tag}.run", lang=" ".join(lang_flags) or "(none)", audio=audio_key)

        cmd = [str(CRISPASR_BIN), "-f", str(audio_path), "-m", str(model_path),
               "--backend", entry["backend"], "--no-prints"]
        if extra_path and extra_flag:
            cmd += [extra_flag, str(extra_path)]
        cmd += lang_flags

        t0_test = time.time()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_s
            )
            elapsed = round(time.time() - t0_test, 2)
            transcript = proc.stdout.strip()
            stderr_tail = proc.stderr.strip()[-400:] if proc.stderr.strip() else ""
            ok = proc.returncode == 0 and (not expect_nonempty or bool(transcript))
            tr = {
                "test_idx": ti,
                "audio": audio_key,
                "lang_flags": lang_flags,
                "note": note,
                "rc": proc.returncode,
                "transcript": transcript[:300],
                "transcript_len": len(transcript),
                "stderr_tail": stderr_tail,
                "elapsed_s": elapsed,
                "expect_nonempty": expect_nonempty,
                "ok": ok,
            }
            status = "PASS" if ok else ("FAIL(empty)" if proc.returncode == 0 and not transcript else "FAIL(crash)")
            step(f"{tag}.done", status=status, rc=proc.returncode,
                 transcript_len=len(transcript), elapsed_s=elapsed)
            print(f"  [{status}] lang={' '.join(lang_flags) or 'none'!r} "
                  f"audio={audio_key} rc={proc.returncode} "
                  f"len={len(transcript)} {elapsed:.1f}s", flush=True)
            if transcript:
                print(f"    → {transcript[:120]!r}", flush=True)
            if not ok and stderr_tail:
                print(f"    stderr: {stderr_tail[-200:]}", flush=True)
        except subprocess.TimeoutExpired:
            elapsed = round(time.time() - t0_test, 2)
            tr = {
                "test_idx": ti, "audio": audio_key, "lang_flags": lang_flags,
                "note": note, "rc": -1, "transcript": "", "transcript_len": 0,
                "stderr_tail": "", "elapsed_s": elapsed,
                "expect_nonempty": expect_nonempty, "ok": False,
                "error": f"timeout after {timeout_s}s",
            }
            step(f"{tag}.timeout", timeout_s=timeout_s)
            print(f"  [TIMEOUT] lang={' '.join(lang_flags)!r} after {timeout_s}s", flush=True)

        test_results.append(tr)

    # ── Delete model to free disk ────────────────────────────────────────────
    step(f"{name}.cleanup")
    shutil.rmtree(model_dir, ignore_errors=True)

    n_pass = sum(1 for t in test_results if t["ok"])
    n_total = len(test_results)
    step(f"{name}.done", pass_rate=f"{n_pass}/{n_total}")

    return {
        "name": name,
        "backend": entry["backend"],
        "size_mb": entry["size_mb"],
        "tests": test_results,
        "pass_rate": f"{n_pass}/{n_total}",
    }


# ── Main loop ────────────────────────────────────────────────────────────────
step("sweep.start", n_backends=len(LANG_TESTS), start_from=START_FROM or "(first)")

started = not bool(START_FROM)
skipped = 0

for entry in LANG_TESTS:
    name = entry["name"]
    if not started:
        if name == START_FROM:
            started = True
        else:
            print(f"[skip] {name} (before START_FROM={START_FROM})", flush=True)
            skipped += 1
            continue

    result = run_one_backend(entry)
    all_results.append(result)

    # Write incremental results after each backend
    RESULTS_PATH.write_text(json.dumps({
        "sweep": "lang-spec-sweep",
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "crispasr_ref": CRISPASR_REF,
        "build_flavour": BUILD_FLAVOUR,
        "results": all_results,
    }, indent=2))
    _push_progress_to_hf(force=True)

step("sweep.done", n_completed=len(all_results), skipped=skipped)

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("LANGUAGE SPEC SWEEP SUMMARY", flush=True)
print("=" * 70, flush=True)
for r in all_results:
    if "error" in r:
        print(f"  ERROR   {r['name']:40s} {r['error']}", flush=True)
        continue
    for t in r["tests"]:
        status = "PASS" if t["ok"] else "FAIL"
        lang = " ".join(t["lang_flags"]) or "(none)"
        print(f"  {status:4s}  {r['name']:40s} lang={lang:12s} "
              f"rc={t['rc']} len={t['transcript_len']:5d}  {t['note'][:60]}", flush=True)

# ── Upload results to HF ────────────────────────────────────────────────────
if UPLOAD and hf_token:
    step("upload.results")
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        slug = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        api.upload_file(
            path_or_fileobj=str(RESULTS_PATH),
            path_in_repo=f"runs/{slug}-lang-spec-sweep.json",
            repo_id="cstr/crispasr-lang-sweep-results",
            repo_type="dataset",
            commit_message=f"lang-spec-sweep {slug}",
        )
        step("upload.done")
        print(f"\nResults uploaded to cstr/crispasr-lang-sweep-results/runs/{slug}-lang-spec-sweep.json",
              flush=True)
    except Exception as exc:
        step("upload.fail", error=str(exc))
        print(f"Upload failed: {exc}", flush=True)
else:
    print(f"\nResults written to {RESULTS_PATH}", flush=True)
    print("(set LANG_SWEEP_UPLOAD=1 and provide HF_TOKEN to upload)", flush=True)

step("script.done", elapsed_total_s=round(time.time() - _T0, 1))
