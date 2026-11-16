#!/usr/bin/env python3
"""Kaggle kernel: re-quantize FastConformer-family GGUFs on HF (#81 conv-pw fix).

Every quantized FastConformer GGUF shipped with F16 conv.pw1/pw2 (the 3D conv
layout dodged crispasr-quantize's 2D-only rule) — ~6x slower than Q8_0 on the
CPU mul_mat path. The fixed quantizer (main, cdc6dbc4) converts them to 2D
Q8_0; all other tensors are byte-copied (same-type requant is a copy), so the
new file is the old file + native pw fix, and Xet dedup keeps uploads small.

Per repo / per quantized file:
  1. download old.gguf → /tmp (kaggle_usage.md #18/#21)
  2. crispasr-quantize old new <type-from-filename>  (0 conversions → skip)
  3. STRICT check: transcribe(old, defaults) == transcribe(new, defaults)
     exactly — the runtime repack (CRISPASR_FC_PW_Q8) makes old ≡ new
  4. SOFT check: transcribe(old, legacy gates) vs new — word overlap ≥ 0.8
     (guards against per-backend quality damage from pw F16→Q8_0); also
     logs legacy-vs-new wall times = per-backend speed evidence
  5. upload to the same path, delete local copies

Re-runs are cheap: step 2 no-ops on already-fixed files.
Push (under $KAGGLE_ACCOUNT): tools/kaggle/fc-pw-requant/push.sh
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")

# ── Phase 0: clone + build CrispASR (CUDA if available, warm ccache) ─────────
print("=== Phase 0: clone + build CrispASR ===", flush=True)
if not REPO.exists():
    subprocess.check_call([
        "git", "clone", "--depth", "1", "-b", "main",
        "https://github.com/CrispStrobe/CrispASR", str(REPO),
    ])
if (REPO / "ggml").is_dir() and not (REPO / "ggml" / "CMakeLists.txt").exists():
    subprocess.check_call(["git", "submodule", "update", "--init", "ggml"], cwd=str(REPO))

if (REPO / "tools" / "kaggle").is_dir():
    sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # bundled fallback
import kaggle_harness as kh  # noqa: E402

kh.init_progress(hf_progress_repo="cstr/crispasr-kaggle-progress")
step = kh.step
step("script.start")

TOKEN = kh.resolve_hf_token("HF_TOKEN")
step("hf_token.resolved", have=bool(TOKEN))

step("deps.begin")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                       "huggingface_hub", "hf_transfer", "soundfile", "gguf"])
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402
step("deps.done")

BUILD = TEMP / "build"
BUILD.mkdir(parents=True, exist_ok=True)
kh.install_build_toolchain()

# Relocate ccache out of /kaggle/working (see issue81-onnx-bench: the 500-file
# output page cap makes ccache.tar unretrievable when .ccache/ sits there).
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
if has_cuda:
    flags = kh.cuda_build_flags(kh.detect_cuda_arch()) + kh.cache_and_link_flags()
else:
    flags = kh.cache_and_link_flags()
cmake_flags = "-DCMAKE_BUILD_TYPE=Release " + " ".join(flags)
ret = subprocess.call(f"cmake -G Ninja -B {BUILD} -S {REPO} {cmake_flags}", shell=True)
if ret != 0 and has_cuda:  # CUDA cmake broken → CPU fallback
    shutil.rmtree(BUILD, ignore_errors=True)
    BUILD.mkdir(parents=True, exist_ok=True)
    has_cuda = False
    cmake_flags = "-DCMAKE_BUILD_TYPE=Release " + " ".join(kh.cache_and_link_flags())
    subprocess.check_call(f"cmake -G Ninja -B {BUILD} -S {REPO} {cmake_flags}", shell=True)
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} -j {kh.safe_build_jobs(has_cuda)} "
        f"--target crispasr-lib crispasr-quantize")
QUANT = BUILD / "bin" / "crispasr-quantize"
LIB = BUILD / "src" / "libcrispasr.so"
assert QUANT.is_file() and LIB.is_file(), "build products missing"
step("build.done", cuda=has_cuda)

WAV = REPO / "samples" / "jfk.wav"
MODELS = TEMP / "models"
MODELS.mkdir(parents=True, exist_ok=True)
API = HfApi(token=TOKEN)

# ── Repo fleet, priority-ordered (a timeout still ships the head) ────────────
# (repo_short, backend_hint or None=auto, language)
FLEET = [
    ("parakeet-ctc-0.6b-GGUF", None, "en"),
    ("parakeet-ctc-1.1b-GGUF", None, "en"),
    ("parakeet-tdt-0.6b-v3-GGUF", None, "en"),
    ("canary-1b-v2-GGUF", "canary", "en"),
    ("canary-ctc-aligner-GGUF", "canary-ctc", "en"),
    ("parakeet-tdt-0.6b-v2-GGUF", None, "en"),
    ("parakeet-tdt-1.1b-GGUF", None, "en"),
    ("parakeet-rnnt-0.6b-GGUF", None, "en"),
    ("parakeet-rnnt-1.1b-GGUF", None, "en"),
    ("parakeet-tdt_ctc-110m-GGUF", None, "en"),
    ("parakeet-tdt_ctc-1.1b-GGUF", None, "en"),
    ("parakeet-tdt-0.6b-ja-GGUF", None, "ja"),
    ("parakeet-ctc-1.1b-ja-GGUF", None, "ja"),
    ("parakeet-unified-en-0.6b-GGUF", None, "en"),
    ("parakeet_de_med-GGUF", None, "de"),
    ("stt-en-fastconformer-ctc-large-GGUF", None, "en"),
    ("stt-en-fastconformer-ctc-xlarge-GGUF", None, "en"),
    ("stt-en-fastconformer-ctc-xxlarge-GGUF", None, "en"),
    ("nemotron-3.5-asr-streaming-0.6b-GGUF", "nemotron", "en"),
    ("nemotron-3.5-asr-streaming-GGUF", "nemotron", "en"),
    ("reazonspeech-nemo-v2-GGUF", "reazonspeech", "ja"),
    ("lfm2-audio-1.5b-GGUF", "lfm2-audio", "en"),
    ("lfm2-audio-1.5b-jp-GGUF", "lfm2-audio", "ja"),
    ("canary-qwen-2.5b-GGUF", "canary-qwen", "en"),
    ("stt-en-fastconformer-hybrid-ctc-large-GGUF", None, "en"),
    ("stt-de-fastconformer-hybrid-ctc-large-GGUF", None, "de"),
    ("stt-es-fastconformer-hybrid-ctc-large-GGUF", None, "es"),
    ("stt-fr-fastconformer-hybrid-ctc-large-GGUF", None, "fr"),
    ("stt-it-fastconformer-hybrid-ctc-large-GGUF", None, "it"),
    ("stt-nl-fastconformer-hybrid-ctc-large-GGUF", None, "nl"),
    ("stt-pl-fastconformer-hybrid-ctc-large-GGUF", None, "pl"),
    ("stt-ru-fastconformer-hybrid-ctc-large-GGUF", None, "ru"),
    ("stt-ua-fastconformer-hybrid-ctc-large-GGUF", None, "uk"),
    ("stt-hr-fastconformer-hybrid-ctc-large-GGUF", None, "hr"),
    ("stt-be-fastconformer-hybrid-ctc-large-GGUF", None, "be"),
    ("stt-ar-fastconformer-hybrid-ctc-large-GGUF", None, "ar"),
    ("stt-fa-fastconformer-hybrid-ctc-large-GGUF", None, "fa"),
    ("stt-ka-fastconformer-hybrid-ctc-large-GGUF", None, "ka"),
    ("stt-hy-fastconformer-hybrid-ctc-large-GGUF", None, "hy"),
    ("stt-uz-fastconformer-hybrid-ctc-large-GGUF", None, "uz"),
    ("stt-kk-ru-fastconformer-hybrid-ctc-large-GGUF", None, "kk"),
]
QUANT_SUFFIXES = ["q8_0", "q6_k", "q5_k", "q5_0", "q4_k", "q4_0", "q3_k", "q2_k"]

# Per-file quantizer overrides: these published q4_k files keep the whole
# transducer decode head (decoder.embed + joint.out + joint.pred) at F16
# (old rule), which current rules would quantize — pin them so ONLY the
# conv pw tensors change (minimal diff, passes the structural gate).
_TDT_HEAD_PIN = ["--tensor-type", r"(decoder\.embed|joint\.out|joint\.pred)\.weight=f16"]
EXTRA_ARGS = {
    "parakeet-tdt-0.6b-v3-q4_k.gguf": _TDT_HEAD_PIN,
    "parakeet_de_med-q4_k.gguf": _TDT_HEAD_PIN,
}
LEGACY_ENV = {"CRISPASR_FC_PW_Q8": "0", "CRISPASR_FC_FUSED_QKV": "0",
              "CRISPASR_FC_ATTN_CONT": "1"}

CHILD = r"""
import os, sys, time
sys.path.insert(0, os.path.join(sys.argv[4], "python"))
os.environ["CRISPASR_LIB_PATH"] = sys.argv[5]
import soundfile as sf
from crispasr import Session
pcm, sr = sf.read(sys.argv[3], dtype="float32")
s = Session(sys.argv[1], n_threads=4, backend=sys.argv[2] or None)
t0 = time.perf_counter()
segs = s.transcribe(pcm, language=sys.argv[6])
dt = time.perf_counter() - t0
print("DT::%.3f" % dt)
print("TRANSCRIPT::" + " | ".join(seg.text for seg in segs))
"""


def transcribe(path, backend, lang, env_extra=None):
    env = dict(os.environ, **(env_extra or {}))
    try:
        r = subprocess.run(
            [sys.executable, "-c", CHILD, str(path), backend or "", str(WAV), str(REPO), str(LIB), lang],
            capture_output=True, text=True, timeout=600, env=env)
    except subprocess.TimeoutExpired:
        return None, None, "transcribe timeout (600s)"
    txt, dt = None, None
    for ln in r.stdout.splitlines():
        if ln.startswith("TRANSCRIPT::"):
            txt = ln[len("TRANSCRIPT::"):]
        elif ln.startswith("DT::"):
            dt = float(ln[4:])
    return txt, dt, r.stderr[-400:]


# ── hang-proof network ops ────────────────────────────────────────────────────
# HfApi.upload_file can strand in CLOSE_WAIT after the commit lands (observed:
# commit visible on HF, client never returns). Run uploads/downloads on daemon
# threads with a join timeout, then verify server/file state to decide success.
import threading  # noqa: E402


def _run_daemon(fn, timeout_s):
    box = {}

    def _t():
        try:
            box["ret"] = fn()
        except Exception as exc:
            box["exc"] = exc

    th = threading.Thread(target=_t, daemon=True)
    th.start()
    th.join(timeout_s)
    return (not th.is_alive()), box.get("ret"), box.get("exc")


def download_with_timeout(repo, path, timeout_s=1800):
    done, ret, exc = _run_daemon(
        lambda: hf_hub_download(repo, path, local_dir=str(MODELS), token=TOKEN), timeout_s)
    if done and exc is None:
        return Path(ret)
    local = MODELS / path
    if local.is_file():  # thread hung after the file fully landed
        return local
    raise RuntimeError(f"download {'hung' if not done else 'failed'}: {exc}")


def upload_with_timeout(local, repo, path, msg, timeout_s=1800):
    size = Path(local).stat().st_size

    def _up():
        API.upload_file(path_or_fileobj=str(local), path_in_repo=path, repo_id=repo,
                        repo_type="model", commit_message=msg)

    done, _, exc = _run_daemon(_up, timeout_s)
    if done and exc is None:
        return True
    # Client hung or errored — trust the server: did the new blob land?
    try:
        for f in HfApi(token=TOKEN).list_repo_tree(repo):
            if f.path == path and getattr(f, "size", None) == size:
                step("upload.recovered", repo=repo, file=path,
                     note="client hung/errored but commit landed")
                return True
    except Exception:
        pass
    raise RuntimeError(f"upload {'hung' if not done else 'failed'}: {exc}")


def push_progress(results):
    """Best-effort per-file progress mirror (own API object, daemon timeout)."""
    def _up():
        blob = json.dumps(results, indent=1).encode()
        HfApi(token=TOKEN).upload_file(
            path_or_fileobj=blob, path_in_repo="fc-pw-requant/results.json",
            repo_id="cstr/crispasr-kaggle-progress", repo_type="dataset",
            commit_message="fc-pw-requant progress")

    _run_daemon(_up, 120)


def words(s):
    return re.sub(r"[^\w ]", "", (s or "").lower()).split()


def overlap(a, b):
    wa, wb = set(words(a)), set(words(b))
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


def structural_ok(old_path, new_path):
    """Non-pw tensors byte-identical; F16 pw tensors became Q8_0."""
    from gguf import GGUFReader
    to = {t.name: t for t in GGUFReader(str(old_path)).tensors}
    tn = {t.name: t for t in GGUFReader(str(new_path)).tensors}
    if set(to) != set(tn):
        return False, "tensor sets differ"
    n_pw = 0
    for name, t_old in to.items():
        t_new = tn[name]
        is_pw = name.endswith("conv.pw1.weight") or name.endswith("conv.pw2.weight")
        if is_pw and t_old.tensor_type.name == "F16":
            if t_new.tensor_type.name != "Q8_0":
                return False, f"{name}: {t_new.tensor_type.name} != Q8_0"
            n_pw += 1
            continue
        if t_old.tensor_type != t_new.tensor_type:
            return False, f"{name}: type changed"
        if t_old.data.tobytes() != t_new.data.tobytes():
            return False, f"{name}: data differs"
    return True, f"{n_pw} pw → Q8_0"


def free_gb():
    return round(shutil.disk_usage(str(MODELS)).free / 1e9, 1)


def process_file(repo, path, backend, lang):
    qtype = None
    base = path[:-5].lower()
    for q in QUANT_SUFFIXES:
        if base.endswith("-" + q) or base.endswith("_" + q):
            qtype = q
            break
    if qtype is None:
        return "skip-unquantized"

    old = download_with_timeout(repo, path)
    new = MODELS / "new" / path
    new.parent.mkdir(parents=True, exist_ok=True)
    try:
        p = subprocess.run([str(QUANT), str(old), str(new), qtype] + EXTRA_ARGS.get(path, []),
                           capture_output=True, text=True, timeout=3600)
        if p.returncode != 0:
            step("file.QUANTIZE-FAIL", repo=repo, file=path, tail=p.stdout[-200:] + p.stderr[-200:])
            return "quantize-fail"
        n_conv = p.stdout.count("quantizing to")
        if n_conv == 0:
            step("file.already-fixed", repo=repo, file=path)
            return "already-fixed"

        ok, why = structural_ok(old, new)
        step("file.structural", repo=repo, file=path, ok=ok, why=why, n_conv=n_conv)
        if not ok:
            return "structural-fail"

        t_old, dt_old, err_o = transcribe(old, backend, lang)
        t_new, dt_new, err_n = transcribe(new, backend, lang)
        tag = "uploaded"
        if t_old is None and t_new is None and (err_o or "")[-120:] == (err_n or "")[-120:]:
            # Backend can't transcribe this input at all (identically on old
            # and new — pre-existing, e.g. reazonspeech via the session ABI).
            # The structural check above already proves new == old + the
            # shipped runtime pw repack, so accept on that basis.
            step("file.structural-only", repo=repo, file=path,
                 err=(err_o or "").replace("\n", " / ")[-200:])
            tag = "uploaded-structural-only"
        elif t_old is None or t_new is None:
            step("file.TRANSCRIBE-FAIL", repo=repo, file=path,
                 old_ok=t_old is not None, new_ok=t_new is not None,
                 err=(err_n or err_o).replace("\n", " / ")[-300:])
            return "transcribe-fail"
        elif t_old != t_new:
            step("file.STRICT-MISMATCH", repo=repo, file=path,
                 old=t_old[:150], new=t_new[:150])
            return "strict-mismatch"
        else:
            # Soft check vs the fully-legacy path (pw F16, no repack): a low
            # overlap on gibberish out-of-domain output (a JA/UZ model on
            # English audio) is not a quality signal, so WARN only — strict
            # equality above is the acceptance (new ≡ old + runtime repack,
            # which is the shipped default behavior either way).
            t_leg, dt_leg, _ = transcribe(old, backend, lang, LEGACY_ENV)
            ov = overlap(t_leg, t_new)
            step("file.verified", repo=repo, file=path, strict="match",
                 legacy_overlap=round(ov, 3), dt_legacy=dt_leg, dt_new=dt_new,
                 speedup=round(dt_leg / dt_new, 2) if dt_leg and dt_new else None,
                 transcript=t_new[:120])
            if ov < 0.8:
                step("file.SOFT-WARN", repo=repo, file=path, overlap=round(ov, 3),
                     legacy=(t_leg or "")[:150], new=t_new[:150])

        upload_with_timeout(new, repo, path,
                            f"Requantize {path}: conv pw1/pw2 F16→Q8_0 "
                            f"(#81 CPU perf fix; all other tensors byte-identical)")
        step("file.UPLOADED", repo=repo, file=path)
        return tag
    finally:
        old.unlink(missing_ok=True)
        new.unlink(missing_ok=True)


# Optional scope file: an `only.txt` pushed next to the script restricts the
# run to the listed repo shortnames (for targeted re-runs without touring
# the whole fleet).
_only_f = Path(__file__).resolve().parent / "only.txt"
ONLY = set(_only_f.read_text().split()) if _only_f.exists() else set()

results = {}
for short, backend, lang in FLEET:
    if ONLY and short not in ONLY:
        continue
    repo = f"cstr/{short}"
    step("repo.begin", repo=repo, free_gb=free_gb())
    try:
        files = [f.path for f in API.list_repo_tree(repo) if f.path.endswith(".gguf")]
    except Exception as exc:
        step("repo.LIST-FAIL", repo=repo, error=str(exc))
        results[repo] = {"error": str(exc)}
        continue
    r = {}
    results[repo] = r
    for path in sorted(files):
        try:
            r[path] = process_file(repo, path, backend, lang)
        except Exception as exc:
            step("file.EXCEPTION", repo=repo, file=path, error=f"{type(exc).__name__}: {exc}")
            r[path] = f"exception: {exc}"
        (WORK / "results.json").write_text(json.dumps(results, indent=1))
        push_progress(results)
    step("repo.done", repo=repo, results=r)

n_up = sum(1 for r in results.values() if isinstance(r, dict)
           for v in r.values() if v == "uploaded")
n_bad = sum(1 for r in results.values() if isinstance(r, dict)
            for v in r.values() if "fail" in str(v) or "mismatch" in str(v) or "exception" in str(v))
step("script.done", uploaded=n_up, failed=n_bad)
print(f"DONE uploaded={n_up} failed={n_bad}", flush=True)
