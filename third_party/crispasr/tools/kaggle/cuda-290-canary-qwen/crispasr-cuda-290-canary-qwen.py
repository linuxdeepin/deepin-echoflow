"""CUDA live validation: #290 canary-qwen long-audio memory + #292 on canary-qwen.

#290: canary-qwen declared CAP_INTERNAL_CHUNKING without a chunker, so long audio
ran one full FastConformer pass -> O(T^2) attention, RSS 384 MiB -> 10.2 GiB, and
truncated output. Verified by capability-bitmask + CI already; this is the live
CUDA confirmation on the device that mattered (reporter used a GPU backend).

Gates (CPU + CUDA, decidable, proof-of-work):
  #290  canary-qwen on ~176 s with DEFAULT handling (chunks) completes NON-EMPTY
        with peak RSS well under the 10.2 GiB regression.
  #292  canary-qwen single-pass --max-new-tokens 64 (truncates) vs 4096 (does
        not) — a second independent #292 backend beyond moss-diarize; EQUAL word
        counts would be the flag-ignored bug.

Full harness regime; hf_hub_download (hf_transfer fast path) + lean q4_k.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
REPO = TEMP / "CrispASR"
BUILD = TEMP / "build"
MODELS = TEMP / "models"
FIX = TEMP / "fixtures"
OUT = WORK / "outputs"
CRISPASR_REPO = "https://github.com/CrispStrobe/CrispASR.git"
CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
for d in (MODELS, FIX, OUT):
    d.mkdir(parents=True, exist_ok=True)

RESULTS = {"tests": [], "bench": [], "env": {}}
_T0 = time.time()


def run(cmd, check=True, env=None, cwd=None, timeout=None, capture=True):
    e = {**os.environ, **(env or {})}
    if capture:
        r = subprocess.run(cmd, env=e, cwd=cwd, timeout=timeout, shell=isinstance(cmd, str),
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    else:
        r = subprocess.run(cmd, env=e, cwd=cwd, timeout=timeout, shell=isinstance(cmd, str))
        r.stdout = ""
    if check and r.returncode != 0:
        print((r.stdout or "")[-8000:], flush=True)
        raise SystemExit(f"command failed ({r.returncode}): {cmd}")
    return r


def record(name, passed, **kw):
    entry = {"test": name, "pass": bool(passed), **kw}
    RESULTS["tests"].append(entry)
    print(("PASS  " if passed else "FAIL  ") + name + "  " + json.dumps(kw), flush=True)
    (WORK / "results.json").write_text(json.dumps(RESULTS, indent=2))
    return passed


# ── cell 1: clone + harness (FULL regime) ──────────────────────────────────
print(json.dumps({"step": "start", "ref": CRISPASR_REF}), flush=True)
if REPO.exists():
    shutil.rmtree(REPO)
run(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF, "--recursive",
     CRISPASR_REPO, str(REPO)], capture=False)
run(["git", "-C", str(REPO), "submodule", "update", "--init", "--recursive"],
    check=False, capture=False, timeout=1800)

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
kh.step("cloned", sha=sha)
RESULTS["env"]["sha"] = sha
RESULTS["env"]["ref"] = CRISPASR_REF
run(["nvidia-smi", "-L"], check=False)
gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
kh.step("gpu", gpu=gpu)
RESULTS["env"]["gpu"] = gpu

# ── cell 2: build (CUDA) ───────────────────────────────────────────────────
kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
kh.step("cuda_arch", arch=arch)
RESULTS["env"]["cuda_arch"] = arch
BUILD.mkdir(parents=True, exist_ok=True)
cmake_args = [
    "cmake", "-S", str(REPO), "-B", str(BUILD),
    "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON",
    "-DCRISPASR_NO_C2PA_NATIVE=ON",
] + kh.cuda_build_flags(arch) + kh.cache_and_link_flags()
run(cmake_args, capture=False)
kh.step("cmake_done")
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli "
        f"-j{kh.safe_build_jobs(gpu=True)}")
kh.step("build_done")

CLI = BUILD / "examples" / "cli" / "crispasr"
if not CLI.exists():
    cands = [c for c in BUILD.rglob("crispasr") if c.is_file() and os.access(c, os.X_OK)]
    if not cands:
        raise SystemExit("crispasr binary not found after build")
    CLI = cands[0]
os.environ["LD_LIBRARY_PATH"] = f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
kh.step("cli", path=str(CLI))

# ── cell 3: downloads ──────────────────────────────────────────────────────
# resolve_hf_token() exports HF_HUB_ENABLE_HF_TRANSFER=1 — the FAST parallel path.
# Use hf_hub_download so that flag actually applies; a single-connection curl (an
# earlier mistake here) crawled at ~355k/s on Kaggle's otherwise-fast pipe. curl
# is kept only as a fallback if the hf library path raises.
token = kh.resolve_hf_token()
run(["pip", "install", "-q", "hf_transfer", "huggingface_hub"], check=False)
kh.step("hf_token", have=bool(token))


def hf_get(repo, filename, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    t0 = time.time()
    try:
        from huggingface_hub import hf_hub_download
        got = hf_hub_download(repo_id=repo, filename=filename, token=token,
                              local_dir=str(dest.parent))
        gp = Path(got)
        if gp.resolve() != dest.resolve():
            shutil.copy(gp, dest)
    except Exception as e:
        print(f"hf_hub_download failed ({e}); falling back to curl", flush=True)
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        hdr = ["-H", f"Authorization: Bearer {token}"] if token else []
        run(["curl", "-fL", "-C", "-", "--retry", "5", "--retry-delay", "5",
             "-o", str(dest), *hdr, url], capture=False, timeout=3600)
    mb = dest.stat().st_size / 1e6
    kh.step("download.done", file=dest.name, mb=round(mb, 1),
            mbps=round(mb / max(time.time() - t0, 0.1), 1))
    return dest


M = {}
# LEAN by design: HF multi-GB pulls crawl on Kaggle (~355k/s degrading), so this
# kernel downloads ONLY what the acceptance test needs. moss-diarize is the
# reporter's EXACT backend (0.9B, q4_k = 1.2 GB) and exercises BOTH #292 parts
# (--max-new-tokens + chunk_id/speaker). tabcnn-f16 is ~2 MB (CUDA-parity bonus).
# The other affected backends (canary-qwen q4_k 3.6 GB, mimo q4_k 4.5 GB) are the
# SAME code pattern and were dropped to keep the run short — CI already compiled
# all 10 and the fix is identical per-backend.
M["canary_qwen"] = hf_get("cstr/canary-qwen-2.5b-GGUF", "canary-qwen-2.5b-q4_k.gguf",
                          MODELS / "canary-qwen-q4_k.gguf")
kh.step("downloads_done")

# ── cell 4: fixtures ───────────────────────────────────────────────────────
JFK = REPO / "samples" / "jfk.wav"
# MED (~55 s): the #292 single-pass cap test. Long enough that cap-4096 >> cap-64,
# short enough that a single FastConformer pass fits GPU memory on every backend.
# LONG (~176 s): the #290 default-chunking test — must stay bounded via chunking.
MED = FIX / "jfk_med.wav"
LONG = FIX / "jfk_long.wav"
run(["ffmpeg", "-y", "-stream_loop", "4", "-i", str(JFK), "-ar", "16000", "-ac", "1", str(MED)], capture=False)
run(["ffmpeg", "-y", "-stream_loop", "15", "-i", str(JFK), "-ar", "16000", "-ac", "1", str(LONG)], capture=False)


def wav_seconds(p):
    with wave.open(str(p), "rb") as w:
        return w.getnframes() / w.getframerate()


kh.step("fixtures", med_s=round(wav_seconds(MED), 1), long_s=round(wav_seconds(LONG), 1))


def transcribe(model, backend, wav, device, extra=None, timeout=1200):
    """Run the CLI; return (ok, text, wall_s, peak_kb).

    Device is selected by the flag, not an env var: CPU = --no-gpu, CUDA = default
    (the only GPU backend built on Kaggle). There is no CRISPASR_DEVICE knob.
    """
    args = [str(CLI), "-m", str(model), "--backend", backend, "-f", str(wav), "--no-prints"]
    if device == "cpu":
        args += ["--no-gpu"]
    if extra:
        args += extra
    t0 = time.time()
    # /usr/bin/time -v gives peak RSS; guard its absence.
    timed = ["/usr/bin/time", "-v"] + args if Path("/usr/bin/time").exists() else args
    r = run(timed, check=False, timeout=timeout)
    wall = time.time() - t0
    out = r.stdout or ""
    peak_kb = 0
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", out)
    if m:
        peak_kb = int(m.group(1))
    # strip the /usr/bin/time trailer to isolate the transcript
    text = re.split(r"\n\tCommand being timed:", out)[0]
    text = re.sub(r"\n\s*(User time|System time|Percent of CPU|Elapsed|Maximum resident|.*resident set|.*page|.*Context|.*Swaps|.*I/O|.*Signals|.*socket|.*Exit status).*", "", text)
    ok = (r.returncode == 0) and bool(text.strip())
    return ok, text.strip(), wall, peak_kb


def word_count(t):
    return len(re.findall(r"\S+", t))


# ── cell 5: #292 — canary-qwen honors --max-new-tokens (both directions) ────
# Single-pass (--chunk-seconds 0), cap 64 (truncates) vs 4096 (does not). A second
# independent #292 backend beyond moss-diarize; equal counts would be the bug.
for device in ["cpu", "cuda"]:
    with kh.build_heartbeat(f"292.canary_qwen.{device}"):
        ok_lo, lo_txt, _, _ = transcribe(M["canary_qwen"], "canary-qwen", JFK, device,
                                         extra=["--chunk-seconds", "0", "--max-new-tokens", "8"])
        ok_hi, hi_txt, _, _ = transcribe(M["canary_qwen"], "canary-qwen", JFK, device,
                                         extra=["--chunk-seconds", "0", "--max-new-tokens", "4096"])
    n_lo, n_hi = word_count(lo_txt), word_count(hi_txt)
    passed = ok_lo and ok_hi and n_hi >= n_lo * 1.5 and n_hi >= n_lo + 20
    record(f"292:canary-qwen:{device}:max_new_tokens_honored", passed,
           words_cap8=n_lo, words_cap4096=n_hi, ratio=round(n_hi / max(n_lo, 1), 2),
           note="11s single-pass jfk: cap 8 MUST truncate vs 4096; EQUAL = flag ignored on a single decode (real bug)")

# ── cell 6: #290 — canary-qwen long audio stays bounded (the real target) ───
# The #290 regression: canary-qwen declared CAP_INTERNAL_CHUNKING with no chunker,
# so long audio ran ONE full FastConformer pass -> O(T^2) attention, RSS 384 MiB
# -> 10.2 GiB, and truncated output. The fix removed the false cap, restoring the
# 30 s auto-chunk fallback. Gate: on ~176 s with DEFAULT handling (no
# --chunk-seconds 0) canary-qwen completes NON-EMPTY with peak RSS well under the
# 10 GiB blowup. CUDA is the device that mattered (the reporter used Vulkan/GPU).
for device in ["cpu", "cuda"]:
    with kh.build_heartbeat(f"290.canary_qwen.{device}"):
        ok, txt, wall, peak_kb = transcribe(M["canary_qwen"], "canary-qwen", LONG, device, timeout=2400)
    peak_gb = round(peak_kb / 1e6, 2)
    # non-empty AND (RSS unavailable OR well under the 10.2 GiB regression)
    passed = ok and word_count(txt) > 0 and (peak_kb == 0 or peak_gb < 8.0)
    record(f"290:canary-qwen:{device}:long_audio_bounded", passed,
           words=word_count(txt), peak_gb=peak_gb, wall_s=round(wall, 1),
           note="default handling chunks long audio; non-empty + no O(T^2) 10 GiB blowup")

# ── summary ────────────────────────────────────────────────────────────────
npass = sum(1 for t in RESULTS["tests"] if t["pass"])
ntot = len(RESULTS["tests"])
RESULTS["summary"] = {"passed": npass, "total": ntot, "wall_s": round(time.time() - _T0, 1)}
(WORK / "results.json").write_text(json.dumps(RESULTS, indent=2))
print(json.dumps({"step": "done", "passed": npass, "total": ntot}), flush=True)
kh.step("done", passed=npass, total=ntot)
