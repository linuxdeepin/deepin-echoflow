"""DIAG: canary-qwen emits all-! (token id 0 = NaN logits) — is it q4_k-specific?
Registry DEFAULT is q8_0 (4.1GB); the earlier test used q4_k (self-chosen).
Diffs q8_0 vs q4_k on jfk (actual generated text) + FUSED_QKV/PW_Q8 toggles.
  q8_0 real + q4_k ! => q4_k QUANT corruption (cf cohere/omnivoice zeroed).
  both !             => forward/port NaN (escalate to NeMo diff harness).
  a toggle fixes q4k => that runtime opt injects the NaN.
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

# also grab the REGISTRY DEFAULT q8_0 (4.1 GB) — the precision real users get.
M["canary_qwen_q8"] = hf_get("cstr/canary-qwen-2.5b-GGUF", "canary-qwen-2.5b-q8_0.gguf",
                             MODELS / "canary-qwen-q8_0.gguf")
kh.step("downloads_done_q8")

JFK = REPO / "samples" / "jfk.wav"

def canary_gen(model_path, env_extra=None, cap=32):
    """Run canary-qwen on jfk; return the 'generated N tokens: <text>' debug line."""
    env = {**os.environ, "CRISPASR_CANARY_QWEN_DEBUG": "1"}
    if env_extra:
        env.update(env_extra)
    args = [str(CLI), "-m", str(model_path), "--backend", "canary-qwen",
            "-f", str(JFK), "--chunk-seconds", "0", "--max-new-tokens", str(cap)]
    r = subprocess.run(args, env=env, capture_output=True, text=True, timeout=1200)
    full = (r.stdout or "") + (r.stderr or "")
    gen = next((l for l in full.splitlines() if "generated" in l and "tokens:" in l), "")
    txt = gen.split("tokens:", 1)[1].strip() if "tokens:" in gen else ""
    is_bang = bool(txt) and set(txt) <= {"!"}
    return {"gen_line": gen[:160], "text": txt[:120], "all_bang": is_bang, "rc": r.returncode}

MATRIX = [
    ("q8_0-default",       "canary_qwen_q8", None),
    ("q4_k-default",       "canary_qwen",    None),
    ("q4_k-FUSED_QKV_off", "canary_qwen",    {"CRISPASR_FC_FUSED_QKV": "0"}),
    ("q4_k-PW_Q8_off",     "canary_qwen",    {"CRISPASR_FC_PW_Q8": "0"}),
    ("q4_k-both_opts_off", "canary_qwen",    {"CRISPASR_FC_FUSED_QKV": "0", "CRISPASR_FC_PW_Q8": "0"}),
]
for name, key, env_extra in MATRIX:
    with kh.build_heartbeat(f"gen.{name}"):
        res = canary_gen(M[key], env_extra)
    ok = (not res["all_bang"]) and bool(res["text"])
    record(f"diag:canary-qwen:{name}", ok, all_bang=res["all_bang"], text=res["text"], rc=res["rc"])
    RESULTS["bench"].append({"config": name, "all_bang": res["all_bang"], "text": res["text"]})
    print(f"  [{name}] all_bang={res['all_bang']} text={res['text']!r}", flush=True)

npass = sum(1 for t in RESULTS["tests"] if t["pass"])
ntot = len(RESULTS["tests"])
RESULTS["summary"] = {"passed": npass, "total": ntot, "wall_s": round(time.time() - _T0, 1)}
(WORK / "results.json").write_text(json.dumps(RESULTS, indent=2))
print(json.dumps({"step": "done", "passed": npass, "total": ntot}), flush=True)
kh.step("done", passed=npass, total=ntot)
