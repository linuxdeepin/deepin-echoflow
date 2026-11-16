"""Re-quantize canary-qwen and ship a WORKING small variant.

The published canary-qwen-2.5b-q4_k.gguf produces NaN logits (all-'!' output);
q8_0 is fine. Root cause (Kaggle quant-diff + GGUF header): tensor TYPES are
correct (token_embd/output F16, only the 196 Qwen3-1.7B LLM projections are
Q4_K) — so it is 4-bit *precision* on this small LLM, not a wrong-tensor bug.

This kernel downloads the F16 source and re-quantizes with today's quantizer at
q4_k, then q5_k, then q6_k, validating EACH by transcribing jfk (must hit the
gold key-words) with the NaN-guarded binary. It uploads the SMALLEST passing
variant:
  - q4_k passes            -> upload q4_k (overwrites the broken file in place).
  - q4_k fails, q5/q6 pass -> atomically replace: delete broken q4_k + add the
                              smallest passing k-quant.
  - none pass              -> delete the broken q4_k; q8_0 stays the only quant.
Full-harness utilities; CPU build (validation is CPU-fine, no CUDA needed).
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

_T0 = time.time()

# gotcha #22: the repo clone + build (thousands of files) and the multi-GB GGUFs
# ALL stage under /kaggle/temp so /kaggle/working stays tiny — only results.json
# + error.txt land there, so `kaggle kernels output` is fast and always carries
# the diagnostics even when the run errors.
TEMP = Path("/kaggle/temp")
OUT = Path("/kaggle/working")
REPO = TEMP / "CrispASR"
BUILD = REPO / "build"
MODELS = TEMP / "models"
for d in (TEMP, OUT, MODELS):
    d.mkdir(parents=True, exist_ok=True)
HF_REPO = "cstr/canary-qwen-2.5b-GGUF"

# Any uncaught exception -> a small retrievable file in /kaggle/working.
import traceback as _tb  # noqa: E402


def _excepthook(et, ev, tb):
    try:
        (OUT / "error.txt").write_text("".join(_tb.format_exception(et, ev, tb)))
    except Exception:
        pass
    sys.__excepthook__(et, ev, tb)


sys.excepthook = _excepthook


def run(cmd, **kw):
    kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    return subprocess.run(cmd, **kw)


# ── clone FIRST, then import the harness from the clone (repo carries it) ────
print(json.dumps({"step": "start"}), flush=True)
if REPO.exists():
    shutil.rmtree(REPO)
run(["git", "clone", "--recursive", "--depth", "1",
     "https://github.com/CrispStrobe/CrispASR.git", str(REPO)], capture_output=False)
run(["git", "-C", str(REPO), "submodule", "update", "--init", "--recursive", "--depth", "1"],
    capture_output=False, timeout=1800)

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
kh.step("cloned", sha=sha)

# ── toolchain + token ───────────────────────────────────────────────────────
kh.install_build_toolchain()
TOKEN = kh.resolve_hf_token("HF_TOKEN")
from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd, CommitOperationDelete  # noqa: E402

# ── build (CPU): crispasr-cli + crispasr-quantize ───────────────────────────
# Stream the build to the console log (capture_output=False) so any Kaggle-only
# compile/link error is visible — the previous run captured stdout and logged
# only the (empty) stderr, hiding the real cause. Build targets one at a time
# (older cmake mishandles multi-target `--target`).
kh.step("configure")
cfg = ["cmake", "-G", "Ninja", "-B", str(BUILD), "-S", str(REPO),
       "-DCMAKE_BUILD_TYPE=Release", "-DCRISPASR_NO_C2PA_NATIVE=ON"] + kh.cache_and_link_flags()
r = run(cfg, capture_output=False)
if r.returncode != 0:
    kh.step("configure.FAIL"); raise SystemExit(1)
jobs = str(min(4, os.cpu_count() or 2))  # CPU C++ TUs on the ~16 GB Kaggle box
for tgt in ("crispasr-cli", "crispasr-quantize"):
    kh.step(f"build.{tgt}")
    with kh.build_heartbeat(f"build.{tgt}"):
        r = run(["cmake", "--build", str(BUILD), "--target", tgt, "-j", jobs], capture_output=False)
    if r.returncode != 0:
        kh.step(f"build.{tgt}.FAIL", rc=r.returncode); raise SystemExit(1)
# The CLI target is `crispasr-cli` but OUTPUT_NAME is `crispasr`, emitted into
# RUNTIME_OUTPUT_DIRECTORY (build/bin). Resolve robustly across layouts.
def find_bin(name):
    p = BUILD / "bin" / name
    if p.exists():
        return p
    cands = [c for c in BUILD.rglob(name) if c.is_file() and os.access(c, os.X_OK)]
    return cands[0] if cands else None


CLI = find_bin("crispasr")
QUANT = find_bin("crispasr-quantize")
if CLI is None or QUANT is None:
    kh.step("build.MISSING", cli=str(CLI), quant=str(QUANT)); raise SystemExit(1)
os.environ["LD_LIBRARY_PATH"] = f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
kh.step("build.done", cli=str(CLI), quant=str(QUANT))

# ── download F16 source ─────────────────────────────────────────────────────
kh.step("download.f16.begin")
with kh.build_heartbeat("download.f16"):
    f16 = Path(hf_hub_download(repo_id=HF_REPO, filename="canary-qwen-2.5b-f16.gguf",
                              local_dir=str(MODELS), token=TOKEN))
kh.step("download.f16.done", gb=round(f16.stat().st_size / 1e9, 2))

JFK = REPO / "samples" / "jfk.wav"
JFK_KEYS = ["fellow", "americans", "country", "ask"]  # gold content words


def _norm(t):
    return re.findall(r"[a-z]+", t.lower())


def validate(gguf):
    """Transcribe jfk with canary-qwen; return (ok, hits, transcript)."""
    r = run([str(CLI), "-m", str(gguf), "--backend", "canary-qwen", "-f", str(JFK),
             "--chunk-seconds", "0", "--no-prints"], timeout=1200)
    lines = [ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip()]
    txt = lines[-1] if lines else ""
    words = set(_norm(txt))
    hits = sum(k in words for k in JFK_KEYS)
    ok = r.returncode == 0 and hits >= 3
    return ok, hits, txt[:180]


# ── quantize + validate, smallest-first ─────────────────────────────────────
ORDER = ["q4_k", "q5_k", "q6_k"]
results = {}
winner = None
for qt in ORDER:
    out = MODELS / f"canary-qwen-2.5b-{qt}.gguf"
    with kh.build_heartbeat(f"quantize.{qt}"):
        r = run([str(QUANT), str(f16), str(out), qt], timeout=1800)
    if not out.exists():
        results[qt] = {"quantized": False, "tail": (r.stderr or "")[-300:]}
        kh.step(f"quantize.{qt}.FAIL", tail=(r.stderr or "")[-300:]); continue
    sz = round(out.stat().st_size / 1e9, 2)
    with kh.build_heartbeat(f"validate.{qt}"):
        ok, hits, txt = validate(out)
    results[qt] = {"quantized": True, "gb": sz, "hits": hits, "pass": ok, "transcript": txt}
    kh.step(f"validate.{qt}", gb=sz, hits=hits, ok=ok, transcript=txt)
    print(f"  [{qt}] {sz}GB hits={hits}/4 pass={ok} :: {txt!r}", flush=True)
    if ok:
        winner = qt
        break  # smallest passing variant found
    out.unlink(missing_ok=True)  # free disk; a failed quant is never uploaded

# ── ship the outcome ────────────────────────────────────────────────────────
api = HfApi(token=TOKEN)
BROKEN = "canary-qwen-2.5b-q4_k.gguf"
if winner == "q4_k":
    with kh.build_heartbeat("upload.q4_k"):
        api.upload_file(path_or_fileobj=str(MODELS / BROKEN), path_in_repo=BROKEN,
                        repo_id=HF_REPO, repo_type="model",
                        commit_message="Fix q4_k: re-quantized F16 (prev file produced NaN logits / all-'!'); ASR-validated on jfk")
    kh.step("uploaded.q4_k", action="replaced broken q4_k in place")
elif winner in ("q5_k", "q6_k"):
    newname = f"canary-qwen-2.5b-{winner}.gguf"
    with kh.build_heartbeat(f"upload.{winner}"):
        api.create_commit(repo_id=HF_REPO, repo_type="model",
                          operations=[
                              CommitOperationAdd(path_in_repo=newname, path_or_fileobj=str(MODELS / newname)),
                              CommitOperationDelete(path_in_repo=BROKEN),
                          ],
                          commit_message=f"q4_k is NaN-corrupt on this small LLM; replace with ASR-validated {winner.upper()} (smallest working k-quant)")
    kh.step(f"uploaded.{winner}", action=f"deleted broken q4_k; added {newname}")
else:
    with kh.build_heartbeat("delete.q4_k"):
        api.delete_file(path_in_repo=BROKEN, repo_id=HF_REPO, repo_type="model",
                        commit_message="Remove NaN-corrupt q4_k (no k-quant validates on this small LLM; use q8_0)")
    kh.step("deleted.q4_k", action="no k-quant passed; removed broken q4_k, q8_0 remains")

RESULTS = {"results": results, "winner": winner, "wall_s": round(time.time() - _T0, 1)}
(OUT / "results.json").write_text(json.dumps(RESULTS, indent=2))
print(json.dumps({"step": "done", "winner": winner, "results": results}), flush=True)
kh.step("done", winner=winner or "none")
