"""Convert efwkjn/whisper-ja-760M (Whisper large-v3-turbo arch, custom 10 240-token
Japanese vocab, #258 custom-vocab path) to CrispASR whisper GGML, quantise to q4_k,
validate by transcribing a Japanese clip, and upload to a PRIVATE HF repo.

Full harness regime: clone CrispASR + import kaggle_harness from the clone, both
datasets (crispasr-hf-token + crispasr-ccache), token via resolve_hf_token, every
long op wrapped in build_heartbeat. Source model has NO license → PRIVATE repo only.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback as _tb
from pathlib import Path

_T0 = time.time()

TEMP = Path("/kaggle/temp")
OUT = Path("/kaggle/working")
REPO = TEMP / "CrispASR"
BUILD = REPO / "build"
MODELS = TEMP / "models"
for d in (TEMP, OUT, MODELS):
    d.mkdir(parents=True, exist_ok=True)

SRC_REPO = "efwkjn/whisper-ja-760M"
HF_REPO = "cstr/whisper-ja-760M-GGML"   # PRIVATE (source has no license)
NAME = "whisper-ja-760m"


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


# ── clone FIRST, then import the harness from the clone ──────────────────────
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
from huggingface_hub import HfApi, snapshot_download  # noqa: E402

# ── build (CPU): crispasr-cli + crispasr-legacy-quantize ────────────────────
kh.step("configure")
cfg = ["cmake", "-G", "Ninja", "-B", str(BUILD), "-S", str(REPO),
       "-DCMAKE_BUILD_TYPE=Release", "-DCRISPASR_NO_C2PA_NATIVE=ON"] + kh.cache_and_link_flags()
r = run(cfg, capture_output=False)
if r.returncode != 0:
    kh.step("configure.FAIL"); raise SystemExit(1)
jobs = str(min(4, os.cpu_count() or 2))
for tgt in ("crispasr-cli", "crispasr-legacy-quantize"):
    kh.step(f"build.{tgt}")
    with kh.build_heartbeat(f"build.{tgt}"):
        r = run(["cmake", "--build", str(BUILD), "--target", tgt, "-j", jobs], capture_output=False)
    if r.returncode != 0:
        kh.step(f"build.{tgt}.FAIL", rc=r.returncode); raise SystemExit(1)


def find_bin(name):
    p = BUILD / "bin" / name
    if p.exists():
        return p
    cands = [c for c in BUILD.rglob(name) if c.is_file() and os.access(c, os.X_OK)]
    return cands[0] if cands else None


CLI = find_bin("crispasr")
QUANT = find_bin("crispasr-legacy-quantize")
if CLI is None or QUANT is None:
    kh.step("build.MISSING", cli=str(CLI), quant=str(QUANT)); raise SystemExit(1)
os.environ["LD_LIBRARY_PATH"] = f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
kh.step("build.done", cli=str(CLI), quant=str(QUANT))

# ── download source model (all HF files) ────────────────────────────────────
SRC = MODELS / "src"
kh.step("download.begin", repo=SRC_REPO)
with kh.build_heartbeat("download.model"):
    snapshot_download(repo_id=SRC_REPO, local_dir=str(SRC), token=TOKEN)
cfg_json = json.loads((SRC / "config.json").read_text())
kh.step("download.done", vocab=cfg_json.get("vocab_size"), n_mels=cfg_json.get("num_mel_bins"))

# ── mel filters (openai/whisper assets) ─────────────────────────────────────
ASSETS = MODELS / "whisper-assets"
(ASSETS / "whisper" / "assets").mkdir(parents=True, exist_ok=True)
run(["curl", "-sL", "-o", str(ASSETS / "whisper" / "assets" / "mel_filters.npz"),
     "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz"],
    capture_output=False)

# ── convert HF safetensors -> f16 GGML ──────────────────────────────────────
CONV_OUT = MODELS / "ggml"
CONV_OUT.mkdir(exist_ok=True)
env = os.environ.copy()
env["USE_TF"] = "0"
kh.step("convert.begin")
with kh.build_heartbeat("convert.f16"):
    r = run([sys.executable, str(REPO / "models" / "convert-h5-to-ggml.py"),
             str(SRC), str(ASSETS), str(CONV_OUT)], env=env, timeout=2400)
if r.returncode != 0:
    kh.step("convert.FAIL", tail=(r.stderr or "")[-500:]); raise SystemExit(1)
f16 = CONV_OUT / "ggml-model.bin"
f16_named = CONV_OUT / f"{NAME}-f16.bin"
f16.rename(f16_named)
kh.step("convert.done", gb=round(f16_named.stat().st_size / 1e9, 2))

# ── quantize f16 -> q4_k ────────────────────────────────────────────────────
q4k = CONV_OUT / f"{NAME}-q4_k.bin"
kh.step("quantize.begin")
with kh.build_heartbeat("quantize.q4_k"):
    r = run([str(QUANT), str(f16_named), str(q4k), "q4_k"], timeout=1800)
if not q4k.exists():
    kh.step("quantize.FAIL", tail=(r.stderr or "")[-500:]); raise SystemExit(1)
kh.step("quantize.done", gb=round(q4k.stat().st_size / 1e9, 2))

# ── validate: transcribe a Japanese clip, require Japanese output ───────────
JA = REPO / "tools" / "kaggle" / "whisper-ja-760M-convert" / "ja_test.wav"


def has_japanese(t):
    return any(0x3040 <= ord(c) <= 0x30FF or 0x3400 <= ord(c) <= 0x9FFF for c in t)


def transcribe(model):
    r = run([str(CLI), "-m", str(model), "--backend", "whisper", "-l", "ja",
             "-f", str(JA), "--no-prints"], timeout=1200)
    lines = [ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip()]
    txt = " ".join(lines)
    return r.returncode, txt


results = {}
for tag, model in (("f16", f16_named), ("q4_k", q4k)):
    with kh.build_heartbeat(f"validate.{tag}"):
        rc, txt = transcribe(model)
    ja = has_japanese(txt)
    ok = rc == 0 and ja and len(txt.strip()) >= 4
    results[tag] = {"rc": rc, "japanese": ja, "len": len(txt), "pass": ok, "transcript": txt[:400]}
    kh.step(f"validate.{tag}", ok=ok, japanese=ja, transcript=txt[:200])
    print(f"  [{tag}] pass={ok} ja={ja} :: {txt[:200]!r}", flush=True)

# ── upload to PRIVATE HF repo (source has no license) ───────────────────────
api = HfApi(token=TOKEN)
if results.get("q4_k", {}).get("pass"):
    api.create_repo(repo_id=HF_REPO, repo_type="model", private=True, exist_ok=True)
    readme = (
        f"# {SRC_REPO} — CrispASR whisper GGML (PRIVATE)\n\n"
        f"CrispASR whisper GGML conversion of [{SRC_REPO}](https://huggingface.co/{SRC_REPO}) "
        f"(Whisper large-v3-turbo arch, custom {cfg_json.get('vocab_size')}-token Japanese vocab).\n\n"
        f"**PRIVATE / evaluation only — the source model has no license.**\n\n"
        f"- `{NAME}-f16.bin` — f16 GGML\n- `{NAME}-q4_k.bin` — q4_k quant\n\n"
        f"Run: `crispasr --backend whisper -m {NAME}-q4_k.bin -l ja -f audio.wav`\n"
    )
    (CONV_OUT / "README.md").write_text(readme)
    for fn in (f"{NAME}-f16.bin", f"{NAME}-q4_k.bin", "README.md"):
        with kh.build_heartbeat(f"upload.{fn}"):
            api.upload_file(path_or_fileobj=str(CONV_OUT / fn), path_in_repo=fn,
                            repo_id=HF_REPO, repo_type="model",
                            commit_message=f"Add {fn} (CrispASR whisper GGML, ASR-validated JA)")
    kh.step("uploaded", repo=HF_REPO, private=True)
    uploaded = True
else:
    kh.step("upload.SKIP", reason="q4_k validation did not produce Japanese output")
    uploaded = False

RESULTS = {"src": SRC_REPO, "hf_repo": HF_REPO, "private": True, "uploaded": uploaded,
           "vocab_size": cfg_json.get("vocab_size"), "results": results,
           "wall_s": round(time.time() - _T0, 1)}
(OUT / "results.json").write_text(json.dumps(RESULTS, indent=2, ensure_ascii=False))
print(json.dumps({"step": "done", "uploaded": uploaded, "results": results}, ensure_ascii=False), flush=True)
kh.step("done", uploaded=uploaded)
