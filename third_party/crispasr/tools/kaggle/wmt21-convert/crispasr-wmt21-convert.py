# %% [markdown]
# # CrispASR — re-convert wmt21-dense-24-wide-en-x with the fixed SP-BPE tokenizer
#
# facebook/wmt21-dense-24-wide-en-x runs through the m2m100 backend. Its GGUF was
# built with the old converter, which (a) mis-scored the SP-BPE merge ranks (id
# offset instead of by-string) and (b) omitted ~390 SP pieces needed as
# intermediate BPE merges — so tokenization diverged from HF and translations
# degraded. The converter is fixed (commit on CrispASR main); this kernel
# re-converts wmt21 with it.
#
# Runs on Kaggle (not the 16 GB Mac) because the model is 18.77 GB (4.7B params,
# 24L d=2048). The converter uses torch.load(mmap=True), so RAM stays low; the
# job is disk + a fast HF link (GPU kernel → internet).
#
# Steps: download model → convert (fixed converter) → GGUF-level tokenizer check
# (intermediates present + scores match SP by-string) → quantize q4_k → upload
# q4_k (replace shipped) + f16 (exact) to cstr/wmt21-dense-24-wide-en-x-GGUF.

# %% [code]
import os
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")

# ── Kaggle regime: clone CrispASR (fixed converter) + import harness ───────────
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
REPO = WORK / "CrispASR"
if not REPO.exists():
    try:
        subprocess.check_call(
            ["git", "clone", "--depth", "1", CRISPASR_URL, str(REPO)])
        sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    except Exception:
        pass  # fall through to bundled harness
if str(REPO / "tools" / "kaggle") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh

kh.init_progress()
step = kh.step
step("script.start")

TOKEN = kh.resolve_hf_token("HF_TOKEN")
step("hf_token.resolved", have=bool(TOKEN))

# ── deps (torch is pre-installed; only small extras) ──────────────────────────
step("install-deps.begin")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                       "gguf", "sentencepiece", "huggingface_hub", "hf_transfer"])
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import HfApi, snapshot_download
step("install-deps.done")

MODEL_ID = "facebook/wmt21-dense-24-wide-en-x"
DST_REPO = "cstr/wmt21-dense-24-wide-en-x-GGUF"


def free_gb(p):
    try:
        return round(shutil.disk_usage(p).free / 1e9, 1)
    except Exception:
        return 0.0


# ── pick the roomiest disk (the model is ~19 GB; /kaggle/working caps ~20 GB) ──
_cands = [("/tmp/wmt21", "/tmp"), ("/kaggle/temp/wmt21", "/kaggle/temp"),
          (str(WORK / "wmt21"), str(WORK))]
_best = max(_cands, key=lambda c: free_gb(c[1]))
BASE = Path(_best[0])
BASE.mkdir(parents=True, exist_ok=True)
step("disk.chosen", dir=str(BASE), free_gb=free_gb(_best[1]),
     probe={d: free_gb(d) for _, d in _cands})

MODEL_DIR = BASE / "model"
F16 = BASE / "wmt21-dense-24-wide-en-x-f16.gguf"
Q4K = BASE / "wmt21-dense-24-wide-en-x-q4_k.gguf"
CONVERTER = REPO / "models" / "convert-m2m100-to-gguf.py"

# ── 1. download the source model ──────────────────────────────────────────────
step("download.begin", model=MODEL_ID, free_gb=free_gb(str(BASE)))
with kh.build_heartbeat("download"):
    snapshot_download(
        repo_id=MODEL_ID, local_dir=str(MODEL_DIR), token=TOKEN,
        allow_patterns=["config.json", "vocab.json", "sentencepiece.bpe.model",
                        "tokenizer_config.json", "special_tokens_map.json",
                        "generation_config.json", "pytorch_model.bin"])
step("download.done", free_gb=free_gb(str(BASE)))

# ── 2. convert with the FIXED converter ───────────────────────────────────────
step("convert.begin", converter=str(CONVERTER))
with kh.build_heartbeat("convert"):
    kh.sh_with_progress(f"{sys.executable} {CONVERTER} --input {MODEL_DIR} --output {F16}")
assert F16.is_file(), "f16 GGUF not produced"
step("convert.done", f16_gb=round(F16.stat().st_size / 1e9, 2), free_gb=free_gb(str(BASE)))

# ── 3. GGUF-level tokenizer validation (no inference needed) ──────────────────
step("validate.begin")
import numpy as np
import sentencepiece as spm
from gguf import GGUFReader

r = GGUFReader(str(F16))


def _kv(name):
    return r.get_field(name)


toks_f = _kv("tokenizer.ggml.tokens")
scr_f = _kv("tokenizer.ggml.scores")
n_tok = len(toks_f.data)


def _kv_u32(name):  # robust scalar read across gguf-py versions
    try:
        f = _kv(name)
        return int(np.array(f.parts[f.data[0]]).flatten()[0])
    except Exception:
        return 0


vocab_size = _kv_u32("m2m100.vocab_size")
n_intermediate = (n_tok - vocab_size) if vocab_size else 0
toks = [bytes(toks_f.parts[i]).decode("utf-8", "replace") for i in toks_f.data]
scores = [float(np.array(scr_f.parts[i])[0]) for i in scr_f.data]
tid = {t: i for i, t in enumerate(toks)}

sp = spm.SentencePieceProcessor()
sp.Load(str(MODEL_DIR / "sentencepiece.bpe.model"))

# (b) scores aligned by string vs SP; (c) BPE reproduces sp.EncodeAsPieces (below).
mism = 0
checked = 0
for t in toks:
    sid = sp.piece_to_id(t)
    if sid == sp.unk_id() and t != sp.id_to_piece(sp.unk_id()):
        continue
    checked += 1
    if abs(scores[tid[t]] - sp.get_score(sid)) > 0.5:
        mism += 1


def bpe(text):
    syms = list("▁" + text.replace(" ", "▁"))
    while True:
        bi, bs = -1, -1e30
        for k in range(len(syms) - 1):
            m = syms[k] + syms[k + 1]
            if m in tid and scores[tid[m]] > bs:
                bs, bi = scores[tid[m]], k
        if bi < 0:
            break
        syms[bi] += syms[bi + 1]
        del syms[bi + 1]
    return syms


bpe_match = 0
sents = ["she quickly finished the assignment yesterday",
         "the delayed international shipment arrived unexpectedly",
         "reconstruction of the neighbourhood took years"]
for s in sents:
    if bpe(s) == sp.EncodeAsPieces(s):
        bpe_match += 1
# Gate on the real signals: scores match SP by-string AND BPE reproduces SP.
validate_ok = mism == 0 and bpe_match == len(sents)
step("validate.done", ok=validate_ok, n_tok=n_tok, vocab_size=vocab_size,
     n_intermediate=n_intermediate, score_mismatch=f"{mism}/{checked}",
     bpe_match=f"{bpe_match}/{len(sents)}")
if not validate_ok:
    step("VALIDATION-FAILED", note="tokenizer did not reproduce SentencePiece — NOT uploading")
    sys.exit(1)

# free the 18.77 GB source before quantizing
del r
shutil.rmtree(MODEL_DIR, ignore_errors=True)
step("source.deleted", free_gb=free_gb(str(BASE)))

# ── 4. quantize f16 → q4_k (v0.8.6 tarball ships crispasr-quantize) ───────────
RELEASE = "v0.8.6"
QUANT = WORK / "bin" / "crispasr-quantize"
QUANT.parent.mkdir(exist_ok=True)
step("quant-binary.download.begin", release=RELEASE)
subprocess.check_call(
    "wget -q https://github.com/CrispStrobe/CrispASR/releases/download/"
    f"{RELEASE}/crispasr-linux-x86_64.tar.gz -O /tmp/c.tar.gz && "
    f"tar -xzf /tmp/c.tar.gz -C {QUANT.parent} --strip-components=1", shell=True)
QUANT.chmod(0o755)
step("quant-binary.download.done", have=QUANT.is_file())

step("quantize.begin", free_gb=free_gb(str(BASE)))
with kh.build_heartbeat("quantize"):
    p = subprocess.run([str(QUANT), str(F16), str(Q4K), "q4_k"],
                       capture_output=True, text=True, timeout=3600)
q4k_ok = p.returncode == 0 and Q4K.is_file()
step("quantize.done", ok=q4k_ok, q4k_gb=round(Q4K.stat().st_size / 1e9, 2) if q4k_ok else 0,
     tail=p.stdout[-300:] if not q4k_ok else "")
if not q4k_ok:
    sys.exit(1)

# ── 5. upload q4_k (replace shipped default) + f16 (exact) ────────────────────
api = HfApi(token=TOKEN)


def upload(path, name):
    step("upload.begin", file=name, gb=round(path.stat().st_size / 1e9, 2))
    with kh.build_heartbeat(f"upload:{name}"):
        api.upload_file(path_or_fileobj=str(path), path_in_repo=name,
                        repo_id=DST_REPO, repo_type="model",
                        commit_message=f"Re-convert with fixed SP-BPE tokenizer: {name}")
    step("upload.done", file=name)


upload(Q4K, "wmt21-dense-24-wide-en-x-q4_k.gguf")
upload(F16, "wmt21-dense-24-wide-en-x-f16.gguf")

step("script.done", uploaded=[DST_REPO], free_gb=free_gb(str(BASE)))
print("DONE — wmt21 re-converted + uploaded with the fixed tokenizer")
