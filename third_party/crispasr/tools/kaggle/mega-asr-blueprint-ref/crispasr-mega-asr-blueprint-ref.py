"""
CrispASR — Mega-ASR bf16 blueprint check for #218 long-form drift

Our 4-bit mega-asr GGUFs loop on the un-chunked 145 s clip ("come on,
come on, …" 2-gram cycles) even with the Q8_0 audio tower. Question: does
the bf16 BLUEPRINT do the same on this (deliberately degraded / in-the-wild)
audio, or is our port/quant at fault?

Mega-ASR = zhifeixie/Mega-ASR: a LoRA adaptation of Qwen/Qwen3-ASR-1.7B
(adapter at subfolder `mega-asr-merged`), driven through the official
`qwen_asr` package. This kernel runs THREE configurations on the raw clip
(RAW decode via `_infer_asr`, BEFORE parse_asr_output's own repetition
cleanup) and prints loop metrics incl. a PHRASE-CYCLE check (unigram runs
miss 2-gram loops — the mega failure mode):

  1. Qwen3-ASR-1.7B base, bf16       — does the base loop long-form?
  2. base + Mega LoRA merged, bf16   — does the shipped mega config loop?
  3. both again with forced language  ("English") — the prefill contract.

Outcome drives PLAN '#218 arc — remaining open threads' item 1.
"""

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path("/kaggle/working")
REPO = Path("/tmp/CrispASR")
AUDIO_DIR = ROOT / "audio"
OUT_DIR = ROOT / "results"
for d in (AUDIO_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

if not REPO.exists():
    subprocess.run(
        f"git clone --depth 1 --branch {os.environ.get('CRISPASR_REF', 'main')} "
        f"https://github.com/CrispStrobe/CrispASR.git {REPO}", shell=True, check=True)
sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress(progress_path=str(ROOT / "progress.jsonl"))
kh.resolve_hf_token()
kh.step("script.start")

# qwen_asr pins transformers==4.57.6 (vendors its own modeling code).
kh.step("deps.begin")
# transformers 4.57.6 needs the pre-1.0 huggingface_hub API. A plain
# version-range downgrade left a MIXED install on Kaggle (old
# _snapshot_download.py + new constants.py → AttributeError) — force a
# coherent reinstall of a known-good hub as the LAST dep step.
kh.sh_with_progress("pip install -q 'transformers==4.57.6' qwen_asr peft librosa soundfile")
kh.sh_with_progress("pip install -q --force-reinstall 'huggingface_hub==0.36.0' hf_transfer")
# kaggle_harness's progress pushes already imported the PREINSTALLED hub —
# its module objects (constants, …) stay cached in sys.modules and mix with
# the 0.36 files imported later (AttributeError on HF_HUB_ENABLE_HF_TRANSFER,
# seen twice). Purge so transformers/qwen_asr import a coherent 0.36.
for _m in [k for k in list(sys.modules) if k.startswith("huggingface_hub")]:
    del sys.modules[_m]
kh.step("deps.done")

WAV_ZIP = AUDIO_DIR / "t32-145s.wav.zip"
WAV = AUDIO_DIR / "t32-145s.wav"
if not WAV.exists():
    kh.sh_with_progress(
        f"curl -sL -o {WAV_ZIP} "
        "https://github.com/user-attachments/files/29652411/t32-145s.wav.zip")
    with zipfile.ZipFile(WAV_ZIP) as z:
        z.extractall(AUDIO_DIR)

import librosa  # noqa: E402
import torch  # noqa: E402
from qwen_asr import Qwen3ASRModel  # noqa: E402

device = "cpu"
try:
    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")  # smoke-test (P100 has no kernels)
        device = "cuda"
except Exception as e:
    print(f"[cuda-smoke] CPU fallback: {e}", flush=True)

audio, _ = librosa.load(str(WAV), sr=16000, mono=True)


def loop_metrics(text: str) -> dict:
    words = [w.strip(".,!?;:—-\"'").lower() for w in text.split() if w.strip()]
    max_uni, cur, prev = 0, 0, None
    for w in words:
        cur = cur + 1 if w == prev else 1
        prev = w
        max_uni = max(max_uni, cur)
    # Phrase-cycle check (n = 2..4): longest immediate repetition of an
    # n-gram — catches "come on, come on, …" that unigram runs miss.
    max_cycle = 0
    for n in (2, 3, 4):
        i = 0
        while i + n <= len(words):
            run = 1
            j = i + n
            while j + n <= len(words) and words[j:j + n] == words[i:i + n]:
                run += 1
                j += n
            max_cycle = max(max_cycle, run if run > 1 else 0)
            i = i + 1 if run == 1 else j
    return {"chars": len(text), "n_words": len(words),
            "max_unigram_run": max_uni, "max_phrase_cycle": max_cycle}


def run_config(label: str, wrapper, language) -> dict:
    raw = wrapper._infer_asr([""], [audio], [language])[0]
    m = loop_metrics(raw)
    kh.step(f"run.{label}", language=str(language), **m)
    (OUT_DIR / f"{label}.txt").write_text(raw)
    print(f"=== {label} (language={language}) {m}")
    print(raw[:900], flush=True)
    return {"metrics": m, "raw_head": raw[:400]}


results = {}
kh.step("base.load.begin")
wrapper = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B", dtype=torch.bfloat16, device_map=device,
    max_inference_batch_size=1, max_new_tokens=512)
kh.step("base.load.done", device=device)
results["base_auto"] = run_config("base_auto", wrapper, None)
results["base_english"] = run_config("base_english", wrapper, "English")

kh.step("lora.merge.begin")
# The adapter declares target_modules='.*' (peft chokes wrapping container
# modules) with keys rooted at 'base_model.model.<module path>'. Merge by
# hand: W += (alpha/r) * B @ A ; r=24, alpha=24 -> scale 1.0.
from huggingface_hub import hf_hub_download  # noqa: E402
from safetensors import safe_open  # noqa: E402

_ad_cfg = json.load(open(hf_hub_download(
    "zhifeixie/Mega-ASR", "mega-asr-merged/adapter_config.json")))
_scale = _ad_cfg["lora_alpha"] / _ad_cfg["r"]
_ad_path = hf_hub_download("zhifeixie/Mega-ASR", "mega-asr-merged/adapter_model.safetensors")
_merged = 0
with safe_open(_ad_path, framework="pt") as _f:
    _keys = [k for k in _f.keys() if k.endswith(".lora_A.weight")]
    for _ka in _keys:
        _kb = _ka.replace(".lora_A.", ".lora_B.")
        _mod_path = _ka[len("base_model.model."):-len(".lora_A.weight")]
        # The adapter was trained against a flatter nesting for the LLM
        # part ('thinker.layers.*'); the qwen_asr modeling nests it under
        # 'thinker.model.layers.*'. Resolve with candidate rewrites.
        _mod = None
        for _cand in (_mod_path,
                      _mod_path.replace("thinker.layers.", "thinker.model.layers.", 1),
                      _mod_path.replace("thinker.embed_tokens", "thinker.model.embed_tokens", 1)):
            try:
                _mod = wrapper.model.get_submodule(_cand)
                break
            except AttributeError:
                continue
        if _mod is None:
            print(f"[lora-merge] UNRESOLVED module path: {_mod_path}", flush=True)
            continue
        _A = _f.get_tensor(_ka).to(torch.float32)
        _B = _f.get_tensor(_kb).to(torch.float32)
        with torch.no_grad():
            _w = _mod.weight
            _w += (_scale * (_B @ _A)).to(_w.dtype).to(_w.device)
        _merged += 1
wrapper.model.eval()
kh.step("lora.merge.done", merged_pairs=_merged, scale=_scale)
results["mega_auto"] = run_config("mega_auto", wrapper, None)
results["mega_english"] = run_config("mega_english", wrapper, "English")

(OUT_DIR / "summary.json").write_text(json.dumps(results, indent=1))
print("=" * 72)
for k, v in results.items():
    print(k, v["metrics"])
kh.step("summary", **{k: v["metrics"]["max_phrase_cycle"] for k, v in results.items()})
kh._push_progress_to_hf(force=True)
kh.step("script.end")
