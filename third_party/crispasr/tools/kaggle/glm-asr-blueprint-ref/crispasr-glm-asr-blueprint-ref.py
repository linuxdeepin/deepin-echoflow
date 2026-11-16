"""
CrispASR — GLM-ASR-Nano HF-blueprint ground truth for #218 long-form parity

Runs the REAL `GlmAsrForConditionalGeneration` (transformers >= 5) on the
canonical 145 s clip and on jfk.wav, greedy, bf16, and dumps:

  - the processor's prompt input_ids (apply_transcription_request)
  - input_features shape (number of 30 s encoder windows) + valid audio tokens
  - the raw greedy generation ids + decoded text (before AND after the
    processor's strip_prefix cleanup)

This is the reference our C++ port must match: the blueprint batch-encodes
all 30 s windows and decodes the WHOLE clip in ONE generate() call (processor
max_audio_len=655 s), unlike the current C++ path (independent 30 s slices).
Results land in /kaggle/working/results/ + progress.jsonl on HF.
"""

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path("/kaggle/working")
# Repo goes to /tmp so kernels_output stays small (results/ + progress only).
REPO = Path("/tmp/CrispASR")
AUDIO_DIR = ROOT / "audio"
OUT_DIR = ROOT / "results"
for d in (AUDIO_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
if not REPO.exists():
    subprocess.run(
        f"git clone --depth 1 --branch {CRISPASR_REF} "
        f"https://github.com/CrispStrobe/CrispASR.git {REPO}",
        shell=True, check=True,
    )
sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress(progress_path=str(ROOT / "progress.jsonl"))
kh.resolve_hf_token()
kh.step("script.start")

# ── deps: transformers >= 5 for GlmAsrForConditionalGeneration ───────────
kh.step("deps.begin")
kh.sh_with_progress("pip install -q -U 'transformers>=5.13' accelerate soundfile librosa")

# NOTE: on a P100 draw (sm_60) the preinstalled torch has no kernels and a
# cu118 force-reinstall breaks transformers-5.x imports (tested: v3 run) —
# so we simply run on CPU there via the smoke-test fallback below.
import transformers  # noqa: E402

kh.step("deps.done", transformers=transformers.__version__)

# ── audio ────────────────────────────────────────────────────────────────
WAV_ZIP = AUDIO_DIR / "t32-145s.wav.zip"
WAV = AUDIO_DIR / "t32-145s.wav"
if not WAV.exists():
    kh.sh_with_progress(
        f"curl -sL -o {WAV_ZIP} "
        "https://github.com/user-attachments/files/29652411/t32-145s.wav.zip"
    )
    with zipfile.ZipFile(WAV_ZIP) as z:
        z.extractall(AUDIO_DIR)
JFK = REPO / "samples" / "jfk.wav"

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from transformers import AutoProcessor, GlmAsrForConditionalGeneration  # noqa: E402

MODEL_ID = "zai-org/GLM-ASR-Nano-2512"
kh.step("model.load.begin")
processor = AutoProcessor.from_pretrained(MODEL_ID)
# cu118 reinstall for sm_60 already happened pre-import when needed; if the
# GPU is still unusable, run on CPU (bf16 CPU is correctness-equivalent for
# the 1.5B nano model; the GPU is only a speed-up).
device = "cpu"
try:
    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")  # smoke-test the kernel image
        device = "cuda"
except Exception as e:
    print(f"[cuda-smoke] falling back to CPU: {e}", flush=True)
model = GlmAsrForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map=device
)
model.eval()
kh.step("model.load.done", device=device)


def transcription_inputs(audio: "np.ndarray"):
    """processor.apply_transcription_request when available (matches the
    blueprint helper exactly); manual chat-template fallback otherwise."""
    if hasattr(processor, "apply_transcription_request"):
        return processor.apply_transcription_request(audio=audio)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": getattr(processor, "default_transcription_prompt",
                                                 "Please transcribe this audio into text")},
            ],
        }
    ]
    return processor.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True, return_dict=True
    )


def run_case(name: str, wav_path: Path) -> dict:
    kh.step(f"case.{name}.begin")
    audio, _sr = librosa.load(str(wav_path), sr=16000, mono=True)
    inputs = transcription_inputs(audio)
    inputs = inputs.to(device)
    if "input_features" in inputs:  # match model dtype (bf16)
        inputs["input_features"] = inputs["input_features"].to(model.dtype)
    in_ids = inputs["input_ids"][0].tolist()
    feats = inputs["input_features"]
    n_audio_tok = int((inputs["input_ids"][0] == processor.audio_token_id).sum())
    with torch.no_grad():
        out = model.generate(
            **inputs, do_sample=False, max_new_tokens=500,
        )
    gen_ids = out[0][inputs["input_ids"].shape[1]:].tolist()
    raw_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
    clean_text = processor._strip_assistant_prefix_and_quotes(raw_text)
    res = {
        "audio_s": round(len(audio) / 16000.0, 2),
        "n_windows": list(feats.shape)[0],
        "features_shape": list(feats.shape),
        "prompt_len": len(in_ids),
        "n_audio_tokens": n_audio_tok,
        "prompt_ids": in_ids,
        "gen_ids": gen_ids,
        "raw_text": raw_text,
        "clean_text": clean_text,
    }
    (OUT_DIR / f"{name}.json").write_text(json.dumps(res, indent=1))
    kh.step(
        f"case.{name}.done",
        n_windows=res["n_windows"],
        prompt_len=res["prompt_len"],
        n_audio_tokens=n_audio_tok,
        gen_tokens=len(gen_ids),
        raw_chars=len(raw_text),
    )
    print(f"--- {name}: prompt head (40): {in_ids[:40]}")
    print(f"--- {name}: prompt tail (25): {in_ids[-25:]}")
    print(f"--- {name}: RAW text:\n{raw_text}\n")
    return res


results = {}
results["jfk"] = run_case("jfk", JFK)
results["t32"] = run_case("t32", WAV)

# Loop metric on the raw long-form text — does the BLUEPRINT loop?
words = [w.strip(".,!?;:\"'").lower() for w in results["t32"]["raw_text"].split() if w.strip()]
max_uni, cur, prev = 0, 0, None
for w in words:
    cur = cur + 1 if w == prev else 1
    prev = w
    max_uni = max(max_uni, cur)
kh.step("summary", t32_max_unigram_run=max_uni, t32_raw_chars=len(results["t32"]["raw_text"]))
print("=" * 72)
print(f"BLUEPRINT t32 max_unigram_run={max_uni}  raw_chars={len(results['t32']['raw_text'])}")
print("=" * 72)

kh._push_progress_to_hf(force=True)
kh.step("script.end")
