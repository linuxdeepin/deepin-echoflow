#!/usr/bin/env python3
"""
Kaggle kernel: Irodori-TTS crispasr-diff reference dump.

Runs the Python Irodori-TTS model and captures stage-by-stage
intermediate activations for diff-testing against the C++ runtime.

Outputs irodori-tts-ref-stages.gguf with:
  - text_embedding, text_state
  - timestep_embed, cond_embed
  - dit_in_proj, dit_block_0, v_pred_step0, ode_step_0

Uploads to cstr/irodori-tts-GGUF on HuggingFace.
"""

import gc
import os
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
os.chdir(str(WORK))

PROGRESS = WORK / "progress.txt"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(PROGRESS, "a") as f:
        f.write(line + "\n")


log("Kernel started — Irodori-TTS reference dump")

# ── Clone CrispASR ──
REPO = WORK / "CrispASR"
if not REPO.exists():
    log("Cloning CrispASR (feat/irodori-tts)...")
    subprocess.check_call([
        "git", "clone", "--depth", "1", "--branch", "feat/irodori-tts",
        "https://github.com/CrispStrobe/CrispASR.git", str(REPO),
    ])

sys.path.insert(0, str(REPO / "tools" / "kaggle"))
try:
    import kaggle_harness as kh
    kh.init_progress()
except Exception:
    pass

log("CrispASR cloned OK")

# ── Install deps ──
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "gguf", "safetensors", "transformers", "sentencepiece",
    "huggingface_hub",
])
log("deps installed")

# ── HF auth ──
token = None
for p in ["/kaggle/input/crispasr-hf-token/hf_token.txt",
          "/kaggle/input/datasets/${KAGGLE_ACCOUNT}/crispasr-hf-token/hf_token.txt"]:
    if os.path.exists(p):
        token = open(p).read().strip()
        break
if token:
    os.environ["HF_TOKEN"] = token

# ── Run the reference dump ──
log("Running dump_reference.py --backend irodori-tts")

os.environ["IRODORI_TEST_TEXT"] = "こんにちは、世界。"
os.environ["IRODORI_SEED"] = "42"
os.environ["IRODORI_ODE_STEPS"] = "2"

ref_output = WORK / "irodori-tts-ref-stages.gguf"

# Create dummy audio file (TTS doesn't use it but dump_reference requires it)
import wave, struct
dummy_wav = REPO / "samples" / "jfk.wav"
if not dummy_wav.exists():
    dummy_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dummy_wav), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<" + "h" * 16000, *([0] * 16000)))

try:
    subprocess.check_call([
        sys.executable, "tools/dump_reference.py",
        "--backend", "irodori-tts",
        "--model-dir", "Aratako/Irodori-TTS-500M-v3",
        "--audio", "samples/jfk.wav",
        "--output", str(ref_output),
    ], cwd=str(REPO))
    log(f"Reference dump: {ref_output} ({ref_output.stat().st_size / 1024:.1f} KB)")
except subprocess.CalledProcessError as e:
    log(f"dump_reference.py failed: {e}")
    # Fallback: try running the backend directly
    log("Trying direct backend import...")
    try:
        sys.path.insert(0, str(REPO / "tools"))
        from reference_backends import irodori_tts as ref_backend
        import numpy as np
        stages = set(ref_backend.DEFAULT_STAGES)
        results = ref_backend.dump(
            model_dir=Path("Aratako/Irodori-TTS-500M-v3"),
            audio=np.zeros(16000, dtype=np.float32),
            stages=stages,
            max_new_tokens=0,
        )
        log(f"Direct dump captured {len(results)} stages")

        # Write to GGUF
        from gguf import GGUFWriter
        writer = GGUFWriter(str(ref_output), "irodori-tts-ref")
        writer.add_string("irodori.ref.test_text", os.environ.get("IRODORI_TEST_TEXT", ""))
        for name, arr in results.items():
            writer.add_tensor(f"ref.{name}", arr.astype(np.float32))
            log(f"  ref.{name}: {arr.shape}")
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        log(f"Reference GGUF: {ref_output} ({ref_output.stat().st_size / 1024:.1f} KB)")
    except Exception as e2:
        import traceback
        log(f"Direct dump also failed: {e2}")
        log(traceback.format_exc())

# ── Upload to HF ──
if ref_output.exists() and ref_output.stat().st_size > 1000:
    log("Uploading reference GGUF to HuggingFace...")
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(ref_output),
            path_in_repo="irodori-tts-ref-stages.gguf",
            repo_id="cstr/irodori-tts-GGUF",
            repo_type="model",
        )
        log("Upload complete!")
    except Exception as e:
        log(f"Upload failed: {e}")
else:
    log("No reference GGUF to upload")

# Cleanup
import shutil
if REPO.exists():
    shutil.rmtree(str(REPO), ignore_errors=True)

log("=== DONE ===")
