#!/usr/bin/env python3
"""VibeVoice-ASR-BitNet diff harness — per-stage cosine parity for each quant variant.

1. Generate reference GGUF from Python (microsoft/VibeVoice-ASR-BitNet safetensors)
2. Generate all 6 quant variant GGUFs
3. Run vibevoice-test-stages for each variant vs reference
4. Upload ref GGUF + all variant GGUFs to HF

Uses the BitNet (1.5B) model — fits comfortably in 30 GB Kaggle RAM.
"""

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

WORK = Path("/kaggle/working")
REPO = Path("/kaggle/temp/CrispASR")
BUILD = REPO / "build"
MODELS = Path("/kaggle/temp/models")
REF_GGUF = MODELS / "vibevoice-bitnet-ref.gguf"
RESULTS = WORK / "diff-results.json"

VARIANTS = [
    ("vibevoice-asr-bitnet-tq2",      "q8_0", "f16"),
    ("vibevoice-asr-bitnet-embed-q8",  "q8_0", "q8_0"),
    ("vibevoice-asr-bitnet-vae-q5",    "q5_0", "f16"),
    ("vibevoice-asr-bitnet-both-q5",   "q5_0", "q8_0"),
    ("vibevoice-asr-bitnet-vae-q4",    "q4_0", "f16"),
    ("vibevoice-asr-bitnet-aggro",     "q4_0", "q8_0"),
]

HF_REPO = "cstr/vibevoice-asr-bitnet-GGUF"

# ── Clone + harness ──────────────────────────────────────────────────────

CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
if not REPO.exists():
    try:
        subprocess.check_call(
            ["git", "clone", "--depth", "1", CRISPASR_URL, str(REPO)])
        subprocess.check_call(
            ["git", "submodule", "update", "--init", "ggml"], cwd=str(REPO))
        sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    except Exception:
        pass

if str(REPO / "tools" / "kaggle") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import kaggle_harness as kh  # noqa: E402
kh.init_progress()

_orig_excepthook = sys.excepthook
def _crash_handler(exc_type, exc_val, exc_tb):
    msg = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
    try:
        (WORK / "error.txt").write_text(msg)
    except Exception:
        pass
    _orig_excepthook(exc_type, exc_val, exc_tb)
sys.excepthook = _crash_handler

# ── Build ────────────────────────────────────────────────────────────────

print("Installing build toolchain + deps")
kh.install_build_toolchain()
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "safetensors", "transformers", "gguf", "librosa", "soundfile"],
               check=False)

hf_token = kh.resolve_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    print("HF token resolved")

cmake_flags = " ".join(kh.cache_and_link_flags())
crispasr_flags = " ".join(kh.crispasr_cmake_flags())
cmake_cmd = (
    f"cmake -G Ninja -S {REPO} -B {BUILD} "
    f"-DCMAKE_BUILD_TYPE=Release "
    f"{cmake_flags} {crispasr_flags} "
    f"-DCRISPASR_BUILD_TESTS=OFF "
    f"-DCRISPASR_BUILD_EXAMPLES=ON "
    f"-DCRISPASR_BUILD_SERVER=OFF"
)
print(f"cmake: {cmake_cmd}")
subprocess.check_call(cmake_cmd, shell=True)

jobs = kh.safe_build_jobs(gpu=True)
print(f"Building with {jobs} jobs")
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} "
        f"--target crispasr-cli --target vibevoice-test-stages -j{jobs}")

CRISPASR_BIN = BUILD / "bin" / "crispasr"
STAGES_BIN = BUILD / "bin" / "vibevoice-test-stages"
assert CRISPASR_BIN.exists(), f"Build failed: {CRISPASR_BIN}"
assert STAGES_BIN.exists(), f"Build failed: {STAGES_BIN}"
print(f"Build OK")

MODELS.mkdir(parents=True, exist_ok=True)

# ── Step 1: Generate reference GGUF from Python ─────────────────────────

if not REF_GGUF.exists():
    print("\n=== Generating reference GGUF from Python ===")
    # Download the BitNet model
    from huggingface_hub import snapshot_download
    model_dir = snapshot_download("microsoft/VibeVoice-ASR-BitNet")
    print(f"  Model dir: {model_dir}")

    # Run dump_reference.py
    dump_cmd = [
        sys.executable, str(REPO / "tools" / "dump_reference.py"),
        "--backend", "vibevoice",
        "--model-dir", model_dir,
        "--audio", str(REPO / "samples" / "jfk.wav"),
        "--output", str(REF_GGUF),
    ]
    print(f"  Dumping: {' '.join(dump_cmd)}")
    r = subprocess.run(dump_cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f"  DUMP FAILED: {r.stderr[-500:]}")
        (WORK / "error.txt").write_text(f"dump_reference failed:\n{r.stderr}")
        sys.exit(1)
    print(f"  Reference GGUF: {REF_GGUF} ({REF_GGUF.stat().st_size / 1e6:.1f} MB)")
else:
    print(f"  Using cached reference: {REF_GGUF}")

# Upload ref GGUF to HF
if hf_token and REF_GGUF.exists():
    print("  Uploading reference GGUF to HF...")
    try:
        from huggingface_hub import HfApi
        HfApi(token=hf_token).upload_file(
            path_or_fileobj=str(REF_GGUF),
            path_in_repo="vibevoice-bitnet-ref.gguf",
            repo_id=HF_REPO, repo_type="model")
        print("  Uploaded")
    except Exception as e:
        print(f"  Upload failed: {e}")

kh.step("ref-gguf-done")

# ── Step 2: Generate + diff each variant ─────────────────────────────────

CONVERTER = REPO / "models" / "convert-vibevoice-bitnet-to-gguf.py"
results = []

for label, vae_q, embed_q in VARIANTS:
    fname = f"{label}.gguf"
    out_gguf = MODELS / fname
    print(f"\n=== {label} (VAE={vae_q}, embed={embed_q}) ===")

    # Convert
    if not out_gguf.exists():
        conv_env = os.environ.copy()
        conv_env["TMPDIR"] = str(MODELS)
        cmd = [
            sys.executable, str(CONVERTER),
            "--input", "microsoft/VibeVoice-ASR-BitNet",
            "--output", str(out_gguf),
            "--vae-quant", vae_q, "--embed-quant", embed_q,
        ]
        print(f"  Converting...")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=conv_env)
        if r.returncode != 0:
            print(f"  CONVERT FAILED: {r.stderr[-300:]}")
            results.append({"label": label, "error": "convert failed"})
            continue
    file_size_mb = out_gguf.stat().st_size / (1024 * 1024)
    print(f"  Size: {file_size_mb:.1f} MB")

    # Run vibevoice-test-stages
    print(f"  Running diff harness...")
    cmd = [str(STAGES_BIN), str(out_gguf), str(REF_GGUF)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    print(r.stdout[-2000:] if r.stdout else "(no stdout)")
    if r.stderr:
        # Filter just the stage lines
        for line in r.stderr.split("\n"):
            if "cos" in line.lower() or "pass" in line.lower() or "fail" in line.lower():
                print(f"  {line}")

    # Parse stage results from output
    stage_results = {}
    for line in (r.stdout + "\n" + r.stderr).split("\n"):
        # Look for lines like: "at_enc_mean: cos_min=0.9999 max_abs=0.001 PASS"
        if "cos_min" in line or "cos_mean" in line:
            stage_results[line.strip()] = True

    # Transcribe for text comparison
    print(f"  Transcribing...")
    tr = subprocess.run(
        [str(CRISPASR_BIN), "-m", str(out_gguf), "--backend", "vibevoice",
         "-f", str(REPO / "samples" / "jfk.wav"), "-t", "4",
         "--language", "en", "--no-prints"],
        capture_output=True, text=True, timeout=600)
    text = ""
    for line in tr.stdout.strip().split("\n"):
        if line and not any(k in line for k in ["firered", "whisper", "crispasr:"]):
            text += line.strip() + " "
    text = text.strip()
    print(f"  Text: {text[:100]}")

    # Upload variant to HF
    if hf_token and out_gguf.exists():
        print(f"  Uploading {fname} to HF...")
        try:
            from huggingface_hub import HfApi
            HfApi(token=hf_token).upload_file(
                path_or_fileobj=str(out_gguf), path_in_repo=fname,
                repo_id=HF_REPO, repo_type="model")
            print(f"  Uploaded")
        except Exception as e:
            print(f"  Upload failed: {e}")

    results.append({
        "label": label, "vae_quant": vae_q, "embed_quant": embed_q,
        "file_size_mb": round(file_size_mb, 1),
        "diff_output": r.stdout[-2000:] if r.stdout else "",
        "diff_stderr": r.stderr[-1000:] if r.stderr else "",
        "text": text,
        "stages": stage_results,
    })

    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    kh.step(f"diff-{label}")

# ── Summary ──────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("DIFF HARNESS RESULTS")
print("=" * 80)
for r in results:
    if "error" in r:
        print(f"{r['label']}: ERROR - {r['error']}")
    else:
        print(f"\n{r['label']} ({r['file_size_mb']:.0f} MB):")
        print(f"  Text: {r['text'][:80]}")
        if r.get("diff_output"):
            for line in r["diff_output"].split("\n"):
                if line.strip():
                    print(f"  {line.strip()}")

with open(RESULTS, "w") as f:
    json.dump(results, f, indent=2)
shutil.copy2(str(RESULTS), str(WORK / "vibeasr-bitnet-diff-results.json"))

print("\nDone.")
