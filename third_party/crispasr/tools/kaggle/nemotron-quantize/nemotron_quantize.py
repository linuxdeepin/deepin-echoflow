"""
CrispASR — Nemotron GGUF quantization + HF upload

Downloads the F16 GGUF, quantizes to Q8_0 and Q4_K using crispasr-quantize,
and uploads all three to cstr/nemotron-3.5-asr-streaming-0.6b-GGUF.

CPU kernel (no GPU needed, just disk + RAM for quantization).
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = REPO / "build"
BRANCH = os.environ.get("CRISPASR_REF", "main")

# ── Clone + harness ──
print(f"[1/5] Cloning CrispASR ({BRANCH})...", flush=True)
Path("/kaggle/working/status.txt").write_text("step1: cloning\n")
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.check_call(
    ["git", "clone", "--depth", "1", "--branch", BRANCH, "--recursive",
     "https://github.com/CrispStrobe/CrispASR.git", str(REPO)], timeout=120)
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh
kh.init_progress()
hf_token = kh.resolve_hf_token()
kh.step("cloned")

# ── Build quantize tool only (much faster than full crispasr) ──
print("[2/5] Building crispasr-quantize...", flush=True)
Path("/kaggle/working/status.txt").write_text("step2: building quantizer\n")
kh.install_build_toolchain()

subprocess.check_call([
    "cmake", "-G", "Ninja", "-B", str(BUILD),
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
    "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
], cwd=str(REPO), timeout=120)

# Build just the quantize target
with kh.build_heartbeat("cmake.build"):
    subprocess.check_call([
        "cmake", "--build", str(BUILD),
        "--target", "crispasr-quantize",
        f"-j{os.cpu_count()}",
    ], cwd=str(REPO), timeout=600)
kh.step("built")

# ── Download F16 GGUF ──
print("[3/5] Downloading F16 GGUF...", flush=True)
Path("/kaggle/working/status.txt").write_text("step3: downloading\n")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

from huggingface_hub import hf_hub_download, HfApi
f16_path = WORK / "nemotron-3.5-asr-streaming-0.6b-f16.gguf"
src = hf_hub_download(
    repo_id="cstr/nemotron-3.5-asr-streaming-0.6b-GGUF",
    filename="nemotron-3.5-asr-streaming-0.6b-f16.gguf",
    cache_dir=str(WORK / "hf_cache"),
)
shutil.copy2(src, str(f16_path))
print(f"  F16: {f16_path.stat().st_size / 1e6:.0f} MB", flush=True)
kh.step("downloaded")

# ── Quantize ──
print("[4/5] Quantizing...", flush=True)
Path("/kaggle/working/status.txt").write_text("step4: quantizing\n")
quantize_bin = str(BUILD / "bin" / "crispasr-quantize")

quants = {
    "q8_0": WORK / "nemotron-3.5-asr-streaming-0.6b-q8_0.gguf",
    "q4_k": WORK / "nemotron-3.5-asr-streaming-0.6b-q4_k.gguf",
}

for qtype, out_path in quants.items():
    print(f"  Quantizing {qtype}...", flush=True)
    t0 = time.time()
    r = subprocess.run(
        [quantize_bin, str(f16_path), str(out_path), qtype],
        capture_output=True, text=True, timeout=300,
    )
    dt = time.time() - t0
    if out_path.exists():
        sz = out_path.stat().st_size / 1e6
        print(f"    {qtype}: {sz:.0f} MB ({dt:.0f}s)", flush=True)
    else:
        print(f"    {qtype}: FAILED (rc={r.returncode})", flush=True)
        if r.stderr:
            print(f"    stderr: {r.stderr[-500:]}", flush=True)
kh.step("quantized")

# ── Upload to HF ──
print("[5/5] Uploading to HuggingFace...", flush=True)
Path("/kaggle/working/status.txt").write_text("step5: uploading\n")
api = HfApi(token=hf_token)

for qtype, out_path in quants.items():
    if out_path.exists():
        print(f"  Uploading {out_path.name}...", flush=True)
        try:
            api.upload_file(
                path_or_fileobj=str(out_path),
                path_in_repo=out_path.name,
                repo_id="cstr/nemotron-3.5-asr-streaming-0.6b-GGUF",
                repo_type="model",
                commit_message=f"Add {qtype.upper()} quantization",
            )
            print(f"    uploaded {out_path.name}", flush=True)
        except Exception as e:
            print(f"    upload failed: {e}", flush=True)

kh.step("uploaded")
Path("/kaggle/working/status.txt").write_text("DONE\n")
print("Done.", flush=True)
