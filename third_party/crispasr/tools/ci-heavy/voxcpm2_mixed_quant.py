#!/usr/bin/env python3
"""Build voxcpm2-q8_0-locdit-f16.gguf: q8_0 everywhere except the LocDiT
diffusion head in F16 (#461).

On Vulkan the CFM head is compute-bound in small matmuls whose cooperative-
matrix kernels favour F16; the 2B text model is bandwidth-bound and prefers
q8_0. Measured on a T4 (Kaggle runs 6-7): CFM 70.2 -> 59.4 ms per audio step
at 10 steps, total 3.47 -> 3.31-3.43 s; full F16 was slower overall (4.0 s).

Gate before the file is kept: a CPU synthesis of the reporter's sentence with
the mixed file must transcribe back exactly (whisper base.en via crispasr).

    gh workflow run heavy-cpu.yml -f script=tools/ci-heavy/voxcpm2_mixed_quant.py -f pip="huggingface_hub"
"""
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

OUT = Path(os.environ.get("HEAVY_OUT", "out"))
SCR = Path(os.environ.get("HEAVY_SCRATCH", "scratch"))
OUT.mkdir(parents=True, exist_ok=True)
SCR.mkdir(parents=True, exist_ok=True)
REPO = Path(__file__).resolve().parents[2]
TEXT = "Hello, this is a short test sentence."
NAME = "voxcpm2-q8_0-locdit-f16.gguf"


def run(cmd, **kw):
    print("$", " ".join(map(str, cmd)), flush=True)
    return subprocess.run([str(c) for c in cmd], **kw)


from huggingface_hub import hf_hub_download  # noqa: E402

b = SCR / "build"
run(["cmake", "-S", REPO, "-B", b, "-DCMAKE_BUILD_TYPE=Release", "-DCRISPASR_OPUS=OFF", "-DCRISPASR_AMR=OFF"],
    check=True, stdout=subprocess.DEVNULL)
run(["cmake", "--build", b, "--target", "crispasr-cli", "crispasr-quantize", f"-j{os.cpu_count() or 4}"],
    check=True, stdout=subprocess.DEVNULL)
cli, quant = b / "bin" / "crispasr", b / "bin" / "crispasr-quantize"
f16 = hf_hub_download("cstr/voxcpm2-GGUF", "voxcpm2-f16.gguf")
mixed = SCR / NAME
r = run([quant, f16, mixed, "q8_0", "--tensor-type", r"^locdit\.=f16"], capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-1500:])
assert r.returncode == 0 and mixed.exists(), "quantize failed"

wav = SCR / "check.wav"
r = run([cli, "--backend", "voxcpm2", "-m", mixed, "--tts", TEXT, "--tts-output", wav, "--seed", "2",
         "-t", str(os.cpu_count() or 4)], capture_output=True, text=True, timeout=3600)
print(r.stderr[-800:])
wm = hf_hub_download("ggerganov/whisper.cpp", "ggml-base.en.bin")
r = run([cli, "-m", wm, "-f", wav, "-np", "-nt"], capture_output=True, text=True, timeout=600)
heard = r.stdout.strip()
norm = lambda s: re.sub(r"[^a-z ]", "", s.lower()).split()
ok = norm(heard) == norm(TEXT)
size = mixed.stat().st_size
summary = [f"### {NAME}\n", f"size: {size / 1e9:.2f} GB", f"ASR roundtrip: {heard!r} -> exact: {ok}"]
if ok:
    h = hashlib.sha256()
    with open(mixed, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    shutil.copy(mixed, OUT / NAME)
    (OUT / "SHA256SUMS").write_text(f"{h.hexdigest()}  {NAME}\n")
(OUT / "summary.md").write_text("\n".join(summary) + "\n")
print("\n".join(summary))
sys.exit(0 if ok else 1)
