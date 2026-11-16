#!/usr/bin/env python3
"""Kaggle GPU kernel: validate + time the Confucius4-TTS ggml GPU path (CUDA).

Run 1 (P100) validated the base CUDA path: CPU-ctrl 685.2s, GPU 183.2s
(3.74x), GPU-persist neutral, roundtrip 8/8 everywhere.  This revision A/Bs
the CFG fusion lever (CRISPASR_CONFUCIUS4_CFG_FUSE=1: cond+uncond in ONE DiT
graph eval per ODE step, seq-concat + block-diagonal mask) on BOTH s2a f16
and s2a q4_k (quant amplifies divergence — dev-guide A/B rule).  Arms (all
fully native `--voice`, jfk.wav prompt):

  GPU           s2a f16, fusion off  — baseline (matches run 1: 183.2s)
  GPU-fuse      s2a f16, fusion on
  GPU-q4k       s2a q4_k, fusion off — q4_k baseline (first q4_k GPU run)
  GPU-q4k-fuse  s2a q4_k, fusion on

Fused PCM will NOT be bit-identical to unfused (masked flash-attn reorders
the fp reductions) — the gates are the whisper roundtrip transcript + rms +
wall time.  Default flip only if fuse wins wall-time AND keeps 8/8
(LEARNING 35).

Push ($KAGGLE_ACCOUNT is the default account; needs ${KAGGLE_ACCOUNT}/crispasr-ccache for the
warm CUDA build):
  python -m kaggle kernels push -p tools/kaggle/confucius4-gpu-verify
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BRANCH = "main"
WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")

TEST_TEXT = "The quick brown fox jumps over the lazy dog."
LANG = "en"
ODE_STEPS = int(os.environ.get("ODE_STEPS", "25"))

print(f"=== clone {BRANCH} ===", flush=True)
if not REPO.exists():
    subprocess.check_call(["git", "clone", "--depth", "1", "-b", BRANCH,
                           "https://github.com/CrispStrobe/CrispASR", str(REPO)])
subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"], cwd=str(REPO))
head = subprocess.run(["git", "log", "--oneline", "-2"], cwd=str(REPO),
                      capture_output=True, text=True).stdout
print("  HEAD:\n   " + "\n   ".join(head.strip().split("\n")))

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh
kh.init_progress()

kh.step("install deps")
kh.sh_with_progress("pip install -q huggingface_hub hf_transfer soundfile")

kh.step("resolve HF token")
hf_token = kh.resolve_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
from huggingface_hub import hf_hub_download

kh.step("GPU check")
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)

kh.step("download GGUFs")
mdir = str(TEMP / "models")


def grab(repo, fname, **kw):
    p = hf_hub_download(repo, fname, local_dir=kw.pop("d", mdir), token=hf_token, **kw)
    print(f"  {fname}: {os.path.getsize(p) / 1024**2:.0f} MB")
    return p


t2s_path = grab("cstr/confucius4-tts-GGUF", "confucius4-tts-t2s-q4_k.gguf")
s2a_q4k = grab("cstr/confucius4-tts-GGUF", "confucius4-tts-s2a-q4_k.gguf")
s2a_f16 = grab("cstr/confucius4-tts-GGUF", "confucius4-tts-s2a-f16.gguf")
grab("cstr/confucius4-tts-GGUF", "confucius4-tts-bigvgan-22k-f16.gguf")
grab("cstr/confucius4-tts-GGUF", "confucius4-tts-w2v-f16.gguf")

kh.step("build CrispASR (CUDA)")
BUILD = TEMP / "build"
BUILD.mkdir(parents=True, exist_ok=True)
kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
flags = kh.cuda_build_flags(arch) + kh.cache_and_link_flags()
kh.sh_with_progress(
    f"cmake -G Ninja -B {BUILD} -S {REPO} -DCMAKE_BUILD_TYPE=Release "
    + " ".join(flags)
)
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"cmake --build {BUILD} -j{kh.safe_build_jobs(gpu=True)} --target crispasr-cli"
    )
crispasr_bin = str(BUILD / "bin" / "crispasr")
print(f"  binary: {crispasr_bin} ({os.path.getsize(crispasr_bin) / 1024**2:.0f} MB)")

prompt_wav = REPO / "samples" / "jfk.wav"

kh.step("synthesis arms")
FUSE = {"CRISPASR_CONFUCIUS4_CFG_FUSE": "1"}
arms = [
    ("GPU", {}, [], s2a_f16),
    ("GPU-fuse", FUSE, [], s2a_f16),
    ("GPU-q4k", {}, [], s2a_q4k),
    ("GPU-q4k-fuse", FUSE, [], s2a_q4k),
]
results = {}
for tag, extra_env, extra_args, s2a_path in arms:
    wav = TEMP / f"c4_gpu_{tag}.wav"
    env = dict(os.environ)
    env.update(extra_env)
    t0 = time.monotonic()
    r = subprocess.run(
        [crispasr_bin, "--backend", "confucius4-tts", "-m", t2s_path,
         "--codec-model", s2a_path, "--tts", TEST_TEXT, "-l", LANG,
         "--voice", str(prompt_wav), "--i-have-rights",
         "--tts-output", str(wav), "--tts-steps", str(ODE_STEPS), "-v"] + extra_args,
        capture_output=True, text=True, timeout=7200, env=env,
    )
    wall = time.monotonic() - t0
    ok = wav.exists() and os.path.getsize(str(wav)) > 100
    print(f"  [{tag}] rc={r.returncode} wall={wall:.1f}s "
          f"wav={'%d B' % os.path.getsize(str(wav)) if ok else 'NONE'}")
    for line in r.stderr.split("\n"):
        if any(k in line for k in ("beam decode:", "flow-matching:", "BigVGAN:", "CUDA",
                                   "voice set", "condition_emb computed", "failed", "error",
                                   "GGML_ASSERT", "unsupported", "CFG fusion", "DiT graph")):
            print(f"  [{tag}] " + line.strip()[:220])
    if r.returncode != 0:
        for line in [l for l in (r.stderr + r.stdout).split("\n") if l.strip()][-25:]:
            print(f"  [{tag}] ! " + line.strip()[:220])
    results[tag] = {"wav": wav, "ok": ok, "wall": wall}

kh.step("ASR roundtrip")
whisper_model = str(TEMP / "models" / "ggml-base.en.bin")
if not os.path.exists(whisper_model):
    import urllib.request
    urllib.request.urlretrieve(
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
        whisper_model)

ORIG = set(w.strip(".,!?").lower() for w in TEST_TEXT.split())
for tag, res in results.items():
    if not res["ok"]:
        print(f"  [{tag}] SKIP: no wav")
        continue
    a = subprocess.run([crispasr_bin, "-m", whisper_model, "-f", str(res["wav"]),
                        "--no-gpu", "--no-prints"],
                       capture_output=True, text=True, timeout=600)
    clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", a.stdout)
    hit = {w for w in ORIG if w in clean.lower()}
    import soundfile as sf
    d, sr = sf.read(str(res["wav"]))
    rms = float(np.sqrt(np.mean(np.square(d)))) if len(d) else 0.0
    print(f"  [{tag}] overlap {len(hit)}/{len(ORIG)} = {100*len(hit)/len(ORIG):.0f}%  "
          f"wall={res['wall']:.1f}s  ({len(d)/sr:.2f}s rms={rms:.4f})")
    print(f"  [{tag}] transcript: {clean.strip()[:300]}")

kh.step("summary")
for base, fused in (("GPU", "GPU-fuse"), ("GPU-q4k", "GPU-q4k-fuse")):
    if results.get(base, {}).get("ok") and results.get(fused, {}).get("ok"):
        b, f = results[base]["wall"], results[fused]["wall"]
        print(f"  {base} {b:.1f}s vs {fused} {f:.1f}s ({b/f:.2f}x, {100*(b-f)/b:+.1f}% saved)")
print("=== Done ===", flush=True)
