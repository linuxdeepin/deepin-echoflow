#!/usr/bin/env python3
"""#302: generic-host CPU baseline + real P100 OmniVoice acceptance gate.

The reporter's Windows status 0xC000001D is a host illegal instruction, not a
missing Pascal CUDA image: CUDA and Vulkan both die while ggml enumerates its
CPU backend. This kernel proves the fixed build still carries sm_60 CUDA, that
all optional host x86 ISAs are disabled in the CMake cache, that the exact
server/voice-clone startup reaches LISTENING, and that a real synthesis decodes
back to the requested sentence.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import wave
from pathlib import Path


WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp")
REPO = TMP / "CrispASR"
BUILD = TMP / "build-issue302"
MODELS = TMP / "models-issue302"
CLI = BUILD / "bin" / "crispasr"
REF = TMP / "issue302-ref-24k.wav"
OUT = WORK / "issue302-omnivoice.wav"
RESULT = WORK / "issue302-result.json"
BRANCH = os.environ.get("CRISPASR_BRANCH", "main")
TEXT = "The quick brown fox jumps over the lazy dog."


def run(cmd, *, cwd=None, env=None, timeout=3600, check=True):
    print("$", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(
        list(map(str, cmd)), cwd=cwd, env=env, timeout=timeout, check=check,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )


TMP.mkdir(parents=True, exist_ok=True)
if not REPO.exists():
    run([
        "git", "clone", "--depth", "1", "--branch", BRANCH,
        "--recurse-submodules", "--shallow-submodules",
        "https://github.com/CrispStrobe/CrispASR.git", REPO,
    ], timeout=2400)

sys.path.insert(0, str(REPO / "tools" / "kaggle"))
try:
    import kaggle_harness as kh
except Exception:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import kaggle_harness as kh

kh.init_progress()
token = kh.resolve_hf_token()
kh.step("clone.done", branch=BRANCH)

gpu = run([
    "nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader",
]).stdout.strip()
print("GPU:", gpu, flush=True)
if "P100" not in gpu or "6.0" not in gpu:
    RESULT.write_text(json.dumps({"verdict": "NEED_P100", "gpu": gpu}, indent=2))
    kh.step("wrong_gpu", gpu=gpu)
    raise SystemExit("This acceptance gate requires a Tesla P100 / sm_60 draw")

kh.install_build_toolchain()
flags = [
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCRISPASR_BUILD_TESTS=OFF",
    "-DCRISPASR_BUILD_EXAMPLES=ON",
    "-DCRISPASR_BUILD_SERVER=ON",
    "-DCRISPASR_PORTABLE_CPU=ON",
] + kh.cuda_build_flags("60") + kh.cache_and_link_flags()

kh.step("build.configure", gpu=gpu)
with kh.build_heartbeat("issue302 configure", 30):
    configured = run(["cmake", "-S", REPO, "-B", BUILD, *flags], timeout=1800)
print(configured.stdout[-4000:], flush=True)

cache = (BUILD / "CMakeCache.txt").read_text()
isa_options = (
    "GGML_NATIVE", "GGML_SSE42", "GGML_AVX", "GGML_AVX_VNNI", "GGML_AVX2",
    "GGML_FMA", "GGML_F16C", "GGML_BMI2", "GGML_AVX512", "GGML_AVX512_VBMI",
    "GGML_AVX512_VNNI", "GGML_AVX512_BF16", "GGML_AMX_TILE", "GGML_AMX_INT8",
    "GGML_AMX_BF16",
)
bad_isa = [name for name in isa_options if f"{name}:BOOL=OFF" not in cache]
if bad_isa:
    raise SystemExit(f"portable cache contract failed; enabled/missing: {bad_isa}")

kh.step("build.compile")
with kh.build_heartbeat("issue302 CUDA build", 30):
    kh.sh_with_progress(
        f"cmake --build {BUILD} --target crispasr-cli "
        f"-j{kh.safe_build_jobs(gpu=True)}"
    )
if not CLI.is_file():
    raise SystemExit("crispasr binary missing after build")

kh.step("models.download")
run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "openai-whisper"])
from huggingface_hub import hf_hub_download

MODELS.mkdir(parents=True, exist_ok=True)
model = Path(hf_hub_download(
    repo_id="cstr/omnivoice-GGUF", filename="omnivoice-q8_0.gguf",
    local_dir=MODELS, token=token,
))
codec = Path(hf_hub_download(
    repo_id="cstr/omnivoice-GGUF", filename="omnivoice-tokenizer-f16.gguf",
    local_dir=MODELS, token=token,
))

# A three-second reference keeps the clone path real without turning the test
# into an 11-second-reference stress run (kaggle_usage.md voice-clone rule).
run([
    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
    "-i", REPO / "samples" / "jfk.wav", "-t", "3", "-ar", "24000", REF,
])

common = [
    CLI, "--backend", "omnivoice", "-m", model, "--codec-model", codec,
    "--voice", REF, "--ref-text", "And so my fellow Americans.",
    "--i-have-rights", "--no-watermark", "--no-c2pa",
    "--accept-marking-responsibility", "--no-spoken-disclaimer", "-t", "4",
]
runtime_env = {
    **os.environ,
    "CRISPASR_OMNIVOICE_CODEC_GPU": "1",
    "CRISPASR_VERBOSE": "1",
}

# Reproduce the reporter's failing phase first: persistent server startup with
# a WAV voice clone configured. It must get beyond model/backend enumeration.
kh.step("server.start")
server_log = WORK / "issue302-server.log"
with server_log.open("w") as log_file:
    proc = subprocess.Popen(
        [*common, "--server", "--host", "127.0.0.1", "--port", "3979"],
        env=runtime_env, text=True, stdout=log_file, stderr=subprocess.STDOUT,
    )
    listening = False
    deadline = time.time() + 600
    while time.time() < deadline and proc.poll() is None:
        time.sleep(2)
        text = server_log.read_text(errors="replace")
        if "listening on 127.0.0.1:3979" in text:
            listening = True
            break
    if listening:
        with urllib.request.urlopen("http://127.0.0.1:3979/health", timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(f"health returned HTTP {response.status}")
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
if not listening:
    print(server_log.read_text(errors="replace")[-8000:], flush=True)
    raise SystemExit("OmniVoice server did not reach LISTENING")

server_text = server_log.read_text(errors="replace")
if "Tesla P100" not in server_text or "compute capability 6.0" not in server_text:
    raise SystemExit("server log does not prove the P100 CUDA backend initialized")
if "cpu-isa: OK — built for baseline x86-64 (no AVX)" not in server_text:
    raise SystemExit("server log does not prove the generic host CPU baseline")

# Decoded-output acceptance: real CUDA synth through the exact voice-clone path,
# then CPU Whisper reads the generated WAV and overlap-gates the requested words.
kh.step("tts.synthesize")
with kh.build_heartbeat("issue302 OmniVoice synthesis", 30):
    synth = run([*common, "--tts", TEXT, "--tts-output", OUT], env=runtime_env, timeout=1800)
print(synth.stdout[-8000:], flush=True)
if "Tesla P100" not in synth.stdout or not OUT.is_file():
    raise SystemExit("synthesis did not prove P100 execution and produce a WAV")

with wave.open(str(OUT), "rb") as wav:
    frames = wav.getnframes()
    rate = wav.getframerate()
    raw = wav.readframes(frames)
duration = frames / max(1, rate)
if duration < 1.0 or not any(raw):
    raise SystemExit(f"invalid synthesis proof: duration={duration:.2f}s")

kh.step("tts.asr_roundtrip", duration=duration)
import whisper

asr = whisper.load_model("base", device="cpu")
transcript = asr.transcribe(str(OUT), language="en", fp16=False)["text"].strip()
norm = lambda s: re.findall(r"[a-z]+", s.lower())
expected_words = norm(TEXT)
heard = set(norm(transcript))
overlap = sum(word in heard for word in expected_words) / len(expected_words)
if overlap < 0.65:
    raise SystemExit(f"roundtrip overlap {overlap:.2f} too low: {transcript!r}")

result = {
    "verdict": "PASS",
    "branch": BRANCH,
    "gpu": gpu,
    "portable_isa_flags_off": len(isa_options),
    "server_listening": listening,
    "audio_duration_s": round(duration, 3),
    "asr_transcript": transcript,
    "word_overlap": round(overlap, 3),
}
RESULT.write_text(json.dumps(result, indent=2))
kh.export_ccache_tar()
kh.step("done", **result)
print(json.dumps(result, indent=2), flush=True)
