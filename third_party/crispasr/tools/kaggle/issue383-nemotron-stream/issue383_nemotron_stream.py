#!/usr/bin/env python3
"""Issue #383: CUDA live stream + NVIDIA Transformers blueprint comparison."""

import json
import os
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = REPO / "build"
RESULTS = WORK / "issue383-results.json"
REF = "fix/383-nemotron-realtime"


def sh(command, **kwargs):
    print("$", command, flush=True)
    return subprocess.run(command, shell=True, check=True, **kwargs)


if not REPO.exists():
    sh(f"git clone --depth 1 --branch {REF} --recursive https://github.com/CrispStrobe/CrispASR {REPO}")
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
if str(REPO / "tools" / "kaggle") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
token = kh.resolve_hf_token()
if token:
    os.environ["HF_TOKEN"] = token
kh.step("clone.done", sha=subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip())

kh.install_build_toolchain()
flags = " ".join(kh.cuda_build_flags() + kh.cache_and_link_flags())
with kh.build_heartbeat("configure"):
    kh.sh_with_progress(f"cmake -S {REPO} -B {BUILD} -G Ninja -DCMAKE_BUILD_TYPE=Release {flags}")
with kh.build_heartbeat("build"):
    kh.sh_with_progress(
        f"cmake --build {BUILD} --target crispasr-cli test-nemotron test-realtime-turn-buffer "
        f"-j{kh.safe_build_jobs(gpu=True)}"
    )
kh.step("build.done")

sh("pip install -q 'huggingface_hub>=0.30' 'transformers>=5.13.0' soundfile accelerate")
from huggingface_hub import hf_hub_download  # noqa: E402

model = hf_hub_download(
    repo_id="cstr/nemotron-3.5-asr-streaming-GGUF",
    filename="nemotron-3.5-asr-streaming-0.6b-q4_k.gguf",
    local_dir=str(WORK / "models"),
)
jfk = REPO / "samples" / "jfk.wav"
env = {**os.environ, "CRISPASR_MODEL_NEMOTRON": model, "CRISPASR_NEMOTRON_STREAMING": "1"}

live = subprocess.run(
    [str(BUILD / "bin/test-nemotron"), "[streaming]"], cwd=REPO, env=env,
    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
)
print(live.stdout, flush=True)
ws = subprocess.run(
    [sys.executable, "-u", str(REPO / "tests/test-server-realtime-api.py"),
     "--backend", "nemotron", "--model", model, "--language", "en"],
    cwd=REPO, env=env,
    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
)
print(ws.stdout, flush=True)
kh.step("cpp.cuda.live.done")

# Run NVIDIA's model-card generator shape verbatim: first centered chunk,
# subsequent non-centered overlapping raw windows, persistent encoder/padding/
# RNNT caches owned by model.generate().
from transformers import AutoModelForRNNT, AutoProcessor  # noqa: E402
from transformers.audio_utils import load_audio  # noqa: E402

model_id = "nvidia/nemotron-3.5-asr-streaming-0.6b"
processor = AutoProcessor.from_pretrained(model_id)
# Kaggle's current PyTorch CUDA wheel starts at sm_70 while its assigned P100
# is sm_60. CrispASR above is tested on CUDA; run this independent architectural
# reference on CPU so framework wheel support cannot invalidate the comparison.
upstream_model = AutoModelForRNNT.from_pretrained(model_id, device_map="cpu")
processor.set_num_lookahead_tokens(6)
audio = load_audio(str(jfk), sampling_rate=processor.feature_extractor.sampling_rate)
first = processor(
    audio[: processor.num_samples_first_audio_chunk],
    sampling_rate=processor.feature_extractor.sampling_rate,
    is_streaming=True, is_first_audio_chunk=True, language="en-US", return_tensors="pt",
).to(upstream_model.device, dtype=upstream_model.dtype)


def features():
    yield first.input_features[:, : processor.num_mel_frames_first_audio_chunk, :]
    mel_idx = processor.num_mel_frames_first_audio_chunk
    hop = processor.feature_extractor.hop_length
    n_fft = processor.feature_extractor.n_fft
    start = mel_idx * hop - n_fft // 2
    while (end := start + processor.num_samples_per_audio_chunk) < audio.shape[0]:
        item = processor(
            audio[start:end], sampling_rate=processor.feature_extractor.sampling_rate,
            is_streaming=True, is_first_audio_chunk=False, language="en-US", return_tensors="pt",
        ).to(upstream_model.device, dtype=upstream_model.dtype)
        yield item.input_features
        mel_idx += processor.num_mel_frames_per_audio_chunk
        start = mel_idx * hop - n_fft // 2


generation_inputs = dict(first)
generation_inputs.pop("input_features", None)
generated = upstream_model.generate(
    **generation_inputs, input_features=features(), return_dict_in_generate=True,
)
upstream_text = processor.decode(generated.sequences, skip_special_tokens=True)
print("UPSTREAM_STREAM_TRANSCRIPT:", upstream_text, flush=True)

result = {
    "git_ref": REF,
    "cpp_stream_test": "passed",
    "websocket_test": "passed",
    "upstream_latency_ms": processor.streaming_latency_ms,
    "upstream_stream_transcript": upstream_text,
    "cpp_test_tail": live.stdout[-3000:],
    "websocket_test_output": ws.stdout,
}
RESULTS.write_text(json.dumps(result, indent=2))
kh.step("all.done", upstream_chars=len(upstream_text))
