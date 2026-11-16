#!/usr/bin/env python3
"""CrispASR canary-qwen diff diagnostic on Kaggle.

Phase 1: Compute Python reference mel using NeMo-identical STFT + mel
         filterbank (torchaudio/librosa). Save to reference GGUF.
Phase 2: Run crispasr-diff canary-qwen to compare C++ vs Python mel.
Phase 3: Full transcription to check end-to-end output.

This kernel finds the first diverging stage in the pipeline.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
PROGRESS = WORK / "progress.jsonl"
T0 = time.time()


def _step(name, **extra):
    rec = {"ts": round(time.time() - T0, 2), "step": name, **extra}
    print(f"[step {rec['ts']:>7.1f}s] {name}" + (f"  {extra}" if extra else ""),
          flush=True)
    try:
        with PROGRESS.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def sh(cmd, cwd=None):
    print(f"$ {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True, cwd=str(cwd) if cwd else None)


_step("start")

try:
    # ── Clone + harness ───────────────────────────────────────────────
    CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"

    _step("clone")
    if not REPO.exists():
        subprocess.check_call(
            ["git", "clone", "--recursive",
             CRISPASR_URL, str(REPO)])
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    try:
        import kaggle_harness as kh
        # Full harness regime: authenticate HF pulls from the attached token
        # dataset (anon pulls get rate-limited); hf_transfer wedges multi-GB
        # Kaggle downloads, so keep the plain resumable downloader.
        kh.resolve_hf_token()
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        kh.init_progress()
    except Exception:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import kaggle_harness as kh
        kh.init_progress()

    kh.step("harness_loaded")

    # ── HF auth ───────────────────────────────────────────────────────
    kh.step("hf_auth")
    hf_token = kh.kaggle_secret("HF_TOKEN")
    if not hf_token:
        hf_token = kh.kaggle_token_from_dataset("hf_token.txt")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    # ── Build ─────────────────────────────────────────────────────────
    kh.step("install_toolchain")
    subprocess.run("apt-get update -qq && apt-get install -y --no-install-recommends "
                   "libopenblas-dev 2>/dev/null || true", shell=True)
    kh.install_build_toolchain()

    kh.step("pip_deps")
    subprocess.run("pip install -q safetensors gguf huggingface_hub soundfile librosa",
                   shell=True, capture_output=True)

    subprocess.run("nvidia-smi", shell=True)

    kh.step("build_configure")
    build_flags = kh.cache_and_link_flags()
    try:
        arch = kh.detect_cuda_arch()
        build_flags += kh.cuda_build_flags(arch)
    except Exception:
        pass

    sh(f"cmake -S {REPO} -B {BUILD} -G Ninja "
       f"-DCMAKE_BUILD_TYPE=Release "
       f"-DCRISPASR_BUILD_TESTS=OFF "
       f"-DCRISPASR_BUILD_EXAMPLES=ON "
       f"-DCRISPASR_BUILD_SERVER=OFF "
       + " ".join(build_flags))

    kh.step("build_compile")
    with kh.build_heartbeat("cmake.build"):
        sh(f"stdbuf -oL -eL cmake --build {BUILD} "
           f"--target crispasr-cli crispasr-quantize crispasr-diff "
           f"-j{kh.safe_build_jobs(gpu=True)}")

    CRISPASR_BIN = BUILD / "bin" / "crispasr"
    QUANTIZE_BIN = BUILD / "bin" / "crispasr-quantize"
    DIFF_BIN = BUILD / "bin" / "crispasr-diff"
    kh.step("build_done")

    # ── Download + convert model ──────────────────────────────────────
    kh.step("download_model")
    MODEL_DIR = WORK / "canary-qwen-hf"
    GGUF_F16 = WORK / "canary-qwen-2.5b-f16.gguf"
    GGUF_Q4K = WORK / "canary-qwen-2.5b-q4_k.gguf"

    from huggingface_hub import snapshot_download, hf_hub_download
    snapshot_download("nvidia/canary-qwen-2.5b",
                      local_dir=str(MODEL_DIR),
                      token=os.environ.get("HF_TOKEN"))

    for tok_file in ("tokenizer.json", "tokenizer_config.json"):
        if not (MODEL_DIR / tok_file).exists():
            hf_hub_download("Qwen/Qwen3-1.7B", tok_file,
                            local_dir=str(MODEL_DIR))
    kh.step("model_downloaded", free_gb=kh.free_gb())

    kh.step("convert_to_gguf")
    os.environ["TMPDIR"] = str(WORK / "tmp")
    (WORK / "tmp").mkdir(exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    sh(f"python {REPO}/models/convert-canary-qwen-to-gguf.py "
       f"--input {MODEL_DIR} --output {GGUF_F16}")
    kh.step("convert_done", size_gb=round(GGUF_F16.stat().st_size / 1e9, 2))

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: Python reference mel dump
    # Match NeMo AudioToMelSpectrogramPreprocessor EXACTLY:
    #   n_fft=512, win=400 (Hann periodic), hop=160, center=True,
    #   128 mels (Slaney), log(x + eps), per_feature z-norm,
    #   preemph=0.97, dither=0 (disabled for determinism)
    # ══════════════════════════════════════════════════════════════════
    kh.step("reference_dump_start")

    import torch
    import numpy as np
    import soundfile as sf
    import gguf as gguf_lib

    JFK_WAV = REPO / "samples" / "jfk.wav"
    audio, sr = sf.read(str(JFK_WAV))
    if sr != 16000:
        import torchaudio
        audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        audio = torchaudio.functional.resample(audio, sr, 16000).squeeze(0).numpy()
    audio = audio.astype(np.float32)
    print(f"  audio: {len(audio)} samples, sr={sr}", flush=True)

    REF_GGUF = WORK / "canary-qwen-ref.gguf"
    ref_writer = gguf_lib.GGUFWriter(str(REF_GGUF), arch="canary_qwen_ref",
                                      use_temp_file=False)

    # ── Mel spectrogram ───────────────────────────────────────────────
    kh.step("compute_mel_reference")

    n_fft = 512
    win_length = 400
    hop_length = 160
    n_mels_target = 128
    sample_rate = 16000
    preemph_coeff = 0.97

    sig = torch.from_numpy(audio).float()

    # Preemphasis (NeMo default: 0.97)
    preemphasized = torch.cat([sig[:1], sig[1:] - preemph_coeff * sig[:-1]])

    # STFT with center padding (matching NeMo center=True)
    window = torch.hann_window(win_length, periodic=True)
    stft_out = torch.stft(preemphasized, n_fft=n_fft, hop_length=hop_length,
                          win_length=win_length, window=window, center=True,
                          return_complex=True)
    power = stft_out.abs().pow(2)  # (n_freqs=257, T_mel)

    # Mel filterbank (Slaney-normalized, htk=False — matching NeMo)
    import librosa
    mel_fb = librosa.filters.mel(sr=sample_rate, n_fft=n_fft,
                                 n_mels=n_mels_target,
                                 htk=False, norm='slaney')
    mel_fb_t = torch.from_numpy(mel_fb).float()  # (128, 257)

    mel = torch.matmul(mel_fb_t, power)  # (128, T_mel)
    print(f"  mel shape (before log/norm): {mel.shape}", flush=True)

    # Log (NeMo: log=True → natural log, with eps guard)
    log_eps = 1.0 / (1 << 24)  # ~5.96e-8, matching NeMo default
    log_mel = torch.log(mel + log_eps)

    # Per-feature z-normalization (NeMo normalize="per_feature")
    mean = log_mel.mean(dim=1, keepdim=True)
    std = log_mel.std(dim=1, keepdim=True)
    std = torch.clamp(std, min=1e-5)
    norm_mel = (log_mel - mean) / std

    # NeMo drops the last frame: feat_len = floor(n_samples / hop_length)
    T_valid = len(audio) // hop_length
    norm_mel = norm_mel[:, :T_valid]

    # Transpose to (T_mel, n_mels) — TimeMels layout matching C++ core_mel
    mel_np = norm_mel.transpose(0, 1).contiguous().numpy().astype(np.float32)
    print(f"  Reference mel shape (T_mel, n_mels): {mel_np.shape}", flush=True)
    print(f"  mel stats: min={mel_np.min():.4f} max={mel_np.max():.4f} "
          f"mean={mel_np.mean():.4f}", flush=True)

    ref_writer.add_tensor("mel_spectrogram", mel_np)
    ref_writer.add_tensor("raw_audio", audio)

    ref_writer.write_header_to_file()
    ref_writer.write_kv_data_to_file()
    ref_writer.write_tensors_to_file()
    ref_writer.close()
    kh.step("reference_dump_done", mel_shape=list(mel_np.shape))

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: crispasr-diff (mel comparison)
    # ══════════════════════════════════════════════════════════════════
    kh.step("run_diff")
    diff_result = subprocess.run(
        [str(DIFF_BIN), "canary-qwen",
         str(GGUF_F16), str(REF_GGUF), str(JFK_WAV)],
        capture_output=True, text=True, timeout=300)

    print("=== DIFF STDOUT ===", flush=True)
    print(diff_result.stdout, flush=True)
    print("=== DIFF STDERR ===", flush=True)
    print(diff_result.stderr, flush=True)
    kh.step("diff_done", exit_code=diff_result.returncode)

    with open(WORK / "diff_result.txt", "w") as f:
        f.write(f"exit_code={diff_result.returncode}\n")
        f.write(f"stdout:\n{diff_result.stdout}\n")
        f.write(f"stderr:\n{diff_result.stderr}\n")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: Quick mel diagnostic — compare shapes + first values
    # ══════════════════════════════════════════════════════════════════
    kh.step("mel_diagnostic")

    # C++ mel via the stage API
    cpp_mel_result = subprocess.run(
        [str(CRISPASR_BIN), "--backend", "canary-qwen",
         "-m", str(GGUF_F16), "-f", str(JFK_WAV),
         "-t", "4", "-l", "en", "--no-prints", "--debug-mode"],
        capture_output=True, text=True, timeout=300,
        env={**os.environ, "CRISPASR_CANARY_QWEN_DEBUG": "1"})
    print("=== DEBUG MODE STDERR (first 2000 chars) ===", flush=True)
    print(cpp_mel_result.stderr[:2000], flush=True)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 4: Full transcription (F16 for accuracy)
    # ══════════════════════════════════════════════════════════════════
    kh.step("transcribe_jfk")
    result = subprocess.run(
        [str(CRISPASR_BIN), "--backend", "canary-qwen",
         "-m", str(GGUF_F16), "-f", str(JFK_WAV),
         "-t", "4", "-l", "en", "--no-prints"],
        capture_output=True, text=True, timeout=600,
        env={**os.environ, "CANARY_QWEN_BENCH": "1",
             "CRISPASR_CANARY_QWEN_DEBUG": "1"})

    print("=== TRANSCRIBE STDOUT ===", flush=True)
    print(result.stdout, flush=True)
    print("=== TRANSCRIBE STDERR ===", flush=True)
    print(result.stderr, flush=True)
    print(f"=== EXIT CODE: {result.returncode} ===", flush=True)

    with open(WORK / "result.txt", "w") as f:
        f.write(f"exit_code={result.returncode}\n")
        f.write(f"stdout={result.stdout}\n")
        f.write(f"stderr={result.stderr[-3000:]}\n")

    kh.step("transcribe_done", exit_code=result.returncode,
            transcript=result.stdout.strip()[:200] if result.stdout else "")

except Exception as exc:
    tb = traceback.format_exc()
    print(f"\n=== EXCEPTION ===\n{tb}", flush=True)
    _step("EXCEPTION", error=str(exc), traceback=tb[-1000:])
    with open(WORK / "error.txt", "w") as f:
        f.write(tb)

_step("save_ccache")
try:
    os.chdir(str(WORK))
    subprocess.run("tar cf ccache.tar .ccache/", shell=True, check=True)
except Exception:
    pass

_step("done", total_s=round(time.time() - T0, 1))
print(f"\nTotal runtime: {time.time() - T0:.0f}s", flush=True)
