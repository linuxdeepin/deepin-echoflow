#!/usr/bin/env python3
"""OmniVoice parity test + ASR roundtrip on Kaggle GPU."""

from __future__ import annotations
import gc
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

WORK = Path("/kaggle/working")
os.chdir(WORK)
LOG = WORK / "progress.txt"

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")

log("=== OmniVoice Parity Test Starting ===")

try:
    # ── Clone CrispASR ──────────────────────────────────────────────
    log("Cloning CrispASR...")
    CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
    CRISPASR_BRANCH = os.environ.get("CRISPASR_BRANCH", "feat/omnivoice-gen-fix")
    _CRISPASR_DIR = WORK / "CrispASR"
    if not _CRISPASR_DIR.exists():
        subprocess.check_call(["git", "clone", "--depth", "1",
            "--branch", CRISPASR_BRANCH,
            CRISPASR_URL, str(_CRISPASR_DIR)])
    log(f"Branch: {CRISPASR_BRANCH}")
    sys.path.insert(0, str(_CRISPASR_DIR / "tools" / "kaggle"))

    try:
        import kaggle_harness as kh
        # Full harness regime: authenticate HF pulls from the attached token
        # dataset (anon pulls get rate-limited); hf_transfer wedges multi-GB
        # Kaggle downloads, so keep the plain resumable downloader.
        kh.resolve_hf_token()
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        kh.init_progress()
        HAS_KH = True
    except Exception as e:
        log(f"kaggle_harness import failed: {e}")
        HAS_KH = False

    # ── Install deps ────────────────────────────────────────────────
    log("Installing deps...")
    # Don't install torch (pre-installed on Kaggle)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "gguf", "safetensors", "soundfile", "openai-whisper"],
                   check=True)

    # Install omnivoice - may need newer transformers
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "omnivoice"],
                       capture_output=True, text=True)
    log(f"omnivoice install: exit={r.returncode}")
    if r.returncode != 0:
        log(f"  stderr: {r.stderr[:500]}")
        # Try installing from source
        log("Trying omnivoice from git...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "git+https://github.com/k2-fsa/OmniVoice.git"],
                       check=False)

    # ── HF auth ─────────────────────────────────────────────────────
    log("Setting up HF auth...")
    hf_token = None
    for path in [
        "/kaggle/input/crispasr-hf-token/hf_token.txt",
        "/kaggle/input/datasets/${KAGGLE_ACCOUNT}/crispasr-hf-token/hf_token.txt",
    ]:
        if os.path.exists(path):
            hf_token = open(path).read().strip()
            break
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        log("HF token loaded")
    else:
        log("WARNING: No HF token found")

    SYN_TEXT = os.environ.get("OMNIVOICE_SYN_TEXT", "Hello, this is a test.")
    MODEL_ID = "k2-fsa/OmniVoice"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Phase 1: Python reference dump
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    log("Phase 1: Loading OmniVoice...")
    import torch
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    try:
        from omnivoice import OmniVoice
        log("omnivoice imported OK")
    except ImportError as e:
        log(f"omnivoice import failed: {e}")
        log("Falling back to direct transformers loading...")
        # Load the model components directly
        from transformers import AutoModel, AutoTokenizer, AutoConfig
        OmniVoice = None

    from transformers import AutoTokenizer

    if OmniVoice is not None:
        # P100 (sm_60) is not supported by recent PyTorch CUDA builds.
        # Load on CPU to avoid CUDA kernel errors with weight_norm in
        # the HiggsAudioV2 tokenizer.
        load_device = "cpu"
        model = OmniVoice.from_pretrained(
            MODEL_ID, device_map=load_device,
            dtype=torch.float32,
        )
        tokenizer = model.text_tokenizer or AutoTokenizer.from_pretrained(MODEL_ID)
        model.text_tokenizer = tokenizer
        log("Model loaded via OmniVoice.from_pretrained()")
    else:
        # Manual load: just load config + weights for embedding comparison
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(MODEL_ID)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        # Load just the safetensors for embedding comparison
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(MODEL_ID, allow_patterns=["*.safetensors", "*.json"])
        from safetensors import safe_open
        st = safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt")
        log(f"Loaded safetensors with {len(st.keys())} tensors")
        model = None

    # Tokenize
    wrapped = f"<|text_start|>{SYN_TEXT}<|text_end|>"
    tok_out = tokenizer(wrapped, return_tensors="pt")
    text_ids = tok_out.input_ids[0]
    T_text = len(text_ids)
    log(f"Text: '{SYN_TEXT}' -> {T_text} tokens")
    log(f"Token IDs: {text_ids.tolist()}")

    stages = {}
    stages["text_input_ids"] = text_ids.numpy().astype(np.float32)

    # Text embeddings
    if model is not None and hasattr(model, 'llm'):
        with torch.no_grad():
            text_embeds = model.llm.embed_tokens(text_ids.unsqueeze(0).to("cpu"))
        stages["text_embeds"] = text_embeds[0].float().cpu().numpy()
        log(f"text_embeds: {stages['text_embeds'].shape}")

        # audio_embd row 0
        with torch.no_grad():
            audio_embd_row = model.audio_embeddings(torch.tensor([0], device="cpu"))
        stages["audio_embd_row0"] = audio_embd_row[0].float().cpu().numpy()

        # Full mixed embeds + LLM forward
        style_text = "<|lang_start|>None<|lang_end|><|instruct_start|>None<|instruct_end|>"
        style_ids = tokenizer(style_text, return_tensors="pt").input_ids[0]
        T_style = len(style_ids)
        T_target = max(int(T_text * 0.5), 10)
        T_total = T_style + T_text + T_target

        n_cb = model.config.num_audio_codebook
        mask_id = model.config.audio_mask_id

        input_ids = torch.zeros(1, n_cb, T_total, dtype=torch.long, device="cpu")
        for cb in range(n_cb):
            input_ids[0, cb, :T_style] = style_ids.to("cpu")
            input_ids[0, cb, T_style:T_style+T_text] = text_ids.to("cpu")
            input_ids[0, cb, T_style+T_text:] = mask_id

        audio_mask = torch.zeros(1, T_total, dtype=torch.bool, device="cpu")
        audio_mask[0, T_style+T_text:] = True

        with torch.no_grad():
            mixed_embeds = model._prepare_embed_inputs(input_ids, audio_mask)
        stages["mixed_embeds"] = mixed_embeds[0].float().cpu().numpy()
        log(f"mixed_embeds: {stages['mixed_embeds'].shape}")

        with torch.no_grad():
            llm_out = model.llm(inputs_embeds=mixed_embeds, return_dict=True)
            hidden = llm_out[0]
        stages["llm_hidden"] = hidden[0].float().cpu().numpy()
        log(f"llm_hidden: {stages['llm_hidden'].shape}")

        with torch.no_grad():
            logits_flat = model.audio_heads(hidden)
            audio_logits = logits_flat.view(1, T_total, n_cb,
                                           model.config.audio_vocab_size).permute(0, 2, 1, 3)
        target_logits = audio_logits[0, :, -T_target:, :]
        stages["audio_logits_target"] = target_logits.float().cpu().numpy()
        log(f"audio_logits_target: {stages['audio_logits_target'].shape}")
    elif st is not None:
        # Minimal: just compare embeddings from safetensors
        embd_w = st.get_tensor("llm.embed_tokens.weight").float()
        text_embeds = embd_w[text_ids.long()]
        stages["text_embeds"] = text_embeds.numpy()
        log(f"text_embeds (from safetensors): {stages['text_embeds'].shape}")
        T_style = 0
        T_target = 10
        T_total = T_text + T_target

    log(f"Stages: {list(stages.keys())}")

    # ── Write reference GGUF ────────────────────────────────────────
    log("Writing reference GGUF...")
    from gguf import GGUFWriter, GGMLQuantizationType
    ref_path = WORK / "omnivoice-ref.gguf"
    w = GGUFWriter(str(ref_path), arch="omnivoice_ref", use_temp_file=False)
    w.add_name("omnivoice-reference")
    w.add_string("omnivoice_ref.syn_text", SYN_TEXT)

    for name, arr in stages.items():
        arr = np.ascontiguousarray(arr.astype(np.float32))
        w.add_tensor(name, arr, raw_dtype=GGMLQuantizationType.F32)
        log(f"  ref tensor: {name:30s} {arr.shape}")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    log(f"Reference GGUF: {ref_path} ({ref_path.stat().st_size/1e6:.1f} MB)")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Phase 2: Build CrispASR
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    log("Phase 2: Building CrispASR...")
    os.chdir(_CRISPASR_DIR)
    subprocess.run(["git", "submodule", "update", "--init", "ggml"], check=True)

    if HAS_KH:
        kh.install_build_toolchain()
        cuda_arch = kh.detect_cuda_arch()
        cmake_flags = kh.cuda_build_flags(cuda_arch)
        cache_flags = kh.cache_and_link_flags()
        n_jobs = kh.safe_build_jobs(gpu=True)
        # cuda_build_flags / cache_and_link_flags return lists
        cf = " ".join(cmake_flags) if isinstance(cmake_flags, list) else str(cmake_flags)
        cc = " ".join(cache_flags) if isinstance(cache_flags, list) else str(cache_flags)
        build_cmd = (
            f"cmake -G Ninja -B build {cf} {cc} "
            f"-DCMAKE_BUILD_TYPE=Release && "
            f"cmake --build build --target crispasr-cli -j{n_jobs}"
        )
        with kh.build_heartbeat("omnivoice-build"):
            kh.sh_with_progress(build_cmd)
    else:
        # Fallback build without harness
        subprocess.run("apt-get update -qq && apt-get install -y -qq ninja-build ccache", shell=True)
        subprocess.run(
            "cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release "
            "-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache && "
            "cmake --build build --target crispasr-cli -j4",
            shell=True, check=True)

    cli_bin = _CRISPASR_DIR / "build" / "bin" / "crispasr"
    assert cli_bin.exists(), f"CLI binary not found at {cli_bin}"
    log(f"CLI built: {cli_bin}")

    # ── Convert model to GGUF ───────────────────────────────────────
    log("Converting OmniVoice to GGUF...")
    gguf_path = WORK / "omnivoice-f16.gguf"
    if not gguf_path.exists():
        subprocess.run([
            sys.executable, "models/convert-omnivoice-to-gguf.py",
            "--input", MODEL_ID, "--output", str(gguf_path),
        ], check=True, env={**os.environ, "OMP_NUM_THREADS": "1",
                            "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
    log(f"GGUF: {gguf_path} ({gguf_path.stat().st_size/1e9:.2f} GB)")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Phase 3: Generate codes + decode + ASR roundtrip
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    log("Phase 3: Code generation via C++...")
    gen_result = subprocess.run(
        [str(cli_bin), "--backend", "omnivoice", "--model", str(gguf_path),
         "--tts", SYN_TEXT, "--no-gpu", "-t", "4"],
        capture_output=True, text=True, timeout=900,
        env={**os.environ, "OMNIVOICE_DEBUG": "1"},
    )
    log(f"Code gen exit: {gen_result.returncode}")
    # Log last 30 lines of stderr
    for line in gen_result.stderr.strip().split("\n")[-30:]:
        log(f"  {line}")

    import re
    m = re.search(r"generated (\d+) codes", gen_result.stderr)
    n_codes = int(m.group(1)) if m else 0
    log(f"Generated {n_codes} codes")

    # ── Decode + ASR via Python OmniVoice ───────────────────────────
    if model is not None and n_codes > 0:
        log("Generating audio via Python OmniVoice.generate()...")
        import soundfile as sf

        try:
            with torch.no_grad():
                audio_list = model.generate(text=SYN_TEXT, language="en")

            if audio_list and len(audio_list) > 0:
                audio = audio_list[0]
                sr = getattr(model, 'sampling_rate', 24000) or 24000
                out_wav = WORK / "omnivoice_output.wav"
                sf.write(str(out_wav), audio, sr)
                duration = len(audio) / sr
                log(f"Audio: {out_wav} ({duration:.2f}s, {sr} Hz, {len(audio)} samples)")

                # ASR roundtrip
                log("ASR roundtrip...")
                try:
                    import whisper
                    asr_model = whisper.load_model("base", device="cpu")
                    result = asr_model.transcribe(str(out_wav), language="en")
                    transcript = result["text"].strip()
                    match = transcript.lower().strip("., !?") == SYN_TEXT.lower().strip("., !?")
                    log(f"")
                    log(f"{'='*60}")
                    log(f"  INPUT:     '{SYN_TEXT}'")
                    log(f"  ASR:       '{transcript}'")
                    log(f"  MATCH:     {match}")
                    log(f"{'='*60}")
                except Exception as e:
                    log(f"ASR failed: {e}")
            else:
                log("Python generate() returned empty audio")
        except Exception as e:
            log(f"Python generate failed: {e}")
            traceback.print_exc()
    else:
        log("Skipping Python decode (model not loaded or no codes)")

    # ── Summary ─────────────────────────────────────────────────────
    summary = {
        "syn_text": SYN_TEXT,
        "n_codes": n_codes,
        "T_text": T_text,
        "stages": list(stages.keys()),
    }
    with open(WORK / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Summary: {json.dumps(summary, indent=2)}")
    log("=== DONE ===")

except Exception as e:
    log(f"FATAL ERROR: {e}")
    traceback.print_exc()
    with open(LOG, "a") as f:
        traceback.print_exc(file=f)
