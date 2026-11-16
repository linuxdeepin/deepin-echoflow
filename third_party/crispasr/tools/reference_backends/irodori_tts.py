"""Irodori-TTS reference dump backend for crispasr-diff.

Loads Aratako/Irodori-TTS (base or VoiceDesign) and captures intermediate
activations at each architectural boundary for comparison against C++.

Stages:
  text_embedding     (T, text_dim)    raw embedding lookup (before encoder blocks)
  text_state         (T, text_dim)    text encoder output (after all blocks + norm)
  cap_state          (T_cap, cap_dim) caption encoder output (VoiceDesign only)
  spk_state_zeros    (T_ref, spk_dim) speaker encoder output with zero reference
  timestep_embed     (timestep_embed_dim,) sinusoidal timestep embedding at t=0.999
  cond_embed         (model_dim*3,)   timestep conditioning MLP output
  dit_in_proj        (model_dim, T_latent) DiT input projection output
  dit_block_0        (model_dim, T_latent) after first DiT block
  v_pred_step0       (latent_dim, T_latent) velocity prediction at ODE step 0
  ode_step_0         (latent_dim, T_latent) x_t after first Euler step

Usage:
  python tools/dump_reference.py --backend irodori-tts \\
      --model-dir Aratako/Irodori-TTS-600M-v3-VoiceDesign \\
      --audio samples/jfk.wav \\
      --output /mnt/volume1/tmp-overflow/irodori-ref.gguf

Environment variables:
  IRODORI_TEST_TEXT      text to synthesize (default: "こんにちは、世界。")
  IRODORI_CAPTION_TEXT   caption for VoiceDesign (default: none)
  IRODORI_SEED           random seed for noise (default: 42)
  IRODORI_ODE_STEPS      number of ODE steps to run (default: 2, for speed)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "text_embedding",
    "text_state",
    "cap_state",
    "spk_state_zeros",
    "timestep_embed",
    "cond_embed",
    "init_noise",
    "dit_in_proj",
    "dit_block_0",
    "v_pred_step0",
    "ode_step_0",
]


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run Irodori-TTS reference forward and return stage captures."""
    import gc
    import json
    import sys

    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    results: Dict[str, np.ndarray] = {}

    # Resolve checkpoint
    model_path = str(model_dir)
    ckpt_path = None
    if Path(model_path).is_file() and model_path.endswith(".safetensors"):
        ckpt_path = model_path
    elif Path(model_path).is_dir():
        for candidate in ["model.safetensors", "*.safetensors"]:
            import glob
            hits = glob.glob(str(Path(model_path) / candidate))
            if hits:
                ckpt_path = hits[0]
                break
    else:
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(repo_id=model_path, filename="model.safetensors")

    if not ckpt_path:
        raise FileNotFoundError(f"No .safetensors found in {model_path}")

    print(f"[irodori-ref] checkpoint: {ckpt_path}")

    # Load config from checkpoint metadata
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
    config = json.loads(metadata.get("config_json", "{}"))

    # Load Irodori-TTS source
    irodori_src = os.environ.get("IRODORI_SRC")
    if irodori_src:
        sys.path.insert(0, irodori_src)
    else:
        # Try common locations
        for candidate in ["/kaggle/temp/irodori-tts-src",
                          "/mnt/volume1/tmp-overflow/irodori-tts-src",
                          str(Path(__file__).parent.parent.parent / "irodori-tts-src")]:
            if Path(candidate).exists():
                sys.path.insert(0, candidate)
                break
        else:
            import subprocess
            irodori_src = "/kaggle/temp/irodori-tts-src" if Path("/kaggle").exists() else "/tmp/irodori-tts-src"
            subprocess.check_call([
                "git", "clone", "--depth", "1",
                "https://github.com/Aratako/Irodori-TTS.git", irodori_src,
            ])
            sys.path.insert(0, irodori_src)

    from irodori_tts.config import ModelConfig
    from irodori_tts.model import TextToLatentRFDiT, get_timestep_embedding, precompute_freqs_cis

    # Build model config
    cfg_fields = {k: v for k, v in config.items() if hasattr(ModelConfig, k)}
    model_cfg = ModelConfig(**cfg_fields)

    # Load tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_cfg.text_tokenizer_repo, use_fast=True)

    # Tokenize test text
    test_text = os.environ.get("IRODORI_TEST_TEXT", "こんにちは、世界。")
    token_ids = [tok.bos_token_id] + tok.encode(test_text, add_special_tokens=False)
    print(f"[irodori-ref] text: '{test_text}' → {len(token_ids)} tokens: {token_ids}")

    # Load model
    print("[irodori-ref] loading model weights...")
    state = load_file(ckpt_path, device="cpu")
    model = TextToLatentRFDiT(model_cfg)
    model.load_state_dict(state, strict=False)
    model.eval()
    del state
    gc.collect()
    print(f"[irodori-ref] model loaded (text_dim={model_cfg.text_dim}, model_dim={model_cfg.model_dim})")

    text_ids = torch.tensor([token_ids], dtype=torch.long)
    text_mask = torch.ones_like(text_ids, dtype=torch.bool)

    with torch.inference_mode():
        # ── Text embedding (before encoder blocks) ──
        if "text_embedding" in stages:
            emb = model.text_encoder.text_embedding(text_ids)
            results["text_embedding"] = emb[0].float().numpy()
            print(f"  text_embedding: {emb.shape}")

        # ── Full text encoder ──
        text_state = model.text_encoder(text_ids, text_mask)
        text_state = model.text_norm(text_state)
        if "text_state" in stages:
            results["text_state"] = text_state[0].float().numpy()
            print(f"  text_state: {text_state.shape}, first 4: {text_state[0, 0, :4].tolist()}")

        # ── Caption encoder (VoiceDesign) ──
        cap_state_for_dit = None
        cap_mask_for_dit = None
        if model_cfg.use_caption_condition and model.caption_encoder is not None:
            caption_text = os.environ.get("IRODORI_CAPTION_TEXT",
                                          "落ち着いた大人の男性。")
            cap_tok_repo = model_cfg.caption_tokenizer_repo_resolved
            cap_tok = AutoTokenizer.from_pretrained(cap_tok_repo, use_fast=True)
            cap_ids_raw = cap_tok.encode(caption_text, add_special_tokens=False)
            if model_cfg.caption_add_bos_resolved and cap_tok.bos_token_id is not None:
                cap_ids_raw = [cap_tok.bos_token_id] + cap_ids_raw
            cap_ids = torch.tensor([cap_ids_raw], dtype=torch.long)
            cap_mask = torch.ones_like(cap_ids, dtype=torch.bool)
            cap_state = model.caption_encoder(cap_ids, cap_mask)
            cap_state = model.caption_norm(cap_state)
            if "cap_state" in stages:
                results["cap_state"] = cap_state[0].float().numpy()
                print(f"  cap_state: {cap_state.shape} (caption: '{caption_text}')")
            cap_state_for_dit = cap_state
            cap_mask_for_dit = cap_mask

        # ── Speaker encoder (zeros = unconditional) ──
        if "spk_state_zeros" in stages and model_cfg.use_speaker_condition_resolved:
            latent_dim = model_cfg.latent_dim * model_cfg.latent_patch_size
            dummy_ref = torch.zeros(1, 10, latent_dim)
            dummy_mask = torch.ones(1, 10, dtype=torch.bool)
            if model.speaker_encoder is not None:
                spk_state = model.speaker_encoder(dummy_ref, dummy_mask)
                spk_state = model.speaker_norm(spk_state)
                results["spk_state_zeros"] = spk_state[0].float().numpy()
                print(f"  spk_state_zeros: {spk_state.shape}")

        # ── Timestep embedding ──
        t_val = 0.999
        t_tensor = torch.tensor([t_val])
        t_emb = get_timestep_embedding(t_tensor, model_cfg.timestep_embed_dim)
        if "timestep_embed" in stages:
            results["timestep_embed"] = t_emb[0].float().numpy()
            print(f"  timestep_embed: {t_emb.shape}")

        # ── Conditioning MLP ──
        cond_embed = model.cond_module(t_emb.to(dtype=model.dtype))
        if "cond_embed" in stages:
            results["cond_embed"] = cond_embed[0].float().numpy()
            print(f"  cond_embed: {cond_embed.shape}")

        # ── DiT forward (1 step) ──
        seed = int(os.environ.get("IRODORI_SEED", "42"))
        T_latent = 10  # small for speed
        latent_d = model_cfg.patched_latent_dim

        torch.manual_seed(seed)
        x_t = torch.randn(1, T_latent, latent_d, dtype=model.dtype)
        if "init_noise" in stages:
            results["init_noise"] = x_t[0].float().numpy()
            print(f"  init_noise: {x_t.shape}")
        cond_1d = cond_embed[:, None, :]  # (1, 1, D*3) broadcast

        # Prepare speaker state (zeros = unconditional, but must not be None
        # when speaker conditioning is enabled in the model config)
        spk_state_for_dit = None
        spk_mask_for_dit = None
        if model_cfg.use_speaker_condition_resolved:
            spk_state_for_dit = torch.zeros(1, 1, model_cfg.speaker_dim, dtype=model.dtype)
            spk_mask_for_dit = torch.zeros(1, 1, dtype=torch.bool)

        # In projection
        x = model.in_proj(x_t)
        if "dit_in_proj" in stages:
            results["dit_in_proj"] = x[0].float().numpy()
            print(f"  dit_in_proj: {x.shape}")

        # RoPE frequencies
        freqs = model._rope_freqs(T_latent, x.device)

        # First DiT block
        if len(model.blocks) > 0:
            x = model.blocks[0](
                x=x, cond_embed=cond_1d,
                text_state=text_state, text_mask=text_mask,
                speaker_state=spk_state_for_dit, speaker_mask=spk_mask_for_dit,
                caption_state=cap_state_for_dit, caption_mask=cap_mask_for_dit,
                freqs_cis=freqs,
            )
            if "dit_block_0" in stages:
                results["dit_block_0"] = x[0].float().numpy()
                print(f"  dit_block_0: {x.shape}")

        # Full forward for velocity prediction
        v_pred = model.forward_with_encoded_conditions(
            x_t=x_t, t=t_tensor.to(dtype=model.dtype),
            text_state=text_state, text_mask=text_mask,
            speaker_state=spk_state_for_dit, speaker_mask=spk_mask_for_dit,
            caption_state=cap_state_for_dit, caption_mask=cap_mask_for_dit,
        )
        if "v_pred_step0" in stages:
            results["v_pred_step0"] = v_pred[0].float().numpy()
            print(f"  v_pred_step0: {v_pred.shape}")

        # Euler step
        dt = -0.999 / 40.0  # first step dt
        x_after = x_t + v_pred * dt
        if "ode_step_0" in stages:
            results["ode_step_0"] = x_after[0].float().numpy()
            print(f"  ode_step_0: {x_after.shape}")

    print(f"[irodori-ref] captured {len(results)} stages")
    return results
