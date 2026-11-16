#!/usr/bin/env python
"""Dump Confucius4-TTS speaker conditioning from a reference wav.

The model is zero-shot: ConfuciusTTS.generate always takes a prompt_wav, and
with zero conditioning even the PyTorch reference produces babble.  Porting
Wav2Vec2-BERT (600M) and the ECAPA-TDNN speaker encoder just to prove the rest
of the pipeline is not worth it, so this runs the reference encoders once and
writes the three tensors the runtime needs:

  w2v_features.bin     (n_frames, 1024)        w2v-BERT layer 17, z-normalised
  style_embedding.bin  (spk_embed_dim,)        CAMPPlus
  prompt_mel.bin       (n_frames, n_mels)      reference mel

The ECAPA speaker encoder is NOT run here: its weights already ship in the T2S
GGUF and the runtime forwards them through core/ecapa_tdnn.h (the same graph
qwen3-tts uses).  Only the w2v-BERT features are still external -- sidon.cpp
has the conformer and the SeamlessM4T frontend, so folding that in later is a
converter job plus a layer-17 hook rather than a new port.

Point CRISPASR_CONFUCIUS4_COND_DIR at the output directory.

Usage:
  python confucius4_dump_conditioning.py --ref-repo <clone> --prompt-wav ref.wav \
      --t2s-ckpt t2s_model.safetensors --out-dir cond/
"""
import argparse
import os
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-repo", required=True,
                    help="clone of github.com/netease-youdao/Confucius4-TTS")
    ap.add_argument("--prompt-wav", required=True)
    ap.add_argument("--t2s-ckpt", default=None, help="t2s_model.safetensors (needed for condition_emb)")
    ap.add_argument("--w2v-stats", default=None, help="wav2vec2bert_stats.pt")
    ap.add_argument("--no-w2v", action="store_true",
                    help="skip condition_emb: writes only style_embedding + prompt_mel, "
                         "which needs neither the 2.4 GB w2v-BERT nor the 2.6 GB T2S "
                         "checkpoint.  Isolates whether the S2A conditioning alone "
                         "is enough to turn the semantic codes into speech.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", default=None, help="defaults to <ref-repo>/config/inference_config.yaml")
    a = ap.parse_args()

    sys.path.insert(0, a.ref_repo)
    import yaml
    import torchaudio
    import safetensors.torch
    from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel
    from huggingface_hub import hf_hub_download

    from external.campplus import CAMPPlus
    from confuciustts.llm.llm import Text2SemanticConfig
    from confuciustts.llm.speaker_encoder import (Qwen3TTSSpeakerEncoder,
                                                  Qwen3TTSSpeakerEncoderConfig)
    from confuciustts.utils.audio_features import mel_spectrogram

    cfg_path = a.config or os.path.join(a.ref_repo, "config", "inference_config.yaml")
    cfg = yaml.safe_load(open(cfg_path))
    audio = cfg["audio"]
    os.makedirs(a.out_dir, exist_ok=True)

    # ---- load and resample the prompt exactly as _load_prompt does ----
    wav, sr = torchaudio.load(a.prompt_wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    sr_tgt = audio["target_sample_rate"]
    wav_16k = wav if sr == 16000 else torchaudio.functional.resample(wav, sr, 16000)
    wav_tgt = wav if sr == sr_tgt else torchaudio.functional.resample(wav, sr, sr_tgt)
    print(f"prompt: {a.prompt_wav}  {wav.shape[-1] / sr:.2f}s @ {sr} Hz")

    # ---- reference mel (prompt_feat) ----
    mel = mel_spectrogram(wav_tgt.float(), sample_rate=sr_tgt, n_fft=audio["n_fft"],
                          hop_length=audio["hop_length"], win_length=audio["win_length"],
                          n_mels=audio["n_mels"], fmin=audio["fmin"], fmax=audio["fmax"])
    prompt_mel = mel.transpose(1, 2).contiguous()[0]          # (T_ref, n_mels)
    print(f"prompt_mel      : {tuple(prompt_mel.shape)}")

    # ---- CAMPPlus style embedding ----
    spk_cfg = cfg["paths"]["style_encoder"]
    style_encoder = CAMPPlus(**spk_cfg.get("init_args", {}))
    spk_state = torch.load(hf_hub_download("funasr/campplus", filename=spk_cfg["checkpoint"]),
                           map_location="cpu")
    if isinstance(spk_state, dict) and "state_dict" in spk_state:
        spk_state = spk_state["state_dict"]
    style_encoder.load_state_dict(spk_state, strict=False)
    style_encoder.eval()
    fbank = torchaudio.compliance.kaldi.fbank(wav_16k, num_mel_bins=80,
                                              sample_frequency=16000, dither=0.0)
    fbank = fbank - fbank.mean(dim=0, keepdim=True)
    with torch.no_grad():
        style = style_encoder(fbank.unsqueeze(0))[0]
    print(f"style_embedding : {tuple(style.shape)}")

    if a.no_w2v:
        for name, arr in (("style_embedding", style), ("prompt_mel", prompt_mel)):
            p = os.path.join(a.out_dir, name + ".bin")
            np.ascontiguousarray(arr.numpy(), dtype=np.float32).tofile(p)
            print(f"wrote {p}  {int(np.prod(arr.shape))} floats")
        print("skipped condition_emb (--no-w2v)")
        return

    if not a.t2s_ckpt or not a.w2v_stats:
        sys.exit("--t2s-ckpt and --w2v-stats are required unless --no-w2v is given")

    # ---- Wav2Vec2-BERT layer 17, z-normalised ----
    w2v_path = cfg["paths"]["w2v_bert_path"]
    fe = SeamlessM4TFeatureExtractor.from_pretrained(w2v_path)
    w2v = Wav2Vec2BertModel.from_pretrained(w2v_path).eval()
    stats = torch.load(a.w2v_stats, map_location="cpu")
    mean, std = stats["mean"], torch.sqrt(stats["var"])
    inputs = fe(wav_16k.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        out = w2v(input_features=inputs["input_features"],
                  attention_mask=inputs.get("attention_mask"),
                  output_hidden_states=True)
    feats = (out.hidden_states[17] - mean) / std                # (1, T_feat, 1024)
    print(f"w2v-bert L17    : {tuple(feats.shape)}")
    del w2v

    # ---- T2S speaker encoder -> the (1, model_dim) condition embedding ----
    # Build the submodule alone: instantiating Text2Semantic would allocate the
    # whole 2.6 GB GPT-2 just to reach speaker_encoder.  Config mirrors
    # Text2Semantic.__init__: mel_dim=speaker_embedding_dim, enc_dim=model_dim.
    t2s_cfg = Text2SemanticConfig(**cfg["t2s_model"])
    spk_enc = Qwen3TTSSpeakerEncoder(Qwen3TTSSpeakerEncoderConfig(
        mel_dim=t2s_cfg.speaker_embedding_dim, enc_dim=t2s_cfg.model_dim))
    with safetensors.torch.safe_open(a.t2s_ckpt, framework="pt", device="cpu") as f:
        sub = {k[len("speaker_encoder."):]: f.get_tensor(k) for k in f.keys()
               if k.startswith("speaker_encoder.")}
    spk_enc.load_state_dict(sub, strict=True)
    spk_enc.eval()
    with torch.no_grad():
        cond_emb = spk_enc(feats)[0]                            # (model_dim,)
    print(f"condition_emb   : {tuple(cond_emb.shape)}")

    for name, arr in (("condition_emb", cond_emb), ("style_embedding", style),
                      ("prompt_mel", prompt_mel), ("w2v_features", feats[0])):
        p = os.path.join(a.out_dir, name + ".bin")
        np.ascontiguousarray(arr.numpy(), dtype=np.float32).tofile(p)
        print(f"wrote {p}  {np.prod(arr.shape)} floats")


if __name__ == "__main__":
    main()
