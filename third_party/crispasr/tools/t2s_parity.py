#!/usr/bin/env python
"""Per-stage parity for the Confucius4 T2S (text -> semantic) stage.

The S2A stage is exact (cos 1.000000) yet the reference S2A produces babble on
the C++'s semantic codes -- so the codes themselves are wrong and the bug is in
T2S.  This drives the reference Text2Semantic on the SAME text ids and speaker
conditioning the runtime used and compares, in execution order:

  condition_emb   -- validates the ggml ECAPA port in core/ecapa_tdnn.h against
                     the reference speaker_encoder
  prefill_logits  -- the first semantic distribution: everything in the prefix
                     (text projector, position embeddings, GPT-2) folds in here
  lm_latent       -- teacher-forced hidden states for the generated codes

Usage:
  python t2s_parity.py --dump-dir <dir> --ref-repo <clone> \
      --t2s-ckpt t2s_model.safetensors --w2v-features w2v.bin
"""
import argparse
import os
import sys

import numpy as np
import torch


def load_shapes(d):
    shp = {}
    with open(os.path.join(d, "shapes.txt")) as f:
        for line in f:
            if line.strip():
                name, dims = line.rstrip("\n").split("\t")
                shp[name] = tuple(int(x) for x in dims.split(","))
    return shp


def load(d, name, shp, dtype=np.float32):
    return np.fromfile(os.path.join(d, name + ".bin"), dtype=dtype).reshape(shp[name])


def cmp(tag, mine, ref):
    mine = np.asarray(mine, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if mine.shape != ref.shape:
        print(f"{tag:22s} SHAPE MISMATCH mine={mine.shape} ref={ref.shape}")
        return
    m, r = mine.ravel(), ref.ravel()
    cos = float(m @ r / (np.linalg.norm(m) * np.linalg.norm(r) + 1e-12))
    print(f"{tag:22s} cos={cos:.6f}  |mine|={np.linalg.norm(m):12.4f}  |ref|={np.linalg.norm(r):12.4f}  "
          f"ratio={np.linalg.norm(m)/(np.linalg.norm(r)+1e-12):7.4f}  max_abs_diff={np.abs(m-r).max():.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--ref-repo", required=True)
    ap.add_argument("--t2s-ckpt", required=True)
    ap.add_argument("--w2v-features", required=True,
                    help="raw f32 (n_frames, 1024) w2v-BERT layer-17 features")
    a = ap.parse_args()

    sys.path.insert(0, a.ref_repo)
    import yaml
    import safetensors.torch
    from confuciustts.llm.llm import Text2Semantic, Text2SemanticConfig

    d = a.dump_dir
    shp = load_shapes(d)
    cfg = yaml.safe_load(open(os.path.join(a.ref_repo, "config", "inference_config.yaml")))

    text_ids = load(d, "text_ids_i32", shp, np.int32)
    codes = load(d, "semantic_codes_i32", shp, np.int32)
    feats = np.fromfile(a.w2v_features, dtype=np.float32).reshape(1, -1, 1024)
    print(f"text_ids={len(text_ids)} codes={len(codes)} w2v={feats.shape}")

    t2s_cfg = Text2SemanticConfig(**cfg["t2s_model"])
    model = Text2Semantic(t2s_cfg)
    model.config.vocab_size = t2s_cfg.semantic_vocab_size
    model.load_state_dict(safetensors.torch.load_file(a.t2s_ckpt, device="cpu"))
    model.eval()

    ti = torch.from_numpy(text_ids.astype(np.int64)).unsqueeze(0)
    cv = torch.from_numpy(feats)

    with torch.no_grad():
        # ---- condition_emb: checks the ggml ECAPA port ----
        cond_ref = model.speaker_encoder(cv)[0].numpy()
        if "condition_emb" in shp:
            cmp("condition_emb (ECAPA)", load(d, "condition_emb", shp), cond_ref)
        else:
            print("condition_emb        not dumped (runtime had no w2v features)")

        # ---- teacher-forced prefix + codes ----
        bos = model.config.start_semantic_token
        sem = torch.cat([torch.full((1, 1), bos, dtype=torch.long),
                         torch.from_numpy(codes.astype(np.int64)).unsqueeze(0)], dim=1)
        embeds = model._prepare_embed_inputs(text_inputs=ti, semantic_codes=sem, condition_vector=cv)
        out = model.transformer(inputs_embeds=embeds, use_cache=False, return_dict=True)
        h = out.last_hidden_state
        logits = model.semantic_head(model.final_norm(h))

        prefix_len = 1 + ti.shape[1]     # condition(1) + text
        # pre-block-0 structural gate: the concatenated prefix embedding
        # [cond | text+pos | BOS+sem_pos0] BEFORE any transformer block.
        # cos < 0.99999 here means the bug is in the embeds, not the stack.
        if "prefix_emb" in shp:
            cmp("prefix_emb (pre-blk0)", load(d, "prefix_emb", shp),
                embeds[0, :prefix_len + 1].numpy())

        # the runtime's prefill logits come from the LAST prefix position (BOS),
        # i.e. the distribution that predicts the first semantic code
        if "prefill_logits" in shp:
            cmp("prefill_logits", load(d, "prefill_logits", shp), logits[0, prefix_len].numpy())
            mine = load(d, "prefill_logits", shp)
            print(f"  argmax  mine={int(np.argmax(mine))}  ref={int(logits[0, prefix_len].argmax())}"
                  f"  first generated code={int(codes[0])}")

        if "lm_latent" in shp:
            lat = load(d, "lm_latent", shp)
            n = min(lat.shape[0], h.shape[1] - prefix_len)
            cmp("lm_latent", lat[:n], h[0, prefix_len:prefix_len + n].numpy())


if __name__ == "__main__":
    main()
