#!/usr/bin/env python3
"""Per-stage reference dump for the beat-this port (§251b-1).

    PYTHONPATH=<beat_this-src> python tools/reference_backends/beat_this.py \
        --ckpt final0.ckpt --logmel fe_ref.bin --output bt_stages.npz

Dumps the activations after each architectural stage so the ggml graph can be
brought up incrementally — first divergence is the bug, per the diff-harness
rule in docs/contributing.md. Debugging a whole 20 M-param forward at once is
how you end up chasing GELU LUTs for a shape error.

Needs the upstream package on PYTHONPATH (MIT):
    git clone --depth 1 https://github.com/CPJKU/beat_this
plus `rotary-embedding-torch` and `einops`.

Input is a (frames, 128) float32 log-mel — use the SAME fixture the front-end
test uses (tools/gen_beat_this_frontend_ref.py), so a stage mismatch can never
be blamed on feature extraction.

CONFIRMED STAGE SHAPES for a 101-frame input (these are the contract the graph
must reproduce; they also confirm the traced architecture):
    stem          (1,  32, 32, 101)     32 ch, freq 128->32
    blk0          (1,  64, 16, 101)     dims double, freq halves, x3
    blk1          (1, 128,  8, 101)
    blk2          (1, 256,  4, 101)
    linear        (1, 101, 512)         concat 256*4=1024 -> 512
    transformer   (1, 101, 512)
    out           beat, downbeat: (1, 101) each
"""
import argparse
import os
import sys
from pathlib import Path

# ⚠ SELF-SHADOWING: this file is `beat_this.py` (the repo's reference-backend
# naming convention) and Python puts a script's own directory FIRST on
# sys.path, so `import beat_this` would resolve to THIS file rather than the
# upstream package. Drop our own directory before importing anything.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--logmel", type=Path, required=True, help="(frames,128) f32")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from beat_this.model.beat_tracker import BeatThis

    m = BeatThis().eval()
    ck = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        # Loud, because a silently-partial load produces plausible activations.
        print(f"WARNING: missing={len(missing)} unexpected={len(unexpected)}")
        for k in list(missing)[:5]:
            print("  missing:", k)

    acts = {}

    def hook(name):
        def f(mod, i, o):
            t = o[0] if isinstance(o, (tuple, list)) else o
            if torch.is_tensor(t):
                acts[name] = t.detach().float().numpy()
        return f

    m.frontend.stem.register_forward_hook(hook("stem"))
    for i in range(3):
        pt = m.frontend.blocks[i].partial
        # Sub-block granularity: attnF/ffF/attnT/ffT are where the port's real
        # failure modes live (RoPE axis, per-head gating, residual placement),
        # so each gets its own reference rather than only the fused output.
        # NOTE these capture the sub-block OUTPUT, i.e. the residual BRANCH
        # value before `x + branch`, not the post-residual activation.
        pt.attnF.register_forward_hook(hook(f"blk{i}_attnF"))
        pt.ffF.register_forward_hook(hook(f"blk{i}_ffF"))
        pt.attnT.register_forward_hook(hook(f"blk{i}_attnT"))
        pt.ffT.register_forward_hook(hook(f"blk{i}_ffT"))
        pt.register_forward_hook(hook(f"blk{i}_partial"))
        m.frontend.blocks[i].register_forward_hook(hook(f"blk{i}"))
    m.frontend.linear.register_forward_hook(hook("linear"))
    m.transformer_blocks.register_forward_hook(hook("transformer"))

    x = np.fromfile(str(args.logmel), dtype=np.float32).reshape(-1, 128)
    with torch.no_grad():
        out = m(torch.tensor(x)[None])
    for k, v in (out.items() if isinstance(out, dict) else [("beat", out)]):
        acts[f"out_{k}"] = v.detach().float().numpy()

    np.savez(str(args.output), **acts)
    for k, v in acts.items():
        print(f"  {k:16s} {v.shape}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
