#!/usr/bin/env python3
"""Executable spec for the BTC chord ggml graph — validates it against PyTorch.

Usage:
    python tools/btc_torch_parity.py <btc-chords.gguf> <btc_model.pt> <BTC-ISMIR19 dir> [ref-dump.gguf]

Passing a fourth argument writes a per-stage reference GGUF for the C++ diff
(`crispasr-diff btc`). The dump includes `input_feat`, so the runtime replays
the EXACT features the spec scored rather than recomputing a CQT — a front-end
difference can then never masquerade as a model parity failure.

Reimplements the BTC forward pass in numpy EXACTLY as src/btc_chords.cpp must
build it, reading the weights back out of the converted GGUF, and scores it
against the real PyTorch model. Keep the two in lockstep: if the C++ graph
changes, change this first and re-run it.

Geometry this pins down (all of it verified against the upstream source, and
every one of these is a silent bug if assumed — see BTC_BLUEPRINT.md):

  * Blocks of exactly `timestep` = 108 frames. The bias mask is built for that
    length; this is NOT a sliding window.
  * Each layer runs TWO attention blocks over the SAME input: a forward one
    masked `triu(-inf, 1)` (causal) and a backward one with that mask
    TRANSPOSED. Outputs are concatenated (256) and projected back to 128.
  * Positional encoding is concat([sin, cos]) — two contiguous halves, NOT
    interleaved.
  * The FFN is Conv(k=3) -> ReLU -> Conv(k=3) with SYMMETRIC (1,1) padding
    (btc_model.py passes padding='both'; the module's documented 'left' option
    would shift the output by a frame).
  * Attention scale multiplies Q by (key_depth // heads) ** -0.5.
  * LayerNorm eps = 1e-6 OUTSIDE the sqrt, and std is UNBIASED (ddof=1).
  * The FFN has a TRAILING ReLU after the second conv — upstream's loop guard
    `if i < len(self.layers)` is always true. Baked into the weights.
  * The head is a bare Linear — output_layer.lstm.* is never called.

The upstream repo is MIT; it is used here only as an oracle (running it
produces facts) and is NOT vendored.
"""

import sys

import numpy as np
from gguf import GGUFReader

if len(sys.argv) not in (4, 5):
    sys.exit(__doc__)
GGUF_PATH, CKPT_PATH, BTC_DIR = sys.argv[1:4]

# ---------------------------------------------------------------- gguf weights
reader = GGUFReader(GGUF_PATH)
W = {t.name: np.array(t.data, dtype=np.float32).reshape([int(d) for d in reversed(t.shape)])
     for t in reader.tensors}


def kv(key, default=None):
    for f in reader.fields.values():
        if f.name == key:
            return f.parts[f.data[0]][0]
    return default


HIDDEN = int(kv("btc.hidden_size", 128))
LAYERS = int(kv("btc.n_layers", 8))
HEADS = int(kv("btc.n_heads", 4))
T = int(kv("btc.timestep", 108))
EPS = float(kv("btc.layer_norm_eps", 1e-6))
MEAN = float(kv("btc.norm_mean"))
STD = float(kv("btc.norm_std"))
FEAT = int(kv("btc.feature_size", 144))
HEAD_DIM = HIDDEN // HEADS


# ---------------------------------------------------------------- numpy forward
def layer_norm(x, gamma, beta):
    # gamma * (x - mean) / (std + eps) + beta, with eps OUTSIDE the sqrt.
    # std is torch's UNBIASED std (ddof=1); numpy defaults to ddof=0, which is
    # a 0.4% error at width 128 — enough to matter (max_abs 1.5e-2 vs 5e-7).
    mu = x.mean(-1, keepdims=True)
    sd = x.std(-1, keepdims=True, ddof=1)
    return gamma * (x - mu) / (sd + EPS) + beta


def timing_signal(length, channels, min_ts=1.0, max_ts=1.0e4):
    pos = np.arange(length, dtype=np.float32)
    n = channels // 2
    inc = np.log(max_ts / min_ts) / (n - 1)
    inv = min_ts * np.exp(np.arange(n, dtype=np.float32) * -inc)
    scaled = pos[:, None] * inv[None, :]
    # CONCATENATED halves, not interleaved.
    return np.concatenate([np.sin(scaled), np.cos(scaled)], axis=1)


def attention(x, p, mask):
    q = x @ W[f"{p}.attn.q.weight"].T
    k = x @ W[f"{p}.attn.k.weight"].T
    v = x @ W[f"{p}.attn.v.weight"].T
    n = x.shape[0]
    q = q.reshape(n, HEADS, HEAD_DIM).transpose(1, 0, 2)
    k = k.reshape(n, HEADS, HEAD_DIM).transpose(1, 0, 2)
    v = v.reshape(n, HEADS, HEAD_DIM).transpose(1, 0, 2)
    q = q * (HEAD_DIM ** -0.5)          # scale applied to Q, not the logits
    logits = q @ k.transpose(0, 2, 1) + mask[None, :, :]
    logits -= logits.max(-1, keepdims=True)
    w = np.exp(logits)
    w /= w.sum(-1, keepdims=True)
    out = (w @ v).transpose(1, 0, 2).reshape(n, HIDDEN)
    return out @ W[f"{p}.attn.o.weight"].T


def ffn(x, p):
    # Conv(k=3) -> ReLU -> Conv(k=3), SYMMETRIC (1,1) padding.
    def conv(sig, wt, b):
        pad = np.zeros((sig.shape[0] + 2, sig.shape[1]), dtype=np.float32)
        pad[1:-1] = sig
        out = np.zeros((sig.shape[0], wt.shape[0]), dtype=np.float32)
        for t in range(sig.shape[0]):
            win = pad[t:t + 3]                    # (3, in)
            out[t] = np.einsum("oik,ki->o", wt, win) + b
        return out

    h = conv(x, W[f"{p}.ffn.0.weight"], W[f"{p}.ffn.0.bias"])
    h = np.maximum(h, 0.0)
    h = conv(h, W[f"{p}.ffn.1.weight"], W[f"{p}.ffn.1.bias"])
    # TRAILING ReLU. Upstream's loop guard is `if i < len(self.layers)`, which
    # is always true (i maxes at len-1) — clearly meant to be len-1, so ReLU
    # fires after the LAST conv too. It is baked into the trained weights, so
    # it must be reproduced, bug or not.
    return np.maximum(h, 0.0)


def block(x, p, mask):
    xn = layer_norm(x, W[f"{p}.norm_mha.gamma"], W[f"{p}.norm_mha.beta"])
    x = x + attention(xn, p, mask)
    xn = layer_norm(x, W[f"{p}.norm_ffn.gamma"], W[f"{p}.norm_ffn.beta"])
    return x + ffn(xn, p)


STAGES = {}


def btc_numpy(feat):
    """feat: (T, 144) raw log-CQT, BEFORE normalisation."""
    STAGES["input_feat"] = feat.astype(np.float32)
    x = (feat - MEAN) / STD
    x = x @ W["embedding_proj.weight"].T            # no bias, no sqrt(d) scaling
    x = x + timing_signal(x.shape[0], HIDDEN)
    STAGES["embed_posenc"] = x.astype(np.float32)

    causal = np.triu(np.full((T, T), -np.inf, dtype=np.float32), 1)
    anti = causal.T.copy()
    n = x.shape[0]
    causal, anti = causal[:n, :n], anti[:n, :n]

    for i in range(LAYERS):
        fwd = block(x, f"layers.{i}.fwd", causal)
        bwd = block(x, f"layers.{i}.bwd", anti)
        if i == 0:
            STAGES["layer0_fwd"] = fwd.astype(np.float32)
            STAGES["layer0_bwd"] = bwd.astype(np.float32)
        cat = np.concatenate([fwd, bwd], axis=1)     # (n, 256)
        x = cat @ W[f"layers.{i}.proj.weight"].T + W[f"layers.{i}.proj.bias"]
        STAGES[f"layer{i}_out"] = x.astype(np.float32)

    x = layer_norm(x, W["final_norm.gamma"], W["final_norm.beta"])
    STAGES["final_norm"] = x.astype(np.float32)
    out = x @ W["output.proj.weight"].T + W["output.proj.bias"]
    STAGES["logits"] = out.astype(np.float32)
    return out


# ---------------------------------------------------------------- torch oracle
sys.path.insert(0, BTC_DIR)
# The 2019 upstream predates numpy 1.20 / PyYAML 6: it uses np.float and calls
# yaml.load() without a Loader. Shim rather than patch their tree — we only run
# it as an oracle and do not vendor it.
for _alias, _real in (("float", float), ("int", int), ("bool", bool), ("object", object)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _real)

import torch                                            # noqa: E402
from btc_model import BTC_model                         # noqa: E402
import yaml                                           # noqa: E402

# Their utils.hparams calls yaml.load() without a Loader (removed in PyYAML 6),
# so read the config directly rather than importing HParams.
with open(f"{BTC_DIR}/run_config.yaml") as fh:
    model_cfg = yaml.safe_load(fh)["model"]
model_cfg["probs_out"] = True
ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
model_cfg["num_chords"] = int(ck["model"]["output_layer.output_projection.weight"].shape[0])

model = BTC_model(config=model_cfg)
model.load_state_dict(ck["model"])
model.eval()

rng = np.random.default_rng(0)
feat = rng.standard_normal((T, FEAT)).astype(np.float32) * 2.0 - 2.0   # log-CQT-ish

with torch.no_grad():
    xt = torch.from_numpy(((feat - MEAN) / STD)[None])
    hidden, _ = model.self_attn_layers(xt)
    ref = model.output_layer.output_projection(hidden)[0].numpy()

mine = btc_numpy(feat)

num = float((mine * ref).sum())
den = float(np.linalg.norm(mine) * np.linalg.norm(ref))
cos = num / den if den else 0.0
mx = float(np.abs(mine - ref).max())
agree = float((mine.argmax(-1) == ref.argmax(-1)).mean())

# Optional reference dump for the C++ per-stage diff (crispasr-diff btc).
if len(sys.argv) > 4:
    import gguf as _g
    w = _g.GGUFWriter(sys.argv[4], "btc-ref")
    for k, v in STAGES.items():
        w.add_tensor(k, np.ascontiguousarray(v, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote reference dump {sys.argv[4]}: {len(STAGES)} stages")

print(f"btc parity: cos={cos:.8f} max_abs={mx:.3e} argmax_agree={agree:.4f} "
      f"shape={mine.shape} chords={mine.shape[-1]}")
if cos < 0.9999 or agree < 1.0:
    sys.exit("FAIL: numpy spec does not match the torch reference")
print("PASS")
