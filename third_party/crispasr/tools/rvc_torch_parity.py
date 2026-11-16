#!/usr/bin/env python3
"""Executable spec + reference dumper for the RVC voice-conversion port (§CB1).

    python tools/rvc_torch_parity.py <rvc.gguf> <f0G*.pth> <configs/vN/<sr>.json> \
        <RVC-repo-dir> [ref-dump.gguf]

WHY THIS EXISTS BEFORE ANY C++. RVC inference is STOCHASTIC, so the usual
"generate audio and compare waveforms" acceptance test is invalid — two runs of
the reference disagree with each other. Everything downstream of the latent
sample is unverifiable unless the noise is REPLAYED. This tool proves the
replay design works, and only then dumps a reference the C++ diff can use.

Two live RNG sites (docs/music-transcription/RVC_BLUEPRINT.md §2):

  A. z_p = (m_p + exp(logs_p) * randn_like(m_p) * 0.66666) * x_mask   models.py:684
  B. SineGen additive noise, voicing-dependent                        models.py:358

Site B's *phase* term is NOT a third site: GeneratorNSF hardcodes
harmonic_num=0, so SineGen.dim == 1 and `rand_ini[:, 0] = 0` zeroes the only
element. Verified here rather than asserted — see check_phase_is_deterministic.

CometBeat's offline rvc.dart exposes the same z_p noise as an injectable
`rnd [1,192,T]` input (cos 0.99994 vs the ONNX graph), so once this tool and
the ggml graph both consume an injected buffer, a three-way deterministic
harness is possible: this reference -> their Dart -> our ggml, identical noise.
"""

import json
import sys
from pathlib import Path

import numpy as np


def make_injector(buffers):
    """Replace torch's RNG with deterministic draws from `buffers`.

    Returns a context manager. Each call to the patched randn_like/rand pops the
    next buffer whose shape matches, so the SAME sequence is reproducible across
    implementations — which seeding alone cannot give us, since different RNGs
    produce different numbers from the same seed. This mirrors how the RVC ONNX
    export exposes its `rnd` input.
    """
    import torch

    class _Injector:
        def __init__(self):
            self.log = []          # (site, shape) in call order
            self._orig_randn_like = torch.randn_like
            self._orig_rand = torch.rand
            self._queue = list(buffers)

        def _next(self, shape, site):
            self.log.append((site, tuple(shape)))
            for i, b in enumerate(self._queue):
                if tuple(b.shape) == tuple(shape):
                    return torch.from_numpy(self._queue.pop(i).astype(np.float32))
            # Nothing supplied for this draw: use zeros and RECORD it, rather
            # than silently falling back to real randomness (which would make
            # the run non-reproducible again without telling anyone).
            self.log[-1] = (site + ":ZEROED", tuple(shape))
            return torch.zeros(*shape, dtype=torch.float32)

        def __enter__(self):
            inj = self

            def randn_like(t, *a, **k):
                return inj._next(t.shape, "randn_like")

            def rand(*size, **k):
                if len(size) == 1 and isinstance(size[0], (tuple, list)):
                    size = tuple(size[0])
                return inj._next(size, "rand")

            torch.randn_like = randn_like
            torch.rand = rand
            return self

        def __exit__(self, *exc):
            torch.randn_like = self._orig_randn_like
            torch.rand = self._orig_rand
            return False

    return _Injector()


def build_model(ckpt_path, cfg_path, repo_dir):
    import torch
    sys.path.insert(0, str(repo_dir))
    from infer.module.models import SynthesizerTrnMs768NSFsid, SynthesizerTrnMs256NSFsid

    cfg = json.load(open(cfg_path))
    m, d = cfg["model"], cfg["data"]
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck

    content_dim = sd["enc_p.emb_phone.weight"].shape[1]
    n_spk = sd["emb_g.weight"].shape[0]
    Klass = SynthesizerTrnMs768NSFsid if content_dim == 768 else SynthesizerTrnMs256NSFsid

    net = Klass(
        d["filter_length"] // 2 + 1,
        m.get("segment_size", 12800) // int(np.prod(m["upsample_rates"])),
        m["inter_channels"], m["hidden_channels"], m["filter_channels"],
        m["n_heads"], m["n_layers"], m["kernel_size"], m["p_dropout"],
        m["resblock"], m["resblock_kernel_sizes"], m["resblock_dilation_sizes"],
        m["upsample_rates"], m["upsample_initial_channel"], m["upsample_kernel_sizes"],
        n_spk, m["gin_channels"], d["sampling_rate"],
        is_half=False,   # required kwarg; fp32 throughout for a clean reference
    )
    # enc_q is training-only; the generator checkpoint has it but infer() never
    # touches it, so a strict load would fail for the wrong reason.
    missing, unexpected = net.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if not k.startswith("enc_q.")]
    if real_missing:
        raise RuntimeError(
            f"{len(real_missing)} weight(s) missing -> the model would be partly random, "
            f"and a reference dumped from it would silently disagree with everything. "
            f"First few: {real_missing[:5]}"
        )
    net.eval()
    return net, cfg, content_dim


def check_phase_is_deterministic(net):
    """Confirm SineGen's random initial phase is structurally zero.

    The blueprint claims harmonic_num == 0 makes rand_ini a 1-element tensor
    that the next line zeroes. Assert it against the built model instead of
    trusting the reading.
    """
    sg = net.dec.m_source.l_sin_gen
    assert sg.harmonic_num == 0, f"harmonic_num={sg.harmonic_num}, expected 0"
    assert sg.dim == 1, f"SineGen.dim={sg.dim}, expected 1"
    print(f"  SineGen: harmonic_num={sg.harmonic_num} dim={sg.dim} -> rand_ini is 1 element, zeroed => phase deterministic")


def main():
    if len(sys.argv) not in (5, 6):
        sys.exit(__doc__)
    gguf_path, ckpt, cfg_path, repo = sys.argv[1:5]
    import torch

    net, cfg, content_dim = build_model(ckpt, cfg_path, repo)
    m, d = cfg["model"], cfg["data"]
    upp = int(np.prod(m["upsample_rates"]))
    sr = int(d["sampling_rate"])
    inter = m["inter_channels"]
    print(f"rvc spec: content_dim={content_dim} inter={inter} sr={sr} upp={upp} ({sr/upp:.0f} fps)")
    check_phase_is_deterministic(net)

    # Deterministic inputs. T frames at 100 Hz.
    rng = np.random.default_rng(0)
    T = 64
    phone = rng.standard_normal((1, T, content_dim)).astype(np.float32) * 0.1
    f0_hz = np.abs(rng.standard_normal((1, T)).astype(np.float32)) * 120.0 + 100.0
    f0_hz[0, T // 3 : T // 3 + 6] = 0.0  # an unvoiced stretch: exercises the uv branch
    pitch_coarse = coarse_pitch(f0_hz).astype(np.int64)
    sid = np.array([0], dtype=np.int64)

    # The injected noise. Site A is (1, inter, T); site B is sized by SineGen.
    noise_zp = rng.standard_normal((1, inter, T)).astype(np.float32)
    noise_sine = rng.standard_normal((1, T * upp, 1)).astype(np.float32)

    caps = {}
    def _hook(name):
        def f(_m,_i,out):
            o = out[0] if isinstance(out,tuple) else out
            caps[name] = o.detach().numpy()
        return f
    # PER-STAGE intermediates for crispasr-diff (HARD RULE #2: start at the
    # earliest layer, first divergence = the bug). Endpoints alone are useless
    # for bisecting -- that is exactly what stalled the first enc_p graph.
    net.enc_p.emb_phone.register_forward_hook(_hook("encp_emb_phone"))
    net.enc_p.lrelu.register_forward_hook(_hook("encp_lrelu"))
    for _l in range(m["n_layers"]):
        net.enc_p.encoder.attn_layers[_l].register_forward_hook(_hook(f"encp_L{_l}_attn"))
        net.enc_p.encoder.norm_layers_1[_l].register_forward_hook(_hook(f"encp_L{_l}_norm1"))
        net.enc_p.encoder.ffn_layers[_l].register_forward_hook(_hook(f"encp_L{_l}_ffn"))
        net.enc_p.encoder.norm_layers_2[_l].register_forward_hook(_hook(f"encp_L{_l}_norm2"))
    net.enc_p.proj.register_forward_hook(_hook("encp_proj"))
    # flow: register_forward_hook does NOT fire here. ResidualCouplingBlock's
    # reverse pass calls `flow.forward(...)` DIRECTLY (models.py:125), and
    # calling .forward() bypasses nn.Module.__call__, which is what dispatches
    # hooks. Wrap the bound method instead.
    def _wrap_flow(mod, name):
        orig = mod.forward
        def wrapped(*a, **k):
            out = orig(*a, **k)
            o_ = out[0] if isinstance(out, tuple) else out
            caps[name] = o_.detach().numpy()
            return out
        mod.forward = wrapped
    for _fi in range(0, 8, 2):
        _wrap_flow(net.flow.flows[_fi], f"flow_c{_fi // 2}")

    net.dec.m_source.register_forward_hook(_hook("har_source"))
    net.dec.m_source.l_sin_gen.register_forward_hook(_hook("sine_raw"))
    net.dec.conv_pre.register_forward_hook(_hook("conv_pre"))
    for _i in range(len(m["upsample_rates"])):
        net.dec.ups[_i].register_forward_hook(_hook(f"ups{_i}"))
        net.dec.noise_convs[_i].register_forward_hook(_hook(f"nc{_i}"))

    def run():
        with make_injector([noise_zp.copy(), noise_sine.copy()]) as inj, torch.no_grad():
            o, x_mask, (z, z_p, m_p, logs_p) = net.infer(
                torch.from_numpy(phone), torch.tensor([T]),
                torch.from_numpy(pitch_coarse), torch.from_numpy(f0_hz),
                torch.from_numpy(sid),
            )
        return o.numpy(), dict(z=z.numpy(), z_p=z_p.numpy(), m_p=m_p.numpy(), logs_p=logs_p.numpy()), inj.log

    a_ref, sa, log1 = run()
    b, sb, log2 = run()
    a = a_ref

    print("  RNG draws intercepted (site, shape), in call order:")
    for site, shape in log1:
        print(f"    {site:22} {shape}")

    same = np.array_equal(a, b)
    print(f"\nDETERMINISM CHECK: two runs with identical injected noise -> "
          f"{'BIT-IDENTICAL' if same else 'DIFFER'}  max_abs={np.abs(a-b).max():.3e}")
    if not same:
        sys.exit(
            "FAIL: injection did not make inference deterministic. Some RNG site is "
            "still unpatched — find it before building anything on this."
        )
    zeroed = [s for s, _ in log1 if s.endswith(":ZEROED")]
    if zeroed:
        print(f"NOTE: {len(zeroed)} draw(s) fell back to zeros (no buffer of that shape supplied): {set(zeroed)}")

    # ---- numpy spec vs torch, for enc_p ----
    from gguf import GGUFReader
    G = {t.name: np.array(t.data, dtype=np.float32).reshape([int(d) for d in reversed(t.shape)])
         for t in GGUFReader(gguf_path).tensors}
    mp_np, logs_np = enc_p_numpy(
        G, phone[0], pitch_coarse[0], m["hidden_channels"], m["n_heads"], m["n_layers"],
        window=(G["enc_p.encoder.attn_layers.0.emb_rel_k"].shape[1] - 1) // 2,
        out_channels=inter,
    )
    print("\nNUMPY SPEC vs TORCH (enc_p):")
    ok = True
    for name, mine, ref in (("m_p", mp_np, sa["m_p"][0]), ("logs_p", logs_np, sa["logs_p"][0])):
        a, b = mine.ravel(), ref.ravel().astype(np.float64)
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        mad = float(np.abs(a - b).max())
        good = cos > 0.99999
        ok &= good
        print(f"  {name:8} {'PASS' if good else 'FAIL'} cos={cos:.8f} max_abs={mad:.3e} "
              f"|mine|={np.linalg.norm(a):.4f} |ref|={np.linalg.norm(b):.4f}")
    # ---- flow (reverse) ----
    g_emb = G["emb_g.weight"][0][:, None].astype(np.float64)  # sid 0
    # (5, 1, 3) = kernel_size, dilation_rate, n_layers -- HARDCODED in
    # models.py:624 (ResidualCouplingBlock(inter, hidden, 5, 1, 3, ...)), not
    # config-derived, so these hold for every checkpoint.
    z_np = flow_numpy(G, sa["z_p"][0], g_emb, n_flows=4, hidden=m["hidden_channels"],
                      n_layers=3, kernel_size=5, dilation_rate=1)
    a, b = z_np.ravel(), sa["z"][0].ravel().astype(np.float64)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    good = cos > 0.99999
    ok &= good
    print(f"NUMPY SPEC vs TORCH (flow, reverse):\n  {'z':8} {'PASS' if good else 'FAIL'} "
          f"cos={cos:.8f} max_abs={np.abs(a-b).max():.3e} |mine|={np.linalg.norm(a):.4f} |ref|={np.linalg.norm(b):.4f}")

    # ---- bisect the source module first ----
    har_np, uv_np = sine_gen_numpy(f0_hz[0], upp, sr, noise_sine)
    for nm, mine in (("sine_raw", har_np),):
        ref = caps[nm][0].ravel() if nm in caps else None
        if ref is not None:
            nn_ = min(len(mine), len(ref)); u,v = mine[:nn_], ref[:nn_].astype(np.float64)
            c = float(u@v/(np.linalg.norm(u)*np.linalg.norm(v)))
            print(f"  BISECT {nm:12} cos={c:.6f} |mine|={np.linalg.norm(u):.4f} |ref|={np.linalg.norm(v):.4f}")

    # ---- dec (NSF vocoder) ----
    dtaps = {}
    audio_np = dec_numpy(G, sa["z"][0], f0_hz[0], g_emb, cfg, noise_sine, taps=dtaps)
    for nm in ["conv_pre"] + [f"{p_}{i}" for i in range(len(m["upsample_rates"])) for p_ in ("ups","nc")]:
        if nm in dtaps and nm in caps:
            u = dtaps[nm].ravel(); v = caps[nm][0].ravel().astype(np.float64)
            nn_ = min(len(u), len(v))
            c = float(u[:nn_] @ v[:nn_] / (np.linalg.norm(u[:nn_]) * np.linalg.norm(v[:nn_])))
            flag = "" if c > 0.99999 else "   <-- FIRST DIVERGENCE" if c < 0.99999 else ""
            print(f"  BISECT {nm:10} cos={c:.6f} len mine={len(u)} ref={len(v)}{flag}")
    # NB: scratch names here must NOT be `a` — `a`/`a_ref` hold the reference
    # audio used by the stage dump below, and shadowing it silently reduced
    # "output_audio" to a single float (making its diff comparison vacuous).
    _u, _v = audio_np.ravel(), a_ref.ravel().astype(np.float64)
    n = min(len(_u), len(_v))
    cos = float(_u[:n] @ _v[:n] / (np.linalg.norm(_u[:n]) * np.linalg.norm(_v[:n])))
    good = cos > 0.999
    ok &= good
    print(f"NUMPY SPEC vs TORCH (dec, audio):\n  {'audio':8} {'PASS' if good else 'FAIL'} "
          f"cos={cos:.8f} max_abs={np.abs(_u[:n]-_v[:n]).max():.3e} "
          f"|mine|={np.linalg.norm(_u[:n]):.4f} |ref|={np.linalg.norm(_v[:n]):.4f} n={n}")

    if not ok:
        sys.exit("FAIL: a numpy spec does not match torch — fix the spec before any ggml.")

    if len(sys.argv) > 5:
        import gguf as _g
        stages = {"input_phone": phone[0], "input_f0": f0_hz, "input_pitch": pitch_coarse.astype(np.float32),
                  "noise_zp": noise_zp, "noise_sine": noise_sine,
                  "m_p": sa["m_p"][0], "logs_p": sa["logs_p"][0], "z_p": sa["z_p"][0], "z": sa["z"][0],
                  "output_audio": a_ref[0]}
        # every captured enc_p sublayer, squeezed of the batch dim
        for _k, _v in ENCP_TAPS.items():
            stages[_k] = _v
        # dec stages: the source module and every upsample/noise-conv stage.
        for _k in ("har_source", "sine_raw", "conv_pre"):
            if _k in caps:
                _v = caps[_k]
                stages["dec_" + _k] = _v[0] if _v.ndim >= 2 and _v.shape[0] == 1 else _v
        for _i in range(len(m["upsample_rates"])):
            for _pfx in ("ups", "nc"):
                _k = f"{_pfx}{_i}"
                if _k in caps:
                    _v = caps[_k]
                    stages["dec_" + _k] = _v[0] if _v.ndim >= 2 and _v.shape[0] == 1 else _v
        for _k, _v in caps.items():
            if _k.startswith("flow_"):
                stages[_k] = _v[0] if _v.ndim >= 2 and _v.shape[0] == 1 else _v
        for _k, _v in caps.items():
            if _k.startswith("encp_"):
                stages[_k] = _v[0] if _v.ndim >= 2 and _v.shape[0] == 1 else _v
        w = _g.GGUFWriter(sys.argv[5], "rvc-ref")
        for k, v in stages.items():
            w.add_tensor(k, np.ascontiguousarray(v, dtype=np.float32))
        w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
        print(f"wrote reference dump {sys.argv[5]}: {len(stages)} stages "
              f"(inputs + BOTH noise buffers + latents + audio)")
    print("PASS")


# ---------------------------------------------------------------------------
# numpy reimplementation — enc_p (TextEncoder)
#
# Traps this encodes, each a silent accuracy bug if assumed:
#   * LeakyReLU slope is 0.1, NOT torch's 0.01 default (models.py:38).
#   * x is scaled by sqrt(hidden_channels) BEFORE the lrelu (models.py:62-63).
#   * Residuals are POST-norm: x = norm(x + sublayer(x)), not pre-norm.
#   * Attention is RELATIVE-position (window 10), not absolute PE.
#   * FFN convs use SAME padding and a plain ReLU (activation is not "gelu"),
#     and the output is re-masked.
#   * LayerNorm is over the CHANNEL dim (modules.py:29-32 transposes first).
# ---------------------------------------------------------------------------

def _layer_norm(x, gamma, beta, eps=1e-5):
    """x: (C, T) -> normalise over C per time step (their LayerNorm transposes)."""
    xt = x.T                                    # (T, C)
    mu = xt.mean(-1, keepdims=True)
    var = xt.var(-1, keepdims=True)
    y = (xt - mu) / np.sqrt(var + eps) * gamma + beta
    return y.T


def _conv1d(x, w, b, pad):
    """x: (Cin, T), w: (Cout, Cin, K) -> (Cout, T) with SAME zero padding."""
    Cin, T = x.shape
    Cout, _, K = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad)))
    out = np.zeros((Cout, T), dtype=np.float64)
    for k in range(K):
        out += w[:, :, k] @ xp[:, k : k + T]
    return out + b[:, None]


def _get_relative_embeddings(emb, T, window):
    """emb: (1, 2w+1, d) -> (2T-1, d), padded/sliced exactly as upstream."""
    pad_len = max(T - (window + 1), 0)
    start = max((window + 1) - T, 0)
    e = emb[0]
    if pad_len > 0:
        e = np.pad(e, ((pad_len, pad_len), (0, 0)))
    return e[start : start + 2 * T - 1]


def _rel_to_abs(x):
    """x: (H, T, 2T-1) -> (H, T, T). The skew from relative to absolute indexing."""
    H, T, _ = x.shape
    x = np.pad(x, ((0, 0), (0, 0), (0, 1)))          # (H, T, 2T)
    flat = x.reshape(H, T * 2 * T)
    flat = np.pad(flat, ((0, 0), (0, T - 1)))        # (H, T*2T + T-1)
    return flat.reshape(H, T + 1, 2 * T - 1)[:, :T, T - 1 :]


ENCP_TAPS = {}


def enc_p_numpy(G, phone, pitch, hidden, n_heads, n_layers, window, out_channels):
    """phone: (T, content_dim), pitch: (T,) int -> (m_p, logs_p), each (out, T)."""
    W = lambda k: G[k].astype(np.float64)
    x = phone.astype(np.float64) @ W("enc_p.emb_phone.weight").T + W("enc_p.emb_phone.bias")
    x = x + W("enc_p.emb_pitch.weight")[pitch]          # embedding lookup
    x = x * np.sqrt(hidden)                             # BEFORE the lrelu
    x = np.where(x < 0, 0.1 * x, x)                     # LeakyReLU(0.1), not 0.01
    x = x.T                                             # (C, T)

    hd = hidden // n_heads
    for i in range(n_layers):
        p = f"enc_p.encoder.attn_layers.{i}."
        q = _conv1d(x, W(p + "conv_q.weight"), W(p + "conv_q.bias"), 0)
        k = _conv1d(x, W(p + "conv_k.weight"), W(p + "conv_k.bias"), 0)
        v = _conv1d(x, W(p + "conv_v.weight"), W(p + "conv_v.bias"), 0)
        T = x.shape[1]
        qh = q.reshape(n_heads, hd, T).transpose(0, 2, 1)   # (H, T, hd)
        kh = k.reshape(n_heads, hd, T).transpose(0, 2, 1)
        vh = v.reshape(n_heads, hd, T).transpose(0, 2, 1)
        scores = (qh / np.sqrt(hd)) @ kh.transpose(0, 2, 1)
        if i == 0:
            ENCP_TAPS["encp_L0_q"] = q.copy()
            ENCP_TAPS["encp_L0_scores_norel"] = scores.copy()
        rel_k = _get_relative_embeddings(W(p + "emb_rel_k"), T, window)
        rl = (qh / np.sqrt(hd)) @ rel_k.T
        if i == 0:
            ENCP_TAPS["encp_L0_rl"] = rl.copy()
        scores = scores + _rel_to_abs(rl)
        if i == 0:
            ENCP_TAPS["encp_L0_scores"] = scores.copy()
        e = np.exp(scores - scores.max(-1, keepdims=True))
        attn = e / e.sum(-1, keepdims=True)
        if i == 0:
            ENCP_TAPS["encp_L0_attn_w"] = attn.copy()
        out = attn @ vh                                     # (H, T, hd)
        if i == 0:
            ENCP_TAPS["encp_L0_v"] = vh.copy()      # (H, T, hd) — hd fastest
            ENCP_TAPS["encp_L0_agg"] = out.copy()   # (H, T, hd) — pre rel_v
        # relative VALUES: upstream adds them via the inverse skew
        rel_v = _get_relative_embeddings(W(p + "emb_rel_v"), T, window)
        out = out + _abs_to_rel(attn) @ rel_v
        out = out.transpose(0, 2, 1).reshape(hidden, T)
        if i == 0:
            ENCP_TAPS["encp_L0_ctx"] = out.copy()   # pre conv_o
        y = _conv1d(out, W(p + "conv_o.weight"), W(p + "conv_o.bias"), 0)
        n1 = f"enc_p.encoder.norm_layers_1.{i}."
        x = _layer_norm(x + y, W(n1 + "gamma"), W(n1 + "beta"))     # POST-norm

        f = f"enc_p.encoder.ffn_layers.{i}."
        w1 = W(f + "conv_1.weight")
        h = _conv1d(x, w1, W(f + "conv_1.bias"), (w1.shape[2] - 1) // 2)
        h = np.maximum(h, 0.0)                                       # plain ReLU
        w2 = W(f + "conv_2.weight")
        y = _conv1d(h, w2, W(f + "conv_2.bias"), (w2.shape[2] - 1) // 2)
        n2 = f"enc_p.encoder.norm_layers_2.{i}."
        x = _layer_norm(x + y, W(n2 + "gamma"), W(n2 + "beta"))

    stats = _conv1d(x, W("enc_p.proj.weight"), W("enc_p.proj.bias"), 0)
    return stats[:out_channels], stats[out_channels:]


# ---------------------------------------------------------------------------
# numpy reimplementation — flow (ResidualCouplingBlock, REVERSE pass)
#
# Traps:
#   * mean_only=True (models.py), so `logs` is ZERO and the coupling is purely
#     ADDITIVE. The reverse is x1 = (x1 - m), not (x1 - m) * exp(-logs).
#   * `flows` interleaves [Coupling, Flip] x 4; the reverse pass walks the whole
#     list backwards, so Flip comes FIRST.
#   * Flip reverses the CHANNEL axis (torch.flip(x, [1])).
#   * The WaveNet is gated: tanh(first half) * sigmoid(second half) of
#     (x_in + g_l), with the speaker conditioning sliced per layer.
#   * Dilated convs with padding = (k*d - d)/2, i.e. SAME for odd k.
# ---------------------------------------------------------------------------

def _wn_numpy(G, prefix, x, g, hidden, n_layers, kernel_size, dilation_rate):
    """WaveNet residual stack. x: (hidden, T), g: (gin, 1) speaker embedding."""
    W = lambda k: G[k].astype(np.float64)
    T = x.shape[1]
    output = np.zeros_like(x)
    # cond_layer projects g once to 2*hidden*n_layers, then each layer slices it.
    gc = W(prefix + "cond_layer.weight")[:, :, 0] @ g[:, 0] + W(prefix + "cond_layer.bias")
    for i in range(n_layers):
        d = dilation_rate ** i
        pad = int((kernel_size * d - d) / 2)
        w = W(prefix + f"in_layers.{i}.weight")
        b = W(prefix + f"in_layers.{i}.bias")
        xp = np.pad(x, ((0, 0), (pad, pad)))
        x_in = np.zeros((w.shape[0], T), dtype=np.float64)
        for k in range(w.shape[2]):
            x_in += w[:, :, k] @ xp[:, k * d : k * d + T]
        x_in += b[:, None]
        g_l = gc[i * 2 * hidden : (i + 1) * 2 * hidden][:, None]
        in_act = x_in + g_l
        acts = np.tanh(in_act[:hidden]) * (1.0 / (1.0 + np.exp(-in_act[hidden:])))
        rw = W(prefix + f"res_skip_layers.{i}.weight")[:, :, 0]
        rb = W(prefix + f"res_skip_layers.{i}.bias")
        rs = rw @ acts + rb[:, None]
        if i < n_layers - 1:
            x = x + rs[:hidden]
            output = output + rs[hidden:]
        else:
            output = output + rs
    return output


def flow_numpy(G, z_p, g, n_flows, hidden, n_layers, kernel_size, dilation_rate):
    """Reverse pass. z_p: (C, T) -> z: (C, T)."""
    W = lambda k: G[k].astype(np.float64)
    x = z_p.astype(np.float64)
    half = x.shape[0] // 2
    # flows = [Coupling, Flip] * n_flows; reverse traversal hits Flip first.
    for idx in range(n_flows - 1, -1, -1):
        x = x[::-1]                                     # Flip: reverse channels
        p = f"flow.flows.{idx * 2}."
        x0, x1 = x[:half], x[half:]
        h = W(p + "pre.weight")[:, :, 0] @ x0 + W(p + "pre.bias")[:, None]
        h = _wn_numpy(G, p + "enc.", h, g, hidden, n_layers, kernel_size, dilation_rate)
        m = W(p + "post.weight")[:, :, 0] @ h + W(p + "post.bias")[:, None]
        x1 = x1 - m                                     # mean_only => no exp(-logs)
        x = np.concatenate([x0, x1], axis=0)
    return x


# ---------------------------------------------------------------------------
# numpy reimplementation — dec (GeneratorNSF)
#
# Traps:
#   * TWO different LeakyReLU slopes in ONE function: the per-stage and ResBlock
#     ones use LRELU_SLOPE = 0.1, but the FINAL pre-conv_post call is a bare
#     F.leaky_relu(x) — torch's 0.01 default (models.py:529).
#   * SineGen accumulates phase with cumsum at the FRAME rate, multiplies by
#     upp, then linear-interpolates to the output rate. Accumulate in float64:
#     the running sum grows without bound while only its fraction matters.
#   * ConvTranspose1d stride=u, padding=(k-u)//2 (models.py:452-459).
#   * Each stage sums num_kernels ResBlocks and DIVIDES by num_kernels.
#   * The source signal is added AFTER the transpose-conv, via noise_convs[i]
#     whose stride is prod(rates[i+1:]).
# ---------------------------------------------------------------------------

def sine_gen_numpy(f0_hz, upp, sr, noise, sine_amp=0.1, noise_std=0.003, voiced_th=0.0):
    """f0_hz: (T,) -> (sine source (T*upp,), uv (T*upp,)). Mirrors SineGen.forward.

    The phase logic is subtle and an approximation does NOT work (a plausible
    rewrite scored cos -0.04 against torch — right amplitude, wrong phase):

      1. rad = (f0/sr) % 1                      at FRAME rate
      2. tmp = cumsum(rad) * upp                still frame rate
      3. tmp -> LINEAR interpolate to out rate, align_corners=True
      4. rad -> NEAREST interpolate to out rate
      5. tmp %= 1; wrap points are where diff(tmp) < 0
      6. phase = cumsum(rad_up + shift), shift = -1 at each wrap point
      7. sine = sin(phase * 2*pi)

    So the linear-interpolated cumsum is used ONLY to LOCATE the wraps; the
    phase itself accumulates over the NEAREST-upsampled per-frame values. Note
    the two interpolations use DIFFERENT modes, and align_corners=True changes
    the sample mapping.
    """
    T = f0_hz.shape[0]
    f0 = f0_hz.astype(np.float64)
    rad = (f0 / sr) % 1.0                                   # (T,), dim == 1
    tmp = np.cumsum(rad) * upp                              # float64: f32 drifts

    N = T * upp
    # linear, align_corners=True: x maps 0..T-1 across 0..N-1
    if T > 1:
        xs = np.arange(N, dtype=np.float64) * (T - 1) / (N - 1)
        tmp_up = np.interp(xs, np.arange(T, dtype=np.float64), tmp)
    else:
        tmp_up = np.repeat(tmp, upp)
    rad_up = np.repeat(rad, upp)                            # nearest

    tmp_up = tmp_up % 1.0
    shift = np.zeros(N, dtype=np.float64)
    shift[1:] = np.where(np.diff(tmp_up) < 0, -1.0, 0.0)

    sines = np.sin(np.cumsum(rad_up + shift) * 2 * np.pi) * sine_amp

    uv = np.repeat((f0 > voiced_th).astype(np.float64), upp)   # nearest
    noise_amp = uv * noise_std + (1 - uv) * sine_amp / 3
    sines = sines * uv + noise_amp * noise.astype(np.float64).ravel()
    return sines, uv


def _conv1d_d(x, w, b, pad, dil=1):
    Cin, T = x.shape
    Cout, _, K = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad)))
    out = np.zeros((Cout, T), dtype=np.float64)
    for k in range(K):
        out += w[:, :, k] @ xp[:, k * dil : k * dil + T]
    return out + b[:, None]


def _conv1d_stride(x, w, b, stride, pad, out_len):
    """Strided Conv1d: w is (Cout, Cin, K)."""
    Cin, T = x.shape
    Cout, _, K = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad)))
    out = np.zeros((Cout, out_len), dtype=np.float64)
    for o in range(out_len):
        seg = xp[:, o * stride : o * stride + K]
        if seg.shape[1] < K:
            seg = np.pad(seg, ((0, 0), (0, K - seg.shape[1])))
        out[:, o] = np.tensordot(w, seg, axes=([1, 2], [0, 1]))
    return out + b[:, None]


def _conv_transpose1d(x, w, b, stride, pad):
    """w: (Cin, Cout, K). Matches torch ConvTranspose1d with output_padding=0."""
    Cin, T = x.shape
    _, Cout, K = w.shape
    full = np.zeros((Cout, (T - 1) * stride + K), dtype=np.float64)
    for t in range(T):
        full[:, t * stride : t * stride + K] += np.tensordot(x[:, t], w, axes=([0], [0]))
    out = full[:, pad : full.shape[1] - pad] if pad else full
    return out + b[:, None]


def _lrelu(x, slope):
    return np.where(x < 0, slope * x, x)


def dec_numpy(G, z, f0_hz, g, cfg, noise_sine, taps=None):
    W = lambda k: G[k].astype(np.float64)
    m = cfg["model"]
    rates, kers = m["upsample_rates"], m["upsample_kernel_sizes"]
    rk, rd = m["resblock_kernel_sizes"], m["resblock_dilation_sizes"]
    upp = int(np.prod(rates))
    num_kernels = len(rk)
    LRELU = 0.1

    har, _uv = sine_gen_numpy(f0_hz, upp, cfg["data"]["sampling_rate"], noise_sine)
    # SourceModuleHnNSF: tanh(linear(sine)), dim 1 -> 1
    har = np.tanh(har[:, None] @ W("dec.m_source.l_linear.weight").T
                  + W("dec.m_source.l_linear.bias"))[:, 0][None, :]   # (1, T*upp)

    x = _conv1d_d(z.astype(np.float64), W("dec.conv_pre.weight"), W("dec.conv_pre.bias"), 3)
    if taps is not None:
        taps["conv_pre"] = x.copy()
    # cond is nn.Conv1d(gin, upsample_initial_channel, 1) -- it HAS a bias.
    # Omitting it is a constant per-channel offset: structurally invisible,
    # cost cos 0.9999 at ups0 and 0.998 on the final audio.
    x = x + (W("dec.cond.weight")[:, :, 0] @ g[:, 0] + W("dec.cond.bias"))[:, None]

    for i, (u, k) in enumerate(zip(rates, kers)):
        x = _lrelu(x, LRELU)
        x = _conv_transpose1d(x, W(f"dec.ups.{i}.weight"), W(f"dec.ups.{i}.bias"), u, (k - u) // 2)
        if taps is not None:
            taps[f"ups{i}"] = x.copy()
        nw = W(f"dec.noise_convs.{i}.weight")
        stride = int(np.prod(rates[i + 1:])) if i + 1 < len(rates) else 1
        pad = (nw.shape[2] - stride) // 2 if stride > 1 else 0
        xs_src = _conv1d_stride(har, nw, W(f"dec.noise_convs.{i}.bias"), stride, pad, x.shape[1])
        if taps is not None:
            taps[f"nc{i}"] = xs_src.copy()
        x = x + xs_src
        acc = None
        for j in range(num_kernels):
            idx = i * num_kernels + j
            h = x
            for d in rd[j]:
                for (c1, c2) in ((0, 0),):
                    pass
            # ResBlock1: 3 (convs1, convs2) pairs, dilations rd[j] on convs1
            for di, dil in enumerate(rd[j]):
                xt = _lrelu(h, LRELU)
                w1 = W(f"dec.resblocks.{idx}.convs1.{di}.weight")
                xt = _conv1d_d(xt, w1, W(f"dec.resblocks.{idx}.convs1.{di}.bias"),
                               (w1.shape[2] * dil - dil) // 2, dil)
                xt = _lrelu(xt, LRELU)
                w2 = W(f"dec.resblocks.{idx}.convs2.{di}.weight")
                xt = _conv1d_d(xt, w2, W(f"dec.resblocks.{idx}.convs2.{di}.bias"),
                               (w2.shape[2] - 1) // 2, 1)
                h = xt + h
            acc = h if acc is None else acc + h
        x = acc / num_kernels

    x = _lrelu(x, 0.01)          # BARE F.leaky_relu -> torch default 0.01, NOT 0.1
    # conv_post is bias=False (models.py:484)
    cpw = W("dec.conv_post.weight")
    x = _conv1d_d(x, cpw, np.zeros(cpw.shape[0]), 3)
    return np.tanh(x)


def _abs_to_rel(x):
    """(H, T, T) -> (H, T, 2T-1); inverse of _rel_to_abs, for relative values."""
    H, T, _ = x.shape
    x = np.pad(x, ((0, 0), (0, 0), (0, T - 1)))
    flat = x.reshape(H, T * (2 * T - 1))
    flat = np.pad(flat, ((0, 0), (T, 0)))
    return flat.reshape(H, T, 2 * T)[:, :, 1:]


def coarse_pitch(f0):
    """f0 (Hz) -> coarse 1..255, exactly as pipeline.py:73-137."""
    f0_min, f0_max = 50.0, 1100.0
    mel_min = 1127 * np.log(1 + f0_min / 700)
    mel_max = 1127 * np.log(1 + f0_max / 700)
    mel = 1127 * np.log(1 + f0 / 700)
    mel = np.where(mel > 0, (mel - mel_min) * 254 / (mel_max - mel_min) + 1, mel)
    mel = np.where(mel <= 1, 1, mel)
    mel = np.where(mel > 255, 255, mel)
    return np.rint(mel)


if __name__ == "__main__":
    main()
