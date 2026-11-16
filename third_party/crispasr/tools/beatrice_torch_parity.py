#!/usr/bin/env python3
"""Dump Beatrice per-stage reference intermediates to a -ref.gguf.

    python tools/beatrice_torch_parity.py \
        --component pitch_estimator --model 104_3_checkpoint_00300000.pt \
        --audio samples/jfk.wav --output beatrice-pitch-ref.gguf \
        --trainer-path <dir containing beatrice_trainer>

This is the HARD RULE #2 artifact: the ggml port is validated stage by stage
against these, earliest layer first, NOT by comparing final output. See
docs/music-transcription/BEATRICE_BLUEPRINT.md.

The forward pass is RE-IMPLEMENTED here step by step rather than captured with
hooks, for two reasons:

  1. It is the executable spec. If my reading of the architecture is wrong, the
     final-output assertion below fails and I find out immediately -- before any
     C++ is written.
  2. register_forward_hook is a known trap in this codebase: it does not fire
     when a module's .forward() is called directly, and during the RVC port it
     silently produced a harness that compared NOTHING while reporting
     "0 FAILED". Beatrice compounds that -- its VectorQuantizer is itself
     installed as a forward hook, so hook-based capture would interact with the
     model's own hooks.

The reimplementation is checked against the module's real forward() at the end;
if they disagree the tool refuses to write a reference file, because a wrong
reference is worse than none -- it makes a broken port look correct.

LAYOUT. Torch stores [batch, channels, time] with TIME fastest. Tensors are
dumped in exactly that memory order, which lands in ggml as ne = [time,
channels] -- the layout ggml_conv_1d expects. Do NOT transpose on the way in or
out. Three separate bugs during the RVC port came from transposing here: the
per-stage cosines went to ~0 on graphs that were in fact correct.
"""

import argparse
import sys

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    import torchaudio
except ImportError:
    sys.exit("error: pip install torch torchaudio")
try:
    from gguf import GGUFWriter
except ImportError:
    sys.exit("error: pip install gguf")


class Dump:
    def __init__(self):
        self.stages = {}

    def __call__(self, name, t):
        a = t.detach().cpu().float().numpy()
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]                      # drop batch; keep [channels, time]
        self.stages[name] = np.ascontiguousarray(a)
        return t


def cos(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def spec_pitch_estimator(m, wav, d):
    """Step-by-step PitchEstimator.forward, dumping every boundary."""
    d("input_wav", wav)

    # --- DSP front end (extract_pitch_features)
    instfreq, corr_diff, energy = m_extract(m, wav, d)

    # --- two embedding branches. GELU is the TANH approximation, not erf.
    x_if = F.gelu(m.instfreq_embed_0(instfreq), approximate="tanh")
    d("instfreq_embed_0_gelu", x_if)
    x_if = m.instfreq_embed_1(x_if)
    d("instfreq_embed_1", x_if)

    x_c = F.gelu(m.corr_embed_0(corr_diff), approximate="tanh")
    d("corr_embed_0_gelu", x_c)
    x_c = m.corr_embed_1(x_c)
    d("corr_embed_1", x_c)

    x = F.gelu(x_if + x_c, approximate="tanh")
    d("branch_sum_gelu", x)

    # --- ConvNeXtStack
    bb = m.backbone
    x = bb.embed(x)
    d("backbone_embed", x)
    # stack-level LayerNorm KEEPS its affine (merge_weights does not fold it)
    x = bb.norm(x.transpose(1, 2)).transpose(1, 2)
    d("backbone_norm", x)

    for i, blk in enumerate(bb.convnext):
        identity = x
        h = blk.dwconv(x)                       # depthwise, strictly causal
        d(f"block{i}_dwconv", h)
        h = h.transpose(1, 2)
        h = blk.norm(h)                         # affine folded away -> normalise only
        h = blk.pwconv1(h)
        h = F.gelu(h, approximate="tanh")
        h = blk.pwconv2(h)
        # gamma / pre_scale / post_scale / post_scale_weight are all identically
        # 1.0 post-merge and are deliberately NOT applied here.
        h = h.transpose(1, 2)
        x = h + identity
        d(f"block{i}_out", x)

    x = bb.final_layer_norm(x.transpose(1, 2)).transpose(1, 2)
    d("backbone_final_norm", x)

    logits = m.head(x)
    d("logits", logits)
    d("energy", energy)

    quantized = m.sample_pitch(logits.clone())
    d("quantized_pitch", quantized.float())
    # The 3 pitch features: [unvoiced_proba, half_pitch_proba, double_pitch_proba].
    # unvoiced_proba is the ONLY voicing signal the model produces -- the
    # quantised bin never returns 0, so without this a consumer cannot tell
    # speech from silence. ConverterNetwork prepends energy to make the 4
    # channels embed_pitch_features consumes.
    _, features = m.sample_pitch(logits.clone(), return_features=True)
    d("pitch_features", features)
    return logits, energy


def spec_phone_extractor(m, wav, d):
    """Step-by-step PhoneExtractor.forward, dumping every boundary.

    NOTE the feature_projection LayerNorm is applied EXPLICITLY here and is NOT
    folded into backbone.embed. Upstream's PhoneExtractor.merge_weights() folds
    it, but that fold is only valid where every kernel tap sees the LayerNorm
    bias -- the zero-padded sequence edges do not, and with use_mha=True
    attention spreads the edge error across the whole sequence (rel 1.7e-02
    measured). The converter skips that fold for the same reason, so reference
    and port agree and both are exact.
    """
    d("input_wav", wav)

    # --- FeatureExtractor: 6 strided convs, /160 overall -> 100 Hz
    fe = m.feature_extractor
    x = F.pad(wav, (40, 40))
    for i in range(6):
        conv = getattr(fe, f"conv{i}")
        x = F.gelu(conv(x), approximate="tanh")
        d(f"fe_conv{i}", x)

    # --- FeatureProjection: LayerNorm over channels (Dropout is eval-inert)
    x = m.feature_projection.norm(x.transpose(1, 2)).transpose(1, 2)
    d("feature_projection", x)

    # --- ConvNeXtStack with MHA
    bb = m.backbone
    x = bb.embed(x)
    d("backbone_embed", x)
    x = bb.norm(x.transpose(1, 2)).transpose(1, 2)
    d("backbone_norm", x)

    # the stack zero-pads the TIME axis up to a multiple of 4 because the MHA
    # reshapes into 4 interleaved subsequences, and trims it back afterwards
    pad_length = -x.size(2) % 4
    if pad_length:
        x = F.pad(x, (0, pad_length))
    t40 = x.size(2) // 4
    attn_mask = torch.ones((t40, t40), dtype=torch.bool, device=x.device).triu(1)
    d("attn_pad_len", torch.tensor([[float(pad_length)]]))

    for i, blk in enumerate(bb.convnext):
        B_, C_, L_ = x.size()
        identity = x
        # interleave: frame t goes to subsequence t%4, position t//4. This is a
        # STRIDED split, not a chunked one -- each of the 4 sequences sees every
        # 4th frame, which is what makes a causal mask over t40 meaningful.
        h = x.view(B_, C_, L_ // 4, 4).permute(0, 3, 2, 1).reshape(B_ * 4, L_ // 4, C_)
        h = blk.attn_norm(h)
        h, _ = blk.mha(h, h, h, attn_mask=attn_mask, is_causal=True, need_weights=False)
        h = h.view(B_, 4, L_ // 4, C_).permute(0, 3, 2, 1).reshape(B_, C_, L_)
        # Dump the BRANCH OUTPUT as well as the post-residual sum. The sum is
        # residual-dominated and nearly blind to the attention: a broken
        # interleave still scores cos 0.99999986 there while wrecking 41 later
        # stages. The delta is what actually tests the MHA.
        d(f"pblock{i}_attn_delta", h)
        x = h + identity
        d(f"pblock{i}_attn", x)

        identity = x
        h = blk.dwconv(x)
        h = h.transpose(1, 2)
        h = blk.norm(h)
        h = blk.pwconv1(h)
        h = F.gelu(h, approximate="tanh")
        h = blk.pwconv2(h)
        h = h.transpose(1, 2)
        x = h + identity
        d(f"pblock{i}_out", x)

    if pad_length:
        x = x[:, :, :-pad_length]
    x = bb.final_layer_norm(x.transpose(1, 2)).transpose(1, 2)
    d("backbone_final_norm", x)

    phone = m.head(F.gelu(x, approximate="tanh"))
    d("phone", phone)
    return phone


def m_extract(m, wav, d):
    """extract_pitch_features, dumped. Constants are the function's defaults."""
    hop, win = 160, 560
    max_corr_period, corr_win_length, cutoff = 256, 304, 64
    assert max_corr_period + corr_win_length == win

    y = wav.squeeze(1)
    pad = (win - hop) // 2
    y = F.pad(y, (pad, pad))
    frames = y.unfold(-1, win, hop).transpose(-2, -1)
    d("dsp_frames", frames)

    spec = torch.fft.rfft(frames, n=win, dim=-2)[..., :cutoff, :]
    log_power = spec.abs().add(1e-5).log10()
    d("dsp_log_power_spec", log_power)

    delta = spec[..., :, 1:] * spec[..., :, :-1].conj()
    delta = delta / delta.abs().add(1e-5)
    delta = torch.cat([torch.zeros_like(delta[..., :, :1]), delta], dim=-1)
    instfreq = torch.cat([log_power, delta.real, delta.imag], dim=-2)
    d("dsp_instfreq_features", instfreq)

    # autocorrelation via FFT of the FLIPPED frames (that is what makes it a
    # correlation rather than a convolution -- there is no conj() here)
    flipped = frames.flip((-2,))
    a = torch.fft.rfft(flipped, n=win, dim=-2)
    b = torch.fft.rfft(frames[..., -corr_win_length:, :], n=win, dim=-2)
    corr = torch.fft.irfft(a * b, n=win, dim=-2)[..., corr_win_length:, :]
    d("dsp_corr", corr)

    energy_c = flipped.square().cumsum(-2)
    energy0 = energy_c[..., corr_win_length - 1 : corr_win_length, :]
    energy_w = energy_c[..., corr_win_length:, :] - energy_c[..., :-corr_win_length, :]
    corr_diff = (energy0 + energy_w) - corr * 2.0
    assert float(corr_diff.min()) >= -1e-3, float(corr_diff.min())
    corr_diff = corr_diff.clamp(min=0.0) * (2.0 / corr_win_length)
    corr_diff = corr_diff.sqrt()
    d("dsp_corr_diff", corr_diff)

    win_cos = torch.signal.windows.cosine(win, device=y.device)[..., None]
    energy = (frames * win_cos).square().sum(-2, keepdim=True)
    energy = energy.clamp(min=1e-3).log10() * 0.5
    d("dsp_energy", energy)
    return instfreq, corr_diff, energy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--component", default="pitch_estimator",
                    choices=["pitch_estimator", "phone_extractor"])
    ap.add_argument("--audio", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--trainer-path")
    ap.add_argument("--max-seconds", type=float, default=4.0)
    args = ap.parse_args()

    if args.trainer_path:
        sys.path.insert(0, args.trainer_path)
    import beatrice_trainer.__main__ as bt

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    if args.component not in ck:
        sys.exit(f"error: {args.model} has {list(ck.keys())}, not '{args.component}'")
    sd = ck[args.component]
    n_blocks = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("backbone.convnext."))
    if args.component == "pitch_estimator":
        m = bt.PitchEstimator(pitch_bins=sd["head.weight"].shape[0],
                              channels=sd["head.weight"].shape[1], n_blocks=n_blocks)
    else:
        m = bt.PhoneExtractor(
            phone_channels=sd["head.weight_v"].shape[0] if "head.weight_v" in sd else sd["head.weight"].shape[0],
            hidden_channels=sd["backbone.embed.weight"].shape[0], n_blocks=n_blocks)
    m.load_state_dict(sd)
    m.eval()
    # Fuse exactly as the converter does, so the reference describes the SHIPPED
    # weights. Dumping pre-merge intermediates would compare the port against a
    # model it is not running. The phone_extractor path deliberately skips the
    # feature_projection fold -- see convert-beatrice-to-gguf.py:fuse_model.
    if args.component == "pitch_estimator":
        m.merge_weights()
    else:
        m.remove_weight_norm()
        m.backbone.merge_weights()

    wav, sr = torchaudio.load(args.audio)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    n = int(args.max_seconds * 16000)
    wav = wav[:, :n][None]  # [1, 1, T]

    d = Dump()
    with torch.inference_mode():
        if args.component == "pitch_estimator":
            logits, energy = spec_pitch_estimator(m, wav, d)
            ref_logits, ref_energy = m(wav)
        else:
            # wav_length must be a multiple of 160 or the extractor warns and the
            # frame count is ambiguous; trim rather than pad so no invented audio
            # reaches the reference.
            n160 = (wav.shape[-1] // 160) * 160
            wav = wav[..., :n160]
            d.stages.clear()
            logits = spec_phone_extractor(m, wav, d)
            ref_logits = m(wav, return_stats=False)
            energy = ref_energy = torch.zeros(1)

    # THE GATE. If the step-by-step spec disagrees with the module's own
    # forward, this reference is wrong and must not be written.
    #
    # The bound is on RELATIVE MAX-ABS, not cosine. Both arms are torch running
    # the same weights, so the control is exactly bit-identical (max_abs 0.0) and
    # anything above f32 rounding is a real discrepancy. Cosine is far too blunt
    # here -- measured: swapping the tanh-approximate GELU for the exact erf one
    # (a genuine bug, blueprint detail 2) still scores cos=0.9999996, which sails
    # through any "cos > 0.999999" check while carrying max_abs 2.4e-02. That is
    # HARD RULE #2b in the concrete.
    # FINITENESS FIRST. Every NaN/Inf comparison is False, so a NaN spec sails
    # through any `rel > TOL` check and writes a reference file full of NaN while
    # exiting 0. Measured: dropping the 1e-5 in the delta_spec normalisation (a
    # divide-by-zero on silent frames) did exactly that.
    for name, a in d.stages.items():
        if not np.isfinite(a).all():
            n_bad = int((~np.isfinite(a)).sum())
            sys.exit(
                f"error: stage '{name}' contains {n_bad} non-finite values.\n"
                "       Refusing to write the reference. NOTE this cannot be caught "
                "by a tolerance check -- NaN compares False against everything."
            )

    c_log, c_en = cos(logits, ref_logits), cos(energy, ref_energy)
    peak = float(ref_logits.abs().max())
    mx = float((logits - ref_logits).abs().max())
    rel = mx / peak
    print(
        f"spec vs module forward(): logits cos={c_log:.10f} max_abs={mx:.3e} "
        f"rel={rel:.2e}  energy cos={c_en:.10f}"
    )
    TOL = 1e-6
    # `energy` is a pitch_estimator output only; the phone path has none, so do
    # not fabricate a check for it. An earlier version compared a zeros(1) stand-in
    # and failed a bit-identical spec while blaming relative max-abs -- a gate that
    # reports the wrong reason is worse than one that does not fire.
    fail_rel = rel > TOL
    fail_energy = args.component == "pitch_estimator" and c_en < 0.999999
    if fail_rel or fail_energy:
        why = []
        if fail_rel:
            why.append(f"relative max-abs {rel:.2e} > {TOL:.0e}")
        if fail_energy:
            why.append(f"energy cos {c_en:.8f} < 0.999999")
        sys.exit(
            "error: the step-by-step spec does NOT reproduce the module's forward() ("
            + "; ".join(why) + ").\n"
            "       Refusing to write a reference file -- a wrong reference makes a "
            "broken port look correct.\n"
            "       NOTE cosine can look perfect here; check max_abs, not cos."
        )

    w = GGUFWriter(str(args.output), "beatrice-ref")
    w.add_string("beatrice.ref.component", args.component)
    w.add_string("beatrice.ref.audio", args.audio)
    w.add_uint32("beatrice.ref.n_samples", int(wav.shape[-1]))
    w.add_uint32("beatrice.ref.n_stages", len(d.stages))
    for name, a in d.stages.items():
        w.add_tensor(f"ref.{name}", a)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    print(f"wrote {args.output}: {len(d.stages)} stages")
    for name, a in d.stages.items():
        print(f"    ref.{name:<28} {tuple(a.shape)}")


if __name__ == "__main__":
    main()
