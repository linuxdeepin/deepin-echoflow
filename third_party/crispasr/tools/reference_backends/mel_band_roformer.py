"""Mel-Band RoFormer reference backend — per-stage dumper for crispasr-diff (§248).

Blueprint: lucidrains/BS-RoFormer `mel_band_roformer.py` (MIT). Weights:
KimberleyJSN/melbandroformer (MIT). NOT Kim's inference repo — it ships no
license, so we never transcribe it; we use only its weights + config values.
Full op-by-op trace lives in docs/mel-band-roformer/PLAN.md.

Capture strategy: forward HOOKS on the real submodules (so we diff what the
model actually ran, never a re-implementation that could drift), plus two
things hooks can't see:

  * the STRUCTURAL band membership (`freq_indices` / `num_bands_per_freq`),
    which is read straight off the model buffers. This is stage 0 on purpose:
    the mel filterbank is used only to decide WHICH bins belong to a band
    (binary, weights discarded) and it carries two hand-tweaks
    (mel[0,0], mel[-1,-1] *= 0.25). An off-by-one here silently corrupts every
    downstream band, and it is far cheaper to catch as an integer mismatch
    than as a cosine drift 8 stages later.
  * the packed STFT, replicated from the traced forward (3 lines) so a
    front-end bug is separable from a band-gather bug.

Stages:
    freq_indices        — (N,)  gather index, band membership      [structural]
    num_bands_per_freq  — (F,)  overlap denominator                [structural]
    stft_packed         — (f*s, t, 2) packed STFT, freq-major/channel-fastest
    band_gathered       — (t, sum(2*f_b*s)) band_split INPUT (post gather+fold)
    band_split_out      — (t, n_bands, dim)
    layer{i}_time       — output of layer i's time transformer
    layer{i}_freq       — output of layer i's freq transformer
    mask_raw            — mask_estimator output, pre scatter/average
    output_vocals       — final separated stem (s, t)
"""

import numpy as np

DEFAULT_STAGES = [
    "freq_indices",
    "num_bands_per_freq",
    "input_audio",
    "stft_packed",
    "band_gathered",
    "band_split_out",
    "layer0_time", "layer0_freq",
    "layer1_time", "layer1_freq",
    "layer5_time", "layer5_freq",
    "mask_raw",
    "output_vocals",
]


def _np(t):
    import torch
    if isinstance(t, torch.Tensor):
        return t.detach().float().cpu().numpy()
    return np.asarray(t)


def dump(model_dir, audio, stages, max_new_tokens=None, **kwargs):
    """Run MelBandRoformer forward with hooks and return per-stage intermediates.

    Args:
        model_dir: path to a checkpoint (.ckpt) or a dir containing one, plus
                   the YAML config. Kim vocals: dim 384, depth 6, num_bands 60,
                   n_fft 2048, hop 441, stereo, num_stems 1.
        audio:     float32 numpy 16 kHz mono PCM (harness convention). Resampled
                   to the model's 44.1 kHz stereo internally.
        stages:    set of stage names to capture.
    Returns:
        {stage_name: numpy_array}
    """
    import torch
    from pathlib import Path

    out = {}
    want = (lambda s: True) if not stages else (lambda s: s in stages)

    model, cfg = _load(model_dir)
    model.eval()

    sr = int(cfg.get("sample_rate", 44100))
    n_fft = int(cfg.get("stft_n_fft", 2048))
    hop = int(cfg.get("stft_hop_length", 441))
    depth = len(model.layers)
    print(f"mel_band_roformer ref: dim={model.band_split.to_features[0][-1].out_features} "
          f"depth={depth} bands={len(model.band_split.dim_inputs)} "
          f"n_fft={n_fft} hop={hop} sr={sr} stems={len(model.mask_estimators)}")

    # ---- stage 0: structural band membership (integers, must match exactly) --
    if want("freq_indices"):
        out["freq_indices"] = _np(model.freq_indices).astype(np.float32)
    if want("num_bands_per_freq"):
        out["num_bands_per_freq"] = _np(model.num_bands_per_freq).astype(np.float32)

    # ---- input: 16 kHz mono -> model sr, stereo ----------------------------
    wav = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
    if sr != 16000:
        import torchaudio
        wav = torchaudio.functional.resample(wav, 16000, sr)
    stereo = bool(getattr(model, "stereo", True))
    wav = wav.expand(2 if stereo else 1, -1).unsqueeze(0).contiguous()  # (1, s, t)

    # ---- stage: the EXACT resampled model input (s, t) --------------------
    # Emitted so the C++ diff can start its STFT from the identical waveform and
    # not eat torchaudio-vs-our-resampler drift (gate input alignment BEFORE
    # trusting per-layer cos — the diff-harness rule).
    if want("input_audio"):
        out["input_audio"] = _np(wav[0])

    # ---- stage: packed STFT (replicates the traced forward exactly) --------
    if want("stft_packed"):
        from einops import rearrange
        b, s, t = wav.shape
        flat = wav.reshape(b * s, t)
        window = model.stft_window_fn(device=wav.device)
        spec = torch.stft(flat, **model.stft_kwargs, window=window, return_complex=True)
        spec = torch.view_as_real(spec)                       # (b*s, f, t, 2)
        spec = spec.reshape(b, s, *spec.shape[1:])
        # stereo folds INTO the freq axis, frequency-major / channel-fastest
        spec = rearrange(spec, "b s f t c -> b (f s) t c")
        out["stft_packed"] = _np(spec[0])

    # ---- hooks on the real submodules --------------------------------------
    handles = []

    def cap_io(name_in, name_out, mod):
        def hook(_m, inp, outp):
            if name_in and want(name_in) and len(inp):
                out[name_in] = _np(inp[0][0])
            if name_out and want(name_out):
                o = outp[0] if isinstance(outp, (tuple, list)) else outp
                out[name_out] = _np(o[0])
        handles.append(mod.register_forward_hook(hook))

    # band_split INPUT is the post-gather/post-fold tensor; OUTPUT is (t,f,d)
    cap_io("band_gathered", "band_split_out", model.band_split)

    for i, layer in enumerate(model.layers):
        # layer = (linear_transformer|None, time_transformer, freq_transformer)
        mods = [m for m in layer]
        time_t, freq_t = mods[-2], mods[-1]
        cap_io(None, f"layer{i}_time", time_t)
        cap_io(None, f"layer{i}_freq", freq_t)

    cap_io(None, "mask_raw", model.mask_estimators[0])

    # ---- run ---------------------------------------------------------------
    with torch.no_grad():
        recon = model(wav)

    for h in handles:
        h.remove()

    if want("output_vocals"):
        r = recon[0] if recon.ndim == 3 else recon[0, 0]
        out["output_vocals"] = _np(r)

    return {k: v for k, v in out.items() if want(k)}


def _load(model_dir):
    """Build MelBandRoformer from a checkpoint + its YAML config.

    Accepts a .ckpt path or a directory holding one (plus *.yaml). Returns
    (model, cfg_model_dict). Uses the MIT lucidrains implementation.
    """
    import torch
    import yaml
    from pathlib import Path

    p = Path(model_dir)
    if p.is_dir():
        ckpts = sorted(list(p.glob("*.ckpt")) + list(p.glob("*.pt")) + list(p.glob("*.bin")))
        yamls = sorted(p.glob("*.yaml")) + sorted(p.glob("*.yml"))
        if not ckpts:
            raise FileNotFoundError(f"no checkpoint (*.ckpt/*.pt/*.bin) in {p}")
        ckpt_path, yaml_path = ckpts[0], (yamls[0] if yamls else None)
    else:
        ckpt_path = p
        cands = sorted(p.parent.glob("*.yaml")) + sorted(p.parent.glob("*.yml"))
        yaml_path = cands[0] if cands else None

    if yaml_path is None:
        raise FileNotFoundError(
            "Mel-Band RoFormer needs its YAML config next to the checkpoint "
            "(e.g. config_melband_roformer_vocals_kim.yaml) — the band layout "
            "(num_bands, n_fft, hop) is not recoverable from the weights alone."
        )

    # The upstream configs use `!!python/tuple` (multi_stft_resolutions_window_sizes),
    # which plain safe_load rejects. Teach SafeLoader that one tag rather than
    # dropping to unsafe_load, so the rest of the document stays safely parsed.
    class _Loader(yaml.SafeLoader):
        pass

    _Loader.add_constructor(
        "tag:yaml.org,2002:python/tuple",
        lambda loader, node: tuple(loader.construct_sequence(node)),
    )

    with open(yaml_path) as f:
        cfg_all = yaml.load(f, Loader=_Loader)
    cfg = dict(cfg_all.get("model", {}))
    audio_cfg = cfg_all.get("audio", {}) or {}
    cfg.setdefault("sample_rate", audio_cfg.get("sample_rate", 44100))

    from bs_roformer.mel_band_roformer import MelBandRoformer
    model = MelBandRoformer(**cfg)

    # Memory discipline: this checkpoint is ~870 MB fp32 and the constructed
    # model is another ~900 MB, so a naive load peaks ~2 GB and can OOM a
    # RAM-tight box. mmap=True keeps the checkpoint storage on disk and
    # assign=True hands those tensors straight to the module (freeing the
    # freshly-allocated params) instead of copying into them — roughly halving
    # peak. Falls back if the checkpoint predates zipfile serialization.
    try:
        sd = torch.load(str(ckpt_path), map_location="cpu", mmap=True, weights_only=False)
        assign = True
    except Exception as e:
        print(f"  mmap load unavailable ({type(e).__name__}), falling back to full read")
        sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        assign = False
    if isinstance(sd, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in sd and isinstance(sd[key], dict):
                sd = sd[key]
                break
    sd = { (k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items() }
    try:
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=assign)
    except TypeError:  # torch < 2.1 has no assign=
        missing, unexpected = model.load_state_dict(sd, strict=False)
    del sd
    if missing:
        print(f"  WARNING missing keys: {len(missing)} (first: {missing[:3]})")
    if unexpected:
        print(f"  WARNING unexpected keys: {len(unexpected)} (first: {unexpected[:3]})")
    return model, cfg
