"""HTDemucs reference backend — per-stage intermediate dumper for crispasr-diff.

Loads the pretrained HTDemucs model, runs forward with intermediate
capture, returns a dict of {stage_name: numpy_array} for the diff harness.

Stages dumped:
    spec_input          — STFT magnitude (CaC) after normalization    (C*2, Fq, T)
    time_input          — normalized waveform input to time branch    (C, T)
    enc_freq_0..3       — freq encoder outputs per layer
    enc_time_0..2       — time encoder outputs per layer
    pre_transformer_z   — freq branch before cross-transformer        (C, F, T)
    pre_transformer_xt  — time branch before cross-transformer        (C, T)
    post_transformer_z  — freq branch after cross-transformer
    post_transformer_xt — time branch after cross-transformer
    dec_freq_0..3       — freq decoder outputs per layer
    output_drums        — final separated source: drums               (C, T)
    output_bass         — final separated source: bass
    output_other        — final separated source: other
    output_vocals       — final separated source: vocals
"""

import numpy as np

DEFAULT_STAGES = [
    "input_wav",
    "spec_input", "time_input",
    "enc0_conv", "enc0_gelu", "enc0_dconv", "enc0_rewrite",
    "ct_in_z", "ct_in_xt",
    "ct_l0_z", "ct_l1_z", "ct_l2_z", "ct_l3_z", "ct_l4_z",
    "ct_l0_xt", "ct_l1_xt", "ct_l2_xt", "ct_l3_xt", "ct_l4_xt",
    "enc_freq_0", "enc_freq_1", "enc_freq_2", "enc_freq_3",
    "enc_time_0", "enc_time_1", "enc_time_2",
    "pre_transformer_z", "pre_transformer_xt",
    "post_transformer_z", "post_transformer_xt",
    "dec_freq_0", "dec_freq_1", "dec_freq_2", "dec_freq_3",
    "spec_drums", "spec_bass", "spec_other", "spec_vocals",
    "time_drums", "time_bass", "time_other", "time_vocals",
    "output_drums", "output_bass", "output_other", "output_vocals",
]


def dump(model_dir, audio, stages, max_new_tokens=None, **kwargs):
    """Run HTDemucs forward and return per-stage intermediates.

    Args:
        model_dir: pretrained model name (e.g. "htdemucs") or path
        audio: float32 numpy array of 16 kHz mono PCM (from the harness).
               We resample to 44100 Hz stereo internally.
        stages: set of stage names to capture
        max_new_tokens: unused (no autoregressive decoding)

    Returns:
        dict of {stage_name: numpy_array}
    """
    import torch
    from demucs.pretrained import get_model
    from demucs.htdemucs import HTDemucs

    # Load model
    bag = get_model(str(model_dir))
    if hasattr(bag, "models"):
        model = bag.models[0]
    else:
        model = bag
    assert isinstance(model, HTDemucs)
    model.eval()
    print(f"htdemucs ref: loaded {model_dir}, depth={model.depth}, "
          f"channels={model.channels}, nfft={model.nfft}")

    # Convert 16 kHz mono input to 44100 Hz stereo
    wav_16k = torch.from_numpy(audio).float().unsqueeze(0)  # (1, T_16k)
    import torchaudio
    wav = torchaudio.functional.resample(wav_16k, 16000, model.samplerate)
    wav = wav.expand(2, -1)  # mono → stereo
    wav = wav.unsqueeze(0)  # (1, 2, T)

    # Pad to training segment length
    training_length = int(model.segment * model.samplerate)
    if wav.shape[-1] < training_length:
        wav = torch.nn.functional.pad(wav, (0, training_length - wav.shape[-1]))
    elif wav.shape[-1] > training_length:
        wav = wav[..., :training_length]
    print(f"htdemucs ref: audio shape {list(wav.shape)}, "
          f"segment={training_length} samples @ {model.samplerate} Hz")

    captures = {}

    def maybe_capture(name, tensor):
        if name in stages:
            if isinstance(tensor, torch.Tensor):
                captures[name] = tensor.squeeze(0).numpy().astype(np.float32)
            else:
                captures[name] = np.asarray(tensor, dtype=np.float32)

    # Dump the exact 44.1 kHz stereo waveform fed to the model. The C++ diff
    # replays THIS instead of re-running its own 16k->44.1k resampler, so a
    # resampler mismatch can never masquerade as a model parity failure.
    maybe_capture("input_wav", wav)

    # Manual forward with intermediate capture (mirrors htdemucs.py forward())
    with torch.no_grad():
        mix = wav
        length = mix.shape[-1]

        # STFT → magnitude (CaC)
        z = model._spec(mix)
        mag = model._magnitude(z).to(mix.device)
        x = mag

        B, C, Fq, T = x.shape
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        maybe_capture("spec_input", x)

        # Time branch input
        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)
        maybe_capture("time_input", xt)

        # Encoder
        saved = []
        saved_t = []
        lengths = []
        lengths_t = []

        for idx, encode in enumerate(model.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(model.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = model.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
                maybe_capture(f"enc_time_{idx}", xt)
            if idx == 0:
                # Bisect encoder layer 0 by replicating HEncLayer.forward so a
                # divergence can be pinned to conv / dconv / rewrite / freq_emb.
                import torch.nn.functional as _F
                _y = encode.conv(x)
                maybe_capture("enc0_conv", _y)
                _y = _F.gelu(encode.norm1(_y))
                maybe_capture("enc0_gelu", _y)
                _b, _c, _fr, _t = _y.shape
                _d = _y.permute(0, 2, 1, 3).reshape(-1, _c, _t)
                _d = encode.dconv(_d)
                _y = _d.view(_b, _fr, _c, _t).permute(0, 2, 1, 3)
                maybe_capture("enc0_dconv", _y)
                _z = encode.norm2(encode.rewrite(_y))
                _z = _F.glu(_z, dim=1)
                maybe_capture("enc0_rewrite", _z)

            x = encode(x, inject)
            if idx == 0 and model.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = model.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + model.freq_emb_scale * emb
            saved.append(x)
            maybe_capture(f"enc_freq_{idx}", x)

        maybe_capture("pre_transformer_z", x)
        maybe_capture("pre_transformer_xt", xt)

        # Cross-transformer
        if model.crosstransformer:
            if model.bottom_channels:
                from einops import rearrange
                b, c, f, t = x.shape
                x = rearrange(x, "b c f t-> b c (f t)")
                x = model.channel_upsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = model.channel_upsampler_t(xt)

            # Replicate CrossTransformerEncoder.forward with per-layer capture.
            # Intermediates are rearranged back to (C, Fr, T1) / (C, T2) so they
            # match the C++ buffers elementwise: the runtime flattens the freq
            # branch fr-major (s = fr*T1 + t1) rather than Python's (t1 fr), and
            # (C, Fr, T1) row-major is exactly that layout.
            from einops import rearrange as _rr
            from demucs.transformer import create_2d_sin_embedding as _c2d
            ct = model.crosstransformer
            _B, _C, _Fr, _T1 = x.shape

            def _cap_z(name, t):
                maybe_capture(name, _rr(t, "b (t1 fr) c -> b c fr t1", t1=_T1))

            def _cap_t(name, t):
                maybe_capture(name, _rr(t, "b t2 c -> b c t2"))

            pe2d = _c2d(_C, _Fr, _T1, x.device, ct.max_period)
            pe2d = _rr(pe2d, "b c fr t1 -> b (t1 fr) c")
            xz = _rr(x, "b c fr t1 -> b (t1 fr) c")
            xz = ct.norm_in(xz)
            xz = xz + ct.weight_pos_embed * pe2d

            _B2, _C2, _T2 = xt.shape
            xtt = _rr(xt, "b c t2 -> b t2 c")
            pe1d = ct._get_pos_embedding(_T2, _B2, _C2, x.device)
            pe1d = _rr(pe1d, "t2 b c -> b t2 c")
            xtt = ct.norm_in_t(xtt)
            xtt = xtt + ct.weight_pos_embed * pe1d

            _cap_z("ct_in_z", xz)
            _cap_t("ct_in_xt", xtt)

            for _i in range(ct.num_layers):
                if _i % 2 == ct.classic_parity:
                    xz = ct.layers[_i](xz)
                    xtt = ct.layers_t[_i](xtt)
                else:
                    _old = xz
                    xz = ct.layers[_i](xz, xtt)
                    xtt = ct.layers_t[_i](xtt, _old)
                _cap_z(f"ct_l{_i}_z", xz)
                _cap_t(f"ct_l{_i}_xt", xtt)

            x = _rr(xz, "b (t1 fr) c -> b c fr t1", t1=_T1)
            xt = _rr(xtt, "b t2 c -> b c t2")

            if model.bottom_channels:
                x = rearrange(x, "b c f t-> b c (f t)")
                x = model.channel_downsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = model.channel_downsampler_t(xt)

        maybe_capture("post_transformer_z", x)
        maybe_capture("post_transformer_xt", xt)

        # Decoder
        for idx, decode in enumerate(model.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))

            offset = model.depth - len(model.tdecoder)
            if idx >= offset:
                tdec = model.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    assert pre.shape[2] == 1, pre.shape
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip_t = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip_t, length_t)

            maybe_capture(f"dec_freq_{idx}", x)

        # Final output
        S = len(model.sources)
        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]

        zout = model._mask(z, x)
        x_spec = model._ispec(zout, training_length)

        xt = xt.view(B, S, -1, training_length)
        xt = xt * stdt[:, None] + meant[:, None]
        output = xt + x_spec

        for s_idx, source_name in enumerate(model.sources):
            # Split the two branches so a divergence can be attributed to the
            # spectrogram path (iSTFT) or the waveform path independently.
            maybe_capture(f"spec_{source_name}", x_spec[:, s_idx])
            maybe_capture(f"time_{source_name}", xt[:, s_idx])
            maybe_capture(f"output_{source_name}", output[:, s_idx])

    print(f"htdemucs ref: captured {len(captures)} stages")
    return captures
