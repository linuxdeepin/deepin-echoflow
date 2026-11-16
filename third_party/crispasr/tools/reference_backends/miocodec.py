"""MioCodec v2 reference backend — per-stage dumper for crispasr-diff.

Model: Aratako/MioCodec-25Hz-44.1kHz-v2 (MIT, 133M params).
Architecture: WavLM encoder → FSQ quantizer → wave decoder (Transformer + AdaLN-Zero + iSTFT).

Capture strategy: run the model's encode+decode pipeline, hook intermediate stages.
For the diff harness we focus on the DECODE path (token → waveform), since the
WavLM encoder is frozen/separate and will be handled by a shared WavLM GGUF.

Stages (decode path):
    content_tokens      — (T,) int32 FSQ token indices from encode
    global_embedding    — (128,) speaker/style vector from encode
    fsq_decoded         — (T, 768) FSQ codebook lookup output
    wave_prenet_out     — (T', 512) after wave_prenet transformer
    wave_prior_net_out  — (512, T') after ResNet prior blocks
    wave_decoder_out    — (T', 512) after AdaLN-Zero transformer
    wave_post_net_out   — (512, T') after ResNet post blocks
    wave_upsampler_out  — (T'', 512) after SnakeBeta upsampler
    istft_mag_phase     — (394, T'') linear projection (mag+phase)
    output_waveform     — (samples,) final reconstructed audio
"""

import numpy as np

DEFAULT_STAGES = [
    "content_tokens",
    "global_embedding",
    "fsq_decoded",
    "wave_prenet_out",
    "wave_prior_net_out",
    "wave_decoder_out",
    "wave_post_net_out",
    "wave_upsampler_out",
    "istft_mag_phase",
    "output_waveform",
]


def _np(t):
    import torch
    if isinstance(t, torch.Tensor):
        return t.detach().float().cpu().numpy()
    return np.asarray(t)


def dump(model_dir, audio, stages, max_new_tokens=None, **kwargs):
    """Run MioCodec encode+decode and return per-stage intermediates.

    Args:
        model_dir: HF repo ID (e.g. "Aratako/MioCodec-25Hz-44.1kHz-v2") or local path.
        audio: float32 numpy array, mono, at model's native sample rate (44100 Hz).
        stages: list of stage names to capture.
        max_new_tokens: unused (not an AR model).

    Returns:
        dict mapping stage name → numpy array.
    """
    import gc
    import sys
    import torch
    import torch.nn.functional as F

    # Suppress FlashAttention warning
    import warnings
    warnings.filterwarnings("ignore", message=".*FlashAttention.*")

    # Stub torchaudio if it can't load (CUDA build on CPU-only VPS).
    # MioCodec's SSL extractor needs torchaudio, but for decode-only
    # reference dumps we can work around it by loading the model without
    # the SSL extractor, then running decode from pre-computed tokens.
    try:
        import torchaudio  # noqa: F401
    except (OSError, ImportError):
        import types

        class _TorchaudioStub(types.ModuleType):
            """Recursive stub that returns itself for any attribute access."""
            __path__ = []
            __file__ = ""
            def __getattr__(self, name):
                sub = _TorchaudioStub(f"{self.__name__}.{name}")
                sub.__path__ = []
                sub.__file__ = ""
                sys.modules[sub.__name__] = sub
                return sub
            def __call__(self, *args, **kwargs):
                return None
            def __iter__(self):
                return iter([])
            def __bool__(self):
                return False

        _ta = _TorchaudioStub("torchaudio")
        sys.modules["torchaudio"] = _ta
        # Pre-register known submodules that miocodec imports
        for sub in ["pipelines", "transforms", "functional", "models",
                    "models.wav2vec2", "models.wav2vec2.model",
                    "models.wav2vec2.components"]:
            full = f"torchaudio.{sub}"
            stub = _TorchaudioStub(full)
            stub.__path__ = []
            stub.__file__ = ""
            sys.modules[full] = stub
            setattr(_ta, sub.split(".")[0], stub)
        print("[miocodec] WARNING: torchaudio unavailable — encode path disabled, decode-only mode")

    from miocodec.model import MioCodecModel, MioCodecModelConfig
    from miocodec.module.fsq import FiniteScalarQuantizer
    from miocodec.module.transformer import Transformer
    from miocodec.module.istft_head import ISTFTHead, ResNetStack, UpSamplerBlock

    model_id = str(model_dir)
    print(f"[miocodec] loading model from {model_id} (decode-only)...")

    # Load weights directly, bypassing SSLFeatureExtractor which needs torchaudio
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    import yaml

    cfg_path = hf_hub_download(model_id, "config.yaml")
    wt_path = hf_hub_download(model_id, "model.safetensors")

    with open(cfg_path) as f:
        raw_cfg = yaml.safe_load(f)
    init = raw_cfg["model"]["init_args"]
    cfg = init["config"]

    # Build decode-path modules manually (skip SSL extractor)
    config = MioCodecModelConfig(**cfg)

    local_quantizer = FiniteScalarQuantizer(
        **init["local_quantizer"]["init_args"])
    wave_prenet = Transformer(**init["wave_prenet"]["init_args"])
    wave_decoder = Transformer(**init["wave_decoder"]["init_args"])

    # Build a minimal model object for decode path
    # We need: local_quantizer, wave_prenet, wave_conv_upsample, wave_prior_net,
    #          wave_decoder, wave_post_net, wave_upsampler, istft_head, config
    import types
    model = types.SimpleNamespace()
    model.config = config
    model.local_quantizer = local_quantizer
    model.wave_prenet = wave_prenet
    model.wave_decoder = wave_decoder

    # Conv upsample
    model.wave_conv_upsample = torch.nn.ConvTranspose1d(
        512, 512, kernel_size=config.wave_upsample_factor, stride=config.wave_upsample_factor
    ) if config.wave_upsample_factor > 1 else None

    # ResNet blocks
    model.wave_prior_net = ResNetStack(
        channels=config.wave_decoder_dim, num_blocks=config.wave_resnet_num_blocks,
        kernel_size=config.wave_resnet_kernel_size, num_groups=config.wave_resnet_num_groups,
        dropout=config.wave_resnet_dropout)
    model.wave_post_net = ResNetStack(
        channels=config.wave_decoder_dim, num_blocks=config.wave_resnet_num_blocks,
        kernel_size=config.wave_resnet_kernel_size, num_groups=config.wave_resnet_num_groups,
        dropout=config.wave_resnet_dropout)

    # Upsampler
    model.wave_upsampler = UpSamplerBlock(
        in_channels=config.wave_decoder_dim,
        upsample_factors=list(config.wave_upsampler_factors),
        kernel_sizes=list(config.wave_upsampler_kernel_sizes),
        num_groups=config.wave_resnet_num_groups,
    ) if config.wave_upsampler_factors else None

    # ISTFT head
    model.istft_head = ISTFTHead(
        dim=config.wave_decoder_dim, n_fft=config.n_fft,
        hop_length=config.hop_length, padding=config.istft_padding)

    # Load weights (strict=False since we skipped SSL etc.)
    state_dict = load_file(wt_path, device="cpu")
    # Load into each submodule
    def _load_sub(mod, prefix, sd):
        sub_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if sub_sd:
            mod.load_state_dict(sub_sd, strict=False)
            return len(sub_sd)
        return 0

    n = 0
    n += _load_sub(model.local_quantizer, "local_quantizer.", state_dict)
    n += _load_sub(model.wave_prenet, "wave_prenet.", state_dict)
    n += _load_sub(model.wave_decoder, "wave_decoder.", state_dict)
    if model.wave_conv_upsample:
        n += _load_sub(model.wave_conv_upsample, "wave_conv_upsample.", state_dict)
    n += _load_sub(model.wave_prior_net, "wave_prior_net.", state_dict)
    n += _load_sub(model.wave_post_net, "wave_post_net.", state_dict)
    if model.wave_upsampler:
        n += _load_sub(model.wave_upsampler, "wave_upsampler.", state_dict)
    n += _load_sub(model.istft_head, "istft_head.", state_dict)
    print(f"[miocodec] loaded {n} tensors into decode-path modules")

    # Set all to eval mode
    for mod in [model.local_quantizer, model.wave_prenet, model.wave_decoder,
                model.wave_prior_net, model.wave_post_net, model.istft_head]:
        mod.eval()
    if model.wave_conv_upsample:
        model.wave_conv_upsample.eval()
    if model.wave_upsampler:
        model.wave_upsampler.eval()

    # Helper: calculate target STFT length (from MioCodecModel)
    def _calc_stft_length(audio_length):
        istft_frames = audio_length // config.hop_length
        if model.wave_upsampler is not None:
            return istft_frames // model.wave_upsampler.total_upsample_factor
        return istft_frames

    sr = config.sample_rate
    print(f"[miocodec] decode-only model ready, sample_rate={sr}")

    # Prepare audio — ensure correct sample rate
    sr = model.config.sample_rate
    if isinstance(audio, np.ndarray):
        waveform = torch.from_numpy(audio).float()
    else:
        waveform = audio.float()

    # MioCodec expects (samples,) mono
    if waveform.dim() > 1:
        waveform = waveform.squeeze()
    if waveform.dim() == 0:
        raise ValueError("Audio is empty")

    print(f"[miocodec] audio: {waveform.shape[0]} samples ({waveform.shape[0]/sr:.2f}s) @ {sr} Hz")

    results = {}

    # ===== ENCODE (or generate synthetic tokens for decode-only) =====
    # Without torchaudio/WavLM, we can't run the full encoder. Generate
    # deterministic synthetic tokens for decode-path parity testing.
    # The C++ side will use the same tokens as input.
    n_audio_samples = len(audio) if isinstance(audio, np.ndarray) else waveform.shape[0]
    # Token rate: 25 Hz after 2× downsample from WavLM 50Hz features
    n_tokens = n_audio_samples // (sr // 25)  # approximate
    print(f"[miocodec] generating {n_tokens} synthetic tokens for decode-path test")

    # Deterministic tokens (cycle through valid range 0..12799)
    content_tokens = torch.arange(n_tokens, dtype=torch.long) % 12800
    # Deterministic global embedding (normalized random-like)
    torch.manual_seed(42)
    global_emb = torch.randn(128)
    global_emb = global_emb / global_emb.norm()

    if "content_tokens" in stages:
        results["content_tokens"] = _np(content_tokens.int())
    if "global_embedding" in stages:
        results["global_embedding"] = _np(global_emb)

    # FSQ decode: token indices → embeddings via codebook lookup
    with torch.no_grad():
        from miocodec.module.fsq import FSQ
        fsq_core = model.local_quantizer.fsq
        content_emb_raw = fsq_core.decode(content_tokens)  # (T, 5) normalized codes
        content_emb = model.local_quantizer.proj_out(content_emb_raw)  # (T, 768)

    if "fsq_decoded" in stages:
        results["fsq_decoded"] = _np(content_emb)

    print(f"[miocodec] FSQ decoded: {content_emb.shape}")

    # ===== DECODE (stage by stage) =====
    # We manually run the decode pipeline to capture intermediates.
    # This mirrors model.forward_wave() but with intermediate captures.

    need_decode = any(s in stages for s in [
        "wave_prenet_out", "wave_prior_net_out", "wave_decoder_out",
        "wave_post_net_out", "wave_upsampler_out", "istft_mag_phase",
        "output_waveform",
    ])

    if need_decode:
        with torch.no_grad():
            # Prepare inputs (same as model.decode)
            target_audio_length = n_audio_samples
            content_embedding = content_emb.unsqueeze(0)  # (1, T, 768)
            global_embedding = global_emb.unsqueeze(0)  # (1, 128)

            # Wave prenet
            local_latent = model.wave_prenet(content_embedding)  # (1, T, 512)
            if "wave_prenet_out" in stages:
                results["wave_prenet_out"] = _np(local_latent.squeeze(0))

            # Conv upsample (2×)
            if model.wave_conv_upsample is not None:
                local_latent = model.wave_conv_upsample(
                    local_latent.transpose(1, 2)
                ).transpose(1, 2)  # (1, T*2, 512)

            # Interpolate to STFT frame length
            stft_length = _calc_stft_length(target_audio_length)
            local_latent = F.interpolate(
                local_latent.transpose(1, 2), size=stft_length,
                mode=model.config.wave_interpolation_mode
            ).transpose(1, 2)  # (1, stft_length, 512)

            # Prior ResNet blocks
            local_latent = model.wave_prior_net(
                local_latent.transpose(1, 2)
            ).transpose(1, 2)  # (1, stft_length, 512)
            if "wave_prior_net_out" in stages:
                results["wave_prior_net_out"] = _np(local_latent.squeeze(0).transpose(0, 1))

            # Wave decoder (Transformer with AdaLN-Zero)
            local_latent = model.wave_decoder(
                local_latent, condition=global_embedding.unsqueeze(1)
            )  # (1, stft_length, 512)
            if "wave_decoder_out" in stages:
                results["wave_decoder_out"] = _np(local_latent.squeeze(0))

            # Post ResNet blocks
            local_latent = model.wave_post_net(
                local_latent.transpose(1, 2)
            ).transpose(1, 2)  # (1, stft_length, 512)
            if "wave_post_net_out" in stages:
                results["wave_post_net_out"] = _np(local_latent.squeeze(0).transpose(0, 1))

            # Upsampler (SnakeBeta + ConvTranspose)
            if model.wave_upsampler is not None:
                local_latent = model.wave_upsampler(
                    local_latent.transpose(1, 2)
                )  # (1, T_up, 512)
                if "wave_upsampler_out" in stages:
                    results["wave_upsampler_out"] = _np(local_latent.squeeze(0))

            # ISTFT head: linear → mag/phase → ISTFT
            if "istft_mag_phase" in stages:
                x_proj = model.istft_head.out(local_latent)  # (1, T_up, 394)
                results["istft_mag_phase"] = _np(x_proj.squeeze(0).transpose(0, 1))

            # Full ISTFT → waveform
            wav_out = model.istft_head(local_latent)  # (1, samples)
            if "output_waveform" in stages:
                results["output_waveform"] = _np(wav_out.squeeze(0))

    print(f"[miocodec] captured {len(results)} stages: {list(results.keys())}")

    # Cleanup
    del model
    gc.collect()

    return results
