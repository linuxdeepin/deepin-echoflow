#!/usr/bin/env python3
"""Convert Beatrice v2 (Project Beatrice) components to GGUF.

    python models/convert-beatrice-to-gguf.py \
        --component pitch_estimator \
        --model 104_3_checkpoint_00300000.pt \
        --output beatrice-pitch-f32.gguf

Reference: `fierce-cats/beatrice-trainer`, `beatrice_trainer/__main__.py`.
Blueprint: docs/music-transcription/BEATRICE_BLUEPRINT.md.

LICENCE. Beatrice's trainer repo is MIT and its README states the source AND the
trained models are MIT, so unlike RVC no acceptance gate is needed. --license
still defaults to `mit` explicitly rather than being silent, because CrispASR's
registry gate matches on the tag and a checkpoint from some other training run
may not share the base's terms.

Beatrice is THREE separate checkpoints (phone_extractor / pitch_estimator /
net_g), each a different file with a different top-level key. This converter
takes one at a time; --component says which, and the checkpoint is verified to
actually contain it rather than trusted from the filename.

Details that are NOT the obvious default:

  * WEIGHT FUSION. `merge_weights()` folds the LayerNorm affine into pwconv1,
    `gamma` into pwconv2, and pre/post scale into dwconv/pwconv2 -- exactly as
    the RVC converter fuses weight_norm. VERIFIED output-preserving on the real
    checkpoint (cos 1.0000000000, max_abs 7.6e-06 = f32 noise). After it,
    gamma / pre_scale / post_scale / post_scale_weight are all identically 1
    and the per-block LayerNorm affine is identity, so all of them are DROPPED
    here and the runtime graph never sees them.
  * ...but the STACK-level `backbone.norm` and `backbone.final_layer_norm` are
    NOT touched by merge_weights and keep real affine parameters. So the port
    needs LayerNorm *with* affine at the stack boundaries and *without* affine
    (normalise only) inside the blocks. Assuming one form everywhere is wrong
    in one place or the other.
  * WSConv1d/WSLinear standardise with torch.var_mean, which is UNBIASED
    (correction=1). numpy's ddof=0 default would mis-scale every weight by
    ~20%. PitchEstimator does not use them, but PhoneExtractor may -- the
    fusion path is shared, so this is asserted rather than assumed.
  * The `.beatrice` dump format (dump_layer) pre-folds the attention scale as
    sqrt on BOTH q and k and reorders heads. We convert from the .pt state_dict
    instead, so that transformation must NOT be applied.
  * CausalConv1d: padding = (k-1)*dilation - delay but trim = (k-1)*dilation -
    2*delay. They are NOT equal when delay > 0, and a delay>0 layer looks
    AHEAD by `delay` frames. Both are emitted so the runtime never re-derives.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    sys.exit("error: pip install torch")
try:
    from gguf import GGUFWriter
except ImportError:
    sys.exit("error: pip install gguf")

ARCH = "beatrice"

# Neutralised by merge_weights() -- identically 1.0 (or identity affine) after
# fusion, so emitting them would ship dead weight and invite the runtime to
# apply them twice.
DROP_SUFFIXES = (
    ".gamma",
    ".pre_scale",
    ".post_scale",
    ".post_scale_weight",
)


def fuse_model(component, m):
    """Apply the convert-time weight transforms for `component`, and ONLY the
    ones verified to preserve the output.

    pitch_estimator: merge_weights() wholesale (verified cos 1.0000000000).

    phone_extractor: NOT wholesale. Bisected on the real checkpoint:

        remove_weight_norm()        rel 0.000e+00   exact
        backbone.merge_weights()    rel 1.15e-06    f32 noise
        merge_weights()  (full)     rel 1.71e-02    BREAKS THE MODEL

    The full method additionally folds `feature_projection.norm`'s affine into
    `backbone.embed`. That fold adds `sum_{i,k} W[o,i,k] * norm_bias[i]` to the
    conv bias, which is only correct if EVERY kernel tap sees norm_bias -- but
    the sequence edges are ZERO-padded, so the taps hanging off the end see 0.
    It is an edge approximation, and because this backbone is `use_mha=True`,
    attention propagates the edge error to every frame: ~3e-04 mid-sequence,
    ~3e-03 at the start.

    So we skip that one fold and keep feature_projection.norm explicit in the
    graph. The result is EXACT where upstream's own inference path is not --
    worth knowing when comparing against a `.beatrice` dump, which carries the
    approximation.
    """
    if component == "pitch_estimator":
        m.merge_weights()
        return ["merge_weights"]
    if component == "phone_extractor":
        m.remove_weight_norm()
        m.backbone.merge_weights()  # deliberately NOT m.merge_weights()
        return ["remove_weight_norm", "backbone.merge_weights (NOT full merge_weights -- see fuse_model)"]
    sys.exit(f"error: no fusion defined for {component}")


def verify_fusion(component, bt, sd, build, fuse):
    """Run the model BEFORE and AFTER the weight transforms and refuse to write
    a GGUF if they disagree.

    This exists because nothing else can catch a bad fusion. The parity dumper's
    gate compares its step-by-step spec against the module's own forward(), and
    a fusion changes BOTH arms identically -- so it passes a fusion that has
    silently altered the model. That is not hypothetical: applying upstream's
    full PhoneExtractor.merge_weights() here moves the output by rel 1.7e-02 and
    the dumper's gate still reports cos 1.0000000000, max_abs 0.000e+00.

    The check is cheap, deterministic, and runs on every conversion.
    """
    torch.manual_seed(0)
    m0, _ = build(bt, component, sd)
    m0.load_state_dict(sd)
    m0.eval()
    # wav_length a multiple of 160 (the feature extractor warns otherwise) and
    # long enough that the post-/160 length is a multiple of 4 for the MHA path.
    wav = torch.randn(1, 1, 160 * 200) * 0.1
    with torch.inference_mode():
        before = m0(wav, return_stats=False) if component == "phone_extractor" else m0(wav)[0]
    fuse(component, m0)
    with torch.inference_mode():
        after = m0(wav, return_stats=False) if component == "phone_extractor" else m0(wav)[0]

    peak = float(before.abs().max())
    mx = float((before - after).abs().max())
    rel = mx / peak if peak > 0 else float("inf")
    TOL = 1e-4
    print(f"fusion check: max_abs={mx:.3e} rel={rel:.3e} (tol {TOL:.0e})")
    if not (rel <= TOL):
        sys.exit(
            f"error: the weight transforms CHANGE THE MODEL (relative {rel:.3e} > {TOL:.0e}).\n"
            "       Refusing to write a GGUF. Bisect which transform is responsible before\n"
            "       shipping it -- see fuse_model()'s docstring for the phone_extractor case,\n"
            "       where upstream's own merge_weights() is an edge approximation."
        )


def build_component(bt, component, sd):
    """Instantiate the module for `component`, geometry read off the weights."""
    if component == "pitch_estimator":
        pitch_bins = sd["head.weight"].shape[0]
        channels = sd["head.weight"].shape[1]
        n_blocks = 1 + max(
            int(k.split(".")[2]) for k in sd if k.startswith("backbone.convnext.")
        )
        m = bt.PitchEstimator(pitch_bins=pitch_bins, channels=channels, n_blocks=n_blocks)
        meta = {
            "pitch_bins": pitch_bins,
            "channels": channels,
            "n_blocks": n_blocks,
            "pitch_bins_per_octave": m.pitch_bins_per_octave,
        }
        return m, meta
    if component == "phone_extractor":
        phone_channels = sd["head.weight_v"].shape[0] if "head.weight_v" in sd else sd["head.weight"].shape[0]
        hidden = sd["backbone.embed.weight"].shape[0]
        n_blocks = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("backbone.convnext."))
        m = bt.PhoneExtractor(phone_channels=phone_channels, hidden_channels=hidden, n_blocks=n_blocks)
        n_heads = m.backbone.convnext[0].mha.num_heads
        return m, {
            "phone_channels": phone_channels,
            "hidden_channels": hidden,
            "n_blocks": n_blocks,
            "n_heads": n_heads,
        }
    sys.exit(f"error: --component {component} not implemented yet")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, help="the .pt for THIS component")
    ap.add_argument(
        "--component",
        required=True,
        choices=["pitch_estimator", "phone_extractor"],
        help="which network this checkpoint holds (verified against its contents)",
    )
    ap.add_argument("--output", required=True)
    ap.add_argument("--dtype", choices=["f32", "f16"], default="f32")
    ap.add_argument(
        "--trainer-path",
        help="dir containing the beatrice_trainer package (else assumed importable)",
    )
    ap.add_argument(
        "--license",
        default="mit",
        help="SPDX-ish tag for THESE WEIGHTS. The trainer repo's own checkpoints "
        "are mit; a checkpoint from another training run may not be.",
    )
    args = ap.parse_args()

    if args.trainer_path:
        sys.path.insert(0, args.trainer_path)
    try:
        import beatrice_trainer.__main__ as bt
    except ImportError as e:
        sys.exit(
            f"error: cannot import beatrice_trainer ({e}).\n"
            "       Fetch beatrice_trainer/{__init__,__main__}.py from "
            "fierce-cats/beatrice-trainer and pass --trainer-path.\n"
            "       It also needs `pip install pyworld`."
        )

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or args.component not in ck:
        have = list(ck.keys()) if isinstance(ck, dict) else type(ck).__name__
        sys.exit(
            f"error: {args.model} does not contain '{args.component}'. It has: {have}\n"
            "       Beatrice ships one component per checkpoint -- phone_extractor, "
            "pitch_estimator and net_g are three DIFFERENT files."
        )
    sd = ck[args.component]

    model, meta = build_component(bt, args.component, sd)
    model.load_state_dict(sd)  # strict: the architecture must match exactly
    model.eval()

    verify_fusion(args.component, bt, sd, build_component, fuse_model)
    transforms = fuse_model(args.component, model)
    fused = model.state_dict()

    # Decide what to drop by PROVING it is neutral, not by trusting a method
    # name. Anything the fusion left as an exact identity is dead weight the
    # runtime must not re-apply; anything else is kept even if we expected it to
    # vanish.
    def is_identity(name, t):
        if name.endswith(DROP_SUFFIXES):  # scalars/vectors folded to 1.0
            return bool(torch.allclose(t, torch.ones_like(t)))
        if name.endswith(".attn_norm.weight"):  # folded into mha.in_proj
            return bool(torch.allclose(t, torch.ones_like(t)))
        if name.endswith(".attn_norm.bias"):
            return bool(torch.allclose(t, torch.zeros_like(t)))
        return False

    for k, t in fused.items():
        if k.endswith(DROP_SUFFIXES) and not is_identity(k, t):
            sys.exit(
                f"error: {k} is not all-ones after fusion "
                f"(max deviation {float((t - 1).abs().max()):.3e}).\n"
                "       Dropping it would silently change the model."
            )

    w = GGUFWriter(str(args.output), ARCH)
    w.add_string("beatrice.component", args.component)
    w.add_string("beatrice.paraphernalia_version", bt.PARAPHERNALIA_VERSION)
    w.add_uint32("beatrice.in_sample_rate", 16000)
    for k, v in meta.items():
        w.add_uint32(f"beatrice.{args.component}.{k}", int(v))

    if args.component == "pitch_estimator":
        # The DSP front end's constants. These are extract_pitch_features()
        # defaults, not config -- carried explicitly so the runtime never
        # re-derives them, and asserted because the function itself asserts it.
        hop, win = 160, 560
        max_corr_period, corr_win_length, cutoff_bin = 256, 304, 64
        assert max_corr_period + corr_win_length == win, "extract_pitch_features invariant"
        w.add_uint32("beatrice.pitch.hop_length", hop)
        w.add_uint32("beatrice.pitch.win_length", win)
        w.add_uint32("beatrice.pitch.max_corr_period", max_corr_period)
        w.add_uint32("beatrice.pitch.corr_win_length", corr_win_length)
        w.add_uint32("beatrice.pitch.instfreq_cutoff_bin", cutoff_bin)
        # CausalConv1d geometry: padding != trim whenever delay > 0.
        emb = model.backbone.embed
        w.add_uint32("beatrice.pitch.embed_kernel_size", int(emb.kernel_size[0]))
        w.add_uint32("beatrice.pitch.embed_padding", int(emb.padding[0]))
        w.add_uint32("beatrice.pitch.embed_trim", int(emb.trim))
        dw = model.backbone.convnext[0].dwconv
        w.add_uint32("beatrice.pitch.dw_kernel_size", int(dw.kernel_size[0]))
        w.add_uint32("beatrice.pitch.dw_padding", int(dw.padding[0]))
        w.add_uint32("beatrice.pitch.dw_trim", int(dw.trim))

    w.add_string("general.license", args.license)
    w.add_string(
        "general.license.description",
        "Beatrice v2. fierce-cats/beatrice-trainer is MIT and its README states "
        "the source and trained models alike are MIT. Checkpoints from other "
        "training runs may carry different terms -- verify before redistributing.",
    )

    emitted, dropped = 0, []
    for k, t in fused.items():
        if is_identity(k, t):
            dropped.append(k)
            continue
        a = t.detach().cpu().float().numpy()
        if args.dtype == "f16" and a.ndim >= 2 and not k.endswith(".bias"):
            a = a.astype(np.float16)
        w.add_tensor(k, np.ascontiguousarray(a))
        emitted += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    if emitted + len(dropped) != len(fused):
        print(
            f"WARNING: accounting does not balance: emitted={emitted} "
            f"dropped={len(dropped)} total={len(fused)}",
            file=sys.stderr,
        )

    print(f"beatrice/{args.component}: " + " ".join(f"{k}={v}" for k, v in meta.items()))
    print(f"transforms applied: {', '.join(transforms)}")
    print(
        f"wrote {args.output}: {emitted} tensors, {len(dropped)} dropped as PROVEN identity "
        f"after fusion, dtype {args.dtype}"
    )
    print(f"NOTE: weights licensed '{args.license}'.")


if __name__ == "__main__":
    main()
