#!/usr/bin/env python
"""Per-stage parity for the Confucius4 S2A stage.

Drives the real PyTorch S2A (confuciustts/flow) on exactly the inputs the C++
runtime dumped (CRISPASR_CONFUCIUS4_DUMP_S2A=<dir>), including the identical
initial noise, and compares the length-regulator output and the final mel.

Usage:
  python s2a_parity.py --dump-dir <dir> --s2a-ckpt <s2a_model.pt> [--cfg 0.7] [--steps 25]
"""
import argparse, os, sys
import numpy as np
import torch

# The Python blueprint (github.com/netease-youdao/Confucius4-TTS) is not vendored;
# point --ref-repo at a clone of it.
def _import_ref(ref_repo):
    sys.path.insert(0, ref_repo)
    from confuciustts.flow.flow import MaskedDiffWithXvec, MaskedDiffWithXvecConfig
    return MaskedDiffWithXvec, MaskedDiffWithXvecConfig


def load_shapes(d):
    shp = {}
    with open(os.path.join(d, "shapes.txt")) as f:
        for line in f:
            if not line.strip():
                continue
            name, dims = line.rstrip("\n").split("\t")
            shp[name] = tuple(int(x) for x in dims.split(","))
    return shp


def load(d, name, shp, dtype=np.float32):
    a = np.fromfile(os.path.join(d, name + ".bin"), dtype=dtype)
    return a.reshape(shp[name])


def cmp(tag, mine, ref):
    mine = np.asarray(mine, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    # Check shapes BEFORE ravel: after ravel every same-sized array matches, so
    # the guard never fired and a transposed comparison read as cos ~ 0 with
    # identical norms.  Norms are transpose-invariant -- that pattern means
    # "same numbers, wrong order", i.e. a harness bug, not a divergence.
    if mine.shape != ref.shape:
        print(f"{tag:22s} SHAPE MISMATCH mine={mine.shape} ref={ref.shape}"
              f"  (|mine|={np.linalg.norm(mine):.4f} |ref|={np.linalg.norm(ref):.4f})")
        return
    mine = mine.ravel()
    ref = ref.ravel()
    nm, nr = np.linalg.norm(mine), np.linalg.norm(ref)
    cos = float(mine @ ref / (nm * nr + 1e-12))
    # magnitudes printed next to cosine: cosine is scale-blind
    print(f"{tag:22s} cos={cos:.6f}  |mine|={nm:12.4f}  |ref|={nr:12.4f}  "
          f"ratio={nm/(nr+1e-12):7.4f}  max_abs_diff={np.abs(mine-ref).max():.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--s2a-ckpt", required=True)
    ap.add_argument("--ref-repo", required=True,
                    help="clone of github.com/netease-youdao/Confucius4-TTS")
    ap.add_argument("--vocode-out", default=None,
                    help="if set, vocode BOTH mels with the real BigVGAN into this dir "
                         "(ref_mel.wav / cpp_mel.wav) so each can be ASR'd separately")
    ap.add_argument("--cfg", type=float, default=0.7)
    ap.add_argument("--steps", type=int, default=25)
    # Conditioned/prompt path (bug 13 lived here with ZERO parity coverage):
    # style_embedding.bin (spk_embed_dim,) + prompt_mel.bin (T_ref, mel_dim)
    # from the conditioning dumper. When given, the ODE mirrors solve_euler
    # with a non-zero prompt span: prompt_cond prepend, prompt_x, spks,
    # per-step prompt re-zero, and the final prompt strip.
    ap.add_argument("--style", default=None, help="style_embedding.bin (spk_embed_dim,)")
    ap.add_argument("--prompt-mel", default=None, help="prompt_mel.bin (T_ref, mel_dim)")
    a = ap.parse_args()

    MaskedDiffWithXvec, MaskedDiffWithXvecConfig = _import_ref(a.ref_repo)

    d = a.dump_dir
    shp = load_shapes(d)
    print("dumped stages:", {k: v for k, v in shp.items()})

    codes = load(d, "semantic_codes_i32", shp, np.int32)
    lm_latent = load(d, "lm_latent", shp)
    z_init = load(d, "z_init", shp)          # (T_mel, mel_dim) row-major
    cond_cpp = load(d, "cond", shp)          # (T_mel, 512)
    mel_cpp = load(d, "mel", shp)            # (T_mel, mel_dim)

    T_sem = codes.shape[0]
    T_total, mel_dim = z_init.shape
    style = prompt_mel_np = None
    if a.style and a.prompt_mel and os.path.exists(a.style) and os.path.exists(a.prompt_mel):
        style = np.fromfile(a.style, dtype=np.float32)
        prompt_mel_np = np.fromfile(a.prompt_mel, dtype=np.float32).reshape(-1, mel_dim)
    T_ref = 0 if prompt_mel_np is None else prompt_mel_np.shape[0]
    T_mel = T_total - T_ref  # regulator target length (prompt span excluded)
    print(f"T_sem={T_sem} T_total={T_total} T_ref={T_ref} T_target={T_mel} mel_dim={mel_dim}")

    # Build from the SHIPPED config, not the dataclass defaults: the checkpoint
    # has estimator_mlp_ratio=3.0 (ff 1536, not 2048), and cfm_t_scheduler stays
    # at its dataclass default "linear" -- ConditionalCFM's own signature says
    # "cosine", but flow.py passes the config value.
    import yaml
    cfg_path = os.path.join(a.ref_repo, "config", "inference_config.yaml")
    s2a_cfg = yaml.safe_load(open(cfg_path))["s2a_model"]
    cfg = MaskedDiffWithXvecConfig(**s2a_cfg)
    print(f"config: ff_intermediate={int(cfg.estimator_hidden_dim * cfg.estimator_mlp_ratio)} "
          f"t_scheduler={cfg.cfm_t_scheduler!r} cfg_rate={cfg.cfm_inference_cfg_rate}")
    model = MaskedDiffWithXvec(cfg)
    state = torch.load(a.s2a_ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=True)
    model.eval()

    sem = torch.from_numpy(codes.astype(np.int64)).unsqueeze(0)
    # The runtime collects one hidden state per decode step, which is T_sem + 1
    # rows (the last one would predict the token after the final code).  The
    # reference keeps exactly T_sem, and the C++ conditioning already slices to
    # T_sem -- so slice here too rather than feeding the extra row.
    lat = torch.from_numpy(lm_latent[:T_sem]).unsqueeze(0)

    with torch.no_grad():
        # --- stage 1: conditioning (encoder_proj + InterpolateRegulator) ---
        semantic_emb = model.input_embedding(sem).transpose(1, 2)
        text_cond = model.encoder_proj(torch.cat([lat, semantic_emb], dim=-1))
        cond_ref, _ = model.length_regulator(text_cond, torch.tensor([T_mel]))
        cmp("cond (regulator)", cond_cpp, cond_ref[0].numpy())

        # --- stage 2: the Euler ODE, driven on the C++ noise ---
        # mirrors ConditionalCFM.solve_euler; with --style/--prompt-mel the
        # prompt span is live (prompt_len == T_ref), matching
        # MaskedDiffWithXvec.inference exactly.
        dec = model.decoder
        x = torch.from_numpy(z_init.T.copy()).unsqueeze(0)   # (1, mel_dim, T_total)
        if T_ref > 0:
            prompt_condition = model.prompt_cond.expand(1, T_ref, -1)
            mu = torch.cat([prompt_condition, cond_ref], dim=1)
            spks = torch.from_numpy(style).unsqueeze(0)
            prompt_x = torch.zeros_like(x)
            prompt_x[..., :T_ref] = torch.from_numpy(prompt_mel_np.T.copy()).unsqueeze(0)
            x[..., :T_ref] = 0
        else:
            mu = cond_ref
            spks = torch.zeros(1, model.spk_embed_dim)
            prompt_x = torch.zeros_like(x)
        mask = torch.ones(1, T_total, dtype=torch.bool)

        t_span = torch.linspace(0, 1, a.steps + 1)
        if cfg.cfm_t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        t, dt = t_span[0], t_span[1] - t_span[0]

        for step in range(1, len(t_span)):
            if a.cfg > 0:
                x_in = torch.cat([x, x], 0)
                p_in = torch.cat([prompt_x, torch.zeros_like(prompt_x)], 0)
                mu_in = torch.cat([mu, torch.zeros_like(mu)], 0)
                t_in = torch.full((2,), float(t))
                s_in = torch.cat([spks, torch.zeros_like(spks)], 0)
                m_in = torch.cat([mask, mask], 0)
                v = dec.estimator(x_in, m_in, mu_in, t_in, s_in, p_in)
                v_c, v_u = torch.split(v, [1, 1], dim=0)
                v = (1.0 + a.cfg) * v_c - a.cfg * v_u
            else:
                v = dec.estimator(x, mask, mu, torch.tensor([float(t)]), spks, prompt_x)
            x = x + dt * v
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
            if T_ref > 0:
                x[..., :T_ref] = 0  # solve_euler re-zeroes the prompt span every step
            print(f"  step {step:2d}/{a.steps} t={float(t):.4f} |v|max={v.abs().max():.4f}", flush=True)

        mel_ref = x[0, :, T_ref:].numpy().T   # prompt stripped, (T_target, mel_dim) row-major
        cmp("mel (final)", mel_cpp, mel_ref)

        # --- stage 2b: inside the estimator, at step 1 ----------------------
        # cond is exact at F16 and t=0 makes the timestep embedding trivially
        # identical, so any divergence here is the estimator itself.  Split the
        # hand-written CPU stages (timestep MLP, input_embed) from the ggml
        # graph, so we know which side to bisect next.
        est = dec.estimator
        if "dit_t1" in shp:
            t1_ref = est.t_embedder(torch.tensor([0.0]))[0].numpy()
            cmp("dit t1 (timestep)", load(d, "dit_t1", shp), t1_ref)
        if "dit_t2" in shp and hasattr(est, "t_embedder2"):
            t2_ref = est.t_embedder2(torch.tensor([0.0]))[0].numpy()
            cmp("dit t2 (wavenet ts)", load(d, "dit_t2", shp), t2_ref)
        z0 = torch.from_numpy(load(d, "z_init", shp).T.copy()).unsqueeze(0)
        if T_ref > 0:
            z0[..., :T_ref] = 0
        if "dit_x_in" in shp:
            x_in_ref = est.input_embed(z0.transpose(1, 2),
                                       prompt_x.transpose(1, 2),
                                       mu, spks)
            cmp("dit x_in (input_embed)", load(d, "dit_x_in", shp), x_in_ref[0].numpy())

        # --- stage 2c: inside the ggml graph, at step 1 ---------------------
        # Everything entering the graph is exact at F16, so the divergence is in
        # the graph.  Hook the reference modules at the same points the runtime
        # taps and report them in execution order: the first one that drops is
        # the bug.
        graph_taps = [k for k in ("dbg_blk00", "dbg_blk06", "dbg_blk12", "dbg_xres",
                                  "dbg_skip", "dbg_wn", "dbg_fin") if k in shp]
        if graph_taps:
            caught = {}

            def grab(name):
                def hook(_m, _inp, out):
                    caught[name] = (out[0] if isinstance(out, tuple) else out).detach()
                return hook

            depth = len(est.transformer_blocks)
            handles = [
                est.transformer_blocks[0].register_forward_hook(grab("dbg_blk00")),
                est.transformer_blocks[depth // 2].register_forward_hook(grab("dbg_blk06")),
                est.transformer_blocks[depth - 1].register_forward_hook(grab("dbg_blk12")),
                est.transformer_norm.register_forward_hook(grab("dbg_xres")),
                est.skip_linear.register_forward_hook(grab("dbg_skip")),
                est.wavenet.register_forward_hook(grab("dbg_wn")),
                est.final_layer.register_forward_hook(grab("dbg_fin")),
            ]
            est(z0, mask, mu, torch.tensor([0.0]), spks, prompt_x)
            for h in handles:
                h.remove()

            print("  inside the graph, step 1 (execution order):")
            for k in graph_taps:
                if k not in caught:
                    continue
                r = caught[k][0].numpy()
                # Every runtime tap lands as (T, C) row-major: a ggml (C, T)
                # tensor has ne[0]=C as its fast axis, so its memory IS (T, C).
                # The reference DiT blocks emit (T, C) too -- only the WaveNet
                # emits (C, T) and needs transposing.
                if k == "dbg_wn":
                    r = r.T
                cmp(f"  {k}", load(d, k, shp), r)

        # --- stage 3: per-step, TEACHER-FORCED velocity ---------------------
        # Recompute each step's velocity from the C++'s OWN state at that step,
        # so no error accumulates between steps.  A first step that already
        # diverges points at the estimator; a smooth decay points at
        # quantization compounding through the ODE.
        have_steps = sorted(k for k in shp if k.startswith("v_step_"))
        if have_steps:
            print("  per-step teacher-forced velocity (no accumulation):")
            for k in have_steps:
                step = int(k.split("_")[-1])
                z_k = load(d, f"z_step_{step:02d}", shp)
                v_k = load(d, f"v_step_{step:02d}", shp)
                t_k = float(t_span[step - 1])
                xk = torch.from_numpy(z_k.T.copy()).unsqueeze(0)
                if a.cfg > 0:
                    v = dec.estimator(torch.cat([xk, xk], 0),
                                      torch.cat([mask, mask], 0),
                                      torch.cat([mu, torch.zeros_like(mu)], 0),
                                      torch.full((2,), t_k),
                                      torch.cat([spks, torch.zeros_like(spks)], 0),
                                      torch.cat([prompt_x, torch.zeros_like(prompt_x)], 0))
                    vc, vu = torch.split(v, [1, 1], dim=0)
                    v = (1.0 + a.cfg) * vc - a.cfg * vu
                else:
                    v = dec.estimator(xk, mask, mu, torch.tensor([t_k]), spks, prompt_x)
                cmp(f"  v step {step:2d} (t={t_k:.3f})", v_k, v[0].numpy().T)
        np.save(os.path.join(d, "mel_ref.npy"), mel_ref)
        print("wrote", os.path.join(d, "mel_ref.npy"))

        # Per-frame statistics: a mel that is globally plausible but locally flat
        # (or saturated at the log_eps floor) shows up here and not in a cosine.
        for tag, m in (("cpp", mel_cpp), ("ref", mel_ref)):
            m = np.asarray(m, dtype=np.float64)
            print(f"  {tag} mel: min={m.min():8.3f} max={m.max():8.3f} mean={m.mean():8.3f} "
                  f"std={m.std():7.3f}  per-frame std (mean)={m.std(axis=1).mean():7.3f}  "
                  f"frames at log_eps floor={(m <= np.log(1e-5) + 1e-3).mean() * 100:.1f}%")

    if a.vocode_out:
        # Vocode BOTH mels with the reference BigVGAN.  If the REFERENCE audio is
        # also unintelligible then the S2A port is not the blocker -- the missing
        # speaker conditioning is (the model is zero-shot and always has a prompt).
        os.makedirs(a.vocode_out, exist_ok=True)
        sys.path.insert(0, a.ref_repo)
        import json as _json
        import scipy.io.wavfile as wavfile
        from huggingface_hub import hf_hub_download
        from external.bigvgan.bigvgan import BigVGAN
        from external.bigvgan.env import AttrDict

        # Build from config + state dict directly: BigVGAN's PyTorchModelHubMixin
        # predates the current huggingface_hub, whose _from_pretrained passes
        # proxies/resume_download that its override does not accept.
        repo = "nvidia/bigvgan_v2_22khz_80band_256x"
        h = AttrDict(_json.load(open(hf_hub_download(repo, "config.json"))))
        voc = BigVGAN(h)
        ckpt = torch.load(hf_hub_download(repo, "bigvgan_generator.pt"),
                          map_location="cpu", weights_only=False)
        voc.load_state_dict(ckpt["generator"])
        voc.remove_weight_norm()
        voc.eval()
        with torch.no_grad():
            for tag, m in (("ref", mel_ref), ("cpp", mel_cpp)):
                t = torch.from_numpy(np.ascontiguousarray(np.asarray(m, dtype=np.float32).T)).unsqueeze(0)
                wav = voc(t).squeeze().cpu().numpy()
                out = os.path.join(a.vocode_out, f"{tag}_mel.wav")
                wavfile.write(out, 22050, (np.clip(wav, -1, 1) * 32767).astype(np.int16))
                print(f"  vocoded {tag}: {out}  ({len(wav) / 22050:.2f}s, peak={np.abs(wav).max():.3f})")


if __name__ == "__main__":
    main()
