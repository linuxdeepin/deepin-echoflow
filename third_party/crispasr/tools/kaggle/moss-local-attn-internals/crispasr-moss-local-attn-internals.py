# %% [markdown]
# # MOSS-TTS-Local 4B — layer-0 attention internals (#249 option 2): flash vs eager
#
# The stop bug is a deterministic graph difference (f32 == f16), seeded at layer 0:
# our attention output is 0.3% off with an EXACT input. Prime suspect: our fused
# ggml flash_attn_ext vs the reference's eager softmax(QK^T/sqrt d)V. Reference-FREE
# decisive check: dump our own layer-0 Q_post_rope / Kfull / Vfull / fa_out (the
# core_attn CRISPASR_CORE_ATTN_DUMP_FA_LAYER hook), recompute eager attention from
# those exact Q/K/V in numpy, and compare to our fa_out.
#   eager != our fa_out -> flash_attn_ext is the bug (scale/mask/accumulation)
#   eager == our fa_out -> flash is fine; the 0.3% is upstream in Q/K/V (rope/qk-norm)
# Also compare our layer-0 attn output (post o_proj) to the HF reference module out.

# %% [code]
import json, os, subprocess, sys, gc, math, shutil
from pathlib import Path
import numpy as np

REPO = Path("/kaggle/temp/CrispASR")
WORK = Path("/kaggle/working")
REF = os.environ.get("CRISPASR_REF", "fix/249-moss")
HF = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
TEXT = "Hello world."
if not REPO.exists():
    subprocess.check_call(["git", "clone", "--recursive", "--depth", "1", "--branch", REF,
                           "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    subprocess.check_call(["git", "-C", str(REPO), "submodule", "update", "--init",
                           "--recursive", "--depth", "1"], timeout=1800)
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402
kh.init_progress(hf_progress_repo="cstr/crispasr-kaggle-progress")
step = kh.step
step("start", ref=REF)
TOKEN = kh.resolve_hf_token("HF_TOKEN")
kh.install_build_toolchain()
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "hf_transfer"])


def robust_download(repo, fname, local_dir, token, tries=3, timeout=900):
    # Run hf_hub_download in a child process with a HARD timeout so an
    # hf_transfer/xet stall (free_gb flat for 56 min — the trap that ate a
    # session) can't hang forever. First try uses hf_transfer; retries disable it
    # (and xet) for a plain, socket-timeout-respecting resumable download.
    import multiprocessing as mp

    def _dl(q, use_ht):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" if use_ht else "0"
        os.environ["HF_HUB_DISABLE_XET"] = "0" if use_ht else "1"
        try:
            from huggingface_hub import hf_hub_download
            q.put(("ok", hf_hub_download(repo, fname, local_dir=local_dir, token=token)))
        except Exception as e:  # noqa: BLE001
            q.put(("err", repr(e)))

    for i in range(tries):
        q = mp.Queue()
        p = mp.Process(target=_dl, args=(q, i == 0))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate(); p.join()
            step("dl.hang.retry", repo=repo, file=fname, attempt=i)
            continue
        if not q.empty():
            statv, val = q.get()
            if statv == "ok":
                return val
            step("dl.err.retry", repo=repo, file=fname, attempt=i, err=val[:200])
    raise RuntimeError(f"download failed after {tries} tries: {repo}/{fname}")

BUILD = REPO / "build"
step("cmake.configure")
subprocess.run(["cmake", "-G", "Ninja", "-B", str(BUILD), "-S", str(REPO),
                "-DCMAKE_BUILD_TYPE=Release"] + kh.crispasr_cmake_flags(), check=True)
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(f"cmake --build {BUILD} --target crispasr-cli -j{kh.safe_build_jobs(gpu=False)}")
CLI = (BUILD / "bin" / "crispasr") if (BUILD / "bin" / "crispasr").exists() else next(iter(BUILD.rglob("crispasr")))
os.environ["LD_LIBRARY_PATH"] = f"{BUILD/'src'}:{BUILD/'ggml'/'src'}:{os.environ.get('LD_LIBRARY_PATH','')}"
step("build.done")

MODELS = Path("/kaggle/temp/models"); MODELS.mkdir(parents=True, exist_ok=True)
with kh.build_heartbeat("download"):
    CODEC = robust_download("cstr/moss-tts-local-v1.5-GGUF", "moss-tts-local-v1.5-codec.gguf", str(MODELS), TOKEN)
    F16 = robust_download("cstr/moss-tts-local-v1.5-GGUF", "moss-tts-local-v1.5-f16.gguf", str(MODELS), TOKEN)

# ── our layer-L attention-internals dump (L=10: the divergence layer) ──────────
LAYER = int(os.environ.get("CRISPASR_DIAG_LAYER", "10"))
fa_path = WORK / "ours_fa.txt"
sub_path = WORK / "ours_sub.txt"
lay_path = WORK / "ours_layers.txt"  # every block's output (input to L = block L-1)
env = {**os.environ, "CRISPASR_CORE_ATTN_DUMP_FA_LAYER": str(LAYER),
       "CRISPASR_MOSS_TTS_LOCAL_DUMP_FA_PATH": str(fa_path),
       "CRISPASR_MOSS_TTS_LOCAL_DUMP_SUBLAYER": str(LAYER),
       "CRISPASR_MOSS_TTS_LOCAL_DUMP_SUBLAYER_PATH": str(sub_path),
       "CRISPASR_MOSS_TTS_LOCAL_DUMP_LAYERS": str(lay_path)}
with kh.build_heartbeat("ours.synth"):
    try:
        subprocess.run([str(CLI), "--backend", "moss-tts-local", "-m", F16, "--codec-model", CODEC,
                        "--tts", TEXT, "--tts-output", str(WORK / "o.wav"), "--no-prints"],
                       env=env, timeout=600, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        pass


def read_ne(path):
    # lines: "<name> ne0 ne1 ne2 ne3 v0 v1 ..." -> {name: (ne, np.array)}
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            p = line.split()
            if not p:
                continue
            ne = [int(x) for x in p[1:5]]
            vals = np.array([float(x) for x in p[5:]], dtype=np.float64)
            out[p[0]] = (ne, vals)
    return out


def read_sub(path):
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            p = line.split()
            if p:
                out[p[0]] = np.array([float(x) for x in p[1:]], dtype=np.float64)
    return out


fa = read_ne(fa_path)
subo = read_sub(sub_path)
ours_blocks = read_sub(lay_path)  # {"blk_N": vec}
step("ours.done", fa_keys=list(fa.keys()), sub_keys=list(subo.keys()), n_blocks=len(ours_blocks))

# ── reference-free flash-vs-eager check ───────────────────────────────────────
result = {"flash_vs_eager": None}
if all(k in fa for k in ("DBG_Q_post_rope", "DBG_Kfull", "DBG_Vfull", "DBG_fa_out")):
    (qne, qv) = fa["DBG_Q_post_rope"]   # ne=(hd, n_q, T)
    (kne, kv) = fa["DBG_Kfull"]         # ne=(hd, Lk, n_q)  (GQA-expanded)
    (vne, vv) = fa["DBG_Vfull"]         # ne=(hd, Lk, n_q)
    (fne, fv) = fa["DBG_fa_out"]        # ne=(hd, n_q, T) expected
    hd, n_q, T = qne[0], qne[1], qne[2]
    Lk = kne[1]
    Q = qv.reshape(T, n_q, hd)          # [t,h,d]
    K = kv.reshape(n_q, Lk, hd)         # [h,k,d]
    V = vv.reshape(n_q, Lk, hd)         # [h,k,d]
    scale = 1.0 / math.sqrt(hd)
    eager = np.zeros((T, n_q, hd), dtype=np.float64)
    for h in range(n_q):
        Qh = Q[:, h, :]                 # (T, hd)
        Kh = K[h]                       # (Lk, hd)
        Vh = V[h]                       # (Lk, hd)
        scores = (Qh @ Kh.T) * scale    # (T, Lk)
        # causal (prefill, n_past=0): key k visible to query t iff k <= t
        mask = np.triu(np.full((T, Lk), -np.inf), k=1)
        scores = scores + mask
        scores -= scores.max(axis=1, keepdims=True)
        w = np.exp(scores); w /= w.sum(axis=1, keepdims=True)
        eager[:, h, :] = w @ Vh         # (T, hd)
    # fa_out reshape: ne=(hd, n_q, T) -> [t,h,d]
    fa_out = fv.reshape(fne[2], fne[1], fne[0]) if fne[2] == T else fv.reshape(T, n_q, hd)
    a, b = eager.reshape(-1), fa_out.reshape(-1)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    l2rel = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9))
    # also last-position only (the token that drives the stop)
    al, bl = eager[-1].reshape(-1), fa_out[-1].reshape(-1)
    cos_last = float(al @ bl / (np.linalg.norm(al) * np.linalg.norm(bl) + 1e-9))
    result["flash_vs_eager"] = {"cos": round(cos, 6), "l2rel": round(l2rel, 6),
                                "cos_last": round(cos_last, 6), "hd": hd, "n_q": n_q, "T": T, "Lk": Lk,
                                "fa_ne": fne, "q_ne": qne}
step("flash_vs_eager", **(result["flash_vs_eager"] or {"err": "missing FA tensors"}))

# ── reference module-output comparison (the known 0.3%) ───────────────────────
step("ref.load")
import torch  # noqa: E402
from transformers import AutoModel, AutoProcessor  # noqa: E402
torch.set_num_threads(os.cpu_count() or 4)
model = AutoModel.from_pretrained(HF, trust_remote_code=True, torch_dtype=torch.float32,
                                  attn_implementation="eager", token=TOKEN).eval()
proc = AutoProcessor.from_pretrained(HF, trust_remote_code=True)
ref = {}


def mk(name):
    def hook(_m, _i, o):
        hs = o[0] if isinstance(o, (tuple, list)) else o
        ref[name] = hs[:, -1, :].detach().float().reshape(-1).numpy().astype(np.float64)
    return hook


layers = model.transformer.layers
lyrL = layers[LAYER]
attn_mod = getattr(lyrL, "self_attn", None) or getattr(lyrL, "attn", None)
if attn_mod is not None:
    attn_mod.register_forward_hook(mk("attn_out"))
    for sub in ("q_norm", "k_norm", "v_proj"):
        m = getattr(attn_mod, sub, None)
        if m is not None:
            m.register_forward_hook(mk(sub))
# block outputs for L-1 (the INPUT to layer L) and L (the output), to measure
# whether layer L amplifies its input divergence
for bl in (LAYER - 1, LAYER):
    if 0 <= bl < len(layers):
        layers[bl].register_forward_hook(mk(f"blk_{bl}"))
msg = proc.build_user_message(text=TEXT)
feat = proc([msg], mode="generation")
input_ids = feat["input_ids"] if isinstance(feat, dict) else feat.input_ids
with kh.build_heartbeat("ref.gen"):
    model.generate(input_ids=input_ids, max_new_frames=1, do_sample=False)
del model
gc.collect()


def cos(a, b):
    if a is None or b is None or a.shape != b.shape:
        return None
    return round(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)), 6)


def l2r(a, b):
    if a is None or b is None or a.shape != b.shape:
        return None
    return round(float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9)), 6)


def last_pos(entry):
    # entry = (ne, flat); tensor is (ne0, ne1, ne2=T) row-fastest ne0 -> last-pos = ne0*ne1 values [h,d]
    if entry is None:
        return None
    ne, flat = entry
    per = ne[0] * ne[1]
    T = ne[2] if ne[2] > 0 else 1
    return flat[(T - 1) * per: T * per]


# our post-qk-norm pre-RoPE Q/K/V (last position) vs the reference module outputs
oq = last_pos(fa.get("DBG_Q_prerope"))
ok = last_pos(fa.get("DBG_K_prerope"))
ov = last_pos(fa.get("DBG_V_new"))
result["prerope"] = {
    "Q_vs_qnorm": {"cos": cos(oq, ref.get("q_norm")), "l2rel": l2r(oq, ref.get("q_norm"))},
    "K_vs_knorm": {"cos": cos(ok, ref.get("k_norm")), "l2rel": l2r(ok, ref.get("k_norm"))},
    "V_vs_vproj": {"cos": cos(ov, ref.get("v_proj")), "l2rel": l2r(ov, ref.get("v_proj"))},
    "sizes": {"oq": None if oq is None else len(oq), "ref_q": None if ref.get("q_norm") is None else len(ref["q_norm"]),
              "ok": None if ok is None else len(ok), "ov": None if ov is None else len(ov)},
}
result["attn_out_vs_ref"] = cos(subo.get(f"sub_attn_{LAYER}"), ref.get("attn_out"))

# block-level divergence: input to layer L (= block L-1 output) vs its output.
# This tells us whether layer L AMPLIFIES a tiny input difference or the ops
# themselves diverge given a nearly-identical input.
in_cos = cos(ours_blocks.get(f"blk_{LAYER-1}"), ref.get(f"blk_{LAYER-1}"))
in_l2 = l2r(ours_blocks.get(f"blk_{LAYER-1}"), ref.get(f"blk_{LAYER-1}"))
out_cos = cos(ours_blocks.get(f"blk_{LAYER}"), ref.get(f"blk_{LAYER}"))
result["block"] = {"layer": LAYER, "input_cos": in_cos, "input_l2rel": in_l2, "output_cos": out_cos}
pr = result["prerope"]
qk_l2 = max([x for x in (pr["Q_vs_qnorm"]["l2rel"], pr["K_vs_knorm"]["l2rel"]) if x is not None] or [None])

# Amplification ratio: how much bigger is the post-qk-norm Q/K divergence than the
# raw input (block L-1) divergence? A matmul/norm shouldn't inflate it much.
amp = (qk_l2 / in_l2) if (qk_l2 and in_l2) else None
result["amplification"] = {"input_l2rel": in_l2, "prerope_qk_l2rel": qk_l2, "ratio_qk_over_input": amp}

if in_l2 is not None and qk_l2 is not None:
    if qk_l2 > 5 * in_l2 and qk_l2 > 0.05:
        verdict = (f"OP BUG at layer {LAYER}: qk-norm/proj INFLATES the input divergence "
                   f"{amp:.0f}x (input l2rel {in_l2:.4g} -> pre-RoPE Q/K l2rel {qk_l2:.4g}). "
                   f"A matmul/rms_norm can't do that on nearly-parallel input -> real op/norm bug.")
    elif qk_l2 <= 3 * in_l2:
        verdict = (f"AMPLIFICATION (not an op bug): pre-RoPE Q/K track the input divergence "
                   f"(input l2rel {in_l2:.4g}, qk l2rel {qk_l2:.4g}, ratio {amp:.1f}); "
                   f"attn_out cos {result['attn_out_vs_ref']} -> the blow-up is in the softmax "
                   f"(peaked/attention-sink at layer {LAYER}), i.e. irreducible sensitivity.")
    else:
        verdict = (f"PARTIAL: qk l2rel {qk_l2:.4g} vs input {in_l2:.4g} (ratio {amp:.1f}) — "
                   f"some norm/proj amplification; inspect Q vs K.")
else:
    verdict = "inconclusive — missing block or FA tensors (check keys/sizes)"
(WORK / "attn_internals.json").write_text(json.dumps({"verdict": verdict, **result}, indent=2))
step("done", verdict=verdict, block=result["block"], amplification=result["amplification"],
     attn_out_vs_ref=result["attn_out_vs_ref"])
print("DONE", verdict, flush=True)

# ── refresh the ccache dataset (gotcha #22): tar to a single file so it lands in
# page 1 of kernel-output, and keep /kaggle/working minimal ──────────────────
try:
    for cand in ("/kaggle/working/.ccache", "/kaggle/temp/.ccache", str(Path.home() / ".ccache")):
        if Path(cand).exists():
            subprocess.run(f"tar cf /kaggle/working/ccache.tar -C {Path(cand).parent} {Path(cand).name}",
                           shell=True, timeout=600)
            # keep /kaggle/working small so ccache.tar lands in output page 1 (#22)
            if cand.startswith("/kaggle/working"):
                shutil.rmtree(cand, ignore_errors=True)
            step("ccache.tar", src=cand, size_mb=round(Path("/kaggle/working/ccache.tar").stat().st_size / 1e6, 1))
            break
except Exception as e:  # noqa: BLE001
    step("ccache.tar.err", err=repr(e)[:150])
