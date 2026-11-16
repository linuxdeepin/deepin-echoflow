# %% [markdown]
# # MOSS-TTS-Local 4B — does eager+F32 reduce the layer-10 divergence? (#249)
#
# The stop A/B was too noisy (stochastic sampling). DETERMINISTIC test: dump the
# layer-10 attention output (sub_attn_10) and the block-9/10 outputs for BOTH our
# eager (default) and flash (CRISPASR_CORE_ATTN_EAGER_F32=0) paths, and compare
# each to the HF reference. If eager pushes cos 0.972 -> ~0.9999, it fixes the
# forward (stop A/B was noise). If eager stays ~0.972, flash-vs-eager is NOT the
# bug and mudler's 4.6e-05 parity comes from something else.

# %% [code]
import json, os, subprocess, sys, gc, math, shutil
from pathlib import Path

REPO = Path("/kaggle/temp/CrispASR")
WORK = Path("/kaggle/working")
REF = os.environ.get("CRISPASR_REF", "fix/249-moss")
LAYER = 10
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
step("start", ref=REF, layer=LAYER)
TOKEN = kh.resolve_hf_token("HF_TOKEN")
kh.install_build_toolchain()
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "hf_transfer"])


def robust_download(repo, fname, local_dir, token, tries=3, timeout=900):
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
            p.terminate(); p.join(); continue
        if not q.empty():
            s, v = q.get()
            if s == "ok":
                return v
    raise RuntimeError(f"download failed: {repo}/{fname}")


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


def read_sub(path):
    out = {}
    if Path(path).exists():
        import numpy as np
        for line in Path(path).read_text().splitlines():
            p = line.split()
            if p:
                out[p[0]] = np.array([float(x) for x in p[1:]], dtype=np.float64)
    return out


# ── our dumps for both attn modes (prefill of the prompt, 1 frame) ────────────
ours = {}
for mode in ("eager", "flash"):
    sub = WORK / f"sub_{mode}.txt"
    lay = WORK / f"lay_{mode}.txt"
    env = {**os.environ, "CRISPASR_MOSS_TTS_LOCAL_DUMP_SUBLAYER": str(LAYER),
           "CRISPASR_MOSS_TTS_LOCAL_DUMP_SUBLAYER_PATH": str(sub),
           "CRISPASR_MOSS_TTS_LOCAL_DUMP_LAYERS": str(lay),
           "CRISPASR_MOSS_TTS_LOCAL_MAX_FRAMES": "2"}
    if mode == "flash":
        env["CRISPASR_CORE_ATTN_EAGER_F32"] = "0"
    with kh.build_heartbeat(f"ours.{mode}"):
        try:
            subprocess.run([str(CLI), "--backend", "moss-tts-local", "-m", F16, "--codec-model", CODEC,
                            "--tts", TEXT, "--tts-output", str(WORK / f"o_{mode}.wav"), "--no-prints"],
                           env=env, timeout=600, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            pass
    d = read_sub(sub); d.update(read_sub(lay))
    ours[mode] = d
    step(f"ours.{mode}.done", keys=[k for k in d if "sub_attn" in k or k == f"blk_{LAYER}"])

# ── reference layer-10 attn output + block 9/10 ───────────────────────────────
step("ref.load")
import numpy as np  # noqa: E402
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
am = getattr(layers[LAYER], "self_attn", None) or getattr(layers[LAYER], "attn", None)
if am is not None:
    am.register_forward_hook(mk("attn_out"))
for bl in (LAYER - 1, LAYER):
    layers[bl].register_forward_hook(mk(f"blk_{bl}"))
feat = proc([proc.build_user_message(text=TEXT)], mode="generation")
input_ids = feat["input_ids"] if isinstance(feat, dict) else feat.input_ids
with kh.build_heartbeat("ref.gen"):
    model.generate(input_ids=input_ids, max_new_frames=1, do_sample=False)
del model
gc.collect()


def cos(a, b):
    if a is None or b is None or a.shape != b.shape:
        return None
    return round(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)), 6)


out = {"layer": LAYER}
for mode in ("eager", "flash"):
    out[mode] = {
        "attn_out_vs_ref": cos(ours[mode].get(f"sub_attn_{LAYER}"), ref.get("attn_out")),
        "blk10_vs_ref": cos(ours[mode].get(f"blk_{LAYER}"), ref.get(f"blk_{LAYER}")),
        "blk9_vs_ref": cos(ours[mode].get(f"blk_{LAYER-1}"), ref.get(f"blk_{LAYER-1}")),
    }
ec = out["eager"]["attn_out_vs_ref"]
fc = out["flash"]["attn_out_vs_ref"]
if ec is not None and fc is not None:
    if ec >= 0.999 and ec - fc > 0.02:
        verdict = f"EAGER FIXES THE FORWARD: attn_out cos eager {ec} vs flash {fc} -> keep eager, stop A/B was noise"
    elif abs(ec - fc) < 0.01:
        verdict = f"NO DIFFERENCE: eager {ec} ~= flash {fc} -> flash-vs-eager is NOT the bug; hunt elsewhere in the forward"
    else:
        verdict = f"PARTIAL: eager {ec} vs flash {fc}"
else:
    verdict = "inconclusive — missing dumps"
(WORK / "eager_layerdiff.json").write_text(json.dumps({"verdict": verdict, **out}, indent=2))
step("done", verdict=verdict, eager=out["eager"], flash=out["flash"])
print("DONE", verdict, flush=True)

try:
    for cand in ("/kaggle/working/.ccache", "/kaggle/temp/.ccache", str(Path.home() / ".ccache")):
        if Path(cand).exists():
            subprocess.run(f"tar cf /kaggle/working/ccache.tar -C {Path(cand).parent} {Path(cand).name}",
                           shell=True, timeout=600)
            if cand.startswith("/kaggle/working"):
                shutil.rmtree(cand, ignore_errors=True)
            break
except Exception:  # noqa: BLE001
    pass
