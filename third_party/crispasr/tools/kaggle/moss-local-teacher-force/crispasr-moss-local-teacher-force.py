# %% [markdown]
# # MOSS-TTS-Local 4B — TEACHER-FORCING diff: bug vs precision (#249)
#
# frame-0 stop logits already match (ours 10.44 vs ref 10.17), so no gross
# forward bug. Ours winds down at ~55, the reference at ~15. This settles it:
# feed the reference's OWN generated frames into our C++ and compare the stop
# logit at each frame.
#   our[f] ~= ref[f] for all f  -> forward is CORRECT; divergence is the sampled
#                                  audio codes / f16-vs-f32 precision (not a bug).
#   our[f] diverges at some f   -> a real forward bug processing the audio
#                                  feedback, at that frame -> fixable in code.
#
# Reference (float32) first, extract frames+logits, free RAM, then our C++
# subprocess teacher-forced. CPU.

# %% [code]
import json, os, re, subprocess, sys, gc
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = Path("/kaggle/temp/CrispASR")
REF = os.environ.get("CRISPASR_REF", "fix/249-moss")
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
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download  # noqa: E402

# ── build our instrumented C++ ────────────────────────────────────────────────
BUILD = REPO / "build"
step("cmake.configure")
subprocess.run(["cmake", "-G", "Ninja", "-B", str(BUILD), "-S", str(REPO),
                "-DCMAKE_BUILD_TYPE=Release"] + kh.crispasr_cmake_flags(), check=True)
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(f"cmake --build {BUILD} --target crispasr-cli -j{kh.safe_build_jobs(gpu=False)}")
CLI = (BUILD / "bin" / "crispasr") if (BUILD / "bin" / "crispasr").exists() else next(iter(BUILD.rglob("crispasr")))
os.environ["LD_LIBRARY_PATH"] = f"{BUILD/'src'}:{BUILD/'ggml'/'src'}:{os.environ.get('LD_LIBRARY_PATH','')}"
step("build.done", cli=str(CLI))

MODELS = Path("/kaggle/temp/models"); MODELS.mkdir(parents=True, exist_ok=True)
with kh.build_heartbeat("download.gguf"):
    F16 = hf_hub_download("cstr/moss-tts-local-v1.5-GGUF", "moss-tts-local-v1.5-f16.gguf", local_dir=str(MODELS))
    CODEC = hf_hub_download("cstr/moss-tts-local-v1.5-GGUF", "moss-tts-local-v1.5-codec.gguf", local_dir=str(MODELS))

TEXT = "Hello world."
SLOT = 151656
N_VQ = 12

# ── reference: sampled run (seed 0), capture frames + per-frame stop logits ────
step("ref.load")
import torch  # noqa: E402
from transformers import AutoModel, AutoProcessor  # noqa: E402
torch.set_num_threads(os.cpu_count() or 4)
HF = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
model = AutoModel.from_pretrained(HF, trust_remote_code=True, torch_dtype=torch.float32,
                                  attn_implementation="eager", token=TOKEN).eval()
proc = AutoProcessor.from_pretrained(HF, trust_remote_code=True)
ref_logits = []
def hook(_m, _i, o):
    t = o.detach().float().reshape(-1, o.shape[-1]); ref_logits.append([round(float(t[-1][0]), 4), round(float(t[-1][1]), 4)])
h = model.local_text_lm_head.register_forward_hook(hook)
msg = proc.build_user_message(text=TEXT)
feat = proc([msg], mode="generation")
input_ids = feat["input_ids"] if isinstance(feat, dict) else feat.input_ids
torch.manual_seed(0)
with kh.build_heartbeat("ref.generate"):
    out = model.generate(input_ids=input_ids, max_new_frames=128, do_sample=True,
                         text_temperature=1.0, temperature=1.7, top_p=0.8, top_k=25)
h.remove()
# generated frames = rows whose text channel == assistant_slot; take the n_vq codes
ids = out[0][1] if isinstance(out, list) else out
ids = ids if hasattr(ids, "shape") else ids
frames = [[int(x) for x in row[1:1 + N_VQ].tolist()] for row in ids if int(row[0]) == SLOT]
ref = [{"f": i, "cont": c, "stop": s, "gap": round(c - s, 4)} for i, (c, s) in enumerate(ref_logits)]
step("ref.done", n_frames=len(frames), n_logits=len(ref), ref_first=ref[:3], ref_tail=ref[-6:])
(WORK / "ref_frames.txt").write_text("\n".join(" ".join(str(c) for c in fr) for fr in frames) + "\n")
del model; gc.collect()

# ── our C++ teacher-forced on the reference's frames ──────────────────────────
step("ours.tf")
env = {**os.environ, "CRISPASR_MOSS_TTS_LOCAL_DUMP_STOP": "1",
       "CRISPASR_MOSS_TTS_LOCAL_FORCE_FRAMES": str(WORK / "ref_frames.txt")}
log = WORK / "ours_tf.log"
with open(log, "w") as fh, kh.build_heartbeat("ours.tf"):
    p = subprocess.run([str(CLI), "--backend", "moss-tts-local", "-m", F16, "--codec-model", CODEC,
                        "--tts", TEXT, "--tts-output", str(WORK / "tf.wav"), "--no-prints"],
                       stdout=fh, stderr=subprocess.STDOUT, env=env, timeout=1800)
logtxt = log.read_text(errors="replace")
ours = [{"f": int(f), "cont": round(float(c), 4), "stop": round(float(s), 4), "gap": round(float(g), 4)}
        for f, c, s, g in re.findall(r"DUMPSTOP frame=(\d+) cont=([\-\d.]+) stop=([\-\d.]+) gap=([\-\d.]+)", logtxt)]
step("ours.done", rc=p.returncode, n=len(ours), tail="" if ours else logtxt[-500:])

# ── compare per frame ─────────────────────────────────────────────────────────
cmp = []
maxdiff = 0.0
for f in range(min(len(ref), len(ours))):
    dg = round(ours[f]["gap"] - ref[f]["gap"], 4)
    maxdiff = max(maxdiff, abs(dg))
    cmp.append({"f": f, "ref_gap": ref[f]["gap"], "our_gap": ours[f]["gap"], "d": dg})
verdict = ("FORWARD CORRECT (teacher-forced gaps match -> divergence is sampled-codes/precision)"
           if maxdiff < 1.0 else
           f"FORWARD DIVERGES (max gap diff {maxdiff:.2f}) -> real forward bug")
(WORK / "tf_diff.json").write_text(json.dumps({"verdict": verdict, "maxdiff": maxdiff,
                                               "cmp": cmp, "ref": ref, "ours": ours}, indent=2))
step("diff", verdict=verdict, maxdiff=round(maxdiff, 3), cmp_tail=cmp[-8:])
print("DONE", verdict, "maxdiff", maxdiff, flush=True)
