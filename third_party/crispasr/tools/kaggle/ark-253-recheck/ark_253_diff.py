#!/usr/bin/env python3
"""Issue #253 ROOT CAUSE — ARK C++ port vs the original Python model, head-to-head.

Runs the SHIPPED C++ (crispasr from main) and the ORIGINAL Python model
(AutoArk-AI/ARK-ASR-3B, apply_chat_template + greedy generate) on the SAME 30 s
windows of the reporter's t501-3.75m.wav, and compares:
  - does the Python model LOOP where the C++ loops? (port bug vs model behavior)
  - do the prompt token IDs match? (C++ hand-builds; Python uses apply_chat_template)
  - special-token handling (C++ leaks endoftext/Human; Python skip_special_tokens=True)
"""
import os, sys, json, re, shutil, subprocess, urllib.request, zipfile
from pathlib import Path

WORK = Path("/kaggle/working"); TMP = Path("/kaggle/temp"); TMP.mkdir(parents=True, exist_ok=True)
REPO = TMP / "CrispASR"; BUILD = TMP / "build"
MODELS = Path("/tmp/models"); MODELS.mkdir(parents=True, exist_ok=True)
REFMODEL = Path("/tmp/refmodel"); REFMODEL.mkdir(parents=True, exist_ok=True)
RESULTS = WORK / "results"; RESULTS.mkdir(parents=True, exist_ok=True)
CRISPASR_REPO = os.environ.get("CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git")
GGUF_REPO = "cstr/ark-asr-3b-GGUF"; ORIG_REPO = "AutoArk-AI/ARK-ASR-3B"
CLIP_URL = "https://github.com/user-attachments/files/29962358/t501-3.75m.wav.zip"

def jstep(n, **kv): print(f"[STEP] {n} " + " ".join(f"{k}={v}" for k, v in kv.items()), flush=True)
def norm(s): return re.sub(r"\s+", " ", (s or "").strip())
def loopscore(text):
    w = norm(text).split(); best = 0
    for k in range(1, 9):
        i = 0
        while i < len(w) - k:
            ph = tuple(w[i:i+k]); r = 1; j = i+k
            while j+k <= len(w) and tuple(w[j:j+k]) == ph: r += 1; j += k
            if r > 1 and r*k > best: best = r*k
            i += 1
    return best  # ~#words consumed by the longest verbatim repeat

# ── build ──
if REPO.exists(): shutil.rmtree(REPO)
subprocess.check_call(["git","clone","--depth","1","--recursive",CRISPASR_REPO,str(REPO)])
sys.path.insert(0, str(REPO/"tools"/"kaggle")); import kaggle_harness as kh; kh.init_progress()
SHA = subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()
jstep("cloned", sha=SHA[:12])
kh.install_build_toolchain()
subprocess.check_call(["cmake","-S",str(REPO),"-B",str(BUILD),"-DCMAKE_BUILD_TYPE=Release","-DBUILD_SHARED_LIBS=ON"])
with kh.build_heartbeat("build"):
    kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli -j{kh.safe_build_jobs(gpu=True)}")
CLI = BUILD/"bin"/"crispasr"; assert CLI.exists(); jstep("built")

# ── downloads ──
import torch, numpy as np, soundfile as sf
from huggingface_hub import hf_hub_download, snapshot_download
TOKEN = kh.resolve_hf_token()
GGUF = hf_hub_download(GGUF_REPO, "ark-asr-3b-q8_0.gguf", local_dir=str(MODELS), token=TOKEN or None)
snapshot_download(ORIG_REPO, local_dir=str(REFMODEL), token=TOKEN or None,
                  allow_patterns=["*.json","*.py","*.safetensors","*.txt","*.model","tokenizer*","merges*","vocab*"])
zp = TMP/"t501.zip"; urllib.request.urlretrieve(CLIP_URL, str(zp))
with zipfile.ZipFile(zp) as z: z.extractall(str(TMP))
WAV0 = next(TMP.glob("t501*.wav"))
a, sr = sf.read(str(WAV0))
if a.ndim > 1: a = a.mean(1)
if sr != 16000:  # resample to 16k (Whisper encoder rate) for BOTH engines
    import math
    x = np.arange(0, len(a), sr/16000.0); x = x[x < len(a)-1].astype(np.int64)
    a = a[x].astype(np.float32); sr = 16000
jstep("ready", dur=round(len(a)/sr,1), gguf=Path(GGUF).name)

# ── load Python model ──
# CPU-only: Kaggle often assigns a P100 (sm_60) whose torch build lacks kernel
# images (cudaErrorNoKernelImageForDevice). The reference backend runs on CPU
# anyway; the C++ (ggml) keeps the GPU. bf16 ~6 GB is fine on the CPU RAM.
from transformers import AutoModelForCausalLM, AutoProcessor
dev = "cpu"
proc = AutoProcessor.from_pretrained(str(REFMODEL), trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(str(REFMODEL), trust_remote_code=True,
                                             torch_dtype=torch.bfloat16, device_map="cpu").eval()
try:
    from reference_backends._reload_guard import reload_if_random_init  # our new guard
    sys.path.insert(0, str(REPO/"tools")); reload_if_random_init(model, REFMODEL)
except Exception: pass
jstep("py_loaded", dev=dev)

WIN = 30*sr; n_win = (len(a)+WIN-1)//WIN
results = {"sha": SHA, "windows": []}
for wi in range(n_win):
    seg = a[wi*WIN:(wi+1)*WIN]
    if len(seg) < sr:  # skip <1s tail
        continue
    t0 = wi*30
    # --- Python: apply_chat_template + greedy ---
    batch = proc.apply_chat_template([{"role":"user","content":[{"type":"audio","array":seg}]}],
                                     add_generation_prompt=True, tokenize=True, return_tensors="pt")
    ids = {k: (v.to(dev) if hasattr(v,"to") else v) for k,v in batch.items()}
    with torch.no_grad():
        gen = model.generate(input_ids=ids["input_ids"], audios=ids.get("audios"),
                             attention_mask=ids.get("attention_mask"), max_new_tokens=256, do_sample=False)
    new = gen[0, batch["input_ids"].shape[1]:]
    py_clean = proc.tokenizer.decode(new, skip_special_tokens=True)
    py_raw = proc.tokenizer.decode(new, skip_special_tokens=False)
    # --- C++: single 30s pass on the same window ---
    wpath = TMP/f"win_{wi}.wav"; sf.write(str(wpath), seg, sr)
    r = subprocess.run([str(CLI),"-m",GGUF,"--language","en","--no-punctuation","-f",str(wpath)],
                       capture_output=True, text=True, timeout=1200,
                       env={**os.environ,"CRISPASR_ARKASR_MAX_SINGLE_PASS_S":"60","CRISPASR_ARKASR_DEBUG_GEN":"1"})
    cpp = norm(" ".join(l for l in r.stdout.splitlines()
                        if l.strip() and not l.lstrip().startswith(("[","whisper","crispasr","load","main:"))))
    rec = {"win": wi, "t0s": t0,
           "py": norm(py_clean)[:400], "py_loop": loopscore(py_clean),
           "cpp": cpp[:400], "cpp_loop": loopscore(cpp),
           "py_prompt_len": int(batch["input_ids"].shape[1])}
    if wi == 0:  # capture the prompt structure once for comparison
        pid = batch["input_ids"][0].tolist()
        rec["py_prompt_head"] = pid[:8]
        rec["py_prompt_tail"] = pid[-6:]
        rec["py_raw_first80"] = py_raw[:80]
    results["windows"].append(rec)
    jstep(f"win{wi}", t0=t0, py_loop=rec["py_loop"], cpp_loop=rec["cpp_loop"])

# summary
py_loops = [w for w in results["windows"] if w["py_loop"] >= 8]
cpp_loops = [w for w in results["windows"] if w["cpp_loop"] >= 8]
results["summary"] = {"n_windows": len(results["windows"]),
                      "py_looping_windows": [w["win"] for w in py_loops],
                      "cpp_looping_windows": [w["win"] for w in cpp_loops]}
(RESULTS/"results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
print("\n===== RESULTS ====="); print(json.dumps(results, ensure_ascii=False, indent=2))
print("RESULTS_JSON " + json.dumps(results, ensure_ascii=False))
jstep("done")
