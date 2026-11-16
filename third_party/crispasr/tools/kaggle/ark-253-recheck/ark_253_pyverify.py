#!/usr/bin/env python3
"""Issue #253 — CLEAN Python reference check. Does the ORIGINAL model transcribe
this clip, or not? (Previous head-to-head used crude decimation + never verified
the audio reached the model.) This one:
  - resamples 24k->16k with anti-aliased polyphase (scipy.resample_poly)
  - VERIFIES batch['audios'] is present + prints its shape
  - prints the FULL Python transcript per 30s window, next to the C++ output
If Python transcribes windows the C++ loops/empties, the C++ runtime is broken.
"""
import os, sys, json, re, shutil, subprocess, urllib.request, zipfile
from pathlib import Path

WORK = Path("/kaggle/working"); TMP = Path("/kaggle/temp"); TMP.mkdir(parents=True, exist_ok=True)
REPO = TMP / "CrispASR"; BUILD = TMP / "build"
MODELS = Path("/tmp/models"); MODELS.mkdir(parents=True, exist_ok=True)
REFMODEL = Path("/tmp/refmodel"); REFMODEL.mkdir(parents=True, exist_ok=True)
RESULTS = WORK / "results"; RESULTS.mkdir(parents=True, exist_ok=True)
GGUF_REPO = "cstr/ark-asr-3b-GGUF"; ORIG_REPO = "AutoArk-AI/ARK-ASR-3B"
CLIP_URL = "https://github.com/user-attachments/files/29962358/t501-3.75m.wav.zip"

def jstep(n, **kv): print(f"[STEP] {n} " + " ".join(f"{k}={v}" for k, v in kv.items()), flush=True)
def norm(s): return re.sub(r"\s+", " ", (s or "").strip())

if REPO.exists(): shutil.rmtree(REPO)
subprocess.check_call(["git","clone","--depth","1","--recursive","https://github.com/CrispStrobe/CrispASR.git",str(REPO)])
sys.path.insert(0, str(REPO/"tools"/"kaggle")); import kaggle_harness as kh; kh.init_progress()
SHA = subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip(); jstep("cloned", sha=SHA[:12])
kh.install_build_toolchain()
subprocess.check_call(["cmake","-S",str(REPO),"-B",str(BUILD),"-DCMAKE_BUILD_TYPE=Release","-DBUILD_SHARED_LIBS=ON"])
with kh.build_heartbeat("build"):
    kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli -j{kh.safe_build_jobs(gpu=True)}")
CLI = BUILD/"bin"/"crispasr"; assert CLI.exists(); jstep("built")

import torch, numpy as np, soundfile as sf
from huggingface_hub import hf_hub_download, snapshot_download
TOKEN = kh.resolve_hf_token()
GGUF = hf_hub_download(GGUF_REPO, "ark-asr-3b-q8_0.gguf", local_dir=str(MODELS), token=TOKEN or None)
snapshot_download(ORIG_REPO, local_dir=str(REFMODEL), token=TOKEN or None,
                  allow_patterns=["*.json","*.py","*.safetensors","*.txt","*.model","tokenizer*","merges*","vocab*"])
zp = TMP/"t501.zip"; urllib.request.urlretrieve(CLIP_URL, str(zp))
with zipfile.ZipFile(zp) as z: z.extractall(str(TMP))
WAV = next(TMP.glob("t501*.wav"))
a, sr = sf.read(str(WAV))
if a.ndim > 1: a = a.mean(1)
a = a.astype(np.float32)
if sr != 16000:
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(16000, sr); a = resample_poly(a, 16000 // g, sr // g).astype(np.float32); sr = 16000
sf.write(str(TMP/"t501_16k.wav"), a, sr)
jstep("audio_ready", dur=round(len(a)/sr,1), sr=sr, rms=round(float((a**2).mean()**0.5),4))

from transformers import AutoModelForCausalLM, AutoProcessor
proc = AutoProcessor.from_pretrained(str(REFMODEL), trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(str(REFMODEL), trust_remote_code=True,
                                             torch_dtype=torch.bfloat16, device_map="cpu").eval()
jstep("py_loaded")

def py_window(seg):
    batch = proc.apply_chat_template([{"role":"user","content":[{"type":"audio","array":seg}]}],
                                     add_generation_prompt=True, tokenize=True, return_tensors="pt")
    keys = {k: (tuple(v.shape) if hasattr(v,"shape") else type(v).__name__) for k,v in batch.items()}
    with torch.no_grad():
        gen = model.generate(input_ids=batch["input_ids"], audios=batch.get("audios"),
                             attention_mask=batch.get("attention_mask"), max_new_tokens=256, do_sample=False)
    new = gen[0, batch["input_ids"].shape[1]:]
    return norm(proc.tokenizer.decode(new, skip_special_tokens=True)), keys

def cpp_window(wpath):
    r = subprocess.run([str(CLI),"-m",GGUF,"--language","en","--no-punctuation","-f",str(wpath)],
                       capture_output=True, text=True, timeout=1200,
                       env={**os.environ,"CRISPASR_ARKASR_MAX_SINGLE_PASS_S":"60"})
    return norm(" ".join(l for l in r.stdout.splitlines()
               if l.strip() and not l.lstrip().startswith(("[","whisper","crispasr","load","main:"))))

WIN = 30*sr; out = {"sha": SHA, "windows": []}
for wi in range(min(8, (len(a)+WIN-1)//WIN)):
    seg = a[wi*WIN:(wi+1)*WIN]
    if len(seg) < sr: continue
    wpath = TMP/f"w{wi}.wav"; sf.write(str(wpath), seg, sr)
    py, keys = py_window(seg)
    cpp = cpp_window(wpath)
    rec = {"win": wi, "t0": wi*30, "py": py[:500], "cpp": cpp[:500]}
    if wi == 0: rec["batch_keys"] = keys
    out["windows"].append(rec)
    jstep(f"win{wi}", t0=wi*30, py_len=len(py.split()), cpp_len=len(cpp.split()))
    print(f"  [win{wi} @{wi*30}s] PY : {py[:220]}", flush=True)
    print(f"  [win{wi} @{wi*30}s] CPP: {cpp[:220]}", flush=True)

(RESULTS/"results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
print("\nBATCH_KEYS(win0):", out["windows"][0].get("batch_keys"))
print("RESULTS_JSON " + json.dumps(out, ensure_ascii=False)); jstep("done")
