#!/usr/bin/env python3
"""Issue #253 — is the Python ARK 'empty' a CPU-bf16 artifact? t501 is clean
dialogue (Parakeet transcribes all of it), yet ARK-Python returned empty. Test
float32 vs bfloat16 on clear-dialogue 30s windows, print token-level behaviour,
and cross-check with Parakeet on the SAME windows.
"""
import os, sys, json, re, shutil, subprocess, urllib.request, zipfile
from pathlib import Path

TMP = Path("/kaggle/temp"); TMP.mkdir(parents=True, exist_ok=True)
REPO = TMP / "CrispASR"; BUILD = TMP / "build"
MODELS = Path("/tmp/models"); MODELS.mkdir(parents=True, exist_ok=True)
REFMODEL = Path("/tmp/refmodel"); REFMODEL.mkdir(parents=True, exist_ok=True)
RESULTS = Path("/kaggle/working/results"); RESULTS.mkdir(parents=True, exist_ok=True)
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
PARA = hf_hub_download("cstr/parakeet-tdt-0.6b-v3-GGUF", "parakeet-tdt-0.6b-v3-q4_k.gguf",
                       local_dir=str(MODELS), token=TOKEN or None) if False else None
snapshot_download(ORIG_REPO, local_dir=str(REFMODEL), token=TOKEN or None,
                  allow_patterns=["*.json","*.py","*.safetensors","*.txt","*.model","tokenizer*","merges*","vocab*"])
zp = TMP/"t501.zip"; urllib.request.urlretrieve(CLIP_URL, str(zp))
with zipfile.ZipFile(zp) as z: z.extractall(str(TMP))
a, sr = sf.read(str(next(TMP.glob("t501*.wav"))))
if a.ndim>1: a=a.mean(1)
a=a.astype(np.float32)
if sr!=16000:
    from scipy.signal import resample_poly; from math import gcd
    g=gcd(16000,sr); a=resample_poly(a,16000//g,sr//g).astype(np.float32); sr=16000
jstep("audio", dur=round(len(a)/sr,1))

from transformers import AutoModelForCausalLM, AutoProcessor
proc = AutoProcessor.from_pretrained(str(REFMODEL), trust_remote_code=True)
EOS = 151645
def run_dtype(dt, windows):
    model = AutoModelForCausalLM.from_pretrained(str(REFMODEL), trust_remote_code=True,
                                                 torch_dtype=dt, device_map="cpu").eval()
    res = {}
    for wi in windows:
        seg = a[wi*30*sr:(wi+1)*30*sr]
        b = proc.apply_chat_template([{"role":"user","content":[{"type":"audio","array":seg}]}],
                                     add_generation_prompt=True, tokenize=True, return_tensors="pt")
        with torch.no_grad():
            g = model.generate(input_ids=b["input_ids"], audios=b.get("audios"),
                               attention_mask=b.get("attention_mask"), max_new_tokens=200, do_sample=False)
        new = g[0, b["input_ids"].shape[1]:].tolist()
        res[wi] = {"n_new": len(new), "first5": new[:5], "eos_first": (len(new) > 0 and new[0] == EOS),
                   "text": norm(proc.tokenizer.decode(new, skip_special_tokens=True))}
        print(f"  [{dt} win{wi}] n_new={len(new)} first5={new[:5]} eos_first={res[wi]['eos_first']} : {res[wi]['text'][:150]}", flush=True)
    del model
    import gc; gc.collect()
    return res

WINS = [3, 6]  # mid-clip dialogue + the bath-gel window (Parakeet transcribes both)
out = {"sha": SHA}
jstep("bf16"); out["bf16"] = run_dtype(torch.bfloat16, WINS)
jstep("fp32"); out["fp32"] = run_dtype(torch.float32, WINS)
# C++ on the same windows
def cpp(wi):
    seg=a[wi*30*sr:(wi+1)*30*sr]; wp=TMP/f"w{wi}.wav"; sf.write(str(wp),seg,sr)
    r=subprocess.run([str(CLI),"-m",GGUF,"--language","en","--no-punctuation","-f",str(wp)],
                     capture_output=True,text=True,timeout=900,env={**os.environ,"CRISPASR_ARKASR_MAX_SINGLE_PASS_S":"60"})
    return norm(" ".join(l for l in r.stdout.splitlines() if l.strip() and not l.lstrip().startswith(("[","whisper","crispasr","load","main:"))))
out["cpp"] = {wi: cpp(wi) for wi in WINS}
for wi in WINS: print(f"  [cpp win{wi}]: {out['cpp'][wi][:150]}", flush=True)

(RESULTS/"results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
print("RESULTS_JSON " + json.dumps(out, ensure_ascii=False)); jstep("done")
