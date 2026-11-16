#!/usr/bin/env python3
"""Issue #253 — the missing CONTROL. Does the canonical Python reference
(tools/reference_backends/arkasr.py) transcribe CLEAN speech (jfk.wav)?

If Python transcribes jfk but is empty on t501 -> my invocation is fine and the
t501 empties are real (the model declines that content) -> the C++ is
HALLUCINATING (forced output). If Python is empty on jfk too -> the Python path
is broken. Whisper on t501-win0 gives ground truth for what's actually said.
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

import numpy as np, soundfile as sf
from huggingface_hub import hf_hub_download, snapshot_download
TOKEN = kh.resolve_hf_token()
GGUF = hf_hub_download(GGUF_REPO, "ark-asr-3b-q8_0.gguf", local_dir=str(MODELS), token=TOKEN or None)
snapshot_download(ORIG_REPO, local_dir=str(REFMODEL), token=TOKEN or None,
                  allow_patterns=["*.json","*.py","*.safetensors","*.txt","*.model","tokenizer*","merges*","vocab*"])

def load16(path):
    a, sr = sf.read(str(path))
    if a.ndim > 1: a = a.mean(1)
    a = a.astype(np.float32)
    if sr != 16000:
        from scipy.signal import resample_poly; from math import gcd
        g = gcd(16000, sr); a = resample_poly(a, 16000//g, sr//g).astype(np.float32)
    return a

JFK = load16(REPO/"samples"/"jfk.wav")                       # clean English speech control
zp = TMP/"t501.zip"; urllib.request.urlretrieve(CLIP_URL, str(zp))
with zipfile.ZipFile(zp) as z: z.extractall(str(TMP))
T501 = load16(next(TMP.glob("t501*.wav")))
T501_W0 = T501[:30*16000]
sf.write(str(TMP/"jfk16.wav"), JFK, 16000); sf.write(str(TMP/"t501_w0.wav"), T501_W0, 16000)
jstep("audio", jfk_s=round(len(JFK)/16000,1), t501w0_s=round(len(T501_W0)/16000,1))

# --- canonical Python reference (arkasr.py dump) ---
sys.path.insert(0, str(REPO/"tools"))
from reference_backends import arkasr
def py(a):
    try:
        r = arkasr.dump(model_dir=REFMODEL, audio=np.asarray(a,dtype=np.float32),
                        stages={"generated_text"}, max_new_tokens=256)
        return norm(r.get("generated_text",""))
    except Exception as e:
        import traceback; traceback.print_exc(); return f"<ERR {e}>"

def cpp(wav):
    r = subprocess.run([str(CLI),"-m",GGUF,"--language","en","--no-punctuation","-f",str(wav)],
                       capture_output=True, text=True, timeout=1200,
                       env={**os.environ,"CRISPASR_ARKASR_MAX_SINGLE_PASS_S":"60"})
    return norm(" ".join(l for l in r.stdout.splitlines()
               if l.strip() and not l.lstrip().startswith(("[","whisper","crispasr","load","main:"))))

# --- whisper ground-truth on t501-win0 (via crispasr's whisper) ---
def whisper(wav):
    try:
        wm = hf_hub_download("ggerganov/whisper.cpp","ggml-base.en.bin",local_dir=str(MODELS),token=None)
    except Exception:
        return "<no whisper model>"
    r = subprocess.run([str(CLI),"-m",wm,"-f",str(wav),"-l","en"], capture_output=True, text=True, timeout=600)
    return norm(" ".join(l for l in r.stdout.splitlines() if l.strip() and not l.lstrip().startswith("[")))

out = {"sha": SHA}
jstep("py_jfk"); out["py_jfk"] = py(JFK);         print("PY  jfk       :", out["py_jfk"][:260], flush=True)
jstep("cpp_jfk"); out["cpp_jfk"] = cpp(TMP/"jfk16.wav"); print("CPP jfk       :", out["cpp_jfk"][:260], flush=True)
jstep("py_t501w0"); out["py_t501w0"] = py(T501_W0); print("PY  t501-win0 :", out["py_t501w0"][:260], flush=True)
jstep("cpp_t501w0"); out["cpp_t501w0"] = cpp(TMP/"t501_w0.wav"); print("CPP t501-win0 :", out["cpp_t501w0"][:260], flush=True)
jstep("whisper_t501w0"); out["whisper_t501w0"] = whisper(TMP/"t501_w0.wav"); print("WHISPER win0  :", out["whisper_t501w0"][:260], flush=True)

(RESULTS/"results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
print("RESULTS_JSON " + json.dumps(out, ensure_ascii=False)); jstep("done")
