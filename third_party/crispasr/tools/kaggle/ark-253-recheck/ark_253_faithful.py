#!/usr/bin/env python3
"""Issue #253 — is the FAITHFUL path (match Python: no EOS-suppression, no seed)
the right behaviour? Compares C++ default vs faithful vs Whisper ground truth on:
  - jfk.wav          (clean speech — must still transcribe)
  - tot.wav          (the ORIGINAL #253 118s clip — must still transcribe, i.e.
                      does 30s windowing ALONE fix it, making the EOS hack moot?)
  - t501-3.75m.wav   (the mixed music+dialogue clip — faithful should match
                      Whisper: empty where there's no speech, no hallucinated loop)
Faithful = CRISPASR_ARKASR_NO_EOS_SUPPRESS=1 + CRISPASR_ARKASR_NO_CHUNK_CONTEXT=1.
"""
import os, sys, json, re, shutil, subprocess, urllib.request, zipfile
from pathlib import Path

TMP = Path("/kaggle/temp"); TMP.mkdir(parents=True, exist_ok=True)
REPO = TMP / "CrispASR"; BUILD = TMP / "build"
MODELS = Path("/tmp/models"); MODELS.mkdir(parents=True, exist_ok=True)
RESULTS = Path("/kaggle/working/results"); RESULTS.mkdir(parents=True, exist_ok=True)
GGUF_REPO = "cstr/ark-asr-3b-GGUF"
CLIPS = {"tot_118s": "https://github.com/user-attachments/files/29944331/tot.wav.zip",
         "t501_225s": "https://github.com/user-attachments/files/29962358/t501-3.75m.wav.zip"}

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

from huggingface_hub import hf_hub_download
TOKEN = kh.resolve_hf_token()
GGUF = hf_hub_download(GGUF_REPO, "ark-asr-3b-q8_0.gguf", local_dir=str(MODELS), token=TOKEN or None)
try: WM = hf_hub_download("ggerganov/whisper.cpp","ggml-base.en.bin",local_dir=str(MODELS),token=None)
except Exception as e: WM = None; print("whisper dl fail", e)

wavs = {"jfk": REPO/"samples"/"jfk.wav"}
for name, url in CLIPS.items():
    zp = TMP/f"{name}.zip"; urllib.request.urlretrieve(url, str(zp))
    with zipfile.ZipFile(zp) as z: z.extractall(str(TMP/name))
    wavs[name] = next((TMP/name).glob("*.wav"))
jstep("clips", **{k: v.name for k, v in wavs.items()})

def ark(wav, faithful):
    env = dict(os.environ)
    if faithful:
        env["CRISPASR_ARKASR_NO_EOS_SUPPRESS"] = "1"
        env["CRISPASR_ARKASR_NO_CHUNK_CONTEXT"] = "1"
    r = subprocess.run([str(CLI),"-m",GGUF,"--language","en","--no-punctuation","-f",str(wav)],
                       capture_output=True, text=True, timeout=3600, env=env)
    return norm(" ".join(l for l in r.stdout.splitlines()
               if l.strip() and not l.lstrip().startswith(("[","whisper","crispasr","load","main:"))))

def whisper(wav):
    if not WM: return "<no model>"
    r = subprocess.run([str(CLI),"-m",WM,"-f",str(wav),"-l","en"], capture_output=True, text=True, timeout=1200)
    return norm(" ".join(l for l in r.stdout.splitlines() if l.strip() and not l.lstrip().startswith("[")))

out = {"sha": SHA, "clips": {}}
for name, wav in wavs.items():
    jstep(f"run_{name}")
    d = {"default": ark(wav, False), "faithful": ark(wav, True), "whisper": whisper(wav)}
    out["clips"][name] = {k: {"words": len(v.split()), "text": v[:400]} for k, v in d.items()}
    for k, v in d.items():
        print(f"  [{name}] {k:9s} ({len(v.split())}w): {v[:180]}", flush=True)

(RESULTS/"results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
print("RESULTS_JSON " + json.dumps(out, ensure_ascii=False)); jstep("done")
