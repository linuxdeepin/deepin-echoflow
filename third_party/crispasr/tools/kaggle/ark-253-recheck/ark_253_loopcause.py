#!/usr/bin/env python3
"""Issue #253 loop root-cause — is the cross-chunk CONDITIONING causing the loop?

C++-only (no Python model → fast, no P100), full clip passed at native rate so
crispasr resamples properly (no crude-resample confound). Runs the reporter's
default command under three configs and captures FULL transcripts + loop score:
  A. default                         (30s windows + cross-chunk conditioning)
  B. CRISPASR_ARKASR_NO_CHUNK_CONTEXT=1   (windows WITHOUT seed prefix)
  C. CRISPASR_ARKASR_NO_EOS_SUPPRESS=1    (baseline, no EOS-at-start suppression)
If the 2:50->end loop vanishes in B, the conditioning is the cause.
"""
import os, sys, json, re, shutil, subprocess, urllib.request, zipfile
from pathlib import Path

WORK = Path("/kaggle/working"); TMP = Path("/kaggle/temp"); TMP.mkdir(parents=True, exist_ok=True)
REPO = TMP / "CrispASR"; BUILD = TMP / "build"
MODELS = Path("/tmp/models"); MODELS.mkdir(parents=True, exist_ok=True)
RESULTS = WORK / "results"; RESULTS.mkdir(parents=True, exist_ok=True)
GGUF_REPO = "cstr/ark-asr-3b-GGUF"
CLIP_URL = "https://github.com/user-attachments/files/29962358/t501-3.75m.wav.zip"

def jstep(n, **kv): print(f"[STEP] {n} " + " ".join(f"{k}={v}" for k, v in kv.items()), flush=True)
def norm(s): return re.sub(r"\s+", " ", (s or "").strip())
def loopscore(text):
    w = norm(text).split(); best, bph = 0, ""
    for k in range(1, 12):
        i = 0
        while i < len(w) - k:
            ph = tuple(w[i:i+k]); r = 1; j = i+k
            while j+k <= len(w) and tuple(w[j:j+k]) == ph: r += 1; j += k
            if r > 1 and r*k > best: best, bph = r*k, " ".join(ph)
            i += 1
    return best, bph[:80]

if REPO.exists(): shutil.rmtree(REPO)
subprocess.check_call(["git","clone","--depth","1","--recursive",CRISPASR_REPO := "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
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
zp = TMP/"t501.zip"; urllib.request.urlretrieve(CLIP_URL, str(zp))
with zipfile.ZipFile(zp) as z: z.extractall(str(TMP))
WAV = next(TMP.glob("t501*.wav")); jstep("ready", wav=WAV.name)

def run(tag, extra_env):
    r = subprocess.run([str(CLI),"-m",GGUF,"--language","en","--no-punctuation","-f",str(WAV)],
                       capture_output=True, text=True, timeout=5400, env={**os.environ, **extra_env})
    txt = norm(" ".join(l for l in r.stdout.splitlines()
               if l.strip() and not l.lstrip().startswith(("[","whisper","crispasr","load","main:"))))
    (RESULTS/f"{tag}.txt").write_text(txt)
    ls, ph = loopscore(txt)
    return {"rc": r.returncode, "n_words": len(txt.split()), "loop_words": ls, "loop_phrase": ph, "text": txt}

out = {"sha": SHA}
cfgs = [("A_default", {}),
        ("B_no_chunk_context", {"CRISPASR_ARKASR_NO_CHUNK_CONTEXT": "1"}),
        ("C_no_eos_suppress", {"CRISPASR_ARKASR_NO_EOS_SUPPRESS": "1"})]
for tag, env in cfgs:
    jstep(f"run_{tag}")
    res = run(tag, env)
    out[tag] = {k: v for k, v in res.items() if k != "text"}
    out[tag]["text_tail"] = res["text"][-260:]
    jstep(f"done_{tag}", words=res["n_words"], loop=res["loop_words"], phrase=res["loop_phrase"])

(RESULTS/"results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
print("\n===== RESULTS ====="); print(json.dumps(out, ensure_ascii=False, indent=2))
print("RESULTS_JSON " + json.dumps(out, ensure_ascii=False)); jstep("done")
