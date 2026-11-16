#!/usr/bin/env python3
"""Issue #253 re-check — ARK ASR on the reporter's 3.75-min real-world clip.

The original two bugs (empty on 118s, scattered dropped windows + stray "p")
are fixed on main (add0df21: cap single-pass at 30s + suppress EOS-at-start).
The reporter's FOLLOW-UP raised new symptoms on t501-3.75m.wav (225s):
  - missing content at 1:03 / 1:29 / 2:01
  - a repetition LOOP from ~2:50 to the end (everything after lost)
  - with --chunk-seconds 7: "endoftextHuman" leaking + other-language chars

This kernel reproduces both commands on the current main build with q8_0 (the
cleanest quant), captures per-window [ark-gen] debug, and analyses the output
for loops / empty windows / special-token leaks / non-Latin hallucinations.
"""
import os, sys, json, re, shutil, subprocess, urllib.request, zipfile
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp"); TMP.mkdir(parents=True, exist_ok=True)
REPO = TMP / "CrispASR"
BUILD = TMP / "build"
MODELS = Path("/tmp/models"); MODELS.mkdir(parents=True, exist_ok=True)  # big model -> ephemeral layer (gotcha #21)
RESULTS = WORK / "results"; RESULTS.mkdir(parents=True, exist_ok=True)
CRISPASR_REPO = os.environ.get("CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git")
CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
GGUF_REPO = "cstr/ark-asr-3b-GGUF"
CLIP_URL = "https://github.com/user-attachments/files/29962358/t501-3.75m.wav.zip"

def jstep(name, **kv):
    print(f"[STEP] {name} " + " ".join(f"{k}={v}" for k, v in kv.items()), flush=True)

# ── clone + build ──────────────────────────────────────────────────────────
if REPO.exists(): shutil.rmtree(REPO)
subprocess.check_call(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF,
                       "--recursive", CRISPASR_REPO, str(REPO)])
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh
kh.init_progress()
SHA = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
ARK_SUBJ = subprocess.check_output(["git", "-C", str(REPO), "log", "-1", "--format=%h %s", "--", "src/ark_asr.cpp"], text=True).strip()
jstep("cloned", sha=SHA[:12], ark_last=ARK_SUBJ[:70])

kh.install_build_toolchain()
subprocess.check_call(["cmake", "-S", str(REPO), "-B", str(BUILD),
                       "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON"])
with kh.build_heartbeat("build"):
    kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli -j{kh.safe_build_jobs(gpu=True)}")
CLI = BUILD / "bin" / "crispasr"
jstep("built", cli=CLI.exists())
assert CLI.exists()

# ── model + clip ───────────────────────────────────────────────────────────
from huggingface_hub import hf_hub_download
TOKEN = kh.resolve_hf_token()
GGUF = hf_hub_download(GGUF_REPO, "ark-asr-3b-q8_0.gguf", local_dir=str(MODELS), token=TOKEN or None)
jstep("model_ready", gguf=Path(GGUF).name)

zp = TMP / "t501.zip"
urllib.request.urlretrieve(CLIP_URL, str(zp))
with zipfile.ZipFile(zp) as z: z.extractall(str(TMP))
WAV = next(TMP.glob("t501*.wav"))
import soundfile as sf
_a, _sr = sf.read(str(WAV))
jstep("clip_ready", wav=WAV.name, dur=round(len(_a) / _sr, 1), sr=_sr)

# ── run ARK (default + chunk-7), capture stdout/stderr/json ─────────────────
def run_ark(tag, extra_args):
    ofbase = RESULTS / f"out_{tag}"
    cmd = [str(CLI), "-m", GGUF, "--language", "en", "--no-punctuation",
           "-f", str(WAV), "-of", str(ofbase), "-ojf"] + extra_args
    env = {**os.environ, "CRISPASR_ARKASR_DEBUG_GEN": "1"}
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=5400, env=env)
    (RESULTS / f"{tag}.stdout.txt").write_text(r.stdout)
    (RESULTS / f"{tag}.stderr.txt").write_text(r.stderr)
    segs = []
    jf = Path(str(ofbase) + ".json")
    if jf.exists():
        try:
            data = json.loads(jf.read_text())
            segs = data if isinstance(data, list) else data.get("transcription", data.get("segments", []))
        except Exception as e:
            jstep(f"json_parse_fail_{tag}", err=str(e)[:100])
    # per-window [ark-gen] lines
    gen = [l for l in r.stderr.splitlines() if "[ark-gen]" in l]
    return {"rc": r.returncode, "stdout": r.stdout, "segs": segs, "gen_lines": gen}

def analyse(res):
    text = " ".join((s.get("text") or s.get("transcription") or "") for s in res["segs"]) if res["segs"] else res["stdout"]
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    # loop: longest run of an identical repeated k-word phrase
    max_rep, rep_phrase = 0, ""
    for k in (1, 2, 3, 4, 5):
        i = 0
        while i < len(words) - k:
            ph = tuple(words[i:i + k]); reps = 1; j = i + k
            while j + k <= len(words) and tuple(words[j:j + k]) == ph:
                reps += 1; j += k
            if reps > max_rep:
                max_rep, rep_phrase = reps, " ".join(ph)
            i += 1
    # special-token leaks + non-Latin
    leaks = sorted(set(re.findall(r"endoftext|<\|[^>]*\|>|im_end|Human|Assistant", text)))
    non_latin = sorted(set(re.findall(r"[^\x00-\x7F\s‘’“”–—.,!?;:'\"()-]", text)))[:20]
    empty_windows = sum(1 for l in res["gen_lines"] if "first_tok=" in l and re.search(r"raw=''", l))
    return {"n_words": len(words), "max_phrase_repeat": max_rep, "rep_phrase": rep_phrase[:60],
            "special_token_leaks": leaks, "non_latin_chars": non_latin,
            "n_windows": len(res["gen_lines"]), "empty_windows": empty_windows,
            "n_segments": len(res["segs"]), "text_tail": text[-300:]}

out = {"sha": SHA, "ark_last": ARK_SUBJ}
for tag, args in [("default", []), ("chunk7", ["--chunk-seconds", "7", "--chunk-overlap", "2"])]:
    jstep(f"run_{tag}")
    res = run_ark(tag, args)
    a = analyse(res)
    out[tag] = {"rc": res["rc"], **a}
    jstep(f"done_{tag}", rc=res["rc"], words=a["n_words"], max_rep=a["max_phrase_repeat"],
          empty_win=a["empty_windows"], leaks=a["special_token_leaks"])

(RESULTS / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
print("\n===== RESULTS =====")
print(json.dumps(out, ensure_ascii=False, indent=2))
print("RESULTS_JSON " + json.dumps(out, ensure_ascii=False))
jstep("done")
