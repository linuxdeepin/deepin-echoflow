# %% [markdown]
# # MOSS-TTS cross-lingual check (#249): does -l <lang> make it speak the target?
#
# subof: cross-lingual clones sound heavily accented because SubtitleEdit can't
# pass a target language. The engine has the mechanism (-l -> the "- Language:"
# prompt field); the fix plumbs it through the server /v1/audio/speech `language`
# field. Validate the ENGINE here (prebuilt v0.8.23, no build): synth German text
# with and without -l de, ASR each (whisper auto-detect), and confirm -l de yields
# German. Also CosyVoice3 cross-lingual (already fixed on #304) as a cross-check.

# %% [code]
import json, os, re, subprocess, sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
if not REPO.exists():
    subprocess.check_call(["git", "clone", "--depth", "1",
                           "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402
kh.init_progress(hf_progress_repo="cstr/crispasr-kaggle-progress")
step = kh.step
step("start")
TOKEN = kh.resolve_hf_token("HF_TOKEN")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "hf_transfer"])
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download  # noqa: E402

BIN = WORK / "bin"; BIN.mkdir(exist_ok=True)
CLI = BIN / "crispasr"
subprocess.check_call(
    "wget -q https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.23/crispasr-linux-x86_64.tar.gz "
    f"-O /tmp/c.tgz && tar -xzf /tmp/c.tgz -C {BIN} --strip-components=1", shell=True)
CLI.chmod(0o755)
os.environ["LD_LIBRARY_PATH"] = f"{BIN}:{os.environ.get('LD_LIBRARY_PATH','')}"
M = Path("/kaggle/temp/models"); M.mkdir(parents=True, exist_ok=True)
with kh.build_heartbeat("download"):
    MOSS = hf_hub_download("cstr/moss-tts-v1.5-GGUF", "moss-tts-v1.5-q6_k.gguf", local_dir=str(M))
    MOSS_CODEC = hf_hub_download("cstr/moss-tts-v1.5-GGUF", "moss-tts-v1.5-codec.gguf", local_dir=str(M))
    WHISPER = hf_hub_download("ggerganov/whisper.cpp", "ggml-large-v3-turbo.bin", local_dir=str(M))

GERMAN = "Guten Morgen, wie geht es Ihnen heute?"


def synth(backend, model, codec, text, lang, out):
    cmd = [str(CLI), "--backend", backend, "-m", model, "--codec-model", codec, "--tts", text,
           "--tts-output", str(out), "--no-prints"]
    if lang:
        cmd += ["-l", lang]
    with kh.build_heartbeat(f"synth.{out.stem}"):
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    return r.returncode, out.exists() and out.stat().st_size > 20000


def asr(wav):
    # whisper auto-detect: transcript + detected language (printed to stderr/stdout)
    r = subprocess.run([str(CLI), "-m", WHISPER, "-f", str(wav), "-l", "auto", "-pp"],
                       capture_output=True, text=True, timeout=600)
    out = (r.stdout or "") + (r.stderr or "")
    lang = None
    m = re.search(r"auto-detected language:\s*([a-z]{2})|lang[=:]\s*([a-z]{2})|detected language:\s*([a-z]{2})", out, re.I)
    if m:
        lang = next(g for g in m.groups() if g)
    txt = " ".join(ln.strip() for ln in (r.stdout or "").splitlines()
                   if ln.strip() and not ln.startswith("[") and "-->" not in ln)
    return lang, txt[:200]


def has_german(t):
    return any(c in t for c in "äöüßÄÖÜ") or any(w in t.lower() for w in ("guten", "morgen", "wie", "geht", "ihnen"))


results = {}
for tag, lang in (("no_lang", None), ("l_de", "de")):
    wav = WORK / f"moss_{tag}.wav"
    rc, ok = synth("moss-tts", MOSS, MOSS_CODEC, GERMAN, lang, wav)
    detected, txt = (asr(wav) if ok else (None, ""))
    results[tag] = {"synth_ok": ok, "rc": rc, "detected_lang": detected, "asr": txt, "german": has_german(txt)}
    step(f"moss.{tag}", **results[tag])

verdict = ("-l de MAKES MOSS SPEAK GERMAN"
           if results.get("l_de", {}).get("german") and results["l_de"].get("detected_lang") == "de"
           else "inconclusive/failed — see results")
(WORK / "xl_check.json").write_text(json.dumps({"verdict": verdict, "results": results}, indent=2))
step("done", verdict=verdict)
print("DONE", verdict, json.dumps(results)[:800], flush=True)
