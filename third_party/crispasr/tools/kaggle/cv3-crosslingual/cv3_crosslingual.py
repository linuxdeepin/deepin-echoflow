#!/usr/bin/env python3
"""
#304 cross-lingual validation — CosyVoice3 clone reading a language different from
the reference voice, WITHOUT the reference-language accent.

subof reported CV3 cross-lingual clones sound heavily accented. Root cause:
CrispASR always ran zero-shot (`text_ids = tokenize(voice.prompt_text) +
tokenize(target_text)`), prepending the reference's own transcript (its
language) → phonetic bias. Fix (branch fix/304-cosyvoice3-se): when a target
language (`-l`/`-tl`) differs from the reference voice's language, drop the
reference TRANSCRIPT (keep the "helpful assistant" framing + the reference SPEECH
tokens for timbre) so the target text drives the phonetics.

This kernel builds crispasr from the branch with CUDA (CV3 runs natively+correct
on CUDA — only Vulkan CPU-routes), then synthesizes:
  1. zeroshot_en_de : voice fleurs-en, GERMAN text, NO -l   (accented baseline)
  2. crossling_en_de: voice fleurs-en, GERMAN text, -l de   (should drop ref text)
  3. samelang_en_en : voice fleurs-en, ENGLISH text, -l en  (must NOT engage — guard)
  4. crossling_de_en: voice fleurs-de, ENGLISH text, -l en  (other direction)
Each output is ASR-round-tripped with whisper-large-v3-turbo using AUTO language
detection — the detected language + transcript are the objective signal (a
de-accented German clone should detect 'de' with the right words; a strongly
en-accented one may mis-detect or garble). The 4 wavs are written to
/kaggle/working for a human listen.

Follows the kaggle_harness regime (clone, heartbeat, HF token, hf_hub_download).
"""
import os, sys, subprocess, json, re, wave, array, math
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp/xl"); TMP.mkdir(parents=True, exist_ok=True)
MODELS = Path("/kaggle/temp/models"); MODELS.mkdir(parents=True, exist_ok=True)
RESULTS = WORK / "crosslingual_results.json"

HERE = Path(__file__).resolve().parent
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
BRANCH = "fix/304-cosyvoice3-se"
CLONE = Path("/kaggle/temp/CrispASR")
if not CLONE.exists():
    try:
        subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                        "--recurse-submodules", CRISPASR_URL, str(CLONE)],
                       check=True, timeout=1200)
    except Exception as e:
        print(f"clone failed: {e}", flush=True)
sys.path.insert(0, str(CLONE / "tools" / "kaggle") if CLONE.exists() else str(HERE))
import kaggle_harness as kh
kh.init_progress()


def step(name, **extra):
    kh.step(name, **extra)


def sh(cmd, timeout=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)


subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hf_transfer", "huggingface_hub"], check=False)
HF_TOKEN = kh.resolve_hf_token()
step("hf.token", present=bool(HF_TOKEN))
from huggingface_hub import hf_hub_download

CV3_REPO = "cstr/cosyvoice3-0.5b-2512-GGUF"
CV3_FILES = ["cosyvoice3-llm-q4_k.gguf", "cosyvoice3-flow-q8_0.gguf", "cosyvoice3-hift-f16.gguf",
             "cosyvoice3-s3tok-f16.gguf", "cosyvoice3-campplus-f16.gguf", "cosyvoice3-voices.gguf"]
WHISPER_REPO, WHISPER_FILE = "ggerganov/whisper.cpp", "ggml-large-v3-turbo.bin"

DE = "Für die besten Aussichten auf einen schönen Tag."
EN = "However, due to the slow communication channels."
# label, voice, text, lang-flag(list), expect_cross_lingual
CASES = [
    ("zeroshot_en_de", "fleurs-en", DE, [], False),
    ("crossling_en_de", "fleurs-en", DE, ["-l", "de"], True),
    ("samelang_en_en", "fleurs-en", EN, ["-l", "en"], False),
    ("crossling_de_en", "fleurs-de", EN, ["-l", "en"], True),
]


def build_crispasr():
    kh.install_build_toolchain()
    src = CLONE
    if not (src / "ggml" / "CMakeLists.txt").exists():
        sh(f"cd {src} && git submodule update --init ggml", timeout=900)
    flags = (["-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF", "-DGGML_AVX2=ON",
              "-DGGML_FMA=ON", "-DGGML_F16C=ON", "-DCRISPASR_OPUS_FETCH=ON"]
             + kh.cuda_build_flags() + kh.cache_and_link_flags())
    step("build.configure", arch=kh.detect_cuda_arch())
    with kh.build_heartbeat("build.cmake", 30):
        r = sh(f"cd {src} && cmake -S . -B build-cuda -G Ninja " + " ".join(flags), timeout=1200)
    if r.returncode != 0:
        step("build.configure_FAILED", err=r.stderr[-2000:])
        raise RuntimeError("cmake configure failed")
    jobs = kh.safe_build_jobs(gpu=True)
    step("build.compile", jobs=jobs)
    with kh.build_heartbeat("build.ninja", 30):
        try:
            kh.sh_with_progress(f"cmake --build build-cuda -j{jobs} --target crispasr", cwd=str(src))
        except Exception as e:
            step("build.compile_FAILED", err=str(e)[-2000:])
            raise
    binp = src / "build-cuda" / "bin" / "crispasr"
    if not binp.exists():
        raise RuntimeError("crispasr binary not produced")
    os.environ["LD_LIBRARY_PATH"] = str(binp.parent) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    ver = sh(f"{binp} --version")
    step("build.ready", backends=[l.split(":")[-1].strip() for l in ver.stdout.splitlines() if "backends" in l])
    try:
        kh.export_ccache_tar()
    except Exception:
        pass
    return str(binp)


def fetch_models():
    for f in CV3_FILES:
        with kh.build_heartbeat(f"models.{f}", 30):
            hf_hub_download(repo_id=CV3_REPO, filename=f, local_dir=str(MODELS), token=HF_TOKEN or None)
    whisp = hf_hub_download(repo_id=WHISPER_REPO, filename=WHISPER_FILE, local_dir=str(MODELS), token=HF_TOKEN or None)
    return str(MODELS / CV3_FILES[0]), whisp


def wav_stats(p):
    try:
        w = wave.open(p, "rb"); a = array.array("h"); a.frombytes(w.readframes(w.getnframes()))
        if not len(a):
            return dict(dur=0, peak=0, rms=0)
        return dict(dur=round(w.getnframes() / w.getframerate(), 2),
                    peak=round(max(abs(x) for x in a) / 32768.0, 4),
                    rms=round(math.sqrt(sum((x / 32768.0) ** 2 for x in a) / len(a)), 4))
    except Exception as e:
        return dict(err=str(e))


def asr_autodetect(binp, whisp, wav):
    # whisper with -l auto → prints "auto-detected language: xx (p=..)" to stderr
    r = sh(f"{binp} -m {whisp} -f {wav} -l auto --no-gpu", timeout=300)
    out = r.stdout + "\n" + r.stderr
    lang = None
    m = re.search(r"auto-detected language[:\s]+([a-z]{2,3})", out, re.I)
    if m:
        lang = m.group(1).lower()
    txt = " ".join(re.sub(r"\[[^\]]*\]", "", l) for l in r.stdout.splitlines() if l.strip().startswith("["))
    return dict(detected_lang=lang, text=re.sub(r"\s+", " ", txt).strip())


def main():
    binp = build_crispasr()
    llm, whisp = fetch_models()
    results = {"cases": {}}
    for label, voice, text, langflag, expect_xl in CASES:
        out = TMP / f"{label}.wav"
        cmd = [binp, "--backend", "cosyvoice3-tts", "-m", llm, "--voice", voice,
               "--tts", text, "--seed", "42", "--tts-output", str(out)] + langflag
        with kh.build_heartbeat(f"synth.{label}", 30):
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        xl_engaged = "cross-lingual" in (r.stdout + r.stderr).lower()
        wpath = WORK / f"{label}.wav"
        if out.exists():
            wpath.write_bytes(out.read_bytes())  # surface for a human listen
        asr = asr_autodetect(binp, whisp, str(out)) if out.exists() else {}
        rec = dict(voice=voice, text=text, langflag=" ".join(langflag) or "(none)",
                   expect_cross_lingual=expect_xl, cross_lingual_engaged=xl_engaged,
                   rc=r.returncode, audio=wav_stats(str(out)), asr=asr,
                   gate_ok=(xl_engaged == expect_xl))
        results["cases"][label] = rec
        step(f"case.{label}", xl=xl_engaged, expect=expect_xl, gate_ok=rec["gate_ok"],
             lang=asr.get("detected_lang"), asr=(asr.get("text") or "")[:70])
        RESULTS.write_text(json.dumps(results, indent=2))
    results["all_gates_ok"] = all(c["gate_ok"] for c in results["cases"].values())
    step("DONE", all_gates_ok=results["all_gates_ok"],
         summary={k: {"xl": v["cross_lingual_engaged"], "lang": v["asr"].get("detected_lang"),
                      "asr": (v["asr"].get("text") or "")[:60]} for k, v in results["cases"].items()})
    RESULTS.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        step("FATAL", err=str(e)[:300])
        traceback.print_exc()
        sys.exit(1)
