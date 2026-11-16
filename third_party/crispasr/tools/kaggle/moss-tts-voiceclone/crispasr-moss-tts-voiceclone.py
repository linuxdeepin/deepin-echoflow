# CrispASR — MOSS-TTS-v1.5 voice-cloning round-trip validation (#249 follow-up)
#
# The codec ENCODER + --voice wiring are in place (moss_tts_set_reference_wav*,
# CLI --voice, c_api set_voice_prompt). This kernel validates the path END-TO-END,
# self-contained (no external speaker corpus), with a closed-loop design:
#
#   R  = TTS(REF_TEXT,   no voice)         # a "target voice" clip (the reference)
#   C  = TTS(CLONE_TEXT, --voice R)        # clone R's voice onto a NEW sentence
#   B  = TTS(CLONE_TEXT, no voice)         # baseline (no cloning) for the same text
#
# Gates:
#   1. NO CRASH: the --voice path returns rc=0 and a non-silent WAV of plausible
#      length (proves the encoder + reference-splice path runs).
#   2. INTELLIGIBLE: ASR(C) reproduces CLONE_TEXT (cloning didn't wreck content).
#   3. CONDITIONING MOVED THE VOICE: speaker-embedding cosine(C, R) > cosine(B, R)
#      — the reference actually pulled the timbre toward R vs the un-cloned
#      baseline. (Speaker embeddings via Resemblyzer's VoiceEncoder, CPU.)
#
# Uses the SHIPPED GGUFs (cstr/moss-tts-v1.5-GGUF) — no re-convert. ccache under
# /kaggle/temp (usage #22). Q4_K on GPU.

import json
import os
import struct
import subprocess
import sys
import time
import traceback
import wave
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
TMP = Path("/tmp")
REPO = TMP / "CrispASR"
BUILD = REPO / "build"
MODELS = TMP / "moss-models"
WORK = Path("/kaggle/working")
RESULTS = WORK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

REF = os.environ.get("CRISPASR_REF", "feat/moss-tts-parity-diff")
HF_GGUF = os.environ.get("MOSS_TTS_GGUF_REPO", "cstr/moss-tts-v1.5-GGUF")
REF_TEXT = os.environ.get("MOSS_TTS_REF_TEXT",
                          "This is the reference speaker reading a calm sentence aloud.")
CLONE_TEXT = os.environ.get("MOSS_TTS_CLONE_TEXT",
                            "Cloning should keep my voice while changing the words entirely.")

_T0 = time.time()
PROGRESS = WORK / "progress.txt"


def log(m):
    line = f"[{round(time.time() - _T0, 1)}s] {m}"
    print(line, flush=True)
    with open(PROGRESS, "a") as f:
        f.write(line + "\n")


def wav_summary(path: Path) -> dict:
    if not path.exists():
        return {"error": "missing"}
    with wave.open(str(path), "rb") as w:
        n, sr, sw, ch = w.getnframes(), w.getframerate(), w.getsampwidth(), w.getnchannels()
        raw = w.readframes(n)
    if sw != 2:
        return {"error": f"sw={sw}"}
    pcm = struct.unpack(f"<{n * ch}h", raw)
    if ch > 1:
        pcm = pcm[::ch]
    if not pcm:
        return {"duration_s": 0.0, "rms": 0.0, "n_samples": 0, "sr": sr}
    rms = ((sum(int(x) * int(x) for x in pcm) / max(1, len(pcm))) ** 0.5) / 32768.0
    return {"duration_s": round(len(pcm) / sr, 3), "rms": round(rms, 6),
            "n_samples": len(pcm), "sr": sr}


def synth(cli, backbone, codec, text, out_wav, voice=None, timeout=2400) -> dict:
    cmd = [str(cli), "--backend", "moss-tts", "-m", backbone, "--codec-model", codec,
           "--tts", text, "--tts-output", str(out_wav), "--no-prints"]
    if voice:
        # --voice with a .wav is voice cloning → requires the consent attestation.
        cmd += ["--voice", str(voice), "--i-have-rights"]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, timeout=timeout, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
        rc, out = r.returncode, r.stdout
    except subprocess.TimeoutExpired as ex:
        rc, out = -1, f"TIMEOUT {timeout}s\n{ex.stdout or ''}"
    (RESULTS / f"{out_wav.stem}.log").write_text(out)
    return {"rc": rc, "elapsed_s": round(time.time() - t0, 1),
            "wav": wav_summary(out_wav) if out_wav.exists() else {"error": "no-wav"},
            "err_excerpt": out[-1024:], "voice": bool(voice)}


def asr(cli, wav, timeout=900) -> str:
    """Return the whisper stdout, ANSI/progress-bar stripped. Do NOT truncate or
    over-filter — word_overlap only needs the transcript words to be PRESENT, and
    the crispasr whisper CLI interleaves device/model-load noise on the same lines
    as the transcript (fireredpunc punctuation model). Keep it all."""
    import re
    if not wav.exists():
        return ""
    cmd = [str(cli), "--backend", "whisper", "-m", "auto", "--auto-download",
           "-f", str(wav), "--no-prints"]
    try:
        r = subprocess.run(cmd, timeout=timeout, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
        raw = r.stdout
        (RESULTS / f"{wav.stem}.asr.log").write_text(raw)
        raw = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", raw)          # strip ANSI
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()
                 and "━" not in ln and "eta " not in ln and "MB/s" not in ln]
        return " ".join(lines)
    except Exception as ex:  # noqa: BLE001
        return f"<asr-error: {type(ex).__name__}>"


def word_overlap(a: str, b: str) -> float:
    """Fraction of b's words present in a (loose intelligibility check)."""
    import re
    wa = set(re.findall(r"[a-z]+", a.lower()))
    wb = re.findall(r"[a-z]+", b.lower())
    if not wb:
        return 0.0
    return sum(1 for w in wb if w in wa) / len(wb)


def spk_cos(venc, wav_a: Path, wav_b: Path):
    """Speaker-embedding cosine between two WAVs via Resemblyzer."""
    from resemblyzer import preprocess_wav
    import numpy as np
    ea = venc.embed_utterance(preprocess_wav(str(wav_a)))
    eb = venc.embed_utterance(preprocess_wav(str(wav_b)))
    return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-9))


def main():
    summary = {"ref": REF, "ref_text": REF_TEXT, "clone_text": CLONE_TEXT, "gates": {}}
    log(f"clone {REF}")
    if not REPO.exists():
        subprocess.check_call(["git", "clone", "--depth", "1", "--branch", REF, "--recursive",
                               "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.init_progress()

    kh.install_build_toolchain()
    arch = kh.detect_cuda_arch()
    env = os.environ.copy()
    env["CCACHE_DIR"] = "/kaggle/temp/.ccache"
    subprocess.run(["cmake", "-G", "Ninja", "-B", str(BUILD), "-S", str(REPO),
                    "-DCMAKE_BUILD_TYPE=Release"] + list(kh.cache_and_link_flags())
                   + list(kh.cuda_build_flags(arch)), env=env, check=True, timeout=300)
    with kh.build_heartbeat("voiceclone build"):
        kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli "
                            f"-j{kh.safe_build_jobs(gpu=True)}")
    cli = BUILD / "bin" / "crispasr"
    if not cli.exists():
        cands = [c for c in BUILD.rglob("crispasr") if c.is_file() and os.access(c, os.X_OK)]
        cli = cands[0] if cands else cli
    os.environ["LD_LIBRARY_PATH"] = f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    log(f"built {cli}")

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import hf_hub_download
    tok = kh.resolve_hf_token()
    q4k = hf_hub_download(HF_GGUF, "moss-tts-v1.5-q4_k.gguf", local_dir=str(MODELS), token=tok)
    codec = hf_hub_download(HF_GGUF, "moss-tts-v1.5-codec.gguf", local_dir=str(MODELS), token=tok)
    log("downloaded shipped GGUFs")

    # R = reference clip (target voice), B = baseline, C = cloned.
    ref_wav = RESULTS / "R_reference.wav"
    base_wav = RESULTS / "B_baseline.wav"
    clone_wav = RESULTS / "C_clone.wav"

    r_R = synth(cli, q4k, codec, REF_TEXT, ref_wav)
    log(f"R (reference): rc={r_R['rc']} wav={r_R['wav']}")
    r_B = synth(cli, q4k, codec, CLONE_TEXT, base_wav)
    log(f"B (baseline):  rc={r_B['rc']} wav={r_B['wav']}")
    r_C = synth(cli, q4k, codec, CLONE_TEXT, clone_wav, voice=ref_wav)
    log(f"C (cloned):    rc={r_C['rc']} wav={r_C['wav']}")
    summary["synth"] = {"R": r_R, "B": r_B, "C": r_C}

    # Gate 1: --voice path ran cleanly and produced audio.
    def ok(res, min_dur=0.4):
        w = res["wav"]
        return res["rc"] == 0 and "error" not in w and w["duration_s"] >= min_dur and w["rms"] >= 1e-4
    g1 = ok(r_C) and ok(r_R) and ok(r_B)
    summary["gates"]["no_crash"] = "PASS" if g1 else "FAIL"
    log(f"GATE1 no_crash: {summary['gates']['no_crash']}")

    # Gate 2: cloned output is intelligible (ASR reproduces CLONE_TEXT).
    asr_C = asr(cli, clone_wav)
    asr_B = asr(cli, base_wav)
    ov_C = word_overlap(asr_C, CLONE_TEXT)
    summary["asr"] = {"C": asr_C, "B": asr_B, "C_overlap": round(ov_C, 3)}
    g2 = ov_C >= 0.5
    summary["gates"]["intelligible"] = "PASS" if g2 else "FAIL"
    log(f"GATE2 intelligible: {summary['gates']['intelligible']} (overlap={ov_C:.2f}) asr_C={asr_C!r}")

    # Gate 3: cloning moved the timbre toward R vs the baseline.
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "resemblyzer"])
        from resemblyzer import VoiceEncoder
        venc = VoiceEncoder("cpu")
        cos_CR = spk_cos(venc, clone_wav, ref_wav)
        cos_BR = spk_cos(venc, base_wav, ref_wav)
        summary["speaker"] = {"cos_clone_ref": round(cos_CR, 4), "cos_base_ref": round(cos_BR, 4),
                              "delta": round(cos_CR - cos_BR, 4)}
        g3 = cos_CR > cos_BR
        summary["gates"]["voice_moved"] = "PASS" if g3 else "FAIL"
        log(f"GATE3 voice_moved: {summary['gates']['voice_moved']} "
            f"(cos_CR={cos_CR:.3f} > cos_BR={cos_BR:.3f}? delta={cos_CR - cos_BR:+.3f})")
    except Exception as e:  # noqa: BLE001
        summary["gates"]["voice_moved"] = f"SKIP: {type(e).__name__}: {e}"
        log(f"GATE3 voice_moved SKIP: {e}")

    summary["all_pass"] = (summary["gates"].get("no_crash") == "PASS"
                           and summary["gates"].get("intelligible") == "PASS"
                           and summary["gates"].get("voice_moved") == "PASS")
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 60 + "\n" + json.dumps(summary, indent=2) + "\n" + "=" * 60)
    log(f"VOICE-CLONE VALIDATION all_pass={summary['all_pass']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        log(f"FATAL: {e}\n{traceback.format_exc()}")
        sys.exit(1)
