# CrispASR — MOSS-TTS-v1.5 Kaggle validation (#249, Phase 6)
#
# The 8B backbone + 1.6B codec won't fit the 8 GB VPS and are tight on the 16 GB
# Mac, so end-to-end validation runs here on a CUDA box (P100/T4).
#
# Flow:
#   1. Clone CrispASR @ CRISPASR_REF (default feat/moss-tts-249), build CUDA
#      (crispasr-cli + crispasr-quantize), warm ccache from the dataset.
#   2. Download MOSS-TTS-v1.5 + MOSS-Audio-Tokenizer to /tmp (~70 GB layer;
#      NEVER /kaggle/working which is ~20 GB — kaggle_usage #18/#21).
#   3. Convert -> F16 backbone + F16 codec (under /tmp). Free the HF snapshot.
#   4. Quantize backbone -> Q4_K.
#   5. GATING: decoded round-trip on F16 AND Q4_K — synthesize a short + a long
#      text, verify non-silent + plausible length, ASR each with whisper, and a
#      proof-of-work check that the long clip yields more words than the short
#      (kaggle_usage #24: an RTF/round-trip is a lie until you prove the work).
#   6. BEST-EFFORT (non-gating): greedy code parity vs the HF reference dumper.
#
# Keep /kaggle/working minimal (progress.txt + results/*.json + a couple WAVs);
# stage everything large under /tmp.

import glob
import json
import os
import struct
import subprocess
import sys
import time
import traceback
import wave
from datetime import datetime, timezone
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

WORK = Path("/kaggle/working")
TMP = Path("/tmp")
REPO = TMP / "CrispASR"                 # clone under /tmp (keeps /working small)
BUILD = REPO / "build"
MODELS = TMP / "moss-models"            # HF snapshots + GGUFs live on the big layer
RESULTS = WORK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

CRISPASR_REF = os.environ.get("CRISPASR_REF", "feat/moss-tts-249")
CRISPASR_REPO = os.environ.get("CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git")
HF_MODEL = os.environ.get("MOSS_TTS_MODEL", "OpenMOSS-Team/MOSS-TTS-v1.5")
HF_CODEC = os.environ.get("MOSS_TTS_CODEC", "OpenMOSS-Team/MOSS-Audio-Tokenizer")

PROGRESS = WORK / "progress.txt"
_T0 = time.time()


def log(msg):
    line = f"[{round(time.time() - _T0, 1)}s] {msg}"
    print(line, flush=True)
    with open(PROGRESS, "a") as f:
        f.write(line + "\n")


def run(cmd, check=True, timeout=None, env=None, cwd=None, capture=True):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    e = os.environ.copy()
    if env:
        e.update(env)
    kw = dict(env=e, cwd=cwd, timeout=timeout, text=True)
    if capture:
        kw.update(stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    r = subprocess.run(cmd, **kw)
    if capture and r.stdout:
        print(r.stdout[-4000:], flush=True)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed (rc={r.returncode}): {cmd}")
    return r


# ── WAV / ASR helpers (from dots-cuda-verify) ──────────────────────────────
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


def asr_roundtrip(cli: Path, wav: Path, timeout=900) -> str:
    if not wav.exists():
        return ""
    cmd = [str(cli), "--backend", "whisper", "-m", "auto", "--auto-download",
           "-f", str(wav), "--no-prints"]
    try:
        r = subprocess.run(cmd, timeout=timeout, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
        (RESULTS / f"{wav.stem}.asr.log").write_text(r.stdout)
        lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
        text = " ".join(ln for ln in lines
                        if not ln.startswith(("[", "whisper", "ggml", "load", "crispasr")))
        return text[-400:]
    except Exception as ex:  # noqa: BLE001
        return f"<asr-error: {type(ex).__name__}>"


def synth(cli: Path, backbone: str, codec: str, text: str, out_wav: Path, timeout=2400) -> dict:
    cmd = [str(cli), "--backend", "moss-tts", "-m", backbone, "--codec-model", codec,
           "--tts", text, "--tts-output", str(out_wav), "--no-prints"]
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
            "err_excerpt": out[-1024:]}


def verdict(res: dict, min_dur: float) -> str:
    if res["rc"] != 0:
        return f"FAIL: rc={res['rc']}"
    w = res["wav"]
    if "error" in w:
        return f"FAIL: wav {w['error']}"
    if w["duration_s"] < min_dur:
        return f"FAIL: too short ({w['duration_s']}s < {min_dur})"
    if w["rms"] < 1e-4:
        return f"FAIL: silent (rms={w['rms']})"
    return "PASS"


SHORT_TEXT = "Hello world."
LONG_TEXT = ("The quick brown fox jumps over the lazy dog. "
             "Speech synthesis should stay intelligible over a longer passage, "
             "so this sentence exercises many autoregressive steps and the codec "
             "sliding window well past the first few frames.")


def main():
    summary = {"ts": datetime.now(timezone.utc).isoformat(), "ref": CRISPASR_REF,
               "phases": {}, "roundtrip": {}, "gates": {}}

    # ── 1. clone + build ───────────────────────────────────────────────────
    log(f"clone {CRISPASR_REF}")
    if not REPO.exists():
        run(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF, "--recursive",
             CRISPASR_REPO, str(REPO)])
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.init_progress()
    sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    summary["sha"] = sha
    log(f"cloned {sha}")

    run(["nvidia-smi", "-L"], check=False)
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
    summary["gpu"] = gpu
    log(f"gpu {gpu}")

    kh.install_build_toolchain()
    arch = kh.detect_cuda_arch()
    log(f"cuda_arch {arch}")
    env = os.environ.copy()
    env["CCACHE_DIR"] = "/kaggle/working/.ccache"
    cmake_args = (["cmake", "-G", "Ninja", "-B", str(BUILD), "-S", str(REPO),
                   "-DCMAKE_BUILD_TYPE=Release"]
                  + list(kh.cache_and_link_flags()) + list(kh.cuda_build_flags(arch)))
    run(cmake_args, env=env, timeout=300)
    jobs = kh.safe_build_jobs(gpu=True)
    with kh.build_heartbeat("moss-tts CUDA build"):
        kh.sh_with_progress(
            f"stdbuf -oL -eL cmake --build {BUILD} "
            f"--target crispasr-cli crispasr-quantize -j{jobs}")
    cli = BUILD / "bin" / "crispasr"
    if not cli.exists():
        cands = [c for c in BUILD.rglob("crispasr") if c.is_file() and os.access(c, os.X_OK)]
        cli = cands[0] if cands else cli
    quant = BUILD / "bin" / "crispasr-quantize"
    if not cli.exists() or not quant.exists():
        raise SystemExit(f"binaries missing: cli={cli.exists()} quant={quant.exists()}")
    os.environ["LD_LIBRARY_PATH"] = f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    summary["phases"]["build"] = "ok"
    log("build ok")

    # ── 2. download + 3. convert ───────────────────────────────────────────
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "huggingface_hub", "safetensors", "gguf"])
    hf_token = kh.resolve_hf_token()
    from huggingface_hub import snapshot_download
    log("download MOSS-TTS-v1.5 + codec -> /tmp")
    src = snapshot_download(HF_MODEL, cache_dir=str(MODELS / "hf"), token=hf_token,
                            allow_patterns=["*.safetensors", "*.json", "merges.txt",
                                            "vocab.json", "tokenizer.json", "added_tokens.json"])
    codec_src = snapshot_download(HF_CODEC, cache_dir=str(MODELS / "hf"), token=hf_token,
                                  allow_patterns=["*.safetensors", "*.json"])
    f16 = MODELS / "moss-tts-v1.5-f16.gguf"
    codec_gguf = MODELS / "moss-tts-v1.5-f16-codec.gguf"
    log("convert -> f16 backbone + codec")
    run([sys.executable, str(REPO / "models" / "convert-moss-tts-to-gguf.py"),
         "--input", src, "--codec", codec_src, "--output", str(f16),
         "--codec-output", str(codec_gguf)], timeout=3600)
    summary["phases"]["convert"] = {"f16_gb": round(f16.stat().st_size / 1e9, 2),
                                    "codec_gb": round(codec_gguf.stat().st_size / 1e9, 2)}
    log(f"converted: {summary['phases']['convert']}")

    # Free the HF snapshot before quantizing (disk headroom, kaggle_usage #18).
    import shutil
    shutil.rmtree(MODELS / "hf", ignore_errors=True)

    # ── 4. quantize backbone -> Q4_K (codec stays F16) ─────────────────────
    q4k = MODELS / "moss-tts-v1.5-q4_k.gguf"
    log("quantize -> q4_k")
    run([str(quant), str(f16), str(q4k), "q4_k"], timeout=1800)
    summary["phases"]["quantize"] = {"q4k_gb": round(q4k.stat().st_size / 1e9, 2)}
    log(f"quantized: {summary['phases']['quantize']}")

    # ── 5. GATING round-trip on F16 AND Q4_K ───────────────────────────────
    for tag, backbone in (("f16", str(f16)), ("q4_k", str(q4k))):
        rt = {}
        rs = synth(cli, backbone, str(codec_gguf), SHORT_TEXT, RESULTS / f"{tag}_short.wav")
        rs["verdict"] = verdict(rs, min_dur=0.2)
        rs["asr"] = asr_roundtrip(cli, RESULTS / f"{tag}_short.wav")
        log(f"{tag} short: {rs['verdict']} rc={rs['rc']} wav={rs['wav']} asr={rs['asr']!r}")

        rl = synth(cli, backbone, str(codec_gguf), LONG_TEXT, RESULTS / f"{tag}_long.wav")
        rl["verdict"] = verdict(rl, min_dur=2.0)
        rl["asr"] = asr_roundtrip(cli, RESULTS / f"{tag}_long.wav")
        log(f"{tag} long: {rl['verdict']} rc={rl['rc']} wav={rl['wav']} asr={rl['asr']!r}")

        # Proof-of-work (#24): the long clip must yield more ASR words + more
        # audio than the short one, else "audio" may be a fixed-time no-op.
        short_words = len((rs["asr"] or "").split())
        long_words = len((rl["asr"] or "").split())
        pow_ok = (rl["verdict"] == "PASS" and rs["verdict"] == "PASS"
                  and long_words > short_words
                  and rl["wav"].get("duration_s", 0) > rs["wav"].get("duration_s", 0))
        rt = {"short": rs, "long": rl, "short_words": short_words,
              "long_words": long_words, "proof_of_work": pow_ok}
        summary["roundtrip"][tag] = rt
        summary["gates"][f"roundtrip_{tag}"] = ("PASS" if pow_ok else "FAIL")
        log(f"{tag} gate: {summary['gates'][f'roundtrip_{tag}']} "
            f"(short_words={short_words} long_words={long_words})")

    # ── 6. BEST-EFFORT greedy code parity vs the HF reference ──────────────
    try:
        log("code parity: HF reference greedy generate")
        ref_dir = MODELS / "ref"
        ref_dir.mkdir(exist_ok=True)
        renv = os.environ.copy()
        renv["MOSS_TTS_MODEL"] = HF_MODEL
        renv["MOSS_TTS_TEXT"] = SHORT_TEXT
        renv["MOSS_TTS_SEED"] = "0"
        renv["MOSS_TTS_MAXNEW"] = "256"
        r = subprocess.run(
            [sys.executable, "-c",
             f"import sys; sys.path.insert(0, r'{REPO / 'tools' / 'reference_backends'}');"
             f"import moss_tts as m; m.run(out_dir=r'{ref_dir}')"],
            env=renv, capture_output=True, text=True, timeout=1800)
        (RESULTS / "ref_dump.log").write_text(r.stdout + "\n---STDERR---\n" + r.stderr)
        ref_codes_p = ref_dir / "codes.npy"
        if r.returncode == 0 and ref_codes_p.exists():
            import numpy as np
            ref_codes = np.load(ref_codes_p)
            summary["gates"]["code_parity"] = {"ref_shape": list(ref_codes.shape),
                                               "note": "C++ greedy-codes compare TODO (ctypes "
                                                       "moss_tts_generate_codes) — ref captured"}
            log(f"ref codes shape={ref_codes.shape}")
        else:
            summary["gates"]["code_parity"] = {"status": "ref-dump-failed",
                                               "stderr_tail": r.stderr[-600:]}
            log("code parity ref dump failed (non-gating) — see ref_dump.log")
    except Exception as e:  # noqa: BLE001
        summary["gates"]["code_parity"] = {"status": f"error: {type(e).__name__}: {e}"}
        log(f"code parity skipped: {e}")

    # ── summary ────────────────────────────────────────────────────────────
    summary["all_gates_pass"] = all(
        v == "PASS" for k, v in summary["gates"].items() if k.startswith("roundtrip_"))
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 60)
    print(json.dumps(summary, indent=2))
    print("=" * 60)
    if not summary["all_gates_pass"]:
        log("ROUND-TRIP GATE FAILED — see results/ logs")
        sys.exit(1)
    log(f"MOSS-TTS-v1.5 PASSES round-trip on {gpu} (F16 + Q4_K)")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        log(f"FATAL: {e}")
        log(traceback.format_exc())
        sys.exit(1)
