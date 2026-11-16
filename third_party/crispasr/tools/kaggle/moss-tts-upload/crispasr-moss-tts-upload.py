# CrispASR — MOSS-TTS-v1.5 convert + quantize + VALIDATE + upload to HF (#249 ship)
#
# The 8B model only converts on a big box, so this Kaggle kernel produces the
# GGUFs and (gated on the Q4_K decoded round-trip passing) uploads them straight
# to HF from inside Kaggle (CPU workers have no internet; GPU workers do).
#
# Flow: clone feat/moss-tts-249 -> build CUDA -> download HF model+codec (/tmp)
#   -> convert F16 backbone+codec -> quantize Q4_K -> validate Q4_K round-trip
#   (SKIP F16 synth on <=16 GB VRAM: the 17 GB F16 backbone won't fit a P100)
#   -> on PASS, create cstr/moss-tts-v1.5-GGUF, upload F16+Q4_K backbones, F16
#   codec, and an Apache-2.0 README, verifying each landed server-side.
#
# HF uploads run on daemon threads with a join timeout + server-side verify
# (upload_file can strand in CLOSE_WAIT after the commit lands — CLAUDE.md note).

import glob
import json
import os
import struct
import subprocess
import sys
import threading
import time
import traceback
import wave
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

TMP = Path("/tmp")
REPO = TMP / "CrispASR"
BUILD = REPO / "build"
MODELS = TMP / "moss-models"
WORK = Path("/kaggle/working")
MODELS.mkdir(parents=True, exist_ok=True)

CRISPASR_REF = os.environ.get("CRISPASR_REF", "feat/moss-tts-249")
CRISPASR_REPO = os.environ.get("CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git")
HF_MODEL = os.environ.get("MOSS_TTS_MODEL", "OpenMOSS-Team/MOSS-TTS-v1.5")
HF_CODEC = os.environ.get("MOSS_TTS_CODEC", "OpenMOSS-Team/MOSS-Audio-Tokenizer")
HF_REPO = os.environ.get("MOSS_TTS_HF_REPO", "cstr/moss-tts-v1.5-GGUF")

_T0 = time.time()
PROGRESS = WORK / "progress.txt"


def log(msg):
    line = f"[{round(time.time() - _T0, 1)}s] {msg}"
    print(line, flush=True)
    with open(PROGRESS, "a") as f:
        f.write(line + "\n")


def run(cmd, timeout=None, env=None, check=True):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    e = os.environ.copy()
    if env:
        e.update(env)
    r = subprocess.run(cmd, env=e, timeout=timeout, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True)
    if r.stdout:
        print(r.stdout[-4000:], flush=True)
    if check and r.returncode != 0:
        raise SystemExit(f"failed rc={r.returncode}: {cmd}")
    return r


def wav_ok(path, min_dur):
    if not path.exists():
        return False, "no-wav"
    with wave.open(str(path), "rb") as w:
        n, sr, sw, ch = w.getnframes(), w.getframerate(), w.getsampwidth(), w.getnchannels()
        raw = w.readframes(n)
    if sw != 2 or n == 0:
        return False, f"sw={sw} n={n}"
    pcm = struct.unpack(f"<{n * ch}h", raw)[::ch]
    rms = ((sum(int(x) * int(x) for x in pcm) / max(1, len(pcm))) ** 0.5) / 32768.0
    dur = len(pcm) / sr
    if dur < min_dur:
        return False, f"short {dur:.2f}s"
    if rms < 1e-4:
        return False, f"silent rms={rms:.5f}"
    return True, f"{dur:.2f}s rms={rms:.3f}"


# ── daemon-thread HF ops (CLOSE_WAIT-safe) ─────────────────────────────────
def _run_daemon(fn, timeout_s):
    box = {}

    def _t():
        try:
            box["ret"] = fn()
        except Exception as e:  # noqa: BLE001
            box["exc"] = e
    th = threading.Thread(target=_t, daemon=True)
    th.start()
    th.join(timeout_s)
    return (not th.is_alive()), box.get("ret"), box.get("exc")


def upload_with_timeout(api, local, repo, path_in_repo, timeout_s=3600):
    log(f"upload.begin {path_in_repo} ({Path(local).stat().st_size / 1e9:.2f} GB)")
    done, _, exc = _run_daemon(
        lambda: api.upload_file(path_or_fileobj=str(local), path_in_repo=path_in_repo,
                                repo_id=repo, repo_type="model",
                                commit_message=f"Add {path_in_repo} (#249)"),
        timeout_s)
    # Verify server-side regardless of whether the client returned (hung-but-landed).
    from huggingface_hub import HfApi
    want = Path(local).stat().st_size
    for _ in range(3):
        try:
            for f in HfApi().list_repo_tree(repo, token=api.token):
                if getattr(f, "path", "") == path_in_repo:
                    got = getattr(f, "size", 0)
                    if got and abs(got - want) < max(1 << 20, want * 0.001):
                        log(f"upload.verified {path_in_repo} ({got / 1e9:.2f} GB, client_done={done})")
                        return True
        except Exception as e:  # noqa: BLE001
            log(f"verify retry: {e}")
    log(f"upload.UNVERIFIED {path_in_repo} done={done} exc={exc}")
    return False


README = """---
license: apache-2.0
base_model:
- OpenMOSS-Team/MOSS-TTS-v1.5
- OpenMOSS-Team/MOSS-Audio-Tokenizer
library_name: gguf
pipeline_tag: text-to-speech
tags: [tts, gguf, crispasr, moss-tts]
---

# MOSS-TTS-v1.5 — GGUF (for CrispASR)

GGUF conversion of [`OpenMOSS-Team/MOSS-TTS-v1.5`](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5)
(MossTTSDelay: a Qwen3-8B backbone emitting 32 RVQ audio codebooks under a delay
pattern) + its [`MOSS-Audio-Tokenizer`](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer)
1.6B transformer codec, for the [CrispASR](https://github.com/CrispStrobe/CrispASR)
`moss-tts` backend.

## Files
| File | What | Size |
|------|------|------|
| `moss-tts-v1.5-q4_k.gguf` | Q4_K backbone (default; audio tables kept F16) | ~7 GB |
| `moss-tts-v1.5-f16.gguf` | F16 backbone (needs >20 GB VRAM or CPU; for re-quant) | ~17 GB |
| `moss-tts-v1.5-codec.gguf` | F16 transformer codec companion | ~3.5 GB |

## Use
```bash
crispasr --backend moss-tts -m moss-tts-v1.5-q4_k.gguf \\
         --codec-model moss-tts-v1.5-codec.gguf \\
         --tts "Hello world." --tts-output out.wav
# or: crispasr --backend moss-tts -m auto --auto-download --tts "..."
```

Validated on CUDA (P100) by decoded round-trip (synthesize → ASR): the Q4_K
backbone produces intelligible, accurate speech end-to-end.

## License
Apache-2.0, inherited from the base MOSS-TTS-v1.5 + MOSS-Audio-Tokenizer models
(OpenMOSS-Team). This repo redistributes derived GGUF weights under the same terms.
"""

SHORT = "Hello world."
LONG = ("The quick brown fox jumps over the lazy dog. Speech synthesis should "
        "stay intelligible over a longer passage, so this sentence exercises many "
        "autoregressive steps and the codec sliding window.")


def main():
    log(f"clone {CRISPASR_REF}")
    if not REPO.exists():
        run(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF, "--recursive",
             CRISPASR_REPO, str(REPO)])
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.init_progress()
    log("cloned " + subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip())

    run(["nvidia-smi", "-L"], check=False)
    vram_mib = int(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True).strip().split("\n")[0])
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
    log(f"gpu {gpu} vram={vram_mib}MiB")

    # ── build ──
    kh.install_build_toolchain()
    arch = kh.detect_cuda_arch()
    env = os.environ.copy()
    env["CCACHE_DIR"] = "/kaggle/working/.ccache"
    run(["cmake", "-G", "Ninja", "-B", str(BUILD), "-S", str(REPO), "-DCMAKE_BUILD_TYPE=Release"]
        + list(kh.cache_and_link_flags()) + list(kh.cuda_build_flags(arch)), env=env, timeout=300)
    with kh.build_heartbeat("moss-tts build"):
        kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} "
                            f"--target crispasr-cli crispasr-quantize -j{kh.safe_build_jobs(gpu=True)}")
    cli = BUILD / "bin" / "crispasr"
    quant = BUILD / "bin" / "crispasr-quantize"
    if not cli.exists() or not quant.exists():
        raise SystemExit("binaries missing after build")
    os.environ["LD_LIBRARY_PATH"] = f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    log("build ok")

    # ── download + convert + quantize ──
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "huggingface_hub", "safetensors", "gguf"])
    hf_token = kh.resolve_hf_token()
    from huggingface_hub import snapshot_download, HfApi
    if not hf_token:
        raise SystemExit("no HF token (need write access to upload)")
    src = snapshot_download(HF_MODEL, cache_dir=str(MODELS / "hf"), token=hf_token,
                            allow_patterns=["*.safetensors", "*.json", "merges.txt", "vocab.json",
                                            "tokenizer.json", "added_tokens.json"])
    codec_src = snapshot_download(HF_CODEC, cache_dir=str(MODELS / "hf"), token=hf_token,
                                  allow_patterns=["*.safetensors", "*.json"])
    f16 = MODELS / "moss-tts-v1.5-f16.gguf"
    codec = MODELS / "moss-tts-v1.5-codec.gguf"
    log("convert")
    run([sys.executable, str(REPO / "models" / "convert-moss-tts-to-gguf.py"),
         "--input", src, "--codec", codec_src, "--output", str(f16), "--codec-output", str(codec)], timeout=3600)
    import shutil
    shutil.rmtree(MODELS / "hf", ignore_errors=True)
    q4k = MODELS / "moss-tts-v1.5-q4_k.gguf"
    log("quantize q4_k")
    run([str(quant), str(f16), str(q4k), "q4_k"], timeout=1800)
    log(f"sizes f16={f16.stat().st_size/1e9:.2f} q4k={q4k.stat().st_size/1e9:.2f} codec={codec.stat().st_size/1e9:.2f}")

    # ── validate (gate): Q4_K round-trip; F16 only if VRAM headroom ──
    def synth(backbone, text, out, min_dur):
        r = subprocess.run([str(cli), "--backend", "moss-tts", "-m", str(backbone),
                            "--codec-model", str(codec), "--tts", text, "--tts-output", str(out), "--no-prints"],
                           timeout=2400, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (WORK / f"{out.stem}.log").write_text(r.stdout[-8000:])
        ok, why = wav_ok(out, min_dur)
        return (r.returncode == 0 and ok), why, r.returncode

    ok_s, why_s, rc_s = synth(q4k, SHORT, WORK / "q4k_short.wav", 0.2)
    ok_l, why_l, rc_l = synth(q4k, LONG, WORK / "q4k_long.wav", 2.0)
    log(f"q4_k short: {'PASS' if ok_s else 'FAIL'} ({why_s} rc={rc_s})")
    log(f"q4_k long:  {'PASS' if ok_l else 'FAIL'} ({why_l} rc={rc_l})")
    q4k_pass = ok_s and ok_l
    if not q4k_pass:
        raise SystemExit("Q4_K round-trip did NOT validate — NOT uploading (gate not met)")

    f16_ok = None
    if vram_mib >= 20000:
        f16_ok, why_f, rc_f = synth(f16, SHORT, WORK / "f16_short.wav", 0.2)
        log(f"f16 short: {'PASS' if f16_ok else 'FAIL'} ({why_f} rc={rc_f})")
    else:
        log(f"f16 round-trip SKIPPED (vram {vram_mib}MiB < 20 GB — 17 GB backbone won't fit)")

    # ── upload (gated on Q4_K pass) ──
    api = HfApi(token=hf_token)
    api.create_repo(HF_REPO, repo_type="model", exist_ok=True, private=False)
    (WORK / "README.md").write_text(README)
    uploads = [
        (q4k, "moss-tts-v1.5-q4_k.gguf"),
        (codec, "moss-tts-v1.5-codec.gguf"),
        (f16, "moss-tts-v1.5-f16.gguf"),
        (WORK / "README.md", "README.md"),
    ]
    results = {}
    for local, name in uploads:
        results[name] = upload_with_timeout(api, local, HF_REPO, name,
                                            timeout_s=5400 if name.endswith("f16.gguf") else 3600)
    # License card check (CLAUDE.md compliance note).
    try:
        card = api.model_info(HF_REPO, expand=["cardData"]).cardData
        log(f"card license = {card.get('license') if card else None}")
    except Exception as e:  # noqa: BLE001
        log(f"card check err: {e}")

    summary = {"repo": HF_REPO, "gpu": gpu, "vram_mib": vram_mib,
               "q4k_pass": q4k_pass, "f16_roundtrip": f16_ok, "uploads": results}
    (WORK / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 60 + "\n" + json.dumps(summary, indent=2) + "\n" + "=" * 60)
    if not all(results.values()):
        raise SystemExit("some uploads UNVERIFIED — see summary.json")
    log(f"SHIPPED: https://huggingface.co/{HF_REPO}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        log(f"FATAL: {e}\n{traceback.format_exc()}")
        sys.exit(1)
