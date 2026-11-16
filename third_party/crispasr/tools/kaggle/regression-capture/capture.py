import os
#!/usr/bin/env python3
"""
CrispASR regression transcript capture — Kaggle GPU kernel.

Builds CrispASR, downloads each backend's GGUF from HF, runs on JFK 11s,
captures the transcript, and writes results to /kaggle/working/transcripts.json.
"""
import json, os, subprocess, sys, time, shutil, traceback
from pathlib import Path

WORK = Path("/kaggle/working")
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
_CRISPASR_DIR = WORK / "CrispASR"
results = {}
progress_file = WORK / "progress.txt"

def log(msg):
    print(msg, flush=True)
    try:
        with open(progress_file, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    except Exception:
        pass

def save_results():
    try:
        with open(WORK / "transcripts.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

def main():
    global results

    # Crash marker
    save_results()
    log("=== Starting regression capture ===")

    # Clone
    if not _CRISPASR_DIR.exists():
        subprocess.check_call(["git", "clone", "--depth", "1",
            CRISPASR_URL, str(_CRISPASR_DIR)])
    sys.path.insert(0, str(_CRISPASR_DIR / "tools" / "kaggle"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    # Harness (best effort)
    try:
        import kaggle_harness as kh
        # Full harness regime: authenticate HF pulls from the attached token
        # dataset (anon pulls get rate-limited); hf_transfer wedges multi-GB
        # Kaggle downloads, so keep the plain resumable downloader.
        kh.resolve_hf_token()
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        kh.init_progress()
        kh.setup_hf_token()
        kh.install_build_toolchain()
    except Exception as e:
        log(f"harness setup: {e}")
        # Fallback HF token
        for p in ["/kaggle/input/crispasr-hf-token/hf_token.txt",
                  "/kaggle/input/datasets/${KAGGLE_ACCOUNT}/crispasr-hf-token/hf_token.txt"]:
            if os.path.exists(p):
                tok = open(p).read().strip()
                os.environ["HF_TOKEN"] = tok
                os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
                break

    # Ensure build tools
    if not shutil.which("cmake"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cmake", "ninja"], check=False)

    # Build
    log("Building...")
    results["_status"] = "building"
    save_results()
    build_dir = _CRISPASR_DIR / "build"
    cmake_args = ["-DCMAKE_BUILD_TYPE=Release", "-DCRISPASR_NO_C2PA_NATIVE=ON"]
    if shutil.which("ninja"):
        cmake_args += ["-G", "Ninja"]
    if shutil.which("ccache"):
        cmake_args += ["-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                       "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"]

    subprocess.check_call(["cmake", "-B", str(build_dir)] + cmake_args,
                          cwd=str(_CRISPASR_DIR))
    subprocess.check_call(["cmake", "--build", str(build_dir), "-j2"],
                          cwd=str(_CRISPASR_DIR))
    log("Build OK")

    CLI = str(build_dir / "bin" / "crispasr")
    JFK = str(_CRISPASR_DIR / "samples" / "jfk.wav")
    MDIR = WORK / "models"
    MDIR.mkdir(exist_ok=True)

    BACKENDS = [
        ("qwen3-asr-0.6b", "qwen3", "cstr/qwen3-asr-0.6b-GGUF", "qwen3-asr-0.6b-q4_k.gguf"),
        ("omniasr-ctc-1b-v2", "omniasr", "cstr/omniASR-CTC-1B-v2-GGUF", "omniasr-ctc-1b-v2-q4_k.gguf"),
        ("kyutai-stt-1b", "kyutai-stt", "cstr/kyutai-stt-1b-GGUF", "kyutai-stt-1b-q4_k.gguf"),
        ("funasr-nano", "funasr", "cstr/funasr-nano-GGUF", "funasr-nano-2512-q8_0.gguf"),
        ("mini-omni2", "mini-omni2", "cstr/mini-omni2-GGUF", "mini-omni2-q4_k.gguf"),
        ("omniasr-llm-300m", "omniasr", "cstr/omniasr-llm-300m-v2-GGUF", "omniasr-llm-300m-v2-q4_k.gguf"),
        ("funasr-mlt-nano", "fun-asr-mlt-nano", "cstr/funasr-mlt-nano-GGUF", "funasr-mlt-nano-2512-f16.gguf"),
        ("voxtral-mini-3b-2507", "voxtral", "cstr/voxtral-mini-3b-2507-GGUF", "voxtral-mini-3b-2507-q4_k.gguf"),
        ("gemma4-e2b-it", "gemma4-e2b", "cstr/gemma4-e2b-it-GGUF", "gemma4-e2b-it-q4_k.gguf"),
        ("granite-4.1-plus", "granite", "cstr/granite-speech-4.1-2b-plus-GGUF", "granite-speech-4.1-2b-plus-q4_k.gguf"),
        ("granite-4.1-nar", "granite", "cstr/granite-speech-4.1-2b-nar-GGUF", "granite-speech-4.1-2b-nar-q4_k.gguf"),
        ("voxtral4b-realtime", "voxtral4b", "cstr/voxtral-mini-4b-realtime-GGUF", "voxtral-mini-4b-realtime-q4_k.gguf"),
        ("moss-audio-4b-instruct", "moss-audio", "cstr/MOSS-Audio-4B-Instruct-GGUF", "moss-audio-4b-instruct-q4_k.gguf"),
        ("mimo-asr", "mimo-asr", "cstr/mimo-asr-GGUF", "mimo-asr-q4_k.gguf"),
        ("vibevoice-asr", "vibevoice", "cstr/vibevoice-asr-GGUF", "vibevoice-asr-q4_k.gguf"),
    ]

    results["_status"] = "running"
    save_results()
    log(f"=== {len(BACKENDS)} backends ===")

    for i, (name, bid, repo, fname) in enumerate(BACKENDS):
        log(f"\n[{i+1}/{len(BACKENDS)}] {name}")
        disk_gb = shutil.disk_usage(str(WORK)).free / (1024**3)
        if disk_gb < 2.0:
            log(f"  SKIP disk={disk_gb:.1f}GB")
            results[name] = {"status": "skip_disk"}
            save_results()
            continue

        # Download
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id=repo, filename=fname,
                           cache_dir=str(MDIR / ".hf"), local_dir=str(MDIR))
            mpath = str(MDIR / fname)
            log(f"  DL OK: {fname}")
        except Exception as e:
            log(f"  DL FAIL: {e}")
            results[name] = {"status": "dl_fail", "error": str(e)}
            save_results()
            continue

        # Run
        t0 = time.time()
        cmd = [CLI, "--backend", bid, "-m", mpath, "-f", JFK, "-np"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            elapsed = time.time() - t0
            transcript = ""
            if r.returncode == 0:
                lines = [l.strip() for l in r.stdout.strip().split('\n') if l.strip()]
                transcript = lines[-1] if lines else ""
            results[name] = {
                "status": "ok" if r.returncode == 0 and transcript else "fail",
                "transcript": transcript,
                "rc": r.returncode,
                "elapsed_s": round(elapsed, 1),
            }
            log(f"  rc={r.returncode} {elapsed:.0f}s: {transcript[:80]}")
        except subprocess.TimeoutExpired:
            results[name] = {"status": "timeout"}
            log(f"  TIMEOUT")
        except Exception as e:
            results[name] = {"status": "error", "error": str(e)}
            log(f"  ERROR: {e}")

        # Cleanup model
        mp = MDIR / fname
        if mp.exists():
            mp.unlink()
        save_results()

    # Summary
    ok = sum(1 for r in results.values() if isinstance(r, dict) and r.get("status") == "ok")
    log(f"\nDONE: {ok}/{len(BACKENDS)} OK")
    results["_status"] = "done"
    save_results()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        results["_crash"] = str(e)
        results["_traceback"] = traceback.format_exc()
        save_results()
        log(f"CRASH: {e}")
        traceback.print_exc()
