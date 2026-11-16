import os
#!/usr/bin/env python3
"""Capture remaining regression transcripts: mimo-asr (with codec) + kugelaudio."""
import json, os, subprocess, sys, time, shutil, traceback
from pathlib import Path

WORK = Path("/kaggle/working")
results = {}

def log(msg):
    print(msg, flush=True)
    try:
        with open(WORK / "progress.txt", "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    except Exception:
        pass

def save():
    try:
        (WORK / "transcripts.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    except Exception:
        pass

def main():
    global results
    save()
    log("=== Regression capture v2 ===")

    # Clone + build
    cdir = WORK / "CrispASR"
    if not cdir.exists():
        subprocess.check_call(["git", "clone", "--depth", "1",
            "https://github.com/CrispStrobe/CrispASR.git", str(cdir)])
    sys.path.insert(0, str(cdir / "tools" / "kaggle"))
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
        log(f"harness: {e}")
        for p in ["/kaggle/input/crispasr-hf-token/hf_token.txt",
                  "/kaggle/input/datasets/${KAGGLE_ACCOUNT}/crispasr-hf-token/hf_token.txt"]:
            if os.path.exists(p):
                os.environ["HF_TOKEN"] = open(p).read().strip()
                break
        if not shutil.which("cmake"):
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cmake", "ninja"], check=False)

    bdir = cdir / "build"
    cmake_args = ["-DCMAKE_BUILD_TYPE=Release", "-DCRISPASR_NO_C2PA_NATIVE=ON"]
    if shutil.which("ninja"): cmake_args += ["-G", "Ninja"]
    if shutil.which("ccache"): cmake_args += ["-DCMAKE_C_COMPILER_LAUNCHER=ccache", "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"]
    subprocess.check_call(["cmake", "-B", str(bdir)] + cmake_args, cwd=str(cdir))
    subprocess.check_call(["cmake", "--build", str(bdir), "-j2"], cwd=str(cdir))
    log("Build OK")

    CLI = str(bdir / "bin" / "crispasr")
    JFK = str(cdir / "samples" / "jfk.wav")
    MDIR = WORK / "models"
    MDIR.mkdir(exist_ok=True)

    from huggingface_hub import hf_hub_download

    def dl(repo, fname):
        hf_hub_download(repo_id=repo, filename=fname, cache_dir=str(MDIR/".hf"), local_dir=str(MDIR))
        return str(MDIR / fname)

    def run(cmd, timeout=600):
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        lines = [l.strip() for l in r.stdout.strip().split('\n') if l.strip()]
        return r.returncode, lines[-1] if lines else "", time.time()

    # 1. mimo-asr (needs --codec-model)
    log("\n[1] mimo-asr")
    try:
        m1 = dl("cstr/mimo-asr-GGUF", "mimo-asr-q4_k.gguf")
        m1c = dl("cstr/mimo-tokenizer-GGUF", "mimo-tokenizer-q4_k.gguf")
        log(f"  DL OK")
        t0 = time.time()
        rc, txt, _ = run([CLI, "--backend", "mimo-asr", "-m", m1, "--codec-model", m1c, "-f", JFK, "-np"])
        elapsed = time.time() - t0
        results["mimo-asr"] = {"status": "ok" if rc == 0 and txt else "fail", "rc": rc,
                                "transcript": txt, "elapsed_s": round(elapsed, 1)}
        log(f"  rc={rc} {elapsed:.0f}s: {txt[:80]}")
    except Exception as e:
        results["mimo-asr"] = {"status": "error", "error": str(e)}
        log(f"  ERROR: {e}")
    # Cleanup
    for f in ["mimo-asr-q4_k.gguf", "mimo-tokenizer-q4_k.gguf"]:
        p = MDIR / f
        if p.exists(): p.unlink()
    save()

    # 2. kugelaudio (5.4GB — download only if disk allows)
    log("\n[2] kugelaudio")
    disk_gb = shutil.disk_usage(str(WORK)).free / (1024**3)
    if disk_gb < 7:
        log(f"  SKIP disk={disk_gb:.1f}GB")
        results["kugelaudio"] = {"status": "skip_disk"}
    else:
        try:
            m2 = dl("cstr/kugelaudio-0-open-GGUF", "kugelaudio-0-open-q4_k.gguf")
            log(f"  DL OK")
            t0 = time.time()
            rc, txt, _ = run([CLI, "--backend", "kugelaudio", "-m", m2, "-f", JFK, "-np"], timeout=900)
            elapsed = time.time() - t0
            results["kugelaudio"] = {"status": "ok" if rc == 0 and txt else "fail", "rc": rc,
                                      "transcript": txt, "elapsed_s": round(elapsed, 1)}
            log(f"  rc={rc} {elapsed:.0f}s: {txt[:80]}")
        except Exception as e:
            results["kugelaudio"] = {"status": "error", "error": str(e)}
            log(f"  ERROR: {e}")
    save()

    # 3. granite-4.1-nar (SIGABRT on CUDA — try CPU-only)
    log("\n[3] granite-4.1-nar (CPU-only attempt)")
    try:
        m3 = dl("cstr/granite-speech-4.1-2b-nar-GGUF", "granite-speech-4.1-2b-nar-q4_k.gguf")
        log(f"  DL OK")
        t0 = time.time()
        rc, txt, _ = run([CLI, "--backend", "granite", "-m", m3, "-f", JFK, "-np", "--no-gpu"], timeout=600)
        elapsed = time.time() - t0
        results["granite-4.1-nar"] = {"status": "ok" if rc == 0 and txt else "fail", "rc": rc,
                                       "transcript": txt, "elapsed_s": round(elapsed, 1)}
        log(f"  rc={rc} {elapsed:.0f}s: {txt[:80]}")
    except Exception as e:
        results["granite-4.1-nar"] = {"status": "error", "error": str(e)}
        log(f"  ERROR: {e}")
    save()

    log("\nDONE")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        results["_crash"] = str(e)
        results["_tb"] = traceback.format_exc()
        save()
        traceback.print_exc()
