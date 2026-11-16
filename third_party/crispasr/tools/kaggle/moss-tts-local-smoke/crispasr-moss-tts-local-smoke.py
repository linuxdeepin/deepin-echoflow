# CrispASR — 4B MossTTSLocal runtime SMOKE test (#249, P5-lite)
#
# Builds moss-tts-local-smoke, converts the real 4B weights -> GGUF (uploads it to
# cstr/moss-tts-local-v1.5-GGUF for reuse), then runs the smoke: generate_codes
# end-to-end (backbone Qwen3-4B + 1-layer local/depth transformer + depth-first
# 12-codebook loop) and checks it produces a valid (12, T) grid without crashing.
# NOT parity — just "does the hand-written runtime RUN on real weights". The 4B
# F16 (~9 GB) fits a P100 16 GB GPU. ccache under /kaggle/temp (usage #22).

import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
TMP = Path("/tmp")
REPO = TMP / "CrispASR"
BUILD = REPO / "build"
MODELS = TMP / "moss-local"
WORK = Path("/kaggle/working")
MODELS.mkdir(parents=True, exist_ok=True)

REF = os.environ.get("CRISPASR_REF", "feat/moss-tts-parity-diff")
HF_MODEL = os.environ.get("MOSS_MODEL", "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5")
GGUF_REPO = os.environ.get("MOSS_GGUF_REPO", "cstr/moss-tts-local-v1.5-GGUF")
TEXT = os.environ.get("MOSS_TEXT", "The quick brown fox jumps over the lazy dog.")
_T0 = time.time()


def log(m):
    print(f"[{round(time.time() - _T0, 1)}s] {m}", flush=True)


def main():
    summary = {"ref": REF, "text": TEXT}
    log(f"clone {REF}")
    if not REPO.exists():
        subprocess.check_call(["git", "clone", "--depth", "1", "--branch", REF, "--recursive",
                               "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.init_progress()
    tok = kh.resolve_hf_token()
    if tok:
        os.environ["HF_TOKEN"] = tok
        os.environ["HUGGING_FACE_HUB_TOKEN"] = tok

    # ── build the smoke exe (pulls in moss_tts_local + crispasr-core + ggml-cuda) ──
    kh.install_build_toolchain()
    arch = kh.detect_cuda_arch()
    env = os.environ.copy()
    env["CCACHE_DIR"] = "/kaggle/temp/.ccache"
    subprocess.run(["cmake", "-G", "Ninja", "-B", str(BUILD), "-S", str(REPO), "-DCMAKE_BUILD_TYPE=Release"]
                   + list(kh.cache_and_link_flags()) + list(kh.cuda_build_flags(arch)),
                   env=env, check=True, timeout=300)
    with kh.build_heartbeat("moss-tts-local-smoke build"):
        kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} --target moss-tts-local-smoke "
                            f"-j{kh.safe_build_jobs(gpu=True)}")
    smoke = None
    for c in BUILD.rglob("moss-tts-local-smoke"):
        if c.is_file() and os.access(c, os.X_OK):
            smoke = c
            break
    if not smoke:
        raise SystemExit("moss-tts-local-smoke not built")
    os.environ["LD_LIBRARY_PATH"] = f"{BUILD/'src'}:{os.environ.get('LD_LIBRARY_PATH','')}"
    log(f"built {smoke}")

    # ── convert the 4B -> GGUF (download if already uploaded, else convert) ──
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gguf", "safetensors", "huggingface_hub"])
    from huggingface_hub import hf_hub_download, snapshot_download, HfApi
    gguf = MODELS / "moss-tts-local-v1.5-f16.gguf"
    try:
        got = hf_hub_download(GGUF_REPO, gguf.name, local_dir=str(MODELS), token=tok)
        gguf = Path(got)
        log(f"downloaded existing GGUF {gguf}")
    except Exception:  # noqa: BLE001
        log("convert 4B -> GGUF")
        src = snapshot_download(HF_MODEL, cache_dir=str(MODELS / "hf"), token=tok,
                                allow_patterns=["*.safetensors", "*.json", "merges.txt", "vocab.json",
                                                "tokenizer.json", "added_tokens.json"])
        cenv = os.environ.copy()
        cenv["TMPDIR"] = str(MODELS)
        subprocess.check_call([sys.executable, str(REPO / "models" / "convert-moss-tts-local-to-gguf.py"),
                               "--input", src, "--output", str(gguf)], env=cenv, timeout=3600)
        try:
            api = HfApi()
            api.create_repo(GGUF_REPO, repo_type="model", exist_ok=True, token=tok)
            api.upload_file(path_or_fileobj=str(gguf), path_in_repo=gguf.name, repo_id=GGUF_REPO,
                            repo_type="model", token=tok)
            summary["uploaded"] = f"{GGUF_REPO}/{gguf.name}"
            log("uploaded GGUF for reuse")
        except Exception as e:  # noqa: BLE001
            log(f"upload skipped: {e}")
    summary["gguf_gb"] = round(gguf.stat().st_size / 1e9, 2)

    # ── run the smoke ──
    log("run moss-tts-local-smoke")
    r = subprocess.run([str(smoke), str(gguf), TEXT, "40"], capture_output=True, text=True, timeout=1800)
    out = r.stdout + "\n--STDERR--\n" + r.stderr
    (WORK / "smoke.log").write_text(out)
    print(out[-4000:], flush=True)
    summary["rc"] = r.returncode
    summary["smoke_pass"] = ("SMOKE PASS" in r.stdout)
    import re
    m = re.search(r"generated grid: n_vq=(\d+) T=(\d+)", r.stdout)
    if m:
        summary["n_vq"], summary["T"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"code range \[(-?\d+), (-?\d+)\], out-of-range=(\d+)", r.stdout)
    if m:
        summary["code_min"], summary["code_max"], summary["oob"] = int(m.group(1)), int(m.group(2)), int(m.group(3))

    (WORK / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 60 + "\n" + json.dumps(summary, indent=2) + "\n" + "=" * 60)
    log(f"SMOKE {'PASS' if summary['smoke_pass'] else 'FAIL'}")
    if not summary["smoke_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
