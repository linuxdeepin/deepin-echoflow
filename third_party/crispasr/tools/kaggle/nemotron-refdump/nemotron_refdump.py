import os, subprocess, sys, shutil
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
OUT_REF = WORK / "nemotron-3.5-asr-streaming-0.6b-ref.gguf"
BRANCH = os.environ.get("CRISPASR_REF", "main")

print(f"[1] cloning CrispASR {BRANCH}", flush=True)
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.check_call([
    "git", "clone", "--depth", "1", "--branch", BRANCH,
    "https://github.com/CrispStrobe/CrispASR.git", str(REPO),
])

sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh
kh.init_progress()
hf_token = kh.resolve_hf_token()
kh.step("cloned", branch=BRANCH, hf_token_ok=bool(hf_token))

print("[2] installing deps", flush=True)
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "torch", "transformers", "safetensors", "gguf",
    "huggingface_hub", "hf_transfer", "nemo_toolkit[asr]",
])
kh.step("deps_installed")

print("[3] running tools/dump_reference.py", flush=True)
sys.path.insert(0, str(REPO / "tools"))
import dump_reference

subprocess.check_call([
    sys.executable, str(REPO / "tools" / "dump_reference.py"),
    "--backend", "nemotron",
    "--model-dir", "nvidia/nemotron-3.5-asr-streaming-0.6b",
    "--audio", str(REPO / "samples" / "jfk.wav"),
    "--output", str(OUT_REF),
])
kh.step("refdump_done", size_mib=round(OUT_REF.stat().st_size / (1024**2), 1))

HF_REPO = "cstr/crispasr-regression-fixtures"
if hf_token:
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    if OUT_REF.exists():
        print(f"[4] uploading ref GGUF ({OUT_REF.stat().st_size / (1024**2):.1f} MiB)", flush=True)
        api.upload_file(
            path_or_fileobj=str(OUT_REF),
            path_in_repo="nemotron-3.5-asr-streaming-0.6b-ref.gguf",
            repo_id=HF_REPO, repo_type="model",
            commit_message="Add nemotron reference activation dump",
        )
        print("[4] uploaded ref GGUF", flush=True)
    kh.step("uploaded")
else:
    print("[4] no HF_TOKEN — staged locally", flush=True)

kh.step("done")
print("[DONE]", flush=True)
