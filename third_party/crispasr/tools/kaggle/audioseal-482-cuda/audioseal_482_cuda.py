#!/usr/bin/env python3
"""Run the CI-built #482 CUDA proof on real GPU hardware; never build on Kaggle."""
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

SCRIPT_VERSION = "audioseal-482-cuda-v1"
BUILD_RUN = 36835772201
BUILD_SHA = "833ad5b58a11b1042da3b91483d13a690ea5c8bb"
BUNDLE_REPO = "cstr/crispasr-audioseal-cuda-proof"
BUNDLE_REVISION = "a433ef5cd2e9197f621414c29825f5306be84091"
BUNDLE_SHA256 = "32285a273bec92c880f000dc2a86a3f2e21c262347ff2f62e19585df04e183f6"
WORK = Path("/kaggle/working")
SCRATCH = Path("/kaggle/temp/audioseal-482")
SCRATCH.mkdir(parents=True, exist_ok=True)
REPO = SCRATCH / "CrispASR"
subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"], check=True)
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/CrispStrobe/CrispASR.git", str(REPO)], check=True)
sys.path.insert(0, str(REPO / "tools/kaggle"))
import kaggle_harness as kh

kh.init_progress()
hf_token = kh.resolve_hf_token()
kh.provenance(SCRIPT_VERSION, REPO)
# Stage the CI artifact privately; Kaggle needs only its existing HF credential.
# Pin both the immutable storage revision and archive checksum.
tar_path = Path(hf_hub_download(BUNDLE_REPO, "audioseal-cuda-proof.tar.gz",
                               repo_type="dataset", revision=BUNDLE_REVISION,
                               token=hf_token, local_dir=SCRATCH / "artifact"))
if hashlib.sha256(tar_path.read_bytes()).hexdigest() != BUNDLE_SHA256:
    raise RuntimeError("proof archive checksum mismatch")
with tarfile.open(tar_path) as archive:
    for member in archive.getmembers():
        if not (member.isfile() or member.isdir()) or member.name.startswith("/") or ".." in Path(member.name).parts:
            raise RuntimeError("unexpected proof archive member")
    archive.extractall(SCRATCH)
bundle = SCRATCH / "bundle"
provenance = json.loads((bundle / "provenance.json").read_text())
if provenance["sha"] != BUILD_SHA:
    raise RuntimeError("proof bundle provenance mismatch")
kh.step("bundle.ready", build_sha=BUILD_SHA, build_run=BUILD_RUN)
model = SCRATCH / "audioseal.gguf"
urllib.request.urlretrieve("https://huggingface.co/cstr/audioseal-GGUF/resolve/main/audioseal.gguf", model)
subprocess.run(["uptime"], check=True)
subprocess.run(["free", "-h"], check=True)
env = dict(os.environ, LD_LIBRARY_PATH=str(bundle) + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
with kh.build_heartbeat("cuda.inference"):
    result = subprocess.run([str(bundle / "audioseal-cuda-proof"), str(model)],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=1800)
(WORK / "cuda-proof.log").write_text(result.stdout)
print(result.stdout, flush=True)
passed = result.returncode == 0 and "AUDIOSEAL_CUDA_PASS" in result.stdout
# The model loader must select CUDA, in addition to the explicit CUDA API check.
passed = passed and "using preferred GPU backend: CUDA" in result.stdout
summary = {"script_version": SCRIPT_VERSION, "build_sha": BUILD_SHA, "build_run": BUILD_RUN,
           "bundle_revision": BUNDLE_REVISION, "bundle_sha256": BUNDLE_SHA256,
           "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version",
                                             "--format=csv,noheader"], text=True).strip(),
           "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
           "rc": result.returncode, "passed": passed}
(WORK / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
kh.step("cuda.result", **summary)
if not passed:
    raise SystemExit("AudioSeal CUDA proof failed")
