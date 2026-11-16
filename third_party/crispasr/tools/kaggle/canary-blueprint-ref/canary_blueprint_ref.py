#!/usr/bin/env python3
"""canary-1b-v2 BLUEPRINT reference transcripts for the #375 long-form port.

Runs NeMo's own `EncDecMultiTaskModel.transcribe()` — which for a single
audio file > 40 s auto-enables the dynamic 30..40 s / 1 s-overlap chunking
with the lcs_alignment merge — on the exact clips the C++ port was validated
against, and writes the transcripts to /kaggle/working/blueprint_<name>.txt.

The C++ side (src/canary.cpp + core/canary_chunk_merge.h, commit 8219b429)
ports that chunking; this kernel produces the decoded-output ground truth to
compare against (HARD RULE #3: the decoded-output roundtrip is the only
acceptance test). Run on GPU purely for speed; the 1B AED on 132 s + 654 s of
audio is minutes on a T4/P100 and would memory-crush a 16 GB laptop (it did).
"""

import subprocess
import sys
import os
from pathlib import Path

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp")
TEMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", "/tmp/hf")  # 6.4 GB .nemo: keep off /kaggle/working (~20 GB cap)
os.environ.setdefault("TMPDIR", "/tmp")

# ---- CrispASR clone: kaggle_harness + repo context (gotcha #26: bundled
# siblings do not ship for script kernels — the clone is the real path).
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
_CRISPASR_DIR = TEMP / "CrispASR"
if not _CRISPASR_DIR.exists():
    try:
        subprocess.check_call(["git", "clone", "--depth", "1", CRISPASR_URL, str(_CRISPASR_DIR)])
    except Exception as e:
        print(f"clone failed: {e}", flush=True)
if str(_CRISPASR_DIR / "tools" / "kaggle") not in sys.path:
    sys.path.insert(0, str(_CRISPASR_DIR / "tools" / "kaggle"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
token = kh.resolve_hf_token()
kh.step("setup", note="harness ready, token resolved")

# ---- deps: torch is preinstalled (gotcha #11); NeMo pinned to the version
# the port was read against, with an unpinned fallback.
with kh.build_heartbeat("pip nemo", 30):
    rc = subprocess.call([sys.executable, "-m", "pip", "install", "-q", "nemo_toolkit[asr]==2.7.3"])
    if rc != 0:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "nemo_toolkit[asr]"])
    # Kaggle's numba-cuda split package breaks `import numba.cuda` inside
    # NeMo's rnnt_loss (AttributeError: numba.cuda.types has no NPDatetime).
    # Match the verified-working local pair: numba 0.61.2 (built-in cuda
    # target) + llvmlite 0.44.0, and drop numba-cuda entirely.
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "numba-cuda"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numba==0.61.2", "llvmlite==0.44.0"])
kh.step("deps", note="nemo installed; numba pinned 0.61.2, numba-cuda removed")

from huggingface_hub import hf_hub_download  # noqa: E402

CLIPS = [
    ("jfk_x12", "en"),
    ("fleurs_60s", "de"),
    ("fleurs_600s", "de"),
]
paths = {}
with kh.build_heartbeat("fixtures", 30):
    for name, _ in CLIPS:
        paths[name] = hf_hub_download(
            "cstr/crispasr-regression-fixtures",
            f"canary-longform/{name}.wav",
            repo_type="dataset",
            token=token,
            local_dir="/tmp/clips",
        )
kh.step("fixtures", note=f"{len(paths)} clips downloaded")

import torch  # noqa: E402

import nemo.collections.asr as nemo_asr  # noqa: E402

# Kaggle assigns P100 (sm_60) or T4 (sm_75) randomly; the preinstalled torch
# ships no sm_60 kernels ("no kernel image is available"). Probe with a real
# op and fall back to CPU — the run is minutes either way and the kernel
# needs GPU enabled anyway for internet access.
device = "cpu"
if torch.cuda.is_available():
    try:
        (torch.zeros(4, device="cuda") + 1).sum().item()
        device = "cuda"
    except Exception as e:
        print(f"CUDA probe failed ({type(e).__name__}); using CPU. cap={torch.cuda.get_device_capability()}", flush=True)
if device == "cpu":
    torch.set_num_threads(os.cpu_count() or 4)

with kh.build_heartbeat("model load", 30):
    model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained("nvidia/canary-1b-v2", map_location=device)
    model = model.to(device)
    model.eval()
kh.step("model", note=f"loaded on {device}; ts_model={model.timestamps_asr_model is not None}")

import nemo  # noqa: E402

print(f"nemo version: {nemo.__version__}", flush=True)

for name, lang in CLIPS:
    with kh.build_heartbeat(f"transcribe {name}", 30):
        out = model.transcribe([paths[name]], batch_size=1, source_lang=lang, target_lang=lang, pnc="yes")
    h = out[0]
    text = h.text if hasattr(h, "text") else str(h)
    (WORK / f"blueprint_{name}.txt").write_text(text + "\n", encoding="utf-8")
    print(f"=== BLUEPRINT {name} ({lang}) ===", flush=True)
    print(text, flush=True)
    kh.step(f"transcribe.{name}", note=f"{len(text.split())} words")

kh.step("done", note="all transcripts written")
print("DONE", flush=True)
