# ─────────────────────────── cell 0 (markdown) ───────────────────────────
# # CrispASR — voxtral-tts PROPER per-stage crispasr-diff on F16
#
# The real HARD-RULE-#2 harness (not the codes-level mudler comparison):
#   python tools/dump_reference.py --backend voxtral-tts --model-dir <m> --output ref.gguf
#   build/bin/crispasr-diff voxtral-tts voxtral-4b-tts-f16.gguf ref.gguf <wav>
#
# The Python dumper is a MANUAL PyTorch LLM forward (no vllm). At F16 the runtime
# should match the BF16 reference per layer (cos>=0.99) — a clean green — vs the
# local Q4_K run where the special AUDIO-token embed row quantized to cos 0.44.
#
# EVERY long phase (build / 16 GB download / dumper / diff) is wrapped in a
# kh.build_heartbeat so the run is pollable mid-flight via the HF progress mirror
# cstr/crispasr-kaggle-progress and RSS/free-GB are visible before any OOM/ENOSPC.
#
# Datasets: ${KAGGLE_ACCOUNT}/crispasr-hf-token, ${KAGGLE_ACCOUNT}/crispasr-ccache.
# GPU + Internet + ~18 GB disk (ref model 8 GB + F16 GGUF 8 GB → /tmp).

# ─────────────────────────── cell 1 (code) ───────────────────────────
import os
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
TMP = Path("/tmp/vtts-cd")
REFMODEL = TMP / "refmodel"
MODELS = TMP / "models"
for d in (REFMODEL, MODELS):
    d.mkdir(parents=True, exist_ok=True)

CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
TEXT = os.environ.get("VOXTRAL_TTS_TEXT", "Hello world.")
VOICE = os.environ.get("VOXTRAL_TTS_VOICE", "neutral_female")


def sh(cmd, check=True, env=None, cwd=None, timeout=None):
    e = {**os.environ, **(env or {})}
    r = subprocess.run(cmd, env=e, cwd=cwd, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(r.stdout[-6000:], flush=True)
    if check and r.returncode != 0:
        raise SystemExit(f"cmd failed ({r.returncode}): {cmd}")
    return r


# ── clone + harness; enable the HF progress mirror BEFORE init_progress ──
if REPO.exists():
    shutil.rmtree(REPO)
sh(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF, "--recursive",
    "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh  # noqa: E402

TOKEN = kh.resolve_hf_token()
if TOKEN:
    os.environ["HF_TOKEN"] = TOKEN  # so init_progress() mirrors to cstr/crispasr-kaggle-progress
kh.init_progress()

sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
kh.step("start", sha=sha)
gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
kh.step("gpu", gpu=gpu)

# ── build crispasr-diff (heartbeat) ──
kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
BUILD.mkdir(exist_ok=True)
sh(["cmake", "-S", str(REPO), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON"]
   + kh.cuda_build_flags(arch) + kh.cache_and_link_flags())
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-diff -j{kh.safe_build_jobs(gpu=True)}")
DIFF = next(c for c in BUILD.rglob("crispasr-diff") if c.is_file() and os.access(c, os.X_OK))
os.environ["LD_LIBRARY_PATH"] = f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
kh.step("built", diff=str(DIFF))

# ── deps for the dumper (NOT torch — pre-installed) ──
with kh.build_heartbeat("pip.deps"):
    sh([sys.executable, "-m", "pip", "install", "-q", "mistral_common", "safetensors", "gguf"], check=False)

# ── downloads → /tmp (heartbeat: watch free_gb climb as ~16 GB lands) ──
from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402
with kh.build_heartbeat("download.refmodel"):
    snapshot_download("mistralai/Voxtral-4B-TTS-2603", local_dir=str(REFMODEL), token=TOKEN or None,
                      allow_patterns=["consolidated.safetensors", "params.json", "tekken.json", "voice_embedding/*"])
kh.step("dl_refmodel_done")
with kh.build_heartbeat("download.f16"):
    F16 = hf_hub_download("cstr/voxtral-4b-tts-GGUF", "voxtral-4b-tts-f16.gguf", local_dir=str(MODELS),
                          token=TOKEN or None)
kh.step("dl_f16_done", size_gb=round(os.path.getsize(F16) / 1e9, 2))

# ── 1) dump the reference (manual PyTorch LLM forward, no vllm) — heartbeat ──
REFGGUF = TMP / "voxtral-tts-ref.gguf"
AUDIO = str(REPO / "samples" / "jfk.wav")  # 16 kHz; unused by voxtral-tts, required by the harness
kh.step("dump_reference_start")
print("\n===== dump_reference.py (manual PyTorch, no vllm) =====", flush=True)
with kh.build_heartbeat("dump_reference"):
    sh([sys.executable, str(REPO / "tools" / "dump_reference.py"), "--backend", "voxtral-tts",
        "--model-dir", str(REFMODEL), "--audio", AUDIO, "--output", str(REFGGUF)],
       env={"VOXTRAL_TTS_TEXT": TEXT, "VOXTRAL_TTS_VOICE": VOICE, "OMP_NUM_THREADS": "4"}, timeout=2400)
kh.step("dump_reference_done", ref_mib=round(os.path.getsize(REFGGUF) / 1e6, 1))

# ── 2) crispasr-diff voxtral-tts <F16> <ref.gguf> — the clean F16 per-layer green ──
kh.step("crispasr_diff_start")
print("\n===== crispasr-diff voxtral-tts (F16 vs BF16 reference) =====", flush=True)
with kh.build_heartbeat("crispasr_diff"):
    r = sh([str(DIFF), "voxtral-tts", F16, str(REFGGUF), AUDIO],
           env={"VOXTRAL_TTS_TEXT": TEXT, "VOXTRAL_TTS_VOICE": VOICE}, check=False, timeout=1200)
verdict = "ALL PASS" if r.returncode == 0 else "divergence"
kh.step("DONE", gpu=gpu, sha=sha[:8], exit=r.returncode, verdict=verdict)
print(f"\n=== DONE gpu={gpu} sha={sha[:8]} exit={r.returncode} ({verdict}) ===", flush=True)
