"""
CrispASR — Parakeet encoder memory policy: CUDA proof (improvements Phase 2).

Question this kernel answers (needs a real NVIDIA card — cannot be done on M1):
does the proactive VRAM-budget policy actually avoid the O(T^2) single-pass
rel-pos-bias OOM that a small-VRAM card hits on long audio (issue #257: a 3.75 min
clip tried to alloc ~1.9 GiB and hit `cudaMalloc … out of memory` on a 3.7 GiB
card), and does the estimate `parakeet_est_singlepass_peak_mb` match the real
CUDA allocation?

Three checks (parakeet-tdt-1.1b, CUDA build from CRISPASR_REF):
  1. ESTIMATE vs REALITY — force single-pass at several clip lengths, poll peak
     GPU VRAM (nvidia-smi), compare to the code's estimate (coeff·T²·H·4 B).
  2. POLICY BOUNDS VRAM — with CRISPASR_PARAKEET_VRAM_BUDGET_MB set, the policy
     must switch to streamed → peak VRAM << single-pass, transcript still full.
  3. OOM AVOIDANCE — pin a torch CUDA buffer to leave only ~`FREE_MB` free
     (simulating the reporter's small card); single-pass must OOM (cudaMalloc
     fail / empty transcript), the budget policy must complete with a full
     transcript. This is the direct proof the policy fixes the reporter's OOM.

Acceptance: (1) estimate within ~2x of measured (the coeff is a heuristic gate,
not an allocator model); (2) policy peak < single-pass peak; (3) single-pass OOMs
where the policy succeeds. Long clips are jfk tiled to a target duration.
"""

import os
import re
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
MODELS = WORK / "models"
CRISPASR = BUILD / "bin" / "crispasr"

CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
# Clip durations (s) for the estimate-vs-reality sweep.
DURATIONS = [int(x) for x in os.environ.get("DURATIONS", "60,120,225").split(",")]
# Simulated small-card free VRAM for the OOM test.
FREE_MB = int(os.environ.get("FREE_MB", "2600"))
MODEL_REPO = os.environ.get("MODEL_REPO", "cstr/parakeet-tdt-1.1b-GGUF")
MODEL_FILE = os.environ.get("MODEL_FILE", "parakeet-tdt-1.1b-q4_k.gguf")
# The code default coefficient / heads for the python-side estimate (mirror
# parakeet_orchestrate.h + hparams: hop 160, subsample 8, 8 heads, coeff 8.0).
MEM_COEFF = float(os.environ.get("MEM_COEFF", "8.0"))
N_HEADS = int(os.environ.get("N_HEADS", "8"))


def _sh(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


print(f"[pre-clone] cloning CrispASR @ {CRISPASR_REF}", flush=True)
if not REPO.exists():
    _sh(f"git clone --depth 1 --branch {CRISPASR_REF} --recursive https://github.com/CrispStrobe/CrispASR {REPO}")

sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
if kh.resolve_hf_token():
    print("[auth] HF token resolved", flush=True)
sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
kh.step("script.start", ref=CRISPASR_REF, sha=sha)


# ── nvidia-smi peak-VRAM poller ────────────────────────────────────────────
class VramPeak:
    def __init__(self, interval=0.05):
        self.interval = interval
        self._stop = threading.Event()
        self.peak = 0
        self.base = self._used()

    @staticmethod
    def _used() -> int:
        try:
            out = subprocess.check_output(
                "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", shell=True, text=True)
            return int(out.strip().splitlines()[0])
        except Exception:
            return 0

    def _loop(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._used())
            time.sleep(self.interval)

    def __enter__(self):
        self.peak = self.base = self._used()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=1)

    @property
    def delta(self) -> int:  # MiB attributable to the run (above the pre-run baseline)
        return max(0, self.peak - self.base)


# ── Build (CUDA) ───────────────────────────────────────────────────────────
kh.step("build.begin")
kh.install_build_toolchain()
cmake_cmd = (f"cmake {REPO} -B{BUILD} -GNinja -DCMAKE_BUILD_TYPE=Release "
             + " ".join(kh.cuda_build_flags()) + " " + " ".join(kh.cache_and_link_flags()))
with kh.build_heartbeat("cmake-configure"):
    kh.sh_with_progress(cmake_cmd)
with kh.build_heartbeat("cmake-build"):
    kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli -- -j{kh.safe_build_jobs(gpu=True)}")
assert CRISPASR.is_file(), "crispasr binary missing"
kh.step("build.done")

# ── Model + clips ──────────────────────────────────────────────────────────
kh.step("download.begin")
MODELS.mkdir(exist_ok=True)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
kh.sh_with_progress("pip install -q huggingface_hub hf_transfer")
from huggingface_hub import hf_hub_download  # noqa: E402

model_path = Path(hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, local_dir=str(MODELS),
                                  local_dir_use_symlinks=False))
kh.step("download.model.done", path=str(model_path))

JFK = REPO / "samples" / "jfk.wav"
assert JFK.is_file()


def tile_clip(dur_s: int) -> Path:
    """Tile jfk.wav to ~dur_s seconds (16 kHz mono) — a controlled long clip."""
    out = WORK / f"clip-{dur_s}s.wav"
    w = wave.open(str(JFK)); sr = w.getframerate(); frames = w.readframes(w.getnframes()); w.close()
    one = len(frames)
    reps = max(1, (dur_s * sr * 2) // one + 1)  # 2 bytes/sample mono
    ow = wave.open(str(out), "w"); ow.setnchannels(1); ow.setsampwidth(2); ow.setframerate(sr)
    ow.writeframes(frames * reps); ow.close()
    return out


def est_singlepass_mb(dur_s: int) -> float:
    # Mirror parakeet_est_singlepass_peak_mb: T_enc = samples/hop/subsample.
    t_enc = (dur_s * 16000 // 160) // 8
    return MEM_COEFF * t_enc * t_enc * N_HEADS * 4.0 / (1024.0 * 1024.0)


def run(clip: Path, env_extra: dict, label: str, timeout=1200) -> dict:
    stem = WORK / f"out-{label}"
    for ext in (".txt",):
        p = stem.with_suffix(ext)
        if p.exists():
            p.unlink()
    cmd = [str(CRISPASR), "-m", str(model_path), "--backend", "parakeet", "-l", "en",
           "-f", str(clip), "-of", str(stem), "-otxt", "-np"]
    env = {**os.environ, **env_extra}
    with VramPeak() as vp:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    log = (r.stdout or "") + (r.stderr or "")
    oom = ("out of memory" in log.lower()) or ("cudamalloc" in log.lower() and "fail" in log.lower())
    txt = stem.with_suffix(".txt")
    text = txt.read_text().strip() if txt.exists() and txt.stat().st_size else ""
    nwords = len(text.split())
    return dict(label=label, peak_delta_mib=vp.delta, oom=oom, words=nwords,
                nonempty=bool(text), rc=r.returncode, log_tail="\n".join(log.splitlines()[-8:]))


# ── Check 1: estimate vs reality (single-pass, unconstrained VRAM) ─────────
kh.step("check1.begin")
print("\n=== CHECK 1: single-pass peak VRAM vs estimate ===", flush=True)
check1 = []
for d in DURATIONS:
    clip = tile_clip(d)
    # Force single-pass, disable the policy, keep full attention.
    r = run(clip, {"CRISPASR_PARAKEET_STREAM_THRESHOLD": "99999",
                   "CRISPASR_PARAKEET_LONGFORM": "0",
                   "CRISPASR_PARAKEET_MEM_POLICY": "off"}, f"single-{d}s")
    est = est_singlepass_mb(d)
    ratio = round(r["peak_delta_mib"] / est, 2) if est else None
    row = dict(dur_s=d, est_mib=round(est), measured_peak_mib=r["peak_delta_mib"],
               est_over_measured=ratio, oom=r["oom"], words=r["words"])
    check1.append(row)
    kh.step(f"check1.{d}s", **row)
    print(f"  {d:>4}s: estimate={round(est):>5} MiB  measured_peak_Δ={r['peak_delta_mib']:>5} MiB  "
          f"ratio={ratio}  words={r['words']}  oom={r['oom']}", flush=True)

# ── Check 2: policy bounds VRAM (budget forces streamed) ───────────────────
kh.step("check2.begin")
print("\n=== CHECK 2: VRAM budget policy bounds peak VRAM ===", flush=True)
long_clip = tile_clip(max(DURATIONS))
budget = max(256, int(est_singlepass_mb(max(DURATIONS)) * 0.5))  # half the single-pass need
sp = run(long_clip, {"CRISPASR_PARAKEET_STREAM_THRESHOLD": "99999", "CRISPASR_PARAKEET_MEM_POLICY": "off"},
         "policy-single")
pol = run(long_clip, {"CRISPASR_PARAKEET_VRAM_BUDGET_MB": str(budget)}, "policy-streamed")
kh.step("check2.result", budget_mib=budget, single_peak_mib=sp["peak_delta_mib"],
        policy_peak_mib=pol["peak_delta_mib"], single_words=sp["words"], policy_words=pol["words"])
print(f"  budget={budget} MiB | single-pass peak_Δ={sp['peak_delta_mib']} MiB ({sp['words']} w) | "
      f"policy peak_Δ={pol['peak_delta_mib']} MiB ({pol['words']} w)", flush=True)

# ── Check 3: OOM avoidance under a simulated small card ─────────────────────
kh.step("check3.begin")
print(f"\n=== CHECK 3: OOM avoidance (torch hog → ~{FREE_MB} MiB free) ===", flush=True)
hog_ok = False
try:
    import torch  # preinstalled on Kaggle
    total = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    free_now = int(subprocess.check_output(
        "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits", shell=True, text=True).splitlines()[0])
    hog_mib = max(0, free_now - FREE_MB)
    hog = torch.empty(int(hog_mib * 1024 * 1024 // 4), dtype=torch.float32, device="cuda") if hog_mib > 0 else None
    torch.cuda.synchronize()
    free_after = int(subprocess.check_output(
        "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits", shell=True, text=True).splitlines()[0])
    print(f"  total={total} MiB, hogged {hog_mib} MiB, free now ≈ {free_after} MiB", flush=True)
    hog_ok = True
    oom_clip = tile_clip(max(DURATIONS))
    single = run(oom_clip, {"CRISPASR_PARAKEET_STREAM_THRESHOLD": "99999", "CRISPASR_PARAKEET_MEM_POLICY": "off"},
                 "oom-single")
    policy = run(oom_clip, {"CRISPASR_PARAKEET_VRAM_BUDGET_MB": str(FREE_MB)}, "oom-policy")
    kh.step("check3.result", free_mib=free_after, single_oom=single["oom"], single_words=single["words"],
            policy_oom=policy["oom"], policy_words=policy["words"])
    print(f"  single-pass: oom={single['oom']} words={single['words']} rc={single['rc']}", flush=True)
    print(f"  budget policy: oom={policy['oom']} words={policy['words']} rc={policy['rc']}", flush=True)
    if not single["nonempty"]:
        print(f"  single-pass stderr tail:\n{single['log_tail']}", flush=True)
    del hog
    torch.cuda.empty_cache()
except Exception as e:  # noqa: BLE001
    print(f"  CHECK 3 skipped ({type(e).__name__}: {e})", flush=True)
    kh.step("check3.skipped", error=str(e))

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print(f"SUMMARY — parakeet memory policy CUDA proof ({sha[:8]})")
print("=" * 72)
print("CHECK 1 estimate vs measured single-pass peak:")
for r in check1:
    print(f"  {r['dur_s']:>4}s est={r['est_mib']:>5} MiB  measured={r['measured_peak_mib']:>5} MiB  "
          f"ratio={r['est_over_measured']}")
print(f"CHECK 2 policy: single peak {sp['peak_delta_mib']} MiB → streamed {pol['peak_delta_mib']} MiB "
      f"(budget {budget}); words {sp['words']}→{pol['words']}")
if hog_ok:
    print("CHECK 3 (small-card sim): see steps — expect single-pass OOM, policy full transcript.")
kh._push_progress_to_hf(force=True)
kh.step("script.end")
