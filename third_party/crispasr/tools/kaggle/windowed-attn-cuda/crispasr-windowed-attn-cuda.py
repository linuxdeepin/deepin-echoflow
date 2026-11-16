#!/usr/bin/env python3
# CUDA cross-check for TRUE windowed (block sliding-chunks) FastConformer attention.
#
# Validates on a real GPU (P100/T4/…) that the windowed local-attention path
# (default when --att-context is set; CRISPASR_FC_WINDOWED_ATTN=0 = legacy
# masked-full) is:
#   (1) CORRECT   — windowed transcript == masked-full transcript, and
#   (2) LIGHTER   — lower peak GPU memory at large single-pass T (the O(T²)
#                   rel-pos bias BD is the hog; windowed makes it O(T·window)),
#   (3) not slower — wall-clock per config.
#
# The M1/Metal A/B already showed identical transcripts + ~3× speed + ~10% peak
# footprint at T=7838 (conv front-end co-dominates; BD win grows with length).
# CUDA is the one backend I can't validate locally — this is that check.
#
# Kaggle setup: GPU accelerator ON, Internet ON. HF_TOKEN optional (model repo
# is public). Run as a script kernel; watch tools/kaggle progress on
# cstr/crispasr-kaggle-progress or the browser UI.

import os
import subprocess
import sys
import threading
import time

REPO = "/kaggle/working/CrispASR"
BUILD = f"{REPO}/build"
MODEL_REPO = "cstr/parakeet-tdt-0.6b-v3-GGUF"   # rel_pos_local_attn-capable (NeMo)
MODEL_FILE = "parakeet-tdt-0.6b-v3-q4_k.gguf"
BRANCH = os.environ.get("CRISPASR_BRANCH", "main")
ATT = os.environ.get("ATT_CONTEXT", "64,64")    # local window (encoder frames)
REPEAT = int(os.environ.get("CLIP_REPEAT", "60"))  # jfk.wav ~11s × 60 ≈ 660s → T≈8k

# ── clone repo early so we can import the shared harness ────────────────────
if not os.path.isdir(REPO):
    subprocess.check_call(
        f"git clone --depth 1 -b {BRANCH} https://github.com/CrispStrobe/CrispASR {REPO}", shell=True)

sys.path.insert(0, f"{REPO}/tools/kaggle")
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
kh.step("script.start", branch=BRANCH, att=ATT, repeat=REPEAT)

# ── toolchain + CUDA build ─────────────────────────────────────────────────
kh.install_build_toolchain()
arch = kh.detect_cuda_arch()          # P100 → "60"
flags = kh.cuda_build_flags(arch) + kh.cache_and_link_flags()
kh.step("build.configure", arch=arch)
kh.sh_with_progress(
    f"cmake -G Ninja -B {BUILD} -S {REPO} -DCMAKE_BUILD_TYPE=Release " + " ".join(flags))
with kh.build_heartbeat("build.crispasr"):
    kh.sh_with_progress(f"cmake --build {BUILD} --target crispasr -j{kh.safe_build_jobs(gpu=True)}")
CLI = f"{BUILD}/bin/crispasr"
assert os.path.isfile(CLI), f"missing {CLI}"

# ── model + long single-pass clip ──────────────────────────────────────────
kh.resolve_hf_token()  # public repo, but sets up auth if a secret exists
from huggingface_hub import hf_hub_download  # noqa: E402
kh.step("model.download", repo=MODEL_REPO, file=MODEL_FILE)
MODEL = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE,
                        local_dir="/kaggle/working/model")

CLIP = "/kaggle/working/long.wav"
listtxt = "/kaggle/working/cat.txt"
with open(listtxt, "w") as f:
    for _ in range(REPEAT):
        f.write(f"file '{REPO}/samples/jfk.wav'\n")
kh.sh(f"ffmpeg -y -f concat -safe 0 -i {listtxt} -c copy {CLIP}")
dur = float(subprocess.run(
    f"ffprobe -v error -show_entries format=duration -of csv=p=0 {CLIP}",
    shell=True, capture_output=True, text=True).stdout.strip() or 0)
kh.step("clip.ready", duration_s=round(dur, 1))


# ── GPU peak-memory poller (nvidia-smi) ────────────────────────────────────
def gpu_used_mib():
    try:
        r = subprocess.run(
            "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
            shell=True, capture_output=True, text=True, timeout=10)
        return max(int(x) for x in r.stdout.split() if x.strip().isdigit())
    except Exception:
        return None


class PeakPoller(threading.Thread):
    def __init__(self, interval=0.2):
        super().__init__(daemon=True)
        self.interval, self.peak, self._stop = interval, 0, threading.Event()

    def run(self):
        while not self._stop.is_set():
            u = gpu_used_mib()
            if u is not None and u > self.peak:
                self.peak = u
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()
        self.join(timeout=2)


def run_cfg(label, extra_env, cli_args):
    env = dict(os.environ)
    env["CRISPASR_PARAKEET_STREAM_THRESHOLD"] = "99999"  # force single-pass
    env["CRISPASR_FC_MEM_DEBUG"] = "1"
    env.update(extra_env)
    idle = gpu_used_mib() or 0
    poll = PeakPoller()
    poll.start()
    t0 = time.time()
    proc = subprocess.run(
        [CLI, "-m", MODEL, "-f", CLIP] + cli_args,
        env=env, capture_output=True, text=True)
    wall = time.time() - t0
    poll.stop()
    T = None
    for line in proc.stderr.splitlines():
        if "fc-encode" in line or "ENGAGED" in line:
            for tok in line.split():
                if tok.startswith("T="):
                    T = tok[2:].rstrip(":")
    text = proc.stdout.strip()
    peak_delta = max(0, poll.peak - idle)
    kh.step(f"run.{label}", wall_s=round(wall, 1), gpu_peak_mib=poll.peak,
            gpu_delta_mib=peak_delta, T=T, chars=len(text), rc=proc.returncode)
    return dict(label=label, wall=wall, peak=poll.peak, delta=peak_delta,
                T=T, text=text, rc=proc.returncode, stderr=proc.stderr[-2000:])


# ── the three configs ──────────────────────────────────────────────────────
cfgs = [
    ("masked_full_local", {"CRISPASR_FC_WINDOWED_ATTN": "0"}, ["--att-context", ATT]),
    ("windowed_local",     {"CRISPASR_FC_WINDOWED_ATTN": "1"}, ["--att-context", ATT]),
    ("full_attention",     {}, []),
    # CUDA uses the MANUAL attention path (fc_gpu_manual_attn default-on), which
    # materializes O(T²) scores+BD — the reporter's ~2 GiB. tiled_full computes the
    # bias one query-block at a time (O(T·block)); this is the config where the win
    # should appear on CUDA (it does NOT on Metal, where flash already avoids scores).
    ("tiled_full",         {"CRISPASR_FC_TILED_ATTN": "1"}, []),
]
results = [run_cfg(*c) for c in cfgs]

# ── verdict ────────────────────────────────────────────────────────────────
by = {r["label"]: r for r in results}
mf, win = by["masked_full_local"], by["windowed_local"]
full, tiled = by["full_attention"], by["tiled_full"]
parity = (mf["text"] == win["text"]) and mf["rc"] == 0 and win["rc"] == 0
mem_win = mf["peak"] - win["peak"]
speedup = (mf["wall"] / win["wall"]) if win["wall"] else 0
# tiled must be bit-exact to full attention; the memory win is the CUDA payoff.
tiled_parity = (full["text"] == tiled["text"]) and full["rc"] == 0 and tiled["rc"] == 0
tiled_mem_win = full["peak"] - tiled["peak"]

print("\n" + "=" * 72)
print(f"CUDA windowed-attn A/B  (arch={arch}, clip={dur:.0f}s, att={ATT})")
print("=" * 72)
for r in results:
    print(f"  {r['label']:20s} T={str(r['T']):>6}  wall={r['wall']:6.1f}s  "
          f"gpu_peak={r['peak']:6d} MiB  chars={len(r['text']):5d}  rc={r['rc']}")
print("-" * 72)
print(f"  PARITY windowed==masked_full : {'IDENTICAL' if parity else 'DIFFER'}")
print(f"  GPU peak masked_full - windowed : {mem_win:+d} MiB "
      f"({mf['peak']} -> {win['peak']})")
print(f"  speedup windowed vs masked_full : {speedup:.2f}x "
      f"({mf['wall']:.1f}s -> {win['wall']:.1f}s)")
print(f"  PARITY tiled==full_attention : {'IDENTICAL' if tiled_parity else 'DIFFER'}")
print(f"  GPU peak full - tiled (CUDA payoff) : {tiled_mem_win:+d} MiB "
      f"({full['peak']} -> {tiled['peak']})")
if not parity or not tiled_parity:
    print("\n  [!] a transcript pair differs — first 300 chars:")
    if not parity:
        print("   masked_full:", mf["text"][:300])
        print("   windowed   :", win["text"][:300])
    if not tiled_parity:
        print("   full       :", full["text"][:300])
        print("   tiled      :", tiled["text"][:300])
print("=" * 72)

kh.step("verdict", parity=parity, gpu_mem_win_mib=mem_win, speedup=round(speedup, 2),
        tiled_parity=tiled_parity, tiled_gpu_mem_win_mib=tiled_mem_win)
kh.step("script.done")
# non-zero exit on parity failure so the kernel status flags it
sys.exit(0 if (parity and tiled_parity) else 1)
