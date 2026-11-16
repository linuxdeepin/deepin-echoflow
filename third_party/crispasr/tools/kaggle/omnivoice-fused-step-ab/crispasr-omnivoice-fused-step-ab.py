"""
CrispASR — OmniVoice stage0 fused step graph A/B on CUDA (#254)

Question this kernel answers: does the fused per-step graph (audio-embedding
lookup + codebook sum + text-embed concat + target-slice logits IN-GRAPH,
token ids as the only per-step upload) close the residual 15-20% RTF gap the
#254 reporter measured vs omnivoice.cpp on CUDA?

The legacy path does, per MaskGIT step: a ~18 MB embedding readback + host
codebook sum + ~5 MB embed re-upload + a full-sequence ~39 MB logits readback
+ single-threaded triple-log-softmax CFG scoring (~13M exp). The fused path
uploads ~140 KB of int32 ids, computes embeddings in-graph, reads back only
the target logit slices, and threads the scoring (rng-order preserved).

Arms (reporter's exact paragraph, q8_0 + tokenizer f16, CODEC_GPU=1):
  legacy : OMNIVOICE_FUSED_STEP=0  (unified CFG auto-on for CUDA)
  fused  : OMNIVOICE_FUSED_STEP=1  (unified CFG auto-on for CUDA)
  fused2 : OMNIVOICE_FUSED_STEP=1 OMNIVOICE_UNIFIED_CFG=0 (is unified still
           the right CUDA default once the host overhead is gone?)

Acceptance: codes byte-identical legacy vs fused (greedy + fixed seed) — hard
gate; then median gen seconds decides. M1 Metal already verified
byte-identical (this kernel is the perf verdict on the reporter's platform
class). Proof-of-work: audio duration must be ~22 s (a crash/no-op mints a
fake win — LEARNINGS 4a).
"""

import os
import re
import subprocess
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
MODELS = WORK / "models"
CRISPASR = BUILD / "bin" / "crispasr"

CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
REPS = int(os.environ.get("REPS", "3"))

TEXT = (
    "CrispASR started as a fork of whisper.cpp and extends that base into a "
    "unified speech engine called crispasr, backed by full ggml C++ runtimes "
    "for major open-weights ASR and TTS architectures. One build, one binary, "
    "one consistent CLI — pick the backend at the command line or let CrispASR "
    "auto-detect it from your GGUF file. See Text-to-Speech for the TTS side."
)


def _sh_preclone(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


print(f"[pre-clone] cloning CrispASR @ {CRISPASR_REF} for shared harness", flush=True)
if not REPO.exists():
    _sh_preclone(
        f"git clone --depth 1 --branch {CRISPASR_REF} --recursive "
        f"https://github.com/CrispStrobe/CrispASR {REPO}"
    )

import sys

sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
if kh.resolve_hf_token():
    print("[auth] HF token resolved", flush=True)
kh.step("script.start", ref=CRISPASR_REF)

sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
kh.step("clone.done", sha=sha, ref=CRISPASR_REF)

# ── Build (CUDA) ──────────────────────────────────────────────────────────
kh.step("build.begin")
kh.install_build_toolchain()
cmake_cmd = (
    f"cmake {REPO} -B{BUILD} -GNinja "
    "-DCMAKE_BUILD_TYPE=Release "
    + " ".join(kh.cuda_build_flags())
    + " "
    + " ".join(kh.cache_and_link_flags())
)
njobs = kh.safe_build_jobs(gpu=True)
with kh.build_heartbeat("cmake-configure"):
    kh.sh_with_progress(cmake_cmd)
kh.step("build.configured")
with kh.build_heartbeat("cmake-build"):
    kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli -- -j{njobs}")
assert CRISPASR.is_file(), "crispasr binary missing after build"
kh.step("build.done", binary=str(CRISPASR))

# ── Download models ───────────────────────────────────────────────────────
kh.step("download.begin")
MODELS.mkdir(exist_ok=True)
kh.sh_with_progress("pip install -q huggingface_hub")
from huggingface_hub import hf_hub_download  # noqa: E402

FILES = [
    ("cstr/omnivoice-GGUF", "omnivoice-q8_0.gguf"),
    ("cstr/omnivoice-GGUF", "omnivoice-tokenizer-f16.gguf"),
]
paths = {}
for repo, fname in FILES:
    p = hf_hub_download(repo_id=repo, filename=fname, local_dir=str(MODELS), local_dir_use_symlinks=False)
    paths[fname] = Path(p)
    kh.step("download.file.done", file=fname, size_mib=Path(p).stat().st_size // (1 << 20))
kh.step("download.done")

TIMING_RX = (
    r"omnivoice: timing — gen ([\d.]+)s \+ decode ([\d.]+)s = ([\d.]+)s "
    r"for ([\d.]+)s audio \(RTF ([\d.]+)"
)


# ── Run one arm ───────────────────────────────────────────────────────────
def run_arm(label: str, env_extra: dict, reps: int = REPS, bench_last: bool = True) -> dict:
    gens, decs, totals, rtfs = [], [], [], []
    audio_s = None
    codes_path = WORK / f"{label}.codes"
    for rep in range(reps):
        env = {
            **os.environ,
            "OMNIVOICE_CODEC_GPU": "1",
            "OMNIVOICE_DUMP_CODES": str(codes_path) if rep == 0 else "",
            **{k: str(v) for k, v in env_extra.items()},
        }
        if bench_last and rep == reps - 1:
            env["OMNIVOICE_BENCH"] = "1"
        cmd = [
            str(CRISPASR), "--threads", "4", "--no-watermark",
            "--backend", "omnivoice-tts",
            "-m", str(paths["omnivoice-q8_0.gguf"]),
            "--codec-model", str(paths["omnivoice-tokenizer-f16.gguf"]),
            "--instruct", "male, middle-aged, low pitch",
            "--tts", TEXT,
            "--tts-output", str(WORK / f"{label}.wav"),
        ]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
        log = (r.stdout or "") + (r.stderr or "")
        m = re.search(TIMING_RX, log)
        if r.returncode != 0 or not m:
            print(f"--- {label} rep{rep} FAILED (rc={r.returncode}) tail ---", flush=True)
            for line in log.splitlines()[-30:]:
                print(line, flush=True)
            kh.step(f"run.{label}.rep{rep}.FAIL", rc=r.returncode)
            continue
        g, d, tot, aud, rtf = (float(m.group(i)) for i in range(1, 6))
        # Proof-of-work: the paragraph must produce ~22 s of audio; a crash or
        # truncated generation would mint a fake RTF win.
        assert aud > 15.0, f"{label}: audio {aud}s too short — not a real run"
        gens.append(g)
        decs.append(d)
        totals.append(tot)
        rtfs.append(rtf)
        audio_s = aud
        if bench_last and rep == reps - 1:
            bench_lines = [ln for ln in log.splitlines() if "omnivoice_bench" in ln]
            agg: dict = {}
            for ln in bench_lines:
                mm = re.search(r"omnivoice_bench:\s+(\S+)\s+([\d.]+) ms", ln)
                if mm:
                    agg.setdefault(mm.group(1), []).append(float(mm.group(2)))
            for stage, vals in agg.items():
                print(f"  [{label}] bench {stage}: n={len(vals)} total={sum(vals):.0f}ms "
                      f"median={sorted(vals)[len(vals)//2]:.1f}ms", flush=True)
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None  # noqa: E731
    res = {
        "gen_med": med(gens), "dec_med": med(decs), "total_med": med(totals),
        "rtf_med": med(rtfs), "audio_s": audio_s, "n_ok": len(gens),
        "codes": codes_path,
    }
    kh.step(f"run.{label}.done", **{k: v for k, v in res.items() if k != "codes"})
    return res


kh.step("run.section.begin", reps=REPS)
legacy = run_arm("legacy", {"OMNIVOICE_FUSED_STEP": 0})
fused = run_arm("fused", {"OMNIVOICE_FUSED_STEP": 1})
fused2 = run_arm("fused2", {"OMNIVOICE_FUSED_STEP": 1, "OMNIVOICE_UNIFIED_CFG": 0})

# ── Byte-identity gate ────────────────────────────────────────────────────
identical = (
    legacy["codes"].exists() and fused["codes"].exists()
    and legacy["codes"].read_bytes() == fused["codes"].read_bytes()
)
identical2 = (
    legacy["codes"].exists() and fused2["codes"].exists()
    and legacy["codes"].read_bytes() == fused2["codes"].read_bytes()
)

speedup = None
if legacy["gen_med"] and fused["gen_med"]:
    speedup = round(legacy["gen_med"] / fused["gen_med"], 3)

verdict = (
    "CORRECTNESS FAIL — do not flip" if not identical else
    ("FLIP TO FUSED" if speedup and speedup > 1.05 else "NO CLEAR WIN — keep opt-in")
)

print("\n" + "=" * 72)
print(f"SUMMARY — OmniVoice stage0 fused step graph A/B ({sha[:8]})")
print("=" * 72)
for name, r in [("legacy", legacy), ("fused", fused), ("fused2(2-fwd)", fused2)]:
    print(f"  {name:14s}: gen={r['gen_med']}s decode={r['dec_med']}s "
          f"total={r['total_med']}s RTF={r['rtf_med']} (n={r['n_ok']}, audio={r['audio_s']}s)")
print(f"  codes legacy==fused : {'BYTE-IDENTICAL' if identical else 'DIFFER (BUG!)'}")
print(f"  codes legacy==fused2: {'BYTE-IDENTICAL' if identical2 else 'DIFFER (BUG!)'}")
print(f"  gen speedup (legacy/fused): {speedup}x")
print(f"  VERDICT: {verdict}")

kh._push_progress_to_hf(force=True)
kh.step("script.end", verdict=verdict, speedup=speedup, identical=identical)
