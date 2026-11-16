# ─────────────────────────── cell 0 (markdown) ───────────────────────────
# # CrispASR — Voxtral-4B-TTS diff harness vs mudler reference C (>99% check)
#
# Rigorous stage-by-stage correctness of the CrispASR voxtral-tts runtime
# against the MIT reference C (github.com/mudler/voxtral-tts.c, validated vs
# vLLM-Omni), on a clean box. Same text + seed + voice; both dump per-frame
# |h| and the 37 codes; we diff them.
#
# Isolates STRUCTURE from QUANT: runs the runtime at F16 (should be >99% vs the
# BF16 reference — proves no structural bug) and Q4_K (shows quant drift).
#
# Stages compared:
#   1. LLM h — per-frame |h| relative error (F16 vs BF16 ref)
#   2. FM codes — semantic (code[0]) exact-match %, acoustic (1..36) exact% + ±1%
#   3. Codec — runtime codec vs reference codec on the SAME ref codes → PCM max|Δ|
# Plus a step-count perf sweep (8/7/6/5) on the GPU.
#
# Datasets: ${KAGGLE_ACCOUNT}/crispasr-hf-token, ${KAGGLE_ACCOUNT}/crispasr-ccache,
#           (reference C source is vendored in the repo under refsrc/).
# Needs GPU + Internet + ~30 GB disk (ref model 8 GB + F16 8 GB + Q4_K 2.3 GB).
# HF token MUST have accepted mistralai/Voxtral-4B-TTS-2603 (gated CC-BY-NC).

# ─────────────────────────── cell 1 (code) — config ──────────────────────
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import wave
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
REFBUILD = WORK / "ref"
# ~18 GB of model downloads (BF16 ref 8 GB + F16 GGUF 8 GB + Q4_K 2.3 GB) must go on
# the ~70 GB ephemeral /tmp layer, NOT the ~20 GB /kaggle/working mount (would ENOSPC).
TMP = Path("/tmp/vtts-diff")
MODELS = TMP / "models"
REFMODEL = TMP / "refmodel"
RESULTS = WORK / "results"  # small (dumps, wavs, summary.json) — keep as downloadable output
for d in (REFBUILD, MODELS, REFMODEL, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

CRISPASR_REPO = os.environ.get("CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git")
CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
ORIG_REPO = "mistralai/Voxtral-4B-TTS-2603"
GGUF_REPO = "cstr/voxtral-4b-tts-GGUF"

TEXT = os.environ.get("VTTS_TEXT", "Hello world.")
VOICE = "neutral_female"
SEED = 42


def step(name, **kv):
    print(f"[{time.strftime('%H:%M:%S')}] STEP {name} " + json.dumps(kv)[:400], flush=True)


def run(cmd, check=True, env=None, cwd=None, timeout=None, quiet=False):
    e = {**os.environ, **(env or {})}
    r = subprocess.run(cmd, env=e, cwd=cwd, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.stdout and not quiet:
        print(r.stdout[-4000:], flush=True)
    if check and r.returncode != 0:
        raise SystemExit(f"cmd failed ({r.returncode}): {' '.join(map(str, cmd))}")
    return r


# ─────────────────────────── cell 2 (code) — clone + build crispasr ───────
step("start", ref=CRISPASR_REF)
if REPO.exists():
    shutil.rmtree(REPO)
run(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF, "--recursive", CRISPASR_REPO, str(REPO)])
sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
step("cloned", sha=sha)
run(["nvidia-smi", "-L"])
gpu_name = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()

kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
BUILD.mkdir(parents=True, exist_ok=True)
run(["cmake", "-S", str(REPO), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON"]
    + kh.cuda_build_flags(arch) + kh.cache_and_link_flags())
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli -j{kh.safe_build_jobs(gpu=True)}")
CLI = BUILD / "examples" / "cli" / "crispasr"
if not CLI.exists():
    CLI = next(c for c in BUILD.rglob("crispasr") if c.is_file() and os.access(c, os.X_OK))
os.environ["LD_LIBRARY_PATH"] = f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
step("crispasr_built", cli=str(CLI))

# ─────────────────────────── cell 3 (code) — build reference C (BLAS/CPU) ─
run(["bash", "-lc", "apt-get install -y -q libopenblas-dev >/dev/null 2>&1 || sudo apt-get install -y -q libopenblas-dev"],
    check=False)
# Reference source is vendored in the repo (MIT, under the diff-harness tool dir) so it
# travels with the clone — no flaky external dataset dependency.
REFSRC = REPO / "tools" / "kaggle" / "voxtral-diff-harness" / "refsrc"
if not (REFSRC / "voxtral_tts.c").exists():
    raise SystemExit(f"vendored reference source missing at {REFSRC}")
step("refsrc", path=str(REFSRC))
for f in REFSRC.iterdir():
    if f.suffix in (".c", ".h") or f.name == "Makefile":
        shutil.copy(f, REFBUILD / f.name)
run(["make", "blas", "-j2"], cwd=str(REFBUILD))
REFBIN = REFBUILD / "voxtral_tts"
if not REFBIN.exists():
    raise SystemExit("reference binary not built")
step("ref_built", bin=str(REFBIN))

# ─────────────────────────── cell 4 (code) — download models ──────────────
from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402
TOKEN = kh.resolve_hf_token()

step("dl_refmodel_start")
snapshot_download(ORIG_REPO, local_dir=str(REFMODEL), token=TOKEN or None,
                  allow_patterns=["consolidated.safetensors", "params.json", "tekken.json", "voice_embedding/*"])
step("dl_refmodel_done", files=sorted(os.listdir(REFMODEL)))

GGUF = {}
# VTTS_DIFF_FULL=1 also compares Q4_K + runs the perf sweep; default (unset) is the fast
# F16-vs-reference structural diagnosis only (Q4_K + perf already captured in a prior run).
_FULL = os.environ.get("VTTS_DIFF_FULL") == "1"
_gguf_list = [("f16", "voxtral-4b-tts-f16.gguf")] + ([("q4_k", "voxtral-4b-tts-q4_k.gguf")] if _FULL else [])
for tag, fn in _gguf_list:
    try:
        GGUF[tag] = hf_hub_download(GGUF_REPO, fn, local_dir=str(MODELS), token=TOKEN or None)
        step("dl_gguf", tag=tag, size_gb=round(os.path.getsize(GGUF[tag]) / 1e9, 2))
    except Exception as ex:  # noqa: BLE001
        step("dl_gguf_fail", tag=tag, err=str(ex)[:200])

# ─────────────────────────── cell 5 (code) — run + parse dumps ────────────
_FRAME_RE = re.compile(r"(?:REF|MINE) frame (\d+) \|h\|=([\d.]+) codes=([\d,]+)")


def parse_frames(text):
    """→ {frame: (|h|, [37 codes])}"""
    out = {}
    for m in _FRAME_RE.finditer(text):
        codes = [int(x) for x in m.group(3).rstrip(",").split(",") if x != ""]
        out[int(m.group(1))] = (float(m.group(2)), codes)
    return out


def run_reference(out_wav, timeout=3600):
    env = {"VTTS_DUMP": "1"}
    cmd = [str(REFBIN), "-d", str(REFMODEL), "-v", VOICE, "-s", str(SEED), "-o", str(out_wav), TEXT]
    r = subprocess.run(cmd, env={**os.environ, **env}, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (RESULTS / "ref.dump.log").write_text(r.stdout)
    return parse_frames(r.stdout)


def run_runtime(gguf, out_wav, extra_env=None, timeout=1800):
    env = {"CRISPASR_VOXTRAL_TTS_DEBUG": "1", **(extra_env or {})}
    cmd = [str(CLI), "--backend", "voxtral-tts", "-m", gguf, "--seed", str(SEED),
           "--tts", TEXT, "--voice", VOICE, "--tts-output", str(out_wav)]
    r = subprocess.run(cmd, env={**os.environ, **env}, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (RESULTS / f"{Path(out_wav).stem}.dump.log").write_text(r.stdout)
    return parse_frames(r.stdout)


step("run_reference")
ref = run_reference(RESULTS / "ref.wav")
step("ref_frames", n=len(ref))
mine = {}
for tag in ("f16", "q4_k"):
    if tag in GGUF:
        step("run_runtime", tag=tag)
        mine[tag] = run_runtime(GGUF[tag], RESULTS / f"mine_{tag}.wav")
        step("mine_frames", tag=tag, n=len(mine[tag]))


# ─────────────────────────── cell 6 (code) — stage diffs ──────────────────
def diff_codes(a, b):
    """a,b = {frame:(|h|,codes)}. Compare over common frames."""
    common = sorted(set(a) & set(b))
    if not common:
        return {"common_frames": 0}
    h_rel, sem_hit, ac_exact, ac_pm1, ac_total = [], 0, 0, 0, 0
    for f in common:
        ha, ca = a[f]
        hb, cb = b[f]
        if ha > 0:
            h_rel.append(abs(hb - ha) / ha)
        if ca and cb and ca[0] == cb[0]:
            sem_hit += 1
        for i in range(1, min(len(ca), len(cb))):
            ac_total += 1
            if ca[i] == cb[i]:
                ac_exact += 1
            if abs(ca[i] - cb[i]) <= 1:
                ac_pm1 += 1
    h_rel.sort()
    # Per-frame trail (printed to stdout so it lands in the always-downloadable kernel log)
    # + first-divergence localization: is frame 0 (prefill-only, no feedback) already off,
    # or does divergence start later (acoustic-feedback cascade)?
    per_frame, first_sem_div, first_ac_div = [], None, None
    for f in common:
        ha, ca = a[f]
        hb, cb = b[f]
        ac_ex = sum(1 for i in range(1, min(len(ca), len(cb))) if ca[i] == cb[i])
        per_frame.append({"f": f, "hrel": round(abs(hb - ha) / ha, 4) if ha else None,
                          "ref_sem": ca[0], "mine_sem": cb[0], "ac_exact": f"{ac_ex}/36"})
        if first_sem_div is None and ca[0] != cb[0]:
            first_sem_div = f
        if first_ac_div is None and any(ca[i] != cb[i] for i in range(1, min(len(ca), len(cb)))):
            first_ac_div = f
    return {
        "common_frames": len(common),
        "ref_frames": len(a), "mine_frames": len(b),
        "h_rel_median": round(h_rel[len(h_rel) // 2], 5) if h_rel else None,
        "h_rel_max": round(max(h_rel), 5) if h_rel else None,
        "semantic_exact_pct": round(100 * sem_hit / len(common), 2),
        "acoustic_exact_pct": round(100 * ac_exact / ac_total, 2) if ac_total else None,
        "acoustic_pm1_pct": round(100 * ac_pm1 / ac_total, 2) if ac_total else None,
        "first_semantic_divergence_frame": first_sem_div,
        "first_acoustic_divergence_frame": first_ac_div,
        "per_frame_first12": per_frame[:12],
    }


summary = {"gpu": gpu_name, "arch": arch, "sha": sha, "text": TEXT, "seed": SEED, "voice": VOICE,
           "ref_frames": len(ref), "stage2_codes": {}, "stage3_codec": {}, "perf": {}}
for tag in mine:
    summary["stage2_codes"][tag] = diff_codes(ref, mine[tag])
    step("stage2", tag=tag, **summary["stage2_codes"][tag])


# ─────────────────────────── cell 7 (code) — stage 3: codec cross-check ───
def wav_pcm(path):
    with wave.open(str(path), "rb") as w:
        n = w.getnframes()
        return list(struct.unpack(f"<{n}h", w.readframes(n)))


if ref:
    codes_file = RESULTS / "ref_codes.txt"
    with open(codes_file, "w") as fh:
        for f in sorted(ref):
            fh.write(",".join(str(c) for c in ref[f][1]) + "\n")
    # reference codec on ref codes
    try:
        subprocess.run([str(REFBIN), "-d", str(REFMODEL), "-v", VOICE, "-o", str(RESULTS / "ref_codec.wav"), TEXT],
                       env={**os.environ, "VTTS_CODEC_FROM_FILE": str(codes_file)}, timeout=900,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # runtime codec on the SAME ref codes (use whichever gguf we have)
        g = GGUF.get("f16") or GGUF.get("q4_k")
        subprocess.run([str(CLI), "--backend", "voxtral-tts", "-m", g, "--tts", TEXT, "--voice", VOICE,
                        "--tts-output", str(RESULTS / "mine_codec.wav")],
                       env={**os.environ, "CRISPASR_VOXTRAL_TTS_CODEC_FROM_FILE": str(codes_file)}, timeout=900,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ra, rb = wav_pcm(RESULTS / "ref_codec.wav"), wav_pcm(RESULTS / "mine_codec.wav")
        n = min(len(ra), len(rb))
        if n:
            maxd = max(abs(ra[i] - rb[i]) for i in range(n)) / 32768.0
            num = sum(ra[i] * rb[i] for i in range(n))
            den = (sum(x * x for x in ra[:n]) ** 0.5) * (sum(x * x for x in rb[:n]) ** 0.5)
            summary["stage3_codec"] = {"n_ref": len(ra), "n_mine": len(rb),
                                       "pcm_max_abs_diff": round(maxd, 5),
                                       "pcm_corr": round(num / den, 6) if den else None}
    except Exception as ex:  # noqa: BLE001
        summary["stage3_codec"] = {"error": str(ex)[:200]}
    step("stage3", **summary["stage3_codec"])


# ─────────────────────────── cell 8 (code) — perf step sweep (VTTS_DIFF_FULL only) ─
_TIM = re.compile(r"LLM-decode\s+([\d.]+)\s+ms/frame,\s+FM\s+([\d.]+)\s+ms/frame")
_FR = re.compile(r"generated\s+(\d+)\s+frames")
g = GGUF.get("q4_k") or GGUF.get("f16")
if g and _FULL:
    for steps in (8, 7, 6, 5):
        env = {"CRISPASR_VOXTRAL_TTS_TIMING": "1"}
        if steps != 8:
            env["CRISPASR_VOXTRAL_TTS_FM_STEPS"] = str(steps)
        vals = []
        fr = None
        for i in range(4):  # 1 warmup + 3
            r = subprocess.run([str(CLI), "--backend", "voxtral-tts", "-m", g, "--seed", str(SEED),
                                "--tts", TEXT, "--voice", VOICE, "--tts-output", str(RESULTS / "perf.wav")],
                               env={**os.environ, **env}, timeout=600, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
            m = _TIM.search(r.stdout)
            f = _FR.search(r.stdout)
            if i > 0 and m:
                vals.append(float(m.group(2)))
            if f:
                fr = int(f.group(1))
        vals.sort()
        summary["perf"][f"steps{steps}"] = {"fm_ms_median": vals[len(vals) // 2] if vals else None, "frames": fr}
    step("perf", **summary["perf"])

# ─────────────────────────── cell 9 (code) — report ───────────────────────
(RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
print("\n" + "=" * 78, flush=True)
print(f"VOXTRAL-TTS DIFF HARNESS vs mudler ref C — {gpu_name}  sha {sha[:8]}", flush=True)
print(f"text={TEXT!r} seed={SEED} voice={VOICE}  ref_frames={len(ref)}", flush=True)
print("=" * 78, flush=True)
print("STAGE 2 (LLM h + FM codes) — runtime vs BF16 reference:", flush=True)
for tag, d in summary["stage2_codes"].items():
    print(f"  {tag:5} | frames {d.get('common_frames')}/{d.get('ref_frames')} | "
          f"|h| relerr med={d.get('h_rel_median')} max={d.get('h_rel_max')} | "
          f"semantic {d.get('semantic_exact_pct')}% exact | "
          f"acoustic {d.get('acoustic_exact_pct')}% exact, {d.get('acoustic_pm1_pct')}% ±1", flush=True)
print("STAGE 2 — first-divergence localization (is frame 0 already off, or a feedback cascade?):", flush=True)
for tag, d in summary["stage2_codes"].items():
    print(f"  {tag}: first semantic divergence @ frame {d.get('first_semantic_divergence_frame')}, "
          f"first acoustic divergence @ frame {d.get('first_acoustic_divergence_frame')}", flush=True)
    print(f"    {'frame':>5} {'|h|rel':>7} {'ref_sem':>8} {'mine_sem':>9} {'ac_exact':>9}", flush=True)
    for pf in d.get("per_frame_first12", []):
        print(f"    {pf['f']:>5} {str(pf['hrel']):>7} {pf['ref_sem']:>8} {pf['mine_sem']:>9} {pf['ac_exact']:>9}", flush=True)
print("STAGE 3 (codec on identical ref codes):", flush=True)
print(f"  {summary['stage3_codec']}", flush=True)
print("PERF (FM ms/frame by ODE steps):", flush=True)
for k, v in summary["perf"].items():
    print(f"  {k}: FM={v['fm_ms_median']} ms/fr, frames={v['frames']}", flush=True)
print("=" * 78, flush=True)
print(json.dumps(summary, indent=2), flush=True)
step("DONE")
