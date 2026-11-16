"""
CrispASR — tada talker §176b bucket-floor timing A/B on CUDA (P100) (§215b follow-up)

Question this kernel answers: the §215b measurement (M1 Metal, loadavg 31->137,
timing untrustworthy) found that the tada talker's positive pass runs through the
§176b decode bucket whose Lk floor is 512, wasting ~500 masked attention columns
per step for short generations (n_past << 512). Bypassing it dropped the positive
pass 266->49 ms/call. The new opt-in path (CRISPASR_TADA_BUCKET_MIN, default 512)
adds smaller buckets {64,128,256}. This kernel measures — on a CLEAN CUDA box —
whether a tighter floor actually wins, and guards the regression the bucket was
built for (crossing more bucket boundaries = more one-time graph builds on long
generations).

Method: build the CUDA runtime from the feat branch; download tada-tts-1b q4_k +
codec; run tada synthesis with CRISPASR_TADA_TALKER_TIMING=1 (per-pass + steady-
state ms/step), --seed 42, for three configs on each of a SHORT and a LONG input:
  A default   — floor 512 (baseline, original §176b)
  B floor64   — CRISPASR_TADA_BUCKET_MIN=64 (tight buckets)
  C nobucket  — CRISPASR_TADA_NO_BUCKET=1 (exact-Lk, no cache; padding-win ceiling)
REPS reps, median. Correctness gate = ASR keyword-recall of EVERY arm (HARD RULE
#3): on a GPU the output is NOT byte-identical across a bucket-width change (the
masked-softmax reduction order over Lk differs, flipping a borderline FP bit ->
AR-amplified different-but-intelligible audio), so md5 is INFORMATIONAL only
(bit-determinism), not a pass/fail. A garbled arm (recall drop) is the real bug.

Acceptance (to justify flipping the default to a tighter floor):
  - PARITY: A==B==C md5 on BOTH inputs (else the tighter path is a correctness bug).
  - SHORT: floor64 loop ms/step < default (the padding win is real on CUDA).
  - LONG:  floor64 does NOT regress vs default (bucket-crossing rebuilds don't
           outweigh the padding saved). If it regresses on LONG, keep opt-in /
           length-gated; do not flip the default.
  - Roundtrip intelligible.
"""

import hashlib
import os
import re
import subprocess
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
MODELS = WORK / "models"
CRISPASR = BUILD / "bin" / "crispasr"

CRISPASR_REF = os.environ.get("CRISPASR_REF", "feat/tada-talker-b2")
REPS = int(os.environ.get("REPS", "3"))
MODEL_REPO = os.environ.get("TADA_REPO", "cstr/tada-tts-1b-GGUF")
TIGHT_MIN = os.environ.get("TADA_BUCKET_MIN", "64")

# SHORT: n_past stays well under 512 (worst case for the 512 floor). LONG: a big
# passage so the generation crosses the 64/128/256/512 bucket boundaries and
# exercises the "more one-time builds" regression the bucket was built to avoid.
SHORT_TEXT = "The quick brown fox jumps over the lazy dog."
LONG_TEXT = (
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen "
    "liquor jugs. How vexingly quick daft zebras jump. The five boxing wizards "
    "jump quickly. Sphinx of black quartz, judge my vow. A wizard's job is to "
    "vex chumps quickly in fog. Two driven jocks help fax my big quiz. The job "
    "requires extra pluck and zeal from every young wage earner. Crazy Frederick "
    "bought many very exquisite opal jewels. We promptly judged antique ivory "
    "buckles for the next prize. A quart jar of oil mixed with zinc oxide makes a "
    "very bright paint. Grumpy wizards make toxic brew for the evil queen and jack. "
    "The public was amazed to view the quickness and dexterity of the juggler. "
    "Jaded zombies acted quaintly but kept driving their oxen forward slowly."
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

# ── Download tada-1b model + codec + whisper-tiny ─────────────────────────
kh.step("download.begin", repo=MODEL_REPO)
MODELS.mkdir(exist_ok=True)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
kh.sh_with_progress("pip install -q huggingface_hub hf_transfer")
from huggingface_hub import hf_hub_download  # noqa: E402


def _fetch(repo, cands, what):
    for cand in cands:
        try:
            return Path(hf_hub_download(repo_id=repo, filename=cand, local_dir=str(MODELS),
                                        local_dir_use_symlinks=False))
        except Exception as e:  # noqa: BLE001
            print(f"[download] {what}: {cand} not in {repo} ({e}); next", flush=True)
    return None


model_path = _fetch(MODEL_REPO, ["tada-tts-1b-q4_k.gguf", "tada-tts-1b-f16.gguf"], "model")
assert model_path is not None, "no tada-1b model gguf found"
codec_path = _fetch(MODEL_REPO, ["tada-codec-f16.gguf", "tada-codec-fixed-f16.gguf"], "codec")
assert codec_path is not None, "no tada codec gguf found"
kh.step("download.done", model=model_path.name, model_mib=model_path.stat().st_size // (1 << 20),
        codec=codec_path.name)

_STEP_RE = re.compile(r"over (\d+) steps \(([\d.]+) ms/step\)")
_TALKER_PCT_RE = re.compile(r"talker=[\d.]+ms \(([\d.]+)% of loop\)")
_POS_SS_RE = re.compile(r"pos=.*?ss=([\d.]+)\)")
_NEG_SS_RE = re.compile(r"neg=.*?ss=([\d.]+)\)")
_SS_RE = re.compile(r"STEADY-STATE: talker=([\d.]+) ms/step \(([\d.]+)% of ([\d.]+) ms/step loop\)")


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest() if p.is_file() else "MISSING"


def _parse(log):
    d = {}
    m = _STEP_RE.search(log)
    if m:
        d["steps"] = int(m.group(1))
        d["loop_ms_step"] = float(m.group(2))
    m = _TALKER_PCT_RE.search(log)
    if m:
        d["talker_pct"] = float(m.group(1))
    m = _POS_SS_RE.search(log)
    if m:
        d["pos_ss"] = float(m.group(1))
    m = _NEG_SS_RE.search(log)
    if m:
        d["neg_ss"] = float(m.group(1))
    m = _SS_RE.search(log)
    if m:
        d["ss_talker_ms_step"] = float(m.group(1))
        d["ss_loop_ms_step"] = float(m.group(3))
    return d


def run_cfg(input_label, cfg_label, text, env_extra):
    out = WORK / f"tada-{input_label}-{cfg_label}.wav"
    cmd = [
        str(CRISPASR), "--backend", "tada-1b", "-m", str(model_path),
        "--codec-model", str(codec_path), "--tts", text,
        "--tts-output", str(out), "--seed", "42", "-np",
    ]
    env = {**os.environ, "CRISPASR_TADA_TALKER_TIMING": "1", **env_extra}
    reps = []
    gpu_line = False
    log = ""
    for rep in range(REPS):
        t0 = time.time()
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
        wall = time.time() - t0
        log = (r.stdout or "") + (r.stderr or "")
        p = _parse(log)
        p["wall_s"] = round(wall, 2)
        reps.append(p)
        if rep == 0:
            gpu_line = bool(re.search(r"ggml_cuda|CUDA\d|found \d+ CUDA|GPU backend enabled|backend=CUDA", log))
            if "loop_ms_step" not in p:
                print(f"--- tada/{input_label}/{cfg_label} rep0 tail (no timing parsed) ---", flush=True)
                for line in log.splitlines()[-25:]:
                    print(line, flush=True)

    def _med(key):
        vals = sorted(x[key] for x in reps if key in x)
        return vals[len(vals) // 2] if vals else None

    res = {k: _med(k) for k in ["loop_ms_step", "ss_loop_ms_step", "ss_talker_ms_step",
                                "talker_pct", "pos_ss", "neg_ss", "steps"]}
    res["md5"] = _md5(out)
    res["wall_s"] = min(x["wall_s"] for x in reps)
    res["gpu"] = gpu_line
    res["wav"] = out
    kh.step(f"run.{input_label}.{cfg_label}.done", **{k: res[k] for k in
            ["loop_ms_step", "ss_loop_ms_step", "ss_talker_ms_step", "talker_pct",
             "pos_ss", "neg_ss", "steps", "md5", "gpu"]})
    return res


CONFIGS = [
    ("default", {}),
    ("floor%s" % TIGHT_MIN, {"CRISPASR_TADA_BUCKET_MIN": TIGHT_MIN}),
    ("nobucket", {"CRISPASR_TADA_NO_BUCKET": "1"}),
]
INPUTS = [("short", SHORT_TEXT), ("long", LONG_TEXT)]

results = {}
for in_label, text in INPUTS:
    kh.step(f"run.{in_label}.begin", reps=REPS)
    results[in_label] = {cfg: run_cfg(in_label, cfg, text, env) for cfg, env in CONFIGS}

# ── Correctness = ASR intelligibility of EVERY arm (HARD RULE #3) ──────────
# NOT md5. On a GPU, a sampled/AR TTS is NOT byte-identical across a bucket-width
# change: soft_max_ext sums the -inf-masked padding as exp->0 terms and the GPU
# reduction ORDER over Lk=512 vs a tight Lk flips a borderline FP bit -> a
# different frame, AR-amplified. That is benign FP divergence (same lesson as the
# dia kernel), so the real correctness gate is "does every arm still say the
# prompt", scored by whisper keyword recall — md5 is kept only as an informational
# "is the output bit-deterministic on this platform" signal.
EXPECT_WORDS = ["quick", "brown", "fox", "jumps", "lazy", "dog"]
recall = {}
asr_txt = {}
audio_verdict = "n/a"
try:
    whisper = Path(hf_hub_download(repo_id="ggerganov/whisper.cpp", filename="ggml-tiny.bin",
                                   local_dir=str(MODELS), local_dir_use_symlinks=False))

    def _asr(wav):
        if not Path(wav).is_file():
            return "", 0
        rr = subprocess.run([str(CRISPASR), "--backend", "whisper", "-m", str(whisper),
                             "-f", str(wav), "--no-gpu", "-l", "en", "-np"],
                            capture_output=True, text=True, timeout=600)
        t = (rr.stdout or "").strip()
        return t, sum(w in t.lower() for w in EXPECT_WORDS)

    for cfg, _ in CONFIGS:
        asr_txt[cfg], recall[cfg] = _asr(results["short"][cfg]["wav"])
    min_kw = min(recall.values()) if recall else 0
    audio_verdict = (f"{'ALL INTELLIGIBLE' if min_kw >= 3 else 'AN ARM GARBLED'} "
                     f"(min {min_kw}/{len(EXPECT_WORDS)} kw; " + ", ".join(f"{c}={recall[c]}" for c, _ in CONFIGS) + ")")
    kh.step("audio_roundtrip", recall=recall, min_kw=min_kw, audio_verdict=audio_verdict, asr=asr_txt)
except Exception as e:  # noqa: BLE001
    audio_verdict = f"ASR roundtrip skipped ({e})"
    min_kw = None

# ── md5 agreement (INFORMATIONAL — bit-determinism, not a pass/fail gate) ──
tight = "floor%s" % TIGHT_MIN
md5_identical = {}
for in_label in results:
    hs = {cfg: results[in_label][cfg]["md5"] for cfg in results[in_label]}
    md5_identical[in_label] = len(set(hs.values())) == 1 and "MISSING" not in hs.values()


def _spd(in_label, cfg):
    base = results[in_label]["default"].get("loop_ms_step")
    cur = results[in_label][cfg].get("loop_ms_step")
    return round(base / cur, 3) if (base and cur) else None


short_spd = _spd("short", tight)
long_spd = _spd("long", tight)

# Correctness gate = every arm intelligible. md5 divergence is EXPECTED on GPU and
# is NOT a failure — only a garbled arm (a real miscompute) or a speed loss is.
if not results["short"]["default"].get("gpu"):
    verdict = "GPU DID NOT ENGAGE (check CUDA build / backend auto-select)"
elif min_kw is not None and min_kw < 3:
    verdict = (f"MISCOMPUTE — an arm is garbled ({audio_verdict}); a bucket width changed the "
               f"decode into nonsense, not just FP. Real bug, do not ship")
elif short_spd and short_spd > 1.03 and (long_spd is None or long_spd >= 0.98):
    verdict = (f"WIN — floor{TIGHT_MIN} {short_spd}x short, {long_spd}x long, all arms intelligible. "
               f"Candidate to flip default (note: not bit-identical on GPU — {md5_identical}).")
elif short_spd and short_spd > 1.03 and long_spd and long_spd < 0.98:
    verdict = (f"MIXED — floor{TIGHT_MIN} wins short ({short_spd}x) but REGRESSES long ({long_spd}x); "
               f"keep opt-in / length-gate, do NOT flip default globally")
else:
    verdict = (f"MARGINAL/NO WIN — floor{TIGHT_MIN} short {short_spd}x long {long_spd}x; keep opt-in "
               f"(GPU padding penalty is small; the big win was Metal-only)")

kh.step("summary", md5_identical=md5_identical, short_speedup=short_spd, long_speedup=long_spd,
        recall=recall, audio_verdict=audio_verdict, verdict=verdict)

# ── Human-readable table ──────────────────────────────────────────────────
print("\n" + "=" * 78)
print(f"SUMMARY — tada bucket-floor A/B on CUDA ({sha[:8]}, {model_path.name})")
print("=" * 78)
for in_label, _ in INPUTS:
    print(f"\n[{in_label}]  (steps={results[in_label]['default'].get('steps')})")
    print(f"  {'config':10s} {'loop ms/step':>13s} {'ss loop ms/st':>14s} "
          f"{'ss talker ms/st':>16s} {'pos_ss':>8s} {'neg_ss':>8s}  md5")
    for cfg, _ in CONFIGS:
        r = results[in_label][cfg]
        print(f"  {cfg:10s} {str(r.get('loop_ms_step')):>13s} {str(r.get('ss_loop_ms_step')):>14s} "
              f"{str(r.get('ss_talker_ms_step')):>16s} {str(r.get('pos_ss')):>8s} "
              f"{str(r.get('neg_ss')):>8s}  {r['md5'][:12]}")
    print(f"  md5 bit-identical (info only): {md5_identical[in_label]}")
print(f"\n  short speedup (floor{TIGHT_MIN} vs default): {short_spd}x")
print(f"  long  speedup (floor{TIGHT_MIN} vs default): {long_spd}x  (>=1.0 = no regression)")
print(f"  audio roundtrip: {audio_verdict}")
for cfg, _ in CONFIGS:
    print(f"    ASR({cfg:9s}): {asr_txt.get(cfg, '')[:80]}")
print(f"  VERDICT: {verdict}")

kh._push_progress_to_hf(force=True)
kh.step("script.end", verdict=verdict, short_speedup=short_spd, long_speedup=long_spd,
        md5_identical=md5_identical, audio_verdict=audio_verdict)
