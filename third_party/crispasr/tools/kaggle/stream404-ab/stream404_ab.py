# ─────────────────────────── cell 0 (markdown) ───────────────────────────
# # CrispASR — #404 streaming-knob A/B on a quiet box (CPU wall + GPU arms)
#
# The exact streaming fixes from the #404 resolution (slice memo +
# tail-capped partials, merged 7eee0412, gated OFF) were proven correct on
# the VPS (finals AND partials byte-identical across arms) but its shared
# load voided every wall clock. This kernel reruns the A/B on an
# UNCONTENDED machine and adds the GPU arms, where per-decode length —
# the tail cap's lever — dominates instead of the CPU's weights-bandwidth
# constant.
#
# Streams (built in-kernel, s16le 16 kHz):
#   jfk2    — samples/jfk.wav x2 with 0.9 s gaps (~24 s, 6 short utterances)
#             exercises the SLICE MEMO (closed slices re-decoded per step)
#   longutt — a ~12 s continuous sentence synthesized in-kernel with
#             chatterbox-turbo (seed 42, GPU) and resampled 24k->16k —
#             exercises the TAIL CAP (one long growing utterance; VAD-split
#             samples like jfk never grow past the 6 s commit trigger)
#
# Arms per stream x device (cpu = --no-gpu -t 4, gpu = default CUDA):
#   off       (both gates off — shipped default)
#   memo      CRISPASR_STREAM_SLICE_MEMO=1
#   memotail  memo + --stream-partial-tail-sec 4
#
# Per run: wall, decode count, total decoded audio-seconds (from
# `cohere: transcribe started audio=..s` lines — load-independent
# telemetry), finals text. PASS requires finals byte-equal to that
# stream+device's off arm (proof-of-work: a fabricated speedup from
# skipped decodes cannot hide, #81/4a). Medians of N_MEASURED interleaved
# rounds for wall.
#
# Model: cohere-transcribe-q4_k (~550 MB) via the registry auto-download.
# Datasets: ${KAGGLE_ACCOUNT}/crispasr-hf-token, ${KAGGLE_ACCOUNT}/crispasr-ccache.

# ─────────────────────────── cell 1 (code) — config ──────────────────────
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/tmp/stream404-ab")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
MODELS = TMP / "models"
STREAMS = TMP / "streams"
RESULTS = WORK / "results"
for d in (MODELS, STREAMS, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

CRISPASR_REPO = os.environ.get("CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git")
CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
N_MEASURED = int(os.environ.get("STREAM404_AB_N", "3"))

STREAM_FLAGS = ["--stream", "--stream-json", "--vad", "--auto-download",
                "--stream-step", "2000", "--stream-length", "12000",
                "--stream-partial-decode-ms", "4000",
                "--stream-final-on-silence-ms", "600"]

ARMS = {
    "off": {"env": {}, "flags": []},
    "memo": {"env": {"CRISPASR_STREAM_SLICE_MEMO": "1"}, "flags": []},
    "memotail": {"env": {"CRISPASR_STREAM_SLICE_MEMO": "1"},
                 "flags": ["--stream-partial-tail-sec", "4"]},
}


def step(name, **kv):
    print(f"[{time.strftime('%H:%M:%S')}] STEP {name} " + json.dumps(kv), flush=True)


def run(cmd, check=True, env=None, cwd=None, timeout=None):
    e = {**os.environ, **(env or {})}
    r = subprocess.run(cmd, env=e, cwd=cwd, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.stdout:
        print(r.stdout, flush=True)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(map(str, cmd))}")
    return r


# ─────────────────────────── cell 2 (code) — clone + CUDA build ───────────
step("start", ref=CRISPASR_REF)
if REPO.exists():
    shutil.rmtree(REPO)
run(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF, "--recursive", CRISPASR_REPO, str(REPO)])

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
step("cloned", sha=subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip())

run(["nvidia-smi", "-L"])
gpu_name = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
step("gpu", gpu=gpu_name)

kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
BUILD.mkdir(parents=True, exist_ok=True)
cmake_args = [
    "cmake", "-S", str(REPO), "-B", str(BUILD),
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=ON",
] + kh.cuda_build_flags(arch) + kh.crispasr_cmake_flags() + kh.cache_and_link_flags()
run(cmake_args)
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli -j{kh.safe_build_jobs(gpu=True)}")
step("build_done")

CLI = BUILD / "examples" / "cli" / "crispasr"
if not CLI.exists():
    cands = [c for c in BUILD.rglob("crispasr") if c.is_file() and os.access(c, os.X_OK)]
    if not cands:
        raise SystemExit("crispasr binary not found after build")
    CLI = cands[0]
os.environ["LD_LIBRARY_PATH"] = f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
os.environ["CRISPASR_CACHE_DIR"] = str(MODELS)
step("cli", path=str(CLI))

# ─────────────────────────── cell 3 (code) — streams + model warm ─────────
import numpy as np  # noqa: E402
import wave  # noqa: E402


def wav_pcm16(path: Path):
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1, path
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


jfk = wav_pcm16(REPO / "samples" / "jfk.wav")
gap = np.zeros(int(0.9 * 16000), dtype=np.int16)
np.concatenate([jfk, gap, jfk, gap]).tofile(STREAMS / "jfk2.s16le")

# longutt: one continuous sentence, no VAD-splittable pauses — the tail
# cap's engagement case (same sentence as the local proof on main 7eee0412).
LONG_TEXT = ("The committee reviewed the annual report in detail and concluded "
             "that the proposed infrastructure changes would require substantially "
             "more coordination between the engineering and operations departments "
             "than originally anticipated during the planning phase.")
long_wav = STREAMS / "longutt-24k.wav"
run([str(CLI), "--backend", "chatterbox-turbo", "-m", "auto", "--auto-download",
     "--seed", "42", "--tts", LONG_TEXT, "--tts-output", str(long_wav), "-t", "4"],
    timeout=3600)
from scipy.signal import resample_poly  # noqa: E402
with wave.open(str(long_wav), "rb") as w:
    assert w.getframerate() == 24000 and w.getnchannels() == 1
    a24 = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
a16 = resample_poly(a24.astype(np.float32), 2, 3)
tail_sil = np.zeros(16000, dtype=np.int16)  # 1 s silence so the final fires
np.concatenate([np.clip(a16, -32768, 32767).astype(np.int16), tail_sil]).tofile(
    STREAMS / "longutt.s16le")
step("streams", jfk2_s=round((len(jfk) + len(gap)) * 2 / 16000, 1),
     longutt_s=round(len(a16) / 16000 + 1, 1))

# Warm the model download once (registry pulls cohere-transcribe-q4_k).
run([str(CLI), "--backend", "cohere", "-m", "auto", "--auto-download", "-l", "en",
     "-t", "4", "--no-prints", "--no-gpu", "-d", "1000", str(REPO / "samples" / "jfk.wav")],
    timeout=3600)
step("model_warm")

AUDIO_RE = re.compile(r"transcribe started\s+n_samples=\d+\s+audio=([\d.]+)s")


def stream_run(stream: Path, lang: str, device: str, arm: str, tag: str, timeout=5400):
    a = ARMS[arm]
    env = {"CRISPASR_COHERE_BENCH": "1", **a["env"]}
    cmd = [str(CLI), "--backend", "cohere", "-m", "auto", "-l", lang, "-t", "4",
           "--no-prints"] + STREAM_FLAGS + a["flags"]
    if device == "cpu":
        cmd.append("--no-gpu")
    t0 = time.time()
    with open(stream, "rb") as fin:
        r = subprocess.run(cmd, stdin=fin, env={**os.environ, **env}, timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    wall = round(time.time() - t0, 2)
    if r.returncode != 0:
        print(r.stderr[-4000:], flush=True)
        raise SystemExit(f"{tag}: rc={r.returncode} — FAIL, not timed")
    finals = []
    for line in r.stdout.splitlines():
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("type") == "final":
            finals.append(ev["text"])
    decoded = [float(m.group(1)) for m in AUDIO_RE.finditer(r.stderr)]
    if not finals or not decoded:
        raise SystemExit(f"{tag}: no finals ({len(finals)}) or no decodes ({len(decoded)}) — FAIL")
    return {"wall_s": wall, "n_decodes": len(decoded),
            "decoded_audio_s": round(sum(decoded), 2), "finals": finals}


# ─────────────────────────── cell 4 (code) — the matrix ───────────────────
report = {"gpu": gpu_name, "n_measured": N_MEASURED, "cases": {}}
CASES = [("jfk2", STREAMS / "jfk2.s16le", "en"), ("longutt", STREAMS / "longutt.s16le", "en")]

for cname, stream, lang in CASES:
    for device in ("gpu", "cpu"):
        key = f"{cname}/{device}"
        step("case_start", case=key)
        # warmup one off-arm run (page cache, GPU clocks)
        stream_run(stream, lang, device, "off", f"{key}/warmup")
        runs = {arm: [] for arm in ARMS}
        n = N_MEASURED if device == "gpu" or cname == "jfk2" else max(1, N_MEASURED - 2)
        for i in range(n):
            for arm in ARMS:  # interleaved: off, memo, memotail per round
                runs[arm].append(stream_run(stream, lang, device, arm, f"{key}/{arm}/m{i}"))
                step("run", case=key, arm=arm, i=i, wall=runs[arm][-1]["wall_s"],
                     decoded_s=runs[arm][-1]["decoded_audio_s"], n_dec=runs[arm][-1]["n_decodes"])
        ref_finals = runs["off"][0]["finals"]
        case = {}
        for arm in ARMS:
            walls = [x["wall_s"] for x in runs[arm]]
            case[arm] = {
                "median_wall_s": statistics.median(walls),
                "walls": walls,
                "decoded_audio_s": runs[arm][0]["decoded_audio_s"],
                "n_decodes": runs[arm][0]["n_decodes"],
                "finals_equal_off": all(x["finals"] == ref_finals for x in runs[arm]),
            }
        case["off"]["finals_text"] = [t[:160] for t in ref_finals]
        report["cases"][key] = case
        step("case_done", case=key,
             off=case["off"]["median_wall_s"], memo=case["memo"]["median_wall_s"],
             memotail=case["memotail"]["median_wall_s"],
             memo_equal=case["memo"]["finals_equal_off"],
             memotail_equal=case["memotail"]["finals_equal_off"])

(RESULTS / "stream404-ab.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
step("done")
