"""CUDA validation: sidon rework, VoxCPM2 VAE guard, and today's music backends.

Three things landed on main without ever running on CUDA. This kernel is the
CUDA arm for all of them, plus benchmark numbers.

  A. sidon (9807c3cab) — the relative-position-bias rework, the bounded DAC
     decode and the padding crop were all validated on Metal only. The bucket
     formulations feed `ggml_get_rows` an index built by an in-graph REPEAT
     (`bucket`) or supplied directly (`bucket-direct`); CUDA asserts
     `src1->nb[0] == ggml_type_size(src1->type)` on that operand where CPU and
     Metal do not, so "correct on Metal" proves nothing here.

  B. voxcpm2 (23847e550, PR #289) — narrowed a `> 500000` sample fallback so
     long VAE graphs stay on CUDA instead of being demoted to CPU. It changes
     the EXISTING TTS decode/encode paths, no test crosses the threshold, and
     it has never run on CUDA. If CUDA does have a practical limit on those
     dispatches this reintroduces a crash for shipped TTS users.

  C. voxcpm2-vae (653f83cee) — brand-new S2S upscaler backend with no published
     GGUF, so nothing has ever executed it. Converted here from
     openbmb/VoxCPM2 `audiovae.pth` (380 MB) + `config.json`.

  D. beat-this / btc-chords / mel-band-roformer — today's music backends, none
     benchmarked on CUDA.

ACCEPTANCE GATE — CPU vs CUDA parity, everywhere.

Judging a beat tracker or chord recogniser by whether it "sounds right" on
synthetic audio conflates model quality with port correctness. The question
this kernel answers is narrower and decidable: does the CUDA path agree with
the CPU path? Every backend therefore runs on BOTH devices and the comparison
is the gate. Musical ground truth (a 120 BPM fixture with a known C-Am-F-G
progression) is reported as a secondary signal, never as a hard failure.

PROOF OF WORK (kaggle_usage.md gotcha #24). A crash that exits fast mints a
fake speedup, so nothing is timed unless it also produced correct output:
non-zero exit or empty/silent output is FAIL and is never reported as a
benchmark number. Durations are asserted against the expected sample counts,
and every restored/synthesised clip is ASR'd back and word-overlap checked.
"""

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time
import wave
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

WORK = Path("/kaggle/working")
# Keep /kaggle/working down to the artifacts we must retrieve: `kaggle kernels
# output` is page-capped at 500 files and does not auto-continue, so a repo
# clone here would bury the results (gotcha #22). Everything bulky lives on the
# ~70 GB ephemeral layer (gotcha #18).
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
REPO = TEMP / "CrispASR"
BUILD = TEMP / "build"
MODELS = TEMP / "models"
FIX = TEMP / "fixtures"
OUT = WORK / "outputs"
CRISPASR_REPO = "https://github.com/CrispStrobe/CrispASR.git"
CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")

for d in (MODELS, FIX, OUT):
    d.mkdir(parents=True, exist_ok=True)

RESULTS = {"tests": [], "bench": [], "env": {}}
_T0 = time.time()


def run(cmd, check=True, env=None, cwd=None, timeout=None, capture=True):
    e = {**os.environ, **(env or {})}
    if capture:
        r = subprocess.run(cmd, env=e, cwd=cwd, timeout=timeout, shell=isinstance(cmd, str),
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    else:
        r = subprocess.run(cmd, env=e, cwd=cwd, timeout=timeout, shell=isinstance(cmd, str))
        r.stdout = ""
    if check and r.returncode != 0:
        print(r.stdout[-8000:], flush=True)
        raise SystemExit(f"command failed ({r.returncode}): {cmd}")
    return r


def record(name, passed, **kw):
    entry = {"test": name, "pass": bool(passed), **kw}
    RESULTS["tests"].append(entry)
    print(("PASS  " if passed else "FAIL  ") + name + "  " + json.dumps(kw), flush=True)
    (WORK / "results.json").write_text(json.dumps(RESULTS, indent=2))
    return passed


# ── cell 1: clone + harness (FULL regime) ──────────────────────────────────
print(json.dumps({"step": "start", "ref": CRISPASR_REF}), flush=True)
if REPO.exists():
    shutil.rmtree(REPO)
run(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF, "--recursive",
     CRISPASR_REPO, str(REPO)], capture=False)
# Belt and braces: a --depth 1 --recursive clone can leave a submodule short.
# main needs BOTH ggml and third_party/c2pa-audio present for cmake to generate
# (a ggml-only init fails with "Cannot find c2pa_native.cpp"); this build passes
# -DCRISPASR_NO_C2PA_NATIVE=ON but the checkout should still be complete.
run(["git", "-C", str(REPO), "submodule", "update", "--init", "--recursive"],
    check=False, capture=False, timeout=1800)

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
if str(REPO / "tools" / "kaggle") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
kh.step("cloned", sha=sha)
RESULTS["env"]["sha"] = sha
RESULTS["env"]["ref"] = CRISPASR_REF

run(["nvidia-smi", "-L"], check=False)
gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
kh.step("gpu", gpu=gpu)
RESULTS["env"]["gpu"] = gpu

# ── cell 2: build ──────────────────────────────────────────────────────────
kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
kh.step("cuda_arch", arch=arch)
RESULTS["env"]["cuda_arch"] = arch

BUILD.mkdir(parents=True, exist_ok=True)
cmake_args = [
    "cmake", "-S", str(REPO), "-B", str(BUILD),
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=ON",
    # Benchmark build: skip the c2pa native submodule (harness regime note).
    "-DCRISPASR_NO_C2PA_NATIVE=ON",
] + kh.cuda_build_flags(arch) + kh.cache_and_link_flags()
run(cmake_args, capture=False)
kh.step("cmake_done")

with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli "
        f"-j{kh.safe_build_jobs(gpu=True)}")
kh.step("build_done")

CLI = BUILD / "examples" / "cli" / "crispasr"
if not CLI.exists():
    cands = [c for c in BUILD.rglob("crispasr") if c.is_file() and os.access(c, os.X_OK)]
    if not cands:
        raise SystemExit("crispasr binary not found after build")
    CLI = cands[0]
os.environ["LD_LIBRARY_PATH"] = f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
kh.step("cli", path=str(CLI))

# ── cell 3: downloads ──────────────────────────────────────────────────────
# resolve_hf_token BEFORE any HF pull; hf_transfer wedges multi-GB Kaggle
# downloads with no resume, so use curl -C - instead.
token = kh.resolve_hf_token()
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
kh.step("hf_token", have=bool(token))


def hf_get(repo, filename, dest, repo_type="model"):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        kh.step("download.cached", file=dest.name)
        return dest
    prefix = "" if repo_type == "model" else f"{repo_type}s/"
    url = f"https://huggingface.co/{prefix}{repo}/resolve/main/{filename}"
    hdr = ["-H", f"Authorization: Bearer {token}"] if token else []
    run(["curl", "-fL", "-C", "-", "--retry", "5", "--retry-delay", "5",
         "-o", str(dest), *hdr, url], capture=False, timeout=3600)
    kh.step("download.done", file=dest.name, mb=round(dest.stat().st_size / 1e6, 1))
    return dest


M = {}
M["sidon"] = hf_get("cstr/Sidon-GGUF", "sidon-v0.1-q8_0.gguf", MODELS / "sidon-q8_0.gguf")
M["voxcpm2"] = hf_get("cstr/voxcpm2-GGUF", "voxcpm2-q4_k.gguf", MODELS / "voxcpm2-q4_k.gguf")
M["beat"] = hf_get("cstr/beat-this-GGUF", "beat-this-f16.gguf", MODELS / "beat-this-f16.gguf")
M["btc"] = hf_get("cstr/btc-chords-GGUF", "btc-chords-large-f16.gguf", MODELS / "btc-chords-large-f16.gguf")
M["sep"] = hf_get("cstr/mel-band-roformer-vocals-GGUF", "mel-band-roformer-vocals-f16.gguf",
                  MODELS / "mel-band-roformer-vocals-f16.gguf")
M["asr"] = hf_get("ggerganov/whisper.cpp", "ggml-base.en.bin", MODELS / "ggml-base.en.bin")
# VAE-only source: config.json + audiovae.pth is all convert --vae-only reads.
VAESRC = MODELS / "voxcpm2-src"
VAESRC.mkdir(parents=True, exist_ok=True)
hf_get("openbmb/VoxCPM2", "config.json", VAESRC / "config.json")
hf_get("openbmb/VoxCPM2", "audiovae.pth", VAESRC / "audiovae.pth")
kh.step("downloads_done")

# ── cell 4: convert the VAE-only GGUF (backend C has no published artifact) ─
M["vae"] = MODELS / "voxcpm2-vae-f32.gguf"
run([sys.executable, "-m", "pip", "install", "-q", "gguf"], check=False, capture=False)
r = run([sys.executable, str(REPO / "models" / "convert-voxcpm2-to-gguf.py"),
         "--input", str(VAESRC), "--output", str(M["vae"]), "--vae-only"],
        check=False, timeout=1800)
vae_ok = M["vae"].exists() and M["vae"].stat().st_size > 0
record("C0.convert_vae_only", vae_ok,
       mb=round(M["vae"].stat().st_size / 1e6, 1) if vae_ok else 0,
       tail=r.stdout[-600:] if not vae_ok else "")
kh.step("vae_converted", ok=vae_ok)

# ── cell 5: fixtures ───────────────────────────────────────────────────────
import numpy as np  # noqa: E402

SR = 16000
BPM = 120.0
BEAT = 60.0 / BPM           # 0.5 s
BARS, BEATS_PER_BAR = 8, 4
PROGRESSION = ["C", "Am", "F", "G", "C", "Am", "F", "G"]  # one chord per bar
CHORD_SEMIS = {"C": [0, 4, 7], "Am": [9, 12, 16], "F": [5, 9, 12], "G": [7, 11, 14]}


def midi_hz(n):
    return 440.0 * (2.0 ** ((n - 69) / 12.0))


def make_music():
    """120 BPM, 8 bars of 4/4, one chord per bar over kick+snare.

    Doubles as ground truth for two backends: beat-this should find a 0.5 s
    inter-beat interval with a downbeat every 4th, and btc should track
    C-Am-F-G. Synthetic, so it is a sanity signal, not a hard gate.
    """
    total = BARS * BEATS_PER_BAR * BEAT
    t = np.arange(int(total * SR)) / SR
    x = np.zeros_like(t)
    for bar in range(BARS):
        root = CHORD_SEMIS[PROGRESSION[bar]]
        b0 = bar * BEATS_PER_BAR * BEAT
        seg = (t >= b0) & (t < b0 + BEATS_PER_BAR * BEAT)
        env = 0.12 * np.exp(-1.2 * (t[seg] - b0))
        for semi in root:
            x[seg] += env * np.sin(2 * np.pi * midi_hz(48 + semi) * t[seg])
        for b in range(BEATS_PER_BAR):
            on = b0 + b * BEAT
            i0 = int(on * SR)
            # kick: decaying low sine; accented on the downbeat
            n = int(0.12 * SR)
            k = np.arange(n) / SR
            amp = 0.9 if b == 0 else 0.6
            x[i0:i0 + n] += amp * np.exp(-28 * k) * np.sin(2 * np.pi * 62 * k)
            if b % 2 == 1:  # snare on 2 and 4
                n2 = int(0.06 * SR)
                rng = np.random.default_rng(1234 + bar * 4 + b)
                k2 = np.arange(n2) / SR
                x[i0:i0 + n2] += 0.35 * np.exp(-45 * k2) * rng.standard_normal(n2)
    return np.clip(x / (np.abs(x).max() + 1e-9) * 0.85, -1, 1)


def read_wav(p):
    with wave.open(str(p), "rb") as w:
        n, sr, ch = w.getnframes(), w.getframerate(), w.getnchannels()
        d = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float64) / 32768.0
    if ch > 1:
        d = d.reshape(-1, ch).mean(1)
    return d, sr


def write_wav(p, x, sr=SR):
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


jfk, jfk_sr = read_wav(REPO / "samples" / "jfk.wav")
assert jfk_sr == SR, f"expected 16 kHz jfk.wav, got {jfk_sr}"
music = make_music()
write_wav(FIX / "music.wav", music)

# Speech over a music bed: separation must pull the speech back out.
n = max(len(music), len(jfk))
mix = np.zeros(n)
mix[:len(music)] += 0.55 * music
mix[:len(jfk)] += 0.85 * jfk
write_wav(FIX / "mix.wav", mix / (np.abs(mix).max() + 1e-9) * 0.9)
write_wav(FIX / "jfk16k.wav", jfk)
kh.step("fixtures", music_s=round(len(music) / SR, 2), mix_s=round(n / SR, 2))

JFK_WORDS = ("and so my fellow americans ask not what your country can do for you "
             "ask what you can do for your country").split()


def wav_stats(p):
    p = Path(p)
    if not p.exists() or p.stat().st_size == 0:
        return None
    x, sr = read_wav(p)
    return {"n": len(x), "sr": sr, "sec": round(len(x) / sr, 3),
            "peak": round(float(np.abs(x).max()), 5),
            "rms": round(float(np.sqrt((x ** 2).mean())), 6),
            "sha": hashlib.sha256(p.read_bytes()).hexdigest()[:16]}


def asr(path, device_cpu=False):
    cmd = [str(CLI), "-m", str(M["asr"]), "-f", str(path), "--no-prints"]
    if device_cpu:
        cmd.append("--no-gpu")
    r = run(cmd, check=False, timeout=1800)
    txt = " ".join(ln.split("]")[-1] for ln in r.stdout.splitlines() if "-->" in ln)
    return " ".join(txt.lower().replace(",", " ").replace(".", " ").split())


def overlap(text, words=JFK_WORDS):
    if not text:
        return 0.0
    got = set(text.split())
    return round(sum(1 for w in words if w in got) / len(words), 3)


# ── cell 6: A — sidon on CUDA ──────────────────────────────────────────────
# The bucket formulations feed get_rows an index that CUDA (unlike CPU/Metal)
# requires to be contiguous. A miscompile here shows up as a crash or garbage,
# so every mode is ASR'd, not just checked for exit 0.
SID_ENVCOMMON = {"CRISPASR_SIDON_DEBUG": "1"}


def sidon_run(tag, out, extra_env, cpu=False, timeout=5400):
    env = {**SID_ENVCOMMON, **extra_env}
    cmd = [str(CLI), "--s2s", "-m", str(M["sidon"]), "-f", str(FIX / "jfk16k.wav"),
           "--s2s-output", str(out)]
    if cpu:
        cmd.append("--no-gpu")
    t = time.time()
    r = run(cmd, check=False, env=env, timeout=timeout)
    wall = time.time() - t
    ws = {}
    for ln in r.stdout.splitlines():
        if "workspace" in ln and "TOTAL" in ln:
            parts = ln.split()
            ws[parts[2]] = float(parts[-2])
    st = wav_stats(out)
    kh.step(f"sidon.{tag}", rc=r.returncode, ok=bool(st), wall=round(wall, 1))
    return {"rc": r.returncode, "wall_s": round(wall, 2), "stats": st,
            "workspace_mib": ws, "tail": r.stdout[-1500:] if r.returncode != 0 else ""}


sid = {}
for mode in ["expand", "bucket", "bucket-direct"]:
    sid[mode] = sidon_run(mode, OUT / f"sidon_{mode}.wav", {"CRISPASR_SIDON_RPE": mode})

# every mode must run AND transcribe; identical transcripts is the parity gate
sid_txt = {}
for mode in sid:
    sid_txt[mode] = asr(OUT / f"sidon_{mode}.wav") if sid[mode]["stats"] else ""
ok = all(sid[m]["rc"] == 0 and sid[m]["stats"] and sid[m]["stats"]["peak"] > 0.01 for m in sid)
record("A1.sidon_rpe_modes_run_on_cuda", ok,
       **{m: {"rc": sid[m]["rc"], "sec": (sid[m]["stats"] or {}).get("sec"),
              "wall_s": sid[m]["wall_s"], "workspace_mib": sid[m]["workspace_mib"]} for m in sid})
record("A2.sidon_rpe_modes_asr_identical",
       ok and len({sid_txt[m] for m in sid_txt}) == 1,
       transcripts={m: sid_txt[m][:120] for m in sid_txt},
       overlap={m: overlap(sid_txt[m]) for m in sid_txt})

# chunked (default 512) vs whole-utterance (0): asserted BIT-EXACT on Metal
sid["nochunk"] = sidon_run("nochunk", OUT / "sidon_nochunk.wav",
                           {"CRISPASR_SIDON_DECODER_CHUNK_FRAMES": "0"})
# Bit-exactness is the right invariant only where the kernels are deterministic
# AND shape-independent. That holds on Metal/CPU; it does NOT hold on CUDA,
# where cuDNN/cuBLAS pick different tiles and reduction orders for a 595-frame
# window than for a 2825-frame whole-utterance decode. So report the MAGNITUDE
# of the difference, not just whether the hashes match — the first run of this
# test reported only SHAs, which proved the outputs differ but could not
# distinguish 1 ULP from a broken decode. Never assert an equality you cannot
# also measure the violation of.
def _max_abs_diff(pa, pb):
    try:
        xa, _ = read_wav(pa)
        xb, _ = read_wav(pb)
        n = min(len(xa), len(xb))
        if n == 0:
            return None, None
        d = np.abs(xa[:n] - xb[:n])
        return float(d.max()), float(d.mean())
    except Exception:
        return None, None


mad, mean_d = _max_abs_diff(OUT / "sidon_bucket-direct.wav", OUT / "sidon_nochunk.wav")
same_sha = (sid["bucket-direct"]["stats"] and sid["nochunk"]["stats"]
            and sid["bucket-direct"]["stats"]["sha"] == sid["nochunk"]["stats"]["sha"])
# int16 LSB = 3.05e-5. Anything at or below a couple of LSBs is float-kernel
# noise, not a decode error; a join failure showed 0.476 on Metal.
BIT_EXACT_TOL = 1e-4
ok_a3 = bool(same_sha) or (mad is not None and mad <= BIT_EXACT_TOL)
record("A3.sidon_chunked_vs_whole_equivalent", ok_a3,
       bit_exact=bool(same_sha), max_abs_diff=mad, mean_abs_diff=mean_d,
       tolerance=BIT_EXACT_TOL, int16_lsb=round(1 / 32768, 8),
       chunked=(sid["bucket-direct"]["stats"] or {}).get("sha"),
       whole=(sid["nochunk"]["stats"] or {}).get("sha"),
       chunked_ws=sid["bucket-direct"]["workspace_mib"], whole_ws=sid["nochunk"]["workspace_mib"],
       note="bit-exact on Metal/CPU; CUDA picks shape-dependent kernels so only "
            "a tight numeric bound is assertable there")

# lookahead: without it the final ~12 ms is a full-scale transient
sid["nolook"] = sidon_run("nolook", OUT / "sidon_nolook.wav", {"CRISPASR_SIDON_LOOKAHEAD": "0"})


def tail_peak(p):
    x, sr = read_wav(p)
    return round(float(np.abs(x[-int(0.012 * sr):]).max()), 5)


# Record unconditionally. The first run guarded this behind "both runs produced
# stats" and one did not, so A4 SILENTLY VANISHED from the results — a skipped
# assertion is indistinguishable from one that was never written, which is the
# same missing-gate criticism levelled at #289's live test. A missing input is a
# FAIL with a reason, never an absence.
if sid["bucket-direct"]["stats"] and sid["nolook"]["stats"]:
    tp_on, tp_off = tail_peak(OUT / "sidon_bucket-direct.wav"), tail_peak(OUT / "sidon_nolook.wav")
    record("A4.sidon_lookahead_kills_tail_transient", tp_on < 0.05,
           tail_peak_with_lookahead=tp_on, tail_peak_without=tp_off)
else:
    record("A4.sidon_lookahead_kills_tail_transient", False,
           reason="a required run produced no output",
           bucket_direct_rc=sid["bucket-direct"]["rc"], nolook_rc=sid["nolook"]["rc"],
           nolook_tail=sid["nolook"]["tail"][-400:])

# CPU arm — the actual parity gate for the graph rework
sid["cpu"] = sidon_run("cpu", OUT / "sidon_cpu.wav", {}, cpu=True)
cpu_txt = asr(OUT / "sidon_cpu.wav") if sid["cpu"]["stats"] else ""
record("A5.sidon_cpu_vs_cuda_asr_parity",
       bool(cpu_txt) and cpu_txt == sid_txt.get("bucket-direct", ""),
       cpu=cpu_txt[:120], cuda=sid_txt.get("bucket-direct", "")[:120],
       cpu_overlap=overlap(cpu_txt), cuda_overlap=overlap(sid_txt.get("bucket-direct", "")),
       cpu_wall_s=sid["cpu"]["wall_s"], cuda_wall_s=sid["bucket-direct"]["wall_s"])
RESULTS["bench"].append({"backend": "sidon", "device": "cuda",
                         "wall_s": sid["bucket-direct"]["wall_s"],
                         "audio_s": (sid["bucket-direct"]["stats"] or {}).get("sec"),
                         "workspace_mib": sid["bucket-direct"]["workspace_mib"]})
RESULTS["bench"].append({"backend": "sidon", "device": "cpu", "wall_s": sid["cpu"]["wall_s"],
                         "audio_s": (sid["cpu"]["stats"] or {}).get("sec")})

# ── cell 7: B — voxcpm2 CUDA VAE guard (>500000 samples) ───────────────────
SHORT_TEXT = "Good morning. This is a short control sentence."
# ~10.4 s at 48 kHz is the 500000-sample threshold; this is comfortably past it.
LONG_TEXT = (
    "The quick brown fox jumps over the lazy dog while the sun sets slowly behind "
    "the distant mountains. Every valley echoes with the sound of running water, and "
    "the evening air carries the scent of pine and woodsmoke across the open fields. "
    "Travellers returning home speak of long roads, quiet villages, and the steady "
    "rhythm of the seasons turning once again toward winter."
)


def tts(tag, text, out, env, timeout=5400, cpu=False):
    cmd = [str(CLI), "-m", str(M["voxcpm2"]), "--tts", text, "--tts-output", str(out)]
    if cpu:
        cmd.append("--no-gpu")
    t = time.time()
    r = run(cmd, check=False, env=env, timeout=timeout)
    wall = time.time() - t
    st = wav_stats(out)
    kh.step(f"voxcpm2.{tag}", rc=r.returncode, ok=bool(st), wall=round(wall, 1))
    return {"rc": r.returncode, "wall_s": round(wall, 2), "stats": st,
            "tail": r.stdout[-2000:] if r.returncode != 0 else ""}


vx = {}
vx["long_graph1"] = tts("long_graph1", LONG_TEXT, OUT / "voxcpm2_long_graph1.wav",
                        {"CRISPASR_VOXCPM2_USE_GRAPH": "1"})
vx["long_graph0"] = tts("long_graph0", LONG_TEXT, OUT / "voxcpm2_long_graph0.wav",
                        {"CRISPASR_VOXCPM2_USE_GRAPH": "0"})
vx["short_graph1"] = tts("short_graph1", SHORT_TEXT, OUT / "voxcpm2_short_graph1.wav",
                         {"CRISPASR_VOXCPM2_USE_GRAPH": "1"})

# The whole point: the long run must exceed 500000 samples, or the narrowed
# guard was never reached and a green result would be meaningless.
n_long = (vx["long_graph1"]["stats"] or {}).get("n", 0)
record("B1.long_synth_crosses_500k_guard", n_long > 500000,
       n_samples=n_long, sec=(vx["long_graph1"]["stats"] or {}).get("sec"),
       threshold=500000,
       note="below threshold => the CUDA carve-out was NOT exercised")
record("B2.voxcpm2_long_graph1_on_cuda", vx["long_graph1"]["rc"] == 0 and n_long > 0,
       rc=vx["long_graph1"]["rc"], wall_s=vx["long_graph1"]["wall_s"],
       stats=vx["long_graph1"]["stats"], tail=vx["long_graph1"]["tail"][-500:])

vx_txt = {k: (asr(OUT / f"voxcpm2_{k}.wav") if vx[k]["stats"] else "") for k in vx}
LONG_KEY = {"quick", "brown", "fox", "mountains", "winter", "water", "evening"}


def keyword_hit(text, keys):
    got = set(text.split())
    return round(len(keys & got) / len(keys), 3)


record("B3.voxcpm2_long_graph1_intelligible",
       keyword_hit(vx_txt["long_graph1"], LONG_KEY) >= 0.5,
       keyword_hit=keyword_hit(vx_txt["long_graph1"], LONG_KEY),
       transcript=vx_txt["long_graph1"][:300])
record("B4.voxcpm2_graph1_vs_graph0_agree",
       keyword_hit(vx_txt["long_graph1"], LONG_KEY) >= keyword_hit(vx_txt["long_graph0"], LONG_KEY) - 0.2,
       graph1_hit=keyword_hit(vx_txt["long_graph1"], LONG_KEY),
       graph0_hit=keyword_hit(vx_txt["long_graph0"], LONG_KEY),
       graph1_sec=(vx["long_graph1"]["stats"] or {}).get("sec"),
       graph0_sec=(vx["long_graph0"]["stats"] or {}).get("sec"))
# proof-of-work: the long clip must be materially longer than the control
record("B5.duration_scales_with_text",
       n_long > 2 * (vx["short_graph1"]["stats"] or {"n": 1e9})["n"],
       long_n=n_long, short_n=(vx["short_graph1"]["stats"] or {}).get("n"))
for k in vx:
    RESULTS["bench"].append({"backend": "voxcpm2-tts", "case": k, "device": "cuda",
                             "wall_s": vx[k]["wall_s"],
                             "audio_s": (vx[k]["stats"] or {}).get("sec")})

# ── cell 8: C — voxcpm2-vae upscaler (never executed anywhere) ─────────────
if vae_ok:
    def vae_run(tag, out, cpu=False):
        cmd = [str(CLI), "--s2s", "-m", str(M["vae"]), "-f", str(FIX / "jfk16k.wav"),
               "--s2s-output", str(out)]
        if cpu:
            cmd.append("--no-gpu")
        t = time.time()
        r = run(cmd, check=False, timeout=5400)
        wall = time.time() - t
        st = wav_stats(out)
        kh.step(f"vae.{tag}", rc=r.returncode, ok=bool(st), wall=round(wall, 1))
        return {"rc": r.returncode, "wall_s": round(wall, 2), "stats": st,
                "tail": r.stdout[-2000:] if r.returncode != 0 else ""}

    v_cuda = vae_run("cuda", OUT / "vae_cuda.wav")
    v_cpu = vae_run("cpu", OUT / "vae_cpu.wav", cpu=True)
    exp_n = len(jfk) * 3  # documented 3x upsample, 16 kHz -> 48 kHz
    record("C1.vae_upscale_cuda_3x", bool(v_cuda["stats"]) and v_cuda["rc"] == 0
           and abs(v_cuda["stats"]["n"] - exp_n) <= 48000,
           rc=v_cuda["rc"], stats=v_cuda["stats"], expected_n=exp_n,
           wall_s=v_cuda["wall_s"], tail=v_cuda["tail"][-500:])
    t_cuda = asr(OUT / "vae_cuda.wav") if v_cuda["stats"] else ""
    t_cpu = asr(OUT / "vae_cpu.wav") if v_cpu["stats"] else ""
    record("C2.vae_cuda_asr_roundtrip", overlap(t_cuda) >= 0.7,
           overlap=overlap(t_cuda), transcript=t_cuda[:200])
    record("C3.vae_cpu_vs_cuda_parity", bool(t_cpu) and t_cpu == t_cuda,
           cpu=t_cpu[:150], cuda=t_cuda[:150],
           cpu_overlap=overlap(t_cpu), cuda_overlap=overlap(t_cuda))
    for nm, v in (("cuda", v_cuda), ("cpu", v_cpu)):
        RESULTS["bench"].append({"backend": "voxcpm2-vae", "device": nm, "wall_s": v["wall_s"],
                                 "audio_s": (v["stats"] or {}).get("sec")})
else:
    record("C1.vae_upscale_cuda_3x", False, skipped="VAE GGUF conversion failed")

# ── cell 9: D — today's music backends, CPU vs CUDA ────────────────────────
def task_run(tag, model, flags, src, cpu=False, timeout=5400):
    cmd = [str(CLI), "-m", str(model), "-f", str(src), *flags]
    if cpu:
        cmd.append("--no-gpu")
    t = time.time()
    r = run(cmd, check=False, timeout=timeout)
    wall = time.time() - t
    kh.step(f"task.{tag}", rc=r.returncode, wall=round(wall, 1))
    return {"rc": r.returncode, "wall_s": round(wall, 2), "out": r.stdout}


# D1 beat-this — ground truth 120 BPM (0.5 s), downbeat every 4th.
# Text format is "%.3f\t%s" (time, beat|downbeat) with a summary on stderr:
# "crispasr: <file>: N beats (M downbeats), X BPM". stdout and stderr are
# merged here, so parse defensively rather than assuming clean output.
def parse_beats(txt):
    times, downs = [], []
    for ln in txt.splitlines():
        p = ln.split()
        if len(p) >= 2 and p[0].replace(".", "", 1).isdigit():
            times.append(float(p[0]))
            if "downbeat" in ln:
                downs.append(float(p[0]))
    return times, downs


def reported_bpm(txt):
    for ln in txt.splitlines():
        if "BPM" in ln:
            for tok in ln.replace(",", " ").split():
                try:
                    v = float(tok)
                except ValueError:
                    continue
                if 20.0 < v < 300.0:
                    return v
    return None


b_cuda = task_run("beats_cuda", M["beat"], ["--beats"], FIX / "music.wav")
b_cpu = task_run("beats_cpu", M["beat"], ["--beats"], FIX / "music.wav", cpu=True)
bt_c, bd_c = parse_beats(b_cuda["out"])
bt_p, bd_p = parse_beats(b_cpu["out"])
(OUT / "beats_cuda.txt").write_text(b_cuda["out"])
(OUT / "beats_cpu.txt").write_text(b_cpu["out"])
ibi = float(np.median(np.diff(bt_c))) if len(bt_c) > 2 else 0.0
record("D1a.beats_cuda_runs", b_cuda["rc"] == 0 and len(bt_c) > 4,
       rc=b_cuda["rc"], n_beats=len(bt_c), n_downbeats=len(bd_c), wall_s=b_cuda["wall_s"])
record("D1b.beats_cpu_vs_cuda_parity",
       len(bt_c) == len(bt_p) and (len(bt_c) == 0 or
                                   max(abs(a - b) for a, b in zip(bt_c, bt_p)) < 0.02),
       n_cuda=len(bt_c), n_cpu=len(bt_p),
       max_delta_s=(round(max((abs(a - b) for a, b in zip(bt_c, bt_p)), default=0), 4)))
# secondary signal only — synthetic audio is not a fair test of a beat tracker
RESULTS["bench"].append({"backend": "beat-this", "device": "cuda", "wall_s": b_cuda["wall_s"],
                         "audio_s": round(len(music) / SR, 2),
                         "median_ibi_s": round(ibi, 4), "expected_ibi_s": BEAT,
                         "implied_bpm": round(60.0 / ibi, 1) if ibi else None,
                         "reported_bpm": reported_bpm(b_cuda["out"]),
                         "expected_bpm": BPM, "signal_only": True})
RESULTS["bench"].append({"backend": "beat-this", "device": "cpu", "wall_s": b_cpu["wall_s"]})

# D2 btc chords — CC-BY-NC-SA, needs the licence attestation.
# Text format ("%.3f\t%.3f\t%s") rather than json: stdout and stderr are merged
# here, and the json writer pretty-prints across many lines, so interleaved log
# output would break a whole-document parse.
CHORD_FLAGS = ["--chords", "--chords-format", "text", "--accept-license", "cc-by-nc-sa-4.0"]
c_cuda = task_run("chords_cuda", M["btc"], CHORD_FLAGS, FIX / "music.wav")
c_cpu = task_run("chords_cpu", M["btc"], CHORD_FLAGS, FIX / "music.wav", cpu=True)
(OUT / "chords_cuda.txt").write_text(c_cuda["out"])
(OUT / "chords_cpu.txt").write_text(c_cpu["out"])


def parse_chords(txt):
    spans = []
    for ln in txt.splitlines():
        p = ln.rstrip("\n").split("\t")
        if len(p) >= 3:
            try:
                spans.append({"start": float(p[0]), "end": float(p[1]), "chord": p[2].strip()})
            except ValueError:
                continue
    return spans


ch_c, ch_p = parse_chords(c_cuda["out"]), parse_chords(c_cpu["out"])
record("D2a.chords_cuda_runs", c_cuda["rc"] == 0 and len(ch_c) > 0,
       rc=c_cuda["rc"], n_spans=len(ch_c), wall_s=c_cuda["wall_s"],
       tail=c_cuda["out"][-400:] if not ch_c else "")
record("D2b.chords_cpu_vs_cuda_parity",
       json.dumps(ch_c, sort_keys=True) == json.dumps(ch_p, sort_keys=True),
       n_cuda=len(ch_c), n_cpu=len(ch_p))
labels = [c["chord"] for c in ch_c][:12]
RESULTS["bench"].append({"backend": "btc-chords", "device": "cuda", "wall_s": c_cuda["wall_s"],
                         "audio_s": round(len(music) / SR, 2), "labels": labels,
                         "expected_progression": PROGRESSION, "signal_only": True})
RESULTS["bench"].append({"backend": "btc-chords", "device": "cpu", "wall_s": c_cpu["wall_s"]})

# D3 mel-band-roformer — separate speech from the music bed, then ASR the
# vocal stem. Real acceptance criterion: the speech must come back.
SEPDIR = OUT / "sep_cuda"
SEPDIR.mkdir(parents=True, exist_ok=True)
s_cuda = task_run("sep_cuda", M["sep"], ["--separate", "--sep-output-dir", str(SEPDIR)],
                  FIX / "mix.wav")
stems = sorted(p.name for p in SEPDIR.glob("*.wav"))
record("D3a.separate_cuda_runs", s_cuda["rc"] == 0 and len(stems) > 0,
       rc=s_cuda["rc"], stems=stems, wall_s=s_cuda["wall_s"], tail=s_cuda["out"][-500:])
voc = next((p for p in SEPDIR.glob("*.wav") if "vocal" in p.name.lower()), None)
inst = next((p for p in SEPDIR.glob("*.wav")
             if "instrum" in p.name.lower() or "other" in p.name.lower()
             or "accomp" in p.name.lower()), None)
mix_txt = asr(FIX / "mix.wav")
voc_txt = asr(voc) if voc else ""
inst_txt = asr(inst) if inst else ""
record("D3b.vocal_stem_recovers_speech", overlap(voc_txt) >= 0.7,
       vocal_overlap=overlap(voc_txt), mix_overlap=overlap(mix_txt),
       instrumental_overlap=overlap(inst_txt), vocal_transcript=voc_txt[:200])
record("D3c.separation_improves_over_mix", overlap(voc_txt) >= overlap(mix_txt),
       vocal=overlap(voc_txt), mix=overlap(mix_txt))
# The assertion with actual discriminating power. D3a-c ALL pass for a no-op
# separation that copies the input into both stems: the mix already transcribes
# cleanly at this music level, so "vocals recovers speech" and "vocals >= mix"
# are satisfied by a passthrough. Requiring the OTHER stem to be empty of speech
# is what proves separation happened. Verified locally on M1: mix and
# mix_vocals both yield the full JFK line, mix_other yields "(upbeat music)".
record("D3d.instrumental_stem_excludes_speech",
       bool(inst) and overlap(inst_txt) < 0.3,
       instrumental_overlap=overlap(inst_txt), vocal_overlap=overlap(voc_txt),
       instrumental_transcript=inst_txt[:120],
       note="a no-op separation passes D3a-c but fails here")
RESULTS["bench"].append({"backend": "mel-band-roformer", "device": "cuda",
                         "wall_s": s_cuda["wall_s"], "audio_s": round(len(mix) / SR, 2),
                         "stems": stems})

# ── cell 10: summary ───────────────────────────────────────────────────────
n_pass = sum(1 for t in RESULTS["tests"] if t["pass"])
n_tot = len(RESULTS["tests"])
RESULTS["summary"] = {"passed": n_pass, "total": n_tot,
                      "failed": [t["test"] for t in RESULTS["tests"] if not t["pass"]],
                      "elapsed_s": round(time.time() - _T0, 1)}
(WORK / "results.json").write_text(json.dumps(RESULTS, indent=2))
print("\n" + "=" * 70, flush=True)
print(json.dumps(RESULTS["summary"], indent=2), flush=True)
print("=" * 70, flush=True)
for t in RESULTS["tests"]:
    print(("PASS  " if t["pass"] else "FAIL  ") + t["test"], flush=True)
kh.step("done", passed=n_pass, total=n_tot)
if n_pass != n_tot:
    raise SystemExit(f"{n_tot - n_pass} of {n_tot} checks FAILED — see results.json")
