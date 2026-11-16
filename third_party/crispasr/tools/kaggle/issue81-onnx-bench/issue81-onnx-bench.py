#!/usr/bin/env python3
"""Kaggle GPU kernel: CrispASR full-fleet benchmark + onnx-asr head-to-head (#81).

Tests all CrispASR backends that fit in Kaggle's time/disk budget.
Head-to-head with onnx-asr for overlapping models (whisper, parakeet, canary).
CrispASR-only RTF for the 20+ backends onnx-asr doesn't support.

Push (under $KAGGLE_ACCOUNT):
  export KAGGLE_API_TOKEN=<$KAGGLE_ACCOUNT token>
  python -m kaggle kernels push -p tools/kaggle/issue81-onnx-bench
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
# Clone the repo into TEMP, NOT /kaggle/working. The git-cloned CrispASR/ tree
# (thousands of files, sorts before ccache.tar) was filling `kaggle kernels
# output`'s 500-file page 1 and burying ccache.tar — so it could never be
# retrieved to refresh the dataset. With the repo in TEMP, /kaggle/working holds
# only ccache.tar + benchmark_results.json (+ the always-fetched log), so
# ccache.tar is reachable in page 1.
REPO = TEMP / "CrispASR"

# ── Phase 0: Clone + build CrispASR with CUDA ───────────────────────────────
print("=== Phase 0: clone + build CrispASR ===", flush=True)
if not REPO.exists():
    subprocess.check_call([
        "git", "clone", "--depth", "1", "-b", "main",
        "https://github.com/CrispStrobe/CrispASR", str(REPO),
    ])
if (REPO / "ggml").is_dir() and not (REPO / "ggml" / "CMakeLists.txt").exists():
    subprocess.check_call(["git", "submodule", "update", "--init", "ggml"], cwd=str(REPO))

if (REPO / "tools" / "kaggle").is_dir():
    sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()  # structured progress + heartbeat plumbing (kaggle_usage.md regime)
# 3-tier HF auth (env -> Kaggle secret -> attached ${KAGGLE_ACCOUNT}/crispasr-hf-token
# dataset) so Phase 3's GGUF pulls are authenticated (avoids anon rate limits;
# the token dataset is attached specifically for this).
kh.resolve_hf_token()
# resolve_hf_token() force-enables hf_transfer, which wedges multi-GB Kaggle
# downloads with no resume (kaggle_usage.md / hf-download note). Some GGUFs here
# are ~1-3 GB, so fall back to the plain resumable downloader.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

BUILD = TEMP / "build"
BUILD.mkdir(parents=True, exist_ok=True)

# Install ninja/ccache/mold AND warm ccache from the attached ${KAGGLE_ACCOUNT}/crispasr-ccache
# dataset (kaggle_usage.md #13/#17). Without this the build runs cold (~21 min);
# warm it's ~3 min. cache_and_link_flags() below only sets the compiler-launcher
# flags — it does NOT install or warm ccache, which is what this call does.
kh.install_build_toolchain()  # sets CCACHE_DIR=/kaggle/working/.ccache + warms it

# Relocate the ccache OUT of /kaggle/working. `kaggle kernels output` is page-
# capped at 500 files and `.ccache/` (thousands of tiny files, sorts before
# ccache.tar) fills page 1 — so the ccache.tar we write can never be downloaded.
# Move the warmed cache to /kaggle/temp so the only ccache artifact left in the
# output is the single ccache.tar (retrievable in page 1). We warm from the
# dataset, not from Kaggle's /kaggle/working persistence, so this is safe.
import shutil  # noqa: E402

_ccache_run = (Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")) / ".ccache"
_warmed = Path("/kaggle/working/.ccache")
if _warmed.exists():
    if _ccache_run.exists():
        shutil.rmtree(_ccache_run, ignore_errors=True)
    shutil.move(str(_warmed), str(_ccache_run))
else:
    _ccache_run.mkdir(parents=True, exist_ok=True)
os.environ["CCACHE_DIR"] = str(_ccache_run)
print(f"  CCACHE_DIR relocated to {_ccache_run}", flush=True)

has_cuda = Path("/usr/local/cuda/bin/nvcc").exists()
print(f"  CUDA available: {has_cuda}")

if has_cuda:
    arch = kh.detect_cuda_arch()
    flags = kh.cuda_build_flags(arch) + kh.cache_and_link_flags()
    # C2PA provenance signing is irrelevant to a benchmark; disabling it avoids
    # the third_party/c2pa-audio submodule (the clone below inits only ggml, so
    # requiring c2pa's sources fails cmake generate — the v1 error).
    cmake_flags = "-DCMAKE_BUILD_TYPE=Release -DCRISPASR_NO_C2PA_NATIVE=ON " + " ".join(flags)
    ret = subprocess.call(f"cmake -G Ninja -B {BUILD} -S {REPO} {cmake_flags}", shell=True)
    if ret != 0:
        print("  CUDA cmake failed, falling back to CPU")
        has_cuda = False
        import shutil
        if BUILD.exists(): shutil.rmtree(BUILD); BUILD.mkdir(parents=True, exist_ok=True)

if not has_cuda:
    subprocess.check_call(
        f"cmake -G Ninja -B {BUILD} -S {REPO} -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF "
        "-DCRISPASR_NO_C2PA_NATIVE=ON",
        shell=True, stdout=subprocess.DEVNULL)

n_jobs = min(os.cpu_count() or 2, 4)
with kh.build_heartbeat("cmake.build"):
    # Capture build output and surface the tail on failure. v7 errored here with
    # stdout=DEVNULL, hiding the cause (a transient glint size_t GCC break on main,
    # since fixed) — a silent build failure is undebuggable from the kernel log.
    _b = subprocess.run(f"cmake --build {BUILD} -j{n_jobs} --target crispasr-cli 2>&1",
                        shell=True, capture_output=True, text=True)
    if _b.returncode != 0:
        print("=== crispasr-cli BUILD FAILED — last 40 lines ===", flush=True)
        print("\n".join((_b.stdout or "").splitlines()[-40:]), flush=True)
        raise SystemExit("crispasr-cli build failed")
CRISPASR = BUILD / "bin" / "crispasr"
print(f"  built: {CRISPASR}")

# ── Phase 1: Install onnx-asr (+ onnxruntime-gpu for the CUDA EP) ────────────
print("\n=== Phase 1: install onnx-asr ===", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                        "onnx-asr", "soundfile", "huggingface_hub", "datasets"])
# Force the GPU build: onnx-asr pulls in CPU `onnxruntime`; both ship the same
# `onnxruntime` module, so uninstall then install -gpu. PIN the CUDA-12 wheel:
# Kaggle's P100 worker has CUDA 12.8, but the LATEST onnxruntime-gpu links
# libcudart.so.13 -> ImportError on 12.8 (this crashed v9). onnxruntime-gpu
# 1.19.2 = CUDA 12 + cuDNN 9. Phase 5 is import-guarded, so if this still can't
# load (e.g. cuDNN mismatch) the kernel skips onnx cleanly and keeps the CrispASR
# numbers rather than crashing.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-gpu"],
               capture_output=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnxruntime-gpu==1.19.2"])

# ── Phase 2: Prepare audio ──────────────────────────────────────────────────
print("\n=== Phase 2: prepare audio ===", flush=True)
import soundfile as sf
import numpy as np

jfk_wav = REPO / "samples" / "jfk.wav"
pcm, sr = sf.read(str(jfk_wav), dtype="float32")
duration = len(pcm) / sr

# LONG clip = REAL VARIED speech, NOT jfk×5. Repeating one 11 s clip lets the
# model/CUDA caches reuse work across identical repeats, inflating the RTF
# (especially on GPU) — a fake "long-audio" number. Concatenate DIFFERENT
# LibriSpeech utterances (16 kHz) up to ~120 s with 0.3 s silences between, so
# every frame is novel content. Falls back to jfk×5 only if the dataset can't load.
long_wav = TEMP / "long_real.wav"
try:
    from datasets import load_dataset
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    gap = np.zeros(int(0.3 * 16000), dtype="float32")
    chunks, tot = [], 0.0
    for row in ds:
        a = np.asarray(row["audio"]["array"], dtype="float32")
        chunks.append(a)
        chunks.append(gap)
        tot += (len(a) + len(gap)) / 16000
        if tot >= 120:
            break
    long_pcm = np.concatenate(chunks)
    sf.write(str(long_wav), long_pcm, 16000)
    long_dur = len(long_pcm) / 16000
    print(f"  JFK: {duration:.1f}s (real), Long: {long_dur:.1f}s (real, {len(chunks)//2} distinct LibriSpeech utterances)")
except Exception as e:  # noqa: BLE001
    print(f"  LibriSpeech load failed ({str(e)[:120]}); falling back to jfk×5", flush=True)
    repeats = max(1, int(60 / duration))
    long_pcm = np.tile(pcm, repeats)
    sf.write(str(long_wav), long_pcm, sr)
    long_dur = len(long_pcm) / sr
    print(f"  JFK: {duration:.1f}s, Long: {long_dur:.1f}s (FALLBACK jfk×{repeats})")

# ── Phase 3: Download all GGUF models ───────────────────────────────────────
print("\n=== Phase 3: download models ===", flush=True)
from huggingface_hub import hf_hub_download

MODELS_DIR = TEMP / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Models to benchmark: (backend, hf_repo, filename, label)
CRISPASR_MODELS = [
    # Head-to-head with onnx-asr.
    # NOTE: parakeet-ctc is a pure CTC (EncDecCTCModelBPE) — it must run on the
    # fastconformer-ctc backend, NOT the `parakeet` transducer backend (which
    # rejects it: "no RNN-T decoder/joint tensors — failed to load"). Using the
    # wrong backend here is exactly the #81 bug that produced the bogus 102.4×/
    # 127.7× (a ~0.5 s failed load timed as if it were inference). See the guard
    # in bench_crispasr() below.
    ("fastconformer-ctc", "cstr/parakeet-ctc-0.6b-GGUF",   "parakeet-ctc-0.6b-q8_0.gguf",     "parakeet-ctc-0.6b"),
    ("parakeet",  "cstr/parakeet-tdt-0.6b-v2-GGUF", "parakeet-tdt-0.6b-v2-q8_0.gguf",  "parakeet-tdt-0.6b"),
    # CrispASR-only (small models that fit in budget)
    ("moonshine", "cstr/moonshine-tiny-GGUF",       "moonshine-tiny-q8_0.gguf",         "moonshine-tiny"),
    ("cohere",    "cstr/cohere-transcribe-GGUF",    "cohere-transcribe-q4_k.gguf",      "cohere-transcribe"),
    ("kyutai-stt","cstr/kyutai-stt-1b-GGUF",        "kyutai-stt-1b-q4_k.gguf",          "kyutai-stt-1b"),
    ("firered-asr","cstr/firered-asr2-aed-GGUF",    "firered-asr2-aed-q4_k.gguf",      "firered-asr2"),
    ("sensevoice","cstr/sensevoice-small-GGUF",     "sensevoice-small-q8_0.gguf",       "sensevoice-small"),
    ("funasr",    "cstr/fun-asr-nano-GGUF",         "fun-asr-nano-q8_0.gguf",           "funasr-nano"),
    ("paraformer","cstr/paraformer-zh-GGUF",        "paraformer-zh-q8_0.gguf",          "paraformer-zh"),
    ("glm-asr",   "cstr/glm-asr-nano-GGUF",        "glm-asr-nano-q4_k.gguf",           "glm-asr-nano"),
]

# Add moonshine tokenizer
MOONSHINE_TOK = None

model_paths = {}
for backend, repo, fname, label in CRISPASR_MODELS:
    try:
        p = hf_hub_download(repo, fname, cache_dir=str(TEMP / "hf"))
        model_paths[label] = p
        sz = Path(p).stat().st_size / (1024**2)
        print(f"  {label}: {sz:.0f} MB")
        # Download moonshine tokenizer alongside
        if "moonshine" in label:
            try:
                MOONSHINE_TOK = hf_hub_download(repo, "tokenizer.bin", cache_dir=str(TEMP / "hf"))
            except Exception:
                pass
    except Exception as e:
        print(f"  {label}: SKIP ({e})")

# ── Phase 4: Benchmark CrispASR fleet ───────────────────────────────────────
print("\n=== Phase 4: benchmark CrispASR fleet ===", flush=True)

gpu_flag = "--gpu-backend cuda" if has_cuda else ""

def bench_crispasr(backend, model_path, audio_path, audio_dur, label, extra_flags="",
                   n_warmup=1, n_runs=3, env_prefix="", force_cpu=False):
    """Benchmark one CrispASR backend. Returns a dict with BOTH:
      wall_rtf  = audio_dur / subprocess walltime (INCLUDES model load each call)
      cli_rtf   = the CLI's own '(Nx realtime)' (load-EXCLUDED, per LEARNINGS #19)
    onnx-asr is timed in-process (load once), so cli_rtf is the fair apples-to-
    apples number; wall_rtf shows the per-call load tax. NOTE: dropped --no-prints
    so the 'transcribed .. in X.Xs (Y.Yx realtime)' line reaches stderr; the
    transcript still comes on stdout.

    env_prefix: shell env-var assignments prepended to the command (e.g.
      'CRISPASR_TDT_BATCH=1 ') so an A/B arm toggles a decode path without a
      rebuild. force_cpu: run '--no-gpu' regardless of the global gpu_flag —
      issue #81's real gap is x86 CPU (OpenBLAS), so the TDT_BATCH A/B is CPU-only."""
    this_gpu = "--no-gpu" if force_cpu else gpu_flag
    cmd = f"{env_prefix}{CRISPASR} --backend {backend} {this_gpu} -m {model_path} -f {audio_path} {extra_flags}"

    for _ in range(n_warmup):
        subprocess.run(cmd, shell=True, capture_output=True, timeout=120)

    times, cli_rtfs, text, ok = [], [], "", True
    rtf_re = re.compile(r"in ([\d.]+)s \(([\d.]+)x realtime\)")
    for i in range(n_runs):
        try:
            t0 = time.perf_counter()
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
            times.append(time.perf_counter() - t0)
            out = (r.stdout or "").strip()
            m = rtf_re.search((r.stderr or "") + (r.stdout or ""))
            if m:
                cli_rtfs.append(float(m.group(2)))
            if r.returncode != 0 or not out:
                ok = False
                if i == 0:
                    tail = (r.stderr or "").strip().split("\n")[-1] if r.stderr else ""
                    text = f"FAILED (rc={r.returncode}): {tail}"[:120]
            elif i == 0:
                text = out  # full transcript (needed to count words for proof-of-work)
        except subprocess.TimeoutExpired:
            times.append(120.0)
            ok = False

    # A crash / empty transcript must NOT be reported as a fast run (the #81
    # `--backend parakeet` on a CTC model bug: exited ~0.5 s -> fake 102/127×).
    if not times or not ok:
        return None
    mean = sum(times) / len(times)
    cli_rtf = max(cli_rtfs) if cli_rtfs else None  # best (least noisy) load-excluded RTF
    return {"wall_s": mean, "wall_rtf": audio_dur / mean, "cli_rtf": cli_rtf, "text": text}

results = {}
for backend, repo, fname, label in CRISPASR_MODELS:
    if label not in model_paths:
        continue
    mp = model_paths[label]
    extra = ""
    if "moonshine" in label and MOONSHINE_TOK:
        extra = f"--moonshine-tokenizer {MOONSHINE_TOK}"

    print(f"\n  [{label}]")
    rj = bench_crispasr(backend, mp, jfk_wav, duration, label, extra)
    if rj:
        # rtf_jfk = load-EXCLUDED CLI RTF (fair vs onnx); wall shows the load tax.
        print(f"    JFK {duration:.0f}s: cli {rj['cli_rtf']}x (load-excl) | wall "
              f"{rj['wall_rtf']:.1f}x (incl {rj['wall_s']:.2f}s load+infer)")
        results[label] = {"rtf_jfk": rj["cli_rtf"], "wall_rtf_jfk": rj["wall_rtf"], "text": rj["text"]}

        # Also test long audio for the head-to-head models
        if "parakeet" in label:
            rl = bench_crispasr(backend, mp, long_wav, long_dur, label, extra, n_warmup=0, n_runs=2)
            if rl:
                nwl = len((rl["text"] or "").split())
                print(f"    Long {long_dur:.0f}s: cli {rl['cli_rtf']}x (load-excl) | wall "
                      f"{rl['wall_rtf']:.1f}x  ({nwl}w)")
                print(f"        long transcript: {(rl['text'] or '')[:220]!r}")
                results[label]["rtf_long"] = rl["cli_rtf"]
                results[label]["wall_rtf_long"] = rl["wall_rtf"]
                results[label]["long_words"] = nwl
    else:
        print(f"    FAILED")
        results[label] = {"rtf_jfk": None, "text": "FAILED"}

# ── Phase 4b: parakeet TDT decode A/B on CPU (issue #81) ─────────────────────
# The real #81 gap is x86 CPU (OpenBLAS): the default per-frame greedy TDT
# decode issues T tiny joint-head sgemms. parakeet_tdt_decode_batched
# (CRISPASR_TDT_BATCH=1) issues one big sgemm over the joint head — better
# BLAS utilization. It already runs unconditionally on the long-form streamed
# path; here we A/B it for short-form to decide whether to flip the default.
# A flip is only safe if it is FASTER *and* byte-equal in transcript (rule 3/4).
print("\n=== Phase 4b: parakeet TDT decode A/B (CPU, issue #81) ===", flush=True)

def _norm_text(t):
    return " ".join((t or "").split()).lower()

tdt_ab = {}
for backend, repo, fname, label in CRISPASR_MODELS:
    if "parakeet" not in label or label not in model_paths:
        continue
    mp = model_paths[label]
    extra = ""
    print(f"\n  [{label}] CPU baseline vs CRISPASR_TDT_BATCH=1")
    for tag, dur, wav in (("jfk", duration, jfk_wav), ("long", long_dur, long_wav)):
        if wav is None:
            continue
        base = bench_crispasr(backend, mp, wav, dur, label, extra,
                              n_warmup=1, n_runs=3, force_cpu=True)
        batch = bench_crispasr(backend, mp, wav, dur, label, extra,
                               n_warmup=1, n_runs=3, env_prefix="CRISPASR_TDT_BATCH=1 ",
                               force_cpu=True)
        if not base or not batch:
            print(f"    {tag}: FAILED (base={bool(base)} batch={bool(batch)})")
            continue
        # Proof-of-work: identical transcript, and a speedup expressed on the
        # load-excluded CLI RTF (falls back to wall RTF if the RTF line was
        # absent). Speedup > 1 => batched is faster.
        identical = _norm_text(base["text"]) == _norm_text(batch["text"])
        b_rtf = base["cli_rtf"] or base["wall_rtf"]
        n_rtf = batch["cli_rtf"] or batch["wall_rtf"]
        speedup = (n_rtf / b_rtf) if (b_rtf and n_rtf) else None
        bw, nw = len((base["text"] or "").split()), len((batch["text"] or "").split())
        print(f"    {tag} {dur:.0f}s CPU: base cli {base['cli_rtf']}x ({bw}w) | "
              f"batch cli {batch['cli_rtf']}x ({nw}w) | "
              f"speedup {speedup:.2f}x | identical={identical}")
        if not identical:
            print(f"      ⚠ TRANSCRIPTS DIFFER — batched decode NOT output-equivalent, do NOT flip default")
            print(f"        base : {(base['text'] or '')[:160]!r}")
            print(f"        batch: {(batch['text'] or '')[:160]!r}")
        tdt_ab[f"{label}:{tag}"] = {
            "base_rtf": base["cli_rtf"], "batch_rtf": batch["cli_rtf"],
            "speedup": speedup, "identical": identical,
            "base_words": bw, "batch_words": nw,
        }
results["tdt_batch_ab"] = tdt_ab

# ── Phase 5: Benchmark onnx-asr (head-to-head models) ───────────────────────
# CPU int8 (the original #81 baseline) AND, when the CUDA EP is available, GPU
# fp32 — the true GPU-vs-GPU comparison the issue is really about. onnx's CUDA
# EP has thin int8 coverage, so the GPU arm uses fp32 (quantization=None).
print("\n=== Phase 5: benchmark onnx-asr ===", flush=True)
# Import-guarded: a broken onnxruntime (e.g. CUDA-13 wheel on a CUDA-12 box, or a
# cuDNN mismatch) must NOT crash the kernel — v9 died here on
# `ImportError: libcudart.so.13`. On failure we skip onnx and keep the CrispASR
# fleet numbers + saved results.
try:
    import onnx_asr
    import onnxruntime as ort
    _avail = ort.get_available_providers()
    cuda_ok = "CUDAExecutionProvider" in _avail
    print(f"  onnxruntime {ort.__version__} providers: {_avail}  (CUDA EP: {cuda_ok})", flush=True)
    ONNX_MODELS = [
        ("nemo-parakeet-ctc-0.6b", "parakeet-ctc-0.6b"),
        ("nemo-parakeet-tdt-0.6b-v2", "parakeet-tdt-0.6b"),
    ]
except Exception as e:  # noqa: BLE001
    print(f"  onnx-asr/onnxruntime unavailable ({str(e)[:160]}) — skipping onnx phase", flush=True)
    cuda_ok = False
    ONNX_MODELS = []


def bench_onnx(onnx_name, quant, providers):
    model = onnx_asr.load_model(onnx_name, quantization=quant, providers=providers)

    # PER-SHAPE warmup (a 55 s clip specialises different kernels than 11 s, and
    # the first call pays cold CUDA JIT) + MEDIAN of N + absolute time. The v10
    # "174.9×" was a tiny-denominator artifact (0.063 s warmed on the same file);
    # reporting absolute ms alongside RTF makes that visible, and per-shape
    # warmup + median stop one cold call from dominating.
    def _timeit(wav, dur, warmups, n):
        for _ in range(warmups):
            model.recognize(str(wav))
        ts, txt = [], ""
        for i in range(n):
            t0 = time.perf_counter()
            r = model.recognize(str(wav))
            ts.append(time.perf_counter() - t0)
            if i == 0:
                txt = str(r)
        ts.sort()
        med = ts[len(ts) // 2]
        # Return FULL transcript + word count so we can PROVE the run transcribed
        # the WHOLE clip: the ~120 s real clip must yield a large word count that
        # ~matches across engines/EPs (onnx-CUDA long_words ≈ onnx-CPU long_words ≈
        # CrispASR long_words). A truncated/no-op fast path would be short+fast =
        # a fake high RTF. (Real varied speech, not jfk×5 — so caches can't reuse
        # work across identical repeats and inflate the GPU number.)
        return dur / med, med, txt, len(txt), len(txt.split())

    rj, mj, tj, cj, wj = _timeit(jfk_wav, duration, 2, 5)
    rl, ml, tl, cl, wl = _timeit(long_wav, long_dur, 1, 3)
    del model
    return {"rtf_jfk": rj, "s_jfk": mj, "rtf_long": rl, "s_long": ml,
            "jfk_words": wj, "long_words": wl, "jfk_text": tj[:60], "long_text": tl}


onnx_results = {}
for onnx_name, label in ONNX_MODELS:
    print(f"\n  [{label} via onnx-asr]")
    onnx_results[label] = {}
    configs = [("cpu_int8", "int8", ["CPUExecutionProvider"])]
    if cuda_ok:
        configs.append(("cuda_fp32", None, ["CUDAExecutionProvider", "CPUExecutionProvider"]))
    for key, quant, provs in configs:
        try:
            r = bench_onnx(onnx_name, quant, provs)
            # PROOF-OF-WORK: long_words must be large (proportional to ~120 s of
            # real speech) and ~match across CPU/GPU EPs; a fast run with few
            # words = truncation/no-op = fake RTF.
            print(f"    {key}: JFK {r['rtf_jfk']:.1f}x ({r['s_jfk'] * 1000:.0f}ms, {r['jfk_words']}w)  "
                  f"Long {r['rtf_long']:.1f}x ({r['s_long']:.2f}s, {r['long_words']}w)", flush=True)
            print(f"        long transcript: {r['long_text'][:220]!r}", flush=True)
            onnx_results[label][key] = r
        except Exception as e:  # noqa: BLE001
            print(f"    {key}: FAILED: {str(e)[:200]}", flush=True)
            onnx_results[label][key] = {"error": str(e)[:200]}

# ── Phase 6: Summary ────────────────────────────────────────────────────────
gpu_name = "unknown"
try:
    gpu_name = subprocess.check_output(
        "nvidia-smi --query-gpu=name --format=csv,noheader", shell=True).decode().strip()
except Exception: pass

print("\n" + "=" * 78)
print("  CrispASR FULL-FLEET BENCHMARK + onnx-asr HEAD-TO-HEAD")
print("=" * 78)
print(f"  GPU: {gpu_name}  |  CUDA build: {has_cuda}  |  Audio: {duration:.0f}s JFK")
print("  METHOD NOTE: CrispASR is timed as a fresh CLI subprocess INCLUDING model")
print("  load each call; onnx-asr loads once then times inference-only. So short-")
print("  clip RTF favours onnx; the LONG-audio column (load amortised) is the fair")
print("  one. onnx-asr does not chunk long audio -> O(T^2) attention blowup there.")
print()
print(f"  {'Backend':<22s} {'Engine':>10s} {'JFK RTF':>10s} {'Long RTF':>10s} {'Notes':>20s}")
print(f"  {'-'*74}")

# Head-to-head
for label in ["parakeet-ctc-0.6b", "parakeet-tdt-0.6b"]:
    ca = results.get(label, {})
    ox = onnx_results.get(label, {})
    ca_rtf = f"{ca.get('rtf_jfk', 0):.1f}x" if ca.get('rtf_jfk') else "FAIL"
    ca_long = f"{ca.get('rtf_long', 0):.1f}x" if ca.get('rtf_long') else "-"
    print(f"  {label:<22s} {'CrispASR':>10s} {ca_rtf:>10s} {ca_long:>10s} {'CUDA Q8_0':>20s}")
    for key, note in [("cpu_int8", "onnx CPU int8"), ("cuda_fp32", "onnx CUDA fp32")]:
        o = ox.get(key, {})
        oj = f"{o.get('rtf_jfk', 0):.1f}x" if o.get("rtf_jfk") else "FAIL"
        ol = f"{o.get('rtf_long', 0):.1f}x" if o.get("rtf_long") else "-"
        print(f"  {'':.<22s} {'onnx-asr':>10s} {oj:>10s} {ol:>10s} {note:>20s}")

print(f"  {'-'*74}")
print(f"  {'CrispASR-only backends':}")

# CrispASR-only
for backend, repo, fname, label in CRISPASR_MODELS:
    if "parakeet" in label:
        continue  # already shown above
    ca = results.get(label, {})
    ca_rtf = f"{ca.get('rtf_jfk', 0):.1f}x" if ca.get('rtf_jfk') else "FAIL"
    print(f"  {label:<22s} {'CrispASR':>10s} {ca_rtf:>10s} {'':>10s} {'CUDA':>20s}")

print(f"\n  Total backends tested: {len(results)} CrispASR + {len(onnx_results)} onnx-asr")
print(f"  CrispASR supports 25+ backends; onnx-asr supports ~10")

# Save JSON
all_results = {
    "gpu": gpu_name, "cuda_build": has_cuda, "audio_duration": duration,
    "crispasr": results, "onnx_asr": onnx_results,
}
with open(WORK / "benchmark_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n  Results saved to {WORK / 'benchmark_results.json'}")

# Refresh the ccache snapshot so the ${KAGGLE_ACCOUNT}/crispasr-ccache dataset can be updated
# from this run (kaggle_usage.md #17 — keep it current or warm builds go stale).
# ccache lives at the RELOCATED CCACHE_DIR (/kaggle/temp/.ccache, out of the
# output). Tar it into /kaggle/working/ccache.tar as the ONLY ccache artifact in
# the output so it's retrievable in page 1 (not buried behind 500 loose files),
# then `kaggle datasets version` from it locally.
try:
    ccache_dir = Path(os.environ.get("CCACHE_DIR", "/kaggle/temp/.ccache"))
    if ccache_dir.is_dir():
        parent = ccache_dir.parent
        subprocess.run(f"tar cf {WORK}/ccache.tar -C {parent} {ccache_dir.name}", shell=True, check=True)
        sz = (WORK / "ccache.tar").stat().st_size / (1024**2)
        print(f"  ccache.tar written to /kaggle/working ({sz:.0f} MB) — the only ccache artifact in "
              f"the output; update ${KAGGLE_ACCOUNT}/crispasr-ccache from it", flush=True)
        subprocess.run("ccache -s 2>/dev/null | tail -6 || true", shell=True)
except Exception as e:  # noqa: BLE001
    print(f"  ccache tar skipped: {e}", flush=True)

print("\n=== Done ===", flush=True)
