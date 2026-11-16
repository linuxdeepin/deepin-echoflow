"""
CrispASR — #218 encoder-quantization audit for cohere-transcribe + glm-asr

The qwen3-asr root cause (PLAN '#218 qwen3-asr long-audio root cause') was
sub-8-bit quantization of the audio encoder compounding per-block drift until
greedy decode flips into repetition loops. cohere-transcribe and glm-asr were
also reported looping in #218 and have NO encoder carve-out in
crispasr-quantize — this kernel measures whether the same mechanism applies,
WITHOUT assuming it transfers.

Method (decoded-output level, the repo's acceptance metric):
  - Build crispasr with CUDA (GPU kernel: internet + fast decode).
  - Fetch the reporter's canonical t32-145s.wav.
  - For each backend, run the SAME 145 s clip through the q4_k GGUF and the
    F16 GGUF with the n-gram loop-fix DISABLED, default dispatcher chunking.
  - Loop metric: max immediate-unigram-run + max short-phrase-cycle length.

Verdict per backend:
  QUANT-DRIFT   q4_k loops, F16 clean  -> add encoder carve-out + rebake
  MODEL-LIMIT   both loop              -> fix_loops is the right mitigation
  CLEAN         neither loops          -> no action
"""

import os
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path("/kaggle/working")
REPO = ROOT / "CrispASR"
BUILD = ROOT / "build"
# Models go to /tmp (writable layer, ~70 GB) — /kaggle/working is capped 20 GB
# and the four GGUFs alone are ~11.5 GB (kaggle_usage.md gotcha #18).
MODELS = Path("/tmp/models")
AUDIO_DIR = ROOT / "audio"
OUT_DIR = ROOT / "results"
for d in (MODELS, AUDIO_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
print(f"[clone] ref={CRISPASR_REF}", flush=True)
if not REPO.exists():
    _clone_cmd = (
        f"git clone --depth 1 --branch {CRISPASR_REF} "
        f"https://github.com/CrispStrobe/CrispASR.git {REPO}"
    )
else:
    _clone_cmd = f"git -C {REPO} pull --ff-only"
if subprocess.run(_clone_cmd, shell=True).returncode != 0:
    raise SystemExit(f"clone failed: {_clone_cmd}")
# git clone --depth 1 does not pull submodules; ggml is one since 2026-07-07
# and cmake fails at add_subdirectory(ggml) without it (#238 lesson).
if not (REPO / "ggml" / "CMakeLists.txt").exists():
    subprocess.run(
        f"git -C {REPO} submodule update --init --recursive --depth 1",
        shell=True, check=True,
    )
sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress(progress_path=str(ROOT / "progress.jsonl"))
kh.resolve_hf_token()
kh.step("script.start")
kh.step("clone.done", sha=sha)

# ── Build (CUDA — the GPU worker is also what gives us internet) ─────────
kh.step("build.begin")
BUILD.mkdir(exist_ok=True)
kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
cmake_cmd = " ".join(
    [
        f"cmake {REPO} -B{BUILD} -GNinja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCRISPASR_BUILD_TESTS=OFF",
    ]
    + kh.cuda_build_flags(arch)
    + kh.cache_and_link_flags()
)
njobs = kh.safe_build_jobs(gpu=True)
with kh.build_heartbeat("cmake-configure"):
    kh.sh_with_progress(cmake_cmd)
with kh.build_heartbeat("cmake-build"):
    kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli -- -j{njobs}")
CRISPASR = BUILD / "bin" / "crispasr"
assert CRISPASR.is_file(), f"crispasr binary missing at {CRISPASR}"
kh.step("build.done", binary=str(CRISPASR), cuda_arch=arch)

# ── Audio: the reporter's canonical 145 s clip (issue #218) ──────────────
kh.step("audio.begin")
WAV_ZIP = AUDIO_DIR / "t32-145s.wav.zip"
WAV = AUDIO_DIR / "t32-145s.wav"
if not WAV.exists():
    kh.sh_with_progress(
        f"curl -sL -o {WAV_ZIP} "
        "https://github.com/user-attachments/files/29652411/t32-145s.wav.zip"
    )
    with zipfile.ZipFile(WAV_ZIP) as z:
        z.extractall(AUDIO_DIR)
assert WAV.is_file(), "t32-145s.wav missing after unzip"
kh.step("audio.done", size_mib=WAV.stat().st_size // (1 << 20))

# ── Models ───────────────────────────────────────────────────────────────
kh.step("download.begin")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
kh.sh_with_progress("pip install -q huggingface_hub hf_transfer")
from huggingface_hub import hf_hub_download  # noqa: E402

CASES = [
    ("cohere", "cstr/cohere-transcribe-03-2026-GGUF", "cohere-transcribe-q4_k.gguf", "q4_k"),
    ("cohere", "cstr/cohere-transcribe-03-2026-GGUF", "cohere-transcribe.gguf", "f16"),
    ("glm-asr", "cstr/glm-asr-nano-GGUF", "glm-asr-nano-q4_k.gguf", "q4_k"),
    ("glm-asr", "cstr/glm-asr-nano-GGUF", "glm-asr-nano.gguf", "f16"),
]
model_path = {}
for backend, repo_id, fname, quant in CASES:
    kh.step(f"download.{backend}.{quant}.begin", repo=repo_id, file=fname)
    p = hf_hub_download(repo_id=repo_id, filename=fname, local_dir=str(MODELS))
    model_path[(backend, quant)] = Path(p)
    kh.step(f"download.{backend}.{quant}.done", size_mib=Path(p).stat().st_size // (1 << 20))
kh.step("download.done")

# ── Loop metrics ─────────────────────────────────────────────────────────


def loop_metrics(text: str) -> dict:
    """Max immediate unigram run + max short-phrase (2-4 words) cycle run."""
    words = [w.strip(".,!?;:—-\"'").lower() for w in text.split()]
    words = [w for w in words if w]
    max_uni, cur = 0, 0
    prev = None
    for w in words:
        cur = cur + 1 if w == prev else 1
        prev = w
        max_uni = max(max_uni, cur)
    max_phrase = 0
    for n in (2, 3, 4):
        i = 0
        while i + n <= len(words):
            run = 1
            j = i + n
            while j + n <= len(words) and words[j : j + n] == words[i : i + n]:
                run += 1
                j += n
            max_phrase = max(max_phrase, run if run > 1 else 0)
            i = i + 1 if run == 1 else j
    # Completion signal: a decode that degenerates into a loop burns its
    # token budget before the clip's final sentence ("...taking the back
    # road and he'll follow"). Loop metrics + truncation together separate
    # "audio genuinely contains repeated shouts" from degenerate decode.
    low = text.lower()
    complete = ("back road" in low) or ("follow" in low[-200:])
    return {
        "max_unigram_run": max_uni,
        "max_phrase_cycle": max_phrase,
        "n_words": len(words),
        "reaches_final_sentence": bool(complete),
    }


def run_one(backend: str, quant: str) -> dict:
    model = model_path[(backend, quant)]
    out_stem = OUT_DIR / f"{backend}-{quant}"
    cmd = [
        str(CRISPASR),
        "-m", str(model),
        "--backend", backend,
        "--language", "en",
        "-f", str(WAV),
        "--no-timestamps",
        "-otxt",
        "-of", str(out_stem),
        "-np",
    ]
    env = dict(os.environ)
    # Disable the shared n-gram collapse (global gate in
    # src/core/ngram_loop_fix.h) so we measure the RAW decode behaviour,
    # not the mitigation. Moss has a separate per-backend gate; covered too.
    env["CRISPASR_NGRAM_LOOPFIX_OFF"] = "1"
    env["CRISPASR_MOSS_TRANSCRIBE_NO_LOOPFIX"] = "1"
    kh.step(f"run.{backend}.{quant}.begin", model=model.name)
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
        rc = r.returncode
        tail = ((r.stdout or "") + (r.stderr or "")).splitlines()[-15:]
    except subprocess.TimeoutExpired:
        rc, tail = -1, ["TIMEOUT"]
    dt = round(time.time() - t0, 1)
    txt_file = out_stem.with_suffix(".txt")
    text = txt_file.read_text(errors="replace") if txt_file.is_file() else ""
    m = loop_metrics(text)
    kh.step(f"run.{backend}.{quant}.done", rc=rc, wall_s=dt, chars=len(text), **m)
    if rc != 0:
        for line in tail:
            print(f"    {line}", flush=True)
    print(f"--- {backend}/{quant} transcript ({len(text)} chars) ---", flush=True)
    print(text[:1500], flush=True)
    if len(text) > 1500:
        print(f"[... {len(text) - 1500} more chars]", flush=True)
    return {"rc": rc, "wall_s": dt, "chars": len(text), **m}


LOOP_THRESHOLD = 6  # runs this long are degenerate, not real speech
results = {}
for backend in ("cohere", "glm-asr"):
    results[backend] = {q: run_one(backend, q) for q in ("q4_k", "f16")}

# ── Verdicts ─────────────────────────────────────────────────────────────
print("\n" + "=" * 76)
print(f"SUMMARY — #218 encoder-quant audit on HEAD ({sha[:8]})")
print("=" * 76)
verdicts = {}
for backend, r in results.items():
    q4_loops = (max(r["q4_k"]["max_unigram_run"], r["q4_k"]["max_phrase_cycle"]) >= LOOP_THRESHOLD
                or not r["q4_k"]["reaches_final_sentence"])
    f16_loops = (max(r["f16"]["max_unigram_run"], r["f16"]["max_phrase_cycle"]) >= LOOP_THRESHOLD
                 or not r["f16"]["reaches_final_sentence"])
    if r["q4_k"]["rc"] != 0 or r["f16"]["rc"] != 0:
        verdict = "RUN-FAILED"
    elif q4_loops and not f16_loops:
        verdict = "QUANT-DRIFT"
    elif q4_loops and f16_loops:
        verdict = "MODEL-LIMIT"
    else:
        verdict = "CLEAN"
    verdicts[backend] = verdict
    print(f"\n  {backend}: {verdict}")
    for q in ("q4_k", "f16"):
        print(
            f"    {q:5s} rc={r[q]['rc']:>3} wall={r[q]['wall_s']:>7}s chars={r[q]['chars']:>6} "
            f"max_uni={r[q]['max_unigram_run']:>3} max_cycle={r[q]['max_phrase_cycle']:>3} "
            f"complete={r[q]['reaches_final_sentence']}"
        )

print("\nInterpretation:")
for backend, v in verdicts.items():
    if v == "QUANT-DRIFT":
        print(f"  {backend}: add encoder carve-out in crispasr-quantize + rebake (qwen3-asr pattern)")
    elif v == "MODEL-LIMIT":
        print(f"  {backend}: loops are inherent at this quant AND f16 — fix_loops is the mitigation")
    elif v == "CLEAN":
        print(f"  {backend}: no loops on the canonical clip at either precision — no action")
    else:
        print(f"  {backend}: a run failed — inspect logs before concluding anything")

kh.step("summary", verdicts=verdicts, sha=sha)
kh._push_progress_to_hf(force=True)
kh.step("script.end")
