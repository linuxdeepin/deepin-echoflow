#!/usr/bin/env python3
"""VibeVoice-ASR-BitNet quantization sweep — generate + test + upload all variants.

Generates 6 GGUF variants, transcribes JFK with each, then uploads all to HF.
"""

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

WORK = Path("/kaggle/working")
REPO = Path("/kaggle/temp/CrispASR")
BUILD = REPO / "build"
MODELS = Path("/kaggle/temp/models")
RESULTS = WORK / "results.json"

# Validation corpus.
#
# This used to be JFK alone, in English, and the model card that came out of it
# said "quality is identical across all variants -- pick any file". VibeVoice-ASR
# advertises English, Chinese, French, Italian, Korean, Portuguese and
# Vietnamese, and #369 reported Korean degrading badly on the same variants that
# transcribe JFK perfectly: on borderline audio the output flipped language
# entirely (Korean -> Italian). One English clip cannot support a claim about
# seven languages, and the equivalence it "verified" was never in question for
# English.
#
# Each entry is (label, audio path relative to REPO, language, reference text).
# A clip is skipped with a warning if it is missing rather than failing the
# sweep, so this runs on a checkout that has only some fixtures — but the
# summary reports which languages actually ran, so a claim can never be made
# from a corpus that silently shrank to English.
CORPUS = [
    ("en-jfk", "samples/jfk.wav", "en",
     "And so, my fellow Americans, ask not what your country can do for you — ask what you can do for your country."),
    ("ko-369", "samples/ko-369.wav", "ko",
     "\ub0b4\uc77c \uc624\uc804\uc5d0 \ud68c\uc758 \uc790\ub8cc\ub97c \ubcf4\ub0b4 \uc8fc\uc138\uc694"),  # synthetic TTS clip from #369
    ("zh-paraformer", "samples/paraformer_zh.wav", "zh", None),
    ("multi", "samples/multispeaker.wav", "en", None),
]

# Kept for the English overlap score, which is the only entry with a reference.
REFERENCE_TEXT = CORPUS[0][3]
JFK_AUDIO = CORPUS[0][1]  # relative to REPO

VARIANTS = [
    ("vibevoice-asr-bitnet-tq2",      "q8_0", "f16"),
    ("vibevoice-asr-bitnet-embed-q8",  "q8_0", "q8_0"),
    ("vibevoice-asr-bitnet-vae-q5",    "q5_0", "f16"),
    ("vibevoice-asr-bitnet-both-q5",   "q5_0", "q8_0"),
    ("vibevoice-asr-bitnet-vae-q4",    "q4_0", "f16"),
    ("vibevoice-asr-bitnet-aggro",     "q4_0", "q8_0"),
]

HF_REPO = "cstr/vibevoice-asr-bitnet-GGUF"

# ── Clone + harness ──────────────────────────────────────────────────────

CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
if not REPO.exists():
    try:
        subprocess.check_call(
            ["git", "clone", "--depth", "1", CRISPASR_URL, str(REPO)])
        subprocess.check_call(
            ["git", "submodule", "update", "--init", "ggml"], cwd=str(REPO))
        sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    except Exception:
        pass

if str(REPO / "tools" / "kaggle") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import kaggle_harness as kh  # noqa: E402
kh.init_progress()

# Global crash handler
_orig_excepthook = sys.excepthook
def _crash_handler(exc_type, exc_val, exc_tb):
    msg = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
    try:
        with open(WORK / "error.txt", "w") as f:
            f.write(msg)
    except Exception:
        pass
    _orig_excepthook(exc_type, exc_val, exc_tb)
sys.excepthook = _crash_handler

# ── Build ────────────────────────────────────────────────────────────────

print("Installing build toolchain")
kh.install_build_toolchain()

# Install converter + upload dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "safetensors", "transformers", "gguf"], check=False)

# HF token
hf_token = kh.resolve_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    print("HF token resolved")
else:
    print("WARNING: no HF token")

cmake_flags = " ".join(kh.cache_and_link_flags())
crispasr_flags = " ".join(kh.crispasr_cmake_flags())
cmake_cmd = (
    f"cmake -G Ninja -S {REPO} -B {BUILD} "
    f"-DCMAKE_BUILD_TYPE=Release "
    f"{cmake_flags} "
    f"{crispasr_flags} "
    f"-DCRISPASR_BUILD_TESTS=OFF "
    f"-DCRISPASR_BUILD_EXAMPLES=ON "
    f"-DCRISPASR_BUILD_SERVER=OFF"
)
print(f"cmake configure: {cmake_cmd}")
subprocess.check_call(cmake_cmd, shell=True)

jobs = kh.safe_build_jobs(gpu=True)
print(f"Building with {jobs} jobs")
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} "
        f"--target crispasr-cli -j{jobs}")

CRISPASR_BIN = BUILD / "bin" / "crispasr"
assert CRISPASR_BIN.exists(), f"Build failed: {CRISPASR_BIN} not found"
print(f"Build OK: {CRISPASR_BIN}")

# ── Generate + test + upload variants ────────────────────────────────────

MODELS.mkdir(parents=True, exist_ok=True)
CONVERTER = REPO / "models" / "convert-vibevoice-bitnet-to-gguf.py"

results = []

for label, vae_q, embed_q in VARIANTS:
    fname = f"{label}.gguf"
    out_gguf = MODELS / fname
    print(f"\n=== {label} (VAE={vae_q}, embed={embed_q}) ===")

    # ── Convert ──
    if not out_gguf.exists():
        t0 = time.time()
        conv_env = os.environ.copy()
        conv_env["TMPDIR"] = str(MODELS)
        cmd = [
            sys.executable, str(CONVERTER),
            "--input", "microsoft/VibeVoice-ASR-BitNet",
            "--output", str(out_gguf),
            "--vae-quant", vae_q,
            "--embed-quant", embed_q,
        ]
        print(f"  Converting...")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                               env=conv_env)
            if r.returncode != 0:
                print(f"  CONVERT FAILED (rc={r.returncode})")
                print(f"  stderr: {r.stderr[-500:]}")
                results.append({
                    "label": label, "vae_quant": vae_q, "embed_quant": embed_q,
                    "error": f"convert failed: {r.stderr[-200:]}",
                })
                continue
            convert_s = time.time() - t0
            print(f"  Converted in {convert_s:.1f}s")
        except subprocess.TimeoutExpired:
            print(f"  CONVERT TIMEOUT")
            results.append({
                "label": label, "vae_quant": vae_q, "embed_quant": embed_q,
                "error": "convert timeout",
            })
            continue
    else:
        print(f"  Using cached {out_gguf}")

    file_size_mb = out_gguf.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")

    # ── Transcribe every clip in the corpus ──
    #
    # Variant equivalence is a per-language claim. Transcribing one clip and
    # generalising is what produced #369, so each variant now runs the whole
    # corpus and the per-clip text is recorded; comparing variants against each
    # other is what shows a quantisation choice costing a language.
    per_clip = {}
    for clip_label, clip_path, clip_lang, _clip_ref in CORPUS:
        wav = REPO / clip_path
        if not wav.exists():
            print(f"  SKIP {clip_label}: {clip_path} not present")
            per_clip[clip_label] = None
            continue
        c = [
            str(CRISPASR_BIN), "-m", str(out_gguf), "--backend", "vibevoice",
            "-f", str(wav), "-t", "4", "--language", clip_lang, "--no-prints",
        ]
        try:
            rr = subprocess.run(c, capture_output=True, text=True, timeout=900)
            tl = [ln.strip() for ln in rr.stdout.strip().split("\n")
                  if ln and not any(k in ln for k in ["firered", "whisper", "crispasr:", "Maximum"])]
            per_clip[clip_label] = " ".join(tl).strip()
            print(f"  [{clip_label}] {per_clip[clip_label][:90]}")
        except Exception as e:
            per_clip[clip_label] = None
            print(f"  [{clip_label}] FAILED: {e}")

    # ── Transcribe (English reference clip, scored) ──
    t0 = time.time()
    cmd = [
        str(CRISPASR_BIN),
        "-m", str(out_gguf),
        "--backend", "vibevoice",
        "-f", str(REPO / JFK_AUDIO),
        "-t", "4",
        "--language", "en",
        "--no-prints",
    ]
    print(f"  Transcribing...")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        wall_s = time.time() - t0

        text_lines = []
        for line in r.stdout.strip().split("\n"):
            if line and not any(k in line for k in
                               ["firered", "whisper", "crispasr:", "Maximum"]):
                text_lines.append(line.strip())
        text = " ".join(text_lines).strip()

        print(f"  Text: {text[:120]}")
        print(f"  Wall: {wall_s:.1f}s")

        ref_words = set(REFERENCE_TEXT.lower().split())
        out_words = set(text.lower().split()) if text else set()
        overlap = len(ref_words & out_words) / len(ref_words) if ref_words else 0.0

        results.append({
            "per_clip": per_clip,
            "label": label,
            "vae_quant": vae_q,
            "embed_quant": embed_q,
            "file_size_mb": round(file_size_mb, 1),
            "wall_s": round(wall_s, 1),
            "text": text,
            "word_overlap": round(overlap, 3),
            "returncode": r.returncode,
        })

    except subprocess.TimeoutExpired:
        print(f"  TRANSCRIBE TIMEOUT")
        results.append({
            "label": label, "vae_quant": vae_q, "embed_quant": embed_q,
            "error": "transcribe timeout",
        })

    # ── Upload to HF ──
    if out_gguf.exists() and hf_token:
        print(f"  Uploading {fname} to {HF_REPO}...")
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.upload_file(
                path_or_fileobj=str(out_gguf),
                path_in_repo=fname,
                repo_id=HF_REPO,
                repo_type="model",
            )
            print(f"  Uploaded {fname}")
        except Exception as e:
            print(f"  Upload failed: {e}")

    # Save incremental results
    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    kh.step(f"variant-{label}")

# ── Summary ──────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("QUANTIZATION SWEEP RESULTS")
print("=" * 80)
print(f"{'Label':<35} {'VAE':<6} {'Embed':<6} {'Size MB':>8} {'Overlap':>8}")
print("-" * 75)

for r in results:
    if "error" in r:
        print(f"{r['label']:<35} {r['vae_quant']:<6} {r['embed_quant']:<6} {'ERROR':>8} {r['error']}")
    else:
        print(f"{r['label']:<35} {r['vae_quant']:<6} {r['embed_quant']:<6} "
              f"{r['file_size_mb']:>8.1f} {r['word_overlap']:>8.3f}")

print(f"\nReference: {REFERENCE_TEXT}")

with open(RESULTS, "w") as f:
    json.dump(results, f, indent=2)
shutil.copy2(str(RESULTS), str(WORK / "vibeasr-bitnet-quant-results.json"))

print("\nDone.")
