#!/usr/bin/env python3
"""#369: is the BitNet gap the ACTIVATION quantization, or the weights?

The BitNet checkpoint transcribes Korean worse than the official demo Space does
on the SAME weights. Everything else has been eliminated:

  * the ternary weights are equivalent to upstream's I2_S — 2 differing values in
    13.76 M, scales equal to f16
  * the sigma-VAE encoder + connectors read cos 0.999926 against upstream's own
    modules on identical 24 kHz input
  * the prompt, the -25 dBFS input normalisation and the resampler are all fixed,
    and the FULL 7B model is now exact on every real fixture through the same
    pipeline

What is left is the one axis where our runtime provably differs from Microsoft's:
how ACTIVATIONS are handled in the ternary matmul. ggml's TQ2_0 multiplies
against block-quantized int8 activations; BitNet.cpp's I2_S quantizes them per
token. The weights are the same either way.

This isolates it. `--lm-quant f16` writes the SAME ternary values
({-mean, 0, +mean}, already collapsed by ternary_quantize) as F16, so ggml takes
an unquantized-activation matmul path. Weights identical, activation handling
different — one variable.

  tq2_0  the published variant (control; must reproduce the known-bad output)
  f16    same weights, unquantized activations
  q8_0   same weights, per-32-block int8 activations — an intermediate point

If f16 recovers the Korean, activation quantization is the cause and the fix is
a higher-precision LM variant (or an I2_S-style activation path). If it does not,
the ternary checkpoint is simply weak here and the model card should say so
instead of implying parity with the full model.

No HF upload: these are experiment artifacts, not releases.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

WORK = Path("/kaggle/working")
STAGE = Path("/kaggle/temp/vvbit")
SRC = STAGE / "checkpoint"
OUT = STAGE / "gguf"
FIX = STAGE / "fixtures"
REPO = Path("/kaggle/temp/CrispASR")
RESULTS = WORK / "results"
for d in (STAGE, SRC, OUT, FIX, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
HF_CHECKPOINT = "microsoft/VibeVoice-ASR-BitNet"
GIST = "https://gist.github.com/Dev0June/80d074bd1f1b588ae56a8ea68f6b4fd3/raw"
GIST_FILES = {
    "ko-test.wav": f"{GIST}/ko-test-sentence.wav",
    "ko-mic-cue-kept.wav": f"{GIST}/ko-mic-cue-kept.wav",
    "ko-mic-cue-lost.wav": f"{GIST}/ko-mic-cue-lost.wav",
}
GT_KO = "내일 오전에 회의 자료를 보내 주세요"
EMPTY_MARKERS = ("[Silence]", "[BLANK_AUDIO]", "[silence]")


def run(argv, *, cwd=None, env=None, timeout=None, check=True):
    argv = [str(x) for x in argv]
    print("$ " + " ".join(argv), flush=True)
    merged = os.environ.copy()
    if env:
        merged.update({str(k): str(v) for k, v in env.items()})
    p = subprocess.run(argv, cwd=str(cwd) if cwd else None, env=merged, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    if check and p.returncode:
        print(p.stdout[-4000:], flush=True)
        raise subprocess.CalledProcessError(p.returncode, argv, output=p.stdout)
    return p


if not (REPO / ".git").exists():
    run(["git", "clone", "--depth", "1", "--recursive", CRISPASR_URL, str(REPO)], timeout=2400)
COMMIT = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
kh.step("provenance", commit=COMMIT)


@contextlib.contextmanager
def hb(label):
    with kh.build_heartbeat(label, interval_s=30):
        yield


def build() -> Path:
    kh.install_build_toolchain()
    seeded, cache = WORK / ".ccache", Path("/kaggle/temp/.ccache")
    if seeded.exists() and not cache.exists():
        shutil.move(str(seeded), str(cache))
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["CCACHE_DIR"] = str(cache)
    arch = kh.detect_cuda_arch()
    bdir = REPO / "build-cuda"
    kh.step("build.configure", arch=arch)
    with hb("build.configure"):
        run(["cmake", "-S", REPO, "-B", bdir, "-G", "Ninja",
             "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF",
             *kh.cuda_build_flags(arch), *kh.cache_and_link_flags()], timeout=2400)
    jobs = kh.safe_build_jobs(gpu=True)
    kh.step("build.compile", jobs=jobs)
    with hb("build.compile"):
        run(["cmake", "--build", bdir, "-j", str(jobs), "--target", "crispasr"], timeout=5400)
    binary = bdir / "bin" / "crispasr"
    if not binary.is_file():
        raise RuntimeError("crispasr did not link")
    return binary


def fetch_fixtures() -> dict:
    import urllib.request
    out = {}
    for name, url in GIST_FILES.items():
        dst = FIX / name
        if not dst.exists():
            with hb(f"fetch.{name}"):
                urllib.request.urlretrieve(url, dst)  # noqa: S310  (fixed gist URL)
        out[name.replace(".wav", "")] = dst
    jfk = REPO / "samples" / "jfk.wav"
    if jfk.is_file():
        out["en-jfk"] = jfk
    return out


def transcribe(binary: Path, model: Path, wav: Path, timeout=3600) -> dict:
    argv = [str(binary), "--backend", "vibevoice-bitnet", "-m", str(model),
            "-ng", "-t", "4", "-f", str(wav), "-nt"]
    print("$ " + " ".join(argv), flush=True)
    p = subprocess.run(argv, text=True, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    text = re.sub(r"^\(Speaker \d+\)\s*", "", p.stdout.strip())
    ok = p.returncode == 0 and bool(text) and not any(m in text for m in EMPTY_MARKERS)
    return {"rc": p.returncode, "text": text, "ok": ok, "stderr_tail": p.stderr[-1200:]}


def preflight_converter(conv: Path) -> None:
    """Prove the converter RUNS before spending eight minutes on 11 GB.

    Run 1 of this kernel downloaded the whole checkpoint and then died on
    `ModuleNotFoundError: No module named 'gguf'` — a two-second failure that
    cost a full GPU session because it was discovered after the download rather
    than before it. HARD RULE #8: never continue past a failed setup step, and
    check the step as early as it can be checked.
    """
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "safetensors", "transformers", "gguf"], check=False, timeout=900)
    p = subprocess.run([sys.executable, str(conv), "--help"], text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
    if p.returncode != 0:
        print(p.stdout[-3000:], flush=True)
        raise RuntimeError("converter cannot even run --help; fix deps before downloading")
    if "--lm-quant" not in p.stdout:
        raise RuntimeError("this checkout's converter has no --lm-quant; the clone is stale")
    print("preflight: converter runs and exposes --lm-quant", flush=True)


def main() -> int:
    token = kh.resolve_hf_token()
    from huggingface_hub import snapshot_download

    binary = build()
    fixtures = fetch_fixtures()

    conv = REPO / "models" / "convert-vibevoice-bitnet-to-gguf.py"
    kh.step("preflight.converter")
    preflight_converter(conv)

    kh.step("download.checkpoint")
    with hb("download.checkpoint"):
        snapshot_download(repo_id=HF_CHECKPOINT, local_dir=str(SRC), token=token or None,
                          allow_patterns=["*.json", "*.safetensors", "*.txt"])

    results: dict = {"commit": COMMIT, "gt_ko": GT_KO, "arms": {}}
    order = [k for k in ("ko-test", "ko-mic-cue-kept", "ko-mic-cue-lost", "en-jfk") if k in fixtures]

    # VAE and embedding held at q8_0/q8_0 in every arm so the ONLY thing that
    # moves is the LM's storage — and with it the activation path.
    for lm_q in ("tq2_0", "f16", "q8_0"):
        gguf_path = OUT / f"vibeasr-bitnet-lm-{lm_q}.gguf"
        kh.step(f"convert.{lm_q}")
        with hb(f"convert.{lm_q}"):
            run([sys.executable, str(conv), "--input", str(SRC), "-o", str(gguf_path),
                 "--lm-quant", lm_q, "--vae-quant", "q8_0", "--embed-quant", "q8_0"],
                timeout=5400)
        size_gb = gguf_path.stat().st_size / 1e9
        print(f"  {lm_q}: {size_gb:.2f} GB", flush=True)
        results["arms"][lm_q] = {"_size_gb": round(size_gb, 2)}
        for name in order:
            kh.step(f"run.{lm_q}.{name}")
            with hb(f"run.{lm_q}.{name}"):
                r = transcribe(binary, gguf_path, fixtures[name])
            results["arms"][lm_q][name] = r
            print(f"[{lm_q:6}] {name:16} ok={r['ok']} :: {r['text']}", flush=True)
            (RESULTS / f"{lm_q}.{name}.txt").write_text(r["text"] + "\n")
        # Free the disk before the next variant: three LM variants at up to
        # ~3 GB each plus an 11 GB checkpoint will not fit otherwise.
        gguf_path.unlink(missing_ok=True)

    (RESULTS / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print("\n" + "=" * 78)
    print(f"GT (ko): {GT_KO}")
    for lm_q, rows in results["arms"].items():
        print(f"\n--- LM stored as {lm_q}  ({rows['_size_gb']} GB)")
        for name in order:
            print(f"  {name:16} ok={str(rows[name]['ok']):5} {rows[name]['text'] or '<EMPTY>'}")
    print("=" * 78, flush=True)
    kh.step("done")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        (RESULTS / "FAILED.txt").write_text(traceback.format_exc())
        raise
