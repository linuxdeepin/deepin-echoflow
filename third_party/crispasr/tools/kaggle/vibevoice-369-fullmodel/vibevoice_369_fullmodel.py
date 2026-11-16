#!/usr/bin/env python3
"""#369: does the FULL VibeVoice-ASR model still lose the language cue?

The reporter's grid (crispasr 0.8.29) showed our full model transcribing four of
five Korean clips correctly and failing on `ko-mic-cue-lost.wav`, while
0xShug0/audio.cpp got that file right at matched q8_0. Since then two defects
have been fixed that are exactly places where audio.cpp was right and we were
wrong, and BOTH were live in the binary they measured:

  * the ASR prompt's hardcoded ids decoded to " configuration audio,thonPEND
    itiz" instead of " seconds audio, please transcribe it with" (b6efe1de)
  * the audio loader resampled 16 -> 24 kHz with miniaudio's linear resampler
    (cos 0.944 vs soxr, -10.3 dB alias on an out-of-band tone) (ac4aa478)

So the open question is empirical, not architectural: with those fixed, does the
full model now get `ko-mic-cue-lost.wav` right? That needs the 4.8 GB q4_k, which
does not fit on the 16 GB dev Mac beside a reference forward — hence this kernel.

Three arms, all on the same binary so only the flag differs:

  A  fixed      current main
  B  legacy     CRISPASR_VIBEVOICE_ASR_PROMPT=legacy + CRISPASR_HQ_RESAMPLE=0
                — restores the two dominant differences. NOT a full 0.8.29
                  rebuild: this binary still has the -25 dBFS input
                  normalisation and the exact-erf GELU, neither of which 0.8.29
                  had, so read arm B as "the same binary without the prompt and
                  resampler fixes", not as the reporter's baseline.
  C  cuda       fixed, on CUDA, with and without CRISPASR_VIBEVOICE_ATTN_PREC
                — the GPU half of the flash-attention precision question, which
                  is a CPU no-op and could not be measured on the Mac at all

Everything large lives under /kaggle/temp; /kaggle/working gets only results.json
and the transcripts (gotcha #22 — a fat working dir makes the real artifact
unreachable past the 500-file output page cap).
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
STAGE = Path("/kaggle/temp/vv369")
MODELS = STAGE / "models"
FIX = STAGE / "fixtures"
REPO = Path("/kaggle/temp/CrispASR")
RESULTS = WORK / "results"
for d in (STAGE, MODELS, FIX, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
CRISPASR_BRANCH = "main"
GGUF_REPO = "cstr/vibevoice-asr-GGUF"
GGUF_FILE = "vibevoice-asr-q4_k.gguf"          # 4.8 GB; q8_0 is 8.8 GB, see note below
BITNET_REPO = "cstr/vibevoice-asr-bitnet-GGUF"
BITNET_FILE = "vibevoice-asr-bitnet-embed-q8.gguf"

# The reporter's two mic files. Regenerable only by them, so they are fetched
# from the gist they published rather than vendored.
GIST = "https://gist.github.com/Dev0June/80d074bd1f1b588ae56a8ea68f6b4fd3/raw"
GIST_FILES = {
    "ko-mic-cue-lost.wav": f"{GIST}/ko-mic-cue-lost.wav",
    "ko-mic-cue-kept.wav": f"{GIST}/ko-mic-cue-kept.wav",
    "ko-test.wav": f"{GIST}/ko-test-sentence.wav",
}
GT = "내일 오전에 회의 자료를 보내 주세요"


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


# HARD RULE #8: a clone failure is fatal. Silently falling back to a bundled
# helper would mean the run no longer describes the commit it claims to.
if not (REPO / ".git").exists():
    run(["git", "clone", "--depth", "1", "--branch", CRISPASR_BRANCH,
         "--recursive", CRISPASR_URL, str(REPO)], timeout=2400)
COMMIT = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
kh.step("provenance", commit=COMMIT)
print(f"CrispASR {COMMIT}", flush=True)


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
    flags = ["-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF",
             *kh.cuda_build_flags(arch), *kh.cache_and_link_flags()]
    bdir = REPO / "build-cuda"
    kh.step("build.configure", arch=arch)
    with hb("build.configure"):
        run(["cmake", "-S", REPO, "-B", bdir, "-G", "Ninja", *flags], timeout=2400)
    jobs = kh.safe_build_jobs(gpu=True)
    kh.step("build.compile", jobs=jobs)
    with hb("build.compile"):
        run(["cmake", "--build", bdir, "-j", str(jobs), "--target", "crispasr"], timeout=5400)
    binary = bdir / "bin" / "crispasr"
    if not binary.is_file():
        raise RuntimeError("crispasr did not link")
    return binary


def fetch_fixtures() -> dict:
    """The gist WAVs, plus the atempo pair that straddles the reporter's flip."""
    import urllib.request
    out = {}
    for name, url in GIST_FILES.items():
        dst = FIX / name
        if not dst.exists():
            with hb(f"fetch.{name}"):
                urllib.request.urlopen(url, timeout=120)  # noqa: S310  (fixed gist URL)
                urllib.request.urlretrieve(url, dst)      # noqa: S310
        out[name.replace(".wav", "")] = dst
    # atempo 0.535 / 0.525 — the minimal pair from issue comment 5335886546.
    for tempo, label in ((0.535, "flip-en"), (0.525, "flip-ko")):
        dst = FIX / f"{label}.wav"
        if not dst.exists():
            run(["ffmpeg", "-v", "error", "-i", str(out["ko-test"]), "-af", f"atempo={tempo}",
                 "-c:a", "pcm_s16le", str(dst), "-y"], timeout=300)
        out[label] = dst
    for k, v in out.items():
        print(f"  fixture {k}: {v.stat().st_size} bytes", flush=True)
    return out


# Transcript sentinels. "[Silence]" is what the CLI emits when the model
# returns no utterance — a WRONG answer that must not be scored as a pass.
EMPTY_MARKERS = ("[Silence]", "[BLANK_AUDIO]", "[silence]")


def transcribe(binary: Path, model: Path, wav: Path, *, backend: str, gpu: bool,
               env: dict | None = None, timeout=3600) -> dict:
    """Run one transcription, keeping stdout (the transcript) apart from stderr.

    ⚠ The first version of this kernel merged the two streams and then filtered
    the transcript out of the mix by prefix. ggml's CUDA banner ("  Device 0:
    Tesla P100...") does not match any of those prefixes, so it landed in the
    `text` field — which meant a run that produced NO transcript still looked
    non-empty, and the proof-of-work check reported "0 of 25 bad" over four arms
    that had actually answered "[Silence]" twice each. Separating the streams
    removes the guesswork; EMPTY_MARKERS covers the rest.
    """
    argv = [binary, "--backend", backend, "-m", model, "-t", "4", "-f", wav, "-nt"]
    if not gpu:
        argv.append("-ng")
    argv = [str(x) for x in argv]
    merged = os.environ.copy()
    if env:
        merged.update({str(k): str(v) for k, v in env.items()})
    print("$ " + " ".join(argv), flush=True)
    p = subprocess.run(argv, env=merged, text=True, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    text = p.stdout.strip()
    # The CLI prefixes diarized output with "(Speaker N) "; keep it out of the
    # comparison but record it.
    speaker = None
    m = re.match(r"^\(Speaker (\d+)\)\s*", text)
    if m:
        speaker = int(m.group(1))
        text = text[m.end():]
    # What backend did it actually pick? An arm labelled "cpu" that silently ran
    # on CUDA would make the CPU/GPU comparison meaningless.
    dev = "cuda" if re.search(r"backend:\s*CUDA", p.stderr) else (
        "cpu" if re.search(r"backend:\s*CPU", p.stderr) else "?")
    ok = p.returncode == 0 and bool(text) and not any(k in text for k in EMPTY_MARKERS)
    return {"rc": p.returncode, "text": text, "speaker": speaker, "device": dev,
            "ok": ok, "stderr_tail": p.stderr[-1500:]}


def main() -> int:
    token = kh.resolve_hf_token()
    from huggingface_hub import hf_hub_download

    binary = build()
    fixtures = fetch_fixtures()

    kh.step("download.full")
    with hb("download.full"):
        full = Path(hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE,
                                    local_dir=str(MODELS), token=token or None))
    kh.step("download.bitnet")
    with hb("download.bitnet"):
        bitnet = Path(hf_hub_download(repo_id=BITNET_REPO, filename=BITNET_FILE,
                                      local_dir=str(MODELS), token=token or None))

    order = ["ko-test", "ko-mic-cue-kept", "ko-mic-cue-lost", "flip-en", "flip-ko"]
    legacy_env = {"CRISPASR_VIBEVOICE_ASR_PROMPT": "legacy", "CRISPASR_HQ_RESAMPLE": "0"}
    results: dict = {"commit": COMMIT, "gt": GT, "model": GGUF_FILE, "arms": {}}

    # Roll the fixes back one at a time. Run 1 showed the full model getting
    # ko-mic-cue-lost.wav RIGHT in both the fixed and the legacy arm — so
    # whatever repaired it is something legacy does not roll back, and the
    # obvious candidate is the -25 dBFS input normalisation (b7a1a71c), which
    # was already in the binary and which I had written off after it failed to
    # help the BITNET checkpoint. nonorm isolates it; v0829 is the closest
    # reconstruction of what the reporter actually ran.
    nonorm_env = {"CRISPASR_VIBEVOICE_NO_INPUT_NORM": "1"}
    v0829_env = {**legacy_env, **nonorm_env, "CRISPASR_VIBEVOICE_GELU_TANH": "1"}
    arms = [
        ("full.cpu.fixed", full, "vibevoice", False, None),
        ("full.cpu.legacy", full, "vibevoice", False, legacy_env),
        ("full.cpu.nonorm", full, "vibevoice", False, nonorm_env),
        ("full.cpu.v0829", full, "vibevoice", False, v0829_env),
        ("full.cuda.fixed", full, "vibevoice", True, None),
        ("full.cuda.prec_default", full, "vibevoice", True,
         {"CRISPASR_VIBEVOICE_ATTN_PREC": "default"}),
        ("bitnet.cpu.fixed", bitnet, "vibevoice-bitnet", False, None),
    ]
    for arm, model, backend, gpu, env in arms:
        results["arms"][arm] = {}
        for name in order:
            kh.step(f"run.{arm}.{name}")
            with hb(f"run.{arm}.{name}"):
                r = transcribe(binary, model, fixtures[name], backend=backend, gpu=gpu, env=env)
            results["arms"][arm][name] = r
            print(f"[{arm:22}] {name:16} rc={r['rc']} dev={r['device']} ok={r['ok']} :: {r['text']}",
                  flush=True)
            (RESULTS / f"{arm}.{name}.txt").write_text(r["text"] + "\n")

    (RESULTS / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))

    print("\n" + "=" * 78)
    print(f"GT: {GT}")
    for arm in results["arms"]:
        rows = results["arms"][arm]
        devs = {r["device"] for r in rows.values()}
        print(f"\n--- {arm}   (backend actually used: {sorted(devs)})")
        for name in order:
            r = rows[name]
            print(f"  {name:16} ok={str(r['ok']):5} {r['text'] or '<EMPTY>'}")
    bad = [f"{a}.{n}" for a, rows in results["arms"].items()
           for n, r in rows.items() if not r["ok"]]
    print(f"\nruns that produced no usable transcript: {len(bad)} — {bad}")
    print("=" * 78, flush=True)
    kh.step("done", unusable=len(bad))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        (RESULTS / "FAILED.txt").write_text(traceback.format_exc())
        raise
