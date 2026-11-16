#!/usr/bin/env python3
"""
Can KugelAudio's Q4_K be the registry default? (sweep follow-up)

KugelAudio was the only multi-GB registry entry defaulting to F16. That F16 is
17.3 GB — over the VRAM of every 16 GB card — so `-m auto` hard-failed on Kaggle's
P100/T4 and on most user GPUs. Q4_K (5.7 GB) is published alongside it and fits,
and is now the default; this run checks the bare `-m auto` path gets it.

The one thing that would justify keeping F16 as the default is Q4_K sounding
worse. The only prior evidence is a commit that switched the DIFF HARNESS to F16
("Q4_K too lossy") — but a per-stage numeric diff legitimately needs F16, and
that says nothing about the audio. So measure the audio: synthesize the same
text at both quants and ASR-roundtrip each (feedback_tts_validation).

F16 will not load on a 16 GB GPU, so it runs on CPU here — slow but the point is
the words, not the wall-clock. If Q4_K holds the transcript, the default moves.

Follows the harness regime (clone in-kernel, import from the clone, heartbeat).
"""
import os, subprocess, sys, json, re, wave, shutil
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp"); TMP.mkdir(parents=True, exist_ok=True)
REPO = TMP / "CrispASR"
BUILD = TMP / "build"
TEXT = ("The quick brown fox jumps over the lazy dog while the morning light "
        "spills across the empty street.")

if not REPO.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--recurse-submodules",
                    "--shallow-submodules", "https://github.com/CrispStrobe/CrispASR.git", str(REPO)],
                   check=True, timeout=2400)

_h = REPO / "tools" / "kaggle"
sys.path.insert(0, str(_h if (_h / "kaggle_harness.py").exists() else Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh._HF_PROGRESS_PATH = "runs/kugelaudio-quant-live.jsonl"
kh.init_progress()
res = {"text": TEXT}


def sh(cmd, timeout=3600):
    print(f"$ {cmd}", flush=True)
    p = subprocess.run(cmd, shell=True, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


kh.step("toolchain")
kh.install_build_toolchain()
token = kh.resolve_hf_token()
if token:
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token

cuda = kh.detect_cuda_arch()
kh.step("cmake", cuda_arch=cuda)
flags = ["-DCMAKE_BUILD_TYPE=Release", "-DCRISPASR_BUILD_TESTS=OFF",
         "-DCRISPASR_BUILD_EXAMPLES=ON", "-DCRISPASR_BUILD_SERVER=OFF",
         "-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={cuda}"] + kh.cache_and_link_flags()
with kh.build_heartbeat("cmake.configure"):
    rc, out = sh(f"cmake -S {REPO} -B {BUILD} -G Ninja " + " ".join(flags))
if rc != 0:
    print(out[-6000:], flush=True); raise SystemExit("configure failed")
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli "
                        f"-j{kh.safe_build_jobs(gpu=True)}")
cli = BUILD / "bin" / "crispasr"
if not cli.exists():
    raise SystemExit("crispasr-cli missing after build")


def dur_s(p):
    try:
        with wave.open(str(p)) as w:
            return round(w.getnframes() / float(w.getframerate()), 2)
    except Exception:  # noqa: BLE001
        return 0.0


def synth(label, extra, timeout):
    out_wav = TMP / f"kugel_{label}.wav"
    with kh.build_heartbeat(f"synth.{label}", 30.0):
        rc, out = sh(f"{cli} --backend kugelaudio -m auto --auto-download {extra} "
                     f"--tts \"{TEXT}\" --tts-output {out_wav} --no-prints", timeout=timeout)
    print(out[-4000:], flush=True)
    res[f"{label}_rc"] = rc
    res[f"{label}_log"] = out[-2500:]
    if rc == 0 and out_wav.exists() and out_wav.stat().st_size > 1000:
        shutil.copy(str(out_wav), str(WORK / out_wav.name))  # for a human listen
        res[f"{label}_dur_s"] = dur_s(out_wav)
        return out_wav
    return None


# Bare `-m auto` — the path a user actually gets. Since the registry default
# moved to Q4_K this must resolve to the 5.7 GB file and fit; if it still pulls
# the F16 it will OOM here, which is exactly the regression worth catching.
kh.step("synth auto (gpu, registry default)")
auto = synth("auto", "", 1800)

# The same weights requested explicitly. Belt-and-braces: if `auto` failed but
# this passes, the bug is the registry default, not the model.
kh.step("synth q4_k (gpu)")
q4 = synth("q4_k", "--model-quant q4_k", 1800)

# F16 is the incumbent. It cannot fit 16 GB of VRAM, so this is the CPU run —
# included to answer "is Q4_K worse?", not to benchmark speed.
kh.step("synth f16 (cpu)")
f16 = synth("f16", "--model-quant f16 --no-gpu", 5400)

kh.step("asr")


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def wer(ref, hyp):
    r, h = norm(ref).split(), norm(hyp).split()
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r)][len(h)] / max(1, len(r))


for label, path in (("auto", auto), ("q4_k", q4), ("f16", f16)):
    if not path:
        continue
    rc, out = sh(f"{cli} --backend parakeet -m auto --auto-download -f {path} --no-prints", timeout=3600)
    text = out.strip().splitlines()[-1] if out.strip() else ""
    res[f"asr_{label}"] = text
    res[f"wer_{label}"] = round(wer(TEXT, text), 4)
    print(f"[{label}] wer={res[f'wer_{label}']} :: {text}", flush=True)

# Verdict. Q4_K earns the default by being intelligible on its own terms — not by
# matching F16 exactly, which no 4-bit quant does. If F16 could not be produced
# here the comparison is simply absent; that is reported, not papered over.
res["q4k_usable"] = res.get("wer_q4_k", 1.0) <= 0.25
res["auto_usable"] = res.get("wer_auto", 1.0) <= 0.25
# Which file `-m auto` actually fetched. Succeeding on a 16 GB GPU already
# implies it is not the 17.3 GB F16, but name the file so the record is explicit
# rather than inferred.
res["auto_downloaded"] = sorted(
    f"{q.name} ({q.stat().st_size/1e9:.1f} GB)"
    for q in Path.home().joinpath(".cache/crispasr").rglob("kugelaudio*.gguf")
)
res["f16_measured"] = f16 is not None
if "wer_f16" in res and "wer_q4_k" in res:
    res["wer_delta_q4k_minus_f16"] = round(res["wer_q4_k"] - res["wer_f16"], 4)
(WORK / "kugelaudio_quant.json").write_text(json.dumps(res, indent=2))
kh.step("done", auto_usable=res["auto_usable"], q4k_usable=res["q4k_usable"],
        wer_auto=res.get("wer_auto"), wer_q4_k=res.get("wer_q4_k"), wer_f16=res.get("wer_f16"))
print(json.dumps(res, indent=2), flush=True)
