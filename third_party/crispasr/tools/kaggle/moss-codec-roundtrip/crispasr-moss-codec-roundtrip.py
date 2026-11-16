#!/usr/bin/env python3
"""
Validate moss_tts_local_codec::encode() by round-trip.

decode() is already trusted — it ships and produces correct audio. So if encode()
is the right analysis inverse, re-encoding decode's output must recover
approximately the codes we started from. A wrong nearest-entry rule (L2 where the
model uses cosine, say) leaves quantizer-0 agreement at chance, 1/1024 ~= 0.1%,
which separates cleanly from a correct inverse.

This exists because encode() was written from INFERENCE, not measurement: the
cosine rule and the reversed attention contexts were reasoned from the v1 codec
runtime. #249 is the standing reminder that plausible-and-wrong survives a long
time unless something measures it.

Follows the harness regime (clone in-kernel, import from the clone, heartbeat).
"""
import os, subprocess, sys, json
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp"); TMP.mkdir(parents=True, exist_ok=True)
REPO = TMP / "CrispASR"
BUILD = TMP / "build"
CODEC_REPO = "cstr/moss-tts-local-v1.5-GGUF"
CODEC_FILE = os.environ.get("MOSS_CODEC_FILE", "moss-tts-local-v1.5-codec-enc.gguf")

if not REPO.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--recurse-submodules",
                    "--shallow-submodules", "https://github.com/CrispStrobe/CrispASR.git", str(REPO)],
                   check=True, timeout=2400)

_h = REPO / "tools" / "kaggle"
sys.path.insert(0, str(_h if (_h / "kaggle_harness.py").exists() else Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
res = {}


def sh(cmd, cwd=None, timeout=3600):
    print(f"$ {cmd}", flush=True)
    p = subprocess.run(cmd, shell=True, cwd=cwd, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


kh.step("toolchain")
kh.install_build_toolchain()
token = kh.resolve_hf_token()
if token:
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token

# CPU-only: the round-trip is tiny (a couple of dozen frames) and this needs no
# GPU, so it stays off the GPU quota entirely.
kh.step("cmake")
# EXAMPLES=ON is required: crispasr-cli lives there, and the ASR leg of the
# acceptance test needs it. With it OFF the build fails late with
# "ninja: error: unknown target 'crispasr-cli'" — after the round-trip has
# already succeeded, which reads as a failed test when it is a failed harness.
flags = ["-DCMAKE_BUILD_TYPE=Release", "-DCRISPASR_BUILD_TESTS=OFF",
         "-DCRISPASR_BUILD_EXAMPLES=ON", "-DCRISPASR_BUILD_SERVER=OFF"] + kh.cache_and_link_flags()
with kh.build_heartbeat("cmake.configure"):
    rc, out = sh(f"cmake -S {REPO} -B {BUILD} -G Ninja " + " ".join(flags))
if rc != 0:
    print(out[-6000:], flush=True); raise SystemExit("configure failed")
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-lib -j{kh.safe_build_jobs(gpu=False)}")

kh.step("compile probe")
probe = REPO / "tools" / "moss-codec" / "moss_codec_roundtrip.cpp"
rc, out = sh(f"c++ -std=gnu++17 -O2 -I {REPO}/src -I {REPO}/ggml/include {probe} "
             f"-o {TMP}/roundtrip -L {BUILD}/src -lcrispasr "
             f"-L {BUILD}/ggml/src -lggml-base -lggml-cpu -lggml "
             f"-Wl,-rpath,{BUILD}/src -Wl,-rpath,{BUILD}/ggml/src")
if rc != 0:
    print(out[-4000:], flush=True); raise SystemExit("probe compile failed")

kh.step("download codec")
from huggingface_hub import hf_hub_download  # noqa: E402
codec = hf_hub_download(repo_id=CODEC_REPO, filename=CODEC_FILE, token=token,
                        local_dir=str(TMP / "codec"))
print(f"codec: {codec} ({Path(codec).stat().st_size/1e9:.2f} GB)", flush=True)

# 1. cheap smoke test — does the encoder invert at all?
kh.step("roundtrip.codes")
with kh.build_heartbeat("roundtrip.codes", 30.0):
    rc, out = sh(f"{TMP}/roundtrip {codec} --codes", timeout=3600)
print(out, flush=True)
res["codes_rc"] = rc
res["codes_output"] = out[-3000:]

# 2. the acceptance test (HARD RULE 3): real speech -> encode -> decode -> ASR.
# Codes-vs-codes agreement cannot decide this — random codes decode to audio off
# the manifold the encoder was trained on, and RVQ is lossy regardless. Only the
# decoded output settles whether encode() is right.
kh.step("roundtrip.speech")
wav_in = REPO / "samples" / "jfk.wav"
wav_out = TMP / "moss_reconstructed.wav"
with kh.build_heartbeat("roundtrip.speech", 30.0):
    rc, out = sh(f"{TMP}/roundtrip {codec} {wav_in} {wav_out}", timeout=5400)
print(out, flush=True)
res["speech_rc"] = rc
res["speech_output"] = out[-3000:]

if rc == 0 and wav_out.exists():
    import shutil
    shutil.copy(str(wav_out), str(WORK / "moss_reconstructed.wav"))  # for a human listen
    kh.step("asr")
    # Build the CLI and transcribe the reconstruction. If encode() is correct the
    # words survive the round-trip; if the latent is misaligned they will not.
    with kh.build_heartbeat("cli.build"):
        kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli "
                            f"-j{kh.safe_build_jobs(gpu=False)}")
    asr = BUILD / "bin" / "crispasr"
    for label, path in (("original", wav_in), ("reconstructed", wav_out)):
        rc2, out2 = sh(f"{asr} --backend parakeet -m auto --auto-download -f {path} --no-prints",
                       timeout=3600)
        text = out2.strip().splitlines()[-1] if out2.strip() else ""
        print(f"[{label}] rc={rc2} :: {text}", flush=True)
        res[f"asr_{label}"] = text
res["rc"] = res.get("speech_rc", 1)
(WORK / "moss_codec_roundtrip.json").write_text(json.dumps(res, indent=2))
kh.step("done", rc=rc)
if rc != 0:
    raise SystemExit(rc)
