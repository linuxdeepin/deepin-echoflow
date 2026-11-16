#!/usr/bin/env python3
"""
Issue #419 — canary outputs Russian as Latin translit (Subtitle Edit,
Windows, GPU). REPRO kernel on a real CUDA box.

Local findings so far (VPS, CPU): canary-1b-v2-q4_k with -l ru produces
proper CYRILLIC on both the single-pass (11 s) and streamed (56 s) paths, and
the GGUF vocab carries 2175 Cyrillic pieces with <|ru|> present — so the
language plumbing and detokenizer are fine, and the model chooses Latin
pieces only in the reporter's environment. The differentials left are the
GPU decode (SE ships GPU builds; --gpu-backend auto) and --sensitivity.

This kernel runs canary on Russian audio (the GigaAM regression fixture —
native Russian speech) on CUDA vs CPU, single-pass and streamed, and
classifies each transcript's script. A Latin-dominated GPU transcript with a
Cyrillic CPU control reproduces #419 and pins it on GPU numerics; all-Cyrillic
means the CUDA arm is clean and the hunt moves to Vulkan.

Results → /kaggle/working/issue419_results.json.
"""
import os, sys, subprocess, json, re, wave, array
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp/i419"); TMP.mkdir(parents=True, exist_ok=True)
MODELS = Path("/kaggle/temp/models"); MODELS.mkdir(parents=True, exist_ok=True)
RESULTS = WORK / "issue419_results.json"

HERE = Path(__file__).resolve().parent
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
BRANCH = "main"
CLONE = Path("/kaggle/temp/CrispASR")
if not CLONE.exists():
    try:
        subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                        "--recurse-submodules", CRISPASR_URL, str(CLONE)],
                       check=True, timeout=1200)
    except Exception as e:
        print(f"clone failed: {e}", flush=True)
sys.path.insert(0, str(CLONE / "tools" / "kaggle") if CLONE.exists() else str(HERE))
import kaggle_harness as kh
kh.init_progress()

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=False)
HF_TOKEN = kh.resolve_hf_token()
kh.step("hf.token", present=bool(HF_TOKEN))
from huggingface_hub import hf_hub_download

CANARY_REPO = "cstr/canary-1b-v2-GGUF"
CANARY_FILE = "canary-1b-v2-q4_k.gguf"
FIXTURE_REPO = "cstr/crispasr-regression-fixtures"
FIXTURE_FILE = "gigaam-ctc/gigaam_example/audio.wav"


def sh(cmd, timeout=None, env=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          timeout=timeout, env=env)


def build_crispasr():
    kh.install_build_toolchain()
    if not (CLONE / "ggml" / "CMakeLists.txt").exists():
        sh(f"cd {CLONE} && git submodule update --init ggml", timeout=900)
    arch = kh.detect_cuda_arch()
    flags = (["-DCMAKE_BUILD_TYPE=Release", "-DCRISPASR_OPUS_FETCH=ON"]
             + kh.cuda_build_flags(arch) + kh.cache_and_link_flags())
    kh.step("build.configure", arch=arch)
    r = sh(f"cd {CLONE} && cmake -G Ninja -B build " + " ".join(flags), timeout=1800)
    if r.returncode != 0:
        kh.step("build.configure_FAILED", stderr=r.stderr[-2000:])
        raise RuntimeError("cmake configure failed")
    jobs = kh.safe_build_jobs(gpu=True)
    with kh.build_heartbeat("build.ninja", interval_s=30):
        kh.sh_with_progress(f"cmake --build build -j{jobs} --target crispasr-cli", cwd=str(CLONE))
    binp = CLONE / "build" / "bin" / "crispasr"
    if not binp.exists():
        raise RuntimeError("crispasr binary not produced")
    os.environ["LD_LIBRARY_PATH"] = str(binp.parent) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    kh.step("build.ready", path=str(binp))
    try:
        kh.export_ccache_tar()
    except Exception:
        pass
    return binp


def script_stats(text):
    cyr = len(re.findall(r"[Ѐ-ӿ]", text))
    lat = len(re.findall(r"[A-Za-z]", text))
    return dict(cyrillic=cyr, latin=lat,
                verdict=("cyrillic" if cyr > 3 * max(1, lat) else
                         "latin_translit" if lat > 3 * max(1, cyr) else "mixed"))


def run_arm(binp, tag, wav, extra, env_extra=None, timeout=1800):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    cmd = f"{binp} --backend canary -m {MODELS / CANARY_FILE} -l ru {extra} -f {wav}"
    with kh.build_heartbeat(f"run.{tag}", interval_s=30):
        try:
            r = sh(cmd, timeout=timeout, env=env)
            rc, out, err = r.returncode, r.stdout or "", r.stderr or ""
        except subprocess.TimeoutExpired:
            rc, out, err = -99, "", "TIMEOUT"
    text = " ".join(l for l in out.splitlines() if l.strip())
    res = dict(tag=tag, rc=rc, transcript=text[:400], **script_stats(text),
               tail=(err[-400:] if rc != 0 else ""))
    kh.step(f"run.{tag}", rc=rc, verdict=res["verdict"],
            cyr=res["cyrillic"], lat=res["latin"], head=text[:80])
    return res


def main():
    binp = build_crispasr()
    for repo, f, kind in ((CANARY_REPO, CANARY_FILE, "model"), (FIXTURE_REPO, FIXTURE_FILE, "dataset")):
        with kh.build_heartbeat(f"dl.{f}", interval_s=30):
            hf_hub_download(repo_id=repo, filename=f, local_dir=str(MODELS),
                            repo_type=kind, token=HF_TOKEN or None)
    ru = MODELS / FIXTURE_FILE

    # 5x loop → 56 s → forces the streamed 30-40 s chunked path.
    src = wave.open(str(ru), "rb")
    a = array.array("h"); a.frombytes(src.readframes(src.getnframes())); sr = src.getframerate(); src.close()
    ru5x = TMP / "ru5x.wav"
    out = wave.open(str(ru5x), "wb"); out.setnchannels(1); out.setsampwidth(2)
    out.setframerate(sr); out.writeframes((a * 5).tobytes()); out.close()

    results = {"runs": {}}
    # The reporter's shape: GPU (auto), plus their exact extra flags.
    results["runs"]["gpu_single"] = run_arm(binp, "gpu_single", ru, "")
    RESULTS.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    results["runs"]["gpu_streamed"] = run_arm(binp, "gpu_streamed", ru5x, "")
    RESULTS.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    results["runs"]["gpu_se_flags"] = run_arm(binp, "gpu_se_flags", ru5x,
                                              "-sl ru --sensitivity aggressive -sp -sow")
    RESULTS.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    results["runs"]["cpu_control"] = run_arm(binp, "cpu_control", ru, "--no-gpu", timeout=2400)

    verdicts = {k: v["verdict"] for k, v in results["runs"].items()}
    reproduced = any(v == "latin_translit" for k, v in verdicts.items() if k.startswith("gpu"))
    results["verdict"] = dict(per_arm=verdicts, translit_reproduced_on_cuda=reproduced,
                              cpu_control_cyrillic=(verdicts.get("cpu_control") == "cyrillic"))
    RESULTS.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)
    kh.step("DONE", **results["verdict"])


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        kh.step("FATAL", err=str(e)[:300])
        traceback.print_exc()
        sys.exit(1)
