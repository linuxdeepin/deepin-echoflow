#!/usr/bin/env python3
"""
Issue #405 — CUDA package left old CPUs with NO cpu backend. PROOF kernel for
the CUDA side of the fix, on a real CUDA box (Kaggle P100/T4).

The report: the linux-x86_64-cuda package builds ONE libggml-cpu.so at
AVX2+FMA+F16C; on the reporter's pre-AVX2 Xeon (Tesla P40 rig)
ggml_backend_score() refuses it, the registry registers only CUDA, and the
process dies on GGML_ASSERT(backend) (--no-gpu, parakeet load) or
GGML_ASSERT(device) (GPU run, whisper LID's make_buft_list).

Fix under test (branch fix/issue-405): the release CUDA legs build with
GGML_CPU_ALL_VARIANTS (loader picks the best CPU module the host supports;
x64 baseline always works) + graceful no-CPU-backend failure paths. The
qemu-Nehalem arms of the proof run on a GH runner
(.github/workflows/linux-isa-fallback-verify.yml) and locally; THIS kernel
proves what needs real CUDA:

  A) the release-CUDA-leg cmake shape (DL + shared + ALL_VARIANTS + CUDA)
     actually configures, builds, and ships the variant set;
  B) parakeet on CUDA with auto whisper-LID — the reporter's exact GPU
     scenario — runs end-to-end with the variant CPU modules present;
  C) --no-gpu runs on the variant CPU backend;
  D) with every libggml-cpu-*.so removed (the reporter's effective registry:
     CUDA only), crispasr exits 1 with the actionable "no CPU ggml backend"
     message instead of the pre-fix SIGABRT.

Results → /kaggle/working/issue405_results.json.
"""
import os, sys, subprocess, json, re, shutil
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp/i405"); TMP.mkdir(parents=True, exist_ok=True)
MODELS = Path("/kaggle/temp/models"); MODELS.mkdir(parents=True, exist_ok=True)
RESULTS = WORK / "issue405_results.json"

HERE = Path(__file__).resolve().parent
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
BRANCH = "fix/issue-405"
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

PARAKEET_REPO = "cstr/parakeet-tdt-0.6b-v3-GGUF"
PARAKEET_FILE = "parakeet-tdt-0.6b-v3-q4_k.gguf"
WHISPER_REPO = "ggerganov/whisper.cpp"
WHISPER_FILE = "ggml-tiny.bin"


def sh(cmd, timeout=None, env=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          timeout=timeout, env=env)


def build_crispasr():
    kh.install_build_toolchain()
    if not (CLONE / "CMakeLists.txt").exists():
        raise RuntimeError("repo clone missing")
    if not (CLONE / "ggml" / "CMakeLists.txt").exists():
        sh(f"cd {CLONE} && git submodule update --init ggml", timeout=900)
    arch = kh.detect_cuda_arch()
    # The release build-linux-x86_64-cuda leg's shape after the #405 change:
    # DL plugins + shared libs + the multi-variant CPU backend.
    flags = (["-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON",
              "-DGGML_BACKEND_DL=ON",
              "-DGGML_NATIVE=OFF", "-DGGML_CPU_ALL_VARIANTS=ON",
              "-DCRISPASR_OPUS_FETCH=ON"]
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
    variants = sorted(p.name for p in binp.parent.glob("libggml-cpu-*.so"))
    kh.step("build.ready", path=str(binp), cpu_variants=variants)
    try:
        kh.export_ccache_tar()
    except Exception:
        pass
    return binp, variants


def run_asr(binp, tag, extra, timeout=1200):
    out_env = dict(os.environ)
    cmd = f"{binp} {extra} -m {MODELS / PARAKEET_FILE} -f {CLONE}/samples/jfk.wav"
    with kh.build_heartbeat(f"run.{tag}", interval_s=30):
        try:
            r = sh(cmd, timeout=timeout, env=out_env)
            rc, out, err = r.returncode, r.stdout or "", r.stderr or ""
        except subprocess.TimeoutExpired:
            rc, out, err = -99, "", "TIMEOUT"
    res = dict(tag=tag, rc=rc,
               transcript_ok=("country" in out.lower()),
               cpu_backend_line=next((l for l in err.splitlines() if "loaded CPU backend" in l), ""),
               no_cpu_msg=("no CPU ggml backend" in err),
               tail=(err[-500:] if rc != 0 else ""))
    kh.step(f"run.{tag}", rc=rc, transcript_ok=res["transcript_ok"],
            cpu_line=res["cpu_backend_line"][-90:], no_cpu_msg=res["no_cpu_msg"])
    return res


def main():
    binp, variants = build_crispasr()

    for repo, f in ((PARAKEET_REPO, PARAKEET_FILE), (WHISPER_REPO, WHISPER_FILE)):
        with kh.build_heartbeat(f"models.{f}", interval_s=30):
            hf_hub_download(repo_id=repo, filename=f, local_dir=str(MODELS), token=HF_TOKEN or None)
    # The LID pass resolves ggml-tiny.bin through the crispasr cache.
    cache = Path.home() / ".cache" / "crispasr"
    cache.mkdir(parents=True, exist_ok=True)
    shutil.copy(MODELS / WHISPER_FILE, cache / WHISPER_FILE)

    results = {"cpu_variants": variants, "runs": {}}
    have_baseline = "libggml-cpu-x64.so" in variants

    # B) the reporter's GPU scenario: parakeet on CUDA, -l auto → whisper LID.
    results["runs"]["gpu"] = run_asr(binp, "gpu", "-l auto")
    RESULTS.write_text(json.dumps(results, indent=2))

    # C) --no-gpu on the variant CPU backend (the arm that hit
    #    GGML_ASSERT(backend) for the reporter).
    results["runs"]["cpu"] = run_asr(binp, "cpu", "--no-gpu", timeout=2400)
    RESULTS.write_text(json.dumps(results, indent=2))

    # D) the reporter's effective registry (CUDA only): remove every CPU
    #    module → must be a clean exit-1 diagnosis, not SIGABRT(134).
    hidden = TMP / "hidden-cpu-modules"
    hidden.mkdir(exist_ok=True)
    for so in binp.parent.glob("libggml-cpu-*.so"):
        shutil.move(str(so), hidden / so.name)
    results["runs"]["no_cpu_modules"] = run_asr(binp, "no_cpu_modules", "-l auto", timeout=600)

    d = results["runs"]["no_cpu_modules"]
    verdict = dict(
        build_has_variants=bool(variants),
        build_has_x64_baseline=have_baseline,
        gpu_ok=bool(results["runs"]["gpu"]["rc"] == 0 and results["runs"]["gpu"]["transcript_ok"]),
        cpu_ok=bool(results["runs"]["cpu"]["rc"] == 0 and results["runs"]["cpu"]["transcript_ok"]),
        graceful_no_cpu=bool(d["rc"] == 1 and d["no_cpu_msg"]),
        pre_fix_would_abort_here=True,  # documented: shipped 0.8.30 rc=134 GGML_ASSERT(backend)
    )
    results["verdict"] = verdict
    RESULTS.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)
    kh.step("DONE", **verdict)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        kh.step("FATAL", err=str(e)[:300])
        traceback.print_exc()
        sys.exit(1)
