#!/usr/bin/env python3
"""
Issue #402 — chatterbox-turbo/Nano T3 crashes in Vulkan flash_attn_ext on
Radeon 780M/RADV; naive attention works. PROOF kernel on real NVIDIA Vulkan.

We have no RADV box, so the shipped policy (chatterbox_attn_policy.h) makes
the explicit softmax(QK^T)V path the DEFAULT whenever the T3 backend is
Vulkan, with CRISPASR_CHATTERBOX_FLASH_ATTN=1 opting back in and the
pre-existing CRISPASR_CHATTERBOX_NAIVE_ATTN=1 debug gate unchanged. This
kernel proves, on a real (NVIDIA) Vulkan driver, that:

  A) vk_default : CRISPASR_CHATTERBOX_T3_GPU=1 → the policy fires
                  ("T3 GPT-2 attention = naive ... backend Vulkan"), the
                  431-step-class AR decode completes, audio is non-silent,
                  and whisper-tiny ASR round-trips the sentence.
  B) vk_flash   : + CRISPASR_CHATTERBOX_FLASH_ATTN=1 → the opt-in reaches
                  flash_attn_ext again (informational: NVIDIA Vulkan doesn't
                  crash like RADV; the run documents both the policy log line
                  and the outcome).
  C) cpu_ref    : --no-gpu reference for the ASR-overlap comparison.

Uses chatterbox-turbo (same run_t3_gpt2_kv GPT-2 path as Nano; there is no
hosted Nano GGUF). Results → /kaggle/working/issue402_results.json.
"""
import os, sys, subprocess, json, re, wave, array, math
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp/i402"); TMP.mkdir(parents=True, exist_ok=True)
MODELS = Path("/kaggle/temp/models"); MODELS.mkdir(parents=True, exist_ok=True)
RESULTS = WORK / "issue402_results.json"

HERE = Path(__file__).resolve().parent
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
BRANCH = "main"  # proof runs used fix/issue-398-402 (merged); main keeps re-runs working
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

CB_REPO = "cstr/chatterbox-turbo-GGUF"
T3_FILE = "chatterbox-turbo-t3-q4_k.gguf"      # reporter reproduced on q4_k + f16
S3GEN_FILE = "chatterbox-turbo-s3gen-f16.gguf" # discover_s3gen()'s preferred turbo name
WHISPER_REPO = "ggerganov/whisper.cpp"
WHISPER_FILE = "ggml-tiny.en.bin"
SENT = "The quick brown fox jumps over the lazy dog."
SEED = 0  # matches the reporter's sampler seed


def sh(cmd, timeout=None, env=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          timeout=timeout, env=env)


def install_vulkan_sdk():
    """Runtime + dev toolchain for building ggml-vulkan (glslc is LunarG-only)."""
    for pkg in ("libvulkan1", "vulkan-tools", "libvulkan-dev", "glslang-tools", "spirv-tools"):
        sh(f"DEBIAN_FRONTEND=noninteractive apt-get install -y -qq {pkg}")
    glslc = sh("which glslc").stdout.strip()
    if not glslc:
        codename = sh("bash -lc '. /etc/os-release; echo $VERSION_CODENAME'").stdout.strip() or "jammy"
        sh("wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc "
           "| tee /etc/apt/trusted.gpg.d/lunarg.asc >/dev/null")
        sh(f"wget -qO /etc/apt/sources.list.d/lunarg-vulkan-{codename}.list "
           f"https://packages.lunarg.com/vulkan/lunarg-vulkan-{codename}.list")
        sh("apt-get update -qq")
        sh("DEBIAN_FRONTEND=noninteractive apt-get install -y -qq vulkan-sdk")
        glslc = sh("which glslc").stdout.strip()
        if not glslc:
            sh("DEBIAN_FRONTEND=noninteractive apt-get install -y -qq shaderc glslc")
            glslc = sh("which glslc").stdout.strip()
    kh.step("vulkan.glslc", path=glslc)
    if glslc:
        os.environ["GLSLC_PATH"] = glslc
    return glslc


def enable_vulkan():
    kh.step("vulkan.install")
    sh("apt-get update -qq")
    install_vulkan_sdk()
    smi = sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
    drv = (smi.stdout.strip().splitlines() or [""])[0]
    major = drv.split(".")[0] if drv else ""
    kh.step("vulkan.nvidia_driver", driver=drv)
    if major:
        sh(f"DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libnvidia-gl-{major}")
    have_glx = bool(sh("ldconfig -p | grep -F libGLX_nvidia.so.0").stdout.strip())
    if have_glx and "NVIDIA" not in sh("vulkaninfo --summary 2>/dev/null").stdout:
        os.makedirs("/usr/share/vulkan/icd.d", exist_ok=True)
        Path("/usr/share/vulkan/icd.d/nvidia_icd.json").write_text(
            '{"file_format_version":"1.0.0","ICD":'
            '{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.277"}}')
    r = sh("vulkaninfo --summary 2>/dev/null")
    devs = [l.split("=")[-1].strip() for l in r.stdout.splitlines() if "deviceName" in l]
    kh.step("vulkan.devices", devices=devs)
    return devs, any("NVIDIA" in d for d in devs)


def build_crispasr():
    kh.install_build_toolchain()
    if not (CLONE / "CMakeLists.txt").exists():
        raise RuntimeError("repo clone missing")
    if not (CLONE / "ggml" / "CMakeLists.txt").exists():
        sh(f"cd {CLONE} && git submodule update --init ggml", timeout=900)
    flags = ["-DCMAKE_BUILD_TYPE=Release", "-DGGML_VULKAN=ON", "-DGGML_CUDA=OFF",
             "-DGGML_NATIVE=OFF", "-DGGML_AVX2=ON", "-DGGML_FMA=ON", "-DGGML_F16C=ON",
             "-DCRISPASR_OPUS_FETCH=ON"] + kh.cache_and_link_flags()
    glslc = os.environ.get("GLSLC_PATH")
    if glslc:
        flags.append(f"-DVulkan_GLSLC_EXECUTABLE={glslc}")
    r = sh(f"cd {CLONE} && cmake -G Ninja -B build-vk " + " ".join(flags), timeout=1200)
    if r.returncode != 0:
        kh.step("build.configure_FAILED", stderr=r.stderr[-2000:])
        raise RuntimeError("cmake configure failed")
    jobs = kh.safe_build_jobs(gpu=False)
    with kh.build_heartbeat("build.ninja", interval_s=30):
        kh.sh_with_progress(f"cmake --build build-vk -j{jobs} --target crispasr-cli", cwd=str(CLONE))
    binp = CLONE / "build-vk" / "bin" / "crispasr"
    if not binp.exists():
        raise RuntimeError("crispasr binary not produced")
    os.environ["LD_LIBRARY_PATH"] = str(binp.parent) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    kh.step("build.ready", path=str(binp))
    try:
        kh.export_ccache_tar()
    except Exception:
        pass
    return str(binp)


def wav_stats(path):
    try:
        w = wave.open(str(path), "rb")
        sr = w.getframerate()
        a = array.array("h"); a.frombytes(w.readframes(w.getnframes())); w.close()
        if not len(a):
            return dict(dur=0.0, peak=0.0, rms=0.0)
        peak = max(abs(x) for x in a) / 32768.0
        rms = math.sqrt(sum((x / 32768.0) ** 2 for x in a) / len(a))
        return dict(dur=round(len(a) / sr, 2), peak=round(peak, 4), rms=round(rms, 4))
    except Exception as e:
        return dict(err=str(e))


def asr(binp, whisp, wav):
    if not Path(wav).exists():
        return ""
    r = sh(f"{binp} -m {whisp} -f {wav} --language en --no-gpu", timeout=600)
    txt = " ".join(re.sub(r"\[[^\]]*\]", "", l) for l in r.stdout.splitlines()
                   if l.strip().startswith("["))
    return re.sub(r"\s+", " ", txt).strip()


def word_overlap(ref, hyp):
    rw = set(re.findall(r"[a-z']+", ref.lower()))
    hw = set(re.findall(r"[a-z']+", hyp.lower()))
    return round(len(rw & hw) / len(rw), 3) if rw else 0.0


def run_tts(binp, t3, tag, env_extra, cli_extra=None, timeout=1800):
    out = TMP / f"cb_{tag}.wav"
    env = dict(os.environ); env.update(env_extra)
    cmd = [binp, "--backend", "chatterbox-turbo", "-m", str(t3), "--tts", SENT,
           "--seed", str(SEED), "--tts-output", str(out)]
    if cli_extra:
        cmd += cli_extra
    rc, err, timed_out = -1, "", False
    with kh.build_heartbeat(f"tts.{tag}", interval_s=30):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
            rc, err = r.returncode, (r.stderr or "")
        except subprocess.TimeoutExpired as e:
            timed_out = True
            err = (e.stderr or b"").decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
    attn_line = next((l for l in err.splitlines() if "T3 GPT-2 attention" in l), "")
    res = dict(tag=tag, rc=rc, timed_out=timed_out, attn_line=attn_line,
               audio=wav_stats(out), wav=str(out),
               stderr_tail=err[-800:] if rc != 0 else "")
    kh.step(f"tts.{tag}", rc=rc, attn=attn_line, audio=res["audio"])
    return res


def main():
    devs, nvidia = enable_vulkan()
    binp = build_crispasr()
    for repo, f in ((CB_REPO, T3_FILE), (CB_REPO, S3GEN_FILE), (WHISPER_REPO, WHISPER_FILE)):
        with kh.build_heartbeat(f"models.{f}", interval_s=30):
            hf_hub_download(repo_id=repo, filename=f, local_dir=str(MODELS), token=HF_TOKEN or None)
    t3 = MODELS / T3_FILE
    whisp = MODELS / WHISPER_FILE

    results = {"env": {"vulkan_devices": devs, "real_nvidia": nvidia}, "runs": {}}
    T3GPU = {"CRISPASR_CHATTERBOX_T3_GPU": "1"}

    # A) the fix's default on Vulkan: naive attention, full synthesis.
    a = run_tts(binp, t3, "vk_default", T3GPU)
    results["runs"]["vk_default"] = a
    RESULTS.write_text(json.dumps(results, indent=2))

    # B) opt back into flash_attn_ext (informational — NVIDIA Vulkan, not RADV).
    b = run_tts(binp, t3, "vk_flash", {**T3GPU, "CRISPASR_CHATTERBOX_FLASH_ATTN": "1"})
    results["runs"]["vk_flash"] = b
    RESULTS.write_text(json.dumps(results, indent=2))

    # C) CPU reference.
    c = run_tts(binp, t3, "cpu_ref", {}, cli_extra=["--no-gpu"], timeout=3000)
    results["runs"]["cpu_ref"] = c

    for tag in ("vk_default", "vk_flash", "cpu_ref"):
        run = results["runs"][tag]
        hyp = asr(binp, whisp, run["wav"])
        run["asr"] = hyp
        run["overlap"] = word_overlap(SENT, hyp)
        kh.step(f"asr.{tag}", text=hyp[:100], overlap=run["overlap"])

    ad, bf, cr = results["runs"]["vk_default"], results["runs"]["vk_flash"], results["runs"]["cpu_ref"]
    verdict = dict(
        policy_naive_on_vulkan=("naive" in ad["attn_line"] and "Vulkan" in ad["attn_line"]),
        policy_flash_optin=("flash_attn_ext" in bf["attn_line"]),
        vk_default_ok=bool(ad["rc"] == 0 and (ad["audio"].get("peak") or 0) > 0.05 and ad["overlap"] >= 0.6),
        vk_flash_outcome=dict(rc=bf["rc"], overlap=bf.get("overlap")),
        cpu_ref_ok=bool(cr["rc"] == 0 and cr["overlap"] >= 0.6),
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
