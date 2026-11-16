#!/usr/bin/env python3
"""
#304 — isolate the flow vs HiFT Vulkan miscompute for CosyVoice3 on REAL NVIDIA Vulkan.

Background: the shipped fix routes CV3 to CPU under Vulkan (SubtitleEdit ships the
Vulkan Windows build to every user). The native-Vulkan path was garbage. On MoltenVK
(M1) I traced it to two independent things:
  1. AR-decode collapse = the ggml_backend_sched weight-less-first-op miscompute
     (LM graph starts with rms_norm(inputs_embeds) on a leaf). FIXED by dispatching
     every CV3 graph through a single-backend gallocr on the GPU backend, bypassing
     the scheduler (branch fix/304-cosyvoice3-se, gated CRISPASR_COSYVOICE3_VULKAN_NATIVE=1).
  2. Even with valid LM tokens the audio was near-silent noise. The pre-HiFT mel is
     HEALTHY on MoltenVK (range ~-12..0.5, no NaN) -> flow works, HiFT is the breaker.

MoltenVK != the Windows/NVIDIA Vulkan SubtitleEdit ships, so this kernel CONFIRMS the
split on a real NVIDIA driver. It builds crispasr from the branch with GGML_VULKAN=ON,
then synthesizes ONE fixed sentence (same seed) under:
   A) --no-gpu                                   (CPU reference, known-good)
   B) CRISPASR_COSYVOICE3_VULKAN_NATIVE=1        (native Vulkan, gallocr LM path)
For each it dumps generated tokens, the pre-HiFT mel (CRISPASR_COSYVOICE3_DUMP_MEL
stats + raw), and the audio, then ASR-roundtrips with whisper-tiny.en.

Verdict emitted to /kaggle/working/cv3_isolate_results.json:
   LM   : Vulkan tokens track CPU for the first-divergence index (prefix match ok).
   FLOW : Vulkan pre-HiFT mel healthy + same range as CPU (no NaN, ~-12..0.5).
   HIFT : healthy mel but garbage audio on Vulkan  => HiFT is the Vulkan breaker.

Follows the kaggle_harness regime: git-clones the repo for the harness, init_progress,
build_heartbeat around the (long) source build + each synth, HF token for model pulls,
real-NVIDIA Vulkan ICD (falls back to llvmpipe which reproduces the same class).
"""
import os, sys, subprocess, json, time, wave, math, array, re, struct
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp/cv3iso"); TMP.mkdir(parents=True, exist_ok=True)
MODELS = Path("/kaggle/temp/models"); MODELS.mkdir(parents=True, exist_ok=True)
RESULTS = WORK / "cv3_isolate_results.json"

# ── kaggle_harness regime: clone repo, import harness, init progress ─────────
HERE = Path(__file__).resolve().parent
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
BRANCH = "fix/304-cosyvoice3-se"
CLONE = Path("/kaggle/temp/CrispASR")
_cloned = CLONE.exists()
if not _cloned:
    try:
        # Shallow on the branch, but FULL submodule clone: ggml is pinned to a
        # specific SHA that is usually NOT the submodule branch tip, so
        # --shallow-submodules (tip-only fetch) fails to find it. A full
        # submodule fetch is small enough and reliable.
        subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                        "--recurse-submodules", CRISPASR_URL, str(CLONE)],
                       check=True, timeout=1200)
        _cloned = True
    except Exception as e:
        print(f"clone failed: {e}", flush=True)
_harness_dir = str(CLONE / "tools" / "kaggle") if _cloned else str(HERE)
sys.path.insert(0, _harness_dir)
import kaggle_harness as kh
kh.init_progress()


def step(name, **extra):
    kh.step(name, **extra)


def sh(cmd, timeout=None, check=False):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          timeout=timeout, check=check)


# HF auth (env -> Kaggle Secret -> mounted crispasr-hf-token dataset)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "hf_transfer", "huggingface_hub"], check=False)
HF_TOKEN = kh.resolve_hf_token()
step("hf.token", present=bool(HF_TOKEN))
from huggingface_hub import hf_hub_download

SENT = "The quick brown fox jumps over the lazy dog."
SEED = 42
VOICE = "fleurs-en"
CV3_REPO = "cstr/cosyvoice3-0.5b-2512-GGUF"
# Companion GGUFs the CV3 backend resolves as siblings of the LLM.
CV3_FILES = [
    "cosyvoice3-llm-q4_k.gguf",
    "cosyvoice3-flow-q8_0.gguf",
    "cosyvoice3-hift-f16.gguf",
    "cosyvoice3-s3tok-f16.gguf",
    "cosyvoice3-campplus-f16.gguf",
    "cosyvoice3-voices.gguf",
]
WHISPER_REPO = "ggerganov/whisper.cpp"
WHISPER_FILE = "ggml-tiny.en.bin"


def install_vulkan_sdk():
    """Install the Vulkan runtime + DEV toolchain needed to BUILD ggml-vulkan.
    ggml/src/ggml-vulkan requires find_package(Vulkan COMPONENTS glslc) — glslc
    is NOT a standalone Ubuntu package; it ships only in the LunarG Vulkan SDK.
    Install each apt package individually (one bad name must not abort the rest),
    then add the LunarG repo for glslc if it's still missing."""
    for pkg in ("libvulkan1", "vulkan-tools", "libvulkan-dev", "glslang-tools", "spirv-tools"):
        sh(f"DEBIAN_FRONTEND=noninteractive apt-get install -y -qq {pkg}")
    glslc = sh("which glslc").stdout.strip()
    if not glslc:
        codename = sh("bash -lc '. /etc/os-release; echo $VERSION_CODENAME'").stdout.strip() or "jammy"
        step("vulkan.lunarg_repo", codename=codename)
        sh("wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc "
           "| tee /etc/apt/trusted.gpg.d/lunarg.asc >/dev/null")
        sh(f"wget -qO /etc/apt/sources.list.d/lunarg-vulkan-{codename}.list "
           f"https://packages.lunarg.com/vulkan/lunarg-vulkan-{codename}.list")
        sh("apt-get update -qq")
        # vulkan-sdk pulls glslc + headers + loader dev; fall back to the
        # narrower shaderc pkg name if the meta-package is unavailable.
        sh("DEBIAN_FRONTEND=noninteractive apt-get install -y -qq vulkan-sdk")
        glslc = sh("which glslc").stdout.strip()
        if not glslc:
            sh("DEBIAN_FRONTEND=noninteractive apt-get install -y -qq shaderc glslc")
            glslc = sh("which glslc").stdout.strip()
    step("vulkan.glslc", path=glslc,
         header=bool(sh("ls /usr/include/vulkan/vulkan.h 2>/dev/null").stdout.strip()
                     or sh("find /usr -name vulkan.h 2>/dev/null | head -1").stdout.strip()))
    if glslc:
        os.environ["GLSLC_PATH"] = glslc
    return glslc


# ── 1. Vulkan enablement: prefer real NVIDIA, fall back to llvmpipe ──────────
def enable_vulkan():
    step("vulkan.install")
    sh("apt-get update -qq")
    install_vulkan_sdk()
    smi = sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
    drv = (smi.stdout.strip().splitlines() or [""])[0]
    major = drv.split(".")[0] if drv else ""
    step("vulkan.nvidia_driver", driver=drv)
    if major:
        sh(f"DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libnvidia-gl-{major}")
    have_glx = bool(sh("ldconfig -p | grep -F libGLX_nvidia.so.0").stdout.strip())
    step("vulkan.libGLX_nvidia_present", present=have_glx)
    if have_glx and "NVIDIA" not in sh("vulkaninfo --summary 2>/dev/null").stdout:
        os.makedirs("/usr/share/vulkan/icd.d", exist_ok=True)
        icd = ('{"file_format_version":"1.0.0","ICD":'
               '{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.277"}}')
        Path("/usr/share/vulkan/icd.d/nvidia_icd.json").write_text(icd)
    r = sh("vulkaninfo --summary 2>/dev/null")
    devs = [l.split("=")[-1].strip() for l in r.stdout.splitlines() if "deviceName" in l]
    step("vulkan.devices", devices=devs)
    nvidia = any("NVIDIA" in d for d in devs)
    step("vulkan.have_glslc", present=bool(sh("which glslc").stdout.strip()))
    return devs, nvidia


# ── 2. Build crispasr from the branch with Vulkan ───────────────────────────
def build_crispasr():
    kh.install_build_toolchain()
    src = CLONE
    if not (src / "CMakeLists.txt").exists():
        raise RuntimeError("repo clone missing — cannot build")
    # ggml submodule must be present (needed for GGML_VULKAN shaders).
    if not (src / "ggml" / "CMakeLists.txt").exists():
        step("build.submodule_init")
        sh(f"cd {src} && git submodule update --init ggml", timeout=900)
    bdir = src / "build-vk"
    flags = ["-DCMAKE_BUILD_TYPE=Release", "-DGGML_VULKAN=ON", "-DGGML_CUDA=OFF",
             "-DGGML_NATIVE=OFF", "-DGGML_AVX2=ON", "-DGGML_FMA=ON", "-DGGML_F16C=ON",
             "-DCRISPASR_OPUS_FETCH=ON"] + kh.cache_and_link_flags()
    glslc = os.environ.get("GLSLC_PATH")
    if glslc:
        flags.append(f"-DVulkan_GLSLC_EXECUTABLE={glslc}")
    else:
        step("build.WARN_no_glslc")  # cmake FindVulkan will very likely fail
    cfg = f"cd {src} && cmake -G Ninja -B build-vk " + " ".join(flags)
    step("build.configure", flags=flags)
    with kh.build_heartbeat("build.cmake", interval_s=30):
        r = sh(cfg, timeout=1200)
    if r.returncode != 0:
        step("build.configure_FAILED", stderr=r.stderr[-2000:], stdout=r.stdout[-1000:])
        raise RuntimeError("cmake configure failed")
    jobs = kh.safe_build_jobs(gpu=False)
    step("build.compile", jobs=jobs)
    with kh.build_heartbeat("build.ninja", interval_s=30):
        try:
            kh.sh_with_progress(f"cmake --build build-vk -j{jobs} --target crispasr", cwd=str(src))
        except Exception as e:
            step("build.compile_FAILED", err=str(e)[-2000:])
            raise
    binp = src / "build-vk" / "bin" / "crispasr"
    if not binp.exists():
        raise RuntimeError("crispasr binary not produced")
    os.environ["LD_LIBRARY_PATH"] = str(binp.parent) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    ver = sh(f"{binp} --version")
    step("build.ready", path=str(binp),
         backends=[l.split(":")[-1].strip() for l in ver.stdout.splitlines() if "backends" in l])
    try:
        kh.export_ccache_tar()
    except Exception:
        pass
    return str(binp)


# ── 3. models ────────────────────────────────────────────────────────────────
def fetch_models():
    step("models.cv3", repo=CV3_REPO)
    for f in CV3_FILES:
        with kh.build_heartbeat(f"models.{f}", interval_s=30):
            hf_hub_download(repo_id=CV3_REPO, filename=f, local_dir=str(MODELS), token=HF_TOKEN or None)
    step("models.whisper")
    whisp = hf_hub_download(repo_id=WHISPER_REPO, filename=WHISPER_FILE,
                            local_dir=str(MODELS), token=HF_TOKEN or None)
    return str(MODELS / CV3_FILES[0]), whisp


# ── 4. analysis helpers ──────────────────────────────────────────────────────
def wav_stats(path):
    try:
        w = wave.open(path, "rb")
        a = array.array("h"); a.frombytes(w.readframes(w.getnframes()))
        if not len(a):
            return dict(dur=0.0, peak=0.0, rms=0.0)
        peak = max(abs(x) for x in a) / 32768.0
        rms = math.sqrt(sum((x / 32768.0) ** 2 for x in a) / len(a))
        return dict(dur=round(w.getnframes() / w.getframerate(), 2),
                    peak=round(peak, 4), rms=round(rms, 4))
    except Exception as e:
        return dict(err=str(e))


def parse_mel_stats(stderr):
    # "cosyvoice3_tts: MEL[142x80] min=-12.4759 max=0.4800 mean=-4.9292 rms=5.3377 nan/inf=0"
    m = re.search(r"MEL\[(\d+)x(\d+)\] min=([-\d.]+) max=([-\d.]+) mean=([-\d.]+) rms=([-\d.]+) nan/inf=(\d+)", stderr)
    if not m:
        return None
    return dict(T=int(m.group(1)), mel=int(m.group(2)), min=float(m.group(3)),
                max=float(m.group(4)), mean=float(m.group(5)), rms=float(m.group(6)),
                nan=int(m.group(7)))


def parse_tokens(dump_path):
    try:
        gen = []
        for line in Path(dump_path).read_text().splitlines():
            if line.startswith("generated_tokens"):
                gen = [int(x) for x in line.split(":", 1)[1].split()]
        return gen
    except Exception:
        return []


def asr(binp, whisp, wav):
    if not Path(wav).exists():
        return ""
    r = sh(f"{binp} -m {whisp} -f {wav} --language en --no-gpu", timeout=300)
    txt = " ".join(re.sub(r"\[[^\]]*\]", "", l) for l in r.stdout.splitlines()
                   if l.strip().startswith("["))
    return re.sub(r"\s+", " ", txt).strip()


def norm_wav(src, dst, target_peak=0.85):
    """Normalize so a quiet-but-valid clip is fairly ASR'd."""
    try:
        w = wave.open(src, "rb"); n = w.getnframes(); sr = w.getframerate()
        a = array.array("h"); a.frombytes(w.readframes(n)); w.close()
        peak = max(abs(x) for x in a) / 32768.0 if a else 0
        if peak <= 0:
            return src
        g = target_peak / peak
        out = array.array("h", [max(-32768, min(32767, int(x * g))) for x in a])
        o = wave.open(dst, "wb"); o.setnchannels(1); o.setsampwidth(2)
        o.setframerate(sr); o.writeframes(out.tobytes()); o.close()
        return dst
    except Exception:
        return src


def run_synth(binp, llm, tag, env_extra, cli_extra=None, timeout=1800):
    out = TMP / f"cv3_{tag}.wav"
    mel = TMP / f"mel_{tag}.bin"
    tok = TMP / f"tok_{tag}.txt"
    env = dict(os.environ)
    env["CRISPASR_COSYVOICE3_DUMP_MEL"] = str(mel)
    env["CRISPASR_COSYVOICE3_DUMP_TOKENS"] = str(tok)
    env.update(env_extra)
    cmd = [binp, "--backend", "cosyvoice3-tts", "-m", llm, "--tts", SENT,
           "--voice", VOICE, "--seed", str(SEED), "--tts-output", str(out)]
    if cli_extra:
        cmd += cli_extra
    stderr, rc, timed_out = "", -1, False
    with kh.build_heartbeat(f"synth.{tag}", interval_s=30):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
            stderr, rc = r.stderr, r.returncode
        except subprocess.TimeoutExpired as e:
            timed_out = True
            stderr = (e.stderr or b"").decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
    return dict(
        tag=tag, rc=rc, timed_out=timed_out,
        mel=parse_mel_stats(stderr),
        n_tokens=len(parse_tokens(tok)),
        tokens=parse_tokens(tok)[:16],
        audio=wav_stats(str(out)),
        stderr_tail=stderr[-800:] if rc != 0 else "",
        wav=str(out),
    )


def main():
    devs, nvidia = enable_vulkan()
    binp = build_crispasr()
    llm, whisp = fetch_models()

    results = {"env": {"vulkan_devices": devs, "real_nvidia": nvidia}, "runs": {}}

    # B) native Vulkan (gallocr LM path) — the ESSENTIAL run, GPU-fast. Runs
    # FIRST so a slow CPU reference can't starve it of session time.
    vk = run_synth(binp, llm, "vk", {"CRISPASR_COSYVOICE3_VULKAN_NATIVE": "1"}, timeout=1200)
    step("run.vk", **{k: vk[k] for k in ("rc", "timed_out", "mel", "n_tokens", "audio")})
    results["runs"]["vk"] = vk
    RESULTS.write_text(json.dumps(results, indent=2))  # persist before the slow CPU ref

    # A) CPU reference (known-good) — explicit --no-gpu. Confirmatory only (I
    # already have a Metal reference locally); the full CV3 pipeline on Kaggle's
    # CPU is very slow, so it's best-effort with a generous, non-fatal timeout.
    cpu = run_synth(binp, llm, "cpu", {}, cli_extra=["--no-gpu"], timeout=3000)
    step("run.cpu", **{k: cpu[k] for k in ("rc", "timed_out", "mel", "n_tokens", "audio")})
    results["runs"]["cpu"] = cpu

    # ASR both (+ normalized for quiet clips)
    for tag in ("cpu", "vk"):
        run = results["runs"][tag]
        raw = asr(binp, whisp, run["wav"])
        nrm = asr(binp, whisp, norm_wav(run["wav"], str(TMP / f"cv3_{tag}_norm.wav")))
        run["asr_raw"] = raw
        run["asr_norm"] = nrm
        step(f"asr.{tag}", raw=raw[:80], norm=nrm[:80])

    # ── verdict ──
    def healthy_mel(m):
        return bool(m) and m["nan"] == 0 and -20 < m["min"] < 5 and -5 < m["max"] < 10 and m["rms"] > 0.5

    cm, vm = cpu.get("mel"), vk.get("mel")
    ct, vt = parse_tokens(TMP / "tok_cpu.txt"), parse_tokens(TMP / "tok_vk.txt")
    k = 0
    while k < min(len(ct), len(vt)) and ct[k] == vt[k]:
        k += 1
    vk_peak = vk["audio"].get("peak", 0) or 0
    cpu_peak = cpu["audio"].get("peak", 0) or 0
    # HiFT is the breaker iff the Vulkan flow mel is HEALTHY yet the Vulkan audio
    # is garbage (near-silent) — a healthy mel through a correct vocoder yields
    # ~0.8-peak speech, so noise can only be the vocoder. The CPU reference is a
    # bonus confirmation (it may time out on Kaggle's slow CPU).
    verdict = dict(
        lm_first_divergence=k, lm_cpu_tokens=len(ct), lm_vk_tokens=len(vt),
        flow_mel_cpu_healthy=healthy_mel(cm), flow_mel_vk_healthy=healthy_mel(vm),
        vk_audio_peak=round(vk_peak, 4), cpu_audio_peak=round(cpu_peak, 4),
        cpu_timed_out=cpu.get("timed_out", False),
        hift_is_breaker=bool(healthy_mel(vm) and vk_peak < 0.2),
        cpu_confirms=bool(cpu_peak > 0.3),
    )
    results["verdict"] = verdict
    step("verdict", **verdict)
    RESULTS.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)
    step("DONE", verdict=verdict)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        step("FATAL", err=str(e)[:300])
        traceback.print_exc()
        sys.exit(1)
