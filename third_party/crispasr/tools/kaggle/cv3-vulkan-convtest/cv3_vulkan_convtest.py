#!/usr/bin/env python3
"""
#304 — minimal ggml-level repro of the Vulkan conv miscompute on real NVIDIA (P100).

CV3 native-Vulkan was root-caused to a conv miscompute: with identical greedy
tokens the LM (no conv) matches CPU 512/512, but the flow mel drifts
(cosine(cpu,vk)=0.961) and the HiFT (all conv) → noise. The LM has no conv; the
flow (conv_pos_embed) and HiFT (conv stack) do. So the suspect primitive is
ggml_conv_1d = IM2COL + MUL_MAT (MUL_MAT is proven-correct by the LM), i.e.
IM2COL on the Vulkan backend.

This kernel builds ggml's OWN test-backend-ops with GGML_VULKAN=ON on a real
Tesla P100 Vulkan device and runs the conv-family ops (IM2COL, CONV_TRANSPOSE_1D,
CONV_2D, CONV_2D_DW, POOL_2D, plus MUL_MAT as a control) against the CPU
reference, printing pass/FAIL + the error magnitude per op/config. That is the
minimal, app-independent repro for an upstream ggml-org/llama.cpp issue and the
substrate for a fix attempt.

Follows the kaggle_harness regime: git-clone the repo (for harness + the ggml
submodule), init_progress, build_heartbeat around the build, real-NVIDIA Vulkan
ICD (#28), HF token for the mirror.
"""
import os, sys, subprocess, json, re
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp/convtest"); TMP.mkdir(parents=True, exist_ok=True)
RESULTS = WORK / "convtest_results.json"

HERE = Path(__file__).resolve().parent
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
BRANCH = "fix/304-cosyvoice3-se"
CLONE = Path("/kaggle/temp/CrispASR")
_cloned = CLONE.exists()
if not _cloned:
    try:
        # ggml submodule pinned to a SHA that's usually not the branch tip, so a
        # full (non-shallow) submodule clone is required.
        subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                        "--recurse-submodules", CRISPASR_URL, str(CLONE)],
                       check=True, timeout=1200)
        _cloned = True
    except Exception as e:
        print(f"clone failed: {e}", flush=True)
sys.path.insert(0, str(CLONE / "tools" / "kaggle") if _cloned else str(HERE))
import kaggle_harness as kh
kh.init_progress()


def step(name, **extra):
    kh.step(name, **extra)


def sh(cmd, timeout=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)


subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=False)
kh.resolve_hf_token()  # enables the live progress mirror


# ── Vulkan dev + runtime (glslc via LunarG; real NVIDIA ICD, #28) ────────────
def install_vulkan_sdk():
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
    if glslc:
        os.environ["GLSLC_PATH"] = glslc
    step("vulkan.glslc", path=glslc)
    return glslc


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
    if sh("ldconfig -p | grep -F libGLX_nvidia.so.0").stdout.strip() and \
       "NVIDIA" not in sh("vulkaninfo --summary 2>/dev/null").stdout:
        os.makedirs("/usr/share/vulkan/icd.d", exist_ok=True)
        Path("/usr/share/vulkan/icd.d/nvidia_icd.json").write_text(
            '{"file_format_version":"1.0.0","ICD":'
            '{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.277"}}')
    r = sh("vulkaninfo --summary 2>/dev/null")
    devs = [l.split("=")[-1].strip() for l in r.stdout.splitlines() if "deviceName" in l]
    step("vulkan.devices", devices=devs)
    return devs, any("NVIDIA" in d for d in devs)


def build_test_backend_ops():
    kh.install_build_toolchain()
    src = CLONE
    if not (src / "ggml" / "CMakeLists.txt").exists():
        step("build.submodule_init")
        sh(f"cd {src} && git submodule update --init ggml", timeout=900)
    flags = ["-DGGML_VULKAN=ON", "-DGGML_BUILD_TESTS=ON", "-DGGML_METAL=OFF",
             "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF",
             "-DGGML_AVX2=ON", "-DGGML_FMA=ON", "-DGGML_F16C=ON"] + kh.cache_and_link_flags()
    glslc = os.environ.get("GLSLC_PATH")
    if glslc:
        flags.append(f"-DVulkan_GLSLC_EXECUTABLE={glslc}")
    cfg = f"cd {src} && cmake -S ggml -B build-ggml-vk -G Ninja " + " ".join(flags)
    step("build.configure")
    with kh.build_heartbeat("build.cmake", 30):
        r = sh(cfg, timeout=900)
    if r.returncode != 0:
        step("build.configure_FAILED", err=r.stderr[-2000:])
        raise RuntimeError("cmake configure failed")
    step("build.compile")
    with kh.build_heartbeat("build.ninja", 30):
        try:
            kh.sh_with_progress("cmake --build build-ggml-vk --target test-backend-ops", cwd=str(src))
        except Exception as e:
            step("build.compile_FAILED", err=str(e)[-2000:])
            raise
    binp = None
    for c in src.rglob("test-backend-ops"):
        if c.is_file() and os.access(c, os.X_OK):
            binp = c
            break
    if not binp:
        raise RuntimeError("test-backend-ops not produced")
    step("build.ready", path=str(binp))
    return str(binp)


def run_op(binp, op, env):
    # test-backend-ops prints per-config lines then a per-backend summary. Capture
    # everything, extract the Vulkan section pass/FAIL + any error magnitudes.
    with kh.build_heartbeat(f"test.{op}", 30):
        r = subprocess.run(f"{binp} test -o {op}", shell=True, capture_output=True,
                           text=True, env=env, timeout=1200)
    out = r.stdout + "\n" + r.stderr
    (TMP / f"tbo_{op}.log").write_text(out)
    esc = re.compile(r"\x1b\[[0-9;]*m")
    clean = esc.sub("", out)
    lines = clean.splitlines()
    # Isolate the Vulkan backend section: "Backend 1/2: Vulkan0" .. "Backend Vulkan0: OK/FAIL"
    vk_verdict, vk_passed = None, None
    in_vk = False
    fail_cfgs = []
    for l in lines:
        if re.match(r"\s*Backend \d+/\d+:\s*Vulkan", l):
            in_vk = True
        elif re.match(r"\s*Backend \d+/\d+:\s*", l):
            in_vk = False
        if in_vk:
            m = re.search(r"(\d+)/(\d+) tests passed", l)
            if m:
                vk_passed = m.group(0)
            if re.search(rf"{op}\b.*\b(FAIL|not OK|NMSE|nan|inf)\b", l, re.I) or \
               ("compare failed" in l.lower()) or (re.search(r"\bFAIL\b", l) and op in l):
                fail_cfgs.append(l.strip()[:200])
        m2 = re.match(rf"\s*Backend Vulkan\S*:\s*(OK|FAIL)", l)
        if m2:
            vk_verdict = m2.group(1)
            in_vk = False
    return dict(op=op, rc=r.returncode, vk_verdict=vk_verdict, vk_passed=vk_passed,
                n_fail_cfgs=len(fail_cfgs), fail_cfgs=fail_cfgs[:15])


def main():
    devs, nvidia = enable_vulkan()
    binp = build_test_backend_ops()
    # Pin the real NVIDIA device (0) if present; llvmpipe still reproduces the
    # class (#28) but the NVIDIA device is the SubtitleEdit-relevant target.
    env = dict(os.environ)
    env["GGML_VK_VISIBLE_DEVICES"] = "0"

    # list the ops test-backend-ops knows, so we filter to real names
    listing = subprocess.run(f"{binp} test -o NONEXISTENT_OP", shell=True,
                             capture_output=True, text=True, env=env)
    step("ops.probe", tail=(listing.stdout + listing.stderr)[-400:])

    # Flow DiT op set. The LM (correct on Vulkan) uses RMS_NORM; the flow uses
    # NORM (LayerNorm) for adaLN — untested by the LM path, prime suspect. conv
    # ops already confirmed OK; keep IM2COL/MUL_MAT as controls. CONV_2D dropped
    # (aborts, and CV3 uses conv_1d only). Run per-op so one abort doesn't cascade.
    OPS = ["NORM", "RMS_NORM", "GROUP_NORM", "L2_NORM",
           "SOFT_MAX", "GELU", "GELU_QUICK", "SILU", "TANH", "SIGMOID",
           "MUL", "ADD", "SUB", "DIV", "SCALE", "SQR", "SQRT", "SIN", "COS",
           "ROPE", "CONCAT", "GET_ROWS", "CPY", "CONT", "PAD", "IM2COL", "MUL_MAT"]
    results = {"env": {"vulkan_devices": devs, "real_nvidia": nvidia}, "ops": {}}
    for op in OPS:
        try:
            rec = run_op(binp, op, env)
        except Exception as e:
            rec = dict(op=op, error=str(e)[:300])
        results["ops"][op] = rec
        step(f"op.{op}", vk=rec.get("vk_verdict"), passed=rec.get("vk_passed"),
             rc=rec.get("rc"), n_fail=rec.get("n_fail_cfgs"),
             first_fail=(rec.get("fail_cfgs") or [""])[0][:150])
        RESULTS.write_text(json.dumps(results, indent=2))

    step("DONE", verdict={op: f"{r.get('vk_verdict')}({r.get('vk_passed')})"
                          + (f" rc={r.get('rc')}" if r.get('rc') else "")
                          for op, r in results["ops"].items()})
    RESULTS.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        step("FATAL", err=str(e)[:300])
        traceback.print_exc()
        sys.exit(1)
