#!/usr/bin/env python3
"""#461 voxcpm2: VAE decode A/B on real Vulkan (Kaggle T4, NVIDIA ICD).

The reporter's steps=6 log: VAE decode 1688 ms of a 4126 ms synthesis (41%).
Arm B sets CRISPASR_VOXCPM2_VAE_DW_SHIFT=1 (depthwise causal convs as shifted
multiply-adds instead of im2col + K-wide batched mat-mul). Reporter's config:
voxcpm2-q8_0, seed 2, CRISPASR_VOXCPM2_INFERENCE_STEPS=6.

Readouts (each can fail):
  - vae_decode ms per arm, alternated after a warm-up (first run pays shader compile)
  - per-op GPU time of the VAE graph ALONE (last GGML_VK_PERF_LOGGER block)
  - PCM diff A vs B: max|diff| relative to max|A|, rms ratio, sample counts
  - ASR roundtrip (whisper base.en through crispasr) on both WAVs
"""
import hashlib, json, os, re, subprocess, sys, time, traceback, wave
from pathlib import Path
OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
REPO = Path("/tmp/CrispASR"); G = Path("/tmp/g"); G.mkdir(exist_ok=True)
REF = "main"
TEXT = "Hello, this is a short test sentence."
BASE_ENV = {"CRISPASR_VOXCPM2_BENCH": "1"}
SHIFT = {"CRISPASR_VOXCPM2_VAE_DW_SHIFT": "1"}
# sha256[:16] of run 3/4's WAVs (identical in both): c90c.. = depthwise-shift
# VAE, 8954.. = legacy conv_1d_dw VAE. Final gate for the Vulkan default: the
# default arm must be the shift output, CRISPASR_VOXCPM2_VAE_DW_SHIFT=0 the legacy.
SHIFT_SHA, LEGACY_SHA = "c90c145bef2bac22", "89541f440a8b10fe"
REF_SHA = {}
res = {"errors": [], "runs": {}}
def save(): (OUT / "result.json").write_text(json.dumps(res, indent=1))
def sh(c, t=None): return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t)


def read_wav(p):
    import numpy as np
    with wave.open(str(p)) as w:
        n, sw = w.getnframes(), w.getsampwidth()
        raw = w.readframes(n)
    return np.frombuffer(raw, dtype={2: "<i2", 4: "<i4"}[sw]).astype("float64")


try:
    subprocess.check_call(["git", "clone", "--depth", "1", "-b", REF, "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    subprocess.run(["git", "submodule", "update", "--init", "--recursive", "--depth", "1"], cwd=str(REPO))
    res["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    sys.path.insert(0, str(REPO / "tools" / "kaggle")); import kaggle_harness as kh
    tok = kh.resolve_hf_token(require=False)
    if tok: os.environ["HF_TOKEN"] = tok
    from huggingface_hub import hf_hub_download
    # Vulkan SDK (glslc) + NVIDIA ICD - recipe from tools/kaggle/voxcpm2-vk-profile
    sh("apt-get update -qq")
    for pkg in ("libvulkan1", "vulkan-tools", "libvulkan-dev", "glslang-tools", "spirv-tools"):
        sh(f"DEBIAN_FRONTEND=noninteractive apt-get install -y -qq {pkg}")
    glslc = sh("which glslc").stdout.strip()
    if not glslc:
        cn = sh("bash -lc '. /etc/os-release; echo $VERSION_CODENAME'").stdout.strip() or "jammy"
        sh("wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | tee /etc/apt/trusted.gpg.d/lunarg.asc >/dev/null")
        sh(f"wget -qO /etc/apt/sources.list.d/lunarg-vulkan-{cn}.list https://packages.lunarg.com/vulkan/lunarg-vulkan-{cn}.list")
        sh("apt-get update -qq"); sh("DEBIAN_FRONTEND=noninteractive apt-get install -y -qq vulkan-sdk")
        glslc = sh("which glslc").stdout.strip()
    drv = (sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader").stdout.strip().splitlines() or [""])[0]
    if drv: sh(f"DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libnvidia-gl-{drv.split('.')[0]}")
    if "NVIDIA" not in sh("vulkaninfo --summary 2>/dev/null").stdout:
        os.makedirs("/usr/share/vulkan/icd.d", exist_ok=True)
        Path("/usr/share/vulkan/icd.d/nvidia_icd.json").write_text(
            '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.277"}}')
    res["vk_devices"] = [l.split("=")[-1].strip() for l in sh("vulkaninfo --summary 2>/dev/null").stdout.splitlines() if "deviceName" in l]
    save()
    kh.install_build_toolchain()
    flags = ["-DGGML_VULKAN=ON", "-DCMAKE_BUILD_TYPE=Release", "-DCRISPASR_OPUS=OFF", "-DCRISPASR_AMR=OFF"] + kh.cache_and_link_flags()
    if glslc: flags.append(f"-DVulkan_GLSLC_EXECUTABLE={glslc}")
    kh.sh(f"cmake -S {REPO} -B {REPO}/build -G Ninja " + " ".join(flags))
    with kh.build_heartbeat("cmake.build"):
        kh.sh(f"cmake --build {REPO}/build -j$(nproc) --target crispasr-cli crispasr-quantize")
    B = REPO / "build" / "bin" / "crispasr"
    MODELS = {q: hf_hub_download("cstr/voxcpm2-GGUF", f"voxcpm2-{q}.gguf", cache_dir=str(G)) for q in ("q8_0", "f16")}
    # Mixed file: q8_0 everywhere except the LocDiT diffusion head at F16 (run 6:
    # F16 speeds the CFM head 16% but the F16 text model costs more than that).
    Q = REPO / "build" / "bin" / "crispasr-quantize"
    mixed = G / "voxcpm2-q8_0-locdit-f16.gguf"
    qr = subprocess.run([str(Q), MODELS["f16"], str(mixed), "q8_0", "--tensor-type", r"^locdit\.=f16"],
                        capture_output=True, text=True, timeout=3600)
    res["quantize"] = {"rc": qr.returncode, "tail": (qr.stdout + qr.stderr)[-1500:],
                       "size_gb": round(mixed.stat().st_size / 1e9, 2) if mixed.exists() else None}
    save()
    if mixed.exists():
        MODELS["q8_locdit_f16"] = str(mixed)
    model = MODELS["q8_0"]

    def vae_graph_ops(err):
        # GGML_VK_PERF_LOGGER prints one block per graph compute. The VAE decode
        # graph is the one with SIN ops (snake activations; nothing else in
        # voxcpm2 uses SIN). Keep only the last such block.
        blocks = re.split(r"^-+\s*\n(?=Vulkan Timings)", err, flags=re.M)
        vae_blocks = [b for b in blocks[1:] if re.search(r"^SIN\b", b, re.M)]
        blk = vae_blocks[-1] if vae_blocks else ""
        ops = {}
        for m in re.finditer(r"^([A-Z_0-9]+)(?:\([^)]*\))?[^\n]*?:\s*(\d+)\s*x\s*([\d.]+)\s*us", blk, re.M):
            ops[m.group(1)] = ops.get(m.group(1), 0.0) + int(m.group(2)) * float(m.group(3))
        return {"n_blocks": len(blocks) - 1, "total_ms": round(sum(ops.values()) / 1000, 1),
                "ops_ms": {k: round(v / 1000, 1) for k, v in sorted(ops.items(), key=lambda kv: -kv[1])[:20]},
                "raw": [l for l in blk.splitlines() if " us" in l][:60]}

    def run(tag, extra=(), env=None, quant="q8_0"):
        model = MODELS[quant]
        wav = OUT / f"{tag}.wav"
        t0 = time.time()
        r = subprocess.run([str(B), "--backend", "voxcpm2", "-m", model, "--tts", TEXT, "--tts-output", str(wav),
                            "--seed", "2", "-v", *extra], capture_output=True, text=True,
                           env=dict(os.environ, **BASE_ENV, **(env or {})), timeout=3600)
        err = r.stderr
        vae = re.findall(r"voxcpm2_bench: vae_decode\s+([\d.]+) ms", err)
        tot = re.findall(r"voxcpm2_bench: synthesize\s+([\d.]+) ms", err)
        ar = re.findall(r"AR loop (\d+) steps, ([\d.]+) ms", err)
        cfm = re.findall(r"voxcpm2\[bench\]:   cfm\s+([\d.]+) ms", err)
        ent = {"rc": r.returncode, "wall_s": round(time.time() - t0, 2),
               "vae_ms": float(vae[-1]) if vae else None, "synth_ms": float(tot[-1]) if tot else None,
               "ar": ar[-1] if ar else None, "cfm_ms_per_step": float(cfm[-1]) if cfm else None,
               "audio_s": None,
               "vae_split": [l.strip() for l in err.splitlines() if "voxcpm2_bench: vae." in l],
               "fallback": [l for l in err.splitlines() if "falling back" in l or "using CPU" in l][:5],
               "tail": err[-1500:] if r.returncode or not vae else ""}
        if env and "GGML_VK_PERF_LOGGER" in env:
            ent["vae_graph"] = vae_graph_ops(err)
        if wav.exists():
            ent["wav_sha"] = hashlib.sha256(wav.read_bytes()).hexdigest()[:16]
            if tag in REF_SHA:
                ent["matches_run3"] = ent["wav_sha"] == REF_SHA[tag]
        res["runs"][tag] = ent; save()
        print(tag, {k: ent[k] for k in ("rc", "vae_ms", "synth_ms", "ar", "fallback", "vae_split")}, flush=True)
        return wav

    # #461 round 3: CFM is 79% at the default 10 steps on the reporter's Arc
    # (RTF 1.28; 6 steps = RTF 0.90 but audible artifacts). q4_K didn't help
    # there. Arms: weight type (q8_0 vs f16: coop-matrix kernels favour F16)
    # x steps (10 default, 8). Warm-up per model pays shader compile.
    for q in ("q8_0", "q8_locdit_f16"):
        if q not in MODELS:
            continue
        run(f"warm_{q}", quant=q)
        for st in ("10", "8"):
            for rep in (1, 2):
                run(f"{q}_s{st}_{rep}", env={"CRISPASR_VOXCPM2_INFERENCE_STEPS": st}, quant=q)
    res["gate_ok"] = True

    for tag in ("q8_0_s10", "q8_0_s8", "q8_locdit_f16_s10", "q8_locdit_f16_s8"):
        v = [res["runs"][k]["vae_ms"] for k in res["runs"] if k.startswith(tag) and res["runs"][k]["vae_ms"]]
        res[f"{tag}_vae_ms_all"] = v
    try:
        import numpy as np
        for a, b in (("q8_0_s10_1", "q8_locdit_f16_s10_1"),):
            A, Bw = read_wav(OUT / f"{a}.wav"), read_wav(OUT / f"{b}.wav")
            n = min(len(A), len(Bw))
            d = np.abs(A[:n] - Bw[:n])
            res[f"pcm_{a}_vs_{b}"] = {"len": [len(A), len(Bw)], "max_abs_A": float(np.abs(A).max()),
                                      "max_diff": float(d.max()), "rel": float(d.max() / max(np.abs(A).max(), 1)),
                                      "rms_ratio": float(np.sqrt((Bw[:n] ** 2).mean() / max((A[:n] ** 2).mean(), 1e-12)))}
    except Exception:
        res["errors"].append("pcm: " + traceback.format_exc())
    save()
    # ASR roundtrip: whisper base.en through the same binary
    try:
        wm = hf_hub_download("ggerganov/whisper.cpp", "ggml-base.en.bin", cache_dir=str(G))
        for tag in ("q8_0_s10_1", "q8_0_s8_1", "q8_locdit_f16_s10_1", "q8_locdit_f16_s8_1"):
            r = subprocess.run([str(B), "-m", wm, "-f", str(OUT / f"{tag}.wav"), "-np", "-nt"],
                               capture_output=True, text=True, timeout=600)
            res.setdefault("asr", {})[tag] = r.stdout.strip() or r.stderr[-300:]
    except Exception:
        res["errors"].append("asr: " + traceback.format_exc())
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
    import shutil; shutil.rmtree(REPO, ignore_errors=True)
