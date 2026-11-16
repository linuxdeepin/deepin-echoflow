#!/usr/bin/env python3
"""
#304 follow-up — Vulkan audit of every CrispASR TTS backend SubtitleEdit ships.

SubtitleEdit downloads the *Vulkan* Windows build for every Windows user and
drives these backends through the crispasr /v1/audio/speech server. cosyvoice3
was confirmed to emit blank/garbled audio on Vulkan (AR decode collapses; conv
vocoder corrupts) and fixed by routing to CPU under Vulkan. This kernel checks
whether the OTHER SE-exposed TTS backends have the same bug on a real Vulkan
driver.

Method per backend: synthesize one sentence under --gpu-backend vulkan and under
--no-gpu (CPU baseline), ASR-roundtrip both with whisper-tiny.en, compare. A
backend is FLAGGED VULKAN_BROKEN when CPU is intelligible but Vulkan collapses.

Uses the prebuilt crispasr-linux-x86_64-vulkan.tar.gz (v0.8.22) — so cosyvoice3
here is UNFIXED and is the positive control (must reproduce the bug). All model
repos are public (cstr/*). Follows the kaggle_harness regime: git-clones the
repo for the harness + samples/jfk.wav, init_progress, heartbeat around every
long op.
"""
import os, sys, subprocess, json, time, wave, math, array, re, urllib.request
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/tmp/ttsaudit"); TMP.mkdir(parents=True, exist_ok=True)
MODELS = Path("/tmp/models"); MODELS.mkdir(parents=True, exist_ok=True)
RESULTS = WORK / "audit_results.json"

# ── kaggle_harness regime: clone repo, import harness, init progress ─────────
# (clone gives us the harness + samples/jfk.wav; bundled copies are the fallback)
HERE = Path(__file__).resolve().parent
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
CLONE = Path("/kaggle/temp/CrispASR")
_cloned = CLONE.exists()
if not _cloned:
    try:
        subprocess.run(["git", "clone", "--depth", "1", CRISPASR_URL, str(CLONE)],
                       check=True, timeout=300)
        _cloned = True
    except Exception as e:
        print(f"clone failed, using bundled fallback: {e}", flush=True)
_harness_dir = str(CLONE / "tools" / "kaggle") if _cloned else str(HERE)
sys.path.insert(0, _harness_dir)
import kaggle_harness as kh
kh.init_progress()

def step(name, **extra):
    kh.step(name, **extra)

# HF auth via the harness (env → Kaggle Secret → mounted crispasr-hf-token
# dataset); exports HF_TOKEN + HUGGING_FACE_HUB_TOKEN + HF_HUB_ENABLE_HF_TRANSFER.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hf_transfer", "huggingface_hub"], check=False)
HF_TOKEN = kh.resolve_hf_token()
step("hf.token", present=bool(HF_TOKEN))
from huggingface_hub import hf_hub_download

RELEASE = "v0.8.22"
VK_TARBALL = f"https://github.com/CrispStrobe/CrispASR/releases/download/{RELEASE}/crispasr-linux-x86_64-vulkan.tar.gz"
WHISPER_TINY = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin"
SENT = "The quick brown fox jumps over the lazy dog near the river."
JFK_TEXT = ("And so my fellow Americans, ask not what your country can do for you, "
            "ask what you can do for your country.")
REF_WAV = str(CLONE / "samples" / "jfk.wav") if _cloned else str(HERE / "jfk.wav")

def sh(cmd, timeout=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)

# ── 1. Vulkan enablement: prefer real NVIDIA, fall back to llvmpipe ──────────
def enable_vulkan():
    step("vulkan.install")
    sh("apt-get update -qq")
    sh("DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
       "libvulkan1 vulkan-tools espeak-ng libespeak-ng1")
    # driver major -> matching NVIDIA GL/Vulkan ICD package
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
        api = "1.3.277"
        icd = ('{"file_format_version":"1.0.0","ICD":'
               f'{{"library_path":"libGLX_nvidia.so.0","api_version":"{api}"}}}}')
        Path("/usr/share/vulkan/icd.d/nvidia_icd.json").write_text(icd)
    r = sh("vulkaninfo --summary 2>/dev/null")
    devs = [l.split("=")[-1].strip() for l in r.stdout.splitlines() if "deviceName" in l]
    step("vulkan.devices", devices=devs)
    nvidia = any("NVIDIA" in d or ("llvmpipe" not in d.lower() and "lavapipe" not in d.lower()) for d in devs)
    return devs, nvidia

def fetch_binaries():
    step("fetch.crispasr_vulkan", release=RELEASE)
    tb = TMP / "crispasr-vulkan.tar.gz"
    urllib.request.urlretrieve(VK_TARBALL, tb)
    sh(f"cd {TMP} && tar xzf {tb}")
    bins = [b for b in TMP.rglob("crispasr") if b.is_file()]
    if not bins:
        raise RuntimeError("crispasr binary not found")
    binp = bins[0]; sh(f"chmod +x {binp}")
    bindir = str(binp.parent)
    os.environ["LD_LIBRARY_PATH"] = bindir + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    ver = sh(f"{binp} --version")
    step("fetch.crispasr_ready", path=str(binp),
         backends=[l.split(":")[-1].strip() for l in ver.stdout.splitlines() if "backends" in l])
    whisp = MODELS / "ggml-tiny.en.bin"
    if not whisp.exists():
        step("fetch.whisper")
        urllib.request.urlretrieve(WHISPER_TINY, whisp)
    return str(binp), str(whisp)

def hf_get(repo, fname, dest_dir):
    # authenticated + resumable + hf_transfer (fast). Falls back to urllib if
    # the hub call fails (public repos still work).
    try:
        return hf_hub_download(repo_id=repo, filename=fname, local_dir=dest_dir,
                               token=HF_TOKEN or None)
    except Exception as e:
        step("hf_get.fallback_urllib", repo=repo, file=fname, err=str(e)[:120])
        dest = Path(dest_dir) / fname
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"https://huggingface.co/{repo}/resolve/main/{fname}", dest)
        return str(dest)

def wav_stats(path):
    try:
        w = wave.open(path, "rb")
        a = array.array("h"); a.frombytes(w.readframes(w.getnframes()))
        if not len(a):
            return dict(dur=0.0, peak=0.0, rms=0.0)
        peak = max(abs(x) for x in a) / 32768.0
        rms = math.sqrt(sum((x / 32768.0) ** 2 for x in a) / len(a))
        return dict(dur=round(w.getnframes() / w.getframerate(), 2), peak=round(peak, 4), rms=round(rms, 4))
    except Exception as e:
        return dict(dur=0.0, peak=0.0, rms=0.0, err=str(e))

def asr(binp, whisp, wav):
    if not Path(wav).exists():
        return ""
    r = sh(f"{binp} -m {whisp} -f {wav} --language en --no-gpu", timeout=300)
    txt = " ".join(re.sub(r"\[[^\]]*\]", "", l) for l in r.stdout.splitlines() if l.strip().startswith("["))
    return re.sub(r"\s+", " ", txt).strip()

def words(s):
    return set(re.findall(r"[a-z']+", s.lower()))

def overlap(a, b):
    wa, wb = words(a), words(b)
    return round(len(wa & wb) / len(wa), 3) if wa else 0.0

# Short 3 s reference — the 11 s jfk.wav makes every clone-synth enormous
# (huge DiT/AR sequence) and burns GPU quota (gotcha #1). Trimmed in main().
import audioop
REF3 = str(TMP / "ref3s.wav")   # 3 s, mono, 24 kHz — qwen3-tts (and others) reject 16 kHz
JFK3_TEXT = "And so my fellow Americans, ask not what your country can do for you"
REF_RATE = 24000

def make_ref3():
    w = wave.open(REF_WAV, "rb")
    sr, sw, ch = w.getframerate(), w.getsampwidth(), w.getnchannels()
    data = w.readframes(min(w.getnframes(), int(3.0 * sr)))
    if ch == 2:
        data = audioop.tomono(data, sw, 0.5, 0.5)
    if sr != REF_RATE:
        data, _ = audioop.ratecv(data, sw, 1, sr, REF_RATE, None)
    o = wave.open(REF3, "wb")
    o.setnchannels(1); o.setsampwidth(sw); o.setframerate(REF_RATE)
    o.writeframes(data); o.close()

CLONE_ARGS = ["--voice", "{ref}", "--ref-text", JFK3_TEXT, "--i-have-rights"]
CLONE_NOTEXT = ["--voice", "{ref}", "--i-have-rights"]

# Ordered fast → slow so the quick backends report before the slow/huge ones.
# f5 (heavy DiT on CPU) and moss (10.5 GB download) run LAST.
BACKENDS = [
    dict(name="cosyvoice3-tts", repo="cstr/cosyvoice3-0.5b-2512-GGUF",
         main="cosyvoice3-llm-q4_k.gguf",
         files=["cosyvoice3-llm-q4_k.gguf", "cosyvoice3-flow-q8_0.gguf",
                "cosyvoice3-hift-f16.gguf", "cosyvoice3-voices.gguf"],
         args=["--voice", "zero_shot"], control=True),
    dict(name="qwen3-tts", repo="cstr/qwen3-tts-0.6b-base-GGUF",
         main="qwen3-tts-12hz-0.6b-base-q8_0.gguf",
         files=["qwen3-tts-12hz-0.6b-base-q8_0.gguf"],
         extra=[("cstr/qwen3-tts-tokenizer-12hz-GGUF", ["qwen3-tts-tokenizer-12hz.gguf"])],
         args=CLONE_ARGS),
    dict(name="vibevoice-1.5b", repo="cstr/vibevoice-1.5b-GGUF",
         main="vibevoice-1.5b-tts-q8_0.gguf",
         files=["vibevoice-1.5b-tts-q8_0.gguf"], args=CLONE_NOTEXT),
    dict(name="voxcpm2-tts", repo="cstr/voxcpm2-GGUF",
         main="voxcpm2-q4_k.gguf",
         files=["voxcpm2-q4_k.gguf", "voxcpm2-ref.gguf"], args=CLONE_ARGS),
    dict(name="zonos", repo="cstr/zonos-v0.1-transformer-GGUF",
         main="zonos-v0.1-transformer-q8_0.gguf",
         files=["zonos-v0.1-transformer-q8_0.gguf"],
         extra=[("cstr/dac-44khz-GGUF", ["dac-44khz-f16.gguf"])],
         args=CLONE_NOTEXT),
    dict(name="indextts", repo="cstr/indextts-1.5-GGUF",
         main="indextts-gpt-q8_0.gguf",
         files=["indextts-gpt-q8_0.gguf", "indextts-bigvgan.gguf"], args=CLONE_NOTEXT),
    dict(name="f5-tts", repo="cstr/f5-tts-GGUF",
         main="f5-tts-v1-base-f16.gguf", files=["f5-tts-v1-base-f16.gguf"],
         args=CLONE_ARGS + ["--tts-steps", "8"], timeout=700),  # 8 ODE steps to fit the cap
    dict(name="moss-tts", repo="cstr/moss-tts-v1.5-GGUF",
         main="moss-tts-v1.5-q4_k.gguf",
         files=["moss-tts-v1.5-q4_k.gguf", "moss-tts-v1.5-codec.gguf"],
         args=["--codec-model", "{dir}/moss-tts-v1.5-codec.gguf"] + CLONE_ARGS, timeout=500),
]

def run_synth(binp, cfg, mdir, mode):
    out = str(TMP / f"{cfg['name']}.{mode}.wav")
    if os.path.exists(out):
        os.remove(out)
    args = [a.replace("{dir}", mdir).replace("{ref}", REF3) for a in cfg["args"]]
    gpuflag = ["--gpu-backend", "vulkan"] if mode == "vulkan" else ["--no-gpu"]
    cmd = ([binp, "-m", f"{mdir}/{cfg['main']}", "--backend", cfg["name"]]
           + gpuflag + args + ["--tts", SENT, "--tts-output", out])
    e = dict(os.environ)
    if mode == "vulkan":
        e["GGML_VK_VISIBLE_DEVICES"] = "0"
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg.get("timeout", 300), env=e)
        rc, err = r.returncode, r.stderr or ""
    except subprocess.TimeoutExpired:
        return out, "timeout", "", ""
    m = re.search(r"generated (\d+) (?:speech )?tokens", err)
    return out, rc, (m.group(1) if m else ""), "\n".join(err.splitlines()[-6:])

def audit_backend(binp, whisp, cfg):
    name = cfg["name"]
    step(f"backend.start", backend=name)
    mdir = str(MODELS / name); os.makedirs(mdir, exist_ok=True)
    rec = dict(backend=name, control=cfg.get("control", False))
    try:
        with kh.build_heartbeat(f"{name}.download", interval_s=30):
            for f in cfg["files"]:
                hf_get(cfg["repo"], f, mdir)
            for erepo, efiles in cfg.get("extra", []):
                for f in efiles:
                    hf_get(erepo, f, mdir)
    except Exception as e:
        rec["error"] = f"download failed: {e}"
        step("backend.download_failed", backend=name, err=str(e))
        return rec
    def one(mode):
        with kh.build_heartbeat(f"{name}.synth.{mode}", interval_s=30):
            out, rc, ntok, tail = run_synth(binp, cfg, mdir, mode)
        st = wav_stats(out) if os.path.exists(out) else dict(dur=0, peak=0, rms=0)
        tx = asr(binp, whisp, out) if os.path.exists(out) else ""
        ov = overlap(SENT, tx)
        rec[mode] = dict(rc=rc, tokens=ntok, dur=st.get("dur"), peak=st.get("peak"),
                         rms=st.get("rms"), asr=tx, overlap_sent=ov)
        step(f"backend.{mode}", backend=name, rc=rc, tokens=ntok, dur=st.get("dur"),
             peak=st.get("peak"), overlap=ov, asr=tx[:60])
        if rc not in (0,):
            step(f"backend.{mode}.stderr", backend=name, tail=tail)
        return isinstance(ov, float) and ov >= 0.5 and (st.get("peak") or 0) > 0.02
    # Vulkan first (the thing under test). If Vulkan already works, skip the
    # CPU baseline entirely — saves ~half the GPU-quota (gotcha #1). Only run
    # the CPU baseline when Vulkan looks broken, to confirm it isn't a bad
    # invocation.
    vk_ok = one("vulkan")
    cpu_ok = True if vk_ok else one("cpu")
    rec["verdict"] = ("VULKAN_BROKEN" if cpu_ok and not vk_ok else
                      "vulkan_ok" if cpu_ok and vk_ok else
                      "cpu_baseline_failed(inconclusive)" if not cpu_ok else "unclear")
    step("backend.verdict", backend=name, verdict=rec["verdict"])
    sh(f"rm -rf {mdir}")  # free disk between backends
    return rec

def main():
    devs, nvidia = enable_vulkan()
    binp, whisp = fetch_binaries()
    make_ref3()
    step("ref.trimmed", path=REF3, stats=wav_stats(REF3))
    results = dict(release=RELEASE, sentence=SENT, vulkan_devices=devs,
                   vulkan_is_nvidia=nvidia, ref_seconds=3, backends=[])
    for cfg in BACKENDS:
        try:
            results["backends"].append(audit_backend(binp, whisp, cfg))
        except Exception as e:
            step("backend.crashed", backend=cfg["name"], err=str(e))
            results["backends"].append(dict(backend=cfg["name"], error=str(e)))
        RESULTS.write_text(json.dumps(results, indent=2))
    step("SUMMARY", devices=devs, nvidia=nvidia)
    for r in results["backends"]:
        step("verdict", backend=r["backend"], v=r.get("verdict", r.get("error", "")))
    RESULTS.write_text(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
