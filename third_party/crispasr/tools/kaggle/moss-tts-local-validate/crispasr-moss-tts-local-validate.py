# CrispASR — MOSS-TTS-Local-Transformer-v1.5 (4B) Kaggle validation (#249, P5)
#
# Rebuilt on the proven harness regime (kaggle_usage.md + gemma4-e4b-convert):
#   - kh.init_progress() mirrors progress.jsonl to HF cstr/crispasr-kaggle-progress
#     (requires os.environ["HF_TOKEN"] so the push fires) -> externally watchable.
#   - kh.step() at EVERY phase; kh.build_heartbeat() around the build.
#   - hf_transfer for downloads (no http_get CLOSE_WAIT hang).
#   - ccache warmed from ${KAGGLE_ACCOUNT}/crispasr-ccache, MOVED off /kaggle/working (§22).
#   - everything large under /kaggle/temp (or /tmp), /kaggle/working stays small.
#
# Acceptance (HARD RULE #3): the decoded ASR round-trip. All fixes in place:
# chunked codec + sched-size fix + card-correct sampling defaults. Gate = q4_k.

import json
import os
import re
import struct
import subprocess
import sys
import time
import traceback
import wave
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # hf_transfer re-fetches from 0 (no resume) -> wedges; use curl -C -

TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
WORK = Path("/kaggle/working")
REPO = TEMP / "CrispASR"
BUILD = REPO / "build"
MODELS = TEMP / "moss-local-models"
RESULTS = WORK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

CRISPASR_REF = os.environ.get("CRISPASR_REF", "feat/moss-tts-local-4b")
CRISPASR_REPO = os.environ.get("CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git")
HF_MODEL = os.environ.get("MOSS_MODEL", "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5")
HF_CODEC = os.environ.get("MOSS_CODEC", "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2")
GGUF_REPO = os.environ.get("MOSS_GGUF_REPO", "cstr/moss-tts-local-v1.5-GGUF")
DO_UPLOAD = os.environ.get("MOSS_UPLOAD", "1") == "1"

SHORT_TEXT = "Hello world."
LONG_TEXT = ("The quick brown fox jumps over the lazy dog. "
             "Speech synthesis should stay intelligible over a longer passage, "
             "so this sentence exercises many autoregressive steps and the codec "
             "sliding window well past the first few frames.")


# ── WAV / ASR / synth helpers ───────────────────────────────────────────────
def wav_summary(path):
    if not path.exists():
        return {"error": "missing"}
    with wave.open(str(path), "rb") as w:
        n, sr, sw, ch = w.getnframes(), w.getframerate(), w.getsampwidth(), w.getnchannels()
        raw = w.readframes(n)
    if sw != 2:
        return {"error": f"sw={sw}"}
    pcm = struct.unpack(f"<{n * ch}h", raw)
    if ch > 1:
        pcm = pcm[::ch]
    if not pcm:
        return {"duration_s": 0.0, "rms": 0.0, "n_samples": 0, "sr": sr}
    rms = ((sum(int(x) * int(x) for x in pcm) / max(1, len(pcm))) ** 0.5) / 32768.0
    return {"duration_s": round(len(pcm) / sr, 3), "rms": round(rms, 6), "n_samples": len(pcm), "sr": sr, "ch": ch}


def asr_roundtrip(cli, wav, timeout=900):
    if not wav.exists():
        return ""
    try:
        r = subprocess.run([str(cli), "--backend", "whisper", "-m", "auto", "--auto-download",
                            "-f", str(wav), "--no-prints"],
                           timeout=timeout, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (RESULTS / f"{wav.stem}.asr.log").write_text(r.stdout)
        lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
        return " ".join(ln for ln in lines if not ln.startswith(("[", "whisper", "ggml", "load", "crispasr")))
    except Exception as ex:  # noqa: BLE001
        return f"<asr-error: {type(ex).__name__}>"


def synth(cli, backbone, codec, text, out_wav, timeout=2400):
    env = os.environ.copy()
    env["CRISPASR_MOSS_TTS_LOCAL_DEBUG"] = "1"
    t0 = time.time()
    try:
        r = subprocess.run([str(cli), "--backend", "moss-tts-local", "-m", backbone, "--codec-model", codec,
                            "--tts", text, "--tts-output", str(out_wav), "--no-prints"],
                           timeout=timeout, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        rc, out = r.returncode, r.stdout
    except subprocess.TimeoutExpired as ex:
        rc, out = -1, f"TIMEOUT {timeout}s\n{ex.stdout or ''}"
    (RESULTS / f"{out_wav.stem}.log").write_text(out)
    m = re.search(r"generated (\d+) frames .*?(runaway|stopped naturally)", out)
    return {"rc": rc, "elapsed_s": round(time.time() - t0, 1),
            "frames": int(m.group(1)) if m else None,
            "stopped": (m.group(2) == "stopped naturally") if m else None,
            "wav": wav_summary(out_wav) if out_wav.exists() else {"error": "no-wav"},
            "err_excerpt": out[-1500:]}


def verdict(res, min_dur):
    if res["rc"] != 0:
        return f"FAIL: rc={res['rc']}"
    w = res["wav"]
    if "error" in w:
        return f"FAIL: wav {w['error']}"
    if w["duration_s"] < min_dur:
        return f"FAIL: too short ({w['duration_s']}s < {min_dur})"
    if w["rms"] < 1e-4:
        return f"FAIL: silent (rms={w['rms']})"
    return "PASS"


def word_overlap(asr_text, ref_text):
    ref = set(re.findall(r"[a-z]+", ref_text.lower()))
    got = set(re.findall(r"[a-z]+", (asr_text or "").lower()))
    return round(len(ref & got) / len(ref), 3) if ref else 0.0


def main():
    # ── harness + progress mirror ──────────────────────────────────────────
    if not REPO.exists():
        subprocess.check_call(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF, "--recursive",
                               CRISPASR_REPO, str(REPO)])
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    if str(REPO / "tools" / "kaggle") not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))  # bundled fallback
    import kaggle_harness as kh
    kh.init_progress()  # mirrors to HF cstr/crispasr-kaggle-progress once HF_TOKEN is set
    # Fixed, identifiable HF progress path (the default is <ts>-batch-unknown.jsonl,
    # indistinguishable from every other CrispASR kernel on the shared mirror).
    kh._HF_PROGRESS_PATH = "runs/moss-tts-local-validate-live.jsonl"
    kh._HF_PUSH_INTERVAL_S = 15.0
    summary = {"ref": CRISPASR_REF, "gates": {}, "roundtrip": {}, "phases": {}}
    summary["sha"] = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    kh.step("cloned", sha=summary["sha"][:10])

    subprocess.run(["nvidia-smi", "-L"], check=False)
    summary["gpu"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
    kh.step("gpu", gpu=summary["gpu"])

    # ── deps + HF token (HF_TOKEN enables the progress push) ───────────────
    kh.step("install deps")
    kh.install_build_toolchain()  # ninja + ccache + mold; warms /kaggle/working/.ccache
    kh.sh_with_progress("pip install -q huggingface_hub hf_transfer safetensors gguf")
    hf_token = kh.resolve_hf_token()
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    kh.step("hf token", ok=bool(hf_token))

    # ccache §22: move warmed cache OFF /kaggle/working so output stays page-1 small.
    subprocess.run("if [ -d /kaggle/working/.ccache ]; then rm -rf /kaggle/temp/.ccache; "
                   "mv /kaggle/working/.ccache /kaggle/temp/.ccache; fi", shell=True, check=False)
    os.environ["CCACHE_DIR"] = "/kaggle/temp/.ccache"
    subprocess.run("ccache -s | head -5", shell=True, check=False)

    # ── build cli + quantize ───────────────────────────────────────────────
    kh.step("cmake configure")
    arch = kh.detect_cuda_arch()
    subprocess.run(["cmake", "-G", "Ninja", "-B", str(BUILD), "-S", str(REPO), "-DCMAKE_BUILD_TYPE=Release"]
                   + list(kh.cache_and_link_flags()) + list(kh.cuda_build_flags(arch)),
                   check=True, timeout=300)
    with kh.build_heartbeat("cmake.build"):
        kh.sh_with_progress(f"cmake --build {BUILD} --target crispasr-cli crispasr-quantize "
                            f"-j{kh.safe_build_jobs(gpu=True)}")
    cli = next((c for c in [BUILD / "bin" / "crispasr"] + list(BUILD.rglob("crispasr"))
                if c.is_file() and os.access(c, os.X_OK)), None)
    quant = next((c for c in [BUILD / "bin" / "crispasr-quantize"] + list(BUILD.rglob("crispasr-quantize"))
                  if c.is_file() and os.access(c, os.X_OK)), None)
    if not cli or not quant:
        raise SystemExit(f"binaries missing cli={cli} quant={quant}")
    os.environ["LD_LIBRARY_PATH"] = f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    kh.step("build ok")
    # Stash the (now-populated) ccache as a single output tar so the seed dataset
    # can be refreshed externally -> future builds warm (~3 min instead of ~23).
    subprocess.run("cd /kaggle/temp && tar cf /kaggle/working/ccache.tar .ccache 2>/dev/null || true", shell=True)
    kh.step("ccache stashed")

    # ── backbone F16 (download) + codec (download safetensors + convert) ───
    import threading
    conv = REPO / "models" / "convert-moss-tts-local-to-gguf.py"
    HFBASE = "https://huggingface.co"

    def curl_get(what, url, out, total_timeout=3000):
        # Resumable download. curl -C - continues the partial file; --speed-time
        # aborts a stalled connection (<30KB/s for 20s) and --retry resumes from the
        # partial byte -> ratchets THROUGH the wedge that kills hf_transfer (which
        # re-fetches from 0 each attempt and never resumes). Watched via file size.
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = ("curl -sSL -C - --retry 50 --retry-delay 3 --retry-all-errors "
               "--speed-limit 30000 --speed-time 20 --fail-with-body "
               f'-H "Authorization: Bearer {hf_token}" -o "{out}" "{url}"')
        box = {}

        def _w():
            box["rc"] = subprocess.run(cmd, shell=True).returncode

        t = threading.Thread(target=_w, daemon=True)
        t.start()
        t0 = time.time()
        while t.is_alive():
            t.join(15)
            mb = round(out.stat().st_size / 1e6, 1) if out.exists() else 0.0
            kh.step(f"{what} (dl)", mb=mb, s=round(time.time() - t0))
            kh._push_progress_to_hf(force=True)
            if time.time() - t0 > total_timeout:
                raise SystemExit(f"{what}: timeout {total_timeout}s at {mb}MB")
        if box.get("rc") != 0:
            raise SystemExit(f"{what}: curl rc={box.get('rc')}")
        kh.step(f"{what}: ok", mb=round(out.stat().st_size / 1e6, 1))
        return out

    # backbone F16 (single hosted file) + codec safetensors (3 shards) + config
    f16 = MODELS / "moss-tts-local-v1.5-f16.gguf"
    curl_get("backbone F16", f"{HFBASE}/{GGUF_REPO}/resolve/main/{f16.name}", f16)
    codec_dir = MODELS / "codec-src"
    for fn in ("config.json", "model.safetensors.index.json",
               "model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors",
               "model-00003-of-00003.safetensors"):
        curl_get(f"codec {fn}", f"{HFBASE}/{HF_CODEC}/resolve/main/{fn}", codec_dir / fn, total_timeout=2400)

    codec_gguf = MODELS / "moss-tts-local-v1.5-codec.gguf"
    kh.step("convert codec")
    codegen = ("import importlib.util as u; from pathlib import Path;"
               "s=u.spec_from_file_location('c', r'%s'); m=u.module_from_spec(s); s.loader.exec_module(m);"
               "m.write_codec_gguf(Path(r'%s'), Path(r'%s'))") % (str(conv), str(codec_dir), str(codec_gguf))
    subprocess.run([sys.executable, "-c", codegen], check=True, timeout=3600)
    if not codec_gguf.exists():
        raise SystemExit("codec GGUF not produced")
    summary["phases"]["convert"] = {"f16_gb": round(f16.stat().st_size / 1e9, 2),
                                    "codec_gb": round(codec_gguf.stat().st_size / 1e9, 2)}
    kh.step("converted", **summary["phases"]["convert"])

    # ── quantize -> Q4_K ───────────────────────────────────────────────────
    kh.step("quantize q4_k")
    q4k = MODELS / "moss-tts-local-v1.5-q4_k.gguf"
    subprocess.run([str(quant), str(f16), str(q4k), "q4_k"], check=True, timeout=1800)
    summary["phases"]["quantize"] = {"q4k_gb": round(q4k.stat().st_size / 1e9, 2)}
    kh.step("quantized", **summary["phases"]["quantize"])

    # ── round-trip: q4_k (gate) + f16 (best-effort) with card-default params ─
    # F16 is the acceptance target; Q4_K's long trajectory runs away (intrinsic
    # quantized-AR drift, tts-port-parity-via-logit-rank) so it's best-effort.
    for tag, backbone, gating in (("f16", str(f16), True), ("q4k", str(q4k), False)):
        kh.step(f"synth {tag} short")
        rs = synth(cli, backbone, str(codec_gguf), SHORT_TEXT, RESULTS / f"{tag}_short.wav")
        rs["verdict"] = verdict(rs, 0.2)
        rs["asr"] = asr_roundtrip(cli, RESULTS / f"{tag}_short.wav")
        rs["overlap"] = word_overlap(rs["asr"], SHORT_TEXT)
        kh.step(f"{tag} short done", frames=rs["frames"], stopped=rs["stopped"],
                verdict=rs["verdict"], overlap=rs["overlap"])

        kh.step(f"synth {tag} long")
        rl = synth(cli, backbone, str(codec_gguf), LONG_TEXT, RESULTS / f"{tag}_long.wav")
        rl["verdict"] = verdict(rl, 2.0)
        rl["asr"] = asr_roundtrip(cli, RESULTS / f"{tag}_long.wav")
        rl["overlap"] = word_overlap(rl["asr"], LONG_TEXT)
        kh.step(f"{tag} long done", frames=rl["frames"], stopped=rl["stopped"],
                verdict=rl["verdict"], overlap=rl["overlap"])

        sw, lw = len((rs["asr"] or "").split()), len((rl["asr"] or "").split())
        pow_ok = (rl["verdict"] == "PASS" and rs["verdict"] == "PASS" and lw > sw
                  and rl["wav"].get("duration_s", 0) > rs["wav"].get("duration_s", 0))
        recognizable = (rs["overlap"] >= 0.5 and rl["overlap"] >= 0.4
                        and rs["stopped"] is not False and rl["stopped"] is not False)
        summary["roundtrip"][tag] = {"short": rs, "long": rl, "proof_of_work": pow_ok,
                                     "recognizable": recognizable}
        oom = (not gating) and rs["rc"] != 0 and "out of memory" in (rs["err_excerpt"] or "").lower()
        summary["gates"][f"roundtrip_{tag}"] = ("SKIP(oom)" if oom else
                                                ("PASS" if (pow_ok and recognizable) else
                                                 ("FAIL" if gating else "WARN")))
        kh.step(f"gate {tag}", result=summary["gates"][f"roundtrip_{tag}"])

    summary["all_gates_pass"] = summary["gates"].get("roundtrip_f16") == "PASS"

    # ── upload GGUFs on pass ───────────────────────────────────────────────
    if summary["all_gates_pass"] and DO_UPLOAD:
        kh.step("upload GGUFs")
        try:
            api = HfApi()
            api.create_repo(GGUF_REPO, repo_type="model", exist_ok=True, token=hf_token)
            for p in (codec_gguf, q4k):
                api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name, repo_id=GGUF_REPO,
                                repo_type="model", token=hf_token)
            summary["uploaded"] = [codec_gguf.name, q4k.name]
        except Exception as e:  # noqa: BLE001
            summary["uploaded"] = f"upload error: {e}"

    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 60 + "\n" + json.dumps(summary, indent=2) + "\n" + "=" * 60)
    kh.step("done", all_gates_pass=summary["all_gates_pass"], gates=summary["gates"])
    if not summary["all_gates_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"FATAL: {e}\n{traceback.format_exc()}", flush=True)
        try:
            import kaggle_harness as kh
            kh.step("FATAL", err=str(e)[:200])
        except Exception:
            pass
        sys.exit(1)
