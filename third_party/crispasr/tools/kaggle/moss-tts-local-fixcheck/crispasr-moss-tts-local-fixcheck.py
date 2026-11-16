# CrispASR — MOSS-TTS-Local 4B stop-fix validation (#249, P5) — CPU, no codec
#
# The model card recommends audio_temperature=1.7 / top_p 0.8 / top_k 25 with the
# stop head SAMPLED (text_temperature 1.0); my old generic defaults (1.0/0.95/50,
# greedy stop) produced a degenerate acoustic trajectory that never reached a
# natural end -> the stop head never fired -> runaway (P5 run1/run2). The fix set
# the defaults to the card values. This validates that fix on MY code, cheaply:
# build moss-tts-local-smoke (CPU), run generate_codes on "Hello world." + a long
# passage, and check the stop head now FIRES at a sane frame count (no codec, no
# GPU, no ASR).

import os
import re
import subprocess
import sys
import time
import json
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
TMP = Path("/tmp")
REPO = TMP / "CrispASR"
BUILD = REPO / "build"
MODELS = TMP / "moss-local"
WORK = Path("/kaggle/working")
RESULTS = WORK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

REF = os.environ.get("CRISPASR_REF", "feat/moss-tts-local-4b")
GGUF_REPO = os.environ.get("MOSS_GGUF_REPO", "cstr/moss-tts-local-v1.5-GGUF")
MAXF = int(os.environ.get("MOSS_MAXF", "300"))
SHORT = "Hello world."
LONG = ("The quick brown fox jumps over the lazy dog. "
        "Speech synthesis should stay intelligible over a longer passage.")
_T0 = time.time()


def log(m):
    print(f"[{round(time.time() - _T0, 1)}s] {m}", flush=True)


def run_smoke(smoke, gguf, tag, text):
    env = os.environ.copy()
    env["CRISPASR_MOSS_TTS_LOCAL_DEBUG"] = "1"
    r = subprocess.run([str(smoke), str(gguf), text, str(MAXF)], capture_output=True, text=True,
                       timeout=3600, env=env)
    out = r.stdout + "\n--STDERR--\n" + r.stderr
    (RESULTS / f"smoke_{tag}.log").write_text(out)
    m = re.search(r"generated (\d+) frames .*?(runaway|stopped naturally)", out)
    frames = int(m.group(1)) if m else None
    stopped = (m.group(2) == "stopped naturally") if m else None
    fired = re.search(r"stop head fired at frame (\d+)", out)
    logits = re.findall(r"frame (\d+): stop_head continue=(-?[\d.]+) stop=(-?[\d.]+)", out)
    traj = [(int(a), float(b), float(c)) for a, b, c in logits]
    return {"rc": r.returncode, "frames": frames, "stopped": stopped,
            "fired_at": int(fired.group(1)) if fired else None,
            "logit_first": traj[:8], "logit_last": traj[-4:]}


def main():
    summary = {"ref": REF, "maxf": MAXF}
    log(f"clone {REF}")
    if not REPO.exists():
        subprocess.check_call(["git", "clone", "--depth", "1", "--branch", REF, "--recursive",
                               "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.init_progress()
    tok = kh.resolve_hf_token()
    summary["sha"] = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()

    # ── build moss-tts-local-smoke (CPU, no CUDA) ──────────────────────────
    kh.install_build_toolchain()
    os.environ["CCACHE_DIR"] = "/kaggle/temp/.ccache"
    env = os.environ.copy()
    subprocess.run(["cmake", "-G", "Ninja", "-B", str(BUILD), "-S", str(REPO), "-DCMAKE_BUILD_TYPE=Release",
                    "-DGGML_CUDA=OFF", "-DGGML_METAL=OFF"] + list(kh.cache_and_link_flags()),
                   env=env, check=True, timeout=300)
    with kh.build_heartbeat("smoke build (cpu)"):
        kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} --target moss-tts-local-smoke "
                            f"-j{kh.safe_build_jobs(gpu=False)}")
    smoke = next((c for c in BUILD.rglob("moss-tts-local-smoke") if c.is_file() and os.access(c, os.X_OK)), None)
    if not smoke:
        raise SystemExit("smoke not built")
    os.environ["LD_LIBRARY_PATH"] = f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH','')}"
    log(f"built {smoke}")

    # ── download the backbone F16 GGUF ─────────────────────────────────────
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import hf_hub_download
    gguf = Path(hf_hub_download(GGUF_REPO, "moss-tts-local-v1.5-f16.gguf", local_dir=str(MODELS), token=tok))
    log(f"gguf {gguf} ({gguf.stat().st_size/1e9:.2f} GB)")

    # ── run generate_codes with the FIXED card-default sampling params ─────
    res = {}
    for tag, text in (("short", SHORT), ("long", LONG)):
        res[tag] = run_smoke(smoke, gguf, tag, text)
        log(f"{tag}: frames={res[tag]['frames']} stopped={res[tag]['stopped']} "
            f"fired_at={res[tag]['fired_at']} rc={res[tag]['rc']}")
    summary["result"] = res

    short_ok = res["short"].get("stopped") is True and (res["short"].get("frames") or 999) < MAXF
    # a natural utterance of "Hello world." should be short (< ~60 frames @ 12.5Hz)
    short_sane = short_ok and (res["short"].get("frames") or 999) < 80
    summary["verdict"] = ("FIX WORKS: 'Hello world' stops at %s frames" % res["short"].get("frames")
                          if short_sane else
                          "STILL RUNAWAY: short=%s frames stopped=%s — params not the (whole) fix"
                          % (res["short"].get("frames"), res["short"].get("stopped")))
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 60 + "\n" + json.dumps(summary, indent=2) + "\n" + "=" * 60)
    log(f"VERDICT: {summary['verdict']}")
    if not short_sane:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        import traceback
        log(f"FATAL: {e}\n{traceback.format_exc()}")
        sys.exit(1)
