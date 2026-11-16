# %% [markdown]
# # MOSS-TTS-Local 4B — is the f16 KV cache the stop bug? (#249) — fast, no build
#
# Per-layer diff pinned the backbone divergence to START at layer 10 (0-9 exact),
# structure identical -> a numerical cause. Prime suspect: our f16 KV cache read
# (ggml computes QK^T on raw f16 K; the reference upcasts). Test cheaply with the
# prebuilt v0.8.23 binary + existing env knobs — does forcing f32 KV make our
# synth stop early (like the reference @~15) instead of running away?

# %% [code]
import json, os, re, subprocess, sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
if not REPO.exists():
    subprocess.check_call(["git", "clone", "--depth", "1",
                           "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402
kh.init_progress(hf_progress_repo="cstr/crispasr-kaggle-progress")
step = kh.step
step("start")
TOKEN = kh.resolve_hf_token("HF_TOKEN")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "hf_transfer"])
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download  # noqa: E402

BIN = WORK / "bin"; BIN.mkdir(exist_ok=True)
CLI = BIN / "crispasr"
subprocess.check_call(
    "wget -q https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.23/crispasr-linux-x86_64.tar.gz "
    f"-O /tmp/c.tgz && tar -xzf /tmp/c.tgz -C {BIN} --strip-components=1", shell=True)
CLI.chmod(0o755)
os.environ["LD_LIBRARY_PATH"] = f"{BIN}:{os.environ.get('LD_LIBRARY_PATH','')}"
MODELS = Path("/kaggle/temp/models"); MODELS.mkdir(parents=True, exist_ok=True)
with kh.build_heartbeat("download"):
    F16 = hf_hub_download("cstr/moss-tts-local-v1.5-GGUF", "moss-tts-local-v1.5-f16.gguf", local_dir=str(MODELS))
    CODEC = hf_hub_download("cstr/moss-tts-local-v1.5-GGUF", "moss-tts-local-v1.5-codec.gguf", local_dir=str(MODELS))

VARIANTS = {
    "default": {},
    "kv_read_f32": {"CRISPASR_KV_READ_F32": "1"},
    "kv_quant_f32": {"CRISPASR_KV_QUANT": "f32"},
    "kv_quant_moss_f32": {"CRISPASR_KV_QUANT": "moss_tts_local:f32"},
}
results = {}
for tag, extra in VARIANTS.items():
    env = {**os.environ, "CRISPASR_MOSS_TTS_LOCAL_DEBUG": "1", **extra}
    log = WORK / f"{tag}.log"
    with open(log, "w") as fh, kh.build_heartbeat(f"synth.{tag}"):
        try:
            subprocess.run([str(CLI), "--backend", "moss-tts-local", "-m", F16, "--codec-model", CODEC,
                            "--tts", "Hello world.", "--tts-output", str(WORK / f"{tag}.wav"), "--no-prints"],
                           stdout=fh, stderr=subprocess.STDOUT, env=env, timeout=1200)
            to = False
        except subprocess.TimeoutExpired:
            to = True
    txt = log.read_text(errors="replace")
    g = re.search(r"generated (\d+) frames .*?(stopped naturally|runaway)", txt)
    results[tag] = {"verdict": g.group(2) if g else ("runaway(timeout)" if to else "?"),
                    "frames": int(g.group(1)) if g else None}
    step(f"synth.{tag}", **results[tag])

verdict = "F32-KV FIXES IT" if any(
    v["verdict"] == "stopped naturally" and v["frames"] and v["frames"] < 30 and tag != "default"
    for tag, v in results.items()) else "KV not the (whole) cause"
(WORK / "kv_test.json").write_text(json.dumps({"verdict": verdict, "results": results}, indent=2))
step("done", verdict=verdict, results=results)
print("DONE", json.dumps(results), flush=True)
