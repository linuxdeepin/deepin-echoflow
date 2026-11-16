# %% [markdown]
# # MOSS-TTS-Local 4B — q6_k/q8_0 RUNAWAY root-cause diagnostic (#249)
#
# FACT: 4B q4_k STOPS (~53s), q6_k/q8_0 RUN AWAY (never stop). The stop head
# (moss.local_text_head), local.*, audio_embed/head are ALL kept F16 by the
# quantizer's is_moss_tts guard in ALL three levels — so the ONLY variable is
# how the backbone (llm.*) is quantized. seed=0 → deterministic, so this is a
# reproducible numerics difference, NOT sampling luck.
#
# Two hypotheses, tested here without guessing:
#   (a) TENSOR-TYPE diff — different k-quant block sizes (q8_0=32 vs q4_k/q6_k=256)
#       can quantize DIFFERENT tensors by dimension divisibility. Diff the three
#       quants' per-tensor {name: type} maps: any tensor F16 in q4_k but quantized
#       in q6_k/q8_0 (or the reverse) is the suspect → keep it F16.
#   (b) RUNTIME stop-logit trace — synth a short line per quant with
#       CRISPASR_MOSS_TTS_LOCAL_DEBUG=1 and capture continue/stop per frame +
#       whether the stop head fires. Shows whether q6_k/q8_0's stop logit stays
#       pinned at continue while q4_k's rises.
#
# CPU (stop behavior must be numerically correct; GPU could confound). Short text
# + a 600s cap per synth: a runaway is conclusive well before 4096 frames.

# %% [code]
import json, os, re, shutil, subprocess, sys
from pathlib import Path

WORK = Path("/kaggle/working")
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
REPO = WORK / "CrispASR"
if not REPO.exists():
    subprocess.check_call(["git", "clone", "--recursive", "--depth", "1", CRISPASR_URL, str(REPO)])
    # cmake needs the ggml submodule; --recursive above usually gets it, but init
    # explicitly so a shallow-submodule hiccup doesn't leave ggml/ empty.
    subprocess.check_call(["git", "-C", str(REPO), "submodule", "update", "--init",
                           "--recursive", "--depth", "1"], timeout=1800)
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress(hf_progress_repo="cstr/crispasr-kaggle-progress")
step = kh.step
step("script.start", sha=subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip())

TOKEN = kh.resolve_hf_token("HF_TOKEN")
kh.install_build_toolchain()
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                       "huggingface_hub", "hf_transfer", "gguf"])
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download  # noqa: E402
import gguf  # noqa: E402

# ── models dir on the roomiest mount ──────────────────────────────────────────
def _free(p):
    try: return shutil.disk_usage(p).free / 1e9
    except Exception: return 0.0
MODELS = max([Path("/kaggle/temp/models"), Path("/tmp/models"), WORK / "models"],
             key=lambda d: _free(d.parent if d.parent.exists() else "/"))
MODELS.mkdir(parents=True, exist_ok=True)

# ── build crispasr-cli + crispasr-quantize (current main) ─────────────────────
BUILD = REPO / "build"
step("cmake.configure")
subprocess.run(["cmake", "-G", "Ninja", "-B", str(BUILD), "-S", str(REPO),
                "-DCMAKE_BUILD_TYPE=Release"] + kh.crispasr_cmake_flags(), check=True)
step("cmake.build.begin")
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(f"cmake --build {BUILD} --target crispasr-cli crispasr-quantize "
                        f"-j{kh.safe_build_jobs(gpu=False)}")
def _find(name):
    p = BUILD / "bin" / name
    return p if p.exists() else next(iter(BUILD.rglob(name)), None)
CLI, QUANT = _find("crispasr"), _find("crispasr-quantize")
assert CLI and QUANT, f"build missing: cli={CLI} quant={QUANT}"
os.environ["LD_LIBRARY_PATH"] = f"{BUILD/'src'}:{BUILD/'ggml'/'src'}:{os.environ.get('LD_LIBRARY_PATH','')}"
step("build.done", cli=str(CLI), quant=str(QUANT))

# ── download 4B f16 + codec ───────────────────────────────────────────────────
REPO_ID = "cstr/moss-tts-local-v1.5-GGUF"
with kh.build_heartbeat("download"):
    F16 = Path(hf_hub_download(REPO_ID, "moss-tts-local-v1.5-f16.gguf", local_dir=str(MODELS)))
    CODEC = Path(hf_hub_download(REPO_ID, "moss-tts-local-v1.5-codec.gguf", local_dir=str(MODELS)))
step("downloaded", f16_gb=round(F16.stat().st_size/1e9, 2))

# ── quantize to the three levels ──────────────────────────────────────────────
QUANTS = {}
for ft in ("q4_k", "q6_k", "q8_0"):
    out = MODELS / f"moss-tts-local-v1.5-{ft}.gguf"
    with kh.build_heartbeat(f"quantize.{ft}"):
        r = subprocess.run([str(QUANT), str(F16), str(out), ft], capture_output=True, text=True, timeout=3600)
    ok = out.exists()
    QUANTS[ft] = out if ok else None
    step(f"quantize.{ft}", ok=ok, gb=round(out.stat().st_size/1e9, 2) if ok else 0,
         tail=r.stdout[-200:] if not ok else "")

# ── (a) per-tensor type map + diff ────────────────────────────────────────────
def type_map(path):
    rd = gguf.GGUFReader(str(path))
    return {t.name: t.tensor_type.name for t in rd.tensors}
maps = {ft: type_map(p) for ft, p in QUANTS.items() if p}
f16map = type_map(F16)
names = sorted(set().union(*[set(m) for m in maps.values()]))
diffs = []
for n in names:
    row = {ft: maps[ft].get(n) for ft in maps}
    if len(set(row.values())) > 1:               # differs across levels
        diffs.append({"tensor": n, "f16": f16map.get(n), **row})
step("tensor-diff", n_tensors=len(names), n_differing=len(diffs), differing=diffs[:40])

# ── (b) runtime stop-logit trace per quant ───────────────────────────────────
SHORT = "Hello world."
def trace(model, tag):
    log = MODELS / f"trace_{tag}.log"
    wav = MODELS / f"out_{tag}.wav"
    env = {**os.environ, "CRISPASR_MOSS_TTS_LOCAL_DEBUG": "1"}
    with open(log, "w") as fh, kh.build_heartbeat(f"synth.{tag}"):
        try:
            subprocess.run([str(CLI), "--backend", "moss-tts-local", "-m", str(model),
                            "--codec-model", str(CODEC), "--tts", SHORT,
                            "--tts-output", str(wav), "--no-prints"],
                           stdout=fh, stderr=subprocess.STDOUT, env=env, timeout=600)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
    txt = log.read_text(errors="replace")
    gen = re.search(r"generated (\d+) frames .*?(stopped naturally|runaway)", txt)
    fired = re.search(r"stop head fired at frame (\d+)", txt)
    logit_lines = re.findall(r"frame (\d+): stop_head continue=([\-\d.]+) stop=([\-\d.]+)", txt)
    return {"timed_out": timed_out,
            "frames": int(gen.group(1)) if gen else None,
            "verdict": gen.group(2) if gen else ("runaway(timeout)" if timed_out else "unknown"),
            "stop_fired_frame": int(fired.group(1)) if fired else None,
            "first_logits": [{"f": int(f), "cont": float(c), "stop": float(s)} for f, c, s in logit_lines[:10]]}

traces = {}
for ft, p in QUANTS.items():
    if p:
        traces[ft] = trace(p, ft)
        step(f"trace.{ft}", **{k: v for k, v in traces[ft].items() if k != "first_logits"})

RESULT = {"differing_tensors": diffs, "traces": traces}
(WORK / "runaway_diag.json").write_text(json.dumps(RESULT, indent=2))
step("script.done", differing=len(diffs),
     verdicts={ft: t["verdict"] for ft, t in traces.items()})
print("DONE", json.dumps(RESULT)[:2000], flush=True)
