# %% [markdown]
# # MOSS-TTS-Local 4B — piece-wise prompt fix validation (#249)
#
# ROOT CAUSE: we tokenized the whole prompt as one string; BPE is not compositional
# so the text boundary drifted ~2 tokens vs the reference processor (which encodes
# each segment separately). Those interior tokens amplify at the layer-10 attention
# sink and break the stop head. Fix = piece-wise assembly. This kernel PROVES it:
#  (1) token parity — our channel-0 ids now == the reference processor's input_ids;
#  (2) stop outcome — q4_k AND f16 now stop (~15-25 frames) instead of running away.

# %% [code]
import json, os, re, subprocess, sys, shutil
from pathlib import Path

REPO = Path("/kaggle/temp/CrispASR")
WORK = Path("/kaggle/working")
REF = os.environ.get("CRISPASR_REF", "fix/249-moss")
HF = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
if not REPO.exists():
    subprocess.check_call(["git", "clone", "--recursive", "--depth", "1", "--branch", REF,
                           "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    subprocess.check_call(["git", "-C", str(REPO), "submodule", "update", "--init",
                           "--recursive", "--depth", "1"], timeout=1800)
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402
kh.init_progress(hf_progress_repo="cstr/crispasr-kaggle-progress")
step = kh.step
step("start", ref=REF)
TOKEN = kh.resolve_hf_token("HF_TOKEN")
kh.install_build_toolchain()
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "hf_transfer"])


def robust_download(repo, fname, local_dir, token, tries=3, timeout=900):
    import multiprocessing as mp

    def _dl(q, use_ht):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" if use_ht else "0"
        os.environ["HF_HUB_DISABLE_XET"] = "0" if use_ht else "1"
        try:
            from huggingface_hub import hf_hub_download
            q.put(("ok", hf_hub_download(repo, fname, local_dir=local_dir, token=token)))
        except Exception as e:  # noqa: BLE001
            q.put(("err", repr(e)))

    for i in range(tries):
        q = mp.Queue(); p = mp.Process(target=_dl, args=(q, i == 0)); p.start(); p.join(timeout)
        if p.is_alive():
            p.terminate(); p.join(); continue
        if not q.empty():
            s, v = q.get()
            if s == "ok":
                return v
    raise RuntimeError(f"download failed: {repo}/{fname}")


BUILD = REPO / "build"
step("cmake.configure")
subprocess.run(["cmake", "-G", "Ninja", "-B", str(BUILD), "-S", str(REPO),
                "-DCMAKE_BUILD_TYPE=Release"] + kh.crispasr_cmake_flags(), check=True)
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(f"cmake --build {BUILD} --target crispasr-cli -j{kh.safe_build_jobs(gpu=False)}")
CLI = (BUILD / "bin" / "crispasr") if (BUILD / "bin" / "crispasr").exists() else next(iter(BUILD.rglob("crispasr")))
os.environ["LD_LIBRARY_PATH"] = f"{BUILD/'src'}:{BUILD/'ggml'/'src'}:{os.environ.get('LD_LIBRARY_PATH','')}"
step("build.done")

M = Path("/kaggle/temp/models"); M.mkdir(parents=True, exist_ok=True)
with kh.build_heartbeat("download"):
    CODEC = robust_download("cstr/moss-tts-local-v1.5-GGUF", "moss-tts-local-v1.5-codec.gguf", str(M), TOKEN)
    Q4 = robust_download("cstr/moss-tts-local-v1.5-GGUF", "moss-tts-local-v1.5-q4_k.gguf", str(M), TOKEN)
    F16 = robust_download("cstr/moss-tts-local-v1.5-GGUF", "moss-tts-local-v1.5-f16.gguf", str(M), TOKEN)

TEXTS = {"hello": "Hello world.", "fox": "The quick brown fox jumps over the lazy dog."}
SEEDS = [7, 42]
MAXF = 200
STOP_RE = re.compile(r"generated (\d+) frames \(max_frames=\d+, (stopped naturally|HIT CAP[^)]*)\)")

# ── (1) TOKEN PARITY: our channel-0 ids vs the reference processor ─────────────
step("tokparity.load")
tokparity = {}
try:
    from transformers import AutoProcessor  # noqa: E402
    proc = AutoProcessor.from_pretrained(HF, trust_remote_code=True)  # HF_TOKEN is in env
except Exception as e:  # noqa: BLE001 — never let tokparity block the stop test
    proc = None
    step("tokparity.load.err", err=repr(e)[:200])
for tag, text in (TEXTS.items() if proc is not None else []):
    ours_ids = None
    idp = WORK / f"pids_{tag}.txt"
    env = {**os.environ, "CRISPASR_MOSS_TTS_LOCAL_DUMP_PROMPT_IDS": str(idp),
           "CRISPASR_MOSS_TTS_LOCAL_MAX_FRAMES": "1"}
    with kh.build_heartbeat(f"tokdump.{tag}"):
        try:
            subprocess.run([str(CLI), "--backend", "moss-tts-local", "-m", Q4, "--codec-model", CODEC,
                            "--tts", text, "--tts-output", str(WORK / "t.wav"), "--no-prints"],
                           env=env, timeout=600, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            pass
    if idp.exists():
        ours_ids = [int(x) for x in idp.read_text().split()]
    feat = proc([proc.build_user_message(text=text)], mode="generation")
    iid = feat["input_ids"] if isinstance(feat, dict) else feat.input_ids
    import numpy as np
    ref_ids = np.asarray(iid)[..., 0].reshape(-1).tolist()  # channel 0
    match = ours_ids == ref_ids
    # first mismatch index for diagnostics
    fm = None
    if not match and ours_ids:
        for i in range(min(len(ours_ids), len(ref_ids))):
            if ours_ids[i] != ref_ids[i]:
                fm = {"idx": i, "ours": ours_ids[max(0, i - 2):i + 3], "ref": ref_ids[max(0, i - 2):i + 3]}
                break
    tokparity[tag] = {"match": match, "n_ours": len(ours_ids or []), "n_ref": len(ref_ids), "first_mismatch": fm}
    step(f"tokparity.{tag}", **tokparity[tag])

# ── (2) STOP OUTCOME: q4_k + f16 across texts/seeds ───────────────────────────
def run_stop(model, text, seed):
    env = {**os.environ, "CRISPASR_MOSS_TTS_LOCAL_DEBUG": "1", "CRISPASR_MOSS_TTS_LOCAL_MAX_FRAMES": str(MAXF)}
    try:
        r = subprocess.run([str(CLI), "--backend", "moss-tts-local", "-m", model, "--codec-model", CODEC,
                            "--tts", text, "--seed", str(seed), "--tts-output", str(WORK / "o.wav")],
                           capture_output=True, text=True, env=env, timeout=900)
        out = (r.stderr or "") + (r.stdout or "")
    except subprocess.TimeoutExpired:
        out = ""
    m = STOP_RE.search(out)
    return {"frames": int(m.group(1)) if m else None, "stopped": bool(m and m.group(2) == "stopped naturally")}


stop = []
for mtag, model in (("q4_k", Q4), ("f16", F16)):
    for tag, text in TEXTS.items():
        for seed in SEEDS:
            with kh.build_heartbeat(f"stop.{mtag}.{tag}.{seed}"):
                res = run_stop(model, text, seed)
            row = {"quant": mtag, "text": tag, "seed": seed, **res}
            stop.append(row)
            step(f"stop.{mtag}.{tag}.{seed}", **res)

n_each = len(TEXTS) * len(SEEDS)
q4_ok = sum(1 for r in stop if r["quant"] == "q4_k" and r["stopped"])
f16_ok = sum(1 for r in stop if r["quant"] == "f16" and r["stopped"])
tok_status = ("not-run" if not tokparity else ("PASS" if all(v["match"] for v in tokparity.values()) else "FAIL"))
stop_ok = q4_ok >= n_each - 1 and f16_ok >= n_each - 1
verdict = (f"{'FIXED' if stop_ok else 'stop NOT fixed'}: token-parity {tok_status}; "
           f"stop q4_k {q4_ok}/{n_each}, f16 {f16_ok}/{n_each}")
(WORK / "promptfix.json").write_text(json.dumps({"verdict": verdict, "token_parity": tokparity, "stop": stop},
                                                indent=2))
step("done", verdict=verdict)
print("DONE", verdict, flush=True)

try:
    for cand in ("/kaggle/working/.ccache", "/kaggle/temp/.ccache", str(Path.home() / ".ccache")):
        if Path(cand).exists():
            subprocess.run(f"tar cf /kaggle/working/ccache.tar -C {Path(cand).parent} {Path(cand).name}",
                           shell=True, timeout=600)
            if cand.startswith("/kaggle/working"):
                shutil.rmtree(cand, ignore_errors=True)
            break
except Exception:  # noqa: BLE001
    pass
