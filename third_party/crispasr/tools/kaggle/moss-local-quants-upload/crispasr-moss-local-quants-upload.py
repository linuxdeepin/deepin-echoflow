# %% [markdown]
# # MOSS-TTS-Local 4B — build q6_k/q8_0 with the prompt fix + validate + upload (#249)
#
# The prompt-tokenization fix (piece-wise assembly) makes the 4B stop reliably, so
# the higher-precision quants are finally usable. This builds the FIXED branch (the
# v0.8.23 binary still has the runaway), quantizes f16 -> q6_k/q8_0, validates each
# with BOTH a natural-stop check AND a TTS->whisper round-trip, and uploads only on
# pass to cstr/moss-tts-local-v1.5-GGUF (which already has f16 + q4_k + codec).

# %% [code]
import json, os, re, subprocess, sys, shutil
from pathlib import Path

REPO = Path("/kaggle/temp/CrispASR")
WORK = Path("/kaggle/working")
REF = os.environ.get("CRISPASR_REF", "fix/249-moss")
REPO_ID = "cstr/moss-tts-local-v1.5-GGUF"
BASE = "moss-tts-local-v1.5"
QUANTS = ["q6_k", "q8_0"]
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
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download, HfApi  # noqa: E402


def robust_download(repo, fname, local_dir, tries=3, timeout=1200):
    import multiprocessing as mp

    def _dl(q, ht):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" if ht else "0"
        os.environ["HF_HUB_DISABLE_XET"] = "0" if ht else "1"
        try:
            q.put(("ok", hf_hub_download(repo, fname, local_dir=local_dir, token=TOKEN)))
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
    kh.sh_with_progress(f"cmake --build {BUILD} --target crispasr-cli crispasr-quantize "
                        f"-j{kh.safe_build_jobs(gpu=False)}")
CLI = (BUILD / "bin" / "crispasr") if (BUILD / "bin" / "crispasr").exists() else next(iter(BUILD.rglob("crispasr")))
QUANT = next(iter(BUILD.rglob("crispasr-quantize")))
os.environ["LD_LIBRARY_PATH"] = f"{BUILD/'src'}:{BUILD/'ggml'/'src'}:{os.environ.get('LD_LIBRARY_PATH','')}"
step("build.done", cli=str(CLI), quant=str(QUANT))

M = Path("/kaggle/temp/models"); M.mkdir(parents=True, exist_ok=True)


def free_gb():
    return round(shutil.disk_usage(str(M)).free / 1e9, 1)


with kh.build_heartbeat("download"):
    CODEC = robust_download(REPO_ID, f"{BASE}-codec.gguf", str(M))
    F16 = robust_download(REPO_ID, f"{BASE}-f16.gguf", str(M))
    WHISPER = robust_download("ggerganov/whisper.cpp", "ggml-tiny.en.bin", str(M))
step("downloaded", free_gb=free_gb())

SYN = "The quick brown fox jumps over the lazy dog."
KEYS = ["quick", "brown", "fox", "lazy", "dog"]
STOP_RE = re.compile(r"generated (\d+) frames \(max_frames=\d+, (stopped naturally|HIT CAP[^)]*)\)")


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).split()


def validate(backbone):
    wav = WORK / f"{Path(backbone).stem}.wav"
    env = {**os.environ, "CRISPASR_MOSS_TTS_LOCAL_DEBUG": "1", "CRISPASR_MOSS_TTS_LOCAL_MAX_FRAMES": "400"}
    with kh.build_heartbeat(f"synth.{Path(backbone).stem}"):
        p = subprocess.run([str(CLI), "--backend", "moss-tts-local", "-m", str(backbone), "--codec-model",
                            str(CODEC), "--tts", SYN, "--seed", "7", "--tts-output", str(wav)],
                           capture_output=True, text=True, env=env, timeout=1800)
    out = (p.stderr or "") + (p.stdout or "")
    m = STOP_RE.search(out)
    stopped = bool(m and m.group(2) == "stopped naturally")
    frames = int(m.group(1)) if m else None
    ok_synth = p.returncode == 0 and wav.is_file() and wav.stat().st_size > 20000
    if not (ok_synth and stopped):
        return False, {"stage": "synth", "stopped": stopped, "frames": frames, "exit": p.returncode}
    pr = subprocess.run([str(CLI), "-m", WHISPER, "-f", str(wav)], capture_output=True, text=True, timeout=900)
    rt = " ".join(re.sub(r"\[[^\]]*\]", "", ln).strip() for ln in pr.stdout.splitlines())
    hits = sum(k in set(norm(rt)) for k in KEYS)
    wav.unlink(missing_ok=True)
    return hits >= 3, {"stage": "roundtrip", "stopped": stopped, "frames": frames, "hits": hits,
                       "transcript": rt[:160]}


produced = {}
for ftype in QUANTS:
    out = M / f"{BASE}-{ftype}.gguf"
    step(f"quantize.{ftype}.begin", free_gb=free_gb())
    with kh.build_heartbeat(f"quantize.{ftype}"):
        q = subprocess.run([str(QUANT), str(F16), str(out), ftype], capture_output=True, text=True, timeout=3600)
    if q.returncode != 0 or not out.is_file():
        step(f"quantize.{ftype}.FAILED", exit=q.returncode, tail=(q.stdout or q.stderr)[-300:])
        produced[ftype] = "quantize-failed"
        continue
    ok, info = validate(out)
    step(f"validate.{ftype}", ok=ok, gb=round(out.stat().st_size / 1e9, 2), **info)
    if not ok:
        produced[ftype] = f"validation-failed: {info}"
        out.unlink(missing_ok=True)
        continue
    with kh.build_heartbeat(f"upload.{ftype}"):
        HfApi(token=TOKEN).upload_file(path_or_fileobj=str(out), path_in_repo=out.name, repo_id=REPO_ID,
                                       repo_type="model",
                                       commit_message=f"Add {ftype} (piece-wise prompt fix; stop+roundtrip validated) (#249)")
    produced[ftype] = f"uploaded ({round(out.stat().st_size / 1e9, 2)} GB, {info.get('frames')} frames)"
    step(f"upload.{ftype}.done")
    out.unlink(missing_ok=True)

(WORK / "quants.json").write_text(json.dumps(produced, indent=2))
step("done", produced=produced)
print("DONE", json.dumps(produced), flush=True)

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
