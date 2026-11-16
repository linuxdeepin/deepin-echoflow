# %% [markdown]
# # MOSS-TTS-Local 4B — does eager+F32 attention fix the stop? (#249) A/B
#
# Diagnosis: our flash_attn_ext gives reduced-precision QK^T scores; at the
# layer-10 near-tied softmax that flips which key wins, drifting the hidden state
# so the 2-way stop head's gap stays high and the 4B runs away. Two reference
# ports (llama.cpp LM decode) use eager mul_mat + GGML_PREC_F32 + soft_max_ext and
# match the reference to ~5e-5. This kernel A/Bs our new eager path (default for
# moss-tts-local) vs flash (CRISPASR_CORE_ATTN_EAGER_F32=0) on the same texts+seeds
# and reports whether eager STOPS where flash runs away.

# %% [code]
import json, os, re, subprocess, sys, shutil, time
from pathlib import Path

REPO = Path("/kaggle/temp/CrispASR")
WORK = Path("/kaggle/working")
REF = os.environ.get("CRISPASR_REF", "fix/249-moss")
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
        q = mp.Queue()
        p = mp.Process(target=_dl, args=(q, i == 0))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate(); p.join(); step("dl.hang.retry", file=fname, attempt=i); continue
        if not q.empty():
            s, v = q.get()
            if s == "ok":
                return v
            step("dl.err.retry", file=fname, attempt=i, err=v[:200])
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

MODELS = Path("/kaggle/temp/models"); MODELS.mkdir(parents=True, exist_ok=True)
with kh.build_heartbeat("download"):
    CODEC = robust_download("cstr/moss-tts-local-v1.5-GGUF", "moss-tts-local-v1.5-codec.gguf", str(MODELS), TOKEN)
    F16 = robust_download("cstr/moss-tts-local-v1.5-GGUF", "moss-tts-local-v1.5-f16.gguf", str(MODELS), TOKEN)

TEXTS = {"hello": "Hello world.", "fox": "The quick brown fox jumps over the lazy dog."}
SEEDS = [7, 42]
MAXF = 120
STOP_RE = re.compile(r"generated (\d+) frames \(max_frames=\d+, (stopped naturally|HIT CAP[^)]*)\)")


def run(text, seed, eager):
    env = {**os.environ, "CRISPASR_MOSS_TTS_LOCAL_DEBUG": "1",
           "CRISPASR_MOSS_TTS_LOCAL_MAX_FRAMES": str(MAXF)}
    if not eager:
        env["CRISPASR_CORE_ATTN_EAGER_F32"] = "0"  # force flash
    cmd = [str(CLI), "--backend", "moss-tts-local", "-m", F16, "--codec-model", CODEC,
           "--tts", text, "--seed", str(seed), "--tts-output", str(WORK / "o.wav")]
    to = False
    t0 = time.monotonic()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
        out = (r.stderr or "") + (r.stdout or "")
    except subprocess.TimeoutExpired as e:
        out = (e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        to = True
    elapsed = time.monotonic() - t0
    m = STOP_RE.search(out)
    frames = int(m.group(1)) if m else None
    stopped = bool(m and m.group(2) == "stopped naturally")
    # per-frame ms is the fair perf metric (eager stops early, flash hits the cap)
    per_frame_ms = round(elapsed / frames * 1000, 1) if frames else None
    return {"frames": frames, "stopped": stopped, "timeout": to,
            "elapsed_s": round(elapsed, 1), "per_frame_ms": per_frame_ms}


results = []
for tag, text in TEXTS.items():
    for seed in SEEDS:
        for eager in (True, False):
            with kh.build_heartbeat(f"{'eager' if eager else 'flash'}.{tag}.{seed}"):
                res = run(text, seed, eager)
            row = {"attn": "eager" if eager else "flash", "text": tag, "seed": seed, **res}
            results.append(row)
            step(f"{row['attn']}.{tag}.{seed}", **res)

eager_stops = [r for r in results if r["attn"] == "eager" and r["stopped"]]
flash_stops = [r for r in results if r["attn"] == "flash" and r["stopped"]]
n_each = len(TEXTS) * len(SEEDS)


def frames_list(attn):
    return [r["frames"] for r in results if r["attn"] == attn and r["stopped"]]


def pf(attn):  # per-frame ms across all runs of this attn (perf, independent of stop)
    xs = sorted(r["per_frame_ms"] for r in results if r["attn"] == attn and r["per_frame_ms"])
    return xs[len(xs) // 2] if xs else None


perf = {"eager_per_frame_ms": pf("eager"), "flash_per_frame_ms": pf("flash")}
perf["eager_vs_flash_slowdown"] = (round(perf["eager_per_frame_ms"] / perf["flash_per_frame_ms"], 2)
                                   if perf["eager_per_frame_ms"] and perf["flash_per_frame_ms"] else None)

verdict = (f"EAGER FIXES IT: eager stopped {len(eager_stops)}/{n_each} (frames {frames_list('eager')}), "
           f"flash stopped {len(flash_stops)}/{n_each} (frames {frames_list('flash')})"
           if len(eager_stops) > len(flash_stops) and len(eager_stops) >= n_each - 1
           else f"inconclusive: eager {len(eager_stops)}/{n_each}, flash {len(flash_stops)}/{n_each}")
(WORK / "eager_ab.json").write_text(json.dumps({"verdict": verdict, "perf": perf, "maxf": MAXF,
                                                "results": results}, indent=2))
step("done", verdict=verdict, perf=perf)
print("DONE", verdict, "| PERF", json.dumps(perf), flush=True)

# refresh ccache dataset (gotcha #22)
try:
    for cand in ("/kaggle/working/.ccache", "/kaggle/temp/.ccache", str(Path.home() / ".ccache")):
        if Path(cand).exists():
            subprocess.run(f"tar cf /kaggle/working/ccache.tar -C {Path(cand).parent} {Path(cand).name}",
                           shell=True, timeout=600)
            if cand.startswith("/kaggle/working"):
                shutil.rmtree(cand, ignore_errors=True)
            step("ccache.tar", src=cand, size_mb=round(Path("/kaggle/working/ccache.tar").stat().st_size / 1e6, 1))
            break
except Exception as e:  # noqa: BLE001
    step("ccache.tar.err", err=repr(e)[:150])
