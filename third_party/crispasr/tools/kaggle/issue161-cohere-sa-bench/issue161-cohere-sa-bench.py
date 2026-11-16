#!/usr/bin/env python3
"""CrispASR — issue #161 cohere self-attention A/B bench.

A) Default (flash_attn_ext SA)
B) CRISPASR_COHERE_LEGACY_SA=1 (manual mul_mat SA, pre-v0.7 path)
"""
import json, os, re, subprocess, sys, time, traceback
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
for s in (sys.stdout, sys.stderr):
    try: s.reconfigure(line_buffering=True)
    except Exception: pass

WORK = Path("/kaggle/working")
CRASH = WORK / "crash.txt"

def main():
    REPO = WORK / "CrispASR"
    BUILD = REPO / "build"
    REF = os.environ.get("CRISPASR_REF", "main")
    URL = "https://github.com/CrispStrobe/CrispASR.git"
    N_RUNS = 5
    MODEL = "cohere-transcribe-q4_k.gguf"

    # ── early progress ────────────────────────────────────────────────
    progress = WORK / "progress.jsonl"
    t0 = time.time()
    def step(name, **kv):
        rec = {"t": round(time.time()-t0, 2), "step": name, **kv}
        print(f"[step] {json.dumps(rec)}", flush=True)
        with open(progress, "a") as f: f.write(json.dumps(rec)+"\n")

    step("start")

    # ── clone ─────────────────────────────────────────────────────────
    step("clone")
    if not REPO.exists():
        subprocess.check_call(["git","clone","--depth","1","-b",REF,URL,str(REPO)])
    else:
        subprocess.check_call(["git","fetch","--depth","1","origin",REF], cwd=str(REPO))
        subprocess.check_call(["git","checkout","FETCH_HEAD"], cwd=str(REPO))

    sys.path.insert(0, str(REPO/"tools"/"kaggle"))
    import kaggle_harness as kh
    kh.init_progress()

    # ── build ─────────────────────────────────────────────────────────
    kh.step("toolchain")
    kh.install_build_toolchain()

    kh.step("configure")
    arch = kh.detect_cuda_arch()
    flags = kh.cuda_build_flags(arch) + kh.cache_and_link_flags()
    with kh.build_heartbeat("cmake.configure"):
        kh.sh(f"cmake -S {REPO} -B {BUILD} -G Ninja -DCMAKE_BUILD_TYPE=Release "
               f"-DCRISPASR_OPUS=OFF -DCRISPASR_AMR=OFF "
               + " ".join(flags))

    kh.step("build")
    with kh.build_heartbeat("cmake.build"):
        kh.sh_with_progress(
            f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli "
            f"-j{kh.safe_build_jobs(gpu=True)}")

    CLI = BUILD / "bin" / "crispasr"
    assert CLI.exists(), f"CLI not at {CLI}"
    kh.step("build_done")

    # ── assets ────────────────────────────────────────────────────────
    kh.step("download_model")
    MODEL_DIR = WORK / "models"; MODEL_DIR.mkdir(exist_ok=True)
    MODEL_PATH = MODEL_DIR / MODEL
    if not MODEL_PATH.exists():
        subprocess.check_call(["wget","-q","--show-progress",
            f"https://huggingface.co/cstr/cohere-transcribe-03-2026-GGUF/resolve/main/{MODEL}",
            "-O", str(MODEL_PATH)])

    AUDIO = REPO / "samples" / "jfk.wav"
    assert AUDIO.exists(), f"Audio not at {AUDIO}"

    kh.step("gpu_info")
    subprocess.run(["nvidia-smi"], check=False)

    # ── bench ─────────────────────────────────────────────────────────
    def bench(label, extra_env=None):
        env = dict(os.environ, COHERE_BENCH="1", COHERE_GAPS="1")
        if extra_env: env.update(extra_env)
        times, last = [], ""
        for i in range(N_RUNS + 1):
            t = time.time()
            r = subprocess.run(
                [str(CLI),"-m",str(MODEL_PATH),"-f",str(AUDIO),
                 "--backend","cohere","-t","4","-np","-l","en"],
                env=env, timeout=120,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            dt = time.time() - t
            tag = "warmup" if i==0 else f"run {i}"
            print(f"  [{label}] {tag}: {dt:.3f}s rc={r.returncode}", flush=True)
            if i > 0: times.append(dt); last = r.stdout
        return times, last

    kh.step("bench_A")
    print("\n=== A) flash_attn_ext (default) ===", flush=True)
    ta, la = bench("flash_attn")

    kh.step("bench_B")
    print("\n=== B) legacy mul_mat SA ===", flush=True)
    tb, lb = bench("legacy_sa", {"CRISPASR_COHERE_LEGACY_SA": "1"})

    # ── report ────────────────────────────────────────────────────────
    kh.step("report")
    def parse(txt):
        kv = {}
        for line in txt.split("\n"):
            m = re.match(r"\s*cohere_bench:\s+(.+?)\s+([\d.]+)\s+ms", line)
            if m: kv[f"bench:{m.group(1).strip()}"] = float(m.group(2))
            for pat, key in [
                (r"cohere:\s+total wall\s+([\d.]+)", "total_wall_ms"),
                (r"cohere:\s+enc compute\s+([\d.]+)", "enc_compute_ms"),
                (r"cohere:\s+dec compute\s+([\d.]+)", "dec_compute_ms"),
                (r"cohere:\s+cross-kv copy\s+([\d.]+)", "cross_kv_ms"),
                (r"cohere:\s+UNACCOUNTED\s+([-\d.]+)", "unaccounted_ms"),
                (r"cohere:\s+cross-kv readback\s*([\d.]+)", "cross_kv_readback_ms"),
                (r"cohere:\s+sched reserve\s+([\d.]+)", "sched_reserve_ms"),
            ]:
                m2 = re.search(pat, line)
                if m2: kv[key] = float(m2.group(1))
        return kv

    pa, pb = parse(la), parse(lb)
    ma = sum(ta)/len(ta) if ta else 0
    mb = sum(tb)/len(tb) if tb else 0

    print("\n" + "="*72)
    print("ISSUE #161 — COHERE SELF-ATTENTION A/B BENCH")
    print("="*72)
    print(f"Audio: jfk.wav | Model: {MODEL} | Runs: {N_RUNS}")
    print(f"\n{'Config':<35} {'Mean':>8} {'Min':>8} {'Max':>8}")
    print("-"*65)
    if ta: print(f"{'A) flash_attn_ext (default)':<35} {ma:>7.3f}s {min(ta):>7.3f}s {max(ta):>7.3f}s")
    if tb: print(f"{'B) legacy mul_mat SA':<35} {mb:>7.3f}s {min(tb):>7.3f}s {max(tb):>7.3f}s")
    if ma>0 and mb>0:
        d = (ma-mb)/mb*100
        print(f"\nDelta: {d:+.1f}%  — {'B (legacy)' if ma>mb else 'A (flash_attn)'} is faster")

    for lbl, p in [("A) flash_attn_ext", pa), ("B) legacy SA", pb)]:
        if p:
            print(f"\n  {lbl} breakdown:")
            for k in sorted(p): print(f"    {k:<30} {p[k]:>10.1f} ms")

    print(f"\nPer-run (s): A={[round(t,3) for t in ta]}  B={[round(t,3) for t in tb]}")

    for lbl, log in [("A", la), ("B", lb)]:
        print(f"\n--- {lbl} cohere stderr ---")
        for line in log.split("\n"):
            if "cohere" in line.lower(): print(f"  {line}")

    with open(WORK/"issue161_results.json","w") as f:
        json.dump({"model":MODEL,"n_runs":N_RUNS,
                   "flash_attn":{"times":ta,"mean":ma,"perf":pa},
                   "legacy_sa":{"times":tb,"mean":mb,"perf":pb}}, f, indent=2)

    kh.step("done", mean_a=round(ma,3), mean_b=round(mb,3))


# ── top-level crash guard ─────────────────────────────────────────────
try:
    main()
except Exception:
    tb = traceback.format_exc()
    print(f"\n!!! CRASH !!!\n{tb}", flush=True)
    try:
        with open(CRASH, "w") as f: f.write(tb)
    except Exception:
        pass
    sys.exit(1)
