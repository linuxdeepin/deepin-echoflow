#!/usr/bin/env python3
"""
Issue #398 — htdemucs GPU path aborts in ggml-cuda/binbcast.cu:293 on the
first separation. PROOF kernel, on a real CUDA box (Kaggle P100/T4).

Root cause: the F16 GGUF stores the DConv GroupNorm affine weights
(`*.dconv.layers.N.4.weight`, 384/768 elements) as F16 — the converter's
keep-F32 rules match ".norm"/".bias"/"scale"/size<256 and these are none of
those. The ggml graph feeds them as src1 of broadcast ggml_mul/ggml_add;
CUDA's binbcast picks the src1 element type from the F32 src0/dst branch and
asserts `nb10 % sizeof(float) == 0` on the 2-byte F16 stride. Quantized files
inherit the same F16 tensors verbatim (q8_0 crashed identically for the
reporter). CPU converts per element and never notices.

Fix under test (branch fix/issue-398-402): htd_bcast_f32() casts any non-F32
weight to F32 in-graph before every broadcast site; the pre-fix graph stays
reachable via CRISPASR_HTDEMUCS_NO_BCAST_CAST=1 for bisection — which is
exactly how this kernel reproduces the crash from a single build:

  A) repro   : GGML=1 GPU=1 NO_BCAST_CAST=1, f16  → expect hard abort with
               the binbcast assert (reproduces the issue verbatim)
  B) fix f16 : GGML=1 GPU=1                        → expect 4 stems written
  C) fix q8_0: GGML=1 GPU=1                        → expect 4 stems written
  D) cpu ref : no GGML env (legacy CPU/BLAS path)  → known-good reference
  E) parity  : per-stem cosine AND |gpu|/|cpu| magnitude ratio vs D
               (cosine is scale-blind — HARD RULE #2b — so both are checked)

Input mirrors the report: a 44.1 kHz stereo WAV (~11 s, resampled from the
repo's samples/jfk.wav). Results land in /kaggle/working/issue398_results.json.
"""
import os, sys, subprocess, json, wave, array, math
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp/i398"); TMP.mkdir(parents=True, exist_ok=True)
MODELS = Path("/kaggle/temp/models"); MODELS.mkdir(parents=True, exist_ok=True)
RESULTS = WORK / "issue398_results.json"

HERE = Path(__file__).resolve().parent
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
BRANCH = "main"  # proof runs used fix/issue-398-402 (merged); main keeps re-runs working
CLONE = Path("/kaggle/temp/CrispASR")
if not CLONE.exists():
    try:
        subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                        "--recurse-submodules", CRISPASR_URL, str(CLONE)],
                       check=True, timeout=1200)
    except Exception as e:
        print(f"clone failed: {e}", flush=True)
sys.path.insert(0, str(CLONE / "tools" / "kaggle") if CLONE.exists() else str(HERE))
import kaggle_harness as kh
kh.init_progress()

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=False)
HF_TOKEN = kh.resolve_hf_token()
kh.step("hf.token", present=bool(HF_TOKEN))
from huggingface_hub import hf_hub_download

HTD_REPO = "cstr/htdemucs-GGUF"
STEMS = ("drums", "bass", "other", "vocals")


def sh(cmd, timeout=None, env=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          timeout=timeout, env=env)


def build_crispasr():
    kh.install_build_toolchain()
    if not (CLONE / "CMakeLists.txt").exists():
        raise RuntimeError("repo clone missing")
    if not (CLONE / "ggml" / "CMakeLists.txt").exists():
        sh(f"cd {CLONE} && git submodule update --init ggml", timeout=900)
    arch = kh.detect_cuda_arch()
    flags = (["-DCMAKE_BUILD_TYPE=Release", "-DCRISPASR_OPUS_FETCH=ON"]
             + kh.cuda_build_flags(arch) + kh.cache_and_link_flags())
    kh.step("build.configure", arch=arch)
    r = sh(f"cd {CLONE} && cmake -G Ninja -B build " + " ".join(flags), timeout=1200)
    if r.returncode != 0:
        kh.step("build.configure_FAILED", stderr=r.stderr[-2000:])
        raise RuntimeError("cmake configure failed")
    jobs = kh.safe_build_jobs(gpu=True)
    with kh.build_heartbeat("build.ninja", interval_s=30):
        kh.sh_with_progress(f"cmake --build build -j{jobs} --target crispasr-cli", cwd=str(CLONE))
    binp = CLONE / "build" / "bin" / "crispasr"
    if not binp.exists():
        raise RuntimeError("crispasr binary not produced")
    os.environ["LD_LIBRARY_PATH"] = str(binp.parent) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    kh.step("build.ready", path=str(binp))
    try:
        kh.export_ccache_tar()
    except Exception:
        pass
    return str(binp)


def make_test_wav():
    """44.1 kHz stereo, mirroring the reporter's input shape."""
    src = CLONE / "samples" / "jfk.wav"
    w = wave.open(str(src), "rb")
    sr = w.getframerate()
    a = array.array("h"); a.frombytes(w.readframes(w.getnframes())); w.close()
    x = [s / 32768.0 for s in a]
    # linear resample sr -> 44100 (fidelity is irrelevant for a crash/parity
    # proof; the graph shapes only depend on length/rate)
    tgt_sr = 44100
    n_out = int(len(x) * tgt_sr / sr)
    y = array.array("h")
    for i in range(n_out):
        t = i * (len(x) - 1) / max(1, n_out - 1)
        i0 = int(t); f = t - i0
        v = x[i0] * (1 - f) + x[min(i0 + 1, len(x) - 1)] * f
        s = max(-32768, min(32767, int(v * 32767)))
        y.append(s); y.append(s)  # stereo
    p = TMP / "test-44k.wav"
    o = wave.open(str(p), "wb"); o.setnchannels(2); o.setsampwidth(2)
    o.setframerate(tgt_sr); o.writeframes(y.tobytes()); o.close()
    kh.step("input.ready", path=str(p), seconds=round(n_out / tgt_sr, 2))
    return p


def read_wav_f32(path):
    w = wave.open(str(path), "rb")
    a = array.array("h"); a.frombytes(w.readframes(w.getnframes())); w.close()
    return [s / 32768.0 for s in a]


def stem_paths(inp):
    stem_base = str(inp)[: -len(".wav")]
    return {s: Path(f"{stem_base}_{s}.wav") for s in STEMS}


def run_separate(binp, model, inp, tag, env_extra, timeout=1800):
    # each arm gets its own copy of the input so stem outputs don't collide
    arm_inp = TMP / f"{tag}.wav"
    arm_inp.write_bytes(Path(inp).read_bytes())
    env = dict(os.environ); env.update(env_extra)
    cmd = f"{binp} --separate --backend htdemucs -m {model} -f {arm_inp}"
    with kh.build_heartbeat(f"sep.{tag}", interval_s=30):
        try:
            r = sh(cmd, timeout=timeout, env=env)
            rc, out = r.returncode, (r.stderr or "") + (r.stdout or "")
        except subprocess.TimeoutExpired:
            rc, out = -99, "TIMEOUT"
    stems = stem_paths(arm_inp)
    written = {s: p.exists() for s, p in stems.items()}
    res = dict(tag=tag, rc=rc, stems_written=sum(written.values()),
               binbcast_assert=("binbcast" in out and "GGML_ASSERT" in out) or "nb10 % sizeof" in out,
               tail=out[-600:])
    kh.step(f"sep.{tag}", rc=rc, stems=res["stems_written"], assert_hit=res["binbcast_assert"])
    return res, stems


def cosine(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return 0.0, 0.0, 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(v * v for v in a[:n])); nb = math.sqrt(sum(v * v for v in b[:n]))
    if na == 0 or nb == 0:
        return 0.0, na, nb
    return dot / (na * nb), na, nb


def main():
    binp = build_crispasr()
    for f in ("htdemucs-f16.gguf", "htdemucs-q8_0.gguf"):
        with kh.build_heartbeat(f"models.{f}", interval_s=30):
            hf_hub_download(repo_id=HTD_REPO, filename=f, local_dir=str(MODELS), token=HF_TOKEN or None)
    f16 = MODELS / "htdemucs-f16.gguf"
    q8 = MODELS / "htdemucs-q8_0.gguf"
    inp = make_test_wav()

    results = {"runs": {}, "parity": {}}
    GPU = {"CRISPASR_HTDEMUCS_GGML": "1", "CRISPASR_HTDEMUCS_GPU": "1"}

    # A) reproduce #398: fix gated OFF -> the pre-fix graph on CUDA
    repro, _ = run_separate(binp, f16, inp, "repro_nofix",
                            {**GPU, "CRISPASR_HTDEMUCS_NO_BCAST_CAST": "1"}, timeout=900)
    results["runs"]["repro_nofix"] = repro
    RESULTS.write_text(json.dumps(results, indent=2))

    # B) the fix, f16
    fix16, stems16 = run_separate(binp, f16, inp, "fix_f16", GPU, timeout=1800)
    results["runs"]["fix_f16"] = fix16
    RESULTS.write_text(json.dumps(results, indent=2))

    # C) the fix, q8_0 (reporter's second crash config)
    fixq8, _ = run_separate(binp, q8, inp, "fix_q8_0", GPU, timeout=1800)
    results["runs"]["fix_q8_0"] = fixq8
    RESULTS.write_text(json.dumps(results, indent=2))

    # D) CPU/BLAS legacy path (known-good reference, no GGML env)
    cpu, stems_cpu = run_separate(binp, f16, inp, "cpu_ref", {}, timeout=3000)
    results["runs"]["cpu_ref"] = cpu

    # E) parity GPU-vs-CPU per stem: cosine + magnitude ratio (HARD RULE #2b)
    for s in STEMS:
        if stems16[s].exists() and stems_cpu[s].exists():
            a = read_wav_f32(stems16[s]); b = read_wav_f32(stems_cpu[s])
            cos, na, nb = cosine(a, b)
            results["parity"][s] = dict(cos=round(cos, 6),
                                        mag_gpu=round(na, 3), mag_cpu=round(nb, 3),
                                        mag_ratio=round(na / nb, 4) if nb else None)
    kh.step("parity", **results["parity"])

    verdict = dict(
        repro_reproduced=bool(repro["binbcast_assert"] or (repro["rc"] != 0 and repro["stems_written"] == 0)),
        fix_f16_ok=bool(fix16["rc"] == 0 and fix16["stems_written"] == 4),
        fix_q8_ok=bool(fixq8["rc"] == 0 and fixq8["stems_written"] == 4),
        cpu_ref_ok=bool(cpu["rc"] == 0 and cpu["stems_written"] == 4),
        parity_ok=bool(results["parity"] and
                       all(v["cos"] > 0.98 and v["mag_ratio"] and 0.9 < v["mag_ratio"] < 1.1
                           for v in results["parity"].values())),
    )
    results["verdict"] = verdict
    RESULTS.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)
    kh.step("DONE", **verdict)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        kh.step("FATAL", err=str(e)[:300])
        traceback.print_exc()
        sys.exit(1)
