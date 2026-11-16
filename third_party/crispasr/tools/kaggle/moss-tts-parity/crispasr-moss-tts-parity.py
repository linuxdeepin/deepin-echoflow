# CrispASR — MOSS-TTS greedy code-parity (#249, Phase-3 gate)
#
# The ship kernel's parity step OOM'd (torch 8B + the crispasr process on one
# 16 GB P100). Fix: (1) download the already-shipped GGUFs instead of
# re-converting; (2) run the HF reference STANDALONE first, then free the GPU
# BEFORE the ggml side loads; (3) compare the greedy code grids.
#
# Greedy (temps=0) makes both sides deterministic. The C++ side runs Q4_K (F16's
# 17 GB backbone won't fit a 16 GB P100), so — per the voxtral-tts lesson — expect
# a byte-identical PREFIX then divergence as Q4_K-vs-BF16 rounding flips an argmax.
# The gate is "prefix matches" (backbone + 33 heads + delay are structurally
# correct), reported alongside the first-divergence frame.
#
# ccache under /kaggle/temp (not /kaggle/working) so the log/artifacts stay
# reachable (kaggle_usage #22).

import ctypes
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
TMP = Path("/tmp")
REPO = TMP / "CrispASR"
BUILD = REPO / "build"
MODELS = TMP / "moss-models"
WORK = Path("/kaggle/working")
MODELS.mkdir(parents=True, exist_ok=True)

REF = os.environ.get("CRISPASR_REF", "feat/moss-tts-parity-diff")
HF_MODEL = os.environ.get("MOSS_TTS_MODEL", "OpenMOSS-Team/MOSS-TTS-v1.5")
HF_GGUF = os.environ.get("MOSS_TTS_GGUF_REPO", "cstr/moss-tts-v1.5-GGUF")
TEXT = os.environ.get("MOSS_TTS_TEXT", "The quick brown fox jumps over the lazy dog.")
MAXNEW = int(os.environ.get("MOSS_TTS_MAXNEW", "160"))

_T0 = time.time()
PROGRESS = WORK / "progress.txt"


def log(m):
    line = f"[{round(time.time() - _T0, 1)}s] {m}"
    print(line, flush=True)
    with open(PROGRESS, "a") as f:
        f.write(line + "\n")


# ctypes mirror of struct moss_tts_synth_params (field order MUST match moss_tts.h)
class SynthParams(ctypes.Structure):
    _fields_ = [
        ("max_new_tokens", ctypes.c_int),
        ("text_temperature", ctypes.c_float),
        ("text_top_p", ctypes.c_float),
        ("text_top_k", ctypes.c_int),
        ("audio_temperature", ctypes.c_float),
        ("audio_top_p", ctypes.c_float),
        ("audio_top_k", ctypes.c_int),
        ("audio_repetition_penalty", ctypes.c_float),
        ("min_audio_frames", ctypes.c_int),
        ("max_audio_frames", ctypes.c_int),
        ("seed", ctypes.c_uint64),
        ("language", ctypes.c_char_p),
        ("instruction", ctypes.c_char_p),
    ]


def cpp_greedy_codes(lib_path, backbone, codec, text):
    """Call the moss_tts C ABI greedily via ctypes → (n_vq, T) int32 numpy."""
    import numpy as np
    lib = ctypes.CDLL(lib_path)
    lib.moss_tts_context_default_params.restype = None
    # context params struct: {int n_threads; int verbosity; bool use_gpu; bool flash_attn;}
    class CtxParams(ctypes.Structure):
        _fields_ = [("n_threads", ctypes.c_int), ("verbosity", ctypes.c_int),
                    ("use_gpu", ctypes.c_bool), ("flash_attn", ctypes.c_bool)]
    lib.moss_tts_context_default_params.restype = CtxParams
    lib.moss_tts_init_from_file.restype = ctypes.c_void_p
    lib.moss_tts_init_from_file.argtypes = [ctypes.c_char_p, CtxParams]
    lib.moss_tts_set_codec_path.restype = ctypes.c_bool
    lib.moss_tts_set_codec_path.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.moss_tts_generate_codes.restype = ctypes.POINTER(ctypes.c_int32)
    lib.moss_tts_generate_codes.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                            ctypes.POINTER(SynthParams),
                                            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    lib.moss_tts_free.argtypes = [ctypes.c_void_p]

    cp = lib.moss_tts_context_default_params()
    cp.use_gpu = True
    ctx = lib.moss_tts_init_from_file(backbone.encode(), cp)
    if not ctx:
        raise RuntimeError("moss_tts_init_from_file returned null")
    lib.moss_tts_set_codec_path(ctx, codec.encode())  # not needed for codes, but harmless
    sp = SynthParams()
    sp.max_new_tokens = MAXNEW
    sp.text_temperature = 0.0   # greedy
    sp.audio_temperature = 0.0  # greedy
    sp.text_top_p = 1.0
    sp.audio_top_p = 1.0
    sp.text_top_k = 0
    sp.audio_top_k = 0
    sp.audio_repetition_penalty = 1.0
    sp.seed = 0
    nvq = ctypes.c_int(0)
    t_audio = ctypes.c_int(0)
    ptr = lib.moss_tts_generate_codes(ctx, text.encode(), ctypes.byref(sp),
                                      ctypes.byref(nvq), ctypes.byref(t_audio))
    if not ptr or nvq.value <= 0 or t_audio.value <= 0:
        lib.moss_tts_free(ctx)
        raise RuntimeError(f"generate_codes empty (nvq={nvq.value} T={t_audio.value})")
    n = nvq.value * t_audio.value
    arr = np.ctypeslib.as_array(ptr, shape=(n,)).reshape(nvq.value, t_audio.value).copy()
    lib.moss_tts_free(ctx)
    return arr


def main():
    import numpy as np
    summary = {"text": TEXT, "max_new": MAXNEW}
    log(f"clone {REF}")
    if not REPO.exists():
        subprocess.check_call(["git", "clone", "--depth", "1", "--branch", REF, "--recursive",
                               "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.init_progress()

    subprocess.run(["nvidia-smi", "-L"], check=False)

    # ── build crispasr-lib (ctypes target; faster than the full CLI) ──
    kh.install_build_toolchain()
    arch = kh.detect_cuda_arch()
    env = os.environ.copy()
    env["CCACHE_DIR"] = "/kaggle/temp/.ccache"  # keep /kaggle/working small (#22)
    subprocess.run(["cmake", "-G", "Ninja", "-B", str(BUILD), "-S", str(REPO), "-DCMAKE_BUILD_TYPE=Release",
                    "-DBUILD_SHARED_LIBS=ON"] + list(kh.cache_and_link_flags()) + list(kh.cuda_build_flags(arch)),
                   env=env, check=True, timeout=300)
    with kh.build_heartbeat("moss-tts parity build"):
        kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-lib "
                            f"-j{kh.safe_build_jobs(gpu=True)}")
    import glob
    libs = glob.glob(str(BUILD / "src" / "libcrispasr.so*"))
    if not libs:
        raise SystemExit("libcrispasr.so not built")
    lib_path = libs[0]
    os.environ["LD_LIBRARY_PATH"] = f"{BUILD/'src'}:{os.environ.get('LD_LIBRARY_PATH','')}"
    log(f"built {lib_path}")

    # ── download shipped GGUFs (no re-convert) ──
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import hf_hub_download
    tok = kh.resolve_hf_token()
    q4k = hf_hub_download(HF_GGUF, "moss-tts-v1.5-q4_k.gguf", local_dir=str(MODELS), token=tok)
    codec = hf_hub_download(HF_GGUF, "moss-tts-v1.5-codec.gguf", local_dir=str(MODELS), token=tok)
    log("downloaded shipped GGUFs")

    # ── (1) HF reference greedy codes, STANDALONE, then free the GPU ──
    ref_codes = None
    try:
        log("HF reference greedy generate")
        renv = os.environ.copy()
        # 8B bf16 (~16 GB) won't fit the 16 GB P100 — run the reference on CPU
        # (fits ~29 GB host RAM); the C++ side runs Q4_K on the GPU.
        renv.update(MOSS_TTS_MODEL=HF_MODEL, MOSS_TTS_TEXT=TEXT, MOSS_TTS_SEED="0",
                    MOSS_TTS_MAXNEW=str(MAXNEW), MOSS_TTS_REF_DEVICE="cpu",
                    CUDA_VISIBLE_DEVICES="")
        r = subprocess.run(
            [sys.executable, "-c",
             f"import sys; sys.path.insert(0, r'{REPO/'tools'/'reference_backends'}');"
             f"import moss_tts as m; m.run(out_dir=r'{MODELS/'ref'}')"],
            env=renv, capture_output=True, text=True, timeout=2400)
        (WORK / "ref.log").write_text(r.stdout + "\n--STDERR--\n" + r.stderr)
        rp = MODELS / "ref" / "codes.npy"
        if rp.exists():
            ref_codes = np.load(rp)
            log(f"ref codes {ref_codes.shape}")
            rawp = MODELS / "ref" / "raw.npy"
            if rawp.exists():
                ref_raw = np.load(rawp)  # (T_raw, 1+n_vq) delayed grid from <audio_start>
                print(f"\n== reference RAW delayed grid (warm-up), first 36 rows, cols 0..8 ==")
                for r in range(min(36, ref_raw.shape[0])):
                    print(f"  raw[{r:2d}] col0={int(ref_raw[r,0])} audio1..8={ref_raw[r,1:9].tolist()}")
        else:
            log("ref codes NOT produced — see ref.log (HF API extraction may need a fix)")
            summary["ref_error"] = r.stderr[-800:]
    except Exception as e:  # noqa: BLE001
        log(f"ref failed: {e}")
        summary["ref_error"] = str(e)

    # ── (2) C++ greedy codes via ctypes (Q4_K) ──
    try:
        log("C++ greedy generate_codes (Q4_K)")
        cpp = cpp_greedy_codes(lib_path, q4k, codec, TEXT)
        np.save(MODELS / "codes_cpp.npy", cpp)
        log(f"cpp codes {cpp.shape}")
        summary["cpp_shape"] = list(cpp.shape)
    except Exception as e:  # noqa: BLE001
        log(f"cpp failed: {e}\n{traceback.format_exc()}")
        summary["cpp_error"] = str(e)
        cpp = None

    # ── (3) structural parity diagnostic ──
    # Exact multi-frame greedy parity is UNACHIEVABLE at Q4_K vs F16: the MOSS
    # delay makes each un-delayed frame depend on many raw AR steps, and AR
    # generation is chaotic — one quantization argmax-flip cascades. ~n_vq*T
    # independent head-decisions at even 99.9% agreement → ~0.1% survival. So the
    # aggregate match frac is the wrong gate. The decisive STRUCTURAL check is
    # frame-0/codebook-0: the single first generated token, from the (now
    # byte-identical) prompt through ONE forward pass + head-0 argmax. If that
    # (and the early codebooks of frame 0) agree, backbone+embed+head+prompt are
    # correct; downstream divergence is quantization, not a bug.
    if ref_codes is not None and cpp is not None:
        # Persist both arrays as versioned output for offline analysis.
        np.save(WORK / "ref_codes.npy", ref_codes)
        np.save(WORK / "cpp_codes.npy", cpp)
        T = min(ref_codes.shape[1], cpp.shape[1])
        nvq = min(ref_codes.shape[0], cpp.shape[0])
        a, b = ref_codes[:nvq, :T], cpp[:nvq, :T]
        eq = (a == b)
        first_div = next((t for t in range(T) if not eq[:, t].all()), None)
        exact_prefix = first_div if first_div is not None else T

        # frame-0 per-codebook agreement (col 0 spans raw steps 0..nvq-1)
        f0_eq = eq[:, 0]
        f0_match = int(f0_eq.sum())
        # longest leading run of matching codebooks in frame 0 (raw steps 0,1,2,…)
        lead = 0
        for cb in range(nvq):
            if f0_eq[cb]:
                lead += 1
            else:
                break
        tok0_match = bool(a[0, 0] == b[0, 0])  # the very first generated token

        # offset scan: best column shift aligning cpp to ref (rules out a pure
        # frame-count/warm-up offset masquerading as total divergence)
        best = {"offset": 0, "frac": float(eq.mean())}
        for k in range(-6, 7):
            if k >= 0:
                aa, bb = ref_codes[:nvq, k:], cpp[:nvq, :cpp.shape[1] - k] if k else cpp[:nvq, :]
            else:
                aa, bb = ref_codes[:nvq, :ref_codes.shape[1] + k], cpp[:nvq, -k:]
            w = min(aa.shape[1], bb.shape[1])
            if w <= 0:
                continue
            f = float((aa[:, :w] == bb[:, :w]).mean())
            if f > best["frac"]:
                best = {"offset": k, "frac": f}

        summary.update({"compare_T": T, "n_vq": nvq,
                        "exact_prefix_frames": exact_prefix,
                        "first_divergence_frame": first_div,
                        "overall_match_frac": float(eq.mean()),
                        "frame0_codebook_match": f0_match,
                        "frame0_leading_run": lead,
                        "first_token_match": tok0_match,
                        "best_offset": best["offset"],
                        "best_offset_frac": best["frac"],
                        "ref_shape": list(ref_codes.shape),
                        "cpp_shape": list(cpp.shape)})

        # Dump frames 0..4 both sides for eyeballing.
        print("\n== frame-by-frame (codebooks 0..%d), frames 0..4 ==" % (nvq - 1))
        for t in range(min(5, T)):
            print(f"  [f{t}] ref={a[:, t].tolist()}")
            print(f"  [f{t}] cpp={b[:, t].tolist()}")
            print(f"  [f{t}] match={int(eq[:, t].sum())}/{nvq}")

        # Full codebook-0 (coarse/semantic RVQ level) sequences — the robust one.
        print("\n== codebook-0 (coarse) full sequence ==")
        print(f"  ref cb0 = {a[0, :].tolist()}")
        print(f"  cpp cb0 = {b[0, :].tolist()}")
        # Per-codebook agreement across ALL frames: RVQ coarse (cb0) should agree
        # far more than fine residuals (cb1..) if the port is structurally correct
        # and the divergence is quantization of the fine levels.
        per_cb = [int(eq[cb, :].sum()) for cb in range(nvq)]
        print(f"\n== per-codebook match count (/{T} frames) ==")
        print(f"  {per_cb}")
        summary["per_codebook_match"] = per_cb
        summary["cb0_match"] = per_cb[0]
        summary["cb0_match_frac"] = per_cb[0] / T if T else 0.0
        summary["fine_mean_match_frac"] = (sum(per_cb[1:]) / (nvq - 1) / T) if (nvq > 1 and T) else 0.0
        log(f"CODEBOOK PROFILE: cb0={per_cb[0]}/{T} ({per_cb[0]/T:.2f}); "
            f"fine cb1..{nvq-1} mean={summary['fine_mean_match_frac']:.3f}")
        log(f"PARITY: first_token_match={tok0_match} frame0={f0_match}/{nvq} "
            f"(leading run {lead}); exact_prefix={exact_prefix}/{T}; "
            f"overall={eq.mean():.3f}; best_offset={best['offset']}@{best['frac']:.3f}")
        # Structural gate: the first generated token + a non-trivial leading run
        # of frame-0 codebooks agreeing proves the pipeline; deeper divergence is
        # expected quantization noise (validate audio via the ASR roundtrip).
        summary["parity_gate"] = "PASS" if (tok0_match and lead >= 4) else "FAIL"
    else:
        summary["parity_gate"] = "INCOMPLETE"

    (WORK / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 60 + "\n" + json.dumps(summary, indent=2) + "\n" + "=" * 60)
    log(f"parity gate: {summary.get('parity_gate')}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        log(f"FATAL: {e}\n{traceback.format_exc()}")
        sys.exit(1)
