# CrispASR — MOSS-TTS step-0 head-0 logit probe (#249)
#
# The greedy code-parity diverges from FRAME 0: the C++ (Q4_K) picks a different
# codebook-0 token at step 0 than the HF reference (BF16). Because the prompt is
# now byte-identical and the delay/gen-slot structure matches, this probe asks
# the decisive question CHEAPLY (no 8B reference, no 17 GB F16): given the same
# prompt, where does the reference's greedy pick (REF_CODE) rank in the C++
# head-0 logit distribution?
#   • REF_CODE is a close runner-up to the C++ argmax  => quantization argmax flip
#     (expected: Q4_K vs BF16 rounding; the audio is still correct — validated by
#     the ASR roundtrip). Exact greedy parity is then unachievable, not a bug.
#   • REF_CODE ranks far down / has a wildly different logit => a real numerics or
#     head-0 divergence worth fixing.
#
# Fast: Q4_K on the GPU, one prefill forward pass. ccache under /kaggle/temp (#22).

import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
TMP = Path("/tmp")
REPO = TMP / "CrispASR"
BUILD = REPO / "build"
MODELS = TMP / "moss-models"
WORK = Path("/kaggle/working")
MODELS.mkdir(parents=True, exist_ok=True)

REF = os.environ.get("CRISPASR_REF", "feat/moss-tts-parity-diff")
HF_GGUF = os.environ.get("MOSS_TTS_GGUF_REPO", "cstr/moss-tts-v1.5-GGUF")
TEXT = os.environ.get("MOSS_TTS_TEXT", "The quick brown fox jumps over the lazy dog.")
# The reference greedy picks at step 0 (codebook 0) and the C++ pick, from the
# parity run — check where each ranks in the C++ head-0 logits.
REF_CODE = int(os.environ.get("MOSS_TTS_REF_CODE", "143"))
CPP_CODE = int(os.environ.get("MOSS_TTS_CPP_CODE", "1021"))

_T0 = time.time()


def log(m):
    print(f"[{round(time.time() - _T0, 1)}s] {m}", flush=True)


class SynthParams(ctypes.Structure):
    _fields_ = [
        ("max_new_tokens", ctypes.c_int), ("text_temperature", ctypes.c_float),
        ("text_top_p", ctypes.c_float), ("text_top_k", ctypes.c_int),
        ("audio_temperature", ctypes.c_float), ("audio_top_p", ctypes.c_float),
        ("audio_top_k", ctypes.c_int), ("audio_repetition_penalty", ctypes.c_float),
        ("min_audio_frames", ctypes.c_int), ("max_audio_frames", ctypes.c_int),
        ("seed", ctypes.c_uint64), ("language", ctypes.c_char_p), ("instruction", ctypes.c_char_p),
    ]


class CtxParams(ctypes.Structure):
    _fields_ = [("n_threads", ctypes.c_int), ("verbosity", ctypes.c_int),
                ("use_gpu", ctypes.c_bool), ("flash_attn", ctypes.c_bool)]


def main():
    import numpy as np
    summary = {"text": TEXT, "ref_code": REF_CODE, "cpp_code": CPP_CODE}
    log(f"clone {REF}")
    if not REPO.exists():
        subprocess.check_call(["git", "clone", "--depth", "1", "--branch", REF, "--recursive",
                               "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.init_progress()

    kh.install_build_toolchain()
    arch = kh.detect_cuda_arch()
    env = os.environ.copy()
    env["CCACHE_DIR"] = "/kaggle/temp/.ccache"
    subprocess.run(["cmake", "-G", "Ninja", "-B", str(BUILD), "-S", str(REPO), "-DCMAKE_BUILD_TYPE=Release",
                    "-DBUILD_SHARED_LIBS=ON"] + list(kh.cache_and_link_flags()) + list(kh.cuda_build_flags(arch)),
                   env=env, check=True, timeout=300)
    with kh.build_heartbeat("logit0 build"):
        kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-lib "
                            f"-j{kh.safe_build_jobs(gpu=True)}")
    import glob
    lib_path = glob.glob(str(BUILD / "src" / "libcrispasr.so*"))[0]
    os.environ["LD_LIBRARY_PATH"] = f"{BUILD/'src'}:{os.environ.get('LD_LIBRARY_PATH','')}"
    log(f"built {lib_path}")

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import hf_hub_download
    tok = kh.resolve_hf_token()
    q4k = hf_hub_download(HF_GGUF, "moss-tts-v1.5-q4_k.gguf", local_dir=str(MODELS), token=tok)
    log("downloaded Q4_K backbone")

    lib = ctypes.CDLL(lib_path)
    lib.moss_tts_context_default_params.restype = CtxParams
    lib.moss_tts_init_from_file.restype = ctypes.c_void_p
    lib.moss_tts_init_from_file.argtypes = [ctypes.c_char_p, CtxParams]
    lib.moss_tts_debug_first_audio_logits.restype = ctypes.POINTER(ctypes.c_float)
    lib.moss_tts_debug_first_audio_logits.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                                      ctypes.POINTER(SynthParams), ctypes.c_int,
                                                      ctypes.POINTER(ctypes.c_int)]
    lib.moss_tts_free.argtypes = [ctypes.c_void_p]

    cp = lib.moss_tts_context_default_params()
    cp.use_gpu = True
    ctx = lib.moss_tts_init_from_file(q4k.encode(), cp)
    if not ctx:
        raise RuntimeError("init null")
    sp = SynthParams()
    sp.max_new_tokens = 4096
    sp.text_temperature = 0.0
    sp.audio_temperature = 0.0
    n = ctypes.c_int(0)
    ptr = lib.moss_tts_debug_first_audio_logits(ctx, TEXT.encode(), ctypes.byref(sp), 0, ctypes.byref(n))
    if not ptr or n.value <= 0:
        raise RuntimeError(f"logits empty (n={n.value})")
    logits = np.ctypeslib.as_array(ptr, shape=(n.value,)).copy()
    lib.moss_tts_free(ctx)

    order = np.argsort(-logits)  # descending
    rank_of = {int(c): int(np.where(order == c)[0][0]) for c in (REF_CODE, CPP_CODE)}
    top = [(int(order[i]), float(logits[order[i]])) for i in range(10)]
    argmax = int(order[0])

    print("\n== C++ Q4_K step-0 head-0 logits ==")
    print(f"  vocab(+pad) = {n.value}")
    print(f"  argmax(cpp) = {argmax}  logit={float(logits[argmax]):.4f}")
    print(f"  ref pick {REF_CODE}: logit={float(logits[REF_CODE]):.4f}  rank={rank_of[REF_CODE]}")
    print(f"  cpp pick {CPP_CODE}: logit={float(logits[CPP_CODE]):.4f}  rank={rank_of[CPP_CODE]}")
    print(f"  gap(cpp_argmax - ref_pick) = {float(logits[argmax] - logits[REF_CODE]):.4f}")
    print("  top-10 (code, logit):")
    for c, lg in top:
        print(f"    {c:5d}  {lg:.4f}")

    summary.update({
        "vocab_full": int(n.value), "argmax_cpp": argmax,
        "argmax_logit": float(logits[argmax]),
        "ref_pick_logit": float(logits[REF_CODE]), "ref_pick_rank": rank_of[REF_CODE],
        "cpp_pick_logit": float(logits[CPP_CODE]), "cpp_pick_rank": rank_of[CPP_CODE],
        "argmax_minus_refpick": float(logits[argmax] - logits[REF_CODE]),
        "top10": top,
    })
    # Interpretation gate: ref pick a near-tie runner-up => quantization flip.
    summary["verdict"] = ("QUANT_FLIP_LIKELY" if rank_of[REF_CODE] <= 8
                          and abs(summary["argmax_minus_refpick"]) < 1.0 else "INVESTIGATE")
    (WORK / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 60 + "\n" + json.dumps(summary, indent=2) + "\n" + "=" * 60)


if __name__ == "__main__":
    main()
