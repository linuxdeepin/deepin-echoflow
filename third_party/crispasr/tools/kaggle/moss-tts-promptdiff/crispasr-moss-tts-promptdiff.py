# CrispASR — MOSS-TTS prompt-token diff (#249)
#
# The greedy code-parity failed from frame 0 — a classic "gate the input first"
# signal. This kernel diffs the PROMPT itself: the HF processor's col-0 text ids
# (input_ids[:,:,0]) vs the C++ mt_build_prompt_text + tokenizer
# (moss_tts_debug_prompt_ids), for the same text. If they differ, the prompt
# template / normalization / special-token handling is the divergence — fix
# mt_build_prompt_text (and add normalize_tts_text) and the greedy codes align.
#
# Fast: the reference side only needs the processor/tokenizer (NO 8B load, no
# generation); the C++ side loads the Q4_K backbone for its vocab. ccache under
# /kaggle/temp (kaggle_usage #22).

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

_T0 = time.time()
PROGRESS = WORK / "progress.txt"


def log(m):
    line = f"[{round(time.time() - _T0, 1)}s] {m}"
    print(line, flush=True)
    with open(PROGRESS, "a") as f:
        f.write(line + "\n")


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


def cpp_prompt_ids(lib_path, backbone, text):
    lib = ctypes.CDLL(lib_path)
    lib.moss_tts_context_default_params.restype = CtxParams
    lib.moss_tts_init_from_file.restype = ctypes.c_void_p
    lib.moss_tts_init_from_file.argtypes = [ctypes.c_char_p, CtxParams]
    lib.moss_tts_debug_prompt_ids.restype = ctypes.POINTER(ctypes.c_int32)
    lib.moss_tts_debug_prompt_ids.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                              ctypes.POINTER(SynthParams), ctypes.POINTER(ctypes.c_int)]
    lib.moss_tts_token_text.restype = ctypes.c_char_p
    lib.moss_tts_token_text.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.moss_tts_free.argtypes = [ctypes.c_void_p]

    cp = lib.moss_tts_context_default_params()
    cp.use_gpu = True
    ctx = lib.moss_tts_init_from_file(backbone.encode(), cp)
    if not ctx:
        raise RuntimeError("init returned null")
    sp = SynthParams()
    sp.max_new_tokens = 4096
    sp.text_temperature = 0.0
    sp.audio_temperature = 0.0
    n = ctypes.c_int(0)
    ptr = lib.moss_tts_debug_prompt_ids(ctx, text.encode(), ctypes.byref(sp), ctypes.byref(n))
    ids = [int(ptr[i]) for i in range(n.value)] if ptr and n.value > 0 else []

    def tok(i):
        s = lib.moss_tts_token_text(ctx, int(i))
        return s.decode("utf-8", "replace") if s else "?"
    toks = [tok(i) for i in ids]
    lib.moss_tts_free(ctx)
    return ids, toks


def main():
    summary = {"text": TEXT}
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
    with kh.build_heartbeat("promptdiff build"):
        kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-lib "
                            f"-j{kh.safe_build_jobs(gpu=True)}")
    import glob
    lib_path = glob.glob(str(BUILD / "src" / "libcrispasr.so*"))[0]
    os.environ["LD_LIBRARY_PATH"] = f"{BUILD/'src'}:{os.environ.get('LD_LIBRARY_PATH','')}"

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import hf_hub_download
    tok_hf = kh.resolve_hf_token()
    q4k = hf_hub_download(HF_GGUF, "moss-tts-v1.5-q4_k.gguf", local_dir=str(MODELS), token=tok_hf)
    log("downloaded Q4_K backbone")

    # ── reference prompt ids (processor only; no 8B) ──
    ref_ids, ref_toks = [], []
    try:
        from transformers import AutoProcessor
        if tok_hf:  # the custom processor's from_pretrained rejects token= — use env
            os.environ.setdefault("HF_TOKEN", tok_hf)
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", tok_hf)
        proc = AutoProcessor.from_pretrained(HF_MODEL, trust_remote_code=True)
        msg = proc.build_user_message(text=TEXT, reference=None, instruction=None, tokens=None,
                                      quality=None, sound_event=None, ambient_sound=None, language=None)
        batch = proc(conversations=[msg], mode="generation", apply_chat_template=True)
        import numpy as np
        iid = np.asarray(batch["input_ids"])
        iid = np.squeeze(iid)
        # (seq, 1+n_vq) -> col 0
        col0 = iid[:, 0] if iid.ndim == 2 else iid
        ref_ids = [int(x) for x in col0.tolist()]
        # decode each id
        t = getattr(proc, "tokenizer", None)
        ref_toks = [t.decode([i]) if t else str(i) for i in ref_ids]
        log(f"ref prompt ids: {len(ref_ids)}")
    except Exception as e:  # noqa: BLE001
        log(f"ref prompt failed: {e}\n{traceback.format_exc()}")
        summary["ref_error"] = str(e)

    # ── C++ prompt ids ──
    cpp_ids, cpp_toks = cpp_prompt_ids(lib_path, q4k, TEXT)
    log(f"cpp prompt ids: {len(cpp_ids)}")

    # ── diff ──
    summary["ref_len"] = len(ref_ids)
    summary["cpp_len"] = len(cpp_ids)
    if ref_ids and cpp_ids:
        L = min(len(ref_ids), len(cpp_ids))
        first = next((i for i in range(L) if ref_ids[i] != cpp_ids[i]), None)
        summary["first_mismatch"] = first
        summary["len_equal"] = (len(ref_ids) == len(cpp_ids))
        summary["prefix_equal_len"] = first if first is not None else L
        summary["ids_identical"] = (ref_ids == cpp_ids)
        lo = max(0, (first or 0) - 4)
        hi = min(L, (first if first is not None else L) + 8)
        print("\n== window around first mismatch ==")
        for i in range(lo, hi):
            r = f"{ref_ids[i]}:{ref_toks[i]!r}" if i < len(ref_ids) else "-"
            c = f"{cpp_ids[i]}:{cpp_toks[i]!r}" if i < len(cpp_ids) else "-"
            mark = "  <==" if (i < L and ref_ids[i] != cpp_ids[i]) else ""
            print(f"  [{i:3d}] ref={r:28s} cpp={c:28s}{mark}")
        # full decoded prompt strings for eyeballing
        summary["ref_prompt_head"] = "".join(ref_toks)[:400]
        summary["cpp_prompt_head"] = "".join(cpp_toks)[:400]

    (WORK / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 60 + "\n" + json.dumps(summary, indent=2) + "\n" + "=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        log(f"FATAL: {e}\n{traceback.format_exc()}")
        sys.exit(1)
