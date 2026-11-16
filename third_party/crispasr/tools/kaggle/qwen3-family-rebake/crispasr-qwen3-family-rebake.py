"""
CrispASR — qwen3-family GGUF rebake (Q8_0 audio-tower floor) + CUDA validation

Two jobs in one CUDA build (#218 follow-ups):

A. CUDA validation of this week's changes (everything so far was verified on
   Metal/CPU only):
     - qwen3-asr 0.6b (fixed GGUF): un-chunked 145 s single pass, full
       attention AND the new CRISP_AUDIO_WINDOWED_ATTN=1 path.
     - glm-asr (merges-baked GGUF): jfk + un-chunked 145 s multi-window
       single pass.
   Pass criteria: transcript reaches the clip's final sentence, no
   degenerate loops (raw decode via CRISPASR_NGRAM_LOOPFIX_OFF=1).

B. Rebake the SIBLING qwen3-family repos whose q4_k files were quantized
   before the Q8_0 audio-tower floor (crispasr-quantize on this HEAD):
     - cstr/qwen3-asr-1.7b-GGUF        (source: f16)
     - cstr/qwen3-asr-1.7b-ja-anime-GGUF (source: q8_0 — no f16 published;
       q8_0 is ~lossless so q8->f32->q4 ≈ f16->q4)
     - cstr/mega-asr-GGUF              (source: f16; also rebake the
       -imatrix variants with the repo's published imatrix)
   Each rebake is validated (jfk transcript + 145 s single-pass completion)
   before upload. Models processed sequentially with deletes in between —
   peak disk ≈ 7 GB under /tmp (70 GB writable layer).
"""

import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path("/kaggle/working")
REPO = Path("/tmp/CrispASR")
BUILD = Path("/tmp/build")
MODELS = Path("/tmp/models")
AUDIO_DIR = ROOT / "audio"
OUT_DIR = ROOT / "results"
for d in (MODELS, AUDIO_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
if not REPO.exists():
    subprocess.run(
        f"git clone --depth 1 --branch {CRISPASR_REF} "
        f"https://github.com/CrispStrobe/CrispASR.git {REPO}",
        shell=True, check=True,
    )
if not (REPO / "ggml" / "CMakeLists.txt").exists():
    subprocess.run(f"git -C {REPO} submodule update --init --recursive --depth 1",
                   shell=True, check=True)
sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress(progress_path=str(ROOT / "progress.jsonl"))
kh.resolve_hf_token()
kh.step("script.start")
sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
kh.step("clone.done", sha=sha)

# ── Build (CUDA) ─────────────────────────────────────────────────────────
kh.step("build.begin")
BUILD.mkdir(exist_ok=True)
kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
cmake_cmd = " ".join(
    [f"cmake {REPO} -B{BUILD} -GNinja", "-DCMAKE_BUILD_TYPE=Release",
     "-DCRISPASR_BUILD_TESTS=OFF"]
    + kh.cuda_build_flags(arch) + kh.cache_and_link_flags()
)
with kh.build_heartbeat("cmake-configure"):
    kh.sh_with_progress(cmake_cmd)
with kh.build_heartbeat("cmake-build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli crispasr-quantize -- "
        f"-j{kh.safe_build_jobs(gpu=True)}"
    )
CRISPASR = BUILD / "bin" / "crispasr"
QUANTIZE = BUILD / "bin" / "crispasr-quantize"
assert CRISPASR.is_file() and QUANTIZE.is_file()
kh.step("build.done", cuda_arch=arch)

# ── Audio ────────────────────────────────────────────────────────────────
WAV_ZIP = AUDIO_DIR / "t32-145s.wav.zip"
WAV = AUDIO_DIR / "t32-145s.wav"
if not WAV.exists():
    kh.sh_with_progress(
        f"curl -sL -o {WAV_ZIP} "
        "https://github.com/user-attachments/files/29652411/t32-145s.wav.zip")
    with zipfile.ZipFile(WAV_ZIP) as z:
        z.extractall(AUDIO_DIR)
JFK = REPO / "samples" / "jfk.wav"

kh.sh_with_progress("pip install -q huggingface_hub hf_transfer")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402


def loop_metrics(text: str) -> dict:
    words = [w.strip(".,!?;:—-\"'").lower() for w in text.split() if w.strip()]
    max_uni, cur, prev = 0, 0, None
    for w in words:
        cur = cur + 1 if w == prev else 1
        prev = w
        max_uni = max(max_uni, cur)
    low = text.lower()
    return {
        "chars": len(text), "max_unigram_run": max_uni,
        "reaches_final_sentence": ("back road" in low) or ("follow" in low[-200:]),
    }


def run_cli(model: Path, backend: str, wav: Path, tag: str, extra=None, env_extra=None) -> dict:
    # NOTE: tags may contain dots (model filenames) — the CLI writes
    # "<of>.txt" by string append, so read the SAME name back; Path
    # .with_suffix() would clobber everything after the first dot (v1 bug:
    # every phase-B validation read a nonexistent file → chars=0 → uploads
    # skipped despite good transcripts).
    tag = tag.replace(".gguf", "")
    out_stem = OUT_DIR / tag
    cmd = [str(CRISPASR), "-m", str(model), "--backend", backend, "-f", str(wav),
           "--no-timestamps", "-np", "-otxt", "-of", str(out_stem)] + (extra or [])
    env = dict(os.environ)
    env["CRISPASR_NGRAM_LOOPFIX_OFF"] = "1"
    env.update(env_extra or {})
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
    txt = Path(str(out_stem) + ".txt")
    text = txt.read_text(errors="replace") if txt.is_file() else ""
    m = {"rc": r.returncode, "wall_s": round(time.time() - t0, 1), **loop_metrics(text)}
    kh.step(f"run.{tag}", **m)
    print(f"--- {tag}: {m}")
    print(text[:600], flush=True)
    return m


# ── Phase A: CUDA validation ─────────────────────────────────────────────
kh.step("phaseA.begin")
q06 = Path(hf_hub_download("cstr/qwen3-asr-0.6b-GGUF", "qwen3-asr-0.6b-q4_k.gguf", local_dir=str(MODELS)))
glm = Path(hf_hub_download("cstr/glm-asr-nano-GGUF", "glm-asr-nano-q4_k.gguf", local_dir=str(MODELS)))

val = {}
val["qwen3_full"] = run_cli(q06, "qwen3", WAV, "qwen3-t32-full",
                            extra=["--language", "en", "--chunk-seconds", "0"])
val["qwen3_windowed"] = run_cli(q06, "qwen3", WAV, "qwen3-t32-windowed",
                                extra=["--language", "en", "--chunk-seconds", "0"],
                                env_extra={"CRISP_AUDIO_WINDOWED_ATTN": "1"})
val["glm_jfk"] = run_cli(glm, "glm-asr", JFK, "glm-jfk")
val["glm_t32"] = run_cli(glm, "glm-asr", WAV, "glm-t32-singlepass",
                         extra=["--chunk-seconds", "0"])
kh.step("phaseA.done", **{k: v["reaches_final_sentence"] for k, v in val.items() if "t32" in k})
q06.unlink()
glm.unlink()

# ── Phase B: family rebake ───────────────────────────────────────────────
# Header-scouted 2026-07-10: cstr/qwen3-asr-1.7b-GGUF q4_k already has a
# Q8_0 audio tower (147 tensors) — NO rebake needed. ja-anime and mega-asr
# q4_k towers are Q4_K (147 tensors each) — rebake below.
FAMILY = [
    # (repo, source file, english_audio, [(target file, qtype, imatrix or None)])
    ("cstr/qwen3-asr-1.7b-ja-anime-GGUF", "qwen3-asr-1.7b-ja-anime-q8_0.gguf", False,
     [("qwen3-asr-1.7b-ja-anime-q4_k.gguf", "q4_k", None)]),
    ("cstr/mega-asr-GGUF", "mega-asr-1.7b-f16.gguf", True,
     [("mega-asr-1.7b-q4_k.gguf", "q4_k", None),
      ("mega-asr-1.7b-q4_k-imatrix.gguf", "q4_k", "mega-asr-1.7b-en-de.imatrix.gguf"),
      ("mega-asr-1.7b-q3_k-imatrix.gguf", "q3_k", "mega-asr-1.7b-en-de.imatrix.gguf")]),
]

api = HfApi()
summary = {}
for repo, src_name, english_audio, targets in FAMILY:
    kh.step(f"rebake.{repo}.begin")
    src = Path(hf_hub_download(repo, src_name, local_dir=str(MODELS)))
    results = {}
    for tgt_name, qtype, imx_name in targets:
        tgt = MODELS / tgt_name
        cmd = [str(QUANTIZE), str(src), str(tgt), qtype]
        if imx_name:
            imx = Path(hf_hub_download(repo, imx_name, local_dir=str(MODELS)))
            cmd += ["--imatrix", str(imx)]
        kh.step(f"quantize.{tgt_name}.begin")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            kh.step(f"quantize.{tgt_name}.FAILED", rc=r.returncode)
            print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
            results[tgt_name] = {"quantize_rc": r.returncode}
            continue
        kh.step(f"quantize.{tgt_name}.done", size_mib=tgt.stat().st_size >> 20)

        # Validate: jfk transcript + 145 s single-pass completion.
        backend = "mega-asr" if "mega" in tgt_name else "qwen3"
        vj = run_cli(tgt, backend, JFK, f"{tgt_name}-jfk", extra=["--language", "en"])
        vt = run_cli(tgt, backend, WAV, f"{tgt_name}-t32",
                     extra=["--language", "en", "--chunk-seconds", "0"])
        # The JA fine-tune may legitimately transcribe the EN clip
        # differently — for it only gate on "coherent, non-degenerate".
        if english_audio:
            ok = (vj["rc"] == 0 and vt["rc"] == 0 and vt["reaches_final_sentence"]
                  and vt["max_unigram_run"] < 8)
        else:
            ok = (vj["rc"] == 0 and vt["rc"] == 0 and vt["chars"] > 150
                  and vt["max_unigram_run"] < 8)
        results[tgt_name] = {"jfk": vj, "t32": vt, "upload": bool(ok)}
        if ok:
            kh.step(f"upload.{tgt_name}.begin")
            api.upload_file(path_or_fileobj=str(tgt), path_in_repo=tgt_name,
                            repo_id=repo, repo_type="model",
                            commit_message=f"rebake {tgt_name}: Q8_0 audio-tower floor (CrispASR #218, {sha[:8]})")
            kh.step(f"upload.{tgt_name}.done")
        else:
            kh.step(f"upload.{tgt_name}.SKIPPED_validation_failed")
        tgt.unlink(missing_ok=True)
    src.unlink(missing_ok=True)
    summary[repo] = results
    kh.step(f"rebake.{repo}.done")

(OUT_DIR / "summary.json").write_text(json.dumps({"validation": val, "rebake": summary}, indent=1, default=str))
print("=" * 72)
print(json.dumps({"validation": val, "rebake": summary}, indent=1, default=str))
kh._push_progress_to_hf(force=True)
kh.step("script.end")
