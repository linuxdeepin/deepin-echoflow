#!/usr/bin/env python3
"""Cohere Transcribe Arabic — end-to-end verification on Kaggle GPU.

Answers, with the CORRECT (upstream) weights:
  1. Does the C++ runtime (crispasr) transcribe real Arabic (not 🎵/music)?
  2. Does the Python HF reference agree (transcript-level parity)?
  3. Numerical parity: crispasr-diff cos per encoder stage vs a Python ref dump.
  4. Short (4s) / medium (8s) / long (40s, chunking path) audio.

C++ builds from GitHub main (the SHIPPED cohere.cpp — WIP masking is NOT on
main). GGUF pulled from cstr/cohere-transcribe-arabic-07-2026-GGUF (the fixed,
published weights). Python model pulled from the gated upstream
CohereLabs/cohere-transcribe-arabic-07-2026 (cstr HF token has access).
"""
import os, sys, json, re, shutil, subprocess, time
from pathlib import Path

WORK = Path("/kaggle/working")
TMP  = Path("/kaggle/temp"); TMP.mkdir(parents=True, exist_ok=True)
REPO = TMP / "CrispASR"          # clone off /kaggle/working (gotcha #22)
BUILD = TMP / "build"
MODELS = TMP / "models";   MODELS.mkdir(parents=True, exist_ok=True)
REFMODEL = TMP / "refmodel"; REFMODEL.mkdir(parents=True, exist_ok=True)
RESULTS = WORK / "results"; RESULTS.mkdir(parents=True, exist_ok=True)

GGUF_REPO = "cstr/cohere-transcribe-arabic-07-2026-GGUF"
ORIG_REPO = "CohereLabs/cohere-transcribe-arabic-07-2026"
CRISPASR_REPO = os.environ.get("CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git")
CRISPASR_REF  = os.environ.get("CRISPASR_REF", "main")
HERE = Path(__file__).resolve().parent

def jstep(name, **kv):
    print(f"[STEP] {name} " + " ".join(f"{k}={v}" for k, v in kv.items()), flush=True)

# ───────────────────────── clone + harness + build ────────────────────────
if REPO.exists(): shutil.rmtree(REPO)
subprocess.check_call(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF,
                       "--recursive", CRISPASR_REPO, str(REPO)])
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh
kh.init_progress()
SHA = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
jstep("cloned", sha=SHA)

kh.install_build_toolchain()
subprocess.check_call(["cmake", "-S", str(REPO), "-B", str(BUILD),
                       "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON"])
with kh.build_heartbeat("build"):
    kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} "
                        f"--target crispasr-cli crispasr-diff -j{kh.safe_build_jobs(gpu=True)}")
CLI  = BUILD / "bin" / "crispasr"
DIFF = BUILD / "bin" / "crispasr-diff"
jstep("built", cli=CLI.exists(), diff=DIFF.exists())
assert CLI.exists(), "crispasr binary missing"

# ───────────────────────── deps + downloads ───────────────────────────────
for pkg in ("gguf", "soundfile"):
    try: __import__(pkg)
    except Exception: subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

from huggingface_hub import hf_hub_download, snapshot_download
TOKEN = kh.resolve_hf_token()
GGUF = {}
for tag, fn in (("q4_k", "cohere-transcribe-arabic-q4_k.gguf"),
                ("f16",  "cohere-transcribe-arabic-f16.gguf")):
    try:
        GGUF[tag] = hf_hub_download(GGUF_REPO, fn, local_dir=str(MODELS), token=TOKEN or None)
    except Exception as e:
        jstep("gguf_dl_fail", tag=tag, err=str(e)[:120])
jstep("gguf_ready", tags=list(GGUF))

try:
    snapshot_download(ORIG_REPO, local_dir=str(REFMODEL), token=TOKEN or None,
                      allow_patterns=["*.json", "*.py", "*.safetensors", "*.model",
                                      "*.txt", "tokenizer*", "*.vocab"])
    PY_OK = True
except Exception as e:
    jstep("refmodel_dl_fail", err=str(e)[:200]); PY_OK = False
jstep("refmodel_ready", ok=PY_OK)

# Read fixtures from the CLONED repo (Kaggle script kernels don't mount
# bundled data files — see kaggle_usage gotcha #19). The wavs are committed
# alongside this script.
CLIPS_DIR = REPO / "tools" / "kaggle" / "cohere-arabic-verify"
CLIPS = {"short_4s": CLIPS_DIR / "ar_clean_4s.wav",
         "med_8s":   CLIPS_DIR / "ar_clean_8s.wav",
         "long_40s": CLIPS_DIR / "ar_fleurs_40s.wav"}

def norm(s): return re.sub(r"\s+", " ", (s or "").strip())

# ───────────────────────── C++ transcription ──────────────────────────────
def cpp_transcribe(gguf, wav, timeout=1800, extra_env=None):
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    r = subprocess.run([str(CLI), "-m", str(gguf), "-f", str(wav), "-l", "ar"],
                       capture_output=True, text=True, timeout=timeout, env=env)
    lines = [l for l in r.stdout.splitlines()
             if l.strip() and not l.lstrip().startswith(("[", "whisper_", "crispasr_", "load", "main:"))]
    return norm(" ".join(lines)), r.returncode

# ───────────────────────── Python HF reference (robust) ────────────────────
def load_python():
    import torch
    _o = torch.stft
    def _s(x, *a, **k):
        w, wl = k.get("window"), k.get("win_length")
        if w is not None and wl is not None and w.shape[-1] != wl: k["win_length"] = int(w.shape[-1])
        return _o(x, *a, **k)
    torch.stft = _s
    # relative imports need a package
    pkg = TMP / "cohere_pkg"; pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("")
    for py in Path(REFMODEL).glob("*.py"): (pkg / py.name).write_text(py.read_text())
    sys.path.insert(0, str(TMP))
    from cohere_pkg.configuration_cohere_asr import CohereAsrConfig
    from cohere_pkg.modeling_cohere_asr import CohereAsrForConditionalGeneration as M
    from cohere_pkg.processing_cohere_asr import CohereAsrProcessor, CohereAsrFeatureExtractor
    from transformers import AutoTokenizer
    try:
        from transformers.generation import GenerationMixin
        if not issubclass(M, GenerationMixin): M.__bases__ = tuple(M.__bases__) + (GenerationMixin,)
    except Exception: pass
    cfg = CohereAsrConfig.from_pretrained(str(REFMODEL))
    model = M.from_pretrained(str(REFMODEL), config=cfg, torch_dtype=torch.float32).eval()
    # CohereAsrPreTrainedModel._init_weights re-initializes all Linear/Conv
    # weights to normal(0, 0.02) on some transformers versions, overwriting the
    # pretrained values -> garbage output. Reload ALL params from safetensors
    # (the fix the real reference backend applies).
    import glob
    from safetensors import safe_open
    msd = dict(model.named_parameters()); nrel = 0
    for sfp in glob.glob(str(Path(REFMODEL) / "*.safetensors")):
        with safe_open(sfp, framework="pt") as sf:
            for k in sf.keys():
                if k in msd:
                    try: msd[k].data.copy_(sf.get_tensor(k).float()); nrel += 1
                    except Exception: pass
    fe = CohereAsrFeatureExtractor.from_pretrained(str(REFMODEL))
    tok = AutoTokenizer.from_pretrained(str(REFMODEL))
    proc = CohereAsrProcessor(feature_extractor=fe, tokenizer=tok)
    # sanity: conv0 weight rms (real weights ~0.1+, random-init ~0.02) + zeroed norms
    import torch.nn as nn
    conv0 = next((m.weight for m in model.encoder.modules()
                  if getattr(getattr(m, "weight", None), "dim", lambda: 0)() >= 3), None)
    rms = float(conv0.norm() / conv0.numel() ** 0.5) if conv0 is not None else None
    z = [n for n, m in model.named_modules() if isinstance(m, nn.LayerNorm) and float(m.weight.abs().max()) < 1e-6]
    jstep("py_loaded", reloaded=nrel, conv0_rms=round(rms, 4) if rms else None, zeroed_norms=len(z))
    return model, proc

def py_transcribe(model, proc, wav):
    return model.transcribe(proc, language="ar", audio_files=[str(wav)])[0]

# ───────────────────────── numerical parity (hooks + crispasr-diff) ────────
def dump_reference(model, proc, wav, out_gguf):
    import torch, numpy as np, soundfile as sf, importlib.util
    a, _ = sf.read(str(wav))
    # mel_spectrogram exactly as the reference backend stores it ([T, n_mels])
    inputs = proc(np.asarray(a, dtype=np.float32), sampling_rate=16000,
                  return_tensors="pt", language="ar")
    mel = inputs["input_features"][0].transpose(0, 1).contiguous().float().numpy()  # [T, n_mels]
    enc = model.encoder; caps = {}
    def cap(nm):
        def h(_m, _i, o):
            t = (o.last_hidden_state if hasattr(o, "last_hidden_state")
                 else (o[0] if isinstance(o, tuple) else o))
            caps[nm] = t.detach().clone()
        return h
    H = [enc.layers[i].register_forward_hook(cap(f"encoder_layer_{i}")) for i in range(len(enc.layers))]
    for at in ("pre_encode", "subsampling", "subsample"):
        if hasattr(enc, at): H.append(getattr(enc, at).register_forward_hook(cap("pre_encode_output"))); break
    H.append(enc.register_forward_hook(cap("encoder_output")))  # final encoder hidden
    text = model.transcribe(proc, language="ar", audio_files=[str(wav)])[0]
    for h in H: h.remove()
    spec = importlib.util.spec_from_file_location("dr", str(REPO / "tools" / "dump_reference.py"))
    dr = importlib.util.module_from_spec(spec); spec.loader.exec_module(dr)
    tens = {k: v[0].detach().cpu().float().numpy() for k, v in caps.items()}
    tens["mel_spectrogram"] = mel
    meta = {"audio": Path(wav).name, "n_samples": int(len(a)), "sample_rate": 16000, "generated_text": str(text)}
    dr.write_gguf_archive(tens, meta, Path(out_gguf))
    return text

# ───────────────────────── run everything ─────────────────────────────────
results = {"sha": SHA, "clips": {}}
for name, wav in CLIPS.items():
    txt, rc = cpp_transcribe(GGUF.get("q4_k") or next(iter(GGUF.values())), wav)
    results["clips"][name] = {"cpp": txt, "cpp_rc": rc, "cpp_music": "🎵" in txt}
    jstep(f"cpp_{name}", rc=rc, music="🎵" in txt, text=txt[:70])

if PY_OK:
    try:
        model, proc = load_python()
        for name, wav in CLIPS.items():
            pt = py_transcribe(model, proc, wav)
            c = results["clips"][name]
            c["py"] = pt
            c["py_music"] = "🎵" in pt
            c["exact_match"] = norm(c["cpp"]) == norm(pt)
            # word-level parity
            cw, pw = norm(c["cpp"]).split(), norm(pt).split()
            common = sum(1 for w in cw if w in pw)
            c["word_overlap"] = round(common / max(1, len(pw)), 3)
            jstep(f"py_{name}", match=c["exact_match"], overlap=c["word_overlap"], text=pt[:70])
        # numerical parity on the med clip (f16 for tightest), best-effort
        try:
            refg = RESULTS / "cohere_ref_med.gguf"
            dump_reference(model, proc, CLIPS["med_8s"], refg)
            g = GGUF.get("f16") or GGUF.get("q4_k")
            dr = subprocess.run([str(DIFF), "cohere", str(g), str(refg), str(CLIPS["med_8s"])],
                                capture_output=True, text=True, timeout=1800)
            tail = dr.stdout[-3000:]
            results["numerical_diff_tail"] = tail
            cos = [float(m) for m in re.findall(r"cos_min[=\s]+([0-9.]+)", tail)]
            results["numerical_cos_min"] = min(cos) if cos else None
            jstep("numerical_diff", cos_min=results.get("numerical_cos_min"))
        except Exception as e:
            results["numerical_diff_tail"] = f"diff failed: {e}"
            jstep("numerical_diff_fail", err=str(e)[:150])
    except Exception as e:
        import traceback; traceback.print_exc()
        results["python_error"] = str(e)[:300]
        jstep("python_fail", err=str(e)[:150])

# ───────────────────────── overhang-masking A/B ───────────────────────────
# HISTORICAL: this A/B asked whether CRISPASR_COHERE_MASK_OVERHANG=1 helped.
# It concluded, and the env var was removed from the C++ side — current
# binaries read no such variable, so on a fresh build both arms below run the
# IDENTICAL default path and "identical: true" is vacuous, not a finding.
RUN_OVERHANG_AB = False  # flip only against a checkout that still reads the var
try:
    if not RUN_OVERHANG_AB:
        raise RuntimeError("skipped: CRISPASR_COHERE_MASK_OVERHANG no longer read by current C++")
    import soundfile as sf
    a40, sr = sf.read(str(CLIPS["long_40s"]))
    clip11 = TMP / "ar_11s.wav"
    sf.write(str(clip11), a40[5 * sr:16 * sr], sr)
    g = GGUF.get("q4_k") or next(iter(GGUF.values()))
    mask_ab = {}
    for nm, wav in [("8s", CLIPS["med_8s"]), ("11s", clip11), ("40s", CLIPS["long_40s"])]:
        off, _ = cpp_transcribe(g, wav)
        on, _ = cpp_transcribe(g, wav, extra_env={"CRISPASR_COHERE_MASK_OVERHANG": "1"})
        mask_ab[nm] = {"off": off, "on": on, "identical": norm(off) == norm(on)}
        jstep(f"mask_ab_{nm}", identical=mask_ab[nm]["identical"])
    results["mask_ab"] = mask_ab
    results["mask_changes_anything"] = any(not v["identical"] for v in mask_ab.values())
except Exception as e:
    results["mask_ab_error"] = str(e)[:200]
    jstep("mask_ab_fail", err=str(e)[:120])

# ───────────────────────── verdict ────────────────────────────────────────
verdict = {}
for name, c in results["clips"].items():
    ok_cpp = (c.get("cpp_rc") == 0) and c.get("cpp") and not c.get("cpp_music")
    ok_py  = ("py" in c) and c.get("py") and not c.get("py_music")
    verdict[name] = {"cpp_real_arabic": bool(ok_cpp), "py_real_arabic": bool(ok_py),
                     "exact_match": c.get("exact_match"), "word_overlap": c.get("word_overlap")}
results["verdict"] = verdict
results["numerical_cos_min"] = results.get("numerical_cos_min")

(RESULTS / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
print("\n===== RESULTS =====")
print(json.dumps(results, ensure_ascii=False, indent=2))
# single-line marker — trivially greppable from the kernel log
print("RESULTS_JSON " + json.dumps(results, ensure_ascii=False))
jstep("done")
