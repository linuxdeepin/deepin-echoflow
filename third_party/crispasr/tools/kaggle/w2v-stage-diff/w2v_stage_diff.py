#!/usr/bin/env python3
"""wav2vec2-xlsr-en (ctc_logits cos 0.908) and hubert-large (0.955) vs
data2vec-base (0.99916, control): which stage diverges first?
transformers hooks (feature_extractor, feature_projection, encoder.layer_norm,
lm_head) vs CRISPASR_WAV2VEC2_DUMP_DIR (cnn_out, feat_proj, after_global_ln,
logits), per-frame cosine. C++ layouts are tried both ways ([C,T] / [T,C]).
"""
import json, os, re, subprocess, sys, traceback
from pathlib import Path
OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
REPO = Path("/tmp/CrispASR"); G = Path("/tmp/g"); G.mkdir(exist_ok=True)
CASES = {
    "wav2vec2": ("jonatasgrosman/wav2vec2-large-xlsr-53-english", "cstr/wav2vec2-large-xlsr-53-english-GGUF", "wav2vec2-xlsr-en.gguf"),
    "hubert": ("facebook/hubert-large-ls960-ft", "cstr/hubert-large-ls960-ft-GGUF", "hubert-large-ls960-ft-f16.gguf"),
    "data2vec": ("facebook/data2vec-audio-base-960h", "cstr/data2vec-audio-960h-GGUF", "data2vec-audio-base-960h-f16.gguf"),
}
res = {"errors": [], "cmp": {}, "hf": {}, "cpp": {}}
def save(): (OUT / "result.json").write_text(json.dumps(res, indent=1))
def pf(U, C):
    cf = (U * C).sum(1) / (np.linalg.norm(U, axis=1) * np.linalg.norm(C, axis=1) + 1e-30)
    return {"cos_min": round(float(cf.min()), 5), "argmin": int(cf.argmin()), "T": int(len(cf)), "cos_mean": round(float(cf.mean()), 5),
            "first8": [round(float(x), 4) for x in cf[:8]], "last8": [round(float(x), 4) for x in cf[-8:]],
            "rms_hf": round(float(np.sqrt((U ** 2).mean())), 4), "rms_cpp": round(float(np.sqrt((C ** 2).mean())), 4)}
try:
    subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    subprocess.run(["git", "submodule", "update", "--init", "--recursive", "--depth", "1"], cwd=str(REPO))
    res["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    sys.path.insert(0, str(REPO / "tools" / "kaggle")); import kaggle_harness as kh
    tok = kh.resolve_hf_token()
    if tok: os.environ["HF_TOKEN"] = tok
    import numpy as np, soundfile as sf, torch
    globals()["np"] = np
    from huggingface_hub import hf_hub_download
    from transformers import AutoFeatureExtractor, AutoModelForCTC
    kh.install_build_toolchain()
    kh.sh(f"cmake -S {REPO} -B {REPO}/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRISPASR_BUILD_TESTS=OFF "
          "-DCRISPASR_BUILD_EXAMPLES=ON -DCRISPASR_BUILD_SERVER=OFF " + " ".join(kh.cache_and_link_flags()))
    with kh.build_heartbeat("cmake.build"):
        kh.sh(f"cmake --build {REPO}/build -j$(nproc) --target crispasr-cli")
    B = REPO / "build" / "bin" / "crispasr"
    wav = REPO / "samples/jfk.wav"
    audio, sr = sf.read(wav, dtype="float32")
    for key, (src, repo, fn) in CASES.items():
        fe = AutoFeatureExtractor.from_pretrained(src)
        model = AutoModelForCTC.from_pretrained(src, torch_dtype=torch.float32).eval()
        base = getattr(model, model.base_model_prefix)
        caps = {}
        def hook(name):
            def h(m, i, o):
                t = o[0] if isinstance(o, (tuple, list)) else o
                caps[name] = t.detach().float().numpy()
            return h
        base.feature_extractor.register_forward_hook(hook("cnn_out"))       # (1, C, T)
        base.feature_projection.register_forward_hook(hook("feat_proj"))    # (1, T, H) (+ extract_features)
        base.encoder.layer_norm.register_forward_hook(hook("encoder_ln"))   # (1, T, H)
        inp = fe(audio, sampling_rate=16000, return_tensors="pt")
        res["hf"][key] = {"do_normalize": bool(getattr(fe, "do_normalize", None)), "stable": bool(getattr(model.config, "do_stable_layer_norm", False)),
                          "feat_proj_layer_norm": getattr(model.config, "feat_proj_layer_norm", None), "conv_bias": getattr(model.config, "conv_bias", None),
                          "input_mean": float(inp.input_values.mean()), "input_std": float(inp.input_values.std())}
        with torch.no_grad():
            caps["logits"] = model(inp.input_values).logits.float().numpy()
        gg = hf_hub_download(repo, fn, cache_dir=str(G))
        d = G / f"dump_{key}"; d.mkdir(exist_ok=True)
        r = subprocess.run([str(B), "-m", gg, "-f", str(wav), "-np", "-ng", "-t", "4"], capture_output=True, text=True,
                           env=dict(os.environ, CRISPASR_WAV2VEC2_DUMP_DIR=str(d)), timeout=1200)
        res["cpp"][key] = {"text": " ".join(r.stdout.split()), "dumps": re.findall(r"DUMP: .*", r.stderr)}
        cmp = {}
        for ck, hk in (("cnn_out", "cnn_out"), ("feat_proj", "feat_proj"), ("after_global_ln", "encoder_ln"), ("logits", "logits")):
            f = d / f"{ck}.bin"
            if not f.exists() or hk not in caps:
                cmp[ck] = "missing"; continue
            U = caps[hk][0]
            if ck == "cnn_out": U = U.T  # (C,T) -> (T,C)
            C = np.fromfile(f, dtype=np.float32)
            best = None
            for cand in (C.reshape(U.shape[0], U.shape[1]) if C.size == U.size else None,
                         C.reshape(U.shape[1], U.shape[0]).T if C.size == U.size else None):
                if cand is None: continue
                s = pf(U, cand)
                if best is None or s["cos_mean"] > best["cos_mean"]: best = s
            cmp[ck] = best if best else {"size_hf": int(U.size), "size_cpp": int(C.size)}
        res["cmp"][key] = cmp
        print(key, res["hf"][key], {k: (v.get("cos_min"), v.get("argmin")) if isinstance(v, dict) else v for k, v in cmp.items()}, flush=True)
        save(); os.remove(gg); del model
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
    import shutil; shutil.rmtree(REPO, ignore_errors=True)
