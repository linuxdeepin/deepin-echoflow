#!/usr/bin/env python3
"""parakeet-tdt-0.6b-v3 and parakeet-tdt_ctc-1.1b diverge from NeMo at encoder
layer 0 (cos 0.75 / 0.94) while tdt-1.1b and rnnt-0.6b match at 0.99999.
NeMo configs: v3 xscaling=false (its GGUF has no key -> C++ defaults true);
tdt_ctc rel_pos_local_attn [128,128] + 1 global token (GGUF has no keys).
crispasr-diff per env arm, every stage printed. CPU build, like GH.
"""
import json, os, re, subprocess, sys, traceback
from pathlib import Path
OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
REPO = Path("/tmp/CrispASR"); G = Path("/tmp/g"); G.mkdir(exist_ok=True)
REF = os.environ.get("CRISPASR_REF", "fix/parakeet-cfg")
FX = ("cstr/crispasr-regression-fixtures", "5dbb78aae08e1e8f9e6a9902dfa83f817b1bde63")
CASES = {
    "v3": ("cstr/parakeet-tdt-0.6b-v3-GGUF", "parakeet-tdt-0.6b-v3.gguf", "parakeet-tdt-0.6b-en/ref.gguf",
           {"default": {}, "xscaling0": {"CRISPASR_PARAKEET_XSCALING": "0"}}),
    "tdt_ctc": ("cstr/parakeet-tdt_ctc-1.1b-GGUF", "parakeet-tdt_ctc-1.1b.gguf", "parakeet-tdt_ctc-1.1b/ref.gguf",
                {"default": {}, "local128_g1": {"CRISPASR_PARAKEET_ATT_CONTEXT": "128,128", "CRISPASR_PARAKEET_GLOBAL_TOKENS": "1"},
                 "xscaling1": {"CRISPASR_PARAKEET_XSCALING": "1"}}),
}
res = {"errors": [], "runs": {}}
def save(): (OUT / "result.json").write_text(json.dumps(res, indent=1))
LINE = re.compile(r"\[(PASS|FAIL)\s*\]\s+(\S+)\s+.*?cos_min=\s*(-?[0-9.eE+-]+)")
try:
    subprocess.check_call(["git", "clone", "--depth", "1", "-b", REF, "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    subprocess.run(["git", "submodule", "update", "--init", "--recursive", "--depth", "1"], cwd=str(REPO))
    res["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    sys.path.insert(0, str(REPO / "tools" / "kaggle")); import kaggle_harness as kh
    tok = kh.resolve_hf_token()
    if tok: os.environ["HF_TOKEN"] = tok
    from huggingface_hub import hf_hub_download
    kh.install_build_toolchain()
    kh.sh(f"cmake -S {REPO} -B {REPO}/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRISPASR_BUILD_TESTS=OFF "
          "-DCRISPASR_BUILD_EXAMPLES=ON -DCRISPASR_BUILD_SERVER=OFF " + " ".join(kh.cache_and_link_flags()))
    with kh.build_heartbeat("cmake.build"):
        kh.sh(f"cmake --build {REPO}/build -j$(nproc) --target crispasr-diff")
    D = REPO / "build" / "bin" / "crispasr-diff"
    for key, (repo, fn, refp, arms) in CASES.items():
        gg = hf_hub_download(repo, fn, cache_dir=str(G))
        ref = hf_hub_download(FX[0], refp, revision=FX[1], cache_dir=str(G))
        for arm, env in arms.items():
            r = subprocess.run([str(D), "parakeet", gg, ref, str(REPO / "samples/jfk.wav")], capture_output=True, text=True,
                               env=dict(os.environ, **env), timeout=1800)
            st = {m.group(2): float(m.group(3)) for m in LINE.finditer(r.stdout)}
            res["runs"][f"{key}|{arm}"] = {"stages": st, "tail": r.stdout[-2500:] if not st else ""}
            print(key, arm, {k: v for k, v in st.items() if not k.startswith("encoder_layer_")},
                  "layer min", min([v for k, v in st.items() if k.startswith("encoder_layer_")] or [None]), flush=True)
            save()
        os.remove(gg)
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
    import shutil
    shutil.rmtree(REPO, ignore_errors=True)
