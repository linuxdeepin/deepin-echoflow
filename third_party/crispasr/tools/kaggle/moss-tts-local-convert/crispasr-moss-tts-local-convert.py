# CrispASR — MOSS-TTS-Local (4B) P1 converter validation (#249 second deliverable)
#
# Runs models/convert-moss-tts-local-to-gguf.py on the REAL 4B weights and reads
# the produced GGUF back to confirm: arch = "moss-tts-local", the metadata KV, and
# every tensor name/shape/dtype. Empirically validates the converter (which is
# already correct-by-construction vs the 438 tensor names) before the runtime is
# built against it. Optionally uploads the F16 GGUF to cstr/moss-tts-local-v1.5-GGUF.
#
# CPU kernel (no GPU needed for conversion). Stage under /tmp (~70 GB), not
# /kaggle/working (~20 GB). HF auth via the ${KAGGLE_ACCOUNT}/crispasr-hf-token dataset.

import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
TMP = Path("/tmp")
REPO = TMP / "CrispASR"
MODELS = TMP / "moss-local"
WORK = Path("/kaggle/working")
MODELS.mkdir(parents=True, exist_ok=True)

REF = os.environ.get("CRISPASR_REF", "feat/moss-tts-parity-diff")
HF_MODEL = os.environ.get("MOSS_MODEL", "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5")
UPLOAD_REPO = os.environ.get("MOSS_UPLOAD_REPO", "")  # set to cstr/... to upload
_T0 = time.time()


def log(m):
    print(f"[{round(time.time() - _T0, 1)}s] {m}", flush=True)


def main():
    summary = {"ref": REF, "model": HF_MODEL}
    log(f"clone {REF}")
    if not REPO.exists():
        subprocess.check_call(["git", "clone", "--depth", "1", "--branch", REF,
                               "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    try:
        import kaggle_harness as kh
        kh.init_progress()
        tok = kh.resolve_hf_token()
    except Exception:  # noqa: BLE001
        tok = os.environ.get("HF_TOKEN")
    if tok:
        os.environ["HF_TOKEN"] = tok
        os.environ["HUGGING_FACE_HUB_TOKEN"] = tok

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "gguf", "safetensors", "huggingface_hub"])

    from huggingface_hub import snapshot_download
    log("download 4B weights + config -> /tmp")
    src = snapshot_download(HF_MODEL, cache_dir=str(MODELS / "hf"), token=tok,
                            allow_patterns=["*.safetensors", "*.json", "merges.txt",
                                            "vocab.json", "tokenizer.json", "added_tokens.json"])
    log(f"downloaded to {src}")

    out = MODELS / "moss-tts-local-v1.5-f16.gguf"
    log("convert -> f16 GGUF")
    env = os.environ.copy()
    env["TMPDIR"] = str(MODELS)  # GGUFWriter temp spill off the small /tmp root
    r = subprocess.run([sys.executable, str(REPO / "models" / "convert-moss-tts-local-to-gguf.py"),
                        "--input", src, "--output", str(out)],
                       env=env, capture_output=True, text=True, timeout=3600)
    (WORK / "convert.log").write_text(r.stdout + "\n--STDERR--\n" + r.stderr)
    print(r.stdout[-3000:], flush=True)
    if r.returncode != 0:
        summary["convert_error"] = r.stderr[-1500:]
        (WORK / "summary.json").write_text(json.dumps(summary, indent=2))
        raise SystemExit(f"convert failed rc={r.returncode}\n{r.stderr[-1500:]}")
    summary["gguf_gb"] = round(out.stat().st_size / 1e9, 2)
    log(f"GGUF written: {summary['gguf_gb']} GB")

    # ── read the GGUF back and validate structure ──
    from gguf import GGUFReader
    reader = GGUFReader(str(out))
    arch = None
    kv = {}
    for field in reader.fields.values():
        try:
            val = field.contents()
        except Exception:  # noqa: BLE001
            val = None
        kv[field.name] = val
        if field.name == "general.architecture":
            arch = val
    tensors = [(t.name, list(t.shape), str(t.tensor_type).split(".")[-1]) for t in reader.tensors]
    summary["arch"] = arch
    summary["n_tensors"] = len(tensors)
    # group tensor prefixes
    import re
    groups = {}
    for name, shape, dt in tensors:
        g = re.sub(r"\.\d+\.", ".N.", name)
        groups.setdefault(g, [0, shape, dt])
        groups[g][0] += 1
    print("\n== GGUF metadata (moss/llm/local) ==")
    for k in sorted(kv):
        if any(k.startswith(p) for p in ("moss", "general.architecture", "general.name")):
            print(f"  {k} = {kv[k]}")
    print("\n== tensor groups (count, sample shape, dtype) ==")
    for g in sorted(groups):
        c, shape, dt = groups[g]
        print(f"  {g:34} x{c:<3} {shape} {dt}")
    summary["tensor_groups"] = {g: v[0] for g, v in groups.items()}

    # sanity checks
    checks = {
        "arch_is_moss_tts_local": arch == "moss-tts-local",
        "has_llm_lm_head": any(n == "llm.lm_head.weight" for n, _, _ in tensors),
        "has_12_audio_embed": sum(1 for n, _, _ in tensors if n.startswith("moss.audio_embed.")) == 12,
        "has_12_audio_head": sum(1 for n, _, _ in tensors if n.startswith("moss.audio_head.")) == 12,
        "has_local_blk": any(n.startswith("local.blk.0.") for n, _, _ in tensors),
        "has_local_text_head": any(n == "moss.local_text_head.weight" for n, _, _ in tensors),
    }
    summary["checks"] = checks
    print("\n== structural checks ==")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    summary["all_checks_pass"] = all(checks.values())

    if UPLOAD_REPO and summary["all_checks_pass"]:
        log(f"upload -> {UPLOAD_REPO}")
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(UPLOAD_REPO, repo_type="model", exist_ok=True, private=False, token=tok)
        api.upload_file(path_or_fileobj=str(out), path_in_repo=out.name,
                        repo_id=UPLOAD_REPO, repo_type="model", token=tok)
        summary["uploaded"] = f"{UPLOAD_REPO}/{out.name}"

    (WORK / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 60 + "\n" + json.dumps(summary, indent=2) + "\n" + "=" * 60)
    log(f"P1 convert validation all_checks_pass={summary['all_checks_pass']}")


if __name__ == "__main__":
    main()
