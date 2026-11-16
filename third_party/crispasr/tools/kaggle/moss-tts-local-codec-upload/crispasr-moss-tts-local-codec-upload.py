# CrispASR — MOSS-TTS-Local codec GGUF convert + upload (#249 ship, CPU, no build)
#
# The 4B port is validated (F16 round-trip overlap 0.969). The backbone F16 GGUF
# is already hosted; the only missing artifact is the decode-only codec GGUF.
# This CPU kernel (no crispasr build): curl-download MOSS-Audio-Tokenizer-v2
# safetensors (curl -C - resume — hf_transfer wedges on Kaggle), run
# write_codec_gguf, and upload to cstr/moss-tts-local-v1.5-GGUF with a
# hang-tolerant + server-verified upload (dev-notes HF pattern).

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
REPO = TEMP / "CrispASR"
MODELS = TEMP / "codec-models"
MODELS.mkdir(parents=True, exist_ok=True)

# feat/moss-tts-local-4b was merged long ago; cloning it pinned this kernel to a
# stale converter, so a re-run would silently reproduce the old decode-only GGUF.
CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
HF_CODEC = os.environ.get("MOSS_CODEC", "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2")
GGUF_REPO = os.environ.get("MOSS_GGUF_REPO", "cstr/moss-tts-local-v1.5-GGUF")
# The encoder-carrying codec is roughly twice the size and is useless until
# encode() lands, so it does NOT overwrite the shipped decode-only file yet:
# Subtitle Edit downloads that path. Ship it under a dev name, validate cloning
# against it, and only then promote it over the published one.
CODEC_NAME = os.environ.get("MOSS_CODEC_NAME", "moss-tts-local-v1.5-codec-enc.gguf")
HFBASE = "https://huggingface.co"


def main():
    if not REPO.exists():
        subprocess.check_call(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF,
                               "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    if str(REPO / "tools" / "kaggle") not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    import kaggle_harness as kh
    kh.init_progress()
    kh._HF_PROGRESS_PATH = "runs/moss-tts-local-codec-upload-live.jsonl"
    kh._HF_PUSH_INTERVAL_S = 15.0
    hf_token = kh.resolve_hf_token()
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    kh.step("start", token=bool(hf_token))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "safetensors", "gguf"])

    def curl_get(what, url, out, total_timeout=2400):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = ("curl -sSL -C - --retry 50 --retry-delay 3 --retry-all-errors "
               "--speed-limit 30000 --speed-time 20 --fail-with-body "
               f'-H "Authorization: Bearer {hf_token}" -o "{out}" "{url}"')
        box = {}

        def _w():
            box["rc"] = subprocess.run(cmd, shell=True).returncode

        t = threading.Thread(target=_w, daemon=True)
        t.start()
        t0 = time.time()
        while t.is_alive():
            t.join(15)
            mb = round(out.stat().st_size / 1e6, 1) if out.exists() else 0.0
            kh.step(f"{what} (dl)", mb=mb, s=round(time.time() - t0))
            kh._push_progress_to_hf(force=True)
            if time.time() - t0 > total_timeout:
                raise SystemExit(f"{what}: timeout at {mb}MB")
        if box.get("rc") != 0:
            raise SystemExit(f"{what}: curl rc={box.get('rc')}")
        kh.step(f"{what}: ok", mb=round(out.stat().st_size / 1e6, 1))

    codec_dir = MODELS / "codec-src"
    for fn in ("config.json", "model.safetensors.index.json",
               "model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors",
               "model-00003-of-00003.safetensors"):
        curl_get(f"codec {fn}", f"{HFBASE}/{HF_CODEC}/resolve/main/{fn}", codec_dir / fn)

    kh.step("convert codec")
    conv = REPO / "models" / "convert-moss-tts-local-to-gguf.py"
    codec_gguf = MODELS / CODEC_NAME
    codegen = ("import importlib.util as u; from pathlib import Path;"
               "s=u.spec_from_file_location('c', r'%s'); m=u.module_from_spec(s); s.loader.exec_module(m);"
               "m.write_codec_gguf(Path(r'%s'), Path(r'%s'))") % (str(conv), str(codec_dir), str(codec_gguf))
    subprocess.run([sys.executable, "-c", codegen], check=True, timeout=3600)
    if not codec_gguf.exists():
        raise SystemExit("codec GGUF not produced")
    sz = codec_gguf.stat().st_size
    kh.step("codec converted", gb=round(sz / 1e9, 3))

    # ── upload with hang-tolerance + server-side verify ────────────────────
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    # Only create the repo if it is genuinely absent. exist_ok=True still POSTs
    # to /api/repos/create, and a fine-grained token that may write to existing
    # repos but not create new ones gets 401 there — which killed this kernel
    # AFTER a clean 4.25 GB conversion, before the upload was even attempted.
    # exist_ok only swallows a 409 Conflict, not a 401.
    try:
        api.repo_info(GGUF_REPO, repo_type="model")
    except Exception:  # noqa: BLE001 — absent, or not visible to this token
        api.create_repo(GGUF_REPO, repo_type="model", exist_ok=True)

    def landed():
        try:
            for f in api.list_repo_tree(GGUF_REPO, repo_type="model"):
                if f.path == CODEC_NAME and getattr(f, "size", 0) == sz:
                    return True
        except Exception:  # noqa: BLE001
            pass
        return False

    kh.step("upload codec")
    box = {}

    def _up():
        try:
            api.upload_file(path_or_fileobj=str(codec_gguf), path_in_repo=CODEC_NAME,
                            repo_id=GGUF_REPO, repo_type="model")
            box["ok"] = True
        except Exception as e:  # noqa: BLE001
            box["e"] = e

    t = threading.Thread(target=_up, daemon=True)
    t.start()
    t0 = time.time()
    ok = False
    while t.is_alive():
        t.join(20)
        kh.step("upload", s=round(time.time() - t0))
        kh._push_progress_to_hf(force=True)
        if landed():  # hung-but-committed = success (dev-notes)
            ok = True
            break
        if time.time() - t0 > 2400:
            break
    ok = ok or box.get("ok", False) or landed()
    kh.step("done", uploaded=ok, codec_gb=round(sz / 1e9, 3),
            where=f"{GGUF_REPO}/{CODEC_NAME}" if ok else None, err=str(box.get("e", ""))[:150])
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        import traceback
        print(f"FATAL: {e}\n{traceback.format_exc()}", flush=True)
        try:
            import kaggle_harness as kh
            kh.step("FATAL", err=str(e)[:200])
        except Exception:
            pass
        sys.exit(1)
