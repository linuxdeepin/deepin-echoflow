"""Convert and publish the five Pocket-TTS language checkpoints for #411.

Artifacts are uploaded immediately after each convert/quantize step so a later
language failure cannot lose completed work. Only one language snapshot and its
two GGUFs coexist on disk at a time.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


WORK = Path("/tmp/crispasr-pocket-tts-411")
REPO = WORK / "CrispASR"
OUT = WORK / "out"
HF_REPO = "cstr/pocket-tts-GGUF"
LANGUAGES = ("german", "spanish", "italian", "portuguese", "french_24l")


def run(args: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, env=env, check=True)


def token() -> str:
    for path in (
        Path("/kaggle/input/crispasr-hf-token/hf_token.txt"),
        Path("/kaggle/input/datasets/${KAGGLE_ACCOUNT}/crispasr-hf-token/hf_token.txt"),
    ):
        if path.is_file():
            return path.read_text().strip()
    raise RuntimeError("HF token dataset is not mounted")


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    hf_token = token()
    env = os.environ.copy()
    env.update({
        "HF_TOKEN": hf_token,
        "HF_HOME": str(WORK / "hf"),
        "TMPDIR": str(WORK / "tmp"),
        "PYTHONNOUSERSITE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    })
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    run(["git", "clone", "--depth", "1", "--branch", "feat/pocket-multilingual",
         "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    run(["git", "submodule", "update", "--init", "--recursive"], cwd=REPO)
    run(["python", "-m", "pip", "install", "-q", "gguf", "sentencepiece", "safetensors",
         "huggingface_hub", "pyyaml"])

    run(["cmake", "-G", "Ninja", "-S", str(REPO), "-B", str(REPO / "build"),
         "-DCMAKE_BUILD_TYPE=Release", "-DCRISPASR_BUILD_TESTS=OFF",
         "-DCRISPASR_BUILD_SERVER=OFF"])
    run(["cmake", "--build", str(REPO / "build"), "--target", "crispasr-quantize", "-j2"])

    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    api.create_repo(HF_REPO, repo_type="model", exist_ok=True)

    for language in LANGUAGES:
        run(["python", str(REPO / "models/convert-pocket-tts-to-gguf.py"),
             "--input", "kyutai/pocket-tts", "--output-dir", str(OUT),
             "--language", language, "--voice-cloning"], env=env)
        f16 = OUT / f"pocket-tts-{language}-f16.gguf"
        q8 = OUT / f"pocket-tts-{language}-q8_0.gguf"
        run([str(REPO / "build/bin/crispasr-quantize"), str(f16), str(q8), "q8_0"])

        for artifact in (f16, q8):
            api.upload_file(path_or_fileobj=artifact, path_in_repo=artifact.name,
                            repo_id=HF_REPO, repo_type="model",
                            commit_message=f"Add Pocket-TTS {language} {artifact.stem.rsplit('-', 1)[-1]}")
            print(f"UPLOADED {artifact.name} {artifact.stat().st_size}", flush=True)
            artifact.unlink()

        # snapshot_download's language-local cache is disposable now. Removing
        # it keeps the next checkpoint from accumulating shared Mimi copies.
        shutil.rmtree(WORK / "hf" / "hub", ignore_errors=True)

    print("POCKET_TTS_411_CONVERSION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
