"""Live Phonon-2 wiring guard: explicit alias, renamed model, repeated calls, CLI LID."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
EXPECTED = "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", required=True)
    parser.add_argument("--cli", required=True)
    parser.add_argument("--require-live", action="store_true")
    args = parser.parse_args()
    model = Path(os.environ.get("CRISPASR_MODEL_PHONON2", ""))
    if not model.is_file():
        print("Phonon-2 model missing: set CRISPASR_MODEL_PHONON2")
        return 1 if args.require_live else 77
    import numpy as np
    from crispasr import Session
    wav = ROOT / "samples/jfk.wav"
    with wave.open(str(wav)) as source:
        assert source.getframerate() == 16000 and source.getnchannels() == 1
        audio = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768
    assert "phonon2" in Session.available_backends(lib_path=args.lib)
    for backend in (None, "phonon2"):
        with Session(str(model), lib_path=args.lib, backend=backend, n_threads=4) as session:
            assert session.backend == "parakeet", session.backend
            for _ in range(2):
                text = " ".join(segment.text for segment in session.transcribe(audio)).strip()
                assert text == EXPECTED, text
    # A neutral basename defeats the CLI filename shortcut. Model metadata
    # must still suppress automatic Whisper language detection.
    with tempfile.TemporaryDirectory(prefix="phonon2-live-") as directory:
        alias = Path(directory) / "renamed.gguf"
        alias.symlink_to(model.resolve())
        for backend in (None, "phonon2"):
            command = [args.cli, "-m", str(alias), "-f", str(wav)]
            if backend:
                command += ["--backend", backend]
            result = subprocess.run(command, capture_output=True, text=True, timeout=300, check=True)
            assert EXPECTED in result.stdout, result.stdout
            assert "whisper_init" not in result.stderr, result.stderr
            assert "detected '" not in result.stderr, result.stderr
    environment = dict(os.environ, CRISPASR_SCHED_PROFILE="1")
    result = subprocess.run([args.cli, "--backend", "phonon2", "-m", str(model), "-f", str(wav),
                             "--no-flash-attn"], capture_output=True, text=True, timeout=300,
                            check=True, env=environment)
    assert EXPECTED in result.stdout, result.stdout
    assert "sched_profile[parakeet.encoder]" in result.stderr, result.stderr
    assert "FLASH_ATTN_EXT" not in result.stderr, result.stderr
    print("PASS: Phonon-2 auto/explicit C ABI and CLI; repeated transcripts; no automatic Whisper LID")
    return 0


if __name__ == "__main__":
    sys.exit(main())
