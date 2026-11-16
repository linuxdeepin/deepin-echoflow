#!/usr/bin/env python
"""test-wyoming-marking.py — EU AI Act Art. 50 marking on the Wyoming surface.

The Wyoming TTS server (--wyoming-port) shipped for four releases emitting
completely unmarked synthetic audio: no watermark, no clone classification, no
spoken disclaimer, no consent gate. The marking work was organised per-surface
from the list in docs/eu-ai-act.md §6.1, and Wyoming was never on it. A Home
Assistant client sending {"voice":{"name":"x"}} got back unmarked, undisclosed
cloned speech.

crispasr_marking::decide_raw_surface() pins the POLICY in the unit tier
(tests/test-marking-policy.cpp). This file pins the thing a unit test cannot
reach: that the synthesize handler actually CALLS it. The original bug was a
missing call, not a wrong rule — a green policy test would have stayed green
throughout.

  1. watermark executes — the built-in detector must report a confidence well
     above the 0.5 chance baseline on Wyoming's output, and near 0.5 on a
     genuinely unmarked render of the same text (`--tts-output x.wav
     --no-watermark --accept-marking-responsibility`; .wav carries C2PA, so
     that opt-out is really honored).
  2. watertight floor — a server launched with --no-watermark must return
     BIT-IDENTICAL audio to the default one, and that audio must still detect
     as watermarked. Raw PCM carries no manifest, so the opt-out must be
     overridden rather than obeyed.

Do NOT weaken case 1 to "the PCM differs from the baseline". It does anyway:
CLI and server synthesis are not bit-identical, so that assertion stays green
with the watermark call REMOVED (verified — mean |delta| 7.1 marked vs 0.4
unmarked, both nonzero). It is the exact shape of a test that cannot go red.
The detector separates cleanly instead: 0.875 vs 0.500 on the same clip.
  3. clone refused without consent — a .wav voice on a server with no
     --i-have-rights must yield an empty audio stream.
  4. consented clone is disclaimed — with --i-have-rights, the same request
     must come back LONGER than the plain synthesis, by the spoken
     AI-disclosure prefix + its 300 ms silence gap.

Needs a TTS GGUF (kokoro is enough — no cloning support required, since the
gates run before synthesis and case 4 uses a provenance-stamped pack).
SKIPs (exit 0) when no model or binary is found. No third-party deps.

Usage:
  python tests/test-wyoming-marking.py [--cache-dir DIR]
  CRISPASR_MODELS_DIR=/path/to/models python tests/test-wyoming-marking.py
"""
import array
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ~10 s of speech. The spread-spectrum detector is length-sensitive: docs/
# eu-ai-act.md §6.7 measures a 100% true-positive rate above 0.65 at 10 s but
# only 78% at 1 s, so a short clip would make this test flaky in the direction
# that matters (a false "not watermarked").
TEXT = ("The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump. "
        "The five boxing wizards jump in quickly. "
        "Sphinx of black quartz, judge my vow.")

# Chance level for the sign-agreement test is 0.5, not 0. Watermarked speech
# lands near 0.85+; this bar sits well clear of both.
WM_THRESHOLD = 0.65


def find_binary():
    for rel in ["build/bin/crispasr", "build/bin/Release/crispasr.exe",
                "build-ninja-compile/bin/crispasr", "bin/crispasr", "bin/crispasr.exe"]:
        p = os.path.join(ROOT, rel)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def find_tts_model(cache_dir):
    """A kokoro backbone GGUF — small, CPU-fast, and ships preset voices."""
    for d in [cache_dir,
              os.environ.get("CRISPASR_TEST_CACHE"),
              os.environ.get("CRISPASR_MODELS_DIR"),
              os.path.expanduser("~/.cache/crispasr")]:
        if d and os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.startswith("kokoro-") and f.endswith(".gguf") and "voice" not in f:
                    return os.path.join(d, f)
    return None


def wyoming_send(sock, msg_type, data):
    hdr = {"type": msg_type}
    data_bytes = b""
    if data:
        data_bytes = json.dumps(data, ensure_ascii=False).encode("utf-8")
        hdr["data_length"] = len(data_bytes)
    sock.sendall((json.dumps(hdr) + "\n").encode())
    if data_bytes:
        sock.sendall(data_bytes)


def wyoming_synthesize(port, text, voice=None, timeout=180.0):
    """Send synthesize, collect audio-chunk payloads. Returns int16 sample list."""
    s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    s.settimeout(timeout)
    data = {"text": {"text": text}}
    if voice:
        data["voice"] = {"name": voice}
    wyoming_send(s, "synthesize", data)

    pcm = b""
    try:
        while True:
            buf = b""
            while True:
                c = s.recv(1)
                if not c:
                    raise EOFError
                if c == b"\n":
                    break
                buf += c
            hdr = json.loads(buf)
            n = hdr.get("data_length", 0)
            while n > 0:
                chunk = s.recv(n)
                if not chunk:
                    raise EOFError
                n -= len(chunk)
            n = hdr.get("payload_length", 0) or 0
            payload = b""
            while len(payload) < n:
                chunk = s.recv(n - len(payload))
                if not chunk:
                    raise EOFError
                payload += chunk
            t = hdr.get("type")
            if t == "audio-chunk":
                pcm += payload
            elif t == "audio-stop":
                break
    except EOFError:
        pass
    finally:
        s.close()
    a = array.array("h")
    a.frombytes(pcm[: len(pcm) // 2 * 2])
    return list(a)


def wait_tcp(host, port, seconds=180):
    for _ in range(seconds):
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(1)
    return False


def start_server(binary, model, http_port, wy_port, extra):
    cmd = [binary, "--server", "-m", model, "--backend", "kokoro",
           "--host", "127.0.0.1", "--port", str(http_port),
           "--wyoming-port", str(wy_port)] + extra
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if not wait_tcp("127.0.0.1", http_port, seconds=180):
        proc.kill()
        return None, ""
    time.sleep(1.0)  # let the Wyoming listener bind
    return proc, ""


def stop(proc):
    if not proc:
        return ""
    proc.kill()
    try:
        err = proc.stderr.read().decode("utf-8", "replace")
    except Exception:
        err = ""
    proc.wait(timeout=10)
    return err


def write_wav(samples, path, rate=24000):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(array.array("h", samples).tobytes())


def watermark_confidence(binary, wav_path):
    """Run the built-in spread-spectrum detector. Returns the confidence, where
    0.5 is chance (32 sign-agreement bins) and unmarked audio averages 0.5 —
    NOT 0. Returns None if the detector could not be read."""
    r = subprocess.run([binary, "--detect-watermark", wav_path],
                       capture_output=True, timeout=300)
    out = (r.stdout + r.stderr).decode("utf-8", "replace")
    for line in out.splitlines():
        if "Watermark confidence:" in line:
            try:
                return float(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def unmarked_baseline(binary, model, out_wav):
    """A genuinely UNMARKED reference. .wav carries a C2PA manifest, so the
    watertight floor honors --no-watermark there — unlike raw PCM."""
    subprocess.run([binary, "--backend", "kokoro", "-m", model, "--tts", TEXT,
                    "--tts-output", out_wav, "--no-watermark",
                    "--accept-marking-responsibility", "--no-prints"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
    if not os.path.isfile(out_wav):
        return None
    with wave.open(out_wav, "rb") as w:
        raw = w.readframes(w.getnframes())
    a = array.array("h")
    a.frombytes(raw)
    return list(a)


def stamped_clone_pack(binary, model, path):
    """A voice pack carrying crispasr.voice.cloned_from_recording=true.

    Built by hand rather than by a baker so this test needs no cloning backend:
    the gate reads provenance from GGUF metadata, so a stamped preset is
    classified exactly as a real baked clone is (reason=pack-provenance)."""
    try:
        import gguf  # noqa
        import numpy as np
    except ImportError:
        return False
    src = None
    d = os.path.dirname(model)
    for f in sorted(os.listdir(d)):
        if f.startswith("kokoro-voice-") and f.endswith(".gguf"):
            src = os.path.join(d, f)
            break
    if not src:
        return False
    r = gguf.GGUFReader(src)
    arch = r.get_field("general.architecture").parts[-1].tobytes().decode()
    w = gguf.GGUFWriter(path, arch)
    for t in r.tensors:
        w.add_tensor(t.name, np.array(t.data))
    w.add_bool("crispasr.voice.cloned_from_recording", True)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return os.path.isfile(path)


def main():
    cache_dir = os.environ.get("CRISPASR_TEST_CACHE", "")
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--cache-dir" and i < len(sys.argv):
            cache_dir = sys.argv[i]
        elif arg.startswith("--cache-dir="):
            cache_dir = arg.split("=", 1)[1]

    binary = find_binary()
    if not binary:
        print("SKIP: crispasr binary not found")
        return 0
    model = find_tts_model(cache_dir)
    if not model:
        print("SKIP: no kokoro-*.gguf found (set CRISPASR_MODELS_DIR)")
        return 0
    print(f"Binary: {binary}\nModel:  {model}")

    tmp = tempfile.mkdtemp(prefix="wyo-marking-")
    passed = failed = 0
    proc = None
    try:
        base = unmarked_baseline(binary, model, os.path.join(tmp, "base.wav"))
        if not base:
            print("SKIP: could not synthesize the unmarked baseline")
            return 0
        print(f"Unmarked baseline: {len(base)} samples")

        # ── 1 + 3: default server (no --i-have-rights) ───────────────────────
        proc, _ = start_server(binary, model, 10406, 10407, ["--no-prints"])
        if not proc:
            print("ERROR: server did not start")
            return 2
        default_pcm = wyoming_synthesize(10407, TEXT)

        print("\n[1] watermark is embedded on the Wyoming surface")
        if not default_pcm:
            print("  x Wyoming returned no audio")
            failed += 1
        else:
            wy_wav = os.path.join(tmp, "wyoming.wav")
            write_wav(default_pcm, wy_wav)
            wy_conf = watermark_confidence(binary, wy_wav)
            base_conf = watermark_confidence(binary, os.path.join(tmp, "base.wav"))
            if wy_conf is None or base_conf is None:
                print("  SKIP: could not read the detector output")
            elif wy_conf >= WM_THRESHOLD and base_conf < WM_THRESHOLD:
                print(f"  ok detector: wyoming={wy_conf:.4f} vs unmarked baseline={base_conf:.4f}")
                passed += 1
            elif base_conf >= WM_THRESHOLD:
                # The control failed, so the comparison proves nothing either way.
                print(f"  x the UNMARKED baseline scored {base_conf:.4f} — control invalid")
                failed += 1
            else:
                print(f"  x wyoming scored {wy_conf:.4f} (< {WM_THRESHOLD}) — not watermarked")
                failed += 1

        print("\n[3] a clone with no operator --i-have-rights is refused")
        refused = wyoming_synthesize(10407, TEXT, voice="victim.wav")
        if refused:
            print(f"  x served {len(refused)} samples for an unconsented clone")
            failed += 1
        else:
            print("  ok empty audio stream")
            passed += 1
        stop(proc)
        proc = None

        # ── 2: the watertight floor overrides --no-watermark ─────────────────
        print("\n[2] --no-watermark is overridden (raw PCM carries no manifest)")
        proc, _ = start_server(binary, model, 10408, 10409,
                               ["--no-prints", "--no-watermark",
                                "--accept-marking-responsibility"])
        if not proc:
            print("ERROR: opt-out server did not start")
            return 2
        optout_pcm = wyoming_synthesize(10409, TEXT)
        if not optout_pcm:
            print("  x opt-out server returned no audio")
            failed += 1
        elif optout_pcm != default_pcm:
            print("  x differs from the default watermarked output — the floor leaked")
            failed += 1
        else:
            # Identical to the default is necessary but not sufficient: with the
            # embed call missing, BOTH are unmarked and equally identical. Ask
            # the detector whether what they agree on is actually marked.
            oo_wav = os.path.join(tmp, "optout.wav")
            write_wav(optout_pcm, oo_wav)
            oo_conf = watermark_confidence(binary, oo_wav)
            if oo_conf is not None and oo_conf >= WM_THRESHOLD:
                print(f"  ok bit-identical to the default and still marked ({oo_conf:.4f})")
                passed += 1
            else:
                print(f"  x identical to the default, but neither is marked ({oo_conf})")
                failed += 1
        stop(proc)
        proc = None

        # ── 4: a consented clone gets the audible disclosure ─────────────────
        print("\n[4] a consented clone is served WITH the spoken disclosure")
        pack = os.path.join(tmp, "cloned_voice.gguf")
        if not stamped_clone_pack(binary, model, pack):
            print("  SKIP: needs the `gguf` python package + a kokoro-voice-*.gguf")
        else:
            proc, _ = start_server(binary, model, 10410, 10411,
                                   ["--no-prints", "--i-have-rights", "test: my own voice"])
            if not proc:
                print("ERROR: consent server did not start")
                return 2
            clone_pcm = wyoming_synthesize(10411, TEXT, voice=pack)
            if not clone_pcm:
                print("  x consented clone returned no audio")
                failed += 1
            elif len(clone_pcm) <= len(default_pcm):
                print(f"  x no disclosure prepended ({len(clone_pcm)} <= {len(default_pcm)} samples)")
                failed += 1
            else:
                extra = len(clone_pcm) - len(default_pcm)
                print(f"  ok {extra} extra samples prepended (disclosure + 300 ms gap)")
                passed += 1
            stop(proc)
            proc = None

        print(f"\n{'=' * 60}\npassed={passed} failed={failed}")
        return 0 if failed == 0 else 1
    finally:
        stop(proc)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
