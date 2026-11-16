#!/usr/bin/env python
"""test-server-ws-stream.py — integration test for the server's --ws-port
real-time WebSocket ASR streaming endpoint.

Boots `crispasr --server --ws-port 0` with a small whisper model, then with a
stdlib-only raw WebSocket client:
  1. verifies the Sec-WebSocket-Accept handshake (regression for the RFC 6455
     magic-GUID bug that made the handshake fail for every spec-compliant
     client), and
  2. streams samples/jfk.wav as binary float32 PCM and asserts a non-empty
     transcription comes back as JSON text frames.

SKIPs (exit 0) without a whisper GGUF. No third-party deps.

Usage: python tests/test-server-ws-stream.py [--port N] [--cache-dir DIR]
"""
import base64
import hashlib
import json
import os
import socket
import struct
import subprocess
import sys
import time
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def find_binary():
    for c in ["build/bin/crispasr", "build-ninja-compile/bin/crispasr", "bin/crispasr"]:
        p = os.path.join(ROOT, c)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def find_whisper(cache_dir):
    cands = []
    for d in [cache_dir, os.environ.get("CRISPASR_TEST_CACHE"), os.environ.get("CRISPASR_MODELS_DIR"),
              os.path.expanduser("~/.cache/crispasr")]:
        if d and os.path.isdir(d):
            for f in os.listdir(d):
                if f.startswith("ggml-") and f.endswith(".bin"):
                    cands.append(os.path.join(d, f))
    return cands[0] if cands else None


def load_pcm_f32_bytes(path):
    w = wave.open(path, "rb")
    n, sr = w.getnframes(), w.getframerate()
    raw = w.readframes(n)
    w.close()
    import array
    a = array.array("h")
    a.frombytes(raw)
    floats = [x / 32768.0 for x in a]
    return struct.pack("<%df" % len(floats), *floats), sr


def ws_frame(payload, opcode=0x2):
    # Client->server frame, masked per RFC 6455.
    b = bytearray([0x80 | opcode])
    n = len(payload)
    mask_bit = 0x80
    if n < 126:
        b.append(mask_bit | n)
    elif n < 65536:
        b.append(mask_bit | 126)
        b += struct.pack(">H", n)
    else:
        b.append(mask_bit | 127)
        b += struct.pack(">Q", n)
    key = os.urandom(4)
    b += key
    b += bytes(payload[i] ^ key[i % 4] for i in range(n))
    return bytes(b)


def read_server_frames(sock, timeout=4.0):
    sock.settimeout(timeout)
    buf = b""
    msgs = []
    try:
        while True:
            chunk = sock.recv(8192)
            if not chunk:
                break
            buf += chunk
            # Parse as many complete (unmasked, server->client) frames as present.
            while len(buf) >= 2:
                ln = buf[1] & 0x7F
                off = 2
                if ln == 126:
                    if len(buf) < 4:
                        break
                    ln = struct.unpack(">H", buf[2:4])[0]
                    off = 4
                elif ln == 127:
                    if len(buf) < 10:
                        break
                    ln = struct.unpack(">Q", buf[2:10])[0]
                    off = 10
                if len(buf) < off + ln:
                    break
                payload = buf[off:off + ln]
                buf = buf[off + ln:]
                opcode = buf and (buf[0] & 0x0F)  # noqa
                msgs.append(payload.decode("utf-8", "replace"))
    except socket.timeout:
        pass
    return msgs


def main():
    port = 11520
    cache_dir = os.environ.get("CRISPASR_TEST_CACHE", "")
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
        elif a.startswith("--port="):
            port = int(a.split("=", 1)[1])
        elif a == "--cache-dir" and i + 1 < len(args):
            cache_dir = args[i + 1]
        elif a.startswith("--cache-dir="):
            cache_dir = a.split("=", 1)[1]

    binary = find_binary()
    if not binary:
        print("SKIP: crispasr binary not found")
        return 0
    model = find_whisper(cache_dir)
    if not model:
        print("SKIP: no whisper ggml-*.bin found (set CRISPASR_TEST_CACHE / CRISPASR_MODELS_DIR)")
        return 0
    sample = os.path.join(ROOT, "samples/jfk.wav")
    if not os.path.isfile(sample):
        print("SKIP: samples/jfk.wav not found")
        return 0

    ws_port = port + 1
    cmd = [binary, "--server", "-m", model, "--backend", "whisper",
           "--host", "127.0.0.1", "--port", str(port), "--ws-port", "0", "--no-prints"]
    if cache_dir:
        cmd += ["--cache-dir", cache_dir]
    print("Model:", model)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        # Wait for /health.
        ready = False
        for _ in range(120):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    ready = True
                    break
            except OSError:
                time.sleep(1)
        if not ready:
            print("ERROR: server did not start")
            return 2
        time.sleep(1)  # let the WS listener bind

        passed = failed = 0

        # 1. Handshake + accept verification.
        key = base64.b64encode(os.urandom(16)).decode()
        req = ("GET / HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nUpgrade: websocket\r\n"
               "Connection: Upgrade\r\nSec-WebSocket-Key: %s\r\n"
               "Sec-WebSocket-Version: 13\r\n\r\n" % (ws_port, key))
        s = socket.create_connection(("127.0.0.1", ws_port), timeout=5)
        s.sendall(req.encode())
        resp = s.recv(4096).decode("utf-8", "replace")
        accept = ""
        for line in resp.split("\r\n"):
            if line.lower().startswith("sec-websocket-accept:"):
                accept = line.split(":", 1)[1].strip()
        expected = base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()
        if "101" in resp.split("\r\n")[0] and accept == expected:
            print("  ✓ handshake Sec-WebSocket-Accept correct (RFC 6455 GUID)")
            passed += 1
        else:
            print("  ✗ bad handshake: status=%r accept=%r expected=%r" % (resp.split(chr(13))[0], accept, expected))
            failed += 1
            s.close()
            print("\nRESULT: %d passed, %d failed" % (passed, failed))
            return 1

        # 2. Stream PCM, collect text frames.
        pcm, sr = load_pcm_f32_bytes(sample)
        chunk = 16000 * 4  # ~1s of float32
        msgs = []
        for i in range(0, len(pcm), chunk):
            s.sendall(ws_frame(pcm[i:i + chunk], opcode=0x2))
            msgs += read_server_frames(s, timeout=0.4)
        msgs += read_server_frames(s, timeout=4.0)
        s.close()

        texts = []
        for m in msgs:
            try:
                texts.append(json.loads(m).get("text", ""))
            except Exception:
                pass
        joined = " ".join(t.strip() for t in texts if t.strip())
        print("  transcription:", joined[:160])
        if joined and len(joined) > 5:
            print("  ✓ streaming returned non-empty transcription")
            passed += 1
        else:
            print("  ✗ no transcription text received")
            failed += 1

        print("\nRESULT: %d passed, %d failed" % (passed, failed))
        return 0 if failed == 0 else 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
