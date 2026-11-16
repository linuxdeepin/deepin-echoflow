#!/usr/bin/env python3
"""Exercise the production PowerShell downloader against real broken responses."""

import collections
import http.server
import json
import pathlib
import socket
import subprocess
import tempfile
import threading


VERSION = "1.4.341.1"
INSTALLER = bytes(range(256)) * 8
SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts/fetch-vulkan-sdk.ps1"


def check_case(mode):
    counts = collections.Counter()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            kind = "metadata" if self.path == "/latest.json" else "installer"
            if kind == "installer":
                assert self.path == f"/{VERSION}/windows/vulkan_sdk.exe", self.path
            counts[kind] += 1
            attempt = counts[kind]
            if kind == "metadata":
                if mode == "metadata_exhausted" or (mode == "recover" and attempt == 1):
                    self.send_error(503)
                    return
                payload = json.dumps({"windows": VERSION}).encode()
            else:
                if mode == "installer_exhausted" or (mode == "recover" and attempt == 1):
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(INSTALLER)))
                    self.end_headers()
                    self.wfile.write(INSTALLER[:32])
                    self.wfile.flush()
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.connection.close()
                    return
                if mode == "recover" and attempt == 2:
                    self.send_error(503)
                    return
                payload = INSTALLER
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Content-Type", "application/json" if kind == "metadata" else "application/octet-stream")
            self.end_headers()
            self.wfile.write(payload)

    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with tempfile.TemporaryDirectory(prefix="vulkan-sdk-retry-") as temp:
                output = pathlib.Path(temp) / "vulkan_sdk.exe"
                root = f"http://127.0.0.1:{server.server_port}"
                result = subprocess.run(
                    ["pwsh", "-NoLogo", "-NoProfile", "-NonInteractive", "-File", str(SCRIPT),
                     "-VersionUri", root + "/latest.json", "-DownloadRoot", root,
                     "-OutFile", str(output), "-MaxAttempts", "3", "-RetryDelaySeconds", "0"],
                    capture_output=True, text=True, timeout=45,
                )
                assert not pathlib.Path(str(output) + ".partial").exists(), result.stdout + result.stderr
                if mode == "recover":
                    assert result.returncode == 0, result.stdout + result.stderr
                    assert output.read_bytes() == INSTALLER
                    assert result.stdout.strip().endswith(VERSION), result.stdout
                    assert counts == {"metadata": 2, "installer": 3}, counts
                else:
                    assert result.returncode != 0, result.stdout + result.stderr
                    assert not output.exists(), "Failed downloads must not expose an installer"
                    expected = {"metadata": 3} if mode == "metadata_exhausted" else {"metadata": 1, "installer": 3}
                    assert counts == expected, counts
                print(f"PASS {mode}: {dict(counts)}")
        finally:
            server.shutdown()
            worker.join()


if __name__ == "__main__":
    for case in ("recover", "metadata_exhausted", "installer_exhausted"):
        check_case(case)
