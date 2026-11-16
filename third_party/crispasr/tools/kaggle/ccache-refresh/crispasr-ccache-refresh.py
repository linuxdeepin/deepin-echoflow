"""Rebuild the crispasr-ccache seed dataset as a SINGLE ccache.tar.

Both account copies of crispasr-ccache were found broken on 2026-07-20: each
held a bare, truncated `.ccache/` tree instead of a `ccache.tar`, so every CUDA
kernel warmed from ~500 stale objects and effectively built cold.

The cause was self-perpetuating and is fixed in kaggle_harness.py: CCACHE_DIR
used to live under /kaggle/working, which is the kernel-output mount that
`kaggle kernels output` serves 500 files at a time with no auto-continue. The
loose cache tree buried ccache.tar past that cap, so refreshing the dataset
downloaded a truncation, which was re-uploaded as the seed, and so on.

This kernel closes the loop: build with the fixed harness (cache on the
ephemeral layer), then emit exactly ONE artifact — /kaggle/working/ccache.tar —
so page 1 of the output contains everything needed.

The tar is compiler cache and carries no account identity, so a single run
refreshes BOTH $KAGGLE_ACCOUNT and $KAGGLE_ACCOUNT copies.

Nothing but ccache.tar (and progress.jsonl) may be written to /kaggle/working —
that is the whole point. The repo and build tree go to /kaggle/temp.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
REPO = TEMP / "CrispASR"
BUILD = TEMP / "build"
CRISPASR_REPO = "https://github.com/CrispStrobe/CrispASR.git"
CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
_T0 = time.time()


def run(cmd, check=True, env=None, cwd=None, timeout=None, capture=False):
    e = {**os.environ, **(env or {})}
    kw = dict(env=e, cwd=cwd, timeout=timeout)
    if capture:
        r = subprocess.run(cmd, **kw, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
        if r.stdout:
            print(r.stdout[-4000:], flush=True)
    else:
        r = subprocess.run(cmd, **kw)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {cmd}")
    return r


print(json.dumps({"step": "start", "ref": CRISPASR_REF}), flush=True)
if REPO.exists():
    shutil.rmtree(REPO)
run(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF, "--recursive",
     CRISPASR_REPO, str(REPO)])
run(["git", "-C", str(REPO), "submodule", "update", "--init", "--recursive"],
    check=False, timeout=1800)

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
if str(REPO / "tools" / "kaggle") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
kh.step("cloned", sha=sha)

# Fail loudly if we cloned a revision predating the CCACHE_DIR fix — otherwise
# this kernel would faithfully reproduce the very truncation it exists to undo.
if not hasattr(kh, "export_ccache_tar"):
    raise SystemExit("cloned harness has no export_ccache_tar() — need main at or "
                     "after the ccache fix; refusing to rebuild a broken dataset")

run(["nvidia-smi", "-L"], check=False)
gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              text=True).strip()
kh.step("gpu", gpu=gpu)

kh.install_build_toolchain()
ccache_dir = os.environ.get("CCACHE_DIR", "")
kh.step("ccache_dir", path=ccache_dir)
if ccache_dir.startswith("/kaggle/working"):
    raise SystemExit(f"CCACHE_DIR is {ccache_dir} — inside the output mount; the "
                     "harness fix did not take effect, aborting")

arch = kh.detect_cuda_arch()
kh.step("cuda_arch", arch=arch)
BUILD.mkdir(parents=True, exist_ok=True)
run(["cmake", "-S", str(REPO), "-B", str(BUILD),
     "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON",
     "-DCRISPASR_NO_C2PA_NATIVE=ON"]
    + kh.cuda_build_flags(arch) + kh.cache_and_link_flags())
kh.step("cmake_done")

with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli "
        f"-j{kh.safe_build_jobs(gpu=True)}")
kh.step("build_done")

# Proof of work: a real build must have produced the binary AND populated the
# cache. Exporting an empty tar would silently re-break the dataset.
cli = BUILD / "examples" / "cli" / "crispasr"
if not cli.exists():
    cands = [c for c in BUILD.rglob("crispasr") if c.is_file() and os.access(c, os.X_OK)]
    if not cands:
        raise SystemExit("build produced no crispasr binary — refusing to export a cache")
run(["ccache", "-s"], check=False, capture=True)

n_files = sum(1 for _ in Path(ccache_dir).rglob("*") if _.is_file()) if ccache_dir else 0
kh.step("ccache_populated", files=n_files)
if n_files <= 501:
    raise SystemExit(f"ccache holds only {n_files} files — that is at/below the "
                     "500-file page cap this dataset exists to escape; aborting")

tar = kh.export_ccache_tar()
if not tar or not Path(tar).exists():
    raise SystemExit("export_ccache_tar() produced nothing")

# /kaggle/working must contain essentially just the tar, or the next download
# pages past 500 files again and we recreate the original bug.
stray = sorted(p.name for p in WORK.iterdir() if p.name not in ("ccache.tar",))
kh.step("done", tar_mb=round(Path(tar).stat().st_size / 1e6, 1),
        files=n_files, working_dir_extra=stray[:20],
        elapsed_s=round(time.time() - _T0, 1))
print(json.dumps({"ccache_tar_mb": round(Path(tar).stat().st_size / 1e6, 1),
                  "cached_files": n_files, "working_dir_extra": stray}, indent=2),
      flush=True)
