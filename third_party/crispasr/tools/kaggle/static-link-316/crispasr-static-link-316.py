# %% [markdown]
# # #316 static-link fix: does crispasr-diff link with BUILD_SHARED_LIBS=OFF?
# Reproduces the AUR/Arch failure (static build) that couldn't link crispasr-diff
# because phonemizer.cpp -> crispasr_cache::ensure_cached_file was unresolved.
# Fix moved crispasr_cache to crispasr-core. GGML_CUDA=OFF (the link bug is CUDA-
# independent); CPU-only, static libs, build the crispasr-diff target end-to-end.
# %% [code]
import os, subprocess, sys
from pathlib import Path
REPO = Path("/kaggle/temp/CrispASR"); WORK = Path("/kaggle/working")
REF = os.environ.get("CRISPASR_REF", "fix/316-static-link")
if not REPO.exists():
    subprocess.check_call(["git","clone","--recursive","--depth","1","--branch",REF,
                           "https://github.com/CrispStrobe/CrispASR.git",str(REPO)])
    subprocess.check_call(["git","-C",str(REPO),"submodule","update","--init","--recursive","--depth","1"],timeout=1800)
sys.path.insert(0,str(REPO/"tools"/"kaggle"))
import kaggle_harness as kh  # noqa: E402
kh.init_progress(hf_progress_repo="cstr/crispasr-kaggle-progress"); step=kh.step
step("start", ref=REF)
kh.resolve_hf_token("HF_TOKEN"); kh.install_build_toolchain()
BUILD = REPO/"build-static"
step("cmake.configure")
r = subprocess.run(["cmake","-G","Ninja","-B",str(BUILD),"-S",str(REPO),
     "-DCMAKE_BUILD_TYPE=Release","-DBUILD_SHARED_LIBS=OFF","-DGGML_CUDA=OFF","-DGGML_VULKAN=OFF",
     "-DCRISPASR_BUILD_TESTS=OFF","-DCRISPASR_BUILD_EXAMPLES=ON"]+kh.crispasr_cmake_flags(),
     capture_output=True, text=True)
step("cmake.configure.done", rc=r.returncode, tail=(r.stderr or r.stdout)[-400:] if r.returncode else "ok")
if r.returncode:
    (WORK/"result.json").write_text('{"verdict":"CONFIGURE FAILED"}'); print("CONFIGURE FAILED", flush=True); raise SystemExit
ok=False; tail=""
with kh.build_heartbeat("build.crispasr-diff"):
    b = subprocess.run(f"cmake --build {BUILD} --target crispasr-diff -j{kh.safe_build_jobs(gpu=False)}",
                       shell=True, capture_output=True, text=True, timeout=7200)
ok = b.returncode==0 and any((BUILD/'bin'/'crispasr-diff').exists() for _ in [0])
tail = (b.stdout+b.stderr)[-1200:]
verdict = "STATIC LINK OK — crispasr-diff built" if ok else "STILL FAILS — see tail"
import json
(WORK/"result.json").write_text(json.dumps({"verdict":verdict,"rc":b.returncode,
    "diff_exists":(BUILD/'bin'/'crispasr-diff').exists(),
    "undef_cache":"ensure_cached_file" in tail}))
step("done", verdict=verdict, rc=b.returncode, tail=tail[-500:])
print("DONE", verdict, flush=True)
