"""#296: validate the FFT iSTFT — correctness cos vs the naive-DFT build + reprofile."""
import json, os, shutil, subprocess, sys, time, wave
from pathlib import Path
import numpy as np

TEMP = Path("/kaggle/temp"); OUT = Path("/kaggle/working")
REPO = TEMP / "CrispASR"; MODELS = TEMP / "models"
for d in (TEMP, OUT, MODELS): d.mkdir(parents=True, exist_ok=True)
REF = "b8e015a79"; FIX = "8ca36951f"
import traceback as _tb
def _eh(et, ev, tb):
    try: (OUT/"error.txt").write_text("".join(_tb.format_exception(et,ev,tb)))
    except Exception: pass
    sys.__excepthook__(et,ev,tb)
sys.excepthook=_eh
def run(cmd, **kw):
    kw.setdefault("capture_output", True); kw.setdefault("text", True)
    return subprocess.run(cmd, **kw)
print(json.dumps({"step":"start"}), flush=True)
if REPO.exists(): shutil.rmtree(REPO)
run(["git","clone","--depth","5","https://github.com/CrispStrobe/CrispASR.git",str(REPO)], capture_output=False)
run(["git","-C",str(REPO),"submodule","update","--init","--recursive","--depth","1"], capture_output=False, timeout=1800)
sys.path.insert(0, os.path.join(str(REPO),"tools","kaggle"))
import kaggle_harness as kh
kh.init_progress(); kh.step("cloned")
kh.install_build_toolchain()
run(["apt-get","install","-y","-q","libopenblas-dev"], capture_output=False)
JOBS=str(min(4, os.cpu_count() or 2))
def build_at(commit, bdir):
    run(["git","-C",str(REPO),"checkout","-q",commit], capture_output=False)
    r=run(["cmake","-G","Ninja","-B",str(bdir),"-S",str(REPO),"-DCMAKE_BUILD_TYPE=Release","-DCRISPASR_NO_C2PA_NATIVE=ON"]+kh.cache_and_link_flags(), capture_output=False)
    if r.returncode: kh.step(f"cfg.FAIL.{commit}"); raise SystemExit(1)
    with kh.build_heartbeat(f"build.{commit}"):
        r=run(["cmake","--build",str(bdir),"--target","crispasr-cli","-j",JOBS], capture_output=False)
    if r.returncode: kh.step(f"build.FAIL.{commit}"); raise SystemExit(1)
    cli=bdir/"bin"/"crispasr"
    if not cli.exists():
        c=[p for p in bdir.rglob("crispasr") if p.is_file() and os.access(p,os.X_OK)]; cli=c[0] if c else None
    if cli is None: kh.step(f"MISSING.{commit}"); raise SystemExit(1)
    return cli
from huggingface_hub import hf_hub_download
MODEL=Path(hf_hub_download(repo_id="cstr/mel-band-roformer-vocals-GGUF", filename="mel-band-roformer-vocals-f16.gguf", local_dir=str(MODELS)))
JFK=REPO/"samples"/"jfk.wav"; CLIP4=TEMP/"jfk4.wav"
run(["ffmpeg","-y","-i",str(JFK),"-t","4",str(CLIP4)], capture_output=False)
def sep(cli, bdir, clip, tag, profile=False):
    env={**os.environ,"LD_LIBRARY_PATH":f"{bdir}/src:"+os.environ.get("LD_LIBRARY_PATH","")}
    if profile: env["CRISPASR_MBR_PROFILE"]="1"
    od=TEMP/f"st_{tag}"; od.mkdir(exist_ok=True)
    t0=time.time(); r=run([str(cli),"--separate","-m",str(MODEL),"-f",str(clip),"--sep-output-dir",str(od)], env=env, timeout=2400)
    return time.time()-t0, od/(Path(clip).stem+"_vocals.wav"), r
def rd(p):
    with wave.open(str(p),"rb") as w: return np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype(np.float32)
def cos(a,b):
    n=min(len(a),len(b)); a,b=a[:n],b[:n]; d=np.linalg.norm(a)*np.linalg.norm(b); return float(np.dot(a,b)/d) if d>0 else 0.0
# REF (naive DFT)
rcli=build_at(REF, TEMP/"b-ref"); rt,rvoc,_=sep(rcli,TEMP/"b-ref",CLIP4,"ref"); kh.step("ref.sep",secs=round(rt,1))
# FIX (FFT iSTFT) — profiled
fcli=build_at(FIX, TEMP/"b-fix"); ft,fvoc,fr=sep(fcli,TEMP/"b-fix",CLIP4,"fix",profile=True); kh.step("fix.sep",secs=round(ft,1))
prof=[l for l in (fr.stderr or "").splitlines() if "mbr-prof" in l]
c=cos(rd(rvoc),rd(fvoc)) if (rvoc.exists() and fvoc.exists()) else None
# 11s fix
f11,v11,_=sep(fcli,TEMP/"b-fix",JFK,"fix11")
R={"clip4_ref_s":round(rt,1),"clip4_fix_s":round(ft,1),"cos_ref_vs_fix":c,"profile":prof,
   "jfk11_fix_s":round(f11,1),"jfk11_completed":v11.exists()}
(OUT/"results.json").write_text(json.dumps(R,indent=2))
print("=== profile ==="); [print("  ",l) for l in prof]
print(json.dumps({"step":"done","cos":c,"clip4_fix_s":R["clip4_fix_s"],"jfk11_s":R["jfk11_fix_s"]}), flush=True)
kh.step("done", cos=c, jfk11_s=R["jfk11_fix_s"])
