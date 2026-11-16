#!/usr/bin/env python3
"""Native CUDA versus independent released mixed-precision CPU oracle and same-box CPU control.

No torch execution: P100 can exercise native CUDA, but its quantized matmul
uses dequant+cuBLAS rather than integer MMQ. Record that coverage limit.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

SCRIPT_VERSION = '2026-10-01.1'
SOURCE_COMMIT = '5e813d39bce0e79b53f956a5715a006108097fb8'
REPO = Path('/kaggle/temp/index-echo-repo')
WORK = Path('/kaggle/working')
MODELS = Path('/kaggle/temp/index-echo-models')
WORK.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)


def run(*command, **kwargs):
    subprocess.run(list(map(str, command)), check=True, **kwargs)


# Read actual hardware before model pulls or builds. No requested-arch guess.
hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,compute_cap,memory.total',
                                    '--format=csv,noheader'], text=True).strip()
print('actual GPU:', hardware, flush=True)
if not hardware:
    raise RuntimeError('GPU session did not provide a CUDA device')
run('git', 'init', REPO)
run('git', '-C', REPO, 'remote', 'add', 'origin', 'https://github.com/CrispStrobe/CrispASR.git')
run('git', '-C', REPO, 'fetch', '--depth=1', 'origin', SOURCE_COMMIT)
run('git', '-C', REPO, 'checkout', 'FETCH_HEAD')
run('git', '-C', REPO, 'submodule', 'update', '--init', '--recursive', '--depth=1')
sys.path.insert(0, str(REPO/'tools/kaggle'))
import kaggle_harness as kh
kh.init_progress()
kh.provenance(SCRIPT_VERSION, REPO)
run(sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub', 'gguf', 'numpy')
from huggingface_hub import HfApi, snapshot_download
from gguf import GGUFReader
import numpy as np
os.environ['HF_TOKEN'] = kh.resolve_hf_token(require=True)
api = HfApi()
revision = api.model_info('cstr/index-echo-2b-GGUF').sha
kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
build = REPO/'build'
flags = ['-DCMAKE_BUILD_TYPE=Release', '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF']
flags += kh.cuda_build_flags(arch) + kh.cache_and_link_flags()
run('cmake', '-S', REPO, '-B', build, '-G', 'Ninja', *flags)
with kh.build_heartbeat('build.cuda', interval_s=30):
    kh.sh_with_progress(f'cmake --build {build} --target crispasr-cli crispasr-diff -j{kh.safe_build_jobs(gpu=True)}', cwd=str(REPO))
result = dict(script_version=SCRIPT_VERSION, source_commit=SOURCE_COMMIT,
              model_revision=revision, hardware=hardware, cuda_arch=arch,
              quantized_integer_mmq=int(arch) >= 61, reference_device='CPU F32 tower/connector, BF16 decoder', cases=[])
failed = []
for cohort in ['f16', 'q8_0']:
    snapshot_download('cstr/index-echo-2b-GGUF', revision=revision, local_dir=MODELS,
        allow_patterns=[f'index-echo-2b-{cohort}.gguf',f'index-echo-2b-decoder-{cohort}.gguf',
                        'reference/*-ref.gguf','reference/jfk-tail.wav'])
    primary = MODELS/f'index-echo-2b-{cohort}.gguf'
    for clip in ['jfk','zh','jfk-tail']:
        audio = MODELS/'reference/jfk-tail.wav' if clip=='jfk-tail' else REPO/'samples'/('paraformer_zh.wav' if clip=='zh' else 'jfk.wav')
        reader = GGUFReader(MODELS/f'reference/{clip}-ref.gguf')
        golden = reader.fields['crispasr.ref.generated_text'].contents()
        del reader
        expected = [line.strip() for line in golden.splitlines() if line.strip() and not line.strip().startswith('[')]
        for gpu in [False,True]:
            tag = f'{cohort}-{clip}-'+('cuda' if gpu else 'cpu')
            env = dict(os.environ, INDEX_ECHO_BENCH='1')
            env.pop('CRISPASR_DIFF_NO_GPU',None)
            if not gpu: env['CRISPASR_DIFF_NO_GPU']='1'
            with (WORK/f'{tag}-diff.log').open('w') as log, kh.build_heartbeat(tag+'.diff',interval_s=30):
                diff = subprocess.run([str(build/'bin/crispasr-diff'),'index-echo',str(primary),
                    str(MODELS/f'reference/{clip}-ref.gguf'),str(audio)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
            prefix=WORK/tag
            command=[str(build/'bin/crispasr'),'-m',str(primary),'-f',str(audio),'-l','auto','-t','4','-osrt','-of',str(prefix)]
            if not gpu: command.append('-ng')
            started=time.perf_counter()
            with (WORK/f'{tag}-cli.log').open('w') as log, kh.build_heartbeat(tag+'.decode',interval_s=30):
                decode=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
            seconds=time.perf_counter()-started
            srt=prefix.with_suffix('.srt')
            actual=[]
            if srt.exists():
                actual=[line.strip() for line in srt.read_text().splitlines() if line.strip()
                        and not line.strip().isdigit() and '-->' not in line]
            ok=diff.returncode==0 and decode.returncode==0 and actual==expected
            result['cases'].append(dict(tag=tag,stage_rc=diff.returncode,decode_rc=decode.returncode,
                                        decoded_match=actual==expected,seconds_including_load=seconds,actual=actual,expected=expected))
            if not ok: failed.append(tag)
            (WORK/'index-echo-cuda.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
            kh.step(tag,passed=ok,seconds=seconds)
    # Keep only the current cohort on the ephemeral disk.
    primary.unlink()
    (MODELS/f'index-echo-2b-decoder-{cohort}.gguf').unlink()
result['failed']=failed
result['conclusive']=not failed
(WORK/'index-echo-cuda.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
if failed: raise RuntimeError('CUDA/control stage or decoded parity failed: '+', '.join(failed))
