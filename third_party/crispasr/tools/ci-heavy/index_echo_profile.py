#!/usr/bin/env python3
"""Same-host CPU benchmark, accepted only with complete output parity.

Each arm runs in a separate process to isolate peak RSS and free the F32
blueprint before loading GGUF. Call after index_echo_validate passes.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH']) / 'index-echo-profile'
parser = argparse.ArgumentParser()
parser.add_argument('--worker', choices=['python', 'f16', 'q8_0'])
parser.add_argument('--iterations', type=int, default=3)
args = parser.parse_args()
assert args.iterations >= 2, 'Need first-call and warm measurements'
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(SCRATCH)
os.environ['OMP_NUM_THREADS'] = '4'


def run(*command, **kwargs):
    subprocess.run(list(map(str, command)), cwd=ROOT, check=True, **kwargs)


def blueprint(source):
    spec = importlib.util.spec_from_file_location('released_index_echo', source / 'infer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if not args.worker:
    from huggingface_hub import HfApi, snapshot_download
    from index_echo_produce_constants import SOURCE, REVISION, DESTINATION
    build = SCRATCH / 'build'
    run('cmake', '-S', ROOT, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
        '-DBUILD_SHARED_LIBS=ON', '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF', '-DGGML_NATIVE=OFF')
    run('cmake', '--build', build, '--target', 'crispasr-lib', '-j', '4')
    model_revision = HfApi().model_info(DESTINATION).sha
    snapshot_download(SOURCE, revision=REVISION, local_dir=SCRATCH / 'source')
    snapshot_download(DESTINATION, revision=model_revision, local_dir=SCRATCH / 'models',
        allow_patterns=['*f16.gguf', '*q8_0.gguf', 'reference/*'])
    os.environ['INDEX_ECHO_PROFILE_LIB'] = str(next(build.rglob('libcrispasr.so')))
    provenance = dict(source=SOURCE, source_revision=REVISION, model_revision=model_revision,
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        threads=4, native_build=False, cpu=subprocess.check_output(['lscpu'], text=True))
    (OUT / 'profile-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    for arm in ['python', 'f16', 'q8_0']:
        with (OUT / f'profile-{arm}.log').open('w') as log:
            run(sys.executable, __file__, '--worker', arm, '--iterations', args.iterations,
                stdout=log, stderr=subprocess.STDOUT)
    results = {arm: json.loads((OUT / f'profile-{arm}.json').read_text()) for arm in ['python', 'f16', 'q8_0']}
    for arm in ['f16', 'q8_0']:
        for clip, iterations in results[arm]['clips'].items():
            gold = results['python']['clips'][clip][0]['segments']
            for iteration in iterations:
                actual = iteration['segments']
                assert len(actual) == len(gold), (arm, clip, actual, gold)
                assert all(a['text'] == g['text'] and abs(a['start']-g['start']) <= .0051 and
                    abs(a['end']-g['end']) <= .0051 for a,g in zip(actual,gold)), (arm,clip,actual,gold)
    (OUT / 'profile.json').write_text(json.dumps(dict(provenance=provenance, results=results), indent=2, ensure_ascii=False)+'\n')
    (OUT / 'summary.md').write_text('Same-host CPU F32 blueprint / native F16 / native Q8 profile: all timed outputs match.\n')
    sys.exit(0)

import numpy as np
import soundfile as sf
from gguf import GGUFReader
source = SCRATCH / 'source'
models = SCRATCH / 'models'
module = blueprint(source)
if args.worker == 'python':
    import torch
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    started = time.perf_counter()
    model = module.AudioTransModel(str(source), device='cpu', dtype=torch.float32)
    sys.path.insert(0, str(ROOT / 'tools'))
    from reference_backends.index_echo import precision_audit
    parameter_dtypes = precision_audit(model)
else:
    sys.path.insert(0, str(ROOT / 'python'))
    from crispasr import Session
    os.environ['INDEX_ECHO_BENCH'] = '1'
    started = time.perf_counter()
    model = Session(str(models / f'index-echo-2b-{args.worker}.gguf'),
                    lib_path=os.environ['INDEX_ECHO_PROFILE_LIB'], n_threads=4)
result = dict(arm=args.worker, precision='CPU F32' if args.worker=='python' else args.worker,
              load_seconds=time.perf_counter()-started, clips={})
if args.worker == 'python': result['parameter_dtypes'] = parameter_dtypes
try:
    for clip in ['jfk','zh','jfk-tail']:
        audio = models/'reference/jfk-tail.wav' if clip=='jfk-tail' else ROOT/'samples'/('paraformer_zh.wav' if clip=='zh' else 'jfk.wav')
        pcm, rate = sf.read(audio, dtype='float32')
        assert rate == 16000 and pcm.ndim == 1
        reader = GGUFReader(models/f'reference/{clip}-ref.gguf')
        text = reader.fields['crispasr.ref.generated_text'].contents()
        parsed, warnings = module.parse_hyp(text)
        assert warnings == 0
        golden = [dict(start=c['st'],end=c['et'],text=c['zh']+'\n'+(c['en'] or '')) for c in parsed]
        del reader
        iterations=[]
        for i in range(args.iterations):
            started=time.perf_counter()
            if args.worker=='python':
                raw,_=model.translate_window(str(audio), [])
                cues,warnings=module.parse_hyp(raw)
                assert warnings==0
                segments=[dict(start=c['st'],end=c['et'],text=c['zh']+'\n'+(c['en'] or '')) for c in cues]
            else:
                segments=[dict(start=c.start,end=c.end,text=c.text) for c in model.transcribe(pcm)]
            elapsed=time.perf_counter()-started
            assert len(segments)==len(golden) and all(a['text']==g['text'] and abs(a['start']-g['start'])<=.0051 and
                abs(a['end']-g['end'])<=.0051 for a,g in zip(segments,golden)), (args.worker,clip,segments,golden)
            iterations.append(dict(iteration=i,seconds=elapsed,audio_seconds=len(pcm)/rate,
                                   realtime_factor=elapsed/(len(pcm)/rate),segments=segments))
            print(args.worker,clip,i,elapsed,flush=True)
        result['clips'][clip]=iterations
finally:
    if args.worker!='python': model.close()
    result['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (OUT/f'profile-{args.worker}.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
