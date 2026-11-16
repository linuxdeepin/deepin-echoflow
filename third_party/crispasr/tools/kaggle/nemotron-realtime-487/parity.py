#!/usr/bin/env python3
"""PR #487 GPU production frontend: frozen-source parity and live server. Version 2."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

REF = 'fix/487-realtime'
WORK = Path('/kaggle/working')
REPO = Path('/kaggle/temp/CrispASR')
BASELINE = 'b33138b057268f573757ded8258e4afef2461ea8'
subprocess.run(['git', 'clone', '--depth', '1', '--recursive', '--branch', REF,
                'https://github.com/CrispStrobe/CrispASR', REPO], check=True)
sys.path.insert(0, str(REPO / 'tools/kaggle'))
import kaggle_harness as kh
kh.init_progress()
os.environ['HF_TOKEN'] = kh.resolve_hf_token(require=True)
kh.step('hardware', script_version=2, sha=subprocess.check_output(
    ['git', '-C', REPO, 'rev-parse', 'HEAD'], text=True).strip())
subprocess.run(['nvidia-smi'], check=True)
subprocess.run(['uptime'], check=True)
subprocess.run(['free', '-h'], check=True)
kh.install_build_toolchain()
build = REPO / 'build'
flags = kh.cuda_build_flags(kh.detect_cuda_arch()) + kh.cache_and_link_flags()
def command(cmd):
    kh.sh_with_progress(shlex.join(list(map(str, cmd))))
with kh.build_heartbeat('configure'):
    command(['cmake', '-S', REPO, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
             '-DGGML_NATIVE=OFF', '-DCRISPASR_BUILD_TESTS=ON', '-DCRISPASR_BUILD_SERVER=ON',
             '-DCRISPASR_OPUS=OFF', '-DCRISPASR_AMR=OFF'] + flags)
with kh.build_heartbeat('build'):
    command(['cmake', '--build', build, '--target', 'test-nemotron', 'crispasr-cli',
             'crispasr-quantize', '-j' + kh.safe_build_jobs(gpu=True)])
# Keep output pagination small; builds and models live in ephemeral scratch.
from huggingface_hub import hf_hub_download, HfApi
cache = kh.export_ccache_tar()
if cache:
    try:
        HfApi(token=os.environ['HF_TOKEN']).upload_file(path_or_fileobj=cache, path_in_repo='ccache.tar',
            repo_id='cstr/crispasr-ccache', repo_type='dataset', commit_message='Refresh from real Nemotron CUDA build')
        Path(cache).unlink()
        kh.step('ccache.upload.pass')
    except Exception as exc:
        # Keep the exported cache artifact; cache publishing is not a correctness gate.
        kh.step('ccache.upload.failed', error_type=type(exc).__name__)

revision = 'bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2'
f16 = hf_hub_download('cstr/nemotron-3.5-asr-streaming-GGUF',
    'nemotron-3.5-asr-streaming-0.6b-f16.gguf', revision=revision)
q8 = Path('/kaggle/temp/nemotron-q8.gguf')
command([build / 'bin/crispasr-quantize', f16, q8, 'q8_0'])
models = {'q8_0': str(q8), 'f16': f16,
          'q4_k': hf_hub_download('cstr/nemotron-3.5-asr-streaming-GGUF',
              'nemotron-3.5-asr-streaming-0.6b-q4_k.gguf', revision=revision)}
source = REPO / 'src/nemotron.cpp'
candidate = source.read_text()
command(['git', '-C', REPO, 'fetch', '--depth', '1', 'origin', BASELINE])
baseline = subprocess.check_output(['git', '-C', REPO, 'show', BASELINE + ':src/nemotron.cpp'], text=True)
results = {'source': subprocess.check_output(['git', '-C', REPO, 'rev-parse', 'HEAD'], text=True).strip(),
           'baseline': BASELINE, 'model_revision': revision, 'script_version': 2,
           'scope': 'GPU full frontend retained; incremental GPU window is not accepted', 'cases': []}
for arm, code in (('baseline', baseline), ('candidate', candidate)):
    source.write_text(code)
    with kh.build_heartbeat('build-' + arm):
        command(['cmake', '--build', build, '--target', 'test-nemotron', 'crispasr-cli',
                 '-j' + kh.safe_build_jobs(gpu=True)])
    for quant, model in models.items():
        capture = WORK / f'{arm}-{quant}.tokens'
        env = dict(os.environ, CRISPASR_MODEL_NEMOTRON=model, CRISPASR_TEST_STREAM_CAPTURE=str(capture))
        log = WORK / f'{arm}-{quant}.log'
        with kh.build_heartbeat(arm + '-' + quant), log.open('w') as output:
            proc = subprocess.run([build / 'bin/test-nemotron',
                'nemotron: GPU production stream captures complete speech'],
                cwd=REPO, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=3600)
        text = log.read_text()
        print(text[-4500:], flush=True)
        assert 'nemotron: backend = CUDA' in text, 'CPU fallback cannot prove GPU correctness'
        assert proc.returncode == 0, (arm, quant)
        if arm == 'candidate':
            assert capture.read_bytes() == (WORK / f'baseline-{quant}.tokens').read_bytes(), quant
        results['cases'].append({'arm': arm, 'quant': quant, 'complete_speech': True,
                                 'exact_baseline_tokens': arm == 'candidate'})
        (WORK / 'receipt.json').write_text(json.dumps(results, indent=2))
        kh.step('cuda.full_frontend.pass', arm=arm, quant=quant)
for mode in ('server', 'server-vad', 'server-long-turn'):
    command_args = [sys.executable, REPO / 'tests/test-server-realtime-api.py',
                    '--backend', 'nemotron', '--model', models['q8_0'], '--language', 'en']
    if mode == 'server-vad':
        command_args += ['--server-vad']
    elif mode == 'server-long-turn':
        command_args += ['--long-turn']
    with kh.build_heartbeat(mode), (WORK / (mode + '.log')).open('w') as output:
        proc = subprocess.run(list(map(str, command_args)), cwd=REPO, stdout=output,
                              stderr=subprocess.STDOUT, timeout=1200)
    print((WORK / (mode + '.log')).read_text()[-5000:], flush=True)
    assert proc.returncode == 0, mode
    results['cases'].append({'protocol': mode, 'passed': True})
    (WORK / 'receipt.json').write_text(json.dumps(results, indent=2))
print('NEMOTRON_GPU_PRODUCTION_AND_SERVER_PASS', flush=True)
