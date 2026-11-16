#!/usr/bin/env python
"""Intel macOS release A/B: identical source, Accelerate, four threads, ISA only."""
import json
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import wave

import numpy as np
from huggingface_hub import hf_hub_download, model_info

REPO = Path.cwd()
OUT = Path(os.environ['HEAVY_OUT'])
SCR = Path(os.environ['HEAVY_SCRATCH'])
OUT.mkdir(parents=True, exist_ok=True)
assert platform.system() == 'Darwin' and platform.machine() == 'x86_64'


def run(cmd, label):
    cmd = list(map(str, cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (OUT / f'{label}.log').write_text(proc.stdout)
    print(label, 'rc=', proc.returncode, proc.stdout[-2500:], flush=True)
    assert proc.returncode == 0, label
    return proc.stdout


def resources(label):
    for cmd in (['uptime'], ['sysctl', 'vm.loadavg', 'hw.memsize'], ['vm_stat']):
        run(cmd, label + '-' + cmd[0])


resources('start')
run(['sysctl', 'machdep.cpu.brand_string', 'machdep.cpu.features', 'machdep.cpu.leaf7_features'], 'hardware')
if not shutil.which('ninja') or not shutil.which('ccache'):
    run(['brew', 'install', 'ninja', 'ccache'], 'tools')
run(['bash', 'scripts/fetch-c2pa.sh'], 'fetch-c2pa')
include = SCR / 'probe.cmake'
include.write_text(f'''add_executable(macos-parakeet-probe "{REPO}/tools/ci-heavy/macos_intel_parakeet_probe.cpp")
target_link_libraries(macos-parakeet-probe PRIVATE parakeet)
set_target_properties(macos-parakeet-probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/bin")
''')
for isa in ('legacy', 'avx2'):
    resources('build-' + isa)
    run(['bash', 'scripts/configure-macos-intel.sh', SCR / isa, isa,
         f'-DCMAKE_PROJECT_crispasr_INCLUDE={include}', '-DCRISPASR_C2PA_FETCH=ON'], 'configure-' + isa)
    run(['cmake', '--build', SCR / isa, '--target', 'macos-parakeet-probe', 'crispasr-cli',
         'crispasr-quantize', '-j3'], 'build-' + isa)
    shutil.copy(SCR / isa / 'CMakeCache.txt', OUT / f'{isa}-CMakeCache.txt')
    # Real package check: moved binaries must resolve the bundled c2pa sidecar.
    bundle = OUT / f'package-{isa}'
    bundle.mkdir(exist_ok=True)
    for exe in ('crispasr', 'crispasr-quantize'):
        shutil.copy(SCR / isa / 'bin' / exe, bundle / exe)
    run(['bash', 'scripts/bundle-c2pa.sh', bundle], 'bundle-' + isa)
    run(['bash', 'scripts/verify-macos-cli.sh', bundle, 'x86_64'], 'verify-' + isa)
    run(['otool', '-L', bundle / 'crispasr'], 'linkage-' + isa)
with wave.open(str(REPO / 'samples/jfk.wav')) as f:
    assert f.getframerate() == 16000 and f.getnchannels() == 1 and f.getsampwidth() == 2
    pcm = np.frombuffer(f.readframes(f.getnframes()), dtype='<i2')
(pcm.astype(np.float32) / 32768).tofile(SCR / 'jfk.f32')
# Long audio exercises the CLI's orchestrator, gap filling and streaming.
long_pcm = np.tile(pcm, 6)
with wave.open(str(SCR / 'long.wav'), 'wb') as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(16000)
    f.writeframes(long_pcm.tobytes())
result = {'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
          'threads': 4, 'long_seconds': len(long_pcm) / 16000, 'models': {}}
repo = 'cstr/parakeet-tdt-0.6b-v3-GGUF'
revision = model_info(repo).sha
result['model_revision'] = revision
expected = 'And so my fellow Americans ask not what your country can do for you ask what you can do for your country'
words = lambda s: re.findall(r'[a-z]+', s.lower())
want = words(expected)
for quant, filename in (('q8_0', 'parakeet-tdt-0.6b-v3-q8_0.gguf'),
                         ('f16', 'parakeet-tdt-0.6b-v3.gguf'),
                         ('q4_k', 'parakeet-tdt-0.6b-v3-q4_k.gguf')):
    resources('download-' + quant)
    model = hf_hub_download(repo, filename, revision=revision)
    entry = {'filename': filename, 'arms': {}}
    for isa in ('legacy', 'avx2'):
        resources(f'time-{quant}-{isa}')
        log = run([SCR / isa / 'bin/macos-parakeet-probe', model, SCR / 'jfk.f32',
                   OUT / f'{quant}-{isa}.f32'], f'probe-{quant}-{isa}')
        flag = int(isa == 'avx2')
        assert f'ISA avx2={flag} fma={flag} f16c={flag} bmi2=0 avx512=0' in log
        text = re.search(r'^TRANSCRIPT (.+)$', log, re.M)[1]
        assert words(text) == want, text
        times = list(map(float, re.findall(r'ENCODER iteration=\d ms=([\d.]+)', log)))
        assert len(times) == 4
        entry['arms'][isa] = {'transcript': text, 'encoder_ms': times,
                              'warm_median_ms': statistics.median(times[1:])}
    a, b = (np.fromfile(OUT / f'{quant}-{isa}.f32', dtype=np.float32).astype(np.float64)
            for isa in ('legacy', 'avx2'))
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    entry['parity'] = {'cosine': float(a @ b / na / nb), 'norm_legacy': float(na),
                       'norm_avx2': float(nb), 'norm_ratio': float(nb / na),
                       'relative_l2': float(np.linalg.norm(a - b) / na)}
    entry['speedup'] = entry['arms']['legacy']['warm_median_ms'] / entry['arms']['avx2']['warm_median_ms']
    assert entry['parity']['cosine'] > .999 and abs(entry['parity']['norm_ratio'] - 1) < .005
    if quant == 'q8_0':
        assert entry['speedup'] > 1.1, entry
        for isa in ('legacy', 'avx2'):
            times = []
            transcripts = []
            for iteration in range(4):
                resources(f'long-{isa}-{iteration}')
                stem = OUT / f'long-{isa}-{iteration}'
                log = run([OUT / f'package-{isa}/crispasr', '--backend', 'parakeet', '-m', model,
                           '-f', SCR / 'long.wav', '-l', 'en', '-t', '4', '-ng', '--output-txt', '-of', stem],
                          f'long-{isa}-{iteration}')
                heard = words(Path(str(stem) + '.txt').read_text())
                # The shipped legacy path inserts one 'and' in the final
                # repetition after gap filling (run 36865090158). Keep this
                # specific observed baseline variant, plus the exact golden;
                # no arbitrary missing/extra words are accepted.
                known_legacy = want * 5 + want[:5] + ['and'] + want[5:]
                assert heard in (want * 6, known_legacy), heard
                transcripts.append(heard)
                timing = re.search(r'transcribed [\d.]+s audio in ([\d.]+)s', log)
                assert timing, log[-1000:]
                times.append(float(timing[1]))
            entry['arms'][isa]['long_words'] = transcripts
            entry['arms'][isa]['long_seconds'] = times
            entry['arms'][isa]['long_warm_median_s'] = statistics.median(times[1:])
        # Candidate cannot add the known insertion if baseline was exact.
        assert max(map(len, entry['arms']['avx2']['long_words'])) <= max(map(len, entry['arms']['legacy']['long_words']))
        entry['long_speedup'] = entry['arms']['legacy']['long_warm_median_s'] / entry['arms']['avx2']['long_warm_median_s']
        assert entry['long_speedup'] > 1.1, entry
    result['models'][quant] = entry
    (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
(OUT / 'summary.md').write_text('Intel macOS Parakeet ISA A/B passed: F16/Q8/Q4 encoder magnitude/cosine,\n'
                              'golden transcripts, warmed medians, and long-audio CLI proof.\n\n```json\n' +
                              json.dumps(result, indent=2) + '\n```\n')
print('MACOS_INTEL_PARAKEET_PASS')
