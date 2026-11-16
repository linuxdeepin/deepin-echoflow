#!/usr/bin/env python
"""Alternate already-validated Intel binaries to bound host timing drift."""
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import wave

from huggingface_hub import hf_hub_download

OUT = Path(os.environ['HEAVY_OUT'])
SCR = Path(os.environ['HEAVY_SCRATCH'])
OUT.mkdir(parents=True, exist_ok=True)
assert platform.system() == 'Darwin' and platform.machine() == 'x86_64'


def run(cmd, label):
    proc = subprocess.run(list(map(str, cmd)), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (OUT / f'{label}.log').write_text(proc.stdout)
    print(label, proc.returncode, proc.stdout[-1200:], flush=True)
    assert proc.returncode == 0, label
    return proc.stdout


def resources(label):
    for cmd in (['uptime'], ['sysctl', 'vm.loadavg', 'hw.memsize'], ['vm_stat']):
        run(cmd, label + '-' + cmd[0])


resources('start')
run(['sysctl', 'machdep.cpu.brand_string'], 'hardware')
proof = SCR / 'proof'
run(['gh', 'run', 'download', '36867331142', '-R', 'CrispStrobe/CrispASR',
     '-n', 'heavy-36867331142', '-D', proof], 'download-build')
original = json.loads((proof / 'result.json').read_text())
model = hf_hub_download('cstr/parakeet-tdt-0.6b-v3-GGUF', 'parakeet-tdt-0.6b-v3-q8_0.gguf',
                        revision=original['model_revision'])
with wave.open('samples/jfk.wav') as f:
    params, pcm = f.getparams(), f.readframes(f.getnframes())
with wave.open(str(SCR / 'long.wav'), 'wb') as f:
    f.setparams(params)
    f.writeframes(pcm * 6)
result = {'build_proof_run': 36867331142, 'binary_source': original['source'],
          'model_revision': original['model_revision'], 'threads': 4, 'cases': {}, 'binary_sha256': {}}
words = lambda s: re.findall(r'[a-z]+', s.lower())
want = words('And so my fellow Americans ask not what your country can do for you ask what you can do for your country')
for isa in ('legacy', 'avx2'):
    cli = proof / f'package-{isa}/crispasr'
    cli.chmod(0o755)  # upload-artifact does not preserve executable mode
    result['binary_sha256'][isa] = hashlib.sha256(cli.read_bytes()).hexdigest()
    version = run([cli, '--version'], 'version-' + isa)
    assert original['source'][:8] in version
for case, audio, count in (('short', Path('samples/jfk.wav'), 1), ('long', SCR / 'long.wav', 6)):
    entry = {'arms': {isa: [] for isa in ('legacy', 'avx2')}, 'pairs': []}
    for iteration in range(4):
        pair = {}
        # First pair warms both shapes; reverse order every other pair.
        order = ('legacy', 'avx2') if iteration % 2 == 0 else ('avx2', 'legacy')
        for isa in order:
            label = f'{case}-{iteration}-{isa}'
            resources(label)
            stem = OUT / label
            log = run([proof / f'package-{isa}/crispasr', '--backend', 'parakeet', '-m', model,
                       '-f', audio, '-l', 'en', '-t', '4', '-ng', '--output-txt', '-of', stem], label)
            heard = words(Path(str(stem) + '.txt').read_text())
            allowed = [want * count]
            if count == 6:
                allowed.append(want * 5 + want[:5] + ['and'] + want[5:])
            assert heard in allowed, heard
            seconds = float(re.search(r'transcribed [\d.]+s audio in ([\d.]+)s', log)[1])
            assert seconds > 0
            pair[isa] = {'seconds': seconds, 'word_count': len(heard)}
            entry['arms'][isa].append(seconds)
        assert pair['avx2']['word_count'] <= pair['legacy']['word_count'], pair
        pair['speedup'] = pair['legacy']['seconds'] / pair['avx2']['seconds']
        entry['pairs'].append(pair)
    entry['warm_medians_s'] = {isa: statistics.median(times[1:]) for isa, times in entry['arms'].items()}
    entry['median_speedup'] = entry['warm_medians_s']['legacy'] / entry['warm_medians_s']['avx2']
    assert all(p['speedup'] > 1.1 for p in entry['pairs'][1:]), entry
    result['cases'][case] = entry
    (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
(OUT / 'summary.md').write_text('Intel macOS alternating timing proof passed.\n\n```json\n' +
                              json.dumps(result, indent=2) + '\n```\n')
print('MACOS_INTEL_ALTERNATING_TIMING_PASS')
