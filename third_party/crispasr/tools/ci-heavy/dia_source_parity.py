#!/usr/bin/env python3
"""Pinned official F32 Dia feedback/logit audit; no framework DAC interception of neural math."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH'])
PIN = '4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8'
MODEL_PIN = '257bc72f9b78182ccc6fa07675a9ae4c1a44e2cd'
TEXT = '[S1] Hello there, how are you doing today? I really hope you are having a wonderful and pleasant time. The weather outside is lovely and bright.'
p = argparse.ArgumentParser()
p.add_argument('--child', choices=('reference', 'native'))
p.add_argument('--model')
p.add_argument('--lib')
p.add_argument('--reference-run')
p.add_argument('--feedback', action='store_true')
p.add_argument('--quant', choices=('f16', 'q8_0'), default='f16')
a = p.parse_args()
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)

if a.child == 'reference':
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    torch.set_num_threads(4)
    source = SCRATCH / 'dia-source'
    subprocess.run(['git', 'clone', 'https://github.com/nari-labs/dia', source], check=True)
    subprocess.run(['git', '-C', source, 'checkout', PIN], check=True)
    # No DAC is needed for a decoder-logit reference. The official neural code
    # and generation loop run unchanged; only its final codec call is skipped.
    sys.modules['dac'] = types.ModuleType('dac')
    sys.path.insert(0, str(source))
    from dia.config import DiaConfig
    from dia.layers import DiaModel
    from dia.model import Dia
    config_path = hf_hub_download('nari-labs/Dia-1.6B', 'config.json', revision=MODEL_PIN)
    weights_path = hf_hub_download('nari-labs/Dia-1.6B', 'model.safetensors', revision=MODEL_PIN)
    config = DiaConfig.load(config_path)
    with torch.device('meta'):
        model = DiaModel(config, torch.float32)
    state = load_file(weights_path, device='cpu')
    model.load_state_dict(state, assign=True, strict=True)
    del state
    for module in model.modules():
        if hasattr(module, 'timescale'):
            fraction = 2.0 * torch.arange(module.embedding_dims // 2) / module.embedding_dims
            module.timescale = (module.min_timescale * (module.max_timescale / module.min_timescale) ** fraction).float()
    assert all(t.dtype == torch.float32 and t.device.type == 'cpu' for t in model.parameters())
    assert all(t.device.type == 'cpu' for t in model.buffers())
    model.eval()
    dia = Dia.__new__(Dia)
    dia.config, dia.model, dia.device, dia.compute_dtype, dia.dac_model = config, model, torch.device('cpu'), torch.float32, None
    inputs, logits = [], []
    original = model.decoder.decode_step
    def capture(tokens, state):
        output = original(tokens, state)
        inputs.append(tokens[0].detach().cpu().numpy().astype('<i4'))
        logits.append(output[:, -1].detach().float().cpu().numpy())
        return output
    model.decoder.decode_step = capture
    dia._generate_output = lambda codes: codes.detach().cpu().numpy()
    torch.manual_seed(42)
    with torch.inference_mode():
        codes = dia.generate(TEXT, max_tokens=128, temperature=1.2, cfg_scale=3., top_p=.95,
                             cfg_filter_top_k=45, use_torch_compile=False, verbose=True)
    np.stack(inputs).tofile(OUT / 'inputs.i32')
    np.stack(logits).astype('<f4').tofile(OUT / 'reference-logits.f32')
    np.save(OUT / 'reference-codes.npy', codes)
    (OUT / 'reference.json').write_text(json.dumps({'source_pin': PIN, 'model_pin': MODEL_PIN,
        'text': TEXT, 'steps': len(inputs), 'shape': np.stack(logits).shape, 'precision': 'F32', 'torch_version': torch.__version__}, indent=2))
    print('DIA_REFERENCE_CAPTURED', len(inputs), flush=True)
    sys.exit(0)

if a.child == 'native':
    if a.feedback:
        os.environ['CRISPASR_DIA_FORCE_OUTPUT_TOKENS'] = str(OUT / 'outputs.i32')
        os.environ['CRISPASR_DIA_DUMP_INPUT_TOKENS'] = str(OUT / 'feedback-inputs.i32')
    else:
        os.environ['CRISPASR_DIA_FORCE_TOKENS'] = str(OUT / 'inputs.i32')
    os.environ['CRISPASR_DIA_DUMP_STEPLOGITS'] = str(OUT / 'native-logits.f32')
    sys.path.insert(0, str(ROOT / 'python'))
    from crispasr import Session
    with Session(a.model, lib_path=a.lib, backend='dia', n_threads=4) as session:
        session.synthesize(TEXT)
    sys.exit(0)

from huggingface_hub import hf_hub_download

def run(cmd, label):
    with (OUT / (label + '.log')).open('w') as log:
        proc = subprocess.run(list(map(str, cmd)), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
    print(label, proc.returncode, (OUT / (label + '.log')).read_text()[-3000:], flush=True)
    assert proc.returncode == 0, label

subprocess.run(['uptime'], check=True)
subprocess.run(['free', '-h'], check=True)
if a.reference_run:
    archive = SCRATCH / 'reference-artifact'
    run(['gh', 'run', 'download', a.reference_run, '-R', 'CrispStrobe/CrispASR', '-D', archive], 'reference-download')
    folder = next(archive.glob('heavy-*'))
    for name in ('reference.json', 'inputs.i32', 'reference-logits.f32', 'reference-codes.npy', 'reference.log'):
        shutil.copyfile(folder / name, OUT / name)
    provenance = json.loads((OUT / 'reference.json').read_text())
    assert provenance['source_pin'] == PIN and provenance['model_pin'] == MODEL_PIN
else:
    run([sys.executable, __file__, '--child', 'reference'], 'reference')
build = SCRATCH / 'dia-build'
run(['cmake', '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON',
    '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF', '-DGGML_BLAS=OFF',
    '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_OPUS=OFF', '-DCRISPASR_AMR=OFF'], 'configure')
run(['cmake', '--build', build, '--target', 'crispasr-lib', '-j4'], 'build')
lib = next(build.rglob('libcrispasr.so'))
revision = '3233fbcb32be47761d2e736857b6d1a075b9ba7e'
model = hf_hub_download('cstr/dia-1.6b-GGUF', f'dia-1.6b-{a.quant}.gguf', revision=revision)
hf_hub_download('cstr/dia-1.6b-GGUF', 'dac-44khz.gguf', revision=revision)
run([sys.executable, __file__, '--child', 'native', '--model', model, '--lib', lib], 'native')
shape = json.loads((OUT / 'reference.json').read_text())['shape']
ref = np.fromfile(OUT / 'reference-logits.f32', dtype='<f4').reshape(shape).astype(np.float64)
actual = np.fromfile(OUT / 'native-logits.f32', dtype='<f4').reshape(shape).astype(np.float64)
cos = np.sum(ref * actual, axis=-1) / np.sqrt(np.sum(ref**2, axis=-1) * np.sum(actual**2, axis=-1))
ratio = np.sqrt(np.sum(actual**2, axis=-1) / np.sum(ref**2, axis=-1))
receipt = {'quant': a.quant, 'cosine': cos.tolist(), 'norm_ratio': ratio.tolist(),
           'minimum_cosine': float(cos.min()), 'max_norm_error': float(abs(ratio-1).max())}
(OUT / 'parity.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('DIA_SOURCE_PARITY', receipt['minimum_cosine'], receipt['max_norm_error'], flush=True)
assert cos.min() >= .9995 and abs(ratio-1).max() < .02, receipt

# Test real delay feedback with official next-input tokens as forced outputs.
inputs = np.fromfile(OUT / 'inputs.i32', dtype='<i4').reshape(-1, 9)
inputs[1:].tofile(OUT / 'outputs.i32')
native_command = [sys.executable, __file__, '--child', 'native', '--model', model, '--lib', lib, '--feedback']
source_path = ROOT / 'src/dia_tts.cpp'
candidate = source_path.read_text()
before = """                if (bos_countdown > 0)
                    bos_countdown--;
                const bool apply_mask = (bos_countdown > 0);"""
after = """                const bool apply_mask = (bos_countdown > 0);"""
assert before in candidate
wrong = candidate.replace(before, after, 1)
marker = """            if (stop) {"""
assert marker in wrong
wrong = wrong.replace(marker, """            if (!dia_force && bos_countdown > 0)
                bos_countdown--;

""" + marker, 1)
try:
    source_path.write_text(wrong)
    run(['cmake', '--build', build, '--target', 'crispasr-lib', '-j4'], 'old-bos-build')
    run(native_command, 'old-bos-feedback')
    observed = np.fromfile(OUT / 'feedback-inputs.i32', dtype='<i4').reshape(-1, 9)
    mismatch = np.argwhere(observed != inputs[:-1])
    assert mismatch.tolist() == [[15, 8]], mismatch
    assert observed[15, 8] == 1026 and inputs[15, 8] == 890
    shutil.copyfile(OUT / 'feedback-inputs.i32', OUT / 'old-bos-inputs.i32')
finally:
    source_path.write_text(candidate)
run(['cmake', '--build', build, '--target', 'crispasr-lib', '-j4'], 'fixed-bos-build')
run(native_command, 'fixed-bos-feedback')
observed = np.fromfile(OUT / 'feedback-inputs.i32', dtype='<i4').reshape(-1, 9)
assert np.array_equal(observed, inputs[:-1]), np.argwhere(observed != inputs[:-1])
receipt['feedback'] = {'steps': len(observed), 'exact_source_inputs': True,
                       'old_mismatch': [[15, 8]], 'old': 1026, 'reference': 890}
(OUT / 'parity.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('DIA_SOURCE_LOGITS_AND_FEEDBACK_PASS', flush=True)
