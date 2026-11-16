"""Capture Phonon-2 encoder stages and gate an FFN experiment against ggml.

Run each mode in a separate process. Supply the same independent reference mel
and quantized model to both; --baseline checks cosine AND magnitude per stage.
This is an A/B kernel check, complementary to the upstream F16 parity gate.
"""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path

import numpy as np
from gguf import GGUFReader


class Params(ct.Structure):
    _fields_ = [('n_threads', ct.c_int), ('use_flash', ct.c_bool),
                ('verbosity', ct.c_int), ('use_gpu', ct.c_bool)]


def compare_stages(stages, baseline):
    rows = []
    for name, values in stages.items():
        ref, got = baseline[name].astype(np.float64), values.astype(np.float64)
        assert got.shape == ref.shape and np.isfinite(got).all()
        ref_norm, got_norm = np.linalg.norm(ref, axis=1), np.linalg.norm(got, axis=1)
        valid = ref_norm > 1e-12
        assert np.all(got_norm[valid] > 0)
        cosine = np.sum(ref[valid] * got[valid], axis=1) / (ref_norm[valid] * got_norm[valid])
        relative_rms = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
        row = dict(stage=name, min_cosine=float(cosine.min()),
                   max_norm_ratio_error=float(np.max(np.abs(got_norm[valid] / ref_norm[valid] - 1))),
                   relative_rms_error=relative_rms)
        row['passed'] = row['min_cosine'] >= .999 and row['max_norm_ratio_error'] < .01 and relative_rms < .01
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lib', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--mode', choices=['ggml', 'repack', 'blas'], required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--report-only', action='store_true',
                        help='retain failed diagnostics without accepting the experiment')
    args = parser.parse_args()
    assert args.threads > 0
    model = GGUFReader(args.model)
    for key, expected in [('n_mels', 128), ('d_model', 1024),
                          ('n_layers', 24), ('subsampling_factor', 8)]:
        field = model.get_field('parakeet.' + key)
        assert field is not None and field.contents() == expected, key
    del model
    os.environ['CRISPASR_PARAKEET_FFN'] = args.mode
    os.environ['CRISPASR_PARAKEET_ENCODER_BLAS'] = '0'
    lib = ct.CDLL(args.lib)
    fp = ct.POINTER(ct.c_float)
    ip = ct.POINTER(ct.c_int)
    lib.parakeet_context_default_params.restype = Params
    lib.parakeet_init_from_file.argtypes = [ct.c_char_p, Params]
    lib.parakeet_init_from_file.restype = ct.c_void_p
    lib.parakeet_free.argtypes = [ct.c_void_p]
    lib.parakeet_run_encoder_dump.argtypes = [ct.c_void_p, fp, ct.c_int, ct.c_int,
                                             ct.POINTER(fp), ct.c_int, ip, ip]
    lib.parakeet_run_encoder_dump.restype = ct.c_int
    mel = next(t.data for t in GGUFReader(args.reference).tensors if t.name == 'mel_spectrogram')
    mel = np.ascontiguousarray(mel.reshape(-1, 128), dtype=np.float32)
    params = lib.parakeet_context_default_params()
    params.n_threads, params.use_gpu, params.verbosity = args.threads, False, 0
    ctx = lib.parakeet_init_from_file(os.fsencode(args.model), params)
    assert ctx, 'model load failed'
    try:
        capacity = (mel.shape[0] + 7) // 8
        arrays = [np.empty((capacity, 1024), dtype=np.float32) for _ in range(25)]
        pointers = (fp * len(arrays))(*(arr.ctypes.data_as(fp) for arr in arrays))
        frames, width = ct.c_int(), ct.c_int()
        status = lib.parakeet_run_encoder_dump(ctx, mel.ctypes.data_as(fp), 128, mel.shape[0],
                                               pointers, len(arrays), ct.byref(frames), ct.byref(width))
        assert status == 0 and 0 < frames.value <= capacity and width.value == 1024
        stages = {'pre_encode_output': arrays[0][:frames.value]}
        stages.update({f'encoder_layer_{i}': arrays[i + 1][:frames.value] for i in range(24)})
        np.savez(args.output, **stages)
    finally:
        lib.parakeet_free(ctx)
    if args.baseline:
        with np.load(args.baseline) as baseline:
            assert set(baseline.files) == set(stages)
            rows = compare_stages(stages, baseline)
        report = dict(mode=args.mode, threads=args.threads, model=args.model,
                      reference=args.reference, baseline=str(args.baseline), stages=rows)
        report['passed'] = all(row['passed'] for row in rows)
        args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
        if not args.report_only:
            assert report['passed'], 'FFN stage parity gate failed; see complete JSON report'


if __name__ == '__main__':
    main()
