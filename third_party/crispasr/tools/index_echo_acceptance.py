#!/usr/bin/env python3
"""Audit retained Index-Echo receipts; never replace exact diagnostic checks.

The released wrapper uses a BF16 decoder; an independently forced F32 source
variant exposes its rounding-sensitive choices. F16 must match one source
variant exactly for each complete case. Q8 allows only 20ms timestamp drift
and must preserve every text line from one complete independent source case.
No edit-distance allowance, transcript truncation or arbitrary synonym list.
"""
import argparse
import hashlib
import json
from pathlib import Path


def compare_case(actual, variants, cohort):
    tolerance = .0051 if cohort == 'f16' else .0201
    for label, expected in variants.items():
        if not expected or len(actual) != len(expected):
            continue
        if any(a['text'] != e['text'] for a, e in zip(actual, expected)):
            continue
        drift = max(abs(a[key] - e[key]) for a, e in zip(actual, expected)
                    for key in ('start', 'end'))
        if drift <= tolerance:
            return dict(source_variant=label, max_timestamp_error_seconds=drift,
                        timestamp_tolerance_seconds=tolerance, exact_text=True)
    raise ValueError('Decoded output exceeds the declared source-precision/quantization bounds')


def audit(validation, f32_validation, released_path, f32_path, dtype_path):
    def read(path):
        return json.loads(path.read_text())
    released, f32, dtype = map(read, (released_path, f32_path, dtype_path))
    if dtype['embedding_dtype'] != 'torch.bfloat16':
        raise ValueError('Released-source dtype audit must identify BF16 decoder')
    if f32['parameter_dtypes']['embedding_dtype'] != 'torch.float32':
        raise ValueError('Independent F32 variant must identify F32 decoder')
    required = {'jfk-en', 'zh-en', 'zh-ja', 'zh-es', 'multi-en'}
    if set(released['cases']) != required or set(f32['cases']) != required:
        raise ValueError('All target languages and two-window context fixture are mandatory')
    provenance = read(validation / 'validation-provenance.json')
    receipts = {}
    f32_cohorts = f32_validation / 'cohort-results.json'
    f32_stages = f32_validation / 'stage-results-f16.json'
    if read(f32_cohorts)['f16'] or read(f32_stages)['failed']:
        raise ValueError('F16 must pass every exact F32 source stage/cache/direct/full-file gate')
    paths = [released_path, f32_path, dtype_path, validation / 'validation-provenance.json',
             f32_cohorts, f32_stages, f32_validation / 'validation-provenance.json']
    for cohort in ('f16', 'q8_0'):
        stage = validation / f'stage-results-{cohort}.json'
        paths.append(stage)
        if read(stage)['failed']:
            raise ValueError('Mandatory stage/magnitude/cache gates failed: ' + cohort)
        # The exact direct-window check remains mandatory, including timestamps.
        direct = validation / f'decoded-{cohort}.json'
        paths.append(direct)
        captures = read(direct)
        if set(captures) != {'jfk', 'zh', 'jfk-tail'}:
            raise ValueError('Missing direct-window cases')
        exact_failures = read(validation / 'cohort-results.json')[cohort]
        if any('full-pipeline decoded mismatch' not in failure for failure in exact_failures):
            raise ValueError('Direct-window or mandatory wiring checks failed')
        pipeline_path = validation / f'pipeline-{cohort}.json'
        paths.append(pipeline_path)
        native = read(pipeline_path)
        if set(native['cases']) != required:
            raise ValueError('Missing native full-file cases')
        cases = {}
        for name in sorted(required):
            candidate = native['cases'][name]
            if candidate['vad_max_abs'] > .01 or candidate['vad_cosine'] < .999:
                raise ValueError('VAD classifier mismatch: ' + name)
            cases[name] = compare_case(candidate['segments'], {
                'released-f32-tower-bf16-decoder': released['cases'][name]['segments'],
                'diagnostic-all-f32': f32['cases'][name]['segments']}, cohort)
            cases[name]['vad_max_abs'] = candidate['vad_max_abs']
        receipts[cohort] = cases
    paths.append(validation / 'cohort-results.json')
    return dict(accepted=True, provenance=provenance, cases=receipts,
                policy='Exact direct-window gates; complete full-file text must match one independently generated source precision variant. F16 timestamp tolerance 5.1ms; Q8 20.1ms.',
                exact_released_full_pipeline=False,
                evidence=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validation', type=Path, required=True)
    p.add_argument('--f32-validation', type=Path, required=True)
    p.add_argument('--released', type=Path, required=True)
    p.add_argument('--f32', type=Path, required=True)
    p.add_argument('--dtype-audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    receipt = audit(args.validation, args.f32_validation, args.released, args.f32, args.dtype_audit)
    args.output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(receipt['cases'], indent=2))
