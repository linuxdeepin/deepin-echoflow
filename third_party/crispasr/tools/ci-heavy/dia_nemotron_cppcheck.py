#!/usr/bin/env python3
"""Run the existing CI cppcheck version/options on the landed runtime changes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[2]
out = Path(os.environ['HEAVY_OUT'])
out.mkdir(parents=True, exist_ok=True)
files = [
    'src/dia_tts.cpp', 'src/dia_tts.h', 'src/dia_sampling.h',
    'src/crispasr_c_api.cpp', 'examples/cli/crispasr_backend_dia.cpp',
    'examples/cli/crispasr_tts_chunking.cpp', 'src/nemotron.cpp',
]
flags = [
    '--error-exitcode=1', '--enable=warning,performance,portability',
    '--inline-suppr', '--std=c++17', '--language=c++', '--quiet',
] + ['--suppress=' + value for value in [
    'missingIncludeSystem', 'unmatchedSuppression', 'duplicateAssignExpression',
    '*:examples/server/*', '*:*httplib.h', '*:*json.hpp', '*:*miniaudio.h',
    '*:*stb_vorbis.c', '*:*rnnoise/*', 'unknownMacro',
    'nullPointerOutOfMemory', 'nullPointerArithmeticOutOfMemory',
    'toomanyconfigs', 'invalidFunctionArg',
]]
command = [
    'docker', 'run', '--rm', '--volume', f'{root}:/src:ro', '--workdir', '/src',
    'ubuntu:22.04', 'bash', '-ec',
    'apt-get update -qq; apt-get install -y --no-install-recommends cppcheck; '
    'test "$(cppcheck --version)" = "Cppcheck 2.7"; cppcheck --version; '
    'exec cppcheck "$@"', 'cppcheck', *flags, *files,
]
subprocess.run(['uptime'], check=True)
subprocess.run(['free', '-h'], check=True)
result = subprocess.run(command, capture_output=True, text=True, timeout=1200)
(out / 'cppcheck.log').write_text(result.stdout + result.stderr)
receipt = {
    'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
    'cppcheck_version': '2.7', 'command': command, 'files': files,
    'sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files},
    'passed': result.returncode == 0, 'exit_code': result.returncode,
    'scope': 'Seven CI-covered changed runtime files; server excluded by existing full-tree CI policy. No Index-Echo files or new suppressions.',
}
(out / 'cppcheck.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(result.stdout + result.stderr, flush=True)
raise SystemExit(result.returncode)
