#!/usr/bin/env python3
"""Verify CLI cap/default behavior without repeating completed C ABI speech."""
import runpy
from pathlib import Path
import sys
sys.argv = [str(Path(__file__).with_name('dia_full_speech.py')), '--quant', 'q8_0',
            '--cli-only', '--limits']
runpy.run_path(sys.argv[0], run_name='__main__')
