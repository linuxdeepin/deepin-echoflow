#!/usr/bin/env python3
"""Independent workflow group for the shipped Dia F16 Metal speech acceptance."""
import runpy
from pathlib import Path
import sys
extra = sys.argv[1:]
sys.argv = [str(Path(__file__).with_name('dia_full_speech.py')), '--quant', 'f16',
            '--matrix', '4', '--limits', '--metal'] + extra
runpy.run_path(sys.argv[0], run_name='__main__')
