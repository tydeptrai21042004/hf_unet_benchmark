#!/usr/bin/env python3
"""Convenience entrypoint for the core-faithful benchmark recipe."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS = [
    'unet',
    'unet_cbam',
    'unetpp',
    'pranet',
    'acsnet',
    'hardnet_mseg',
    'polyp_pvt',
    'caranet',
    'cfanet',
    'hsnet',
    'proposal_hf_unet_lite',
    'proposal_hf_unet',
]


def main() -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / 'scripts' / 'benchmark_all.py'),
        '--config-dir', 'configs/faithful',
        '--models', ','.join(MODELS),
    ]
    print('[RUN]', ' '.join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
