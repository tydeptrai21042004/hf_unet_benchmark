from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / 'configs'

MODELS = [
    'unet',
    'unetpp',
    'pranet',
    'acsnet',
    'hardnet_mseg',
    'polyp_pvt',
    'caranet',
    'proposal_hf_unet',
]

SHARED_KEYS = {
    'data': ['image_size', 'num_workers', 'pin_memory'],
    'train': ['epochs', 'optimizer', 'scheduler', 'weight_decay', 'mixed_precision', 'grad_clip', 'threshold'],
    'eval': ['threshold'],
}

SOFT_KEYS = {
    'data': ['batch_size'],
    'train': ['lr', 'loss', 'aux_loss_weight'],
    'model': ['norm', 'act'],
}


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def collect() -> Dict[str, Dict[str, Any]]:
    configs: Dict[str, Dict[str, Any]] = {}
    for model in MODELS:
        path = CONFIG_DIR / f'{model}.yaml'
        configs[model] = load_yaml(path)
    return configs


def compare_section(configs: Dict[str, Dict[str, Any]], section: str, key: str) -> Dict[str, Any]:
    out = {}
    for model, cfg in configs.items():
        out[model] = cfg.get(section, {}).get(key)
    return out


def unique_values(values: Dict[str, Any]) -> set[str]:
    return {json.dumps(v, sort_keys=True) for v in values.values()}


def main() -> None:
    configs = collect()
    report: Dict[str, Any] = {
        'strict_mismatches': {},
        'soft_mismatches': {},
        'warnings': [],
    }

    for section, keys in SHARED_KEYS.items():
        for key in keys:
            values = compare_section(configs, section, key)
            if len(unique_values(values)) > 1:
                report['strict_mismatches'][f'{section}.{key}'] = values

    for section, keys in SOFT_KEYS.items():
        for key in keys:
            values = compare_section(configs, section, key)
            if len(unique_values(values)) > 1:
                report['soft_mismatches'][f'{section}.{key}'] = values

    hf = configs['proposal_hf_unet']
    unet = configs['unet']
    if hf.get('train', {}).get('loss') != unet.get('train', {}).get('loss'):
        report['warnings'].append('HF-U-Net and U-Net use different losses; backbone-matched comparison would be cleaner with the same loss.')
    if hf.get('data', {}).get('batch_size') != unet.get('data', {}).get('batch_size'):
        report['warnings'].append('HF-U-Net and U-Net use different batch sizes; consider matching effective batch size or using gradient accumulation.')
    report['warnings'].append('Architectural fairness still needs manual review: HF-U-Net adds CBAM decoder attention, SE, gate, and spectral regularization beyond a plain U-Net bottleneck swap.')
    report['warnings'].append("Baseline configs are harmonized across models, but paper-faithful reproduction still depends on matching each paper's loss, pretraining, and evaluation protocol.")

    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
