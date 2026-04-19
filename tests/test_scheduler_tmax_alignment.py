from __future__ import annotations

from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_ROOT = PROJECT_ROOT / "configs"


def _iter_cfgs():
    for path in sorted(CONFIG_ROOT.rglob("*.yaml")):
        yield path, yaml.safe_load(path.read_text(encoding="utf-8"))


def test_all_cosine_scheduler_configs_use_tmax_30():
    checked = 0
    for path, cfg in _iter_cfgs():
        train = cfg.get("train", {})
        if str(train.get("scheduler", "")).lower() == "cosine" and "t_max" in train:
            assert train["t_max"] == 30, (str(path), train["t_max"])
            checked += 1
    assert checked > 0


def test_paper_facing_30_epoch_configs_keep_tmax_matched_to_epoch_budget():
    checked = 0
    for path, cfg in _iter_cfgs():
        train = cfg.get("train", {})
        if train.get("epochs") == 30 and str(train.get("scheduler", "")).lower() == "cosine":
            assert train.get("t_max") == 30, (str(path), train.get("t_max"), train.get("epochs"))
            checked += 1
    assert checked > 0
