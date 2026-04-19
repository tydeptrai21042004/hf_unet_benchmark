from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, MutableMapping, Optional

from .registry import create_model


def _to_dict(config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, dict):
        return deepcopy(config)
    return deepcopy(dict(config))


ALIASES = {
    "hf_unet": "proposal_hf_unet",
    "proposal": "proposal_hf_unet",
    "hf_unet_lite": "proposal_hf_unet_lite",
    "proposal_lite": "proposal_hf_unet_lite",
}


def build_model(name: str, config: Optional[Mapping[str, Any]] = None, **overrides: Any):
    cfg = _to_dict(config)
    cfg.pop("name", None)
    cfg.update(overrides)
    cfg.pop("name", None)
    model_name = ALIASES.get(name.lower(), name.lower())
    return create_model(model_name, **cfg)


__all__ = ["build_model"]
