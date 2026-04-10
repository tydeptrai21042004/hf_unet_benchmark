from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


class ConfigError(RuntimeError):
    pass


class DotDict(dict):
    """Dictionary with attribute access for nested config use."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @staticmethod
    def from_mapping(mapping: Mapping[str, Any]) -> "DotDict":
        result = DotDict()
        for key, value in mapping.items():
            if isinstance(value, Mapping):
                result[key] = DotDict.from_mapping(value)
            else:
                result[key] = value
        return result


def deep_update(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            deep_update(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def load_yaml(path: str | Path) -> Dict[str, Any]:
    if yaml is None:
        raise ConfigError("PyYAML is required to load YAML config files.")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Expected YAML object mapping at top level, got {type(data).__name__}")
    return data


def dump_yaml(data: Mapping[str, Any], path: str | Path) -> None:
    if yaml is None:
        raise ConfigError("PyYAML is required to write YAML config files.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(data), f, sort_keys=False, allow_unicode=True)


def load_config(
    *paths: str | Path,
    overrides: Optional[Mapping[str, Any]] = None,
    as_dotdict: bool = True,
) -> Dict[str, Any] | DotDict:
    merged: Dict[str, Any] = {}
    for path in paths:
        cfg = load_yaml(path)
        deep_update(merged, cfg)
    if overrides:
        deep_update(merged, overrides)
    return DotDict.from_mapping(merged) if as_dotdict else merged


__all__ = ["ConfigError", "DotDict", "deep_update", "load_yaml", "dump_yaml", "load_config"]
