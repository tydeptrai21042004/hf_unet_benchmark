from __future__ import annotations

from typing import Any, Callable, Dict, Type

import torch.nn as nn


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


class RegistryError(KeyError):
    pass


def register_model(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        key = name.strip().lower()
        if key in MODEL_REGISTRY:
            raise RegistryError(f"Model '{key}' is already registered.")
        MODEL_REGISTRY[key] = cls
        cls.model_name = key
        return cls

    return decorator


def get_model_class(name: str) -> Type[nn.Module]:
    key = name.strip().lower()
    if key not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise RegistryError(f"Unknown model '{name}'. Available: {available}")
    return MODEL_REGISTRY[key]


def create_model(name: str, **kwargs: Any) -> nn.Module:
    cls = get_model_class(name)
    return cls(**kwargs)


__all__ = ["MODEL_REGISTRY", "RegistryError", "register_model", "get_model_class", "create_model"]
