from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True, warn_only: bool = True) -> int:
    """Seed common RNGs and optionally enable PyTorch deterministic algorithms.

    Args:
        seed: Global random seed.
        deterministic: If True, enable deterministic CuDNN behavior and ask
            PyTorch to enforce deterministic algorithms when possible.
        warn_only: When deterministic=True, downgrade unsupported deterministic
            CUDA ops from RuntimeError to warning. This is useful for paper
            benchmark code where some backbones (e.g. PVT-style attention blocks)
            use CUDA ops without deterministic backward implementations.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass

    return seed


__all__ = ["seed_everything"]
