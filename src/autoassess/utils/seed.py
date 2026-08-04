"""Reproducibility utilities — seed every random source."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set the random seed for Python, NumPy, and PyTorch (CPU + CUDA).

    Args:
        seed: Integer seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensures deterministic cuDNN behaviour (may slow training slightly).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
