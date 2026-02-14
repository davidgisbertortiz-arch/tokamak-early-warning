"""Centralized runtime defaults and seed handling."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_SEED = 42
DEFAULT_DATA_PATH = "data/raw/DL_DataFrame.h5"
DEFAULT_REPORTS_DIR = Path("reports")


@dataclass(frozen=True)
class RunConfig:
    seed: int = DEFAULT_SEED
    data_path: str = DEFAULT_DATA_PATH

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # Torch is optional for all scripts except TCN training.
        pass
