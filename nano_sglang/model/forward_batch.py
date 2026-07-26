from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ForwardBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor | None = None
    past_key_values: Any = None
    use_cache: bool = True
