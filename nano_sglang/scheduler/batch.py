from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from nano_sglang.scheduler.request import GenerationRequest


class BatchMode(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class ScheduleBatch:
    requests: list[GenerationRequest]
    mode: BatchMode

    def __len__(self) -> int:
        return len(self.requests)
