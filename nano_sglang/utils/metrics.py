from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter


@dataclass
class GenerationMetrics:
    prompt_tokens: int = 0
    generated_tokens: int = 0
    total_seconds: float = 0.0
    time_to_first_token: float | None = None

    @property
    def tokens_per_second(self) -> float:
        if self.total_seconds == 0:
            return 0.0
        return self.generated_tokens / self.total_seconds


class Timer:
    def __init__(self):
        self.start = perf_counter()

    def elapsed(self) -> float:
        return perf_counter() - self.start
