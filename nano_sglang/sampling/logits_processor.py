from __future__ import annotations

import torch


class LogitsProcessor:
    def __init__(
        self,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int | None = None,
    ):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature > 0:
            logits = logits / self.temperature

        if self.top_k is not None and self.top_k > 0:
            values, _ = torch.topk(logits, k=min(self.top_k, logits.shape[-1]))
            cutoff = values[..., -1, None]
            logits = logits.masked_fill(logits < cutoff, float("-inf"))

        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative > self.top_p
            remove[..., 0] = False

            filtered = torch.zeros_like(remove).scatter(-1, sorted_indices, remove)
            logits = logits.masked_fill(filtered, float("-inf"))

        return logits
