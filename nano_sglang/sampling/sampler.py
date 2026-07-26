from __future__ import annotations

import torch

from nano_sglang.sampling.logits_processor import LogitsProcessor


class Sampler:
    def __init__(
        self,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int | None = None,
        seed: int | None = None,
    ):
        self.temperature = temperature
        self.processor = LogitsProcessor(temperature, top_p, top_k)
        self.generator = None
        if seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)

    def sample(self, logits: torch.Tensor) -> int:
        next_token_logits = logits[:, -1, :]

        if self.temperature == 0:
            return int(torch.argmax(next_token_logits, dim=-1)[0].item())

        processed = self.processor(next_token_logits)
        probs = torch.softmax(processed, dim=-1)
        token = torch.multinomial(probs, num_samples=1, generator=self.generator)
        return int(token[0, 0].item())
