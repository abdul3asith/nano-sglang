from __future__ import annotations

from typing import Any

import torch

from .forward_batch import ForwardBatch


class ModelRunner:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def forward(self, batch: ForwardBatch):
        return self.model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            past_key_values=batch.past_key_values,
            use_cache=batch.use_cache,
        )

    @torch.no_grad()
    def prefill(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ):
        outputs = self.forward(
            ForwardBatch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
        )
        return outputs.logits, outputs.past_key_values

    @torch.no_grad()
    def decode_one_token(self, token_id: torch.Tensor, past_key_values: Any):
        outputs = self.forward(
            ForwardBatch(
                input_ids=token_id,
                past_key_values=past_key_values,
                use_cache=True,
            )
        )
        return outputs.logits, outputs.past_key_values
