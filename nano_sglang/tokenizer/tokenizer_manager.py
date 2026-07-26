from __future__ import annotations

import torch


class TokenizerManager:
    def __init__(self, tokenizer, device: str):
        self.tokenizer = tokenizer
        self.device = device

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(self, text: str) -> dict[str, torch.Tensor]:
        return self.tokenizer(text, return_tensors="pt").to(self.device)

    def encode_many(self, prompts: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def decode_token(self, token_id: int) -> str:
        return self.decode([token_id])

    @property
    def eos_token_id(self) -> int | None:
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int | None:
        return self.tokenizer.pad_token_id
