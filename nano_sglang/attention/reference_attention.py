from __future__ import annotations

import math

import torch


def reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """Readable scaled dot-product attention.

    Expected shapes:
    query/key/value: [batch, heads, seq_len, head_dim]
    """

    scores = query @ key.transpose(-2, -1)
    scores = scores / math.sqrt(query.shape[-1])

    if causal:
        q_len = query.shape[-2]
        k_len = key.shape[-2]
        mask = torch.ones(q_len, k_len, device=query.device, dtype=torch.bool).tril(
            diagonal=k_len - q_len
        )
        scores = scores.masked_fill(~mask, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    return probs @ value
