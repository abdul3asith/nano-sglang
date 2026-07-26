from __future__ import annotations

import torch

from nano_sglang.attention.reference_attention import reference_attention


def gather_paged_kv(
    key_blocks: torch.Tensor,
    value_blocks: torch.Tensor,
    block_ids: list[int],
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not block_ids:
        raise ValueError("block_ids cannot be empty")

    keys = key_blocks[block_ids].reshape(-1, *key_blocks.shape[2:])[:seq_len]
    values = value_blocks[block_ids].reshape(-1, *value_blocks.shape[2:])[:seq_len]
    return keys, values


def paged_attention(
    query: torch.Tensor,
    key_blocks: torch.Tensor,
    value_blocks: torch.Tensor,
    block_ids: list[int],
    seq_len: int,
    causal: bool = True,
) -> torch.Tensor:
    """Naive PyTorch paged attention.

    This gathers pages into a contiguous tensor first. It is intentionally simple
    and useful as a correctness reference before writing kernels.
    """

    key, value = gather_paged_kv(key_blocks, value_blocks, block_ids, seq_len)
    key = key.transpose(0, 1).unsqueeze(0)
    value = value.transpose(0, 1).unsqueeze(0)
    return reference_attention(query, key, value, causal=causal)
