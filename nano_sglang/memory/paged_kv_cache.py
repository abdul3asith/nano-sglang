from __future__ import annotations

import torch

from nano_sglang.memory.block_manager import BlockManager
from nano_sglang.memory.page_table import PageTable


class PagedKVCache:
    """Fixed-block KV cache used for learning paged allocation.

    This is not wired into Hugging Face attention yet. It demonstrates the memory
    layout and page-table mechanics that a paged attention kernel would consume.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int,
        num_blocks: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cpu",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.block_manager = BlockManager(num_blocks)
        self.page_table = PageTable()
        self.seq_lens: dict[str, int] = {}

        shape = (num_layers, num_blocks, block_size, num_heads, head_dim)
        self.keys = torch.empty(shape, dtype=dtype, device=device)
        self.values = torch.empty(shape, dtype=dtype, device=device)

    def new_sequence(self, request_id: str) -> None:
        self.page_table.new_sequence(request_id)
        self.seq_lens[request_id] = 0

    def _ensure_block(self, request_id: str, token_offset: int) -> tuple[int, int]:
        block_offset = token_offset // self.block_size
        slot = token_offset % self.block_size
        blocks = self.page_table.blocks(request_id)
        while len(blocks) <= block_offset:
            block_id = self.block_manager.allocate()
            self.page_table.append_block(request_id, block_id)
            blocks.append(block_id)
        return blocks[block_offset], slot

    def append(self, request_id: str, layer: int, key: torch.Tensor, value: torch.Tensor) -> None:
        if request_id not in self.seq_lens:
            self.new_sequence(request_id)

        token_offset = self.seq_lens[request_id]
        block_id, slot = self._ensure_block(request_id, token_offset)
        self.keys[layer, block_id, slot].copy_(key)
        self.values[layer, block_id, slot].copy_(value)

        if layer == self.num_layers - 1:
            self.seq_lens[request_id] += 1

    def gather(self, request_id: str, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        blocks = self.page_table.blocks(request_id)
        seq_len = self.seq_lens[request_id]
        if not blocks:
            raise KeyError(f"No blocks allocated for request={request_id}")

        keys = self.keys[layer, blocks].reshape(-1, self.num_heads, self.head_dim)[:seq_len]
        values = self.values[layer, blocks].reshape(-1, self.num_heads, self.head_dim)[:seq_len]
        return keys, values

    def free(self, request_id: str) -> None:
        for block_id in self.page_table.pop_sequence(request_id):
            self.block_manager.free(block_id)
        self.seq_lens.pop(request_id, None)
