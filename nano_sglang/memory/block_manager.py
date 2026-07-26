from __future__ import annotations


class BlockManager:
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.free_blocks = list(range(num_blocks))

    @property
    def used_blocks(self) -> int:
        return self.num_blocks - len(self.free_blocks)

    def allocate(self) -> int:
        if not self.free_blocks:
            raise MemoryError("No free KV-cache blocks available")
        return self.free_blocks.pop()

    def free(self, block_id: int) -> None:
        if block_id < 0 or block_id >= self.num_blocks:
            raise ValueError(f"Invalid block id: {block_id}")
        if block_id not in self.free_blocks:
            self.free_blocks.append(block_id)
