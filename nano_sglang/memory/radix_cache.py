from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RadixNode:
    children: dict[int, "RadixNode"] = field(default_factory=dict)
    value: Any = None


class RadixCache:
    """Tiny token-prefix cache.

    This stores arbitrary values by token sequence and returns the longest
    matching prefix. It is deliberately simple and CPU-only.
    """

    def __init__(self):
        self.root = RadixNode()

    def insert(self, token_ids: list[int], value: Any) -> None:
        node = self.root
        for token_id in token_ids:
            node = node.children.setdefault(token_id, RadixNode())
        node.value = value

    def longest_prefix(self, token_ids: list[int]) -> tuple[list[int], Any]:
        node = self.root
        best_prefix: list[int] = []
        best_value = None
        current: list[int] = []

        for token_id in token_ids:
            if token_id not in node.children:
                break
            node = node.children[token_id]
            current.append(token_id)
            if node.value is not None:
                best_prefix = list(current)
                best_value = node.value

        return best_prefix, best_value

    def clear(self) -> None:
        self.root = RadixNode()
