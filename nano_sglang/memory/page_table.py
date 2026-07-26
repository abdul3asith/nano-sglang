from __future__ import annotations


class PageTable:
    def __init__(self):
        self._tables: dict[str, list[int]] = {}

    def new_sequence(self, request_id: str) -> None:
        self._tables[request_id] = []

    def append_block(self, request_id: str, block_id: int) -> None:
        self._tables.setdefault(request_id, []).append(block_id)

    def blocks(self, request_id: str) -> list[int]:
        return list(self._tables.get(request_id, []))

    def pop_sequence(self, request_id: str) -> list[int]:
        return self._tables.pop(request_id, [])
