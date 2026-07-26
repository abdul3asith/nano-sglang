from __future__ import annotations

import torch


class ContiguousKVCache:
    """Simple per-request KV storage for learning and tests.

    Shape convention per layer:
    key/value token tensors are [num_heads, head_dim].
    gather() returns [seq_len, num_heads, head_dim].
    """

    def __init__(self):
        self._keys: dict[str, dict[int, list[torch.Tensor]]] = {}
        self._values: dict[str, dict[int, list[torch.Tensor]]] = {}

    def new_sequence(self, request_id: str) -> None:
        self._keys[request_id] = {}
        self._values[request_id] = {}

    def append(self, request_id: str, layer: int, key: torch.Tensor, value: torch.Tensor) -> None:
        if request_id not in self._keys:
            self.new_sequence(request_id)
        self._keys[request_id].setdefault(layer, []).append(key.detach())
        self._values[request_id].setdefault(layer, []).append(value.detach())

    def gather(self, request_id: str, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        keys = self._keys[request_id].get(layer, [])
        values = self._values[request_id].get(layer, [])
        if not keys:
            raise KeyError(f"No KV entries for request={request_id}, layer={layer}")
        return torch.stack(keys, dim=0), torch.stack(values, dim=0)

    def free(self, request_id: str) -> None:
        self._keys.pop(request_id, None)
        self._values.pop(request_id, None)

    def sequence_length(self, request_id: str, layer: int = 0) -> int:
        return len(self._keys.get(request_id, {}).get(layer, []))
