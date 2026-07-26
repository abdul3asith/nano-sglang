from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


class RequestStatus(str, Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODING = "decoding"
    FINISHED = "finished"


@dataclass
class GenerationRequest:
    prompt: str
    max_new_tokens: int
    request_id: str = field(default_factory=lambda: uuid4().hex)
    status: RequestStatus = RequestStatus.WAITING
    prompt_token_ids: list[int] = field(default_factory=list)
    generated_token_ids: list[int] = field(default_factory=list)
    past_key_values: Any = None
    finish_reason: str | None = None

    @property
    def is_finished(self) -> bool:
        return self.status == RequestStatus.FINISHED

    @property
    def generated_tokens(self) -> int:
        return len(self.generated_token_ids)

    def append_token(self, token_id: int) -> None:
        self.generated_token_ids.append(token_id)

    def mark_finished(self, reason: str) -> None:
        self.status = RequestStatus.FINISHED
        self.finish_reason = reason
