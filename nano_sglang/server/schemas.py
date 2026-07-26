from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GenerateRequest:
    prompt: str
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None


@dataclass
class GenerateResponse:
    text: str
    request_id: str
    finish_reason: str | None
    generated_token_ids: list[int] = field(default_factory=list)
