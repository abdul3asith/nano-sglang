from __future__ import annotations

from collections import deque

from .batch import BatchMode, ScheduleBatch
from .request import GenerationRequest, RequestStatus


class Scheduler:
    """Naive FIFO scheduler.

    It admits requests up to max_batch_size and then serves active requests in
    round-robin decode steps. The engine currently executes each request
    independently, but this keeps request lifecycle logic separate from model
    execution.
    """

    def __init__(self, max_batch_size: int):
        self.max_batch_size = max_batch_size
        self.waiting: deque[GenerationRequest] = deque()
        self.active: list[GenerationRequest] = []
        self.finished: list[GenerationRequest] = []

    def add_request(self, request: GenerationRequest) -> None:
        self.waiting.append(request)

    def has_work(self) -> bool:
        return bool(self.waiting or self.active)

    def admit(self) -> ScheduleBatch | None:
        admitted: list[GenerationRequest] = []
        while self.waiting and len(self.active) < self.max_batch_size:
            request = self.waiting.popleft()
            request.status = RequestStatus.PREFILL
            self.active.append(request)
            admitted.append(request)

        if not admitted:
            return None
        return ScheduleBatch(admitted, BatchMode.PREFILL)

    def decode_batch(self) -> ScheduleBatch | None:
        ready = [request for request in self.active if request.status == RequestStatus.DECODING]
        if not ready:
            return None
        return ScheduleBatch(ready, BatchMode.DECODE)

    def retire_finished(self) -> list[GenerationRequest]:
        still_active: list[GenerationRequest] = []
        retired: list[GenerationRequest] = []
        for request in self.active:
            if request.is_finished:
                retired.append(request)
            else:
                still_active.append(request)

        self.active = still_active
        self.finished.extend(retired)
        return retired
