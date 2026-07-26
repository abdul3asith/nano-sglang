from nano_sglang.scheduler import GenerationRequest, RequestStatus, Scheduler


def test_scheduler_admits_and_retires_requests():
    scheduler = Scheduler(max_batch_size=1)
    request = GenerationRequest(prompt="hello", max_new_tokens=1)

    scheduler.add_request(request)
    batch = scheduler.admit()

    assert batch is not None
    assert batch.requests == [request]
    assert request.status == RequestStatus.PREFILL

    request.mark_finished("length")
    retired = scheduler.retire_finished()

    assert retired == [request]
    assert not scheduler.has_work()
