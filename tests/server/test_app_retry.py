from video2tasks.server.app import _requeue_empty_result


def test_requeue_empty_result_appends_to_queue_tail_and_keeps_counting() -> None:
    job_queue = [{"task_id": "existing"}]
    retry_counts = {}
    job = {"task_id": "target"}

    attempt1 = _requeue_empty_result(job_queue, retry_counts, "target", job)
    attempt2 = _requeue_empty_result(job_queue, retry_counts, "target", job)

    assert attempt1 == 1
    assert attempt2 == 2
    assert retry_counts == {"target": 2}
    assert job_queue == [
        {"task_id": "existing"},
        {"task_id": "target"},
        {"task_id": "target"},
    ]
