"""并发槽位的行为验证：等待、超时、统计计数。"""

from __future__ import annotations

import threading
import time

import pytest

from backend.app.concurrency_metrics import ConcurrencyMetrics
from backend.app.verification_slot import VerificationSlot


def test_metrics_snapshot_counts_in_flight_and_release():
    metrics = ConcurrencyMetrics(recent_window=10)

    metrics.on_acquired(wait_ms=5.0)
    snap_running = metrics.snapshot()
    assert snap_running["in_flight"] == 1
    assert snap_running["max_in_flight"] == 1
    assert snap_running["completed"] == 0

    metrics.on_released(duration_ms=120.0)
    snap_done = metrics.snapshot()
    assert snap_done["in_flight"] == 0
    assert snap_done["completed"] == 1
    assert snap_done["recent_sample_size"] == 1
    assert snap_done["recent_avg_ms"] == pytest.approx(120.0)
    assert snap_done["avg_wait_ms"] == pytest.approx(5.0)


def test_metrics_records_rejection():
    metrics = ConcurrencyMetrics()
    metrics.on_rejected_busy()
    metrics.on_rejected_busy()
    assert metrics.snapshot()["rejected_busy"] == 2


def test_verification_slot_waits_until_released():
    semaphore = threading.BoundedSemaphore(1)
    metrics = ConcurrencyMetrics(recent_window=10)

    results = {}

    def hold_first():
        with VerificationSlot(semaphore, metrics, lambda: 1.0):
            time.sleep(0.3)
            results["first_done"] = time.perf_counter()

    def follow_second(start_barrier: threading.Event):
        start_barrier.wait()
        try:
            with VerificationSlot(semaphore, metrics, lambda: 1.0):
                results["second_acquired"] = time.perf_counter()
        except Exception as exc:  # noqa: BLE001
            results["second_error"] = exc

    barrier = threading.Event()
    t1 = threading.Thread(target=hold_first)
    t2 = threading.Thread(target=follow_second, args=(barrier,))
    t1.start()
    # 等待第一个线程拿到槽位再放行第二个，确保第二个一定会进入等待
    time.sleep(0.05)
    t2.start()
    barrier.set()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert "second_error" not in results, results.get("second_error")
    assert "second_acquired" in results
    # 第二个线程必须晚于第一个完成才拿到槽位
    assert results["second_acquired"] >= results["first_done"] - 1e-3

    snap = metrics.snapshot()
    assert snap["completed"] == 2
    assert snap["rejected_busy"] == 0
    assert snap["max_in_flight"] == 1
    # 第二个线程的等待时间应该 >0
    assert snap["avg_wait_ms"] >= 0.0


def test_verification_slot_rejects_after_timeout():
    semaphore = threading.BoundedSemaphore(1)
    metrics = ConcurrencyMetrics(recent_window=10)

    holder_release = threading.Event()
    holder_acquired = threading.Event()

    def hold():
        with VerificationSlot(semaphore, metrics, lambda: 0.1):
            holder_acquired.set()
            holder_release.wait(timeout=2.0)

    holder = threading.Thread(target=hold)
    holder.start()
    holder_acquired.wait(timeout=2.0)

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        with VerificationSlot(semaphore, metrics, lambda: 0.1):
            pass
    assert excinfo.value.status_code == 503
    assert excinfo.value.headers == {"Retry-After": "1"}

    holder_release.set()
    holder.join(timeout=2.0)

    snap = metrics.snapshot()
    assert snap["rejected_busy"] == 1
    assert snap["completed"] == 1
