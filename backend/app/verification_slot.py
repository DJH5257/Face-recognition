from __future__ import annotations

import math
from threading import BoundedSemaphore
from typing import Callable

from fastapi import HTTPException

from .concurrency_metrics import ConcurrencyMetrics, now_ms


class VerificationSlot:
    """Acquire one verification slot, waiting briefly before rejecting as busy."""

    def __init__(
        self,
        semaphore: BoundedSemaphore,
        metrics: ConcurrencyMetrics,
        wait_seconds: Callable[[], float],
    ) -> None:
        self._semaphore = semaphore
        self._metrics = metrics
        self._wait_seconds = wait_seconds
        self._slot_start = 0.0

    def __enter__(self) -> "VerificationSlot":
        wait_seconds = max(0.0, float(self._wait_seconds()))
        start = now_ms()
        if wait_seconds <= 0:
            acquired = self._semaphore.acquire(blocking=False)
        else:
            acquired = self._semaphore.acquire(timeout=wait_seconds)
        wait_ms = now_ms() - start
        if not acquired:
            self._metrics.on_rejected_busy()
            retry_after = str(max(1, math.ceil(wait_seconds)))
            raise HTTPException(
                status_code=503,
                detail="当前人脸核验请求较多，请稍后重试",
                headers={"Retry-After": retry_after},
            )
        self._metrics.on_acquired(wait_ms)
        self._slot_start = now_ms()
        return self

    def __exit__(self, *_args) -> bool:
        duration_ms = now_ms() - self._slot_start
        self._semaphore.release()
        self._metrics.on_released(duration_ms)
        return False
