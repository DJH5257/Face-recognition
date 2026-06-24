"""轻量的核验并发与耗时统计，仅在单进程内有效。

只用标准库实现，避免引入 Redis / Prometheus 等额外依赖；
适合当前单实例部署阶段做容量观察与压测参考。
"""

from __future__ import annotations

from collections import deque
from threading import Lock
from time import monotonic
from typing import Deque, Dict


class ConcurrencyMetrics:
    """记录正在跑的请求数、拒绝次数、最近若干次耗时。"""

    def __init__(self, recent_window: int = 200) -> None:
        self._lock = Lock()
        self._in_flight = 0
        self._max_in_flight = 0
        self._completed = 0
        self._rejected_busy = 0
        self._wait_total_ms = 0.0
        self._wait_count = 0
        self._latencies_ms: Deque[float] = deque(maxlen=max(1, recent_window))
        self._recent_window = max(1, recent_window)

    # ----- 槽位生命周期 -----
    def on_acquired(self, wait_ms: float) -> None:
        with self._lock:
            self._in_flight += 1
            if self._in_flight > self._max_in_flight:
                self._max_in_flight = self._in_flight
            self._wait_total_ms += max(0.0, wait_ms)
            self._wait_count += 1

    def on_released(self, duration_ms: float) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            self._completed += 1
            self._latencies_ms.append(max(0.0, duration_ms))

    def on_rejected_busy(self) -> None:
        with self._lock:
            self._rejected_busy += 1

    # ----- 查询 -----
    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            latencies = sorted(self._latencies_ms)
            count = len(latencies)
            avg = sum(latencies) / count if count else 0.0
            p95 = latencies[min(count - 1, int(round(count * 0.95)) - 1)] if count else 0.0
            p50 = latencies[count // 2] if count else 0.0
            avg_wait = self._wait_total_ms / self._wait_count if self._wait_count else 0.0
            return {
                "in_flight": self._in_flight,
                "max_in_flight": self._max_in_flight,
                "completed": self._completed,
                "rejected_busy": self._rejected_busy,
                "recent_window": self._recent_window,
                "recent_sample_size": count,
                "recent_avg_ms": round(avg, 2),
                "recent_p50_ms": round(p50, 2),
                "recent_p95_ms": round(p95, 2),
                "avg_wait_ms": round(avg_wait, 2),
                "wait_count": self._wait_count,
            }


def now_ms() -> float:
    return monotonic() * 1000.0
