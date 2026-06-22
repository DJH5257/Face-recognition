from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Dict, Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class _FailureBucket:
    count: int
    window_started_at: datetime
    blocked_until: Optional[datetime] = None


class InMemoryFailureLimiter:
    def __init__(self) -> None:
        self._lock = Lock()
        self._buckets: Dict[str, _FailureBucket] = {}

    def retry_after_seconds(self, key: str) -> int:
        now = _utcnow()
        with self._lock:
            bucket = self._buckets.get(key)
            if not bucket or not bucket.blocked_until:
                return 0
            if bucket.blocked_until <= now:
                self._buckets.pop(key, None)
                return 0
            return max(1, int((bucket.blocked_until - now).total_seconds()))

    def record_failure(self, key: str, max_failures: int, window_seconds: int, cooldown_seconds: int) -> int:
        if max_failures <= 0 or window_seconds <= 0 or cooldown_seconds <= 0:
            return 0
        now = _utcnow()
        with self._lock:
            bucket = self._buckets.get(key)
            if not bucket or now - bucket.window_started_at > timedelta(seconds=window_seconds):
                bucket = _FailureBucket(count=0, window_started_at=now)
                self._buckets[key] = bucket
            bucket.count += 1
            if bucket.count >= max_failures:
                bucket.blocked_until = now + timedelta(seconds=cooldown_seconds)
                return cooldown_seconds
            return 0

    def clear(self, key: str) -> None:
        with self._lock:
            self._buckets.pop(key, None)
