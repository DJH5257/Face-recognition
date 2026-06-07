from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Dict, List, Optional
from uuid import uuid4

import numpy as np


@dataclass
class Enrollment:
    id: str
    embedding: np.ndarray
    bbox: List[float]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Challenge:
    id: str
    actions: List[str]
    expires_at: datetime
    enrollment_id: Optional[str] = None
    liveness_passed: bool = False
    live_embedding: Optional[np.ndarray] = None
    face_matched: bool = False
    face_similarity: Optional[float] = None
    action_results: List[dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    verified_at: Optional[datetime] = None

    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > self.expires_at


class MemoryStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self.enrollments: Dict[str, Enrollment] = {}
        self.challenges: Dict[str, Challenge] = {}

    def create_enrollment(self, embedding: np.ndarray, bbox: List[float]) -> Enrollment:
        item = Enrollment(id=str(uuid4()), embedding=embedding, bbox=bbox)
        with self._lock:
            self.enrollments[item.id] = item
        return item

    def get_enrollment(self, enrollment_id: str) -> Optional[Enrollment]:
        with self._lock:
            return self.enrollments.get(enrollment_id)

    def create_challenge(
        self,
        actions: List[str],
        ttl_seconds: int,
        enrollment_id: Optional[str] = None,
    ) -> Challenge:
        item = Challenge(
            id=str(uuid4()),
            actions=actions,
            enrollment_id=enrollment_id,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds),
        )
        with self._lock:
            self.challenges[item.id] = item
        return item

    def get_challenge(self, challenge_id: str) -> Optional[Challenge]:
        with self._lock:
            return self.challenges.get(challenge_id)


store = MemoryStore()
