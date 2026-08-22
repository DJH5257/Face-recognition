from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from secrets import token_urlsafe
from threading import Lock
from typing import Dict, List, Optional
from uuid import uuid4

import numpy as np

from .config import get_settings
from .template_crypto import TemplateCryptoError, embedding_from_blob, embedding_to_blob


class _ClosingConnection(sqlite3.Connection):
    """Preserve transaction handling and close each SQLite file handle."""

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat()


def _from_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _embedding_to_blob(embedding: np.ndarray) -> tuple[bytes, int]:
    settings = get_settings()
    return embedding_to_blob(
        embedding,
        settings.template_encryption_key,
        settings.template_encryption_required,
    )


def _embedding_from_blob(blob: bytes, dimension: int) -> np.ndarray:
    settings = get_settings()
    return embedding_from_blob(blob, dimension, settings.template_encryption_key)


def _token_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _action_results_to_json(action_results: Optional[List[dict]]) -> Optional[str]:
    if action_results is None:
        return None
    return json.dumps(action_results, ensure_ascii=False, separators=(",", ":"))


def _action_results_from_json(value: Optional[str]) -> List[dict]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, column_type: str) -> None:
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
    if column_name not in columns:
        default_value = "''" if column_type.upper().startswith("TEXT") else "0"
        conn.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type} NOT NULL DEFAULT {default_value}"
        )


def _ensure_nullable_column(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_type: str,
) -> None:
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
    if column_name not in columns:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")


@dataclass
class Enrollment:
    id: str
    embedding: np.ndarray
    bbox: List[float]
    created_at: datetime = field(default_factory=_utcnow)


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
    created_at: datetime = field(default_factory=_utcnow)
    verified_at: Optional[datetime] = None

    def is_expired(self) -> bool:
        return _utcnow() > self.expires_at


@dataclass
class FaceTemplate:
    template_id: str
    subject_type: str
    subject_id: str
    embedding: np.ndarray
    bbox: List[float]
    template_version: int
    status: str
    source_type: str
    source_image_hash: Optional[str]
    created_at: datetime
    activated_at: Optional[datetime]
    revoked_at: Optional[datetime]


@dataclass
class VerificationSession:
    session_id: str
    request_id: Optional[str]
    subject_type: str
    subject_id: str
    admin_id: Optional[str]
    scene: str
    business_event_id: str
    record_id: Optional[str]
    action: Optional[str]
    template_id: str
    template_version: int
    expected_actions: List[str]
    upload_token: str
    status: str
    result_code: str
    issued_at: datetime
    expires_at: datetime
    verifying_started_at: Optional[datetime]
    verified_at: Optional[datetime]
    proof_id: Optional[str]
    action_results: List[dict] = field(default_factory=list)

    def is_expired(self) -> bool:
        return _utcnow() > self.expires_at


@dataclass
class VerificationProof:
    proof_id: str
    session_id: str
    business_event_id: str
    subject_type: str
    subject_id: str
    admin_id: Optional[str]
    scene: str
    record_id: Optional[str]
    action: Optional[str]
    template_id: str
    template_version: int
    status: str
    result_code: str
    issued_at: datetime
    expires_at: datetime
    finalized_at: Optional[datetime]

    def is_expired(self) -> bool:
        return _utcnow() > self.expires_at

    @property
    def valid(self) -> bool:
        return self.status == "ready" and not self.is_expired()


class Store:
    backend_name = "sqlite"

    def __init__(self, database_path: Optional[str] = None) -> None:
        settings = get_settings()
        self.database_path = Path(database_path or settings.database_path).expanduser()
        self._lock = Lock()
        self.enrollments: Dict[str, Enrollment] = {}
        self.challenges: Dict[str, Challenge] = {}
        self._init_db()

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
            expires_at=_utcnow() + timedelta(seconds=ttl_seconds),
        )
        with self._lock:
            self.challenges[item.id] = item
        return item

    def get_challenge(self, challenge_id: str) -> Optional[Challenge]:
        with self._lock:
            return self.challenges.get(challenge_id)

    def create_template(
        self,
        subject_type: str,
        subject_id: str,
        embedding: np.ndarray,
        bbox: List[float],
        source_type: str,
        source_image_hash: Optional[str],
    ) -> FaceTemplate:
        template_id = str(uuid4())
        now = _utcnow()
        embedding_blob, dimension = _embedding_to_blob(embedding)
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            latest = conn.execute(
                """
                SELECT COALESCE(MAX(template_version), 0) AS version
                FROM face_templates
                WHERE subject_type = ? AND subject_id = ?
                """,
                (subject_type, subject_id),
            ).fetchone()
            template_version = int(latest["version"]) + 1
            conn.execute(
                """
                UPDATE face_templates
                SET status = 'revoked', revoked_at = ?
                WHERE subject_type = ? AND subject_id = ? AND status = 'active'
                """,
                (_to_iso(now), subject_type, subject_id),
            )
            conn.execute(
                """
                INSERT INTO face_templates (
                    template_id, subject_type, subject_id, embedding_blob,
                    embedding_dimension, bbox_json, template_version, status,
                    source_type, source_image_hash, created_at, activated_at, revoked_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, NULL)
                """,
                (
                    template_id,
                    subject_type,
                    subject_id,
                    embedding_blob,
                    dimension,
                    json.dumps([float(value) for value in bbox]),
                    template_version,
                    source_type,
                    source_image_hash,
                    _to_iso(now),
                    _to_iso(now),
                ),
            )
            conn.commit()
        template = self.get_template(template_id)
        if template is None:
            raise RuntimeError("template insert succeeded but could not be loaded")
        return template

    def get_template(self, template_id: str) -> Optional[FaceTemplate]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM face_templates WHERE template_id = ?",
                (template_id,),
            ).fetchone()
        return _row_to_template(row) if row else None

    def get_active_template(self, subject_type: str, subject_id: str) -> Optional[FaceTemplate]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM face_templates
                WHERE subject_type = ? AND subject_id = ? AND status = 'active'
                ORDER BY template_version DESC
                LIMIT 1
                """,
                (subject_type, subject_id),
            ).fetchone()
        return _row_to_template(row) if row else None

    def list_templates(self, subject_type: str, subject_id: str) -> List[FaceTemplate]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM face_templates
                WHERE subject_type = ? AND subject_id = ?
                ORDER BY template_version DESC
                """,
                (subject_type, subject_id),
            ).fetchall()
        return [_row_to_template(row) for row in rows]

    def revoke_template(self, template_id: str) -> Optional[FaceTemplate]:
        now = _utcnow()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM face_templates WHERE template_id = ?",
                (template_id,),
            ).fetchone()
            if row is None:
                conn.rollback()
                return None
            if row["status"] == "active":
                conn.execute(
                    """
                    UPDATE face_templates
                    SET status = 'revoked', revoked_at = ?
                    WHERE template_id = ? AND status = 'active'
                    """,
                    (_to_iso(now), template_id),
                )
            updated = conn.execute(
                "SELECT * FROM face_templates WHERE template_id = ?",
                (template_id,),
            ).fetchone()
            conn.commit()
        return _row_to_template(updated) if updated else None

    def create_verification_session(
        self,
        request_id: Optional[str],
        subject_type: str,
        subject_id: str,
        admin_id: Optional[str],
        scene: str,
        business_event_id: str,
        record_id: Optional[str],
        action: Optional[str],
        expected_actions: List[str],
        ttl_seconds: int,
    ) -> Optional[VerificationSession]:
        template = self.get_active_template(subject_type, subject_id)
        if template is None:
            return None
        if request_id:
            existing = self.get_session_by_request_id(request_id)
            if existing is not None:
                return existing

        session_id = str(uuid4())
        upload_token = token_urlsafe(32)
        now = _utcnow()
        expires_at = now + timedelta(seconds=ttl_seconds)
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    """
                    INSERT INTO verification_sessions (
                        session_id, request_id, subject_type, subject_id, admin_id,
                        scene, business_event_id, record_id, action, template_id,
                        template_version, expected_actions_json, upload_token,
                        upload_token_hash, status, result_code, issued_at, expires_at,
                        verified_at, proof_id
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'issued', 'ISSUED', ?, ?, NULL, NULL)
                    """,
                    (
                        session_id,
                        request_id,
                        subject_type,
                        subject_id,
                        admin_id,
                        scene,
                        business_event_id,
                        record_id,
                        action,
                        template.template_id,
                        template.template_version,
                        json.dumps(expected_actions),
                        upload_token,
                        _token_hash(upload_token),
                        _to_iso(now),
                        _to_iso(expires_at),
                    ),
                )
                conn.commit()
            except sqlite3.IntegrityError:
                conn.rollback()
                if request_id:
                    return self.get_session_by_request_id(request_id)
                raise
        return self.get_session(session_id)

    def recent_failed_action_names(
        self,
        subject_type: str,
        subject_id: str,
        scene: str,
        limit: int,
    ) -> List[str]:
        if limit <= 0:
            return []
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT action_results_json
                FROM verification_sessions
                WHERE subject_type = ?
                  AND subject_id = ?
                  AND scene = ?
                  AND status = 'failed'
                  AND action_results_json IS NOT NULL
                ORDER BY verified_at DESC
                LIMIT ?
                """,
                (subject_type, subject_id, scene, limit),
            ).fetchall()
        failed_actions: List[str] = []
        known_actions = {"blink", "mouth_open", "shake_head", "nod_head", "smile"}
        for row in rows:
            for item in _action_results_from_json(row["action_results_json"]):
                action = item.get("action")
                if action in known_actions and item.get("passed") is False:
                    failed_actions.append(str(action))
        return failed_actions

    def get_session(self, session_id: str) -> Optional[VerificationSession]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM verification_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return _row_to_session(row) if row else None

    def get_session_by_request_id(self, request_id: str) -> Optional[VerificationSession]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM verification_sessions WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        return _row_to_session(row) if row else None

    def verify_upload_token(self, session_id: str, upload_token: str) -> bool:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT upload_token_hash FROM verification_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return bool(row and row["upload_token_hash"] == _token_hash(upload_token))

    def start_session_verification(self, session_id: str) -> Optional[str]:
        started_at = _to_iso(_utcnow())
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                UPDATE verification_sessions
                SET status = 'verifying',
                    result_code = 'VERIFYING',
                    verifying_started_at = ?,
                    verified_at = NULL
                WHERE session_id = ? AND status = 'issued'
                """,
                (started_at, session_id),
            )
            conn.commit()
        return started_at if row.rowcount == 1 else None

    def recover_stale_session_verification(self, session_id: str, timeout_seconds: int) -> int:
        if timeout_seconds <= 0:
            return 0
        now = _utcnow()
        stale_before = _to_iso(now - timedelta(seconds=timeout_seconds))
        now_iso = _to_iso(now)
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            expired = conn.execute(
                """
                UPDATE verification_sessions
                SET status = 'failed',
                    result_code = 'EXPIRED_SESSION',
                    verifying_started_at = NULL,
                    verified_at = ?
                WHERE session_id = ?
                  AND status = 'verifying'
                  AND (verifying_started_at IS NULL OR verifying_started_at <= ?)
                  AND expires_at <= ?
                """,
                (now_iso, session_id, stale_before, now_iso),
            )
            reset = conn.execute(
                """
                UPDATE verification_sessions
                SET status = 'issued',
                    result_code = 'ISSUED',
                    verifying_started_at = NULL,
                    verified_at = NULL
                WHERE session_id = ?
                  AND status = 'verifying'
                  AND (verifying_started_at IS NULL OR verifying_started_at <= ?)
                  AND expires_at > ?
                """,
                (session_id, stale_before, now_iso),
            )
            conn.commit()
        return expired.rowcount + reset.rowcount

    def mark_session_failed(
        self,
        session_id: str,
        result_code: str,
        verifying_started_at: Optional[str] = None,
        action_results: Optional[List[dict]] = None,
    ) -> bool:
        now = _utcnow()
        action_results_json = _action_results_to_json(action_results)
        with self._lock, self._connect() as conn:
            if verifying_started_at:
                row = conn.execute(
                    """
                    UPDATE verification_sessions
                    SET status = 'failed',
                        result_code = ?,
                        verified_at = ?,
                        verifying_started_at = NULL,
                        action_results_json = COALESCE(?, action_results_json)
                    WHERE session_id = ?
                      AND status = 'verifying'
                      AND verifying_started_at = ?
                    """,
                    (result_code, _to_iso(now), action_results_json, session_id, verifying_started_at),
                )
            else:
                row = conn.execute(
                    """
                    UPDATE verification_sessions
                    SET status = 'failed',
                        result_code = ?,
                        verified_at = ?,
                        verifying_started_at = NULL,
                        action_results_json = COALESCE(?, action_results_json)
                    WHERE session_id = ?
                      AND status IN ('issued', 'verifying')
                    """,
                    (result_code, _to_iso(now), action_results_json, session_id),
                )
        return row.rowcount == 1

    def reset_session_verification(
        self,
        session_id: str,
        verifying_started_at: Optional[str] = None,
    ) -> bool:
        with self._lock, self._connect() as conn:
            if verifying_started_at:
                row = conn.execute(
                    """
                    UPDATE verification_sessions
                    SET status = 'issued',
                        result_code = 'ISSUED',
                        verifying_started_at = NULL,
                        verified_at = NULL
                    WHERE session_id = ?
                      AND status = 'verifying'
                      AND verifying_started_at = ?
                    """,
                    (session_id, verifying_started_at),
                )
            else:
                row = conn.execute(
                    """
                    UPDATE verification_sessions
                    SET status = 'issued',
                        result_code = 'ISSUED',
                        verifying_started_at = NULL,
                        verified_at = NULL
                    WHERE session_id = ?
                      AND status = 'verifying'
                    """,
                    (session_id,),
                )
        return row.rowcount == 1

    def create_proof_for_session(
        self,
        session_id: str,
        result_code: str,
        ttl_seconds: int,
        verifying_started_at: Optional[str] = None,
        action_results: Optional[List[dict]] = None,
    ) -> Optional[VerificationProof]:
        now = _utcnow()
        expires_at = now + timedelta(seconds=ttl_seconds)
        action_results_json = _action_results_to_json(action_results)
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM verification_proofs WHERE session_id = ?",
                (session_id,),
            ).fetchone()

            if existing is not None:
                conn.commit()
                return _row_to_proof(existing)

            session = conn.execute(
                """
                SELECT * FROM verification_sessions
                WHERE session_id = ? AND status = 'verifying'
                """,
                (session_id,),
            ).fetchone()
            if session is None or (
                verifying_started_at and session["verifying_started_at"] != verifying_started_at
            ):
                conn.rollback()
                return None

            proof_id = str(uuid4())
            conn.execute(
                """
                INSERT INTO verification_proofs (
                    proof_id, session_id, business_event_id, subject_type, subject_id,
                    admin_id, scene, record_id, action, template_id, template_version,
                    status, result_code, issued_at, expires_at, finalized_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ready', ?, ?, ?, NULL)
                """,
                (
                    proof_id,
                    session_id,
                    session["business_event_id"],
                    session["subject_type"],
                    session["subject_id"],
                    session["admin_id"],
                    session["scene"],
                    session["record_id"],
                    session["action"],
                    session["template_id"],
                    session["template_version"],
                    result_code,
                    _to_iso(now),
                    _to_iso(expires_at),
                ),
            )
            conn.execute(
                """
                UPDATE verification_sessions
                SET status = 'passed',
                    result_code = ?,
                    verified_at = ?,
                    proof_id = ?,
                    verifying_started_at = NULL,
                    action_results_json = COALESCE(?, action_results_json)
                WHERE session_id = ?
                  AND status = 'verifying'
                  AND (? IS NULL OR verifying_started_at = ?)
                """,
                (
                    result_code,
                    _to_iso(now),
                    proof_id,
                    action_results_json,
                    session_id,
                    verifying_started_at,
                    verifying_started_at,
                ),
            )
            proof = conn.execute(
                "SELECT * FROM verification_proofs WHERE proof_id = ?",
                (proof_id,),
            ).fetchone()
            conn.commit()
        return _row_to_proof(proof) if proof else None

    def get_proof(self, proof_id: str) -> Optional[VerificationProof]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM verification_proofs WHERE proof_id = ?",
                (proof_id,),
            ).fetchone()
        return _row_to_proof(row) if row else None

    def finalize_proof(self, proof_id: str, business_event_id: str) -> Optional[VerificationProof]:
        now = _utcnow()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            proof = conn.execute(
                "SELECT * FROM verification_proofs WHERE proof_id = ?",
                (proof_id,),
            ).fetchone()
            if proof is None:
                conn.rollback()
                return None
            if proof["business_event_id"] != business_event_id:
                conn.rollback()
                raise ValueError("proof 与业务事件不一致")
            if proof["status"] == "ready":
                expires_at = _from_iso(proof["expires_at"])
                if expires_at and _utcnow() > expires_at:
                    conn.execute(
                        "UPDATE verification_proofs SET status = 'expired' WHERE proof_id = ?",
                        (proof_id,),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE verification_proofs
                        SET status = 'finalized', finalized_at = ?
                        WHERE proof_id = ?
                        """,
                        (_to_iso(now), proof_id),
                    )
            updated = conn.execute(
                "SELECT * FROM verification_proofs WHERE proof_id = ?",
                (proof_id,),
            ).fetchone()
            conn.commit()
        return _row_to_proof(updated) if updated else None

    def _connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self.database_path),
            timeout=15,
            isolation_level=None,
            factory=_ClosingConnection,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=15000")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS face_templates (
                    template_id TEXT PRIMARY KEY,
                    subject_type TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    embedding_blob BLOB NOT NULL,
                    embedding_dimension INTEGER NOT NULL,
                    bbox_json TEXT NOT NULL,
                    template_version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_image_hash TEXT,
                    created_at TEXT NOT NULL,
                    activated_at TEXT,
                    revoked_at TEXT
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_face_templates_active
                    ON face_templates(subject_type, subject_id)
                    WHERE status = 'active';

                CREATE INDEX IF NOT EXISTS idx_face_templates_subject
                    ON face_templates(subject_type, subject_id, template_version);

                CREATE TABLE IF NOT EXISTS verification_sessions (
                    session_id TEXT PRIMARY KEY,
                    request_id TEXT UNIQUE,
                    subject_type TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    admin_id TEXT,
                    scene TEXT NOT NULL,
                    business_event_id TEXT NOT NULL,
                        record_id TEXT,
                        action TEXT,
                        template_id TEXT NOT NULL,
                        template_version INTEGER NOT NULL,
                        expected_actions_json TEXT NOT NULL,
                    upload_token TEXT NOT NULL,
                    upload_token_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                        result_code TEXT NOT NULL,
                        issued_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL,
                        verifying_started_at TEXT,
                        verified_at TEXT,
                        proof_id TEXT,
                        action_results_json TEXT
                    );

                CREATE INDEX IF NOT EXISTS idx_verification_sessions_event
                    ON verification_sessions(business_event_id, scene, subject_type, subject_id);

                CREATE TABLE IF NOT EXISTS verification_proofs (
                    proof_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL UNIQUE,
                    business_event_id TEXT NOT NULL,
                    subject_type TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    admin_id TEXT,
                    scene TEXT NOT NULL,
                    record_id TEXT,
                    action TEXT,
                    template_id TEXT NOT NULL,
                    template_version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    result_code TEXT NOT NULL,
                    issued_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    finalized_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_verification_proofs_event
                    ON verification_proofs(business_event_id, scene, subject_type, subject_id);
                """
            )
            _ensure_column(conn, "verification_sessions", "upload_token", "TEXT")
            _ensure_nullable_column(conn, "verification_sessions", "verifying_started_at", "TEXT")
            _ensure_nullable_column(conn, "verification_sessions", "action_results_json", "TEXT")

    def check_schema(self) -> Dict[str, object]:
        with self._lock, self._connect() as conn:
            tables = {}
            for table_name, required in (
                ("face_templates", _SQLITE_TEMPLATE_COLUMNS),
                ("verification_sessions", _SQLITE_SESSION_COLUMNS),
                ("verification_proofs", _SQLITE_PROOF_COLUMNS),
            ):
                columns = {
                    str(row["name"])
                    for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
                }
                missing = sorted(required - columns)
                if missing:
                    raise RuntimeError(
                        f"SQLite 状态表 {table_name} 缺少字段：{', '.join(missing)}"
                    )
                tables[table_name] = {"table": table_name, "columns": sorted(columns)}
        return {"backend": self.backend_name, "tables": tables}


class MemoryStore(Store):
    """Compatibility alias for tests and older demo imports."""


def _row_to_template(row: sqlite3.Row) -> FaceTemplate:
    return FaceTemplate(
        template_id=row["template_id"],
        subject_type=row["subject_type"],
        subject_id=row["subject_id"],
        embedding=_embedding_from_blob(row["embedding_blob"], int(row["embedding_dimension"])),
        bbox=[float(value) for value in json.loads(row["bbox_json"])],
        template_version=int(row["template_version"]),
        status=row["status"],
        source_type=row["source_type"],
        source_image_hash=row["source_image_hash"],
        created_at=_from_iso(row["created_at"]) or _utcnow(),
        activated_at=_from_iso(row["activated_at"]),
        revoked_at=_from_iso(row["revoked_at"]),
    )


def _row_to_session(row: sqlite3.Row) -> VerificationSession:
    return VerificationSession(
        session_id=row["session_id"],
        request_id=row["request_id"],
        subject_type=row["subject_type"],
        subject_id=row["subject_id"],
        admin_id=row["admin_id"],
        scene=row["scene"],
        business_event_id=row["business_event_id"],
        record_id=row["record_id"],
        action=row["action"],
        template_id=row["template_id"],
        template_version=int(row["template_version"]),
        expected_actions=list(json.loads(row["expected_actions_json"])),
        upload_token=row["upload_token"],
        status=row["status"],
        result_code=row["result_code"],
        issued_at=_from_iso(row["issued_at"]) or _utcnow(),
        expires_at=_from_iso(row["expires_at"]) or _utcnow(),
        verifying_started_at=_from_iso(row["verifying_started_at"]),
        verified_at=_from_iso(row["verified_at"]),
        proof_id=row["proof_id"],
        action_results=_action_results_from_json(row["action_results_json"]),
    )


def _row_to_proof(row: sqlite3.Row) -> VerificationProof:
    status = row["status"]
    expires_at = _from_iso(row["expires_at"]) or _utcnow()
    if status == "ready" and _utcnow() > expires_at:
        status = "expired"
    return VerificationProof(
        proof_id=row["proof_id"],
        session_id=row["session_id"],
        business_event_id=row["business_event_id"],
        subject_type=row["subject_type"],
        subject_id=row["subject_id"],
        admin_id=row["admin_id"],
        scene=row["scene"],
        record_id=row["record_id"],
        action=row["action"],
        template_id=row["template_id"],
        template_version=int(row["template_version"]),
        status=status,
        result_code=row["result_code"],
        issued_at=_from_iso(row["issued_at"]) or _utcnow(),
        expires_at=expires_at,
        finalized_at=_from_iso(row["finalized_at"]),
    )


_SQLITE_TEMPLATE_COLUMNS = {
    "template_id", "subject_type", "subject_id", "embedding_blob",
    "embedding_dimension", "bbox_json", "template_version", "status",
    "source_type", "source_image_hash", "created_at", "activated_at", "revoked_at",
}
_SQLITE_SESSION_COLUMNS = {
    "session_id", "request_id", "subject_type", "subject_id", "admin_id", "scene",
    "business_event_id", "record_id", "action", "template_id", "template_version",
    "expected_actions_json", "upload_token", "upload_token_hash", "status", "result_code",
    "issued_at", "expires_at", "verifying_started_at", "verified_at", "proof_id",
    "action_results_json",
}
_SQLITE_PROOF_COLUMNS = {
    "proof_id", "session_id", "business_event_id", "subject_type", "subject_id",
    "admin_id", "scene", "record_id", "action", "template_id", "template_version",
    "status", "result_code", "issued_at", "expires_at", "finalized_at",
}


def _build_store():
    settings = get_settings()
    if settings.state_db_dsn.strip():
        from .mysql_store import MySQLStore

        return MySQLStore(settings)
    return MemoryStore()


store = _build_store()
