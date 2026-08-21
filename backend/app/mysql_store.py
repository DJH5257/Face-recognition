from __future__ import annotations

from contextlib import closing
import re
import sqlite3
from threading import Lock
from typing import Any, Dict, Iterable, Optional, Sequence
from urllib.parse import unquote, urlparse

from .stores import Challenge, Enrollment, Store


class MySQLStoreError(RuntimeError):
    """Raised when the configured MySQL state store is unavailable or invalid."""


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_REQUIRED_COLUMNS = {
    "face_templates": {
        "template_id",
        "subject_type",
        "subject_id",
        "embedding_blob",
        "embedding_dimension",
        "bbox_json",
        "template_version",
        "status",
        "source_type",
        "source_image_hash",
        "created_at",
        "activated_at",
        "revoked_at",
    },
    "verification_sessions": {
        "session_id",
        "request_id",
        "subject_type",
        "subject_id",
        "admin_id",
        "scene",
        "business_event_id",
        "record_id",
        "action",
        "template_id",
        "template_version",
        "expected_actions_json",
        "upload_token",
        "upload_token_hash",
        "status",
        "result_code",
        "issued_at",
        "expires_at",
        "verifying_started_at",
        "verified_at",
        "proof_id",
        "action_results_json",
    },
    "verification_proofs": {
        "proof_id",
        "session_id",
        "business_event_id",
        "subject_type",
        "subject_id",
        "admin_id",
        "scene",
        "record_id",
        "action",
        "template_id",
        "template_version",
        "status",
        "result_code",
        "issued_at",
        "expires_at",
        "finalized_at",
    },
}


class _BufferedResult:
    def __init__(self, rows: Sequence[Dict[str, Any]], rowcount: int) -> None:
        self._rows = list(rows)
        self.rowcount = int(rowcount)

    def fetchone(self) -> Optional[Dict[str, Any]]:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[Dict[str, Any]]:
        return list(self._rows)


class _MySQLConnectionAdapter:
    """Expose the small sqlite-style connection surface used by Store."""

    def __init__(self, connection: Any, table_names: Dict[str, str]) -> None:
        self._connection = connection
        self._table_names = table_names

    def __enter__(self) -> "_MySQLConnectionAdapter":
        return self

    def __exit__(self, exc_type, _exc, _traceback) -> None:
        try:
            if exc_type is None:
                self._connection.commit()
            else:
                self._connection.rollback()
        finally:
            self._connection.close()

    def execute(self, query: str, params: Iterable[Any] = ()) -> _BufferedResult:
        normalized = query.strip().rstrip(";").upper()
        if normalized == "BEGIN IMMEDIATE":
            self._connection.begin()
            return _BufferedResult([], 0)

        translated = _translate_query(query, self._table_names)
        with closing(self._connection.cursor()) as cursor:
            try:
                cursor.execute(translated, tuple(params))
            except Exception as exc:
                if exc.__class__.__name__ == "IntegrityError":
                    raise sqlite3.IntegrityError(str(exc)) from exc
                raise
            rows = cursor.fetchall() if cursor.description else []
            return _BufferedResult(rows, cursor.rowcount)

    def commit(self) -> None:
        self._connection.commit()

    def rollback(self) -> None:
        self._connection.rollback()

    def close(self) -> None:
        self._connection.close()


def _translate_query(query: str, table_names: Dict[str, str]) -> str:
    translated = query
    for logical_name, physical_name in table_names.items():
        translated = re.sub(
            rf"\b{re.escape(logical_name)}\b",
            f"`{physical_name}`",
            translated,
        )
    return translated.replace("?", "%s")


class MySQLStore(Store):
    """MySQL-backed persistent template, verification-session and proof store."""

    backend_name = "mysql"

    def __init__(self, settings: Any) -> None:
        self.dsn = str(settings.state_db_dsn or "").strip()
        self.connect_timeout = max(1, int(settings.state_db_connect_timeout_seconds))
        self.read_timeout = max(1, int(settings.state_db_read_timeout_seconds))
        self.write_timeout = max(1, int(settings.state_db_write_timeout_seconds))
        self.table_names = {
            "face_templates": self._validate_identifier(settings.state_db_template_table),
            "verification_sessions": self._validate_identifier(settings.state_db_session_table),
            "verification_proofs": self._validate_identifier(settings.state_db_proof_table),
        }
        if not self.dsn:
            raise MySQLStoreError("FACE_DEMO_STATE_DB_DSN 未配置")
        self._parse_dsn()
        self._lock = Lock()
        self.enrollments: Dict[str, Enrollment] = {}
        self.challenges: Dict[str, Challenge] = {}
        self._init_db()

    @staticmethod
    def _validate_identifier(value: str) -> str:
        value = str(value or "").strip()
        if not value or not _IDENTIFIER.fullmatch(value):
            raise MySQLStoreError(f"非法状态存储表名：{value!r}")
        return value

    def _parse_dsn(self):
        parsed = urlparse(self.dsn)
        if parsed.scheme not in {"mysql", "mysql+pymysql"}:
            raise MySQLStoreError("FACE_DEMO_STATE_DB_DSN 必须使用 mysql:// 或 mysql+pymysql://")
        if not parsed.hostname or not parsed.path.strip("/"):
            raise MySQLStoreError("状态数据库 DSN 缺少 host 或 database")
        return parsed

    def _connect(self) -> _MySQLConnectionAdapter:
        parsed = self._parse_dsn()
        try:
            import pymysql
            from pymysql.cursors import DictCursor
        except ImportError as exc:
            raise MySQLStoreError("MySQL 状态存储已启用，但未安装 PyMySQL") from exc

        try:
            connection = pymysql.connect(
                host=parsed.hostname,
                port=parsed.port or 3306,
                user=unquote(parsed.username or ""),
                password=unquote(parsed.password or ""),
                database=unquote(parsed.path.strip("/")),
                charset="utf8mb4",
                autocommit=False,
                cursorclass=DictCursor,
                connect_timeout=self.connect_timeout,
                read_timeout=self.read_timeout,
                write_timeout=self.write_timeout,
            )
        except Exception as exc:
            raise MySQLStoreError(f"连接 MySQL 状态数据库失败：{exc}") from exc
        return _MySQLConnectionAdapter(connection, self.table_names)

    def _init_db(self) -> None:
        template_table = self.table_names["face_templates"]
        session_table = self.table_names["verification_sessions"]
        proof_table = self.table_names["verification_proofs"]
        ddl = [
            f"""
            CREATE TABLE IF NOT EXISTS `{template_table}` (
                `template_id` varchar(64) NOT NULL,
                `subject_type` varchar(32) NOT NULL,
                `subject_id` varchar(64) NOT NULL,
                `embedding_blob` longblob NOT NULL,
                `embedding_dimension` int unsigned NOT NULL,
                `bbox_json` text NOT NULL,
                `template_version` int unsigned NOT NULL,
                `status` varchar(16) NOT NULL,
                `source_type` varchar(32) NOT NULL,
                `source_image_hash` varchar(128) DEFAULT NULL,
                `created_at` varchar(40) NOT NULL,
                `activated_at` varchar(40) DEFAULT NULL,
                `revoked_at` varchar(40) DEFAULT NULL,
                PRIMARY KEY (`template_id`),
                KEY `idx_face_service_template_subject` (`subject_type`, `subject_id`, `template_version`),
                KEY `idx_face_service_template_active` (`subject_type`, `subject_id`, `status`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            f"""
            CREATE TABLE IF NOT EXISTS `{session_table}` (
                `session_id` varchar(64) NOT NULL,
                `request_id` varchar(128) DEFAULT NULL,
                `subject_type` varchar(32) NOT NULL,
                `subject_id` varchar(64) NOT NULL,
                `admin_id` varchar(64) DEFAULT NULL,
                `scene` varchar(64) NOT NULL,
                `business_event_id` varchar(128) NOT NULL,
                `record_id` varchar(128) DEFAULT NULL,
                `action` varchar(64) DEFAULT NULL,
                `template_id` varchar(64) NOT NULL,
                `template_version` int unsigned NOT NULL,
                `expected_actions_json` text NOT NULL,
                `upload_token` varchar(128) NOT NULL,
                `upload_token_hash` char(64) NOT NULL,
                `status` varchar(16) NOT NULL,
                `result_code` varchar(64) NOT NULL,
                `issued_at` varchar(40) NOT NULL,
                `expires_at` varchar(40) NOT NULL,
                `verifying_started_at` varchar(40) DEFAULT NULL,
                `verified_at` varchar(40) DEFAULT NULL,
                `proof_id` varchar(64) DEFAULT NULL,
                `action_results_json` longtext,
                PRIMARY KEY (`session_id`),
                UNIQUE KEY `uniq_face_service_session_request` (`request_id`),
                KEY `idx_face_service_session_event` (`business_event_id`, `scene`, `subject_type`, `subject_id`),
                KEY `idx_face_service_session_status` (`status`, `expires_at`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            f"""
            CREATE TABLE IF NOT EXISTS `{proof_table}` (
                `proof_id` varchar(64) NOT NULL,
                `session_id` varchar(64) NOT NULL,
                `business_event_id` varchar(128) NOT NULL,
                `subject_type` varchar(32) NOT NULL,
                `subject_id` varchar(64) NOT NULL,
                `admin_id` varchar(64) DEFAULT NULL,
                `scene` varchar(64) NOT NULL,
                `record_id` varchar(128) DEFAULT NULL,
                `action` varchar(64) DEFAULT NULL,
                `template_id` varchar(64) NOT NULL,
                `template_version` int unsigned NOT NULL,
                `status` varchar(16) NOT NULL,
                `result_code` varchar(64) NOT NULL,
                `issued_at` varchar(40) NOT NULL,
                `expires_at` varchar(40) NOT NULL,
                `finalized_at` varchar(40) DEFAULT NULL,
                PRIMARY KEY (`proof_id`),
                UNIQUE KEY `uniq_face_service_proof_session` (`session_id`),
                KEY `idx_face_service_proof_event` (`business_event_id`, `scene`, `subject_type`, `subject_id`),
                KEY `idx_face_service_proof_status` (`status`, `expires_at`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
        ]
        try:
            with self._connect() as conn:
                for statement in ddl:
                    # DDL already contains validated physical table names.
                    with closing(conn._connection.cursor()) as cursor:
                        cursor.execute(statement)
        except MySQLStoreError:
            raise
        except Exception as exc:
            raise MySQLStoreError(f"初始化 MySQL 状态表失败：{exc}") from exc

    def check_schema(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"backend": self.backend_name, "tables": {}}
        try:
            with self._connect() as conn:
                for logical_name, physical_name in self.table_names.items():
                    rows = conn.execute(f"SHOW COLUMNS FROM `{physical_name}`").fetchall()
                    columns = {str(row["Field"]) for row in rows}
                    missing = sorted(_REQUIRED_COLUMNS[logical_name] - columns)
                    if missing:
                        raise MySQLStoreError(
                            f"状态表 {physical_name} 缺少字段：{', '.join(missing)}"
                        )
                    result["tables"][logical_name] = {
                        "table": physical_name,
                        "columns": sorted(columns),
                    }
            return result
        except MySQLStoreError:
            raise
        except Exception as exc:
            raise MySQLStoreError(f"检查 MySQL 状态表失败：{exc}") from exc
