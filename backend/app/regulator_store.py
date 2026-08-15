from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import re
from typing import Any, Dict
from urllib.parse import unquote, urlparse


class RegulatorStoreError(RuntimeError):
    """Raised when a regulator result cannot be persisted."""


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_REQUIRED_COLUMNS = {
    "business_event_id",
    "record_id",
    "subject_type",
    "subject_id",
    "scene",
    "action",
    "status",
    "result_code",
    "result_msg",
    "created_at",
    "updated_at",
}


def _unix_now() -> int:
    return int(datetime.now(timezone.utc).timestamp())


class RegulatorStore:
    """Persist only fa_face_verify_regulator_status results.

    This is intentionally separate from the face service's SQLite session/proof
    store. Existing business tables and their existing write paths are untouched.
    """

    def __init__(self, settings: Any) -> None:
        self.enabled = bool(settings.regulator_db_enabled or settings.regulator_db_dsn)
        self.dsn = (settings.regulator_db_dsn or "").strip()
        self.table = self._validate_identifier(settings.regulator_db_table)
        self.connect_timeout = max(1, int(settings.regulator_db_connect_timeout_seconds))
        self.read_timeout = max(1, int(settings.regulator_db_read_timeout_seconds))
        self.write_timeout = max(1, int(settings.regulator_db_write_timeout_seconds))
        self._columns_cache: set[str] | None = None
        if self.enabled and not self.dsn:
            raise RegulatorStoreError("监管结果库已启用，但 FACE_DEMO_REGULATOR_DB_DSN 未配置")

    @staticmethod
    def _validate_identifier(value: str) -> str:
        value = (value or "").strip()
        if not value or not _IDENTIFIER.fullmatch(value):
            raise RegulatorStoreError(f"非法监管结果表名：{value!r}")
        return value

    def _connect(self):
        if not self.enabled:
            raise RegulatorStoreError("监管结果库未启用")
        parsed = urlparse(self.dsn)
        if parsed.scheme not in {"mysql", "mysql+pymysql"}:
            raise RegulatorStoreError("FACE_DEMO_REGULATOR_DB_DSN 必须使用 mysql:// 或 mysql+pymysql://")
        if not parsed.hostname or not parsed.path.strip("/"):
            raise RegulatorStoreError("监管结果库 DSN 缺少 host 或 database")
        try:
            import pymysql
            from pymysql.cursors import DictCursor
        except ImportError as exc:
            raise RegulatorStoreError("监管结果库已启用，但未安装 PyMySQL") from exc

        return pymysql.connect(
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

    def _columns(self, conn) -> set[str]:
        if self._columns_cache is None:
            with closing(conn.cursor()) as cursor:
                cursor.execute(f"SHOW COLUMNS FROM `{self.table}`")
                self._columns_cache = {str(row["Field"]) for row in cursor.fetchall()}
        return self._columns_cache

    def check_schema(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        try:
            with closing(self._connect()) as conn:
                columns = self._columns(conn)
                missing = sorted(_REQUIRED_COLUMNS - columns)
                if missing:
                    raise RegulatorStoreError(
                        f"监管结果表 {self.table} 缺少字段：{', '.join(missing)}"
                    )
                return {"enabled": True, "table": self.table, "columns": sorted(columns)}
        except RegulatorStoreError:
            raise
        except Exception as exc:
            raise RegulatorStoreError(f"监管结果表连接或结构检查失败：{exc}") from exc

    def record_result(
        self,
        session: Any,
        status: str,
        result_code: str,
        result_msg: str,
    ) -> None:
        if not self.enabled:
            return
        if status not in {"success", "failed", "expired"}:
            raise RegulatorStoreError(f"非法监管状态：{status}")

        now = _unix_now()
        values = {
            "business_event_id": str(session.business_event_id),
            "record_id": session.record_id,
            "subject_type": str(session.subject_type),
            "subject_id": str(session.subject_id),
            "scene": str(session.scene),
            "action": str(session.action or ("login" if session.scene == "login" else "audit")),
            "status": status,
            "result_code": str(result_code)[:64],
            "result_msg": str(result_msg or result_code)[:255],
            "created_at": now,
            "updated_at": now,
        }
        try:
            with closing(self._connect()) as conn, closing(conn.cursor()) as cursor:
                columns = self._columns(conn)
                missing = sorted(_REQUIRED_COLUMNS - columns)
                if missing:
                    raise RegulatorStoreError(
                        f"监管结果表 {self.table} 缺少字段：{', '.join(missing)}"
                    )
                fields = list(_REQUIRED_COLUMNS)
                # Keep the SQL deterministic and use parameters for every value.
                fields = [
                    "business_event_id",
                    "record_id",
                    "subject_type",
                    "subject_id",
                    "scene",
                    "action",
                    "status",
                    "result_code",
                    "result_msg",
                    "created_at",
                    "updated_at",
                ]
                quoted = ", ".join(f"`{field}`" for field in fields)
                placeholders = ", ".join(["%s"] * len(fields))
                updates = ", ".join(
                    f"`{field}` = VALUES(`{field}`)"
                    for field in fields
                    if field not in {"business_event_id", "created_at"}
                )
                cursor.execute(
                    f"INSERT INTO `{self.table}` ({quoted}) VALUES ({placeholders}) "
                    f"ON DUPLICATE KEY UPDATE {updates}",
                    [values[field] for field in fields],
                )
                conn.commit()
        except RegulatorStoreError:
            raise
        except Exception as exc:
            raise RegulatorStoreError(f"写入监管人脸核验结果失败：{exc}") from exc
