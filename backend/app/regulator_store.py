from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import json
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
_DOCTOR_FACE_LOG_REQUIRED_COLUMNS = {
    "id",
    "doctor_id",
    "action",
    "detail",
    "record_id",
    "verify_time",
    "ip",
    "user_agent",
    "create_time",
}


def _unix_now() -> int:
    return int(datetime.now(timezone.utc).timestamp())


class RegulatorStore:
    """Persist verification results to the configured existing result tables.

    This is intentionally separate from the face service's SQLite session/proof
    store. The doctor log is optional and is written only for doctor sessions.
    """

    def __init__(self, settings: Any) -> None:
        self.doctor_face_log_enabled = bool(settings.doctor_face_log_enabled)
        self.enabled = bool(
            settings.regulator_db_enabled
            or settings.regulator_db_dsn
            or self.doctor_face_log_enabled
        )
        self.dsn = (settings.regulator_db_dsn or "").strip()
        self.table = self._validate_identifier(settings.regulator_db_table)
        self.connect_timeout = max(1, int(settings.regulator_db_connect_timeout_seconds))
        self.read_timeout = max(1, int(settings.regulator_db_read_timeout_seconds))
        self.write_timeout = max(1, int(settings.regulator_db_write_timeout_seconds))
        self.doctor_face_log_table = self._validate_identifier(settings.doctor_face_log_table)
        self._columns_cache: set[str] | None = None
        self._doctor_face_log_columns_cache: set[str] | None = None
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

    def _doctor_face_log_columns(self, conn) -> set[str]:
        if self._doctor_face_log_columns_cache is None:
            with closing(conn.cursor()) as cursor:
                cursor.execute(f"SHOW COLUMNS FROM `{self.doctor_face_log_table}`")
                self._doctor_face_log_columns_cache = {
                    str(row["Field"]) for row in cursor.fetchall()
                }
        return self._doctor_face_log_columns_cache

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
                result = {"enabled": True, "table": self.table, "columns": sorted(columns)}
                if self.doctor_face_log_enabled:
                    doctor_log_columns = self._doctor_face_log_columns(conn)
                    doctor_log_missing = sorted(
                        _DOCTOR_FACE_LOG_REQUIRED_COLUMNS - doctor_log_columns
                    )
                    if doctor_log_missing:
                        raise RegulatorStoreError(
                            f"医生人脸核验日志表 {self.doctor_face_log_table} 缺少字段："
                            f"{', '.join(doctor_log_missing)}"
                        )
                    result["doctor_face_log"] = {
                        "enabled": True,
                        "table": self.doctor_face_log_table,
                        "columns": sorted(doctor_log_columns),
                    }
                else:
                    result["doctor_face_log"] = {"enabled": False}
                return result
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
        *,
        client_ip: str = "",
        user_agent: str = "",
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
                if self.doctor_face_log_enabled and str(session.subject_type) == "doctor":
                    self._record_doctor_face_log(
                        conn,
                        cursor,
                        session,
                        status,
                        result_code,
                        result_msg,
                        client_ip,
                        user_agent,
                    )
                conn.commit()
        except RegulatorStoreError:
            raise
        except Exception as exc:
            raise RegulatorStoreError(f"写入监管人脸核验结果失败：{exc}") from exc

    def _record_doctor_face_log(
        self,
        conn: Any,
        cursor: Any,
        session: Any,
        status: str,
        result_code: str,
        result_msg: str,
        client_ip: str,
        user_agent: str,
    ) -> None:
        columns = self._doctor_face_log_columns(conn)
        missing = sorted(_DOCTOR_FACE_LOG_REQUIRED_COLUMNS - columns)
        if missing:
            raise RegulatorStoreError(
                f"医生人脸核验日志表 {self.doctor_face_log_table} 缺少字段："
                f"{', '.join(missing)}"
            )

        business_action = str(
            session.action or ("login" if session.scene == "login" else "audit")
        )
        log_action = "login" if business_action == "login" else "audit"
        action_label = "登录" if log_action == "login" else "审核"
        result_label = {
            "success": "通过",
            "failed": "失败",
            "expired": "已过期",
        }[status]
        safe_result_msg = str(result_msg or result_code)
        if status != "success":
            # The existing monitor treats any detail containing "通过" as success.
            safe_result_msg = safe_result_msg.replace("未通过", "失败").replace("通过", "成功")
        detail = json.dumps(
            {
                "business_event_id": str(session.business_event_id),
                "status": status,
                "result": result_label,
                "result_code": str(result_code),
                "result_msg": safe_result_msg,
                "scene": str(session.scene),
                "action": business_action,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        cursor.execute(
            f"INSERT INTO `{self.doctor_face_log_table}` "
            "(`doctor_id`, `action`, `detail`, `record_id`, `verify_time`, `ip`, "
            "`user_agent`, `create_time`) "
            "VALUES (%s, %s, %s, %s, NOW(), %s, %s, UNIX_TIMESTAMP())",
            [
                str(session.subject_id)[:50],
                f"liveness_verify_{status}-{log_action}-{action_label}"[:50],
                detail,
                str(session.record_id or "")[:50],
                str(client_ip or "")[:50],
                str(user_agent or "")[:500],
            ],
        )
