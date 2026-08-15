from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import json
import re
from typing import Any, Dict, Iterable, Optional
from urllib.parse import unquote, urlparse


class BusinessStoreError(RuntimeError):
    """Raised when the configured business database cannot satisfy a write."""


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_AUDIT_KEYS = {
    "business_event_id",
    "event_id",
    "verify_event_id",
    "session_id",
    "proof_id",
    "subject_id",
    "doctor_id",
    "result",
    "verify_result",
    "result_code",
    "status",
    "verify_status",
}


def _db_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)


class BusinessStore:
    """MySQL adapter for the chbzg fa_face_* business tables.

    The face service keeps embeddings, sessions and proofs in its local store. This
    adapter mirrors the business-facing identity, event and consumption state into
    chbzg's MySQL database and fails closed when that mirror cannot be updated.
    """

    def __init__(self, settings: Any) -> None:
        self.enabled = bool(settings.business_db_enabled or settings.business_db_dsn)
        self.dsn = (settings.business_db_dsn or "").strip()
        self.connect_timeout = max(1, int(settings.business_db_connect_timeout_seconds))
        self.read_timeout = max(1, int(settings.business_db_read_timeout_seconds))
        self.write_timeout = max(1, int(settings.business_db_write_timeout_seconds))
        self.profile_table = self._validate_identifier(settings.business_db_profile_table)
        self.event_table = self._validate_identifier(settings.business_db_event_table)
        self.consumption_table = self._validate_identifier(settings.business_db_consumption_table)
        self.log_table = self._validate_identifier(settings.business_db_log_table)
        self.regulator_table = self._validate_identifier(settings.business_db_regulator_table)
        self.strict_audit = bool(settings.business_db_strict_audit)
        self._column_cache: Dict[str, set[str]] = {}
        if self.enabled and not self.dsn:
            raise BusinessStoreError("业务数据库已启用，但 FACE_DEMO_BUSINESS_DB_DSN 未配置")

    @staticmethod
    def _validate_identifier(value: str) -> str:
        value = (value or "").strip()
        if not value or not _IDENTIFIER.fullmatch(value):
            raise BusinessStoreError(f"非法业务表名：{value!r}")
        return value

    @staticmethod
    def _quote_identifier(value: str) -> str:
        return f"`{value}`"

    def _connect(self):
        if not self.enabled:
            raise BusinessStoreError("业务数据库未启用")
        parsed = urlparse(self.dsn)
        if parsed.scheme not in {"mysql", "mysql+pymysql"}:
            raise BusinessStoreError("FACE_DEMO_BUSINESS_DB_DSN 必须使用 mysql:// 或 mysql+pymysql://")
        if not parsed.hostname or not parsed.path.strip("/"):
            raise BusinessStoreError("业务数据库 DSN 缺少 host 或 database")
        try:
            import pymysql
            from pymysql.cursors import DictCursor
        except ImportError as exc:
            raise BusinessStoreError("业务数据库已启用，但未安装 PyMySQL") from exc

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

    def _columns(self, conn, table: str) -> set[str]:
        if table not in self._column_cache:
            with closing(conn.cursor()) as cursor:
                cursor.execute(f"SHOW COLUMNS FROM {self._quote_identifier(table)}")
                self._column_cache[table] = {str(row["Field"]) for row in cursor.fetchall()}
        return self._column_cache[table]

    def _require_columns(self, conn, table: str, required: Iterable[str]) -> set[str]:
        columns = self._columns(conn, table)
        missing = sorted(set(required) - columns)
        if missing:
            raise BusinessStoreError(f"业务表 {table} 缺少字段：{', '.join(missing)}")
        return columns

    @staticmethod
    def _pick(values: Dict[str, Any], columns: set[str]) -> Dict[str, Any]:
        return {key: value for key, value in values.items() if key in columns and value is not None}

    def _insert(self, cursor, table: str, values: Dict[str, Any]) -> None:
        if not values:
            raise BusinessStoreError(f"业务表 {table} 没有可写入字段")
        fields = list(values)
        quoted = ", ".join(self._quote_identifier(item) for item in fields)
        placeholders = ", ".join(["%s"] * len(fields))
        cursor.execute(
            f"INSERT INTO {self._quote_identifier(table)} ({quoted}) VALUES ({placeholders})",
            [values[field] for field in fields],
        )

    def _update(self, cursor, table: str, values: Dict[str, Any], where: Dict[str, Any]) -> None:
        if not values:
            return
        assignments = ", ".join(f"{self._quote_identifier(key)} = %s" for key in values)
        predicates = " AND ".join(f"{self._quote_identifier(key)} = %s" for key in where)
        cursor.execute(
            f"UPDATE {self._quote_identifier(table)} SET {assignments} WHERE {predicates}",
            [*values.values(), *where.values()],
        )

    def _one(self, cursor, table: str, where: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        predicates = " AND ".join(f"{self._quote_identifier(key)} = %s" for key in where)
        cursor.execute(
            f"SELECT * FROM {self._quote_identifier(table)} WHERE {predicates} LIMIT 1",
            list(where.values()),
        )
        return cursor.fetchone()

    def check_schema(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        try:
            with closing(self._connect()) as conn:
                profile = self._require_columns(
                    conn,
                    self.profile_table,
                    {"subject_type", "subject_id", "active_template_id", "template_version", "status"},
                )
                event = self._require_columns(
                    conn,
                    self.event_table,
                    {"business_event_id", "subject_type", "subject_id", "scene", "status"},
                )
                consumption = self._require_columns(
                    conn,
                    self.consumption_table,
                    {"proof_id", "business_event_id", "subject_type", "subject_id"},
                )
                optional = {}
                for table in (self.log_table, self.regulator_table):
                    if table:
                        columns = self._columns(conn, table)
                        if self.strict_audit and not (_AUDIT_KEYS & columns):
                            raise BusinessStoreError(f"审计表 {table} 没有可识别的业务字段")
                        optional[table] = sorted(columns)
                return {
                    "enabled": True,
                    "profile_columns": sorted(profile),
                    "event_columns": sorted(event),
                    "consumption_columns": sorted(consumption),
                    "optional_columns": optional,
                }
        except BusinessStoreError:
            raise
        except Exception as exc:
            raise BusinessStoreError(f"业务数据库连接或表检查失败：{exc}") from exc

    def get_active_profile(self, subject_type: str, subject_id: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            with closing(self._connect()) as conn, closing(conn.cursor()) as cursor:
                self._require_columns(
                    conn,
                    self.profile_table,
                    {"subject_type", "subject_id", "active_template_id", "template_version", "status"},
                )
                return self._one(
                    cursor,
                    self.profile_table,
                    {"subject_type": subject_type, "subject_id": subject_id},
                )
        except BusinessStoreError:
            raise
        except Exception as exc:
            raise BusinessStoreError(f"读取人脸模板映射失败：{exc}") from exc

    def sync_profile(self, template: Any) -> None:
        if not self.enabled:
            return
        values = {
            "subject_type": template.subject_type,
            "subject_id": template.subject_id,
            "active_template_id": template.template_id,
            "template_version": template.template_version,
            "avatar_hash": template.source_image_hash,
            "status": "active",
            "failure_code": None,
            "updated_at": _db_now(),
        }
        try:
            with closing(self._connect()) as conn, closing(conn.cursor()) as cursor:
                columns = self._require_columns(
                    conn,
                    self.profile_table,
                    {"subject_type", "subject_id", "active_template_id", "template_version", "status"},
                )
                current = self._one(
                    cursor,
                    self.profile_table,
                    {"subject_type": template.subject_type, "subject_id": template.subject_id},
                )
                filtered = self._pick(values, columns)
                if current:
                    self._update(
                        cursor,
                        self.profile_table,
                        filtered,
                        {"subject_type": template.subject_type, "subject_id": template.subject_id},
                    )
                else:
                    self._insert(cursor, self.profile_table, filtered)
                conn.commit()
        except BusinessStoreError:
            raise
        except Exception as exc:
            raise BusinessStoreError(f"写入人脸模板映射失败：{exc}") from exc

    def record_event_issued(self, session: Any) -> None:
        if not self.enabled:
            return
        values = {
            "business_event_id": session.business_event_id,
            "request_id": session.request_id,
            "subject_type": session.subject_type,
            "subject_id": session.subject_id,
            "admin_id": session.admin_id,
            "scene": session.scene,
            "record_id": session.record_id,
            "action": session.action,
            "required": 1,
            "session_id": session.session_id,
            "status": "session_issued",
            "result_code": "ISSUED",
            "expires_at": session.expires_at.replace(tzinfo=None),
            "created_at": _db_now(),
            "policy_snapshot": json.dumps(
                {"expected_actions": session.expected_actions}, ensure_ascii=True, separators=(",", ":")
            ),
        }
        try:
            with closing(self._connect()) as conn, closing(conn.cursor()) as cursor:
                columns = self._require_columns(
                    conn,
                    self.event_table,
                    {"business_event_id", "subject_type", "subject_id", "scene", "status"},
                )
                current = self._one(cursor, self.event_table, {"business_event_id": session.business_event_id})
                filtered = self._pick(values, columns)
                if current:
                    for field in ("subject_type", "subject_id", "scene", "record_id", "action"):
                        if field in current and current[field] is not None and values.get(field) != str(current[field]):
                            raise BusinessStoreError(f"业务事件 {session.business_event_id} 字段冲突：{field}")
                    if current.get("session_id") not in (None, "", session.session_id):
                        raise BusinessStoreError(f"业务事件 {session.business_event_id} 已绑定其它 session")
                    self._update(cursor, self.event_table, filtered, {"business_event_id": session.business_event_id})
                else:
                    self._insert(cursor, self.event_table, filtered)
                conn.commit()
        except BusinessStoreError:
            raise
        except Exception as exc:
            raise BusinessStoreError(f"写入人脸核验业务事件失败：{exc}") from exc

    def record_event_result(
        self,
        session: Any,
        passed: bool,
        result_code: str,
        proof_id: Optional[str] = None,
        action_results: Optional[list] = None,
    ) -> None:
        if not self.enabled:
            return
        values = {
            "status": "verified" if passed else "failed",
            "result_code": result_code,
            "proof_id": proof_id,
            "session_id": session.session_id,
            "verified_at": _db_now() if passed else None,
            "last_error": None if passed else result_code,
            "action_results_json": json.dumps(action_results or [], ensure_ascii=True, separators=(",", ":")),
        }
        try:
            with closing(self._connect()) as conn, closing(conn.cursor()) as cursor:
                columns = self._require_columns(
                    conn,
                    self.event_table,
                    {"business_event_id", "subject_type", "subject_id", "scene", "status"},
                )
                event = self._one(cursor, self.event_table, {"business_event_id": session.business_event_id})
                if not event:
                    raise BusinessStoreError(f"未找到业务事件：{session.business_event_id}")
                self._update(cursor, self.event_table, self._pick(values, columns), {"business_event_id": session.business_event_id})
                self._write_audit(cursor, conn, session, passed, result_code, proof_id, "verified")
                conn.commit()
        except BusinessStoreError:
            raise
        except Exception as exc:
            raise BusinessStoreError(f"更新人脸核验业务事件失败：{exc}") from exc

    def consume_proof(self, proof: Any, business_event_id: str) -> None:
        if not self.enabled:
            return
        try:
            with closing(self._connect()) as conn, closing(conn.cursor()) as cursor:
                event_columns = self._require_columns(
                    conn,
                    self.event_table,
                    {"business_event_id", "subject_type", "subject_id", "scene", "status"},
                )
                consumption_columns = self._require_columns(
                    conn,
                    self.consumption_table,
                    {"proof_id", "business_event_id", "subject_type", "subject_id"},
                )
                event = self._one(cursor, self.event_table, {"business_event_id": business_event_id})
                if not event:
                    raise BusinessStoreError(f"未找到业务事件：{business_event_id}")
                self._assert_proof_binding(proof, event)

                existing = self._one(cursor, self.consumption_table, {"proof_id": proof.proof_id})
                by_event = self._one(cursor, self.consumption_table, {"business_event_id": business_event_id})
                if existing or by_event:
                    existing = existing or by_event
                    if existing.get("proof_id") != proof.proof_id:
                        raise BusinessStoreError("业务事件已经消费过其它 proof")
                else:
                    values = {
                        "proof_id": proof.proof_id,
                        "business_event_id": business_event_id,
                        "scene": proof.scene,
                        "subject_type": proof.subject_type,
                        "subject_id": proof.subject_id,
                        "record_id": proof.record_id,
                        "action": proof.action,
                        "consumed_at": _db_now(),
                    }
                    self._insert(cursor, self.consumption_table, self._pick(values, consumption_columns))

                event_values = {
                    "status": "consumed",
                    "proof_id": proof.proof_id,
                    "consumed_at": _db_now(),
                    "finalize_status": "succeeded",
                    "finalize_attempts": 1,
                }
                self._update(cursor, self.event_table, self._pick(event_values, event_columns), {"business_event_id": business_event_id})
                self._write_audit(cursor, conn, proof, True, "CONSUMED", proof.proof_id, "consumed")
                conn.commit()
        except BusinessStoreError:
            raise
        except Exception as exc:
            raise BusinessStoreError(f"消费人脸核验 proof 失败：{exc}") from exc

    @staticmethod
    def _assert_proof_binding(proof: Any, event: Dict[str, Any]) -> None:
        for field in ("subject_type", "subject_id", "scene"):
            if field in event and event[field] is not None and str(event[field]) != str(getattr(proof, field)):
                raise BusinessStoreError(f"proof 与业务事件不一致：{field}")
        if event.get("proof_id") not in (None, "", proof.proof_id):
            raise BusinessStoreError("业务事件已经绑定其它 proof")

    def _write_audit(
        self,
        cursor,
        conn,
        subject: Any,
        passed: bool,
        result_code: str,
        proof_id: Optional[str],
        status: str,
    ) -> None:
        values = {
            "business_event_id": getattr(subject, "business_event_id", None),
            "event_id": getattr(subject, "business_event_id", None),
            "verify_event_id": getattr(subject, "business_event_id", None),
            "session_id": getattr(subject, "session_id", None),
            "proof_id": proof_id,
            "subject_type": getattr(subject, "subject_type", None),
            "subject_id": getattr(subject, "subject_id", None),
            "doctor_id": getattr(subject, "subject_id", None)
            if getattr(subject, "subject_type", None) == "doctor"
            else None,
            "scene": getattr(subject, "scene", None),
            "verify_scene": getattr(subject, "scene", None),
            "action": getattr(subject, "action", None),
            "verify_type": getattr(subject, "action", None),
            "result": "pass" if passed else "fail",
            "verify_result": "pass" if passed else "fail",
            "result_status": "success" if passed else "failed",
            "result_code": result_code,
            "status": status,
            "verify_status": status,
            "created_at": _db_now(),
            "create_time": _db_now(),
            "verified_at": _db_now(),
            "verify_time": _db_now(),
            "consumed_at": _db_now() if status == "consumed" else None,
            "message": result_code,
            "remark": result_code,
        }
        for table in (self.log_table, self.regulator_table):
            if not table:
                continue
            try:
                columns = self._columns(conn, table)
                table_values = dict(values)
                if table == self.regulator_table:
                    table_values["status"] = "succeeded" if passed else "failed"
                    table_values["verify_status"] = table_values["status"]
                    table_values["result_status"] = table_values["status"]
                available = self._pick(table_values, columns)
                if available:
                    # Regulator status is one row per business event in the
                    # deployed schema; update it on retries. The doctor log is
                    # append-only and records each attempt.
                    event_key = next(
                        (key for key in ("business_event_id", "event_id", "verify_event_id") if key in columns),
                        None,
                    )
                    existing = (
                        self._one(cursor, table, {event_key: table_values[event_key]})
                        if table == self.regulator_table and event_key and table_values.get(event_key)
                        else None
                    )
                    if existing:
                        self._update(cursor, table, available, {event_key: table_values[event_key]})
                    else:
                        self._insert(cursor, table, available)
                elif self.strict_audit:
                    raise BusinessStoreError(f"审计表 {table} 没有可识别的业务字段")
            except Exception:
                if self.strict_audit:
                    raise
