from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
import sys
from typing import Any, Dict, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config import get_settings
from backend.app.stores import store


TABLES = {
    "face_templates": (
        "template_id",
        (
            "template_id", "subject_type", "subject_id", "embedding_blob",
            "embedding_dimension", "bbox_json", "template_version", "status",
            "source_type", "source_image_hash", "created_at", "activated_at", "revoked_at",
        ),
    ),
    "verification_sessions": (
        "session_id",
        (
            "session_id", "request_id", "subject_type", "subject_id", "admin_id", "scene",
            "business_event_id", "record_id", "action", "template_id", "template_version",
            "expected_actions_json", "upload_token", "upload_token_hash", "status", "result_code",
            "issued_at", "expires_at", "verifying_started_at", "verified_at", "proof_id",
            "action_results_json",
        ),
    ),
    "verification_proofs": (
        "proof_id",
        (
            "proof_id", "session_id", "business_event_id", "subject_type", "subject_id",
            "admin_id", "scene", "record_id", "action", "template_id", "template_version",
            "status", "result_code", "issued_at", "expires_at", "finalized_at",
        ),
    ),
}


def _normalized(value: Any) -> Any:
    if isinstance(value, memoryview):
        return value.tobytes()
    return value


def _assert_same(
    table_name: str,
    primary_key: str,
    source: Dict[str, Any],
    target: Dict[str, Any],
    fields: Iterable[str],
) -> None:
    differences = [
        field
        for field in fields
        if _normalized(source[field]) != _normalized(target[field])
    ]
    if differences:
        raise RuntimeError(
            f"目标表 {table_name} 的 {primary_key}={source[primary_key]!r} "
            f"与 SQLite 字段不一致：{', '.join(differences)}"
        )


def migrate(source_path: Path, target: Any, check_only: bool = False) -> Dict[str, Dict[str, int]]:
    if not source_path.is_file():
        raise RuntimeError(f"SQLite 文件不存在：{source_path}")

    source = sqlite3.connect(str(source_path))
    source.row_factory = sqlite3.Row
    summary: Dict[str, Dict[str, int]] = {}
    try:
        with target._lock, target._connect() as destination:
            destination.execute("BEGIN IMMEDIATE")
            for table_name, (primary_key, fields) in TABLES.items():
                source_rows = source.execute(
                    f"SELECT {', '.join(fields)} FROM {table_name} ORDER BY {primary_key}"
                ).fetchall()
                inserted = 0
                matched = 0
                for source_row in source_rows:
                    values = {field: source_row[field] for field in fields}
                    existing = destination.execute(
                        f"SELECT {', '.join(fields)} FROM {table_name} "
                        f"WHERE {primary_key} = ?",
                        (values[primary_key],),
                    ).fetchone()
                    if existing is not None:
                        _assert_same(table_name, primary_key, values, existing, fields)
                        matched += 1
                        continue
                    if check_only:
                        raise RuntimeError(
                            f"目标表 {table_name} 缺少 {primary_key}={values[primary_key]!r}"
                        )
                    placeholders = ", ".join(["?"] * len(fields))
                    destination.execute(
                        f"INSERT INTO {table_name} ({', '.join(fields)}) "
                        f"VALUES ({placeholders})",
                        [values[field] for field in fields],
                    )
                    inserted += 1
                summary[table_name] = {
                    "source": len(source_rows),
                    "inserted": inserted,
                    "matched": matched,
                }
    finally:
        source.close()

    # Loading active templates also verifies that encrypted blobs still decrypt.
    source = sqlite3.connect(str(source_path))
    try:
        active_subjects = source.execute(
            "SELECT subject_type, subject_id FROM face_templates WHERE status = 'active'"
        ).fetchall()
    finally:
        source.close()
    for subject_type, subject_id in active_subjects:
        if target.get_active_template(str(subject_type), str(subject_id)) is None:
            raise RuntimeError(f"迁移后无法加载有效模板：{subject_type}/{subject_id}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate face-service state from SQLite to MySQL")
    parser.add_argument("--source", help="SQLite source file; defaults to FACE_DEMO_DATABASE_PATH")
    parser.add_argument("--check-only", action="store_true", help="Compare only; do not insert rows")
    args = parser.parse_args()

    settings = get_settings()
    source_path = Path(args.source or settings.database_path).expanduser().resolve()
    target = store
    if getattr(target, "backend_name", "") != "mysql":
        raise RuntimeError("FACE_DEMO_STATE_DB_DSN 未配置，当前状态存储不是 MySQL")
    summary = migrate(source_path, target, check_only=args.check_only)
    for table_name, counts in summary.items():
        print(
            f"{table_name}: source={counts['source']} "
            f"inserted={counts['inserted']} matched={counts['matched']}"
        )


if __name__ == "__main__":
    main()
