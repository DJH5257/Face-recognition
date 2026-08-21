import sqlite3
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from backend.app.mysql_store import (
    MySQLStore,
    MySQLStoreError,
    _MySQLConnectionAdapter,
    _translate_query,
)


def _settings(**overrides):
    values = {
        "state_db_dsn": "mysql+pymysql://face:secret@127.0.0.1:3306/test_face",
        "state_db_template_table": "fa_face_service_template",
        "state_db_session_table": "fa_face_service_session",
        "state_db_proof_table": "fa_face_service_proof",
        "state_db_connect_timeout_seconds": 3,
        "state_db_read_timeout_seconds": 5,
        "state_db_write_timeout_seconds": 5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self.description = None
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def execute(self, query, params=()):
        self.connection.queries.append((query, params))
        self.description = ("column",) if query.lstrip().upper().startswith("SELECT") else None
        self.rowcount = 1

    def fetchall(self):
        return [{"template_id": "template-1"}] if self.description else []

    def close(self):
        return None


class _FakeConnection:
    def __init__(self):
        self.queries = []
        self.begins = 0
        self.commits = 0
        self.rollbacks = 0
        self.closed = 0

    def cursor(self):
        return _FakeCursor(self)

    def begin(self):
        self.begins += 1

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed += 1


class MySQLStoreTests(unittest.TestCase):
    def test_query_translation_uses_configured_tables_and_parameters(self):
        translated = _translate_query(
            "SELECT * FROM face_templates WHERE template_id = ?",
            {
                "face_templates": "fa_face_service_template",
                "verification_sessions": "fa_face_service_session",
                "verification_proofs": "fa_face_service_proof",
            },
        )

        self.assertEqual(
            translated,
            "SELECT * FROM `fa_face_service_template` WHERE template_id = %s",
        )

    def test_connection_adapter_buffers_rows_and_commits_context(self):
        connection = _FakeConnection()
        adapter = _MySQLConnectionAdapter(
            connection,
            {
                "face_templates": "fa_face_service_template",
                "verification_sessions": "fa_face_service_session",
                "verification_proofs": "fa_face_service_proof",
            },
        )

        with adapter as conn:
            conn.execute("BEGIN IMMEDIATE")
            result = conn.execute(
                "SELECT template_id FROM face_templates WHERE template_id = ?",
                ("template-1",),
            )

        self.assertEqual(result.fetchone(), {"template_id": "template-1"})
        self.assertEqual(connection.begins, 1)
        self.assertEqual(connection.commits, 1)
        self.assertEqual(connection.closed, 1)
        self.assertIn("`fa_face_service_template`", connection.queries[0][0])
        self.assertEqual(connection.queries[0][1], ("template-1",))

    def test_context_rolls_back_on_error(self):
        connection = _FakeConnection()
        adapter = _MySQLConnectionAdapter(connection, {})

        with self.assertRaisesRegex(RuntimeError, "stop"):
            with adapter:
                raise RuntimeError("stop")

        self.assertEqual(connection.rollbacks, 1)
        self.assertEqual(connection.commits, 0)

    def test_invalid_dsn_and_table_names_fail_before_connecting(self):
        with self.assertRaises(MySQLStoreError):
            MySQLStore(_settings(state_db_template_table="face;DROP TABLE doctor"))
        with self.assertRaises(MySQLStoreError):
            MySQLStore(_settings(state_db_dsn="sqlite:///face.db"))

    def test_integrity_error_is_normalized_for_store_idempotency(self):
        class IntegrityError(Exception):
            pass

        class FailingCursor(_FakeCursor):
            def execute(self, query, params=()):
                raise IntegrityError("duplicate")

        connection = _FakeConnection()
        connection.cursor = lambda: FailingCursor(connection)
        adapter = _MySQLConnectionAdapter(connection, {})

        with self.assertRaises(sqlite3.IntegrityError):
            adapter.execute("INSERT INTO t (id) VALUES (?)", (1,))

    def test_constructor_uses_validated_table_mapping(self):
        with patch.object(MySQLStore, "_init_db", return_value=None):
            store = MySQLStore(_settings())

        self.assertEqual(store.backend_name, "mysql")
        self.assertEqual(
            store.table_names,
            {
                "face_templates": "fa_face_service_template",
                "verification_sessions": "fa_face_service_session",
                "verification_proofs": "fa_face_service_proof",
            },
        )


if __name__ == "__main__":
    unittest.main()
