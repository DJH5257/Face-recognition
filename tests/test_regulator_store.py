import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.app.regulator_store import RegulatorStore, RegulatorStoreError


def _settings(**overrides):
    values = {
        "regulator_db_enabled": False,
        "regulator_db_dsn": "",
        "regulator_db_table": "fa_face_verify_regulator_status",
        "regulator_db_connect_timeout_seconds": 3,
        "regulator_db_read_timeout_seconds": 5,
        "regulator_db_write_timeout_seconds": 5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeCursor:
    def __init__(self):
        self.queries = []

    def execute(self, query, params=None):
        self.queries.append((query, params))

    def fetchall(self):
        return [{"Field": field} for field in (
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
        )]

    def close(self):
        return None


class _FakeConnection:
    def __init__(self):
        self.cursor_instance = _FakeCursor()
        self.commits = 0

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.commits += 1

    def close(self):
        return None


class RegulatorStoreTests(unittest.TestCase):
    def test_disabled_store_does_not_connect(self):
        store = RegulatorStore(_settings())
        store._connect = lambda: self.fail("disabled store attempted a database connection")
        store.record_result(
            SimpleNamespace(
                business_event_id="event-1",
                record_id=None,
                subject_type="doctor",
                subject_id="83",
                scene="login",
                action="login",
            ),
            "success",
            "PASS",
            "ok",
        )

    def test_enabled_store_requires_dsn(self):
        with self.assertRaises(RegulatorStoreError):
            RegulatorStore(_settings(regulator_db_enabled=True))

    def test_table_name_and_status_are_validated(self):
        with self.assertRaises(RegulatorStoreError):
            RegulatorStore(_settings(regulator_db_table="fa_face_status;DROP TABLE users"))

        store = RegulatorStore(
            _settings(
                regulator_db_enabled=True,
                regulator_db_dsn="mysql+pymysql://user:pass@localhost/chbzg_test",
            )
        )
        with self.assertRaises(RegulatorStoreError):
            store.record_result(SimpleNamespace(), "pending", "PENDING", "pending")

    def test_result_is_written_only_to_regulator_table(self):
        connection = _FakeConnection()
        pymysql_module = types.ModuleType("pymysql")
        pymysql_module.connect = lambda **_kwargs: connection
        cursors_module = types.ModuleType("pymysql.cursors")
        cursors_module.DictCursor = object

        store = RegulatorStore(
            _settings(
                regulator_db_enabled=True,
                regulator_db_dsn="mysql+pymysql://user:pass@localhost/chbzg_test",
            )
        )
        session = SimpleNamespace(
            business_event_id="event-1",
            record_id="record-1",
            subject_type="doctor",
            subject_id="83",
            scene="login",
            action="login",
        )

        with patch.dict(sys.modules, {"pymysql": pymysql_module, "pymysql.cursors": cursors_module}):
            store.record_result(session, "failed", "FAIL_ACTION", "action failed")

        self.assertEqual(connection.commits, 1)
        self.assertEqual(len(connection.cursor_instance.queries), 2)
        schema_query, _ = connection.cursor_instance.queries[0]
        write_query, params = connection.cursor_instance.queries[1]
        self.assertIn("SHOW COLUMNS FROM `fa_face_verify_regulator_status`", schema_query)
        self.assertIn("INSERT INTO `fa_face_verify_regulator_status`", write_query)
        self.assertIn("ON DUPLICATE KEY UPDATE", write_query)
        self.assertNotIn("fa_face_profile", write_query)
        self.assertNotIn("fa_face_verify_event", write_query)
        self.assertNotIn("fa_face_verify_consumption", write_query)
        self.assertEqual(params[:7], ["event-1", "record-1", "doctor", "83", "login", "login", "failed"])


if __name__ == "__main__":
    unittest.main()
