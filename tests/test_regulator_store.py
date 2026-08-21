import sys
import types
import unittest
import json
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
        "doctor_face_log_enabled": False,
        "doctor_face_log_table": "fa_doctor_face_verify_log",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeCursor:
    def __init__(self):
        self.queries = []

    def execute(self, query, params=None):
        self.queries.append((query, params))

    def fetchall(self):
        if self.queries and "fa_doctor_face_verify_log" in self.queries[-1][0]:
            return [{"Field": field} for field in (
                "id",
                "doctor_id",
                "action",
                "detail",
                "record_id",
                "verify_time",
                "ip",
                "user_agent",
                "create_time",
            )]
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

    def test_enabled_doctor_log_is_associated_with_verification_event(self):
        connection = _FakeConnection()
        pymysql_module = types.ModuleType("pymysql")
        pymysql_module.connect = lambda **_kwargs: connection
        cursors_module = types.ModuleType("pymysql.cursors")
        cursors_module.DictCursor = object

        store = RegulatorStore(
            _settings(
                regulator_db_enabled=True,
                regulator_db_dsn="mysql+pymysql://user:pass@localhost/chbzg_test",
                doctor_face_log_enabled=True,
            )
        )
        session = SimpleNamespace(
            business_event_id="event-doctor-1",
            record_id="record-9",
            subject_type="doctor",
            subject_id="219",
            scene="login",
            action="login",
        )

        with patch.dict(sys.modules, {"pymysql": pymysql_module, "pymysql.cursors": cursors_module}):
            store.record_result(
                session,
                "success",
                "PASS",
                "活体检测通过，且为本人",
                client_ip="203.0.113.8",
                user_agent="Face Test Browser",
            )

        self.assertEqual(connection.commits, 1)
        self.assertEqual(len(connection.cursor_instance.queries), 4)
        doctor_schema_query, _ = connection.cursor_instance.queries[2]
        doctor_write_query, doctor_params = connection.cursor_instance.queries[3]
        self.assertIn("SHOW COLUMNS FROM `fa_doctor_face_verify_log`", doctor_schema_query)
        self.assertIn("INSERT INTO `fa_doctor_face_verify_log`", doctor_write_query)
        self.assertIn("NOW()", doctor_write_query)
        self.assertIn("UNIX_TIMESTAMP()", doctor_write_query)
        self.assertEqual(doctor_params[0], "219")
        self.assertEqual(doctor_params[1], "liveness_verify_success-login-登录")
        detail = json.loads(doctor_params[2])
        self.assertEqual(detail["business_event_id"], "event-doctor-1")
        self.assertEqual(detail["status"], "success")
        self.assertEqual(detail["result"], "通过")
        self.assertEqual(doctor_params[3:], ["record-9", "203.0.113.8", "Face Test Browser"])

    def test_failed_doctor_log_does_not_look_successful_to_existing_monitor(self):
        connection = _FakeConnection()
        pymysql_module = types.ModuleType("pymysql")
        pymysql_module.connect = lambda **_kwargs: connection
        cursors_module = types.ModuleType("pymysql.cursors")
        cursors_module.DictCursor = object
        store = RegulatorStore(
            _settings(
                regulator_db_dsn="mysql+pymysql://user:pass@localhost/chbzg_test",
                doctor_face_log_enabled=True,
            )
        )
        session = SimpleNamespace(
            business_event_id="event-doctor-2",
            record_id=None,
            subject_type="doctor",
            subject_id="219",
            scene="doctor_audit",
            action="audit",
        )

        with patch.dict(sys.modules, {"pymysql": pymysql_module, "pymysql.cursors": cursors_module}):
            store.record_result(
                session,
                "failed",
                "FAIL_ACTION",
                "活体检测未通过",
            )

        _, doctor_params = connection.cursor_instance.queries[3]
        self.assertEqual(doctor_params[1], "liveness_verify_failed-audit-审核")
        self.assertNotIn("通过", doctor_params[2])
        detail = json.loads(doctor_params[2])
        self.assertEqual(detail["result"], "失败")
        self.assertEqual(detail["result_msg"], "活体检测失败")

    def test_pharmacist_result_does_not_write_doctor_log(self):
        connection = _FakeConnection()
        pymysql_module = types.ModuleType("pymysql")
        pymysql_module.connect = lambda **_kwargs: connection
        cursors_module = types.ModuleType("pymysql.cursors")
        cursors_module.DictCursor = object
        store = RegulatorStore(
            _settings(
                regulator_db_dsn="mysql+pymysql://user:pass@localhost/chbzg_test",
                doctor_face_log_enabled=True,
            )
        )
        session = SimpleNamespace(
            business_event_id="event-pharmacist-1",
            record_id=None,
            subject_type="pharmacist",
            subject_id="7",
            scene="login",
            action="login",
        )

        with patch.dict(sys.modules, {"pymysql": pymysql_module, "pymysql.cursors": cursors_module}):
            store.record_result(session, "success", "PASS", "通过")

        queries = [query for query, _params in connection.cursor_instance.queries]
        self.assertFalse(any("fa_doctor_face_verify_log" in query for query in queries))

    def test_ready_schema_check_includes_doctor_log_table(self):
        connection = _FakeConnection()
        pymysql_module = types.ModuleType("pymysql")
        pymysql_module.connect = lambda **_kwargs: connection
        cursors_module = types.ModuleType("pymysql.cursors")
        cursors_module.DictCursor = object
        store = RegulatorStore(
            _settings(
                regulator_db_dsn="mysql+pymysql://user:pass@localhost/chbzg_test",
                doctor_face_log_enabled=True,
            )
        )

        with patch.dict(sys.modules, {"pymysql": pymysql_module, "pymysql.cursors": cursors_module}):
            schema = store.check_schema()

        self.assertTrue(schema["doctor_face_log"]["enabled"])
        self.assertEqual(schema["doctor_face_log"]["table"], "fa_doctor_face_verify_log")


if __name__ == "__main__":
    unittest.main()
