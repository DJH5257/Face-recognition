import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend.app.stores import MemoryStore


class ProductionStorageTests(unittest.TestCase):
    def test_template_versions_persist_and_only_latest_is_active(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "face.sqlite3"
            store = MemoryStore(db_path)

            first = store.create_template(
                subject_type="doctor",
                subject_id="83",
                embedding=np.array([1.0, 0.0], dtype=np.float64),
                bbox=[0, 0, 10, 10],
                source_type="avatar",
                source_image_hash="hash-a",
            )
            second = store.create_template(
                subject_type="doctor",
                subject_id="83",
                embedding=np.array([0.0, 1.0], dtype=np.float64),
                bbox=[1, 1, 11, 11],
                source_type="manual_enroll",
                source_image_hash="hash-b",
            )

            reopened = MemoryStore(db_path)
            active = reopened.get_active_template("doctor", "83")
            loaded_first = reopened.get_template(first.template_id)

            self.assertEqual(second.template_id, active.template_id)
            self.assertEqual(active.template_version, 2)
            self.assertEqual(active.status, "active")
            self.assertEqual(loaded_first.status, "revoked")
            self.assertEqual(active.embedding.dtype, np.float32)
            np.testing.assert_allclose(active.embedding, np.array([0.0, 1.0], dtype=np.float32))

    def test_session_is_request_id_idempotent_and_token_bound(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = MemoryStore(Path(tmp_dir) / "face.sqlite3")
            template = store.create_template(
                subject_type="doctor",
                subject_id="83",
                embedding=np.array([1.0, 0.0], dtype=np.float32),
                bbox=[0, 0, 1, 1],
                source_type="avatar",
                source_image_hash="hash",
            )

            first = store.create_verification_session(
                request_id="event-1",
                subject_type="doctor",
                subject_id="83",
                admin_id="2112",
                scene="login",
                business_event_id="biz-1",
                record_id=None,
                action="login",
                expected_actions=["blink"],
                ttl_seconds=180,
            )
            second = store.create_verification_session(
                request_id="event-1",
                subject_type="doctor",
                subject_id="83",
                admin_id="2112",
                scene="login",
                business_event_id="biz-1",
                record_id=None,
                action="login",
                expected_actions=["mouth_open"],
                ttl_seconds=180,
            )

            self.assertEqual(first.session_id, second.session_id)
            self.assertEqual(first.upload_token, second.upload_token)
            self.assertEqual(first.template_id, template.template_id)
            self.assertTrue(store.verify_upload_token(first.session_id, first.upload_token))
            self.assertFalse(store.verify_upload_token(first.session_id, "wrong-token"))
            self.assertTrue(store.start_session_verification(first.session_id))
            self.assertFalse(store.start_session_verification(first.session_id))

    def test_session_requires_active_template(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = MemoryStore(Path(tmp_dir) / "face.sqlite3")

            session = store.create_verification_session(
                request_id="event-1",
                subject_type="doctor",
                subject_id="missing",
                admin_id=None,
                scene="login",
                business_event_id="biz-1",
                record_id=None,
                action="login",
                expected_actions=["blink"],
                ttl_seconds=180,
            )

            self.assertIsNone(session)

    def test_proof_lifecycle_persists_and_finalize_is_bound_to_business_event(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "face.sqlite3"
            store = MemoryStore(db_path)
            store.create_template(
                subject_type="doctor",
                subject_id="83",
                embedding=np.array([1.0, 0.0], dtype=np.float32),
                bbox=[0, 0, 1, 1],
                source_type="avatar",
                source_image_hash="hash",
            )
            session = store.create_verification_session(
                request_id="event-1",
                subject_type="doctor",
                subject_id="83",
                admin_id="2112",
                scene="doctor_audit",
                business_event_id="biz-1",
                record_id="record-9",
                action="audit",
                expected_actions=["blink"],
                ttl_seconds=180,
            )

            proof = store.create_proof_for_session(session.session_id, "PASS", ttl_seconds=180)
            reopened = MemoryStore(db_path)
            loaded = reopened.get_proof(proof.proof_id)

            self.assertTrue(loaded.valid)
            self.assertEqual(loaded.business_event_id, "biz-1")
            self.assertEqual(loaded.record_id, "record-9")
            with self.assertRaises(ValueError):
                reopened.finalize_proof(proof.proof_id, "other-event")

            finalized = reopened.finalize_proof(proof.proof_id, "biz-1")
            repeated = reopened.finalize_proof(proof.proof_id, "biz-1")

            self.assertEqual(finalized.status, "finalized")
            self.assertFalse(finalized.valid)
            self.assertEqual(repeated.status, "finalized")

    def test_expired_proof_is_not_valid(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = MemoryStore(Path(tmp_dir) / "face.sqlite3")
            store.create_template(
                subject_type="doctor",
                subject_id="83",
                embedding=np.array([1.0, 0.0], dtype=np.float32),
                bbox=[0, 0, 1, 1],
                source_type="avatar",
                source_image_hash="hash",
            )
            session = store.create_verification_session(
                request_id=None,
                subject_type="doctor",
                subject_id="83",
                admin_id=None,
                scene="login",
                business_event_id="biz-1",
                record_id=None,
                action="login",
                expected_actions=["blink"],
                ttl_seconds=180,
            )

            proof = store.create_proof_for_session(session.session_id, "PASS", ttl_seconds=-1)

            self.assertEqual(store.get_proof(proof.proof_id).status, "expired")
            finalized = store.finalize_proof(proof.proof_id, "biz-1")
            self.assertEqual(finalized.status, "expired")
            self.assertFalse(finalized.valid)


if __name__ == "__main__":
    unittest.main()
