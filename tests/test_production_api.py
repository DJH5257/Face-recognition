import base64
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from fastapi import BackgroundTasks, HTTPException

from backend.app import main
from backend.app.schemas import (
    FramePayload,
    ProofFinalizeRequest,
    ProofIntrospectRequest,
    TemplateCreateRequest,
    VerificationSessionCreateRequest,
    VerificationSessionVerifyRequest,
)
from backend.app.stores import MemoryStore


class FakeFace:
    def __init__(self, embedding):
        self.embedding = np.asarray(embedding, dtype=np.float32)
        self.bbox = [0.0, 0.0, 10.0, 10.0]


class FakeFaceEngine:
    def __init__(self, faces):
        self._faces = faces

    def extract_faces(self, _frame):
        return self._faces


def make_image_payload():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("could not encode jpeg")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def make_verify_request():
    return VerificationSessionVerifyRequest(
        frames=[
            FramePayload(
                action="blink",
                index=0,
                timestamp=0,
                image="data:image/jpeg;base64,AAAA",
            )
        ]
    )


class ProductionApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.original_store = main.store
        self.original_key = main.settings.internal_api_key
        self.original_insightface_root = main.settings.insightface_root
        self.original_face_engine = main.face_engine
        self.original_evaluate = main._evaluate_liveness_frames
        main.store = MemoryStore(Path(self.tmp.name) / "face.sqlite3")
        main.settings.internal_api_key = "test-key"
        insightface_root = Path(self.tmp.name) / "insightface-home" / ".insightface"
        model_dir = insightface_root / "models" / main.settings.insightface_model
        model_dir.mkdir(parents=True, exist_ok=True)
        for name in [
            "det_10g.onnx",
            "w600k_r50.onnx",
            "2d106det.onnx",
            "1k3d68.onnx",
            "genderage.onnx",
        ]:
            (model_dir / name).touch()
        main.settings.insightface_root = str(insightface_root)
        main.face_engine = FakeFaceEngine([FakeFace([1.0, 0.0])])

    def tearDown(self):
        main.store = self.original_store
        main.settings.internal_api_key = self.original_key
        main.settings.insightface_root = self.original_insightface_root
        main.face_engine = self.original_face_engine
        main._evaluate_liveness_frames = self.original_evaluate
        self.tmp.cleanup()

    def test_internal_routes_require_api_key(self):
        with self.assertRaises(HTTPException) as ctx:
            main._require_internal_api_key(x_face_api_key="wrong-key")

        self.assertEqual(ctx.exception.status_code, 401)

    def test_ready_endpoint_reports_ok(self):
        ready = main.ready()

        self.assertTrue(ready["ok"])
        self.assertTrue(ready["database_parent_exists"])
        self.assertEqual(ready["missing_models"], [])

    def test_template_session_verify_proof_and_finalize_flow(self):
        template_response = main.create_template(
            TemplateCreateRequest(
                subject_type="doctor",
                subject_id="83",
                image=make_image_payload(),
                source_type="avatar",
            )
        )
        self.assertEqual(template_response.subject_type, "doctor")
        self.assertEqual(template_response.subject_id, "83")

        session_payload = VerificationSessionCreateRequest(
            request_id="req-1",
            subject_type="doctor",
            subject_id="83",
            admin_id="2112",
            scene="doctor_audit",
            business_event_id="biz-1",
            record_id="record-9",
            action="audit",
        )
        first_session = main.create_verification_session(BackgroundTasks(), session_payload)
        second_session = main.create_verification_session(BackgroundTasks(), session_payload)

        self.assertEqual(first_session.session_id, second_session.session_id)
        self.assertEqual(first_session.upload_token, second_session.upload_token)
        self.assertGreaterEqual(len(first_session.actions), 1)
        self.assertLessEqual(len(first_session.actions), 3)

        with self.assertRaises(HTTPException) as wrong_token:
            main.verify_verification_session(
                first_session.session_id,
                make_verify_request(),
                authorization="Bearer wrong-token",
            )
        self.assertEqual(wrong_token.exception.status_code, 401)

        def fake_pass(*_args, **_kwargs):
            return main.VerificationEvaluation(
                passed=True,
                result_code="PASS",
                anti_spoofing_passed=True,
                anti_spoofing_score=0.91,
                face_matched=True,
                similarity=0.88,
                threshold=main.settings.face_match_threshold,
                action_results=[
                    {
                        "action": "face_match",
                        "label": "人脸一致性",
                        "passed": True,
                        "score": 0.88,
                        "detail": "mock",
                    }
                ],
                live_embedding=np.array([1.0, 0.0], dtype=np.float32),
                message="活体检测通过，且为本人",
            )

        main._evaluate_liveness_frames = fake_pass
        verify_response = main.verify_verification_session(
            first_session.session_id,
            make_verify_request(),
            authorization=f"Bearer {first_session.upload_token}",
        )
        self.assertTrue(verify_response.passed)
        self.assertEqual(verify_response.result_code, "PASS")
        self.assertIsNotNone(verify_response.proof_id)

        with self.assertRaises(HTTPException) as replay:
            main.verify_verification_session(
                first_session.session_id,
                make_verify_request(),
                authorization=f"Bearer {first_session.upload_token}",
            )
        self.assertEqual(replay.exception.status_code, 409)

        introspected = main.introspect_proof(
            ProofIntrospectRequest(proof_id=verify_response.proof_id)
        )
        self.assertTrue(introspected.valid)
        self.assertEqual(introspected.subject_type, "doctor")
        self.assertEqual(introspected.subject_id, "83")
        self.assertEqual(introspected.scene, "doctor_audit")
        self.assertEqual(introspected.business_event_id, "biz-1")
        self.assertEqual(introspected.record_id, "record-9")

        with self.assertRaises(HTTPException) as wrong_event:
            main.finalize_proof(
                verify_response.proof_id,
                ProofFinalizeRequest(business_event_id="other-event"),
            )
        self.assertEqual(wrong_event.exception.status_code, 409)

        finalized = main.finalize_proof(
            verify_response.proof_id,
            ProofFinalizeRequest(business_event_id="biz-1"),
        )
        self.assertTrue(finalized.finalized)

        after_finalize = main.introspect_proof(
            ProofIntrospectRequest(proof_id=verify_response.proof_id)
        )
        self.assertFalse(after_finalize.valid)
        self.assertEqual(after_finalize.status, "finalized")

    def test_failed_verification_does_not_issue_proof(self):
        main.create_template(
            TemplateCreateRequest(
                subject_type="doctor",
                subject_id="84",
                image=make_image_payload(),
            )
        )
        session = main.create_verification_session(
            BackgroundTasks(),
            VerificationSessionCreateRequest(
                subject_type="doctor",
                subject_id="84",
                scene="login",
                business_event_id="biz-2",
            ),
        )

        def fake_fail(*_args, **_kwargs):
            return main.VerificationEvaluation(
                passed=False,
                result_code="FAIL_FACE_MISMATCH",
                anti_spoofing_passed=True,
                anti_spoofing_score=0.9,
                face_matched=False,
                similarity=0.12,
                threshold=main.settings.face_match_threshold,
                action_results=[
                    {
                        "action": "face_match",
                        "label": "人脸一致性",
                        "passed": False,
                        "score": 0.12,
                        "detail": "mock mismatch",
                    }
                ],
                live_embedding=None,
                message="人脸比对失败，不是同一个人",
            )

        main._evaluate_liveness_frames = fake_fail
        response = main.verify_verification_session(
            session.session_id,
            make_verify_request(),
            authorization=f"Bearer {session.upload_token}",
        )

        self.assertFalse(response.passed)
        self.assertEqual(response.result_code, "FAIL_FACE_MISMATCH")
        self.assertIsNone(response.proof_id)


if __name__ == "__main__":
    unittest.main()
