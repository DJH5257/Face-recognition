import base64
import unittest
from types import SimpleNamespace

import cv2
import numpy as np
from fastapi import HTTPException
from pydantic import ValidationError

from backend.app.config import Settings
from backend.app.face_engine import decode_base64_image
from backend.app import main
from backend.app.liveness import ACTION_LABELS, random_actions
from backend.app.main import (
    _analyze_live_faces,
    _match_live_face,
    _select_deep_check_frames,
    _validate_frame_sequence,
    _validate_frame_uniqueness,
)
from backend.app.schemas import ChallengeRequest, CompareRequest, FramePayload, VerifyLivenessRequest
from backend.app.schemas import (
    ProofFinalizeRequest,
    ProofIntrospectRequest,
    TemplateCreateRequest,
    VerificationSessionCreateRequest,
    VerificationSessionVerifyRequest,
)


def make_frame(action, index, timestamp):
    return SimpleNamespace(
        action=action,
        index=index,
        timestamp=timestamp,
        image="",
    )


class FakeFace:
    def __init__(self, offset):
        self.embedding = np.full((4,), offset, dtype=np.float32)
        self.bbox = [offset, offset, offset + 10.0, offset + 10.0]


class MultiFaceEngine:
    def extract_faces(self, _frame):
        return [FakeFace(1.0), FakeFace(20.0)]


class FakeAntiSpoofingModel:
    def predict(self, *_args, **_kwargs):
        return SimpleNamespace(
            live_score=0.92,
            passed=True,
            frame_scores=[0.91, 0.93],
            class_probs=[0.03, 0.92, 0.05],
            passed_frame_ratio=1.0,
        )


class VerificationGuardTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()
        with main.store._lock:
            main.store.enrollments.clear()
            main.store.challenges.clear()

    def test_valid_frame_sequence_allows_two_second_capture(self):
        frames = [
            make_frame("blink", index, index * 200)
            for index in range(11)
        ]

        _validate_frame_sequence(frames, ["blink"], self.settings)

    def test_frame_sequence_rejects_short_capture(self):
        frames = [
            make_frame("blink", index, index * 80)
            for index in range(10)
        ]

        with self.assertRaises(HTTPException) as ctx:
            _validate_frame_sequence(frames, ["blink"], self.settings)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("采集时间过短", ctx.exception.detail)

    def test_frame_sequence_rejects_non_contiguous_indexes(self):
        frames = [
            make_frame("blink", 0, 0),
            make_frame("blink", 1, 200),
            make_frame("blink", 3, 400),
            make_frame("blink", 4, 600),
            make_frame("blink", 5, 800),
            make_frame("blink", 6, 1000),
            make_frame("blink", 7, 1200),
            make_frame("blink", 8, 1400),
            make_frame("blink", 9, 1600),
        ]

        with self.assertRaises(HTTPException) as ctx:
            _validate_frame_sequence(frames, ["blink"], self.settings)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("帧序号不连续", ctx.exception.detail)

    def test_frame_uniqueness_rejects_repeated_frames(self):
        frame = np.full((32, 32, 3), 128, dtype=np.uint8)
        result = _validate_frame_uniqueness([frame.copy() for _ in range(10)], self.settings)

        self.assertFalse(result["passed"])
        self.assertEqual(result["action"], "frame_uniqueness")

    def test_frame_uniqueness_accepts_distinct_frames(self):
        frames = []
        for index in range(10):
            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            frame[:, :, :] = index * 20
            frames.append(frame)

        result = _validate_frame_uniqueness(frames, self.settings)

        self.assertTrue(result["passed"])

    def test_decode_base64_image_rejects_invalid_payload(self):
        with self.assertRaises(ValueError):
            decode_base64_image("data:image/jpeg;base64,not valid base64!!", self.settings.max_upload_bytes)

    def test_decode_base64_image_accepts_jpeg_data_url(self):
        image = np.zeros((12, 12, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        payload = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")

        decoded = decode_base64_image(payload, self.settings.max_upload_bytes)

        self.assertEqual(decoded.shape[:2], (12, 12))

    def test_compare_request_rejects_client_threshold(self):
        with self.assertRaises(ValidationError):
            CompareRequest(
                enrollment_id="enrollment",
                challenge_id="challenge",
                threshold=0.01,
            )

    def test_request_models_reject_extra_fields(self):
        with self.assertRaises(ValidationError):
            ChallengeRequest(enrollment_id="enrollment", threshold=0.01)

        with self.assertRaises(ValidationError):
            FramePayload(
                action="blink",
                index=0,
                timestamp=0,
                image="data:image/jpeg;base64,AAAA",
                extra="not-allowed",
            )

        frame = FramePayload(
            action="blink",
            index=0,
            timestamp=0,
            image="data:image/jpeg;base64,AAAA",
        )
        with self.assertRaises(ValidationError):
            VerifyLivenessRequest(
                challenge_id="challenge",
                enrollment_id="enrollment",
                frames=[frame],
                threshold=0.01,
            )

        with self.assertRaises(ValidationError):
            TemplateCreateRequest(
                subject_type="doctor",
                subject_id="83",
                image="data:image/jpeg;base64,AAAA",
                threshold=0.01,
            )

        with self.assertRaises(ValidationError):
            VerificationSessionCreateRequest(
                subject_type="doctor",
                subject_id="83",
                scene="login",
                business_event_id="event",
                template_id="client-must-not-choose",
            )

        with self.assertRaises(ValidationError):
            VerificationSessionVerifyRequest(frames=[frame], subject_id="83")

        with self.assertRaises(ValidationError):
            ProofIntrospectRequest(proof_id="proof", scene="login")

        with self.assertRaises(ValidationError):
            ProofFinalizeRequest(business_event_id="event", threshold=0.01)

    def test_random_actions_use_default_one_to_three_unique_actions(self):
        action_names = set(ACTION_LABELS)
        for _ in range(100):
            actions = random_actions()

            self.assertGreaterEqual(len(actions), 1)
            self.assertLessEqual(len(actions), 3)
            self.assertEqual(len(actions), len(set(actions)))
            self.assertTrue(set(actions).issubset(action_names))

            configured_actions = random_actions(
                self.settings.liveness_action_min_count,
                self.settings.liveness_action_max_count,
            )
            self.assertGreaterEqual(len(configured_actions), 1)
            self.assertLessEqual(len(configured_actions), 3)

    def test_live_face_analysis_rejects_multiple_faces(self):
        original_engine = main.face_engine
        main.face_engine = MultiFaceEngine()
        try:
            frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(8)]

            result = _analyze_live_faces(frames, self.settings)
        finally:
            main.face_engine = original_engine

        self.assertFalse(result.passed)
        self.assertIsNone(result.embedding)
        self.assertIn("多张人脸", result.detail)

    def test_deep_check_frames_are_distributed_by_action(self):
        settings = Settings(face_match_embedding_sample=6, anti_spoofing_sample_frames=6)
        actions = ["blink", "mouth_open", "shake_head", "nod_head"]
        frames_by_action = {
            action: [
                np.full((8, 8, 3), action_index * 50 + frame_index, dtype=np.uint8)
                for frame_index in range(11)
            ]
            for action_index, action in enumerate(actions)
        }

        selected = _select_deep_check_frames(frames_by_action, actions, settings)

        self.assertEqual(len(selected), 8)
        selected_values = [int(frame[0, 0, 0]) for frame in selected]
        for action_index in range(len(actions)):
            self.assertTrue(any(action_index * 50 <= value < action_index * 50 + 11 for value in selected_values))

    def test_face_match_requires_threshold_ratio_and_min_similarity(self):
        enrollment = np.array([1.0, 0.0], dtype=np.float32)
        good_live = [
            np.array([1.0, 0.0], dtype=np.float32),
            _normalized([0.95, 0.05]),
            _normalized([0.9, 0.1]),
        ]
        weak_tail_live = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]
        settings = Settings(
            face_match_threshold=0.8,
            face_match_min_threshold=0.6,
            face_match_min_pass_ratio=0.65,
        )

        self.assertTrue(_match_live_face(enrollment, good_live, settings)["passed"])
        self.assertFalse(_match_live_face(enrollment, weak_tail_live, settings)["passed"])

    def test_compare_requires_verified_liveness(self):
        enrollment = main.store.create_enrollment(np.array([1.0, 0.0], dtype=np.float32), [0, 0, 1, 1])
        challenge = main.store.create_challenge(["blink"], ttl_seconds=180, enrollment_id=enrollment.id)

        with self.assertRaises(HTTPException) as ctx:
            main.compare_faces(
                CompareRequest(
                    enrollment_id=enrollment.id,
                    challenge_id=challenge.id,
                )
            )

        self.assertEqual(ctx.exception.status_code, 403)
        self.assertIn("活体检测未通过", ctx.exception.detail)

    def test_invalid_liveness_attempt_consumes_challenge(self):
        enrollment = main.store.create_enrollment(np.array([1.0, 0.0], dtype=np.float32), [0, 0, 1, 1])
        challenge = main.store.create_challenge(["blink"], ttl_seconds=180, enrollment_id=enrollment.id)
        payload = VerifyLivenessRequest(
            challenge_id=challenge.id,
            enrollment_id=enrollment.id,
            frames=[
                FramePayload(
                    action="blink",
                    index=0,
                    timestamp=0,
                    image="data:image/jpeg;base64,AAAA",
                )
            ],
        )

        with self.assertRaises(HTTPException) as first_ctx:
            main.verify_liveness(payload)

        self.assertEqual(first_ctx.exception.status_code, 400)
        self.assertIsNotNone(challenge.verified_at)

        with self.assertRaises(HTTPException) as second_ctx:
            main.verify_liveness(payload)

        self.assertEqual(second_ctx.exception.status_code, 409)
        self.assertIn("活体挑战已使用", second_ctx.exception.detail)

    def test_compare_rejects_cross_enrollment_challenge(self):
        enrollment = main.store.create_enrollment(np.array([1.0, 0.0], dtype=np.float32), [0, 0, 1, 1])
        other = main.store.create_enrollment(np.array([0.0, 1.0], dtype=np.float32), [0, 0, 1, 1])
        challenge = main.store.create_challenge(["blink"], ttl_seconds=180, enrollment_id=enrollment.id)
        challenge.liveness_passed = True
        challenge.live_embedding = enrollment.embedding

        with self.assertRaises(HTTPException) as ctx:
            main.compare_faces(
                CompareRequest(
                    enrollment_id=other.id,
                    challenge_id=challenge.id,
                )
            )

        self.assertEqual(ctx.exception.status_code, 403)
        self.assertIn("活体挑战与基准人脸不一致", ctx.exception.detail)

    def test_liveness_success_then_compare_passes_for_same_person(self):
        enrollment_embedding = np.array([1.0, 0.0], dtype=np.float32)
        live_embeddings = [
            enrollment_embedding,
            _normalized([0.95, 0.05]),
            _normalized([0.9, 0.1]),
        ]
        enrollment = main.store.create_enrollment(enrollment_embedding, [0, 0, 10, 10])
        challenge = main.store.create_challenge(["blink"], ttl_seconds=180, enrollment_id=enrollment.id)

        with patched_verification_pipeline(live_embeddings, passed_actions=True):
            live = main.verify_liveness(make_verify_request(challenge.id, enrollment.id))
            compared = main.compare_faces(
                CompareRequest(
                    enrollment_id=enrollment.id,
                    challenge_id=challenge.id,
                )
            )

        self.assertTrue(live.liveness_passed)
        self.assertTrue(live.face_matched)
        self.assertTrue(compared.matched)
        self.assertEqual(compared.message, "验证成功，是本人")

    def test_liveness_rejects_different_person_even_when_actions_and_pad_pass(self):
        enrollment = main.store.create_enrollment(np.array([1.0, 0.0], dtype=np.float32), [0, 0, 10, 10])
        challenge = main.store.create_challenge(["blink"], ttl_seconds=180, enrollment_id=enrollment.id)
        other_person_embeddings = [
            np.array([0.0, 1.0], dtype=np.float32),
            _normalized([0.1, 0.9]),
            _normalized([0.2, 0.8]),
        ]

        with patched_verification_pipeline(other_person_embeddings, passed_actions=True):
            live = main.verify_liveness(make_verify_request(challenge.id, enrollment.id))

        self.assertFalse(live.liveness_passed)
        self.assertFalse(live.face_matched)
        self.assertEqual(live.message, "人脸比对失败，不是同一个人")

        with self.assertRaises(HTTPException) as ctx:
            main.compare_faces(
                CompareRequest(
                    enrollment_id=enrollment.id,
                    challenge_id=challenge.id,
                )
            )

        self.assertEqual(ctx.exception.status_code, 403)
        self.assertIn("活体检测未通过", ctx.exception.detail)


def _normalized(values):
    arr = np.asarray(values, dtype=np.float32)
    return arr / max(float(np.linalg.norm(arr)), 1e-12)


def make_verify_request(challenge_id, enrollment_id):
    return VerifyLivenessRequest(
        challenge_id=challenge_id,
        enrollment_id=enrollment_id,
        frames=[
            FramePayload(
                action="blink",
                index=index,
                timestamp=index * 200,
                image="data:image/jpeg;base64,AAAA",
            )
            for index in range(11)
        ],
    )


class patched_verification_pipeline:
    def __init__(self, live_embeddings, passed_actions):
        self.live_embeddings = live_embeddings
        self.passed_actions = passed_actions
        self.original_decode = main.decode_base64_image
        self.original_face_check = main._analyze_live_faces
        self.original_actions = main.verify_liveness_actions
        self.original_spoofing = main.anti_spoofing_model

    def __enter__(self):
        self.decoded_frame_index = 0
        main.decode_base64_image = self._decode_frame
        main._analyze_live_faces = self._face_check
        main.verify_liveness_actions = self._actions
        main.anti_spoofing_model = FakeAntiSpoofingModel()
        return self

    def __exit__(self, *_args):
        main.decode_base64_image = self.original_decode
        main._analyze_live_faces = self.original_face_check
        main.verify_liveness_actions = self.original_actions
        main.anti_spoofing_model = self.original_spoofing

    def _face_check(self, frames, _settings, **_kwargs):
        embedding = main._mean_embedding(self.live_embeddings)
        return main.LiveFaceCheck(
            passed=True,
            embedding=embedding,
            embeddings=self.live_embeddings,
            bbox=[0, 0, 10, 10],
            bboxes=[[0, 0, 10, 10] for _ in frames],
            face_frame_ratio=1.0,
            consistency_score=1.0,
            consistency_pass_ratio=1.0,
            detail="mock live face",
        )

    def _actions(self, *_args, **_kwargs):
        return [
            {
                "action": "blink",
                "label": "请眨眼",
                "passed": self.passed_actions,
                "score": 1.0 if self.passed_actions else 0.0,
                "detail": "mock action",
            }
        ]

    def _decode_frame(self, *_args, **_kwargs):
        value = (self.decoded_frame_index * 20) % 255
        self.decoded_frame_index += 1
        return np.full((32, 32, 3), value, dtype=np.uint8)


if __name__ == "__main__":
    unittest.main()
