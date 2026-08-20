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
from backend.app.liveness import ACTION_LABELS, FrameMetrics, _judge_action, _judge_blink_action, random_actions
from backend.app.main import (
    _analyze_live_faces,
    _fold_abs_pose_angle,
    _match_live_face,
    _validate_face_bbox_quality,
    _select_deep_check_frames,
    _select_pose_quality_frames,
    _validate_pose_quality,
    _validate_frame_sequence,
    _validate_frame_uniqueness,
    _validate_sampled_image_quality,
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

    def test_valid_frame_sequence_allows_high_rate_blink_capture(self):
        frames = [
            make_frame("blink", index, index * 80)
            for index in range(26)
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

    def test_image_quality_rejects_flat_dark_frames(self):
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(6)]

        result = _validate_sampled_image_quality(frames, self.settings, "图像质量")

        self.assertFalse(result["passed"])
        self.assertEqual(result["action"], "image_quality")

    def test_image_quality_accepts_textured_normal_frames(self):
        frames = [_textured_frame(96, 96) for _ in range(6)]

        result = _validate_sampled_image_quality(frames, self.settings, "图像质量")

        self.assertTrue(result["passed"], result["detail"])

    def test_face_bbox_quality_rejects_tiny_face(self):
        result = _validate_face_bbox_quality(
            [1, 1, 5, 5],
            (200, 200, 3),
            self.settings.image_quality_template_min_face_ratio,
            self.settings,
            "基准人脸画面",
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["action"], "face_bbox_quality")

    def test_pose_quality_rejects_frames_without_landmarks(self):
        frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
        original_face_mesh = main.mp.solutions.face_mesh.FaceMesh
        original_extract_metrics = main._extract_metrics

        class DummyFaceMesh:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        main.mp.solutions.face_mesh.FaceMesh = DummyFaceMesh
        main._extract_metrics = lambda *_args, **_kwargs: None

        try:
            result = _validate_pose_quality(frames, self.settings, "姿态质量", 55.0, 55.0)
        finally:
            main.mp.solutions.face_mesh.FaceMesh = original_face_mesh
            main._extract_metrics = original_extract_metrics

        self.assertFalse(result["passed"])
        self.assertEqual(result["action"], "pose_quality")

    def test_pose_angle_fold_handles_rotation_ambiguity(self):
        self.assertAlmostEqual(_fold_abs_pose_angle(179.5), 0.5)
        self.assertAlmostEqual(_fold_abs_pose_angle(-179.0), 1.0)
        self.assertAlmostEqual(_fold_abs_pose_angle(35.0), 35.0)

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

    def test_verify_request_models_allow_configured_frame_ceiling(self):
        frames = [
            FramePayload(
                action="blink",
                index=index,
                timestamp=index * 80,
                image="data:image/jpeg;base64,AAAA",
            )
            for index in range(90)
        ]

        VerifyLivenessRequest(challenge_id="challenge", frames=frames)
        VerificationSessionVerifyRequest(frames=frames)

        too_many = frames + [
            FramePayload(
                action="blink",
                index=90,
                timestamp=7200,
                image="data:image/jpeg;base64,AAAA",
            )
        ]
        with self.assertRaises(ValidationError):
            VerifyLivenessRequest(challenge_id="challenge", frames=too_many)
        with self.assertRaises(ValidationError):
            VerificationSessionVerifyRequest(frames=too_many)

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
        blink_values = [value for value in selected_values if value < 11]
        self.assertTrue(all(value <= 2 or value >= 8 for value in blink_values))

    def test_pose_quality_frames_skip_head_motion_when_stable_actions_exist(self):
        settings = Settings(face_match_embedding_sample=6, anti_spoofing_sample_frames=6)
        actions = ["shake_head", "blink", "nod_head"]
        frames_by_action = {
            action: [
                np.full((8, 8, 3), action_index * 50 + frame_index, dtype=np.uint8)
                for frame_index in range(11)
            ]
            for action_index, action in enumerate(actions)
        }

        selected = _select_pose_quality_frames(frames_by_action, actions, settings)

        self.assertTrue(selected)
        selected_values = [int(frame[0, 0, 0]) for frame in selected]
        self.assertTrue(all(50 <= value < 61 for value in selected_values))

    def test_blink_event_detects_relative_close_and_recovery(self):
        ears = np.array(
            [0.27, 0.27, 0.268, 0.265, 0.24, 0.18, 0.145, 0.18, 0.235, 0.265, 0.27, 0.27],
            dtype=np.float32,
        )

        result = _judge_blink_action("blink", ears, self.settings)

        self.assertTrue(result["passed"], result["detail"])
        self.assertIn("baseline", result["detail"])

    def test_blink_event_rejects_static_eye_series(self):
        ears = np.full((12,), 0.27, dtype=np.float32)

        result = _judge_blink_action("blink", ears, self.settings)

        self.assertFalse(result["passed"])

    def test_blink_event_accepts_five_percent_relaxed_drop(self):
        ears = np.array(
            [0.24, 0.24, 0.24, 0.24, 0.201, 0.201, 0.201, 0.24, 0.24, 0.24, 0.24, 0.24],
            dtype=np.float32,
        )

        result = _judge_blink_action("blink", ears, self.settings)

        self.assertTrue(result["passed"], result["detail"])

    def test_blink_event_accepts_low_baseline_with_clear_close_and_recovery(self):
        ears = np.array(
            [0.145, 0.145, 0.145, 0.14, 0.095, 0.055, 0.085, 0.135, 0.19, 0.19, 0.19],
            dtype=np.float32,
        )

        result = _judge_blink_action("blink", ears, self.settings)

        self.assertTrue(result["passed"], result["detail"])
        self.assertIn("baseline_min", result["detail"])

    def test_blink_event_accepts_clear_close_at_capture_end(self):
        ears = np.array(
            [0.20, 0.20, 0.198, 0.195, 0.19, 0.18, 0.14, 0.095, 0.055, 0.05, 0.05],
            dtype=np.float32,
        )

        result = _judge_blink_action("blink", ears, self.settings)

        self.assertTrue(result["passed"], result["detail"])

    def test_smile_accepts_small_width_change_with_mouth_motion(self):
        metrics = [
            FrameMetrics(ear=0.25, mar=0.20, mouth_width_ratio=0.410, yaw=0.0, pitch=0.0),
            FrameMetrics(ear=0.25, mar=0.21, mouth_width_ratio=0.416, yaw=0.0, pitch=0.0),
            FrameMetrics(ear=0.25, mar=0.22, mouth_width_ratio=0.422, yaw=0.0, pitch=0.0),
            FrameMetrics(ear=0.25, mar=0.225, mouth_width_ratio=0.436, yaw=0.0, pitch=0.0),
            FrameMetrics(ear=0.25, mar=0.223, mouth_width_ratio=0.438, yaw=0.0, pitch=0.0),
            FrameMetrics(ear=0.25, mar=0.219, mouth_width_ratio=0.437, yaw=0.0, pitch=0.0),
        ]

        result = _judge_action("smile", metrics, self.settings)

        self.assertTrue(result["passed"], result["detail"])
        self.assertIn("MAR delta", result["detail"])

    def test_default_nod_threshold_is_relaxed_five_percent(self):
        self.assertEqual(self.settings.nod_pitch_range_threshold, 8.0)

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


def _textured_frame(width, height):
    y, x = np.indices((height, width))
    base = 120 + ((x * 13 + y * 17) % 80)
    texture = (((x // 4 + y // 4) % 2) * 45)
    gray = np.clip(base + texture - 20, 45, 225).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


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
        self.original_image_quality = main._validate_sampled_image_quality
        self.original_bbox_quality = main._validate_live_face_bbox_quality
        self.original_pose_quality = main._validate_pose_quality

    def __enter__(self):
        self.decoded_frame_index = 0
        main.decode_base64_image = self._decode_frame
        main._analyze_live_faces = self._face_check
        main.verify_liveness_actions = self._actions
        main.anti_spoofing_model = FakeAntiSpoofingModel()
        main._validate_sampled_image_quality = self._quality
        main._validate_live_face_bbox_quality = self._quality
        main._validate_pose_quality = self._quality
        return self

    def __exit__(self, *_args):
        main.decode_base64_image = self.original_decode
        main._analyze_live_faces = self.original_face_check
        main.verify_liveness_actions = self.original_actions
        main.anti_spoofing_model = self.original_spoofing
        main._validate_sampled_image_quality = self.original_image_quality
        main._validate_live_face_bbox_quality = self.original_bbox_quality
        main._validate_pose_quality = self.original_pose_quality

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

    def _quality(self, *_args, **_kwargs):
        return {
            "action": "image_quality",
            "label": "图像质量",
            "passed": True,
            "score": 1.0,
            "detail": "mock quality",
        }


if __name__ == "__main__":
    unittest.main()
