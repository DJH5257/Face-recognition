from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from .config import Settings


ACTION_LABELS: Dict[str, str] = {
    "blink": "请眨眼",
    "mouth_open": "请张嘴",
    "shake_head": "请左右摇头",
    "nod_head": "请上下点头",
    "smile": "请微笑",
}


def random_actions(
    min_count: int = 1,
    max_count: int = 3,
    weights: Optional[Dict[str, float]] = None,
    reduced_actions: Optional[Iterable[str]] = None,
    reduction_factor: float = 0.5,
) -> List[str]:
    max_count = max(1, min(max_count, len(ACTION_LABELS)))
    min_count = max(1, min(min_count, max_count))
    count = min_count + secrets.randbelow(max_count - min_count + 1)
    available = list(ACTION_LABELS.keys())
    selected: List[str] = []
    reduced_set = set(reduced_actions or [])
    normalized_weights = weights or {}
    factor = min(max(float(reduction_factor), 0.05), 1.0)

    while available and len(selected) < count:
        weighted = []
        total = 0.0
        for action in available:
            weight = max(float(normalized_weights.get(action, 1.0)), 0.05)
            if action in reduced_set:
                weight = max(weight * factor, 0.05)
            weighted.append((action, weight))
            total += weight
        pick = secrets.randbelow(max(1, int(total * 1000))) / 1000.0
        cursor = 0.0
        chosen = weighted[-1][0]
        for action, weight in weighted:
            cursor += weight
            if pick < cursor:
                chosen = action
                break
        selected.append(chosen)
        available.remove(chosen)
    return selected


@dataclass
class FrameMetrics:
    ear: float
    mar: float
    mouth_width_ratio: float
    yaw: float
    pitch: float


def verify_liveness_actions(
    frames_by_action: Dict[str, List[np.ndarray]],
    expected_actions: Iterable[str],
    settings: Settings,
) -> List[dict]:
    results: List[dict] = []
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        for action in expected_actions:
            frames = frames_by_action.get(action, [])
            metrics = [_extract_metrics(frame, face_mesh) for frame in frames]
            metrics = [item for item in metrics if item is not None]
            result = _judge_action(action, metrics, settings)
            result["label"] = ACTION_LABELS[action]
            result["face_frames"] = len(metrics)
            result["total_frames"] = len(frames)
            face_frame_ratio = len(metrics) / max(len(frames), 1)
            if result["passed"] and face_frame_ratio < settings.min_face_frame_ratio:
                result["passed"] = False
                result["detail"] += f"，有效人脸帧占比不足 {face_frame_ratio:.0%}"
            results.append(result)
    return results


def _judge_action(action: str, metrics: List[FrameMetrics], settings: Settings) -> dict:
    if len(metrics) < settings.min_frames_per_action:
        return {
            "action": action,
            "passed": False,
            "score": 0.0,
            "detail": f"有效人脸帧不足：{len(metrics)}/{settings.min_frames_per_action}",
        }

    ears = np.array([m.ear for m in metrics], dtype=np.float32)
    mars = np.array([m.mar for m in metrics], dtype=np.float32)
    smiles = np.array([m.mouth_width_ratio for m in metrics], dtype=np.float32)
    yaws = np.array([m.yaw for m in metrics], dtype=np.float32)
    pitches = np.array([m.pitch for m in metrics], dtype=np.float32)

    if action == "blink":
        return _judge_blink_action(action, ears, settings)

    if action == "mouth_open":
        min_mar = float(np.min(mars))
        max_mar = float(np.max(mars))
        mar_delta = max_mar - min_mar
        passed = (
            max_mar > settings.mouth_mar_threshold
            and mar_delta > settings.mouth_mar_delta_threshold
        )
        return {
            "action": action,
            "passed": passed,
            "score": round(max_mar, 4),
            "detail": f"MAR min={min_mar:.3f}, max={max_mar:.3f}, delta={mar_delta:.3f}",
        }

    if action == "shake_head":
        yaw_range = float(np.max(yaws) - np.min(yaws))
        passed = yaw_range > settings.shake_yaw_range_threshold
        return {
            "action": action,
            "passed": passed,
            "score": round(yaw_range, 4),
            "detail": f"yaw range={yaw_range:.1f} deg",
        }

    if action == "nod_head":
        pitch_range = float(np.max(pitches) - np.min(pitches))
        passed = pitch_range > settings.nod_pitch_range_threshold
        return {
            "action": action,
            "passed": passed,
            "score": round(pitch_range, 4),
            "detail": f"pitch range={pitch_range:.1f} deg",
        }

    if action == "smile":
        smile_delta = float(np.max(smiles) - np.min(smiles))
        mar_delta = float(np.max(mars) - np.min(mars))
        assisted_threshold = settings.smile_delta_threshold * 0.93
        passed = smile_delta > settings.smile_delta_threshold or (
            smile_delta >= assisted_threshold and mar_delta >= 0.018
        )
        return {
            "action": action,
            "passed": passed,
            "score": round(smile_delta, 4),
            "detail": (
                f"mouth width ratio delta={smile_delta:.3f}, "
                f"MAR delta={mar_delta:.3f}"
            ),
        }

    return {"action": action, "passed": False, "score": 0.0, "detail": "未知动作"}


def _judge_blink_action(action: str, ears: np.ndarray, settings: Settings) -> dict:
    smoothed = _smooth_series(ears)
    edge_count = max(1, int(np.ceil(len(smoothed) * 0.2)))
    edge_values = np.concatenate([smoothed[:edge_count], smoothed[-edge_count:]])
    baseline = float(np.median(edge_values))
    trough_index = int(np.argmin(smoothed))
    trough = float(smoothed[trough_index])
    drop = baseline - trough
    drop_threshold = max(0.036, baseline * 0.145)
    baseline_threshold = settings.blink_ear_threshold * 0.9
    relaxed_baseline_threshold = max(0.12, settings.blink_ear_threshold * 0.6)

    before_peak = float(np.max(smoothed[:trough_index])) if trough_index > 0 else 0.0
    after_peak = float(np.max(smoothed[trough_index + 1:])) if trough_index + 1 < len(smoothed) else 0.0
    closed_threshold = baseline - drop_threshold * 0.55
    closed_ratio = float(np.mean(smoothed <= closed_threshold))
    recovery_ok = after_peak >= baseline * 0.76
    endpoint_close_ok = (
        trough_index >= len(smoothed) - max(2, int(np.ceil(len(smoothed) * 0.2)))
        and drop >= drop_threshold * 1.2
        and before_peak >= baseline * 0.85
        and closed_ratio <= 0.60
    )
    baseline_ok = baseline >= baseline_threshold or (
        baseline >= relaxed_baseline_threshold
        and drop >= drop_threshold
        and before_peak >= baseline * 0.85
        and (recovery_ok or endpoint_close_ok)
    )
    passed = (
        baseline_ok
        and drop >= drop_threshold
        and before_peak >= baseline * 0.85
        and (recovery_ok or endpoint_close_ok)
        and 0 < trough_index
        and closed_ratio <= 0.70
    )
    strong_blink = (
        drop >= drop_threshold * 1.8
        and trough <= baseline * 0.65
        and before_peak >= baseline * 0.80
        and recovery_ok
        and 0 < trough_index
        and trough_index < len(smoothed) - 1
        and closed_ratio <= 0.58
    )
    passed = passed or strong_blink
    return {
        "action": action,
        "passed": bool(passed),
        "score": round(float(max(drop, 0.0)), 4),
        "detail": (
            f"EAR baseline={baseline:.3f}, trough={trough:.3f}, "
            f"drop={drop:.3f}/{drop_threshold:.3f}, "
            f"baseline_min={relaxed_baseline_threshold:.3f}, "
            f"recovery={after_peak:.3f}, closed_ratio={closed_ratio:.0%}, "
            f"strong_blink={strong_blink}"
        ),
    }


def _smooth_series(values: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        return values.astype(np.float32)
    padded = np.pad(values.astype(np.float32), (1, 1), mode="edge")
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _extract_metrics(bgr: np.ndarray, face_mesh) -> Optional[FrameMetrics]:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None

    h, w = bgr.shape[:2]
    landmarks = result.multi_face_landmarks[0].landmark

    left_ear = _eye_aspect_ratio(landmarks, [33, 160, 158, 133, 153, 144], w, h)
    right_ear = _eye_aspect_ratio(landmarks, [362, 385, 387, 263, 373, 380], w, h)
    ear = (left_ear + right_ear) / 2.0

    mouth_vertical = _distance(landmarks[13], landmarks[14], w, h)
    mouth_horizontal = _distance(landmarks[61], landmarks[291], w, h)
    mar = mouth_vertical / max(mouth_horizontal, 1e-6)

    xs = np.array([pt.x for pt in landmarks], dtype=np.float32)
    face_width = float(np.max(xs) - np.min(xs))
    mouth_width_ratio = float((landmarks[291].x - landmarks[61].x) / max(face_width, 1e-6))

    pitch, yaw = _estimate_head_pose(landmarks, w, h)
    return FrameMetrics(
        ear=float(ear),
        mar=float(mar),
        mouth_width_ratio=mouth_width_ratio,
        yaw=float(yaw),
        pitch=float(pitch),
    )


def _eye_aspect_ratio(landmarks, idx: List[int], width: int, height: int) -> float:
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in idx]
    vertical = _distance(p2, p6, width, height) + _distance(p3, p5, width, height)
    horizontal = 2.0 * _distance(p1, p4, width, height)
    return vertical / max(horizontal, 1e-6)


def _distance(a, b, width: int, height: int) -> float:
    ax, ay = a.x * width, a.y * height
    bx, by = b.x * width, b.y * height
    return float(np.hypot(ax - bx, ay - by))


def _estimate_head_pose(landmarks, width: int, height: int) -> Tuple[float, float]:
    image_points = np.array(
        [
            (landmarks[1].x * width, landmarks[1].y * height),
            (landmarks[152].x * width, landmarks[152].y * height),
            (landmarks[33].x * width, landmarks[33].y * height),
            (landmarks[263].x * width, landmarks[263].y * height),
            (landmarks[61].x * width, landmarks[61].y * height),
            (landmarks[291].x * width, landmarks[291].y * height),
        ],
        dtype=np.float64,
    )
    model_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1),
        ],
        dtype=np.float64,
    )
    focal_length = float(width)
    camera_matrix = np.array(
        [[focal_length, 0, width / 2], [0, focal_length, height / 2], [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    ok, rotation_vector, _ = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return 0.0, 0.0
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles, *_ = cv2.RQDecomp3x3(rotation_matrix)
    pitch = float(angles[0])
    yaw = float(angles[1])
    return pitch, yaw
