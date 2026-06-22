from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import BoundedSemaphore, Lock
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from .config import Settings


@dataclass
class AntiSpoofingResult:
    live_score: float
    passed: bool
    frame_scores: List[float]
    class_probs: List[float]
    passed_frame_ratio: float


class AntiSpoofingModel:
    """MiniFASNet-V2 based presentation-attack detector.

    Public MiniFASNet ONNX exports are inconsistent about output class order.
    The live class is configurable through FACE_DEMO_ANTI_SPOOFING_LIVE_CLASS_INDEX.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._load_lock = Lock()
        self._detector_lock = Lock()
        max_concurrent = max(1, int(self.settings.anti_spoofing_max_concurrent_inferences))
        self._inference_gate = BoundedSemaphore(max_concurrent)
        self._sessions: Optional[List[Tuple[ort.InferenceSession, str, float]]] = None
        self._face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def predict(
        self,
        frames: Iterable[np.ndarray],
        bbox: List[float],
        bboxes: Optional[Iterable[Optional[List[float]]]] = None,
    ) -> AntiSpoofingResult:
        selected = _sample_frames(list(frames), self.settings.anti_spoofing_sample_frames)
        selected_bboxes = _sample_bboxes(list(bboxes or []), len(selected))
        scores: List[float] = []
        probs_by_frame: List[np.ndarray] = []
        for index, frame in enumerate(selected):
            frame_bbox = selected_bboxes[index] if index < len(selected_bboxes) else None
            frame_bbox = frame_bbox or self._detect_face_bbox(frame) or bbox
            with self._inference_gate:
                model_probs = []
                for session, input_name, scale in self._ensure_sessions():
                    crop = _crop_face(frame, frame_bbox, scale)
                    if crop is None:
                        continue
                    tensor = _preprocess(crop)
                    logits = session.run(None, {input_name: tensor})[0][0]
                    model_probs.append(_softmax(logits))
            if not model_probs:
                continue
            probs = np.mean(np.stack(model_probs, axis=0), axis=0)
            probs_by_frame.append(probs)
            live_index = min(max(self.settings.anti_spoofing_live_class_index, 0), len(probs) - 1)
            scores.append(float(probs[live_index]))

        live_score = float(np.mean(scores)) if scores else 0.0
        passed_frames = [
            score for score in scores if score >= self.settings.anti_spoofing_threshold
        ]
        passed_frame_ratio = len(passed_frames) / max(len(scores), 1)
        class_probs = (
            np.mean(np.stack(probs_by_frame, axis=0), axis=0).astype(float).tolist()
            if probs_by_frame
            else [0.0, 0.0, 0.0]
        )
        passed = (
            len(scores) >= self.settings.anti_spoofing_min_valid_frames
            and live_score >= self.settings.anti_spoofing_threshold
            and passed_frame_ratio >= self.settings.anti_spoofing_min_passed_frame_ratio
        )
        return AntiSpoofingResult(
            live_score=live_score,
            passed=passed,
            frame_scores=scores,
            class_probs=class_probs,
            passed_frame_ratio=passed_frame_ratio,
        )

    def preload(self) -> None:
        self._ensure_sessions()

    def _ensure_sessions(self) -> List[Tuple[ort.InferenceSession, str, float]]:
        if self._sessions is None:
            with self._load_lock:
                if self._sessions is None:
                    model_paths = [
                        item.strip()
                        for item in self.settings.anti_spoofing_model_path.split(",")
                        if item.strip()
                    ]
                    if not model_paths:
                        raise FileNotFoundError("未配置反欺骗模型")
                    scales = _parse_scales(self.settings.anti_spoofing_model_scales, len(model_paths))
                    sessions: List[Tuple[ort.InferenceSession, str, float]] = []
                    for item, scale in zip(model_paths, scales):
                        model_path = Path(item)
                        if not model_path.is_absolute():
                            model_path = Path.cwd() / model_path
                        if not model_path.exists():
                            raise FileNotFoundError(f"反欺骗模型不存在：{model_path}")
                        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
                        sessions.append((session, session.get_inputs()[0].name, scale))
                    self._sessions = sessions
        return self._sessions

    def _detect_face_bbox(self, bgr: np.ndarray) -> Optional[List[float]]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        with self._detector_lock:
            faces = self._face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(60, 60),
            )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda item: int(item[2]) * int(item[3]))
        return [float(x), float(y), float(x + w), float(y + h)]


def _sample_frames(frames: List[np.ndarray], count: int) -> List[np.ndarray]:
    if len(frames) <= count:
        return frames
    indexes = np.linspace(0, len(frames) - 1, count).round().astype(int)
    return [frames[int(index)] for index in indexes]


def _sample_bboxes(bboxes: List[Optional[List[float]]], count: int) -> List[Optional[List[float]]]:
    if not bboxes:
        return []
    if len(bboxes) <= count:
        return bboxes
    indexes = np.linspace(0, len(bboxes) - 1, count).round().astype(int)
    return [bboxes[int(index)] for index in indexes]


def _parse_scales(value: str, count: int) -> List[float]:
    scales = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not scales:
        scales = [2.7]
    while len(scales) < count:
        scales.append(scales[-1])
    return scales[:count]


def _crop_face(bgr: np.ndarray, bbox: List[float], scale: float) -> Optional[np.ndarray]:
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0:
        return None

    scaled_w = box_w * scale
    scaled_h = box_h * scale
    left = max(0, int(round(x1 - (scaled_w - box_w) / 2.0)))
    top = max(0, int(round(y1 - (scaled_h - box_h) / 2.0)))
    right = min(w, int(round(left + scaled_w)))
    bottom = min(h, int(round(top + scaled_h)))
    if right <= left or bottom <= top:
        return None
    return bgr[top:bottom, left:right]


def _preprocess(crop: np.ndarray) -> np.ndarray:
    resized = cv2.resize(crop, (80, 80), interpolation=cv2.INTER_LINEAR)
    tensor = resized.astype(np.float32).transpose(2, 0, 1)
    return np.expand_dims(tensor, axis=0)


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    logits = logits - float(np.max(logits))
    probs = np.exp(logits)
    return probs / max(float(np.sum(probs)), 1e-12)
