from __future__ import annotations

import base64
import binascii
import os
from dataclasses import dataclass
from threading import Lock
from typing import List, Optional

import cv2
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/face-verify-demo-matplotlib")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from insightface.app import FaceAnalysis

from .config import Settings


@dataclass
class FaceEmbedding:
    embedding: np.ndarray
    bbox: List[float]


class FaceEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = Lock()
        self._app: Optional[FaceAnalysis] = None

    def _ensure_loaded(self) -> FaceAnalysis:
        if self._app is None:
            app = FaceAnalysis(
                name=self.settings.insightface_model,
                providers=self.settings.insightface_providers,
            )
            app.prepare(
                ctx_id=self.settings.insightface_ctx_id,
                det_size=(self.settings.insightface_det_size, self.settings.insightface_det_size),
            )
            self._app = app
        return self._app

    def extract_largest_face(self, bgr: np.ndarray) -> Optional[FaceEmbedding]:
        faces = self.extract_faces(bgr)
        if not faces:
            return None
        return max(faces, key=lambda item: _bbox_area(np.asarray(item.bbox, dtype=np.float32)))

    def extract_faces(self, bgr: np.ndarray) -> List[FaceEmbedding]:
        app = self._ensure_loaded()
        with self._lock:
            faces = app.get(bgr)
        if not faces:
            return []

        results: List[FaceEmbedding] = []
        for face in faces:
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                embedding = face.embedding
                embedding = embedding / max(np.linalg.norm(embedding), 1e-12)
            results.append(
                FaceEmbedding(
                    embedding=np.asarray(embedding, dtype=np.float32),
                    bbox=[float(x) for x in face.bbox],
                )
            )
        return results


def _bbox_area(bbox: np.ndarray) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def decode_image_bytes(data: bytes, max_bytes: int) -> np.ndarray:
    if len(data) > max_bytes:
        raise ValueError("图片太大")
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("无法解析图片")
    return bgr


def decode_base64_image(value: str, max_bytes: int) -> np.ndarray:
    value = value.strip()
    if "," in value and value.startswith("data:"):
        header, value = value.split(",", 1)
        if ";base64" not in header:
            raise ValueError("图片 data URL 必须使用 base64 编码")
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("图片 base64 编码无效") from exc
    return decode_image_bytes(raw, max_bytes=max_bytes)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    b = b / max(float(np.linalg.norm(b)), 1e-12)
    return float(np.dot(a, b))
