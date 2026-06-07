from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .anti_spoofing import AntiSpoofingModel
from .config import Settings, get_settings
from .face_engine import FaceEngine, cosine_similarity, decode_base64_image, decode_image_bytes
from .liveness import ACTION_LABELS, random_actions, verify_liveness_actions
from .schemas import (
    ChallengeRequest,
    ChallengeResponse,
    CompareRequest,
    CompareResponse,
    EnrollResponse,
    VerifyLivenessRequest,
    VerifyLivenessResponse,
)
from .stores import store


settings = get_settings()
face_engine = FaceEngine(settings)
anti_spoofing_model = AntiSpoofingModel(settings)

app = FastAPI(title="Face Verification Liveness Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).resolve().parents[1] / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@dataclass
class LiveFaceCheck:
    passed: bool
    embedding: Optional[np.ndarray]
    embeddings: list
    bbox: Optional[list]
    bboxes: list
    face_frame_ratio: float
    consistency_score: float
    consistency_pass_ratio: float
    detail: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/enroll", response_model=EnrollResponse)
async def enroll_face(file: UploadFile = File(...)) -> EnrollResponse:
    raw = await file.read()
    try:
        bgr = decode_image_bytes(raw, settings.max_upload_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    faces = face_engine.extract_faces(bgr)
    if not faces:
        raise HTTPException(status_code=400, detail="上传图片未检测到人脸")
    if len(faces) > 1:
        raise HTTPException(status_code=400, detail="上传图片检测到多张人脸，请只上传本人清晰正脸")

    face = faces[0]
    enrollment = store.create_enrollment(face.embedding, face.bbox)
    return EnrollResponse(
        enrollment_id=enrollment.id,
        bbox=enrollment.bbox,
        message="基准人脸上传成功，已提取人脸特征",
    )


@app.post("/api/liveness/challenge", response_model=ChallengeResponse)
def create_liveness_challenge(
    payload: Optional[ChallengeRequest] = Body(default=None),
) -> ChallengeResponse:
    if payload is None or not payload.enrollment_id:
        raise HTTPException(status_code=400, detail="请先上传基准人脸")
    enrollment = store.get_enrollment(payload.enrollment_id)
    if enrollment is None:
        raise HTTPException(status_code=404, detail="基准人脸不存在，请重新上传")

    challenge = store.create_challenge(
        random_actions(settings.liveness_action_min_count, settings.liveness_action_max_count),
        ttl_seconds=settings.challenge_ttl_seconds,
        enrollment_id=payload.enrollment_id,
    )
    return ChallengeResponse(
        challenge_id=challenge.id,
        actions=challenge.actions,
        labels=ACTION_LABELS,
    )


@app.post("/api/liveness/verify", response_model=VerifyLivenessResponse)
def verify_liveness(payload: VerifyLivenessRequest) -> VerifyLivenessResponse:
    challenge = store.get_challenge(payload.challenge_id)
    if challenge is None:
        raise HTTPException(status_code=404, detail="活体挑战不存在或已过期")
    if challenge.is_expired():
        raise HTTPException(status_code=410, detail="活体挑战已过期，请重新生成")
    if challenge.verified_at is not None:
        raise HTTPException(status_code=409, detail="活体挑战已使用，请重新生成")
    if not payload.frames:
        raise HTTPException(status_code=400, detail="没有收到摄像头帧")

    if challenge.enrollment_id and payload.enrollment_id and payload.enrollment_id != challenge.enrollment_id:
        raise HTTPException(status_code=403, detail="活体挑战与基准人脸不一致，请重新验证")

    enrollment_id = challenge.enrollment_id or payload.enrollment_id
    if not enrollment_id:
        raise HTTPException(status_code=400, detail="请先上传基准人脸")
    enrollment = store.get_enrollment(enrollment_id)
    if enrollment is None:
        raise HTTPException(status_code=404, detail="基准人脸不存在，请重新上传")

    challenge.verified_at = datetime.now(timezone.utc)
    _validate_frame_sequence(payload.frames, challenge.actions, settings)

    frames_by_action = defaultdict(list)
    all_frames = []
    for frame in payload.frames:
        if len(frame.image) > settings.max_frame_base64_chars:
            raise HTTPException(status_code=400, detail="单帧图片太大，请降低摄像头采集分辨率")
        try:
            bgr = decode_base64_image(frame.image, settings.max_upload_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        frames_by_action[frame.action].append(bgr)
        all_frames.append(bgr)

    if not all_frames:
        raise HTTPException(status_code=400, detail="没有收到有效动作帧")

    frame_uniqueness = _validate_frame_uniqueness(all_frames, settings)

    live_face_check = _analyze_live_faces(all_frames, settings)
    if live_face_check.embedding is None or live_face_check.bbox is None:
        raise HTTPException(status_code=400, detail=live_face_check.detail or "摄像头画面未检测到人脸")

    action_results = verify_liveness_actions(frames_by_action, challenge.actions, settings)
    action_passed = all(item["passed"] for item in action_results)
    action_results.append(frame_uniqueness)
    action_results.append(
        {
            "action": "face_consistency",
            "label": "人脸稳定性",
            "passed": live_face_check.passed,
            "score": round(live_face_check.consistency_score, 4),
            "detail": live_face_check.detail,
        }
    )

    try:
        spoof_result = anti_spoofing_model.predict(
            all_frames,
            live_face_check.bbox,
            live_face_check.bboxes,
        )
    except Exception:
        raise HTTPException(status_code=503, detail="防翻拍模型暂不可用，请稍后重试")
    class_probs = (spoof_result.class_probs + [0.0, 0.0, 0.0])[:3]
    class0_prob, class1_prob, class2_prob = class_probs
    action_results.append(
        {
            "action": "anti_spoofing",
            "label": "防翻拍检测",
            "passed": spoof_result.passed,
            "score": round(spoof_result.live_score, 4),
            "detail": (
                f"live score={spoof_result.live_score:.3f}, "
                f"class0={class0_prob:.3f}, class1={class1_prob:.3f}, class2={class2_prob:.3f}, "
                f"live_class={settings.anti_spoofing_live_class_index}, "
                f"passed_frame_ratio={spoof_result.passed_frame_ratio:.0%}, "
                f"scales={settings.anti_spoofing_model_scales}, "
                f"threshold={settings.anti_spoofing_threshold:.2f}, "
                f"frames={len(spoof_result.frame_scores)}"
            ),
        }
    )

    face_match = _match_live_face(enrollment.embedding, live_face_check.embeddings, settings)
    action_results.append(
        {
            "action": "face_match",
            "label": "人脸一致性",
            "passed": face_match["passed"],
            "score": round(face_match["similarity"], 4),
            "detail": face_match["detail"],
        }
    )

    liveness_passed = (
        action_passed
        and frame_uniqueness["passed"]
        and live_face_check.passed
        and spoof_result.passed
        and face_match["passed"]
    )

    challenge.liveness_passed = liveness_passed
    challenge.live_embedding = live_face_check.embedding if liveness_passed else None
    challenge.face_matched = face_match["passed"]
    challenge.face_similarity = face_match["similarity"]
    challenge.action_results = action_results
    if liveness_passed:
        message = "活体检测通过，且为本人"
    elif action_passed and live_face_check.passed and spoof_result.passed and not face_match["passed"]:
        message = "人脸比对失败，不是同一个人"
    else:
        message = "活体检测失败"

    return VerifyLivenessResponse(
        challenge_id=challenge.id,
        liveness_passed=liveness_passed,
        anti_spoofing_passed=spoof_result.passed,
        anti_spoofing_score=round(spoof_result.live_score, 4),
        face_matched=face_match["passed"],
        similarity=round(face_match["similarity"], 4),
        threshold=settings.face_match_threshold,
        action_results=action_results,
        message=message,
    )


@app.post("/api/compare", response_model=CompareResponse)
def compare_faces(payload: CompareRequest) -> CompareResponse:
    enrollment = store.get_enrollment(payload.enrollment_id)
    if enrollment is None:
        raise HTTPException(status_code=404, detail="基准人脸不存在，请重新上传")

    challenge = store.get_challenge(payload.challenge_id)
    if challenge is None:
        raise HTTPException(status_code=404, detail="活体挑战不存在或已过期")

    if challenge.enrollment_id and payload.enrollment_id != challenge.enrollment_id:
        raise HTTPException(status_code=403, detail="活体挑战与基准人脸不一致，请重新验证")

    if not challenge.liveness_passed or challenge.live_embedding is None:
        raise HTTPException(status_code=403, detail="活体检测未通过，不能进行人脸比对")

    threshold = settings.face_match_threshold
    similarity = cosine_similarity(enrollment.embedding, challenge.live_embedding)
    matched = similarity >= threshold
    return CompareResponse(
        liveness_passed=True,
        similarity=round(similarity, 4),
        threshold=threshold,
        matched=matched,
        message="验证成功，是本人" if matched else "人脸比对失败，不是同一个人",
    )


def _validate_frame_sequence(frames: list, expected_actions: list, settings: Settings) -> None:
    if len(frames) > settings.max_total_frames:
        raise HTTPException(status_code=400, detail=f"摄像头帧过多：{len(frames)}/{settings.max_total_frames}")

    expected_set = set(expected_actions)
    submitted_actions = [frame.action for frame in frames]
    unexpected = sorted({action for action in submitted_actions if action not in expected_set})
    if unexpected:
        raise HTTPException(status_code=400, detail=f"收到非本次挑战动作：{', '.join(unexpected)}")

    sequence = []
    for action in submitted_actions:
        if not sequence or sequence[-1] != action:
            sequence.append(action)
    if sequence != expected_actions:
        raise HTTPException(status_code=400, detail="动作提交顺序与本次挑战不一致")

    previous_timestamp = None
    for frame in frames:
        if previous_timestamp is not None and frame.timestamp < previous_timestamp:
            raise HTTPException(status_code=400, detail="摄像头帧时间戳顺序不正确")
        previous_timestamp = frame.timestamp

    frames_by_action = defaultdict(list)
    for frame in frames:
        frames_by_action[frame.action].append(frame)

    for action in expected_actions:
        action_frames = frames_by_action.get(action, [])
        frame_count = len(action_frames)
        if frame_count < settings.min_frames_per_action:
            raise HTTPException(
                status_code=400,
                detail=f"{ACTION_LABELS.get(action, action)} 有效帧不足：{frame_count}/{settings.min_frames_per_action}",
            )
        if frame_count > settings.max_frames_per_action:
            raise HTTPException(
                status_code=400,
                detail=f"{ACTION_LABELS.get(action, action)} 帧数过多：{frame_count}/{settings.max_frames_per_action}",
            )

        indexes = [frame.index for frame in action_frames]
        if indexes != list(range(frame_count)):
            raise HTTPException(status_code=400, detail=f"{ACTION_LABELS.get(action, action)} 帧序号不连续")

        timestamps = [float(frame.timestamp) for frame in action_frames]
        duration_ms = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
        if duration_ms < settings.min_action_duration_ms:
            raise HTTPException(
                status_code=400,
                detail=f"{ACTION_LABELS.get(action, action)} 采集时间过短：{duration_ms:.0f}ms/{settings.min_action_duration_ms}ms",
            )
        if duration_ms > settings.max_action_duration_ms:
            raise HTTPException(
                status_code=400,
                detail=f"{ACTION_LABELS.get(action, action)} 采集时间过长：{duration_ms:.0f}ms/{settings.max_action_duration_ms}ms",
            )
        gaps = [
            timestamps[index + 1] - timestamps[index]
            for index in range(len(timestamps) - 1)
        ]
        if gaps and max(gaps) > settings.max_frame_gap_ms:
            raise HTTPException(
                status_code=400,
                detail=f"{ACTION_LABELS.get(action, action)} 帧间隔过大，请保持连续采集",
            )


def _validate_frame_uniqueness(frames: list, settings: Settings) -> dict:
    if not frames:
        return {
            "action": "frame_uniqueness",
            "label": "帧连续性",
            "passed": False,
            "score": 0.0,
            "detail": "没有收到摄像头帧",
        }

    hashes = [_frame_hash(frame, settings.duplicate_frame_hash_size) for frame in frames]
    unique_ratio = len(set(hashes)) / max(len(hashes), 1)
    passed = unique_ratio >= settings.min_unique_frame_ratio
    return {
        "action": "frame_uniqueness",
        "label": "帧连续性",
        "passed": passed,
        "score": round(unique_ratio, 4),
        "detail": (
            f"unique_ratio={unique_ratio:.0%}, "
            f"threshold={settings.min_unique_frame_ratio:.0%}, frames={len(hashes)}"
        ),
    }


def _frame_hash(frame: np.ndarray, hash_size: int) -> tuple:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    return tuple(int(value) for value in resized.flatten())


def _analyze_live_faces(frames: list, settings: Settings) -> LiveFaceCheck:
    total_frames = len(frames)
    sample_count = max(settings.face_match_embedding_sample, 6)
    sample_indexes = _evenly_spaced_indexes(total_frames, sample_count)
    sample_set = set(sample_indexes)

    faces = []
    bboxes_by_frame = [None] * total_frames
    for index, frame in enumerate(frames):
        if index not in sample_set:
            continue
        frame_faces = face_engine.extract_faces(frame)
        if not frame_faces:
            continue
        if len(frame_faces) > 1:
            return LiveFaceCheck(
                passed=False,
                embedding=None,
                embeddings=[],
                bbox=None,
                bboxes=bboxes_by_frame,
                face_frame_ratio=0.0,
                consistency_score=0.0,
                consistency_pass_ratio=0.0,
                detail="摄像头画面检测到多张人脸，请确保画面中只有本人",
            )
        face = max(frame_faces, key=lambda item: _bbox_area(item.bbox))
        faces.append(face)
        bboxes_by_frame[index] = face.bbox

    if not faces:
        return LiveFaceCheck(
            passed=False,
            embedding=None,
            embeddings=[],
            bbox=None,
            bboxes=bboxes_by_frame,
            face_frame_ratio=0.0,
            consistency_score=0.0,
            consistency_pass_ratio=0.0,
            detail="未检测到可用于比对的人脸",
        )

    sampled_total = max(len(sample_indexes), 1)
    face_frame_ratio = len(faces) / sampled_total
    embedding = _mean_embedding([face.embedding for face in faces])
    consistency_scores = [cosine_similarity(face.embedding, embedding) for face in faces]
    consistency_score = float(min(consistency_scores)) if consistency_scores else 0.0
    consistency_pass_ratio = sum(
        score >= settings.face_consistency_threshold for score in consistency_scores
    ) / max(len(consistency_scores), 1)
    passed = (
        face_frame_ratio >= settings.min_face_frame_ratio
        and consistency_pass_ratio >= settings.face_consistency_min_pass_ratio
    )
    detail = (
        f"face_frames={len(faces)}/{sampled_total}({face_frame_ratio:.0%}), "
        f"min_similarity={consistency_score:.4f}, "
        f"pass_ratio={consistency_pass_ratio:.0%}, "
        f"threshold={settings.face_consistency_threshold:.2f}"
    )
    bboxes = np.asarray([face.bbox for face in faces], dtype=np.float32)
    bbox = np.mean(bboxes, axis=0).astype(float).tolist()
    return LiveFaceCheck(
        passed=passed,
        embedding=embedding,
        embeddings=[face.embedding for face in faces],
        bbox=bbox,
        bboxes=bboxes_by_frame,
        face_frame_ratio=face_frame_ratio,
        consistency_score=consistency_score,
        consistency_pass_ratio=consistency_pass_ratio,
        detail=detail,
    )


def _evenly_spaced_indexes(total: int, count: int) -> list:
    if total <= 0:
        return []
    if total <= count:
        return list(range(total))
    indexes = np.linspace(0, total - 1, count).round().astype(int)
    return sorted(set(int(idx) for idx in indexes))


def _mean_embedding(embeddings: list) -> np.ndarray:
    embedding = np.mean(np.stack(embeddings, axis=0), axis=0)
    norm = max(float(np.linalg.norm(embedding)), 1e-12)
    return (embedding / norm).astype(np.float32)


def _match_live_face(
    enrollment_embedding: np.ndarray,
    live_embeddings: list,
    settings: Settings,
) -> dict:
    if not live_embeddings:
        return {
            "passed": False,
            "similarity": 0.0,
            "detail": "未检测到可用于人脸一致性比对的活体人脸",
        }

    scores = [cosine_similarity(enrollment_embedding, embedding) for embedding in live_embeddings]
    sorted_scores = sorted(scores, reverse=True)
    top_k = max(1, len(sorted_scores) // 2 + (len(sorted_scores) % 2))
    top_scores = sorted_scores[:top_k]
    mean_score = float(np.mean(scores))
    top_mean_score = float(np.mean(top_scores))
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    pass_ratio = sum(score >= settings.face_match_threshold for score in scores) / max(len(scores), 1)
    passed = (
        top_mean_score >= settings.face_match_threshold
        and max_score >= settings.face_match_threshold
        and pass_ratio >= settings.face_match_min_pass_ratio
        and min_score >= settings.face_match_min_threshold
    )
    return {
        "passed": passed,
        "similarity": top_mean_score,
        "detail": (
            f"top{top_k}_mean={top_mean_score:.4f}, mean={mean_score:.4f}, "
            f"max={max_score:.4f}, min={min_score:.4f}, "
            f"pass_ratio={pass_ratio:.0%}, "
            f"threshold={settings.face_match_threshold:.2f}, "
            f"min_threshold={settings.face_match_min_threshold:.2f}"
        ),
    }


def _bbox_area(bbox: list) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
