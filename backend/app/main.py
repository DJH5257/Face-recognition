from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import secrets
from threading import BoundedSemaphore
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from fastapi import BackgroundTasks, Body, Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .anti_spoofing import AntiSpoofingModel
from .config import Settings, get_settings
from .face_engine import FaceEngine, cosine_similarity, decode_base64_image, decode_image_bytes
from .liveness import ACTION_LABELS, _extract_metrics, random_actions, verify_liveness_actions
from .rate_limit import InMemoryFailureLimiter
from .schemas import (
    ChallengeRequest,
    ChallengeResponse,
    CompareRequest,
    CompareResponse,
    EnrollResponse,
    ProofFinalizeRequest,
    ProofFinalizeResponse,
    ProofIntrospectRequest,
    ProofIntrospectResponse,
    TemplateCreateRequest,
    TemplateCreateResponse,
    TemplateInfo,
    TemplateListResponse,
    TemplateRevokeResponse,
    VerifyLivenessRequest,
    VerifyLivenessResponse,
    VerificationSessionCreateRequest,
    VerificationSessionCreateResponse,
    VerificationSessionVerifyRequest,
    VerificationSessionVerifyResponse,
)
from .stores import store
from .template_crypto import TemplateCryptoError


settings = get_settings()
face_engine = FaceEngine(settings)
anti_spoofing_model = AntiSpoofingModel(settings)
failure_limiter = InMemoryFailureLimiter()
verification_semaphore = BoundedSemaphore(max(1, settings.verification_max_concurrent_requests))

app = FastAPI(title="Face Verification Liveness Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
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


@dataclass
class VerificationEvaluation:
    passed: bool
    result_code: str
    anti_spoofing_passed: bool
    anti_spoofing_score: float
    face_matched: bool
    similarity: Optional[float]
    threshold: float
    action_results: list
    live_embedding: Optional[np.ndarray]
    message: str


@dataclass
class ImageQualitySummary:
    passed: bool
    score: float
    detail: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/ready")
def ready() -> dict:
    model_paths = _configured_model_paths(settings) + _configured_insightface_model_paths(settings)
    missing_models = [str(path) for path in model_paths if not path.exists()]
    database_parent = getattr(store, "database_path", Path("data/face_verify.sqlite3")).parent
    ok = database_parent.exists() and not missing_models
    if not ok:
        raise HTTPException(
            status_code=503,
            detail={
                "ok": False,
                "database_parent_exists": database_parent.exists(),
                "missing_models": missing_models,
            },
        )
    return {
        "ok": True,
        "database_parent_exists": True,
        "missing_models": [],
    }


@app.post("/api/enroll", response_model=EnrollResponse)
def enroll_face(file: UploadFile = File(...)) -> EnrollResponse:
    raw = file.file.read()
    try:
        bgr = decode_image_bytes(raw, settings.max_upload_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    image_quality = _validate_sampled_image_quality([bgr], settings, "基准图像质量")
    if not image_quality["passed"]:
        raise HTTPException(status_code=400, detail=f"基准图片质量不达标：{image_quality['detail']}")

    faces = face_engine.extract_faces(bgr)
    if not faces:
        raise HTTPException(status_code=400, detail="上传图片未检测到人脸")
    if len(faces) > 1:
        raise HTTPException(status_code=400, detail="上传图片检测到多张人脸，请只上传本人清晰正脸")

    face = faces[0]
    face_quality = _validate_face_bbox_quality(
        face.bbox,
        bgr.shape,
        settings.image_quality_template_min_face_ratio,
        settings,
        "基准人脸画面",
    )
    if not face_quality["passed"]:
        raise HTTPException(status_code=400, detail=f"基准人脸画面不达标：{face_quality['detail']}")

    pose_quality = _validate_pose_quality(
        [bgr],
        settings,
        "基准姿态质量",
        settings.image_quality_template_max_abs_yaw,
        settings.image_quality_template_max_abs_pitch,
    )
    if not pose_quality["passed"]:
        raise HTTPException(status_code=400, detail=f"基准人脸姿态不达标：{pose_quality['detail']}")

    enrollment = store.create_enrollment(face.embedding, face.bbox)
    return EnrollResponse(
        enrollment_id=enrollment.id,
        bbox=enrollment.bbox,
        message="基准人脸上传成功，已提取人脸特征",
    )


@app.post("/api/liveness/challenge", response_model=ChallengeResponse)
def create_liveness_challenge(
    background_tasks: BackgroundTasks,
    payload: Optional[ChallengeRequest] = Body(default=None),
) -> ChallengeResponse:
    if payload is None or not payload.enrollment_id:
        raise HTTPException(status_code=400, detail="请先上传基准人脸")
    enrollment = store.get_enrollment(payload.enrollment_id)
    if enrollment is None:
        raise HTTPException(status_code=404, detail="基准人脸不存在，请重新上传")

    actions = _select_liveness_actions(settings)
    challenge = store.create_challenge(
        actions,
        ttl_seconds=settings.challenge_ttl_seconds,
        enrollment_id=payload.enrollment_id,
    )
    background_tasks.add_task(_preload_anti_spoofing_model)
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

    with _VerificationSlot():
        evaluation = _evaluate_liveness_frames(
            payload.frames,
            challenge.actions,
            enrollment.embedding,
            settings,
        )

    challenge.liveness_passed = evaluation.passed
    challenge.live_embedding = evaluation.live_embedding if evaluation.passed else None
    challenge.face_matched = evaluation.face_matched
    challenge.face_similarity = evaluation.similarity
    challenge.action_results = evaluation.action_results

    return VerifyLivenessResponse(
        challenge_id=challenge.id,
        liveness_passed=evaluation.passed,
        anti_spoofing_passed=evaluation.anti_spoofing_passed,
        anti_spoofing_score=evaluation.anti_spoofing_score,
        face_matched=evaluation.face_matched,
        similarity=round(evaluation.similarity, 4) if evaluation.similarity is not None else None,
        threshold=settings.face_match_threshold,
        action_results=evaluation.action_results,
        message=evaluation.message,
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


def _require_internal_api_key(x_face_api_key: str = Header(default="")) -> bool:
    if not settings.internal_api_key:
        raise HTTPException(status_code=503, detail="服务鉴权未配置")
    if not secrets.compare_digest(x_face_api_key, settings.internal_api_key):
        raise HTTPException(status_code=401, detail="服务鉴权失败")
    return True


@app.post("/v1/internal/templates", response_model=TemplateCreateResponse)
def create_template(
    payload: TemplateCreateRequest,
    _authorized: bool = Depends(_require_internal_api_key),
) -> TemplateCreateResponse:
    try:
        bgr = decode_base64_image(payload.image, settings.max_upload_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    image_quality = _validate_sampled_image_quality([bgr], settings, "基准图像质量")
    if not image_quality["passed"]:
        raise HTTPException(status_code=400, detail=f"基准图片质量不达标：{image_quality['detail']}")

    faces = face_engine.extract_faces(bgr)
    if not faces:
        raise HTTPException(status_code=400, detail="基准图片未检测到人脸")
    if len(faces) > 1:
        raise HTTPException(status_code=400, detail="基准图片检测到多张人脸，请只登记本人清晰正脸")

    face = faces[0]
    face_quality = _validate_face_bbox_quality(
        face.bbox,
        bgr.shape,
        settings.image_quality_template_min_face_ratio,
        settings,
        "基准人脸画面",
    )
    if not face_quality["passed"]:
        raise HTTPException(status_code=400, detail=f"基准人脸画面不达标：{face_quality['detail']}")

    pose_quality = _validate_pose_quality(
        [bgr],
        settings,
        "基准姿态质量",
        settings.image_quality_template_max_abs_yaw,
        settings.image_quality_template_max_abs_pitch,
    )
    if not pose_quality["passed"]:
        raise HTTPException(status_code=400, detail=f"基准人脸姿态不达标：{pose_quality['detail']}")

    source_hash = hashlib.sha256(payload.image.encode("utf-8")).hexdigest()
    try:
        template = store.create_template(
            subject_type=payload.subject_type,
            subject_id=payload.subject_id,
            embedding=face.embedding,
            bbox=face.bbox,
            source_type=payload.source_type,
            source_image_hash=source_hash,
        )
    except TemplateCryptoError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return TemplateCreateResponse(
        template_id=template.template_id,
        template_version=template.template_version,
        subject_type=template.subject_type,
        subject_id=template.subject_id,
        status=template.status,
        bbox=template.bbox,
        message="新版基准人脸模板已生效",
    )


@app.get(
    "/v1/internal/templates/subjects/{subject_type}/{subject_id}",
    response_model=TemplateListResponse,
)
def list_subject_templates(
    subject_type: str,
    subject_id: str,
    _authorized: bool = Depends(_require_internal_api_key),
) -> TemplateListResponse:
    try:
        templates = store.list_templates(subject_type, subject_id)
    except TemplateCryptoError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    active = next((item for item in templates if item.status == "active"), None)
    return TemplateListResponse(
        subject_type=subject_type,
        subject_id=subject_id,
        active_template_id=active.template_id if active else None,
        templates=[_template_info(item) for item in templates],
    )


@app.post("/v1/internal/templates/{template_id}/revoke", response_model=TemplateRevokeResponse)
def revoke_template(
    template_id: str,
    _authorized: bool = Depends(_require_internal_api_key),
) -> TemplateRevokeResponse:
    template = store.revoke_template(template_id)
    if template is None:
        raise HTTPException(status_code=404, detail="人脸模板不存在")
    return TemplateRevokeResponse(
        template_id=template.template_id,
        revoked=template.status == "revoked",
        status=template.status,
    )


@app.post("/v1/internal/verification-sessions", response_model=VerificationSessionCreateResponse)
def create_verification_session(
    background_tasks: BackgroundTasks,
    payload: VerificationSessionCreateRequest,
    _authorized: bool = Depends(_require_internal_api_key),
) -> VerificationSessionCreateResponse:
    actions = _select_liveness_actions(
        settings,
        payload.subject_type,
        payload.subject_id,
        payload.scene,
    )
    try:
        session = store.create_verification_session(
            request_id=payload.request_id,
            subject_type=payload.subject_type,
            subject_id=payload.subject_id,
            admin_id=payload.admin_id,
            scene=payload.scene,
            business_event_id=payload.business_event_id,
            record_id=payload.record_id,
            action=payload.action,
            expected_actions=actions,
            ttl_seconds=settings.verification_session_ttl_seconds,
        )
    except TemplateCryptoError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if session is None:
        raise HTTPException(status_code=404, detail="未找到有效人脸模板，请先登记基准人脸")

    background_tasks.add_task(_preload_anti_spoofing_model)
    return VerificationSessionCreateResponse(
        session_id=session.session_id,
        upload_token=session.upload_token,
        actions=session.expected_actions,
        labels=ACTION_LABELS,
        expires_at=session.expires_at,
    )


@app.post(
    "/v1/verification-sessions/{session_id}/verify",
    response_model=VerificationSessionVerifyResponse,
)
def verify_verification_session(
    session_id: str,
    payload: VerificationSessionVerifyRequest,
    request: Request,
    authorization: Optional[str] = Header(default=None),
) -> VerificationSessionVerifyResponse:
    store.recover_stale_session_verification(
        session_id,
        settings.verification_processing_timeout_seconds,
    )
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="核验会话不存在")
    if session.is_expired():
        store.mark_session_failed(session_id, "EXPIRED_SESSION")
        raise HTTPException(status_code=410, detail="核验会话已过期，请重新发起")
    if session.status != "issued":
        raise HTTPException(status_code=409, detail="核验会话已使用，请重新发起")

    upload_token = _extract_bearer_token(authorization)
    if not upload_token or not store.verify_upload_token(session_id, upload_token):
        raise HTTPException(status_code=401, detail="上传令牌无效")

    try:
        template = store.get_template(session.template_id)
    except TemplateCryptoError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if template is None or template.status != "active":
        store.mark_session_failed(session_id, "FAIL_TEMPLATE_INVALID")
        raise HTTPException(status_code=409, detail="人脸模板无效，请重新登记")

    failure_key = _failure_limit_key(session, _client_ip(request))
    _raise_if_failure_limited(failure_key, settings)

    verification_attempt = store.start_session_verification(session_id)
    if not verification_attempt:
        raise HTTPException(status_code=409, detail="核验会话已使用，请重新发起")

    try:
        with _VerificationSlot():
            evaluation = _evaluate_liveness_frames(
                payload.frames,
                session.expected_actions,
                template.embedding,
                settings,
            )
    except HTTPException as exc:
        result_code = _result_code_from_exception(exc)
        if result_code == "ERROR_MODEL_UNAVAILABLE":
            store.reset_session_verification(session_id, verification_attempt)
        else:
            store.mark_session_failed(session_id, result_code, verification_attempt)
        raise
    except Exception as exc:
        store.mark_session_failed(session_id, "ERROR_INTERNAL", verification_attempt)
        raise HTTPException(status_code=500, detail="核验处理失败，请稍后重试") from exc

    proof_id = None
    if evaluation.passed:
        proof = store.create_proof_for_session(
            session_id=session_id,
            result_code="PASS",
            ttl_seconds=settings.proof_ttl_seconds,
            verifying_started_at=verification_attempt,
            action_results=evaluation.action_results,
        )
        if proof is None:
            raise HTTPException(status_code=409, detail="核验会话已超时，请重新提交")
        proof_id = proof.proof_id
    else:
        store.mark_session_failed(
            session_id,
            evaluation.result_code,
            verification_attempt,
            action_results=evaluation.action_results,
        )
    _record_failure_limit_result(failure_key, evaluation.passed, settings)

    return VerificationSessionVerifyResponse(
        session_id=session_id,
        passed=evaluation.passed,
        result_code=evaluation.result_code,
        proof_id=proof_id,
        anti_spoofing_passed=evaluation.anti_spoofing_passed,
        anti_spoofing_score=evaluation.anti_spoofing_score,
        face_matched=evaluation.face_matched,
        similarity=round(evaluation.similarity, 4) if evaluation.similarity is not None else None,
        threshold=evaluation.threshold,
        action_results=evaluation.action_results,
        message=evaluation.message,
    )


@app.post("/v1/internal/proofs/introspect", response_model=ProofIntrospectResponse)
def introspect_proof(
    payload: ProofIntrospectRequest,
    _authorized: bool = Depends(_require_internal_api_key),
) -> ProofIntrospectResponse:
    proof = store.get_proof(payload.proof_id)
    if proof is None:
        return ProofIntrospectResponse(
            valid=False,
            proof_id=payload.proof_id,
            status="not_found",
            result_code="PROOF_NOT_FOUND",
        )
    return _proof_response(proof)


@app.post("/v1/internal/proofs/{proof_id}/finalize", response_model=ProofFinalizeResponse)
def finalize_proof(
    proof_id: str,
    payload: ProofFinalizeRequest,
    _authorized: bool = Depends(_require_internal_api_key),
) -> ProofFinalizeResponse:
    try:
        proof = store.finalize_proof(proof_id, payload.business_event_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if proof is None:
        raise HTTPException(status_code=404, detail="proof 不存在")
    return ProofFinalizeResponse(
        proof_id=proof.proof_id,
        finalized=proof.status == "finalized",
        status=proof.status,
    )


def _extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        return None
    token = authorization[len(prefix):].strip()
    return token or None


def _template_info(template) -> TemplateInfo:
    return TemplateInfo(
        template_id=template.template_id,
        template_version=template.template_version,
        subject_type=template.subject_type,
        subject_id=template.subject_id,
        status=template.status,
        source_type=template.source_type,
        source_image_hash=template.source_image_hash,
        bbox=template.bbox,
        created_at=template.created_at,
        activated_at=template.activated_at,
        revoked_at=template.revoked_at,
    )


def _proof_response(proof) -> ProofIntrospectResponse:
    return ProofIntrospectResponse(
        valid=proof.valid,
        proof_id=proof.proof_id,
        status=proof.status,
        result_code=proof.result_code,
        session_id=proof.session_id,
        subject_type=proof.subject_type,
        subject_id=proof.subject_id,
        admin_id=proof.admin_id,
        scene=proof.scene,
        business_event_id=proof.business_event_id,
        record_id=proof.record_id,
        action=proof.action,
        template_id=proof.template_id,
        template_version=proof.template_version,
        issued_at=proof.issued_at,
        expires_at=proof.expires_at,
        finalized_at=proof.finalized_at,
    )


def _result_code_from_exception(exc: HTTPException) -> str:
    if exc.status_code == 410:
        return "EXPIRED_SESSION"
    if exc.status_code == 503:
        return "ERROR_MODEL_UNAVAILABLE"
    if exc.status_code == 429:
        return "RATE_LIMITED"
    return "FAIL_PROTOCOL"


def _action_weights_from_settings(settings: Settings) -> dict:
    weights = {}
    for item in settings.liveness_action_weights.split(","):
        if "=" not in item:
            continue
        action, value = item.split("=", 1)
        action = action.strip()
        if action not in ACTION_LABELS:
            continue
        try:
            weights[action] = max(float(value), 0.05)
        except ValueError:
            continue
    return weights


def _recent_actions_to_reduce(subject_type: str, subject_id: str, scene: str, settings: Settings) -> set:
    window = max(0, settings.liveness_action_recent_failure_window)
    if window < 2:
        return set()
    failed = store.recent_failed_action_names(subject_type, subject_id, scene, window)
    if len(failed) < window:
        return set()
    first = failed[0]
    if first and all(action == first for action in failed[:window]):
        return {first}
    return set()


def _select_liveness_actions(
    settings: Settings,
    subject_type: Optional[str] = None,
    subject_id: Optional[str] = None,
    scene: Optional[str] = None,
) -> list:
    reduced = set()
    if subject_type and subject_id and scene:
        reduced = _recent_actions_to_reduce(subject_type, subject_id, scene, settings)
    return random_actions(
        settings.liveness_action_min_count,
        settings.liveness_action_max_count,
        weights=_action_weights_from_settings(settings),
        reduced_actions=reduced,
        reduction_factor=settings.liveness_action_recent_failure_factor,
    )


def _failure_limit_key(session, client_ip: str) -> str:
    return "|".join([
        session.subject_type,
        session.subject_id,
        session.scene,
        client_ip or "unknown",
    ])


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    return request.client.host if request.client else "unknown"


def _raise_if_failure_limited(key: str, settings: Settings) -> None:
    if not settings.verification_failure_limit_enabled:
        return
    retry_after = failure_limiter.retry_after_seconds(key)
    if retry_after > 0:
        raise HTTPException(
            status_code=429,
            detail="尝试次数较多，请稍后再试",
            headers={"Retry-After": str(retry_after)},
        )


def _record_failure_limit_result(key: str, passed: bool, settings: Settings) -> None:
    if not settings.verification_failure_limit_enabled:
        return
    if passed:
        failure_limiter.clear(key)
        return
    failure_limiter.record_failure(
        key,
        settings.verification_failure_max_attempts,
        settings.verification_failure_window_seconds,
        settings.verification_failure_cooldown_seconds,
    )


class _VerificationSlot:
    def __enter__(self):
        acquired = verification_semaphore.acquire(blocking=False)
        if not acquired:
            raise HTTPException(
                status_code=503,
                detail="当前人脸核验请求较多，请稍后重试",
                headers={"Retry-After": "3"},
            )
        return self

    def __exit__(self, *_args):
        verification_semaphore.release()
        return False


def _evaluate_liveness_frames(
    frames: list,
    expected_actions: list,
    reference_embedding: np.ndarray,
    settings: Settings,
) -> VerificationEvaluation:
    _validate_frame_sequence(frames, expected_actions, settings)

    frames_by_action = defaultdict(list)
    all_frames = []
    for frame in frames:
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

    action_results = []
    image_quality = _validate_sampled_image_quality(all_frames, settings, "图像质量")
    action_results.append(image_quality)
    if not image_quality["passed"]:
        return VerificationEvaluation(
            passed=False,
            result_code="FAIL_FACE_QUALITY",
            anti_spoofing_passed=False,
            anti_spoofing_score=0.0,
            face_matched=False,
            similarity=None,
            threshold=settings.face_match_threshold,
            action_results=action_results,
            live_embedding=None,
            message="摄像头画面质量不达标",
        )

    frame_uniqueness = _validate_frame_uniqueness(all_frames, settings)
    action_results.append(frame_uniqueness)
    if not frame_uniqueness["passed"]:
        return VerificationEvaluation(
            passed=False,
            result_code="FAIL_DUPLICATE_FRAMES",
            anti_spoofing_passed=False,
            anti_spoofing_score=0.0,
            face_matched=False,
            similarity=None,
            threshold=settings.face_match_threshold,
            action_results=action_results,
            live_embedding=None,
            message="摄像头帧重复度过高",
        )

    liveness_action_results = verify_liveness_actions(frames_by_action, expected_actions, settings)
    action_passed = all(item["passed"] for item in liveness_action_results)
    action_results.extend(liveness_action_results)
    if not action_passed:
        return VerificationEvaluation(
            passed=False,
            result_code="FAIL_ACTION",
            anti_spoofing_passed=False,
            anti_spoofing_score=0.0,
            face_matched=False,
            similarity=None,
            threshold=settings.face_match_threshold,
            action_results=action_results,
            live_embedding=None,
            message="动作检测失败",
        )

    deep_check_frames = _select_deep_check_frames(frames_by_action, expected_actions, settings)
    pose_check_frames = _select_pose_quality_frames(frames_by_action, expected_actions, settings)
    pose_quality = _validate_pose_quality(
        pose_check_frames,
        settings,
        "姿态质量",
        settings.image_quality_live_max_abs_yaw,
        settings.image_quality_live_max_abs_pitch,
    )
    action_results.append(pose_quality)
    if not pose_quality["passed"]:
        return VerificationEvaluation(
            passed=False,
            result_code="FAIL_FACE_QUALITY",
            anti_spoofing_passed=False,
            anti_spoofing_score=0.0,
            face_matched=False,
            similarity=None,
            threshold=settings.face_match_threshold,
            action_results=action_results,
            live_embedding=None,
            message="摄像头人脸姿态不达标",
        )

    live_face_check = _analyze_live_faces(
        deep_check_frames,
        settings,
        sample_count=len(deep_check_frames),
    )
    action_results.append(
        {
            "action": "face_consistency",
            "label": "人脸稳定性",
            "passed": live_face_check.passed,
            "score": round(live_face_check.consistency_score, 4),
            "detail": live_face_check.detail,
        }
    )

    if live_face_check.embedding is None or live_face_check.bbox is None:
        result_code = "FAIL_MULTI_FACE" if "多张人脸" in live_face_check.detail else "FAIL_FACE_QUALITY"
        return VerificationEvaluation(
            passed=False,
            result_code=result_code,
            anti_spoofing_passed=False,
            anti_spoofing_score=0.0,
            face_matched=False,
            similarity=None,
            threshold=settings.face_match_threshold,
            action_results=action_results,
            live_embedding=None,
            message=live_face_check.detail or "摄像头画面未检测到人脸",
        )

    face_bbox_quality = _validate_live_face_bbox_quality(
        deep_check_frames,
        live_face_check.bboxes,
        settings,
    )
    action_results.append(face_bbox_quality)
    if not face_bbox_quality["passed"]:
        return VerificationEvaluation(
            passed=False,
            result_code="FAIL_FACE_QUALITY",
            anti_spoofing_passed=False,
            anti_spoofing_score=0.0,
            face_matched=False,
            similarity=None,
            threshold=settings.face_match_threshold,
            action_results=action_results,
            live_embedding=None,
            message="摄像头人脸画面质量不达标",
        )

    try:
        spoof_result = anti_spoofing_model.predict(
            deep_check_frames,
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

    face_match = _match_live_face(reference_embedding, live_face_check.embeddings, settings)
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
        live_face_check.passed
        and face_bbox_quality["passed"]
        and spoof_result.passed
        and face_match["passed"]
    )

    if liveness_passed:
        result_code = "PASS"
        message = "活体检测通过，且为本人"
    elif not live_face_check.passed:
        result_code = "FAIL_FACE_INCONSISTENT"
        message = "人脸稳定性检测失败"
    elif not face_bbox_quality["passed"]:
        result_code = "FAIL_FACE_QUALITY"
        message = "摄像头人脸画面质量不达标"
    elif not spoof_result.passed:
        result_code = "FAIL_PAD"
        message = "防翻拍检测失败"
    elif not face_match["passed"]:
        result_code = "FAIL_FACE_MISMATCH"
        message = "人脸比对失败，不是同一个人"
    else:
        result_code = "FAIL_PROTOCOL"
        message = "活体检测失败"

    return VerificationEvaluation(
        passed=liveness_passed,
        result_code=result_code,
        anti_spoofing_passed=spoof_result.passed,
        anti_spoofing_score=round(spoof_result.live_score, 4),
        face_matched=face_match["passed"],
        similarity=face_match["similarity"],
        threshold=settings.face_match_threshold,
        action_results=action_results,
        live_embedding=live_face_check.embedding if liveness_passed else None,
        message=message,
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


def _validate_sampled_image_quality(frames: list, settings: Settings, label: str) -> dict:
    if not frames:
        return {
            "action": "image_quality",
            "label": label,
            "passed": False,
            "score": 0.0,
            "detail": "没有可检测的图像",
        }

    indexes = _evenly_spaced_indexes(
        len(frames),
        min(settings.image_quality_sample_frames, len(frames)),
    )
    summaries = [_measure_image_quality(frames[index], settings) for index in indexes]
    passed_items = [item for item in summaries if item.passed]
    pass_ratio = len(passed_items) / max(len(summaries), 1)
    passed = pass_ratio >= 0.67
    laplacians = [item.score for item in summaries]
    brightness_values = [_gray_mean(frames[index]) for index in indexes]
    contrast_values = [_gray_std(frames[index]) for index in indexes]
    return {
        "action": "image_quality",
        "label": label,
        "passed": passed,
        "score": round(pass_ratio, 4),
        "detail": (
            f"pass_ratio={pass_ratio:.0%}, "
            f"laplacian_median={float(np.median(laplacians)):.1f}, "
            f"brightness_median={float(np.median(brightness_values)):.1f}, "
            f"contrast_median={float(np.median(contrast_values)):.1f}, "
            f"frames={len(summaries)}"
        ),
    }


def _measure_image_quality(frame: np.ndarray, settings: Settings) -> ImageQualitySummary:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    passed = (
        laplacian >= settings.image_quality_min_laplacian
        and settings.image_quality_min_brightness <= brightness <= settings.image_quality_max_brightness
        and contrast >= settings.image_quality_min_contrast
    )
    return ImageQualitySummary(
        passed=passed,
        score=laplacian,
        detail=(
            f"laplacian={laplacian:.1f}, brightness={brightness:.1f}, "
            f"contrast={contrast:.1f}"
        ),
    )


def _gray_mean(frame: np.ndarray) -> float:
    return float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))


def _gray_std(frame: np.ndarray) -> float:
    return float(np.std(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))


def _validate_face_bbox_quality(
    bbox: list,
    frame_shape: tuple,
    min_face_ratio: float,
    settings: Settings,
    label: str,
) -> dict:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = [float(value) for value in bbox]
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    box_area = box_w * box_h
    image_area = max(float(width * height), 1.0)
    area_ratio = box_area / image_area
    clipped_x1 = min(max(x1, 0.0), float(width))
    clipped_y1 = min(max(y1, 0.0), float(height))
    clipped_x2 = min(max(x2, 0.0), float(width))
    clipped_y2 = min(max(y2, 0.0), float(height))
    clipped_area = max(0.0, clipped_x2 - clipped_x1) * max(0.0, clipped_y2 - clipped_y1)
    visible_ratio = clipped_area / max(box_area, 1.0)
    passed = (
        min_face_ratio <= area_ratio <= settings.image_quality_max_face_ratio
        and visible_ratio >= 0.90
    )
    return {
        "action": "face_bbox_quality",
        "label": label,
        "passed": passed,
        "score": round(area_ratio, 4),
        "detail": (
            f"face_area={area_ratio:.1%}, visible={visible_ratio:.0%}, "
            f"min={min_face_ratio:.1%}, max={settings.image_quality_max_face_ratio:.0%}"
        ),
    }


def _validate_pose_quality(
    frames: list,
    settings: Settings,
    label: str,
    max_abs_yaw: float,
    max_abs_pitch: float,
) -> dict:
    if not frames:
        return {
            "action": "pose_quality",
            "label": label,
            "passed": False,
            "score": 0.0,
            "detail": "没有可检测的姿态帧",
        }
    indexes = _evenly_spaced_indexes(
        len(frames),
        min(settings.image_quality_sample_frames, len(frames)),
    )
    metrics = []
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=len(frames) == 1,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        for index in indexes:
            item = _extract_metrics(frames[index], face_mesh)
            if item is not None:
                metrics.append(item)
    if not metrics:
        return {
            "action": "pose_quality",
            "label": label,
            "passed": False,
            "score": 0.0,
            "detail": "未检测到可用于姿态判断的人脸关键点",
        }
    passed_items = [
        item for item in metrics
        if _fold_abs_pose_angle(item.yaw) <= max_abs_yaw
        and _fold_abs_pose_angle(item.pitch) <= max_abs_pitch
    ]
    pass_ratio = len(passed_items) / max(len(metrics), 1)
    yaw_values = [_fold_abs_pose_angle(item.yaw) for item in metrics]
    pitch_values = [_fold_abs_pose_angle(item.pitch) for item in metrics]
    passed = pass_ratio >= 0.67
    return {
        "action": "pose_quality",
        "label": label,
        "passed": passed,
        "score": round(pass_ratio, 4),
        "detail": (
            f"pass_ratio={pass_ratio:.0%}, "
            f"yaw_max={max(yaw_values):.1f}/{max_abs_yaw:.1f}, "
            f"pitch_max={max(pitch_values):.1f}/{max_abs_pitch:.1f}, "
            f"frames={len(metrics)}/{len(indexes)}"
        ),
    }


def _fold_abs_pose_angle(value: float) -> float:
    angle = abs(float(value)) % 360.0
    if angle > 180.0:
        angle = 360.0 - angle
    if angle > 90.0:
        angle = 180.0 - angle
    return abs(angle)


def _validate_live_face_bbox_quality(frames: list, bboxes: list, settings: Settings) -> dict:
    checks = []
    for frame, bbox in zip(frames, bboxes):
        if bbox is None:
            continue
        checks.append(
            _validate_face_bbox_quality(
                bbox,
                frame.shape,
                settings.image_quality_live_min_face_ratio,
                settings,
                "人脸画面质量",
            )
        )
    if not checks:
        return {
            "action": "face_bbox_quality",
            "label": "人脸画面质量",
            "passed": False,
            "score": 0.0,
            "detail": "没有可检测的人脸框",
        }
    pass_ratio = sum(1 for item in checks if item["passed"]) / len(checks)
    area_values = [item["score"] for item in checks]
    passed = pass_ratio >= 0.67
    return {
        "action": "face_bbox_quality",
        "label": "人脸画面质量",
        "passed": passed,
        "score": round(pass_ratio, 4),
        "detail": (
            f"pass_ratio={pass_ratio:.0%}, "
            f"face_area_median={float(np.median(area_values)):.1%}, "
            f"frames={len(checks)}"
        ),
    }


def _frame_hash(frame: np.ndarray, hash_size: int) -> tuple:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    return tuple(int(value) for value in resized.flatten())


def _preload_anti_spoofing_model() -> None:
    try:
        anti_spoofing_model.preload()
    except Exception:
        pass


def _configured_model_paths(settings: Settings) -> list:
    paths = []
    for item in settings.anti_spoofing_model_path.split(","):
        item = item.strip()
        if not item:
            continue
        path = Path(item)
        if not path.is_absolute():
            path = Path.cwd() / path
        paths.append(path)
    return paths


def _configured_insightface_model_paths(settings: Settings) -> list:
    model_root = Path(settings.insightface_root).expanduser()
    model_dir = model_root / "models" / settings.insightface_model
    return [
        model_dir / "det_10g.onnx",
        model_dir / "w600k_r50.onnx",
        model_dir / "2d106det.onnx",
        model_dir / "1k3d68.onnx",
        model_dir / "genderage.onnx",
    ]


def _select_deep_check_frames(
    frames_by_action: dict,
    expected_actions: list,
    settings: Settings,
) -> list:
    target = max(
        settings.face_match_embedding_sample,
        settings.anti_spoofing_sample_frames,
        len(expected_actions),
    )
    if not expected_actions:
        return []
    per_action = max(1, (target + len(expected_actions) - 1) // len(expected_actions))
    selected = []
    for action in expected_actions:
        frames = frames_by_action.get(action, [])
        indexes = _select_action_deep_check_indexes(action, len(frames), per_action)
        selected.extend(frames[index] for index in indexes)
    return selected


def _select_pose_quality_frames(
    frames_by_action: dict,
    expected_actions: list,
    settings: Settings,
) -> list:
    stable_actions = [
        action
        for action in expected_actions
        if action not in {"shake_head", "nod_head"}
    ]
    if not stable_actions:
        stable_actions = [
            action
            for action in expected_actions
            if action != "shake_head"
        ]
    if stable_actions:
        return _select_deep_check_frames(frames_by_action, stable_actions, settings)
    return _select_deep_check_frames(frames_by_action, expected_actions, settings)


def _select_action_deep_check_indexes(action: str, total: int, count: int) -> list:
    if total <= 0:
        return []
    if total <= count:
        return list(range(total))

    if action == "blink":
        edge_count = max(1, total // 4)
        candidates = list(range(edge_count)) + list(range(total - edge_count, total))
    else:
        margin = max(0, total // 4)
        candidates = list(range(margin, total - margin)) or list(range(total))

    if len(candidates) < count:
        candidates = list(range(total))
    return [candidates[index] for index in _evenly_spaced_indexes(len(candidates), count)]


def _extract_live_faces(frame: np.ndarray, settings: Settings) -> list:
    try:
        return face_engine.extract_faces(frame, det_size=settings.insightface_live_det_size)
    except TypeError:
        return face_engine.extract_faces(frame)


def _analyze_live_faces(
    frames: list,
    settings: Settings,
    sample_count: Optional[int] = None,
) -> LiveFaceCheck:
    total_frames = len(frames)
    sample_count = sample_count if sample_count is not None else max(settings.face_match_embedding_sample, 6)
    sample_indexes = _evenly_spaced_indexes(total_frames, sample_count)
    sample_set = set(sample_indexes)

    faces = []
    bboxes_by_frame = [None] * total_frames
    for index, frame in enumerate(frames):
        if index not in sample_set:
            continue
        frame_faces = _extract_live_faces(frame, settings)
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
