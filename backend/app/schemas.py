from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class EnrollResponse(BaseModel):
    enrollment_id: str
    bbox: List[float]
    message: str


class ChallengeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enrollment_id: Optional[str] = None


class ChallengeResponse(BaseModel):
    challenge_id: str
    actions: List[str]
    labels: Dict[str, str]


class FramePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str
    index: int = Field(..., ge=0, description="0-based frame index within the action")
    timestamp: float = Field(..., ge=0, description="Client capture timestamp in milliseconds")
    image: str = Field(
        ...,
        max_length=768 * 1024,
        description="JPEG frame as a data URL or raw base64 string",
    )


class VerifyLivenessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    challenge_id: str
    enrollment_id: Optional[str] = None
    frames: List[FramePayload] = Field(..., min_length=1, max_length=72)


class ActionResult(BaseModel):
    action: str
    label: str
    passed: bool
    score: float
    detail: str


class VerifyLivenessResponse(BaseModel):
    challenge_id: str
    liveness_passed: bool
    anti_spoofing_passed: bool
    anti_spoofing_score: float
    face_matched: bool
    similarity: Optional[float] = None
    threshold: Optional[float] = None
    action_results: List[ActionResult]
    message: str


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enrollment_id: str
    challenge_id: str


class CompareResponse(BaseModel):
    liveness_passed: bool
    similarity: float
    threshold: float
    matched: bool
    message: str


class TemplateCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject_type: str = Field(..., min_length=1, max_length=32)
    subject_id: str = Field(..., min_length=1, max_length=64)
    image: str = Field(
        ...,
        max_length=12 * 1024 * 1024,
        description="Template image as a data URL or raw base64 string",
    )
    source_type: str = Field(default="avatar", max_length=32)
    request_id: Optional[str] = Field(default=None, max_length=128)


class TemplateCreateResponse(BaseModel):
    template_id: str
    template_version: int
    subject_type: str
    subject_id: str
    status: str
    bbox: List[float]
    message: str


class VerificationSessionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: Optional[str] = Field(default=None, max_length=128)
    subject_type: str = Field(..., min_length=1, max_length=32)
    subject_id: str = Field(..., min_length=1, max_length=64)
    admin_id: Optional[str] = Field(default=None, max_length=64)
    scene: str = Field(..., min_length=1, max_length=64)
    business_event_id: str = Field(..., min_length=1, max_length=128)
    record_id: Optional[str] = Field(default=None, max_length=128)
    action: Optional[str] = Field(default=None, max_length=64)


class VerificationSessionCreateResponse(BaseModel):
    session_id: str
    upload_token: str
    actions: List[str]
    labels: Dict[str, str]
    expires_at: datetime


class VerificationSessionVerifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frames: List[FramePayload] = Field(..., min_length=1, max_length=72)


class VerificationSessionVerifyResponse(BaseModel):
    session_id: str
    passed: bool
    result_code: str
    proof_id: Optional[str] = None
    anti_spoofing_passed: bool
    anti_spoofing_score: float
    face_matched: bool
    similarity: Optional[float] = None
    threshold: Optional[float] = None
    action_results: List[ActionResult]
    message: str


class ProofIntrospectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proof_id: str = Field(..., min_length=1, max_length=128)


class ProofIntrospectResponse(BaseModel):
    valid: bool
    proof_id: str
    status: str
    result_code: str
    session_id: Optional[str] = None
    subject_type: Optional[str] = None
    subject_id: Optional[str] = None
    admin_id: Optional[str] = None
    scene: Optional[str] = None
    business_event_id: Optional[str] = None
    record_id: Optional[str] = None
    action: Optional[str] = None
    template_id: Optional[str] = None
    template_version: Optional[int] = None
    issued_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    finalized_at: Optional[datetime] = None


class ProofFinalizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    business_event_id: str = Field(..., min_length=1, max_length=128)


class ProofFinalizeResponse(BaseModel):
    proof_id: str
    finalized: bool
    status: str
