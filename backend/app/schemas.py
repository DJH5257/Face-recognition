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
