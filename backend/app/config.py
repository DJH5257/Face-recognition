from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="FACE_DEMO_")

    database_path: str = "data/face_verify.sqlite3"
    internal_api_key: str = "change-me-face-internal-key"
    cors_origins: str = "*"
    verification_session_ttl_seconds: int = 300
    proof_ttl_seconds: int = 300

    insightface_model: str = "buffalo_l"
    insightface_root: str = "~/.insightface"
    insightface_det_size: int = 480
    insightface_ctx_id: int = -1
    face_match_threshold: float = 0.42
    face_match_min_threshold: float = 0.30
    face_match_min_pass_ratio: float = 0.65
    face_match_embedding_sample: int = 6

    liveness_action_min_count: int = 1
    liveness_action_max_count: int = 3
    challenge_ttl_seconds: int = 180
    min_frames_per_action: int = 6
    max_frames_per_action: int = 18
    max_total_frames: int = 72
    min_action_duration_ms: int = 1500
    max_action_duration_ms: int = 5000
    max_frame_gap_ms: int = 900
    min_unique_frame_ratio: float = 0.65
    duplicate_frame_hash_size: int = 8
    min_face_frame_ratio: float = 0.57
    face_consistency_threshold: float = 0.42
    face_consistency_min_pass_ratio: float = 0.66
    blink_ear_threshold: float = 0.20
    mouth_mar_threshold: float = 0.34
    mouth_mar_delta_threshold: float = 0.07
    smile_delta_threshold: float = 0.031
    shake_yaw_range_threshold: float = 16.0
    nod_pitch_range_threshold: float = 10.5

    anti_spoofing_model_path: str = "models/MiniFASNetV1SE.onnx,models/MiniFASNetV2.yakhyo.onnx"
    anti_spoofing_model_scales: str = "4.0,2.7"
    anti_spoofing_threshold: float = 0.35
    anti_spoofing_min_valid_frames: int = 4
    anti_spoofing_min_passed_frame_ratio: float = 0.60
    anti_spoofing_live_class_index: int = 1
    anti_spoofing_sample_frames: int = 6

    max_upload_bytes: int = 8 * 1024 * 1024
    max_frame_base64_chars: int = 768 * 1024

    @property
    def insightface_providers(self) -> List[str]:
        return ["CPUExecutionProvider"]

    @property
    def cors_origin_list(self) -> List[str]:
        origins = [item.strip() for item in self.cors_origins.split(",") if item.strip()]
        return origins or ["*"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
