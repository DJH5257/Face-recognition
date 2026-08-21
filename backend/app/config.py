from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="FACE_DEMO_")

    database_path: str = "data/face_verify.sqlite3"
    internal_api_key: str = "change-me-face-internal-key"
    cors_origins: str = "*"
    verification_session_ttl_seconds: int = 300
    verification_processing_timeout_seconds: int = 90
    proof_ttl_seconds: int = 300
    template_encryption_key: str = ""
    template_encryption_required: bool = False

    # Optional writers for existing verification result tables.
    regulator_db_enabled: bool = False
    regulator_db_dsn: str = ""
    regulator_db_table: str = "fa_face_verify_regulator_status"
    regulator_db_connect_timeout_seconds: int = 3
    regulator_db_read_timeout_seconds: int = 5
    regulator_db_write_timeout_seconds: int = 5
    doctor_face_log_enabled: bool = False
    doctor_face_log_table: str = "fa_doctor_face_verify_log"

    insightface_model: str = "buffalo_l"
    insightface_root: str = "~/.insightface"
    insightface_det_size: int = 480
    insightface_live_det_size: int = 320
    insightface_ctx_id: int = -1
    insightface_max_concurrent_inferences: int = 2
    face_match_threshold: float = 0.42
    face_match_min_threshold: float = 0.30
    face_match_min_pass_ratio: float = 0.65
    face_match_embedding_sample: int = 5

    liveness_action_min_count: int = 1
    liveness_action_max_count: int = 2
    challenge_ttl_seconds: int = 180
    min_frames_per_action: int = 6
    max_frames_per_action: int = 48
    max_total_frames: int = 90
    min_action_duration_ms: int = 1500
    max_action_duration_ms: int = 5000
    max_frame_gap_ms: int = 900
    min_unique_frame_ratio: float = 0.55
    duplicate_frame_hash_size: int = 12
    min_face_frame_ratio: float = 0.57
    face_consistency_threshold: float = 0.42
    face_consistency_min_pass_ratio: float = 0.66
    blink_ear_threshold: float = 0.20
    mouth_mar_threshold: float = 0.34
    mouth_mar_delta_threshold: float = 0.07
    smile_delta_threshold: float = 0.028
    shake_yaw_range_threshold: float = 14.0
    nod_pitch_range_threshold: float = 8.0

    anti_spoofing_model_path: str = "models/MiniFASNetV1SE.onnx,models/MiniFASNetV2.yakhyo.onnx"
    anti_spoofing_model_scales: str = "4.0,2.7"
    anti_spoofing_threshold: float = 0.35
    anti_spoofing_min_valid_frames: int = 4
    anti_spoofing_min_passed_frame_ratio: float = 0.60
    anti_spoofing_live_class_index: int = 1
    anti_spoofing_sample_frames: int = 6
    anti_spoofing_max_concurrent_inferences: int = 2

    verification_max_concurrent_requests: int = 10
    verification_slot_wait_seconds: float = 3.0
    verification_metrics_window: int = 200
    verification_failure_limit_enabled: bool = True
    verification_failure_max_attempts: int = 5
    verification_failure_window_seconds: int = 300
    verification_failure_cooldown_seconds: int = 180

    image_quality_sample_frames: int = 6
    image_quality_min_laplacian: float = 12.0
    image_quality_min_brightness: float = 35.0
    image_quality_max_brightness: float = 225.0
    image_quality_min_contrast: float = 10.0
    image_quality_template_min_face_ratio: float = 0.025
    image_quality_live_min_face_ratio: float = 0.02
    image_quality_max_face_ratio: float = 0.92
    image_quality_template_max_abs_yaw: float = 35.0
    image_quality_template_max_abs_pitch: float = 35.0
    image_quality_live_max_abs_yaw: float = 55.0
    image_quality_live_max_abs_pitch: float = 55.0

    liveness_action_weights: str = "mouth_open=24,shake_head=24,nod_head=24,blink=18,smile=18"
    liveness_action_recent_failure_window: int = 2
    liveness_action_recent_failure_factor: float = 0.5

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
