from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class RealtimeConfig:
    pipeline_mode: str = "split"
    raw_worker_count: int = 2
    camera_input_fps: int = 30
    target_fps: int = 15
    frame_width: int = 640
    frame_height: int = 480
    jpeg_quality: int = 72
    chunk_frames: int = 15
    chunk_queue_max: int = 2
    prepared_queue_max: int = 2
    recent_window_frames: int = 16
    max_utterance_seconds: float = 20.0
    finalized_hold_seconds: float = 10.0
    preview_initial_frames: int = 30
    preview_stride_frames: int = 15
    preview_num_beams: int = 1
    preview_max_length: int = 96
    final_num_beams: int = 2
    final_max_length: int = 128
    endpoint_min_utterance_ms: int = 600
    endpoint_resume_grace_ms: int = 500
    endpoint_finalize_pause_ms: int = 1500
    endpoint_scene_absence_ms: int = 2000
    endpoint_false_start_reset_ms: int = 1200
    endpoint_hand_motion_threshold: float = 0.004
    endpoint_upper_body_motion_threshold: float = 0.018
    write_stable_repeats: int = 2
    write_min_tokens: int = 1
    session_idle_timeout_ms: int = 120_000

    @property
    def max_utterance_frames(self) -> int:
        return max(1, int(round(self.target_fps * self.max_utterance_seconds)))

    @classmethod
    def from_env(cls) -> "RealtimeConfig":
        target_fps = int(os.environ.get("SIGN_TRANSLATION_REALTIME_TARGET_FPS", "15"))
        legacy_max_frames = os.environ.get("SIGN_TRANSLATION_REALTIME_MAX_UTTERANCE_FRAMES")
        if "SIGN_TRANSLATION_REALTIME_MAX_UTTERANCE_SECONDS" in os.environ:
            max_utterance_seconds = float(
                os.environ["SIGN_TRANSLATION_REALTIME_MAX_UTTERANCE_SECONDS"]
            )
        elif legacy_max_frames is not None:
            max_utterance_seconds = max(1.0, float(legacy_max_frames) / max(target_fps, 1))
        else:
            max_utterance_seconds = 20.0

        return cls(
            pipeline_mode="split",
            raw_worker_count=max(1, int(os.environ.get("SIGN_TRANSLATION_REALTIME_RAW_WORKERS", "2"))),
            camera_input_fps=int(os.environ.get("SIGN_TRANSLATION_REALTIME_CAMERA_INPUT_FPS", "30")),
            target_fps=target_fps,
            frame_width=int(os.environ.get("SIGN_TRANSLATION_REALTIME_FRAME_WIDTH", "640")),
            frame_height=int(os.environ.get("SIGN_TRANSLATION_REALTIME_FRAME_HEIGHT", "480")),
            jpeg_quality=int(os.environ.get("SIGN_TRANSLATION_REALTIME_JPEG_QUALITY", "72")),
            chunk_frames=int(os.environ.get("SIGN_TRANSLATION_REALTIME_CHUNK_FRAMES", "15")),
            chunk_queue_max=int(os.environ.get("SIGN_TRANSLATION_REALTIME_CHUNK_QUEUE_MAX", "2")),
            prepared_queue_max=int(os.environ.get("SIGN_TRANSLATION_REALTIME_PREPARED_QUEUE_MAX", "2")),
            recent_window_frames=int(os.environ.get("SIGN_TRANSLATION_REALTIME_WINDOW_FRAMES", "16")),
            max_utterance_seconds=max_utterance_seconds,
            finalized_hold_seconds=float(
                os.environ.get("SIGN_TRANSLATION_REALTIME_FINALIZED_HOLD_SECONDS", "10")
            ),
            preview_initial_frames=int(os.environ.get("SIGN_TRANSLATION_REALTIME_PREVIEW_INITIAL_FRAMES", "30")),
            preview_stride_frames=int(os.environ.get("SIGN_TRANSLATION_REALTIME_PREVIEW_STRIDE_FRAMES", "15")),
            preview_num_beams=int(os.environ.get("SIGN_TRANSLATION_REALTIME_NUM_BEAMS", "1")),
            preview_max_length=int(os.environ.get("SIGN_TRANSLATION_REALTIME_MAX_LENGTH", "96")),
            final_num_beams=int(os.environ.get("SIGN_TRANSLATION_REALTIME_FINAL_NUM_BEAMS", "2")),
            final_max_length=int(os.environ.get("SIGN_TRANSLATION_REALTIME_FINAL_MAX_LENGTH", "128")),
            endpoint_min_utterance_ms=int(
                os.environ.get("SIGN_TRANSLATION_REALTIME_ENDPOINT_MIN_UTTERANCE_MS", "600")
            ),
            endpoint_resume_grace_ms=int(
                os.environ.get("SIGN_TRANSLATION_REALTIME_ENDPOINT_RESUME_GRACE_MS", "500")
            ),
            endpoint_finalize_pause_ms=int(
                os.environ.get("SIGN_TRANSLATION_REALTIME_ENDPOINT_FINALIZE_PAUSE_MS", "1500")
            ),
            endpoint_scene_absence_ms=int(
                os.environ.get("SIGN_TRANSLATION_REALTIME_ENDPOINT_SCENE_ABSENCE_MS", "2000")
            ),
            endpoint_false_start_reset_ms=int(
                os.environ.get("SIGN_TRANSLATION_REALTIME_ENDPOINT_FALSE_START_RESET_MS", "1200")
            ),
            endpoint_hand_motion_threshold=float(
                os.environ.get("SIGN_TRANSLATION_REALTIME_ENDPOINT_HAND_MOTION_THRESHOLD", "0.004")
            ),
            endpoint_upper_body_motion_threshold=float(
                os.environ.get(
                    "SIGN_TRANSLATION_REALTIME_ENDPOINT_UPPER_BODY_MOTION_THRESHOLD",
                    "0.018",
                )
            ),
            write_stable_repeats=int(os.environ.get("SIGN_TRANSLATION_REALTIME_WRITE_STABLE_REPEATS", "2")),
            write_min_tokens=int(os.environ.get("SIGN_TRANSLATION_REALTIME_WRITE_MIN_TOKENS", "1")),
            session_idle_timeout_ms=int(
                os.environ.get("SIGN_TRANSLATION_REALTIME_IDLE_TIMEOUT_MS", "120000")
            ),
        )
