from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import django
import numpy as np
from django.apps import apps as django_apps

from ai_core.dinov2_features import extract_embeddings_from_frames, get_dino_embedder
from ai_core.inference import generate_text_from_features, warm_translation_model
from ai_core.kpe_mediapipe import video_holistic
from sign_translation_engine.runtime.bootstrap import get_model_config

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SharedRealtimeRuntime:
    config: dict
    hand_embedder: object
    face_embedder: object
    hand_lock: threading.Lock
    face_lock: threading.Lock
    decode_lock: threading.Lock

    def embed_hands(self, left_frame: np.ndarray, right_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        left_embeddings, right_embeddings = self.embed_hands_batch([left_frame], [right_frame])
        return left_embeddings[0], right_embeddings[0]

    def embed_hands_batch(
        self,
        left_frames: list[np.ndarray],
        right_frames: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(left_frames) != len(right_frames):
            raise ValueError("left_frames and right_frames must have the same length.")
        if not left_frames:
            return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

        combined_frames = list(left_frames) + list(right_frames)
        with self.hand_lock:
            embeddings = extract_embeddings_from_frames(
                combined_frames,
                self.config["dino_hands_model_path"],
                embedder=self.hand_embedder,
            )
        midpoint = len(left_frames)
        return embeddings[:midpoint], embeddings[midpoint:]

    def embed_face(self, face_frame: np.ndarray) -> np.ndarray:
        embeddings = self.embed_face_batch([face_frame])
        return embeddings[0]

    def embed_face_batch(self, face_frames: list[np.ndarray]) -> np.ndarray:
        if not face_frames:
            return np.empty((0,), dtype=np.float32)
        with self.face_lock:
            embeddings = extract_embeddings_from_frames(
                face_frames,
                self.config["dino_face_model_path"],
                embedder=self.face_embedder,
            )
        return embeddings

    def decode(
        self,
        face_embeddings: np.ndarray,
        left_hand_embeddings: np.ndarray,
        right_hand_embeddings: np.ndarray,
        pose_embeddings: np.ndarray,
        *,
        generation_num_beams: int,
        generation_max_length: int,
    ) -> str:
        with self.decode_lock:
            return generate_text_from_features(
                face_embeddings=face_embeddings,
                left_hand_embeddings=left_hand_embeddings,
                right_hand_embeddings=right_hand_embeddings,
                body_posture_embeddings=pose_embeddings,
                model_config=self.config["slt_model_config"],
                model_checkpoint=self.config["slt_model_checkpoint"],
                tokenizer_checkpoint=self.config["slt_tokenizer_checkpoint"],
                output_dir=self.config["temp_dir"],
                generation_num_beams=generation_num_beams,
                generation_max_length=generation_max_length,
            )


_RUNTIME: SharedRealtimeRuntime | None = None
_RUNTIME_LOCK = threading.Lock()
_RUNTIME_STATUS_LOCK = threading.Lock()
_RUNTIME_STATUS: dict[str, object] = {
    "status": "cold",
    "message": "Runtime chua duoc khoi dong.",
    "started_at_ms": None,
    "ready_at_ms": None,
    "error_at_ms": None,
}
_RUNTIME_WARMUP_THREAD: threading.Thread | None = None


def _set_runtime_status(status: str, message: str, *, timestamp_ms: float | None = None) -> None:
    now_ms = timestamp_ms if timestamp_ms is not None else time.time() * 1000.0
    with _RUNTIME_STATUS_LOCK:
        _RUNTIME_STATUS["status"] = status
        _RUNTIME_STATUS["message"] = message
        if status == "warming":
            _RUNTIME_STATUS["started_at_ms"] = now_ms
            _RUNTIME_STATUS["ready_at_ms"] = None
            _RUNTIME_STATUS["error_at_ms"] = None
        elif status == "ready":
            if _RUNTIME_STATUS.get("started_at_ms") is None:
                _RUNTIME_STATUS["started_at_ms"] = now_ms
            _RUNTIME_STATUS["ready_at_ms"] = now_ms
            _RUNTIME_STATUS["error_at_ms"] = None
        elif status == "error":
            if _RUNTIME_STATUS.get("started_at_ms") is None:
                _RUNTIME_STATUS["started_at_ms"] = now_ms
            _RUNTIME_STATUS["error_at_ms"] = now_ms


def get_runtime_status() -> dict[str, object]:
    global _RUNTIME
    with _RUNTIME_STATUS_LOCK:
        snapshot = dict(_RUNTIME_STATUS)
    snapshot["ready"] = _RUNTIME is not None
    return snapshot


def _warm_runtime_background() -> None:
    try:
        get_shared_realtime_runtime()
    except Exception:
        logger.exception("Realtime runtime | background warm failed")


def start_runtime_warmup() -> bool:
    global _RUNTIME_WARMUP_THREAD
    with _RUNTIME_STATUS_LOCK:
        if _RUNTIME is not None:
            return False
        if _RUNTIME_WARMUP_THREAD is not None and _RUNTIME_WARMUP_THREAD.is_alive():
            return False
        _RUNTIME_WARMUP_THREAD = threading.Thread(
            target=_warm_runtime_background,
            name="realtime-runtime-warmup",
            daemon=True,
        )
        _RUNTIME_WARMUP_THREAD.start()
        return True


def get_shared_realtime_runtime() -> SharedRealtimeRuntime:
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME

    with _RUNTIME_LOCK:
        if _RUNTIME is not None:
            return _RUNTIME

        _set_runtime_status("warming", "Dang tai va lam nong mo hinh realtime...")
        if not django_apps.ready:
            logger.info("Realtime runtime | django setup start")
            django.setup()
            logger.info("Realtime runtime | django setup done")
        logger.info("Realtime runtime | model config init start")
        try:
            config = get_model_config()

            logger.info("Realtime runtime | warm detector start")
            video_holistic(
                video_input=[],
                face_model_path=config["mediapipe_face_model_path"],
                hand_model_path=config["mediapipe_hands_model_path"],
            )
            logger.info("Realtime runtime | warm detector done")

            logger.info("Realtime runtime | warm hand embedder start")
            hand_embedder = get_dino_embedder(config["dino_hands_model_path"])
            logger.info("Realtime runtime | warm hand embedder done")

            logger.info("Realtime runtime | warm face embedder start")
            face_embedder = get_dino_embedder(config["dino_face_model_path"])
            logger.info("Realtime runtime | warm face embedder done")

            logger.info("Realtime runtime | warm translation model start")
            warm_translation_model(
                model_config=config["slt_model_config"],
                model_checkpoint=config["slt_model_checkpoint"],
                tokenizer_checkpoint=config["slt_tokenizer_checkpoint"],
                output_dir=config["temp_dir"],
            )
            logger.info("Realtime runtime | warm translation model done")

            _RUNTIME = SharedRealtimeRuntime(
                config=config,
                hand_embedder=hand_embedder,
                face_embedder=face_embedder,
                hand_lock=threading.Lock(),
                face_lock=threading.Lock(),
                decode_lock=threading.Lock(),
            )
            _set_runtime_status("ready", "Runtime realtime da san sang.")
            return _RUNTIME
        except Exception as exc:
            _set_runtime_status("error", f"Loi runtime: {exc}")
            raise
