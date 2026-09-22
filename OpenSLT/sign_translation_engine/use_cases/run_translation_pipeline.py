import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import decord

from ai_core.body_features import process_pose_landmarks
from ai_core.crop_face import FaceExtractor
from ai_core.crop_hands import HandExtractor
from ai_core.dinov2_features import extract_embeddings_from_frames, get_dino_embedder
from ai_core.inference import test
from ai_core.kpe_mediapipe import video_holistic

from sign_translation_engine.artifacts.video_outputs import (
    render_landmarks_frames,
    render_subtitle_frames,
    write_video,
)
from sign_translation_engine.runtime.bootstrap import get_model_config

logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    text: str
    landmarks_video: Path | None = None
    left_hand_video: Path | None = None
    right_hand_video: Path | None = None
    face_video: Path | None = None
    subtitle_video: Path | None = None


class TranslationEngine:
    def __init__(self):
        self.config = get_model_config()
        self.hand_extractor = HandExtractor()
        self.face_extractor = FaceExtractor()
        self.hand_embedder = None
        self.face_embedder = None

    def _ensure_embedder_cache(self) -> None:
        if self.hand_embedder is None:
            self.hand_embedder = get_dino_embedder(self.config["dino_hands_model_path"])
        if self.face_embedder is None:
            self.face_embedder = get_dino_embedder(self.config["dino_face_model_path"])

    def _load_video_frames(self, video_path: Path):
        reader = decord.VideoReader(str(video_path))
        original_fps = reader.get_avg_fps()
        stride = max(1, int(round(original_fps / 15.0)))
        indices = list(range(0, len(reader), stride))
        frames = reader.get_batch(indices).asnumpy()
        sampled_fps = original_fps / stride if stride > 0 else 15.0
        return frames, sampled_fps

    def process(
        self,
        video_path: Path,
        output_dir: Path,
        include_landmarks_video: bool,
        include_crop_videos: bool,
        job_id: int | None = None,
        progress_callback=None,
    ) -> TranslationResult:
        def emit(stage: str, **details) -> None:
            detail_text = " | ".join(f"{key}={value}" for key, value in details.items())
            prefix = f"job={job_id} | " if job_id is not None else ""
            suffix = f" | {detail_text}" if detail_text else ""
            logger.info("Pipeline | %sstage=%s%s", prefix, stage, suffix)
            if progress_callback is not None:
                progress_callback(stage, **details)

        overall_start = time.perf_counter()

        stage_start = time.perf_counter()
        frames, sampled_fps = self._load_video_frames(video_path)
        emit(
            "video_loaded",
            elapsed_seconds=round(time.perf_counter() - stage_start, 3),
            sampled_frames=len(frames),
            sampled_fps=round(sampled_fps, 3),
        )

        stage_start = time.perf_counter()
        landmarks, landmark_timing = video_holistic(
            video_input=frames,
            face_model_path=self.config["mediapipe_face_model_path"],
            hand_model_path=self.config["mediapipe_hands_model_path"],
            return_timing=True,
        )
        emit(
            "landmarks_extracted",
            elapsed_seconds=round(time.perf_counter() - stage_start, 3),
            **landmark_timing,
        )

        stage_start = time.perf_counter()
        self._ensure_embedder_cache()
        emit("embedder_cache_ready", elapsed_seconds=round(time.perf_counter() - stage_start, 3))

        landmarks_video_path = None
        if include_landmarks_video:
            stage_start = time.perf_counter()
            landmarks_video_path = output_dir / "landmarks_overlay.mp4"
            rendered_frames = render_landmarks_frames(frames, landmarks)
            write_video(rendered_frames, landmarks_video_path, fps=sampled_fps)
            emit(
                "landmarks_video_written",
                elapsed_seconds=round(time.perf_counter() - stage_start, 3),
                path=str(landmarks_video_path),
            )

        stage_start = time.perf_counter()
        left_hand_frames, right_hand_frames = self.hand_extractor.extract_hand_frames(frames, landmarks)
        face_frames = self.face_extractor.extract_face_frames(frames, landmarks)
        emit(
            "crop_frames_ready",
            elapsed_seconds=round(time.perf_counter() - stage_start, 3),
            left_frames=len(left_hand_frames),
            right_frames=len(right_hand_frames),
            face_frames=len(face_frames),
        )

        left_hand_video_path = None
        right_hand_video_path = None
        face_video_path = None
        if include_crop_videos:
            stage_start = time.perf_counter()
            left_hand_video_path = output_dir / "left_hand.mp4"
            right_hand_video_path = output_dir / "right_hand.mp4"
            face_video_path = output_dir / "face_cues.mp4"
            write_video(left_hand_frames, left_hand_video_path, fps=sampled_fps)
            write_video(right_hand_frames, right_hand_video_path, fps=sampled_fps)
            write_video(face_frames, face_video_path, fps=sampled_fps)
            emit(
                "crop_videos_written",
                elapsed_seconds=round(time.perf_counter() - stage_start, 3),
                left_hand_path=str(left_hand_video_path),
                right_hand_path=str(right_hand_video_path),
                face_path=str(face_video_path),
            )

        hand_frames = left_hand_frames + right_hand_frames
        stage_start = time.perf_counter()
        hand_embeddings = extract_embeddings_from_frames(
            hand_frames,
            self.config["dino_hands_model_path"],
            embedder=self.hand_embedder,
        )
        left_count = len(left_hand_frames)
        left_hand_embeddings = hand_embeddings[:left_count]
        right_hand_embeddings = hand_embeddings[left_count:]
        emit(
            "hand_embeddings_ready",
            elapsed_seconds=round(time.perf_counter() - stage_start, 3),
            total_hand_frames=len(hand_frames),
        )

        stage_start = time.perf_counter()
        face_embeddings = extract_embeddings_from_frames(
            face_frames,
            self.config["dino_face_model_path"],
            embedder=self.face_embedder,
        )
        emit("face_embeddings_ready", elapsed_seconds=round(time.perf_counter() - stage_start, 3))

        stage_start = time.perf_counter()
        pose_embeddings = process_pose_landmarks(landmarks)
        emit("pose_embeddings_ready", elapsed_seconds=round(time.perf_counter() - stage_start, 3))

        stage_start = time.perf_counter()
        text = test(
            face_embeddings=face_embeddings,
            left_hand_embeddings=left_hand_embeddings,
            right_hand_embeddings=right_hand_embeddings,
            body_posture_embeddings=pose_embeddings,
            model_config=self.config["slt_model_config"],
            model_checkpoint=self.config["slt_model_checkpoint"],
            tokenizer_checkpoint=self.config["slt_tokenizer_checkpoint"],
            output_dir=self.config["temp_dir"],
        )
        emit(
            "translation_ready",
            elapsed_seconds=round(time.perf_counter() - stage_start, 3),
            total_elapsed_seconds=round(time.perf_counter() - overall_start, 3),
            text_length=len(text),
        )

        subtitle_video_path = None
        subtitle_frames = render_subtitle_frames(frames, text, reveal_fraction=0.75)
        if subtitle_frames:
            stage_start = time.perf_counter()
            subtitle_video_path = output_dir / "subtitle_video.mp4"
            write_video(subtitle_frames, subtitle_video_path, fps=sampled_fps)
            emit(
                "subtitle_video_written",
                elapsed_seconds=round(time.perf_counter() - stage_start, 3),
                path=str(subtitle_video_path),
                reveal_fraction=0.75,
                word_count=len(text.split()),
            )
        else:
            emit(
                "subtitle_video_skipped",
                reason="no_text_tokens",
                text_length=len(text),
            )

        return TranslationResult(
            text=text,
            landmarks_video=landmarks_video_path,
            left_hand_video=left_hand_video_path,
            right_hand_video=right_hand_video_path,
            face_video=face_video_path,
            subtitle_video=subtitle_video_path,
        )


_ENGINE = None
_ENGINE_LOCK = threading.Lock()


def get_translation_engine():
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = TranslationEngine()
    return _ENGINE
