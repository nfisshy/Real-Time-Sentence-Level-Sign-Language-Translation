from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ai_core.body_features import PoseProcessor
from ai_core.crop_face import FaceExtractor
from ai_core.crop_hands import HandExtractor
from ai_core.kpe_mediapipe import HolisticDetector

from .config import RealtimeConfig
from .endpoint import EndpointDecision, EndpointDetector
from .runtime import SharedRealtimeRuntime
from .text_stabilizer import TextStabilizer


@dataclass(slots=True)
class PendingChunk:
    chunk_index: int
    start_ts_ms: float
    end_ts_ms: float
    frame_bytes_list: list[bytes]
    enqueued_at_perf: float

    def extend(self, other: "PendingChunk") -> None:
        self.frame_bytes_list.extend(other.frame_bytes_list)
        self.end_ts_ms = other.end_ts_ms


@dataclass(slots=True)
class PreparedEvent:
    kind: str
    left_inputs: list[np.ndarray] | None = None
    right_inputs: list[np.ndarray] | None = None
    face_inputs: list[np.ndarray] | None = None
    pose_embeddings: list[np.ndarray] | None = None
    payload_status: str | None = None
    payload_status_label: str | None = None
    reason_label: str | None = None


@dataclass(slots=True)
class PreparedChunk:
    chunk_index: int
    start_ts_ms: float
    end_ts_ms: float
    events: list[PreparedEvent]
    mediapipe_elapsed_seconds: float
    enqueued_at_perf: float = 0.0


@dataclass(slots=True)
class AnalyzedFrame:
    frame_timestamp_ms: int
    frame_landmarks: dict[str, Any] | None
    left_crop: np.ndarray | None
    right_crop: np.ndarray | None
    face_crop: np.ndarray | None
    pose_embedding: np.ndarray | None
    hand_present: bool
    face_present: bool
    pose_present: bool
    pose_points: np.ndarray | None
    left_hand_points: np.ndarray | None
    right_hand_points: np.ndarray | None


@dataclass(slots=True)
class AnalyzedChunk:
    chunk_index: int
    start_ts_ms: float
    end_ts_ms: float
    frames: list[AnalyzedFrame]
    mediapipe_elapsed_seconds: float


@dataclass(slots=True)
class PrepareWorkerContext:
    detector: HolisticDetector
    hand_extractor: HandExtractor
    face_extractor: FaceExtractor
    pose_processor: PoseProcessor


class RealtimeSession:
    def _create_prepare_context(self) -> PrepareWorkerContext:
        return PrepareWorkerContext(
            detector=HolisticDetector(
                face_model_path=self.runtime.config["mediapipe_face_model_path"],
                hand_model_path=self.runtime.config["mediapipe_hands_model_path"],
            ),
            hand_extractor=HandExtractor(),
            face_extractor=FaceExtractor(),
            pose_processor=PoseProcessor(),
        )

    def __init__(self, runtime: SharedRealtimeRuntime, config: RealtimeConfig):
        self.runtime = runtime
        self.config = config
        self._parallel_prepare_enabled = self.config.raw_worker_count > 1
        self._prepare_contexts = [self._create_prepare_context() for _ in range(max(1, self.config.raw_worker_count))]
        self.detector = self._prepare_contexts[0].detector
        self.hand_extractor = self._prepare_contexts[0].hand_extractor
        self.face_extractor = self._prepare_contexts[0].face_extractor
        self.pose_processor = self._prepare_contexts[0].pose_processor
        self.stabilizer = TextStabilizer()
        self.endpoint_detector = EndpointDetector(
            target_fps=config.target_fps,
            min_utterance_ms=config.endpoint_min_utterance_ms,
            resume_grace_ms=config.endpoint_resume_grace_ms,
            finalize_pause_ms=config.endpoint_finalize_pause_ms,
            scene_absence_ms=config.endpoint_scene_absence_ms,
            false_start_reset_ms=config.endpoint_false_start_reset_ms,
            hand_motion_threshold=config.endpoint_hand_motion_threshold,
            upper_body_motion_threshold=config.endpoint_upper_body_motion_threshold,
        )

        self.recent_left_hand_embeddings: deque[np.ndarray] = deque(maxlen=config.recent_window_frames)
        self.recent_right_hand_embeddings: deque[np.ndarray] = deque(maxlen=config.recent_window_frames)
        self.recent_face_embeddings: deque[np.ndarray] = deque(maxlen=config.recent_window_frames)
        self.recent_pose_embeddings: deque[np.ndarray] = deque(maxlen=config.recent_window_frames)

        self.utterance_left_hand_embeddings: deque[np.ndarray] = deque(maxlen=config.max_utterance_frames)
        self.utterance_right_hand_embeddings: deque[np.ndarray] = deque(maxlen=config.max_utterance_frames)
        self.utterance_face_embeddings: deque[np.ndarray] = deque(maxlen=config.max_utterance_frames)
        self.utterance_pose_embeddings: deque[np.ndarray] = deque(maxlen=config.max_utterance_frames)

        self.transcript_segments: list[str] = []
        self.current_draft_text = ""

        self.last_decode_ts_ms = 0.0
        self.last_activity_ts_ms = time.time() * 1000.0
        self.session_started_ts_ms = self.last_activity_ts_ms
        self._finalized_hold_deadline_ms: float | None = None
        self.first_draft_ts_ms: float | None = None
        self.first_final_ts_ms: float | None = None
        self.ingest_parse_seconds_total = 0.0
        self.ingest_enqueue_seconds_total = 0.0
        self.ingest_endpoint_seconds_total = 0.0
        self.raw_queue_wait_seconds_total = 0.0
        self.prepared_queue_wait_seconds_total = 0.0
        self.prepared_enqueue_wait_seconds_total = 0.0
        self.jpeg_decode_seconds_total = 0.0
        self.rgb_convert_seconds_total = 0.0
        self.crop_pose_seconds_total = 0.0
        self.mediapipe_chunk_seconds_total = 0.0
        self.gpu_chunk_seconds_total = 0.0
        self.hand_embed_seconds_total = 0.0
        self.face_embed_seconds_total = 0.0
        self.preview_decode_seconds_total = 0.0
        self.final_decode_seconds_total = 0.0
        self.last_payload: dict[str, object] = {
            "status": "idle",
            "status_label": "San sang bat dau",
            "display_text": "",
            "draft_text": "",
            "final_text": "",
            "finalized_hold_countdown_ms": None,
            "updated_at_ms": self.session_started_ts_ms,
        }

        blank_shape = (*self.hand_extractor.output_size, 3)
        self.blank_hand_frame = np.zeros(blank_shape, dtype=np.uint8)
        self.blank_face_frame = np.zeros((*self.face_extractor.output_size, 3), dtype=np.uint8)

        self.counters: dict[str, int] = {
            "chunks_received": 0,
            "chunks_coalesced": 0,
            "chunks_processed": 0,
            "prepared_chunks_enqueued": 0,
            "prepared_chunks_processed": 0,
            "preview_decode_count": 0,
            "final_decode_count": 0,
            "decode_wait_count": 0,
            "frames_received_total": 0,
            "frames_processed_total": 0,
            "frames_collectable_total": 0,
            "frames_not_collectable_total": 0,
            "preview_skipped_due_to_backpressure": 0,
            "preview_skipped_due_to_gpu_backpressure": 0,
        }

        self._state_lock = threading.RLock()
        self._raw_queue_condition = threading.Condition(self._state_lock)
        self._prepared_queue_condition = threading.Condition(self._state_lock)
        self._raw_chunk_queue: deque[PendingChunk] = deque()
        self._prepared_chunk_queue: deque[PreparedChunk] = deque()
        self._analyzed_chunks_by_index: dict[int, AnalyzedChunk] = {}
        self._next_analyzed_chunk_index = 0
        self._worker_should_stop = False
        self._closed = False
        self._decode_dirty = False
        self._next_preview_frame_target = self.config.preview_initial_frames
        self._raw_workers_active = 0
        self._raw_queue_depth_peak = 0
        self._prepared_queue_depth_peak = 0

        self._raw_workers = [
            threading.Thread(
                target=self._split_raw_worker_loop,
                args=(worker_index,),
                name=f"realtime-raw-{worker_index}-{id(self)}",
                daemon=True,
            )
            for worker_index in range(max(1, self.config.raw_worker_count))
        ]
        self._gpu_worker = threading.Thread(
            target=self._split_gpu_worker_loop,
            name=f"realtime-gpu-{id(self)}",
            daemon=True,
        )
        for raw_worker in self._raw_workers:
            raw_worker.start()
        self._gpu_worker.start()

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._worker_should_stop = True
            self._raw_chunk_queue.clear()
            self._prepared_chunk_queue.clear()
            self._analyzed_chunks_by_index.clear()
            self._raw_queue_condition.notify_all()
            self._prepared_queue_condition.notify_all()

        for raw_worker in self._raw_workers:
            if raw_worker.is_alive():
                raw_worker.join(timeout=20)
        if self._gpu_worker is not None and self._gpu_worker.is_alive():
            self._gpu_worker.join(timeout=20)
        for context in self._prepare_contexts:
            context.detector.close()

    def _extract_pose_points(self, frame_landmarks: dict[str, Any] | None) -> np.ndarray | None:
        if frame_landmarks is None:
            return None
        pose_landmarks = frame_landmarks.get("pose_landmarks")
        if not pose_landmarks:
            return None
        pose = pose_landmarks[0]
        pose_indices = self.pose_processor.pose_indices
        return np.array([[pose[index][0], pose[index][1]] for index in pose_indices], dtype=np.float32)

    def _extract_hand_points(
        self,
        *,
        frame_landmarks: dict[str, Any] | None,
        hand_extractor: HandExtractor,
        image_shape: tuple[int, int, int],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if frame_landmarks is None or frame_landmarks.get("pose_landmarks") is None:
            return None, None

        left_hand_landmarks, right_hand_landmarks = hand_extractor.select_hands(
            frame_landmarks["pose_landmarks"][0],
            frame_landmarks.get("hand_landmarks"),
            image_shape,
        )

        left_points = None
        if left_hand_landmarks is not None:
            left_points = np.array(
                [[point[0], point[1]] for point in left_hand_landmarks],
                dtype=np.float32,
            )

        right_points = None
        if right_hand_landmarks is not None:
            right_points = np.array(
                [[point[0], point[1]] for point in right_hand_landmarks],
                dtype=np.float32,
            )

        return left_points, right_points

    def _build_endpoint_decision(
        self,
        *,
        frame_timestamp_ms: int,
        frame_landmarks: dict[str, Any] | None,
        left_crop: np.ndarray | None,
        right_crop: np.ndarray | None,
        face_crop: np.ndarray | None,
        left_hand_points: np.ndarray | None,
        right_hand_points: np.ndarray | None,
    ) -> EndpointDecision:
        pose_points = self._extract_pose_points(frame_landmarks)
        body_present = pose_points is not None
        return self.endpoint_detector.update(
            frame_timestamp_ms=frame_timestamp_ms,
            hand_present=left_crop is not None or right_crop is not None,
            face_present=face_crop is not None,
            pose_present=pose_points is not None,
            body_present=body_present,
            left_hand_points=left_hand_points,
            right_hand_points=right_hand_points,
            pose_points=pose_points,
        )

    def _finalize_reason_label(self, finalize_reason: str | None) -> str:
        if finalize_reason == "user_left_scene":
            return "Da hoan tat cau khi roi khoi khung"
        return "Da hoan tat cau"

    def _clear_committed_transcript(self) -> None:
        self.transcript_segments.clear()

    def _start_finalized_hold(self) -> None:
        self._finalized_hold_deadline_ms = (
            time.time() * 1000.0 + self.config.finalized_hold_seconds * 1000.0
        )

    def _clear_finalized_hold(self) -> None:
        self._finalized_hold_deadline_ms = None

    def _finalized_hold_countdown_ms(self) -> float | None:
        if self._finalized_hold_deadline_ms is None:
            return None
        return max(0.0, self._finalized_hold_deadline_ms - time.time() * 1000.0)

    def _finalized_hold_active(self) -> bool:
        remaining_ms = self._finalized_hold_countdown_ms()
        return remaining_ms is not None and remaining_ms > 0.0

    def _expire_finalized_hold_if_needed(self) -> None:
        remaining_ms = self._finalized_hold_countdown_ms()
        if remaining_ms is None or remaining_ms > 0.0:
            return
        self._clear_committed_transcript()
        self._clear_finalized_hold()
        self.endpoint_detector.mark_idle()

    def _decision_during_finalized_hold(self, decision: EndpointDecision) -> EndpointDecision:
        return EndpointDecision(
            state="finalized_waiting",
            hand_present=decision.hand_present,
            face_present=decision.face_present,
            pose_present=decision.pose_present,
            body_present=decision.body_present,
            scene_present=decision.scene_present,
            collectable_frame=False,
            signing_present=False,
            hand_motion_energy=decision.hand_motion_energy,
            upper_body_motion_energy=decision.upper_body_motion_energy,
            utterance_collect_ms=decision.utterance_collect_ms,
            pause_ms=decision.pause_ms,
            scene_absence_ms=decision.scene_absence_ms,
            should_finalize=False,
            finalize_reason=None,
            did_start_utterance=False,
            should_start_new_utterance=False,
            should_drop_false_start=False,
        )

    def process_chunk_bytes(
        self,
        *,
        frame_bytes_list: list[bytes],
        start_ts_ms: float,
        end_ts_ms: float,
        chunk_index: int,
    ) -> dict[str, object]:
        normalized_frames = [bytes(frame_bytes) for frame_bytes in frame_bytes_list if frame_bytes]
        with self._raw_queue_condition:
            if self._closed:
                return dict(self.last_payload)
            if not normalized_frames:
                return dict(self.last_payload)

            self.last_activity_ts_ms = end_ts_ms
            self.counters["chunks_received"] += 1
            self.counters["frames_received_total"] += len(normalized_frames)
            pending_chunk = PendingChunk(
                chunk_index=chunk_index,
                start_ts_ms=start_ts_ms,
                end_ts_ms=end_ts_ms,
                frame_bytes_list=normalized_frames,
                enqueued_at_perf=time.perf_counter(),
            )
            if self._parallel_prepare_enabled and self.counters["chunks_received"] == 1:
                self._next_analyzed_chunk_index = pending_chunk.chunk_index

            if len(self._raw_chunk_queue) >= self.config.chunk_queue_max and self._raw_chunk_queue:
                self._raw_chunk_queue[-1].extend(pending_chunk)
                self.counters["chunks_coalesced"] += 1
                self.counters["preview_skipped_due_to_backpressure"] += 1
            else:
                self._raw_chunk_queue.append(pending_chunk)
                self._raw_queue_depth_peak = max(self._raw_queue_depth_peak, len(self._raw_chunk_queue))

            if self.last_payload.get("status") in {"idle", "stopped"}:
                self.last_payload = self._payload(
                    status="streaming",
                    status_label="Dang nhan chunk...",
                )

            snapshot = dict(self.last_payload)
            self._raw_queue_condition.notify()
            return snapshot

    def record_ingest_metrics(
        self,
        *,
        parse_seconds: float,
        enqueue_seconds: float,
        endpoint_seconds: float,
    ) -> None:
        with self._state_lock:
            self.ingest_parse_seconds_total += parse_seconds
            self.ingest_enqueue_seconds_total += enqueue_seconds
            self.ingest_endpoint_seconds_total += endpoint_seconds

    def snapshot_payload(self) -> dict[str, object]:
        with self._state_lock:
            return dict(self.last_payload)

    def snapshot_stats(self) -> dict[str, object]:
        with self._state_lock:
            uptime_ms = max(1.0, time.time() * 1000.0 - self.session_started_ts_ms)
            uptime_s = uptime_ms / 1000.0
            counters = dict(self.counters)
            final_text = self._build_final_text()
            return {
                "pipeline_mode": self.config.pipeline_mode,
                "uptime_seconds": round(uptime_s, 3),
                "raw_queue_depth": len(self._raw_chunk_queue),
                "raw_queue_depth_peak": self._raw_queue_depth_peak,
                "prepared_queue_depth": len(self._prepared_chunk_queue),
                "prepared_queue_depth_peak": self._prepared_queue_depth_peak,
                "decode_dirty": self._decode_dirty,
                "chunks_received": counters["chunks_received"],
                "chunks_coalesced": counters["chunks_coalesced"],
                "chunks_processed": counters["chunks_processed"],
                "prepared_chunks_enqueued": counters["prepared_chunks_enqueued"],
                "prepared_chunks_processed": counters["prepared_chunks_processed"],
                "preview_decode_count": counters["preview_decode_count"],
                "final_decode_count": counters["final_decode_count"],
                "decode_wait_count": counters["decode_wait_count"],
                "frames_received_total": counters["frames_received_total"],
                "frames_processed_total": counters["frames_processed_total"],
                "frames_collectable_total": counters["frames_collectable_total"],
                "frames_not_collectable_total": counters["frames_not_collectable_total"],
                "preview_skipped_due_to_backpressure": counters["preview_skipped_due_to_backpressure"],
                "preview_skipped_due_to_gpu_backpressure": counters["preview_skipped_due_to_gpu_backpressure"],
                "chunk_receive_rate": round(counters["chunks_received"] / uptime_s, 3),
                "chunk_process_rate": round(counters["chunks_processed"] / uptime_s, 3),
                "frame_receive_rate": round(counters["frames_received_total"] / uptime_s, 3),
                "frame_process_rate": round(counters["frames_processed_total"] / uptime_s, 3),
                "frame_collectable_rate": round(counters["frames_collectable_total"] / uptime_s, 3),
                "utterance_frames": self._utterance_frame_count(),
                "current_draft_text": self.current_draft_text,
                "current_final_text": final_text,
                "first_draft_elapsed_seconds": (
                    None
                    if self.first_draft_ts_ms is None
                    else round((self.first_draft_ts_ms - self.session_started_ts_ms) / 1000.0, 3)
                ),
                "first_final_elapsed_seconds": (
                    None
                    if self.first_final_ts_ms is None
                    else round((self.first_final_ts_ms - self.session_started_ts_ms) / 1000.0, 3)
                ),
                "mediapipe_chunk_seconds_total": round(self.mediapipe_chunk_seconds_total, 3),
                "gpu_chunk_seconds_total": round(self.gpu_chunk_seconds_total, 3),
                "ingest_parse_seconds_total": round(self.ingest_parse_seconds_total, 3),
                "ingest_enqueue_seconds_total": round(self.ingest_enqueue_seconds_total, 3),
                "ingest_endpoint_seconds_total": round(self.ingest_endpoint_seconds_total, 3),
                "raw_queue_wait_seconds_total": round(self.raw_queue_wait_seconds_total, 3),
                "prepared_queue_wait_seconds_total": round(self.prepared_queue_wait_seconds_total, 3),
                "prepared_enqueue_wait_seconds_total": round(self.prepared_enqueue_wait_seconds_total, 3),
                "jpeg_decode_seconds_total": round(self.jpeg_decode_seconds_total, 3),
                "rgb_convert_seconds_total": round(self.rgb_convert_seconds_total, 3),
                "crop_pose_seconds_total": round(self.crop_pose_seconds_total, 3),
                "hand_embed_seconds_total": round(self.hand_embed_seconds_total, 3),
                "face_embed_seconds_total": round(self.face_embed_seconds_total, 3),
                "preview_decode_seconds_total": round(self.preview_decode_seconds_total, 3),
                "final_decode_seconds_total": round(self.final_decode_seconds_total, 3),
                "last_status": self.last_payload.get("status"),
                "last_status_label": self.last_payload.get("status_label"),
                "last_payload_updated_at_ms": self.last_payload.get("updated_at_ms"),
                "last_decode_ts_ms": round(self.last_decode_ts_ms, 3) if self.last_decode_ts_ms else 0.0,
            }

    def stop(self) -> dict[str, object]:
        with self._state_lock:
            if self._closed:
                return dict(self.last_payload)
            self._worker_should_stop = True
            self._raw_queue_condition.notify_all()
            self._prepared_queue_condition.notify_all()

        for raw_worker in self._raw_workers:
            if raw_worker.is_alive():
                raw_worker.join(timeout=20)
        if self._gpu_worker is not None and self._gpu_worker.is_alive():
            self._gpu_worker.join(timeout=20)

        payload = self.finalize(reason_label="Da hoan tat cau cuoi")
        with self._state_lock:
            self.last_payload = payload
            return dict(payload)

    def finalize(self, reason_label: str) -> dict[str, object]:
        final_text = ""
        if self._utterance_frame_count() > 0:
            self.counters["final_decode_count"] += 1
            decode_start = time.perf_counter()
            final_text = self._decode_utterance(
                generation_num_beams=self.config.final_num_beams,
                generation_max_length=self.config.final_max_length,
            ).strip()
            self.final_decode_seconds_total += time.perf_counter() - decode_start
            if not final_text:
                final_text = self.current_draft_text.strip()
            final_text = " ".join(final_text.split())
            if final_text:
                self.transcript_segments.append(final_text)
                if self.first_final_ts_ms is None:
                    self.first_final_ts_ms = time.time() * 1000.0

        self._reset_current_utterance(reset_detector=False)
        self.endpoint_detector.mark_finalized_waiting()
        if final_text:
            self._start_finalized_hold()
        return self._payload(
            status="finalized" if final_text else "listening",
            status_label=reason_label if final_text else "Da chot cau, dang cho reset cau moi",
        )

    def _split_raw_worker_loop(self, worker_index: int) -> None:
        worker_context = self._prepare_contexts[min(worker_index, len(self._prepare_contexts) - 1)]
        while True:
            pending_chunk = self._pop_raw_chunk()
            if pending_chunk is None:
                return

            try:
                with self._state_lock:
                    self._raw_workers_active += 1
                if self._parallel_prepare_enabled:
                    analyzed_chunk = self._analyze_chunk_sync(pending_chunk, worker_context)
                    if not self._submit_analyzed_chunk(analyzed_chunk):
                        return
                else:
                    prepared_chunk = self._prepare_chunk_sync(pending_chunk)
                    if not self._enqueue_prepared_chunk(prepared_chunk):
                        return
            except Exception:
                with self._state_lock:
                    self.last_payload = self._payload(
                        status="error",
                        status_label="Loi xu ly MediaPipe",
                    )
            finally:
                with self._prepared_queue_condition:
                    self._raw_workers_active = max(0, self._raw_workers_active - 1)
                    self._prepared_queue_condition.notify_all()

    def _split_gpu_worker_loop(self) -> None:
        while True:
            prepared_chunk = self._pop_prepared_chunk()
            if prepared_chunk is None:
                return

            try:
                payload = self._process_prepared_chunk_sync(prepared_chunk)
            except Exception:
                payload = self._payload(
                    status="error",
                    status_label="Loi xu ly GPU chunk",
                )

            with self._state_lock:
                self.last_payload = payload

    def _pop_raw_chunk(self) -> PendingChunk | None:
        with self._raw_queue_condition:
            while not self._raw_chunk_queue and not self._worker_should_stop:
                self._raw_queue_condition.wait(timeout=0.5)
            if not self._raw_chunk_queue and self._worker_should_stop:
                return None
            pending_chunk = self._raw_chunk_queue.popleft()
            self.raw_queue_wait_seconds_total += time.perf_counter() - pending_chunk.enqueued_at_perf
            return pending_chunk

    def _enqueue_prepared_chunk(self, prepared_chunk: PreparedChunk) -> bool:
        with self._prepared_queue_condition:
            wait_start = time.perf_counter()
            while len(self._prepared_chunk_queue) >= self.config.prepared_queue_max and not self._closed:
                self._prepared_queue_condition.wait(timeout=0.25)
            self.prepared_enqueue_wait_seconds_total += time.perf_counter() - wait_start
            if self._closed:
                return False
            prepared_chunk.enqueued_at_perf = time.perf_counter()
            self._prepared_chunk_queue.append(prepared_chunk)
            self.counters["prepared_chunks_enqueued"] += 1
            self._prepared_queue_depth_peak = max(self._prepared_queue_depth_peak, len(self._prepared_chunk_queue))
            self._prepared_queue_condition.notify_all()
            return True

    def _submit_analyzed_chunk(self, analyzed_chunk: AnalyzedChunk) -> bool:
        ready_chunks: list[PreparedChunk] = []
        with self._prepared_queue_condition:
            self._analyzed_chunks_by_index[analyzed_chunk.chunk_index] = analyzed_chunk
            while self._next_analyzed_chunk_index in self._analyzed_chunks_by_index:
                ordered_chunk = self._analyzed_chunks_by_index.pop(self._next_analyzed_chunk_index)
                ready_chunks.append(self._build_prepared_chunk_from_analyzed(ordered_chunk))
                self._next_analyzed_chunk_index += 1

        for prepared_chunk in ready_chunks:
            if not self._enqueue_prepared_chunk(prepared_chunk):
                return False
        return True

    def _pop_prepared_chunk(self) -> PreparedChunk | None:
        with self._prepared_queue_condition:
            while (
                not self._prepared_chunk_queue
                and not (
                    self._worker_should_stop
                    and not self._raw_chunk_queue
                    and self._raw_workers_active == 0
                    and not self._analyzed_chunks_by_index
                )
            ):
                self._prepared_queue_condition.wait(timeout=0.5)
            if (
                not self._prepared_chunk_queue
                and self._worker_should_stop
                and not self._raw_chunk_queue
                and self._raw_workers_active == 0
                and not self._analyzed_chunks_by_index
            ):
                return None
            prepared_chunk = self._prepared_chunk_queue.popleft()
            self.prepared_queue_wait_seconds_total += time.perf_counter() - prepared_chunk.enqueued_at_perf
            self._prepared_queue_condition.notify_all()
            return prepared_chunk

    def _prepare_chunk_sync(self, pending_chunk: PendingChunk) -> PreparedChunk:
        return self._prepare_chunk_sync_with_context(pending_chunk, self._prepare_contexts[0])

    def _analyze_chunk_sync(
        self,
        pending_chunk: PendingChunk,
        worker_context: PrepareWorkerContext,
    ) -> AnalyzedChunk:
        frames: list[AnalyzedFrame] = []
        mediapipe_elapsed_seconds = 0.0
        jpeg_decode_total = 0.0
        rgb_convert_total = 0.0
        crop_pose_total = 0.0
        frames_processed = 0

        frame_step_ms = 1000.0 / max(self.config.target_fps, 1)
        if len(pending_chunk.frame_bytes_list) > 1:
            frame_step_ms = (pending_chunk.end_ts_ms - pending_chunk.start_ts_ms) / max(
                len(pending_chunk.frame_bytes_list) - 1,
                1,
            )

        for frame_index, frame_bytes in enumerate(pending_chunk.frame_bytes_list):
            frames_processed += 1
            frame_timestamp_ms = int(round(pending_chunk.start_ts_ms + frame_step_ms * frame_index))
            decode_start = time.perf_counter()
            decode_buffer = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame_bgr = cv2.imdecode(decode_buffer, cv2.IMREAD_COLOR)
            jpeg_decode_total += time.perf_counter() - decode_start
            if frame_bgr is None:
                frames.append(
                    AnalyzedFrame(
                        frame_timestamp_ms=frame_timestamp_ms,
                        frame_landmarks=None,
                        left_crop=None,
                        right_crop=None,
                        face_crop=None,
                        pose_embedding=None,
                        hand_present=False,
                        face_present=False,
                        pose_present=False,
                        pose_points=None,
                        left_hand_points=None,
                        right_hand_points=None,
                    )
                )
                continue

            convert_start = time.perf_counter()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb_convert_total += time.perf_counter() - convert_start
            stage_start = time.perf_counter()
            _stats, frame_landmarks = worker_context.detector.detect_frame_landmarks(
                frame_rgb,
                frame_timestamp_ms=frame_timestamp_ms,
            )
            mediapipe_elapsed_seconds += time.perf_counter() - stage_start

            crop_pose_start = time.perf_counter()
            left_crop, right_crop = worker_context.hand_extractor.extract_hand_crops_from_frame(frame_rgb, frame_landmarks)
            face_crop = worker_context.face_extractor.extract_face_crop_from_frame(frame_rgb, frame_landmarks)
            left_hand_points, right_hand_points = self._extract_hand_points(
                frame_landmarks=frame_landmarks,
                hand_extractor=worker_context.hand_extractor,
                image_shape=frame_rgb.shape,
            )
            hand_present = left_crop is not None or right_crop is not None
            face_present = face_crop is not None
            pose_points = self._extract_pose_points(frame_landmarks)
            pose_present = pose_points is not None
            pose_embedding = (
                worker_context.pose_processor.process_frame_landmarks(frame_landmarks)
                if hand_present
                else None
            )
            crop_pose_total += time.perf_counter() - crop_pose_start
            frames.append(
                AnalyzedFrame(
                    frame_timestamp_ms=frame_timestamp_ms,
                    frame_landmarks=frame_landmarks,
                    left_crop=left_crop,
                    right_crop=right_crop,
                    face_crop=face_crop,
                    pose_embedding=pose_embedding,
                    hand_present=hand_present,
                    face_present=face_present,
                    pose_present=pose_present,
                    pose_points=pose_points,
                    left_hand_points=left_hand_points,
                    right_hand_points=right_hand_points,
                )
            )

        with self._state_lock:
            self.counters["frames_processed_total"] += frames_processed
            self.jpeg_decode_seconds_total += jpeg_decode_total
            self.rgb_convert_seconds_total += rgb_convert_total
            self.crop_pose_seconds_total += crop_pose_total
            self.mediapipe_chunk_seconds_total += mediapipe_elapsed_seconds

        return AnalyzedChunk(
            chunk_index=pending_chunk.chunk_index,
            start_ts_ms=pending_chunk.start_ts_ms,
            end_ts_ms=pending_chunk.end_ts_ms,
            frames=frames,
            mediapipe_elapsed_seconds=mediapipe_elapsed_seconds,
        )

    def _prepare_chunk_sync_with_context(
        self,
        pending_chunk: PendingChunk,
        worker_context: PrepareWorkerContext,
    ) -> PreparedChunk:
        events: list[PreparedEvent] = []
        segment_left_inputs: list[np.ndarray] = []
        segment_right_inputs: list[np.ndarray] = []
        segment_face_inputs: list[np.ndarray] = []
        segment_pose_embeddings: list[np.ndarray] = []
        mediapipe_elapsed_seconds = 0.0

        frame_step_ms = 1000.0 / max(self.config.target_fps, 1)
        if len(pending_chunk.frame_bytes_list) > 1:
            frame_step_ms = (pending_chunk.end_ts_ms - pending_chunk.start_ts_ms) / max(
                len(pending_chunk.frame_bytes_list) - 1,
                1,
            )

        for frame_index, frame_bytes in enumerate(pending_chunk.frame_bytes_list):
            self.counters["frames_processed_total"] += 1
            frame_timestamp_ms = int(round(pending_chunk.start_ts_ms + frame_step_ms * frame_index))
            decode_start = time.perf_counter()
            decode_buffer = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame_bgr = cv2.imdecode(decode_buffer, cv2.IMREAD_COLOR)
            self.jpeg_decode_seconds_total += time.perf_counter() - decode_start
            if frame_bgr is None:
                continue

            convert_start = time.perf_counter()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.rgb_convert_seconds_total += time.perf_counter() - convert_start
            stage_start = time.perf_counter()
            _stats, frame_landmarks = worker_context.detector.detect_frame_landmarks(
                frame_rgb,
                frame_timestamp_ms=frame_timestamp_ms,
            )
            mediapipe_elapsed_seconds += time.perf_counter() - stage_start

            crop_pose_start = time.perf_counter()
            left_crop, right_crop = worker_context.hand_extractor.extract_hand_crops_from_frame(frame_rgb, frame_landmarks)
            face_crop = worker_context.face_extractor.extract_face_crop_from_frame(frame_rgb, frame_landmarks)
            left_hand_points, right_hand_points = self._extract_hand_points(
                frame_landmarks=frame_landmarks,
                hand_extractor=worker_context.hand_extractor,
                image_shape=frame_rgb.shape,
            )
            decision = self._build_endpoint_decision(
                frame_timestamp_ms=frame_timestamp_ms,
                frame_landmarks=frame_landmarks,
                left_crop=left_crop,
                right_crop=right_crop,
                face_crop=face_crop,
                left_hand_points=left_hand_points,
                right_hand_points=right_hand_points,
            )
            self.last_activity_ts_ms = pending_chunk.end_ts_ms

            self._expire_finalized_hold_if_needed()
            if self._finalized_hold_active():
                self.endpoint_detector.mark_finalized_waiting()
                decision = self._decision_during_finalized_hold(decision)

            if decision.should_finalize:
                self._flush_segment_inputs(
                    events,
                    segment_left_inputs,
                    segment_right_inputs,
                    segment_face_inputs,
                    segment_pose_embeddings,
                )
                events.append(
                    PreparedEvent(
                        kind="finalize",
                        reason_label=self._finalize_reason_label(decision.finalize_reason),
                    )
                )
                self.crop_pose_seconds_total += time.perf_counter() - crop_pose_start
                continue

            if decision.should_drop_false_start:
                self._flush_segment_inputs(
                    events,
                    segment_left_inputs,
                    segment_right_inputs,
                    segment_face_inputs,
                    segment_pose_embeddings,
                )
                payload = self._payload_for_non_collecting_frame(decision)
                events.append(
                    PreparedEvent(
                        kind="drop_false_start",
                        payload_status=str(payload["status"]),
                        payload_status_label=str(payload["status_label"]),
                    )
                )
                self.crop_pose_seconds_total += time.perf_counter() - crop_pose_start
                continue

            if decision.collectable_frame:
                self.counters["frames_collectable_total"] += 1
                segment_left_inputs.append(left_crop if left_crop is not None else self.blank_hand_frame)
                segment_right_inputs.append(right_crop if right_crop is not None else self.blank_hand_frame)
                segment_face_inputs.append(face_crop if face_crop is not None else self.blank_face_frame)
                segment_pose_embeddings.append(worker_context.pose_processor.process_frame_landmarks(frame_landmarks))
                self.crop_pose_seconds_total += time.perf_counter() - crop_pose_start
                continue

            self.counters["frames_not_collectable_total"] += 1
            self._flush_segment_inputs(
                events,
                segment_left_inputs,
                segment_right_inputs,
                segment_face_inputs,
                segment_pose_embeddings,
            )
            payload = self._payload_for_non_collecting_frame(decision)
            events.append(
                PreparedEvent(
                    kind="status",
                    payload_status=str(payload["status"]),
                    payload_status_label=str(payload["status_label"]),
                )
            )
            self.crop_pose_seconds_total += time.perf_counter() - crop_pose_start

        self._flush_segment_inputs(
            events,
            segment_left_inputs,
            segment_right_inputs,
            segment_face_inputs,
            segment_pose_embeddings,
        )
        self.mediapipe_chunk_seconds_total += mediapipe_elapsed_seconds
        return PreparedChunk(
            chunk_index=pending_chunk.chunk_index,
            start_ts_ms=pending_chunk.start_ts_ms,
            end_ts_ms=pending_chunk.end_ts_ms,
            events=events,
            mediapipe_elapsed_seconds=mediapipe_elapsed_seconds,
        )

    def _build_prepared_chunk_from_analyzed(self, analyzed_chunk: AnalyzedChunk) -> PreparedChunk:
        events: list[PreparedEvent] = []
        segment_left_inputs: list[np.ndarray] = []
        segment_right_inputs: list[np.ndarray] = []
        segment_face_inputs: list[np.ndarray] = []
        segment_pose_embeddings: list[np.ndarray] = []

        for frame in analyzed_chunk.frames:
            decision = self.endpoint_detector.update(
                frame_timestamp_ms=frame.frame_timestamp_ms,
                hand_present=frame.hand_present,
                face_present=frame.face_present,
                pose_present=frame.pose_present,
                body_present=frame.pose_present,
                left_hand_points=frame.left_hand_points,
                right_hand_points=frame.right_hand_points,
                pose_points=frame.pose_points,
            )
            self.last_activity_ts_ms = analyzed_chunk.end_ts_ms

            self._expire_finalized_hold_if_needed()
            if self._finalized_hold_active():
                self.endpoint_detector.mark_finalized_waiting()
                decision = self._decision_during_finalized_hold(decision)

            if decision.should_finalize:
                self._flush_segment_inputs(
                    events,
                    segment_left_inputs,
                    segment_right_inputs,
                    segment_face_inputs,
                    segment_pose_embeddings,
                )
                events.append(
                    PreparedEvent(
                        kind="finalize",
                        reason_label=self._finalize_reason_label(decision.finalize_reason),
                    )
                )
                continue

            if decision.should_drop_false_start:
                self._flush_segment_inputs(
                    events,
                    segment_left_inputs,
                    segment_right_inputs,
                    segment_face_inputs,
                    segment_pose_embeddings,
                )
                payload = self._payload_for_non_collecting_frame(decision)
                events.append(
                    PreparedEvent(
                        kind="drop_false_start",
                        payload_status=str(payload["status"]),
                        payload_status_label=str(payload["status_label"]),
                    )
                )
                continue

            if decision.collectable_frame:
                self.counters["frames_collectable_total"] += 1
                segment_left_inputs.append(frame.left_crop if frame.left_crop is not None else self.blank_hand_frame)
                segment_right_inputs.append(frame.right_crop if frame.right_crop is not None else self.blank_hand_frame)
                segment_face_inputs.append(frame.face_crop if frame.face_crop is not None else self.blank_face_frame)
                segment_pose_embeddings.append(
                    frame.pose_embedding
                    if frame.pose_embedding is not None
                    else self.pose_processor.process_frame_landmarks(frame.frame_landmarks)
                )
                continue

            self.counters["frames_not_collectable_total"] += 1
            self._flush_segment_inputs(
                events,
                segment_left_inputs,
                segment_right_inputs,
                segment_face_inputs,
                segment_pose_embeddings,
            )
            payload = self._payload_for_non_collecting_frame(decision)
            events.append(
                PreparedEvent(
                    kind="status",
                    payload_status=str(payload["status"]),
                    payload_status_label=str(payload["status_label"]),
                )
            )

        self._flush_segment_inputs(
            events,
            segment_left_inputs,
            segment_right_inputs,
            segment_face_inputs,
            segment_pose_embeddings,
        )
        return PreparedChunk(
            chunk_index=analyzed_chunk.chunk_index,
            start_ts_ms=analyzed_chunk.start_ts_ms,
            end_ts_ms=analyzed_chunk.end_ts_ms,
            events=events,
            mediapipe_elapsed_seconds=analyzed_chunk.mediapipe_elapsed_seconds,
        )

    def _flush_segment_inputs(
        self,
        events: list[PreparedEvent],
        left_inputs: list[np.ndarray],
        right_inputs: list[np.ndarray],
        face_inputs: list[np.ndarray],
        pose_embeddings: list[np.ndarray],
    ) -> None:
        if not left_inputs:
            return

        events.append(
            PreparedEvent(
                kind="collect",
                left_inputs=list(left_inputs),
                right_inputs=list(right_inputs),
                face_inputs=list(face_inputs),
                pose_embeddings=list(pose_embeddings),
            )
        )
        left_inputs.clear()
        right_inputs.clear()
        face_inputs.clear()
        pose_embeddings.clear()

    def _process_prepared_chunk_sync(self, prepared_chunk: PreparedChunk) -> dict[str, object]:
        payload: dict[str, object] | None = None
        gpu_stage_start = time.perf_counter()

        for event in prepared_chunk.events:
            if event.kind == "collect":
                hand_embed_start = time.perf_counter()
                left_hand_embeddings, right_hand_embeddings = self.runtime.embed_hands_batch(
                    event.left_inputs or [],
                    event.right_inputs or [],
                )
                self.hand_embed_seconds_total += time.perf_counter() - hand_embed_start
                face_embed_start = time.perf_counter()
                face_embeddings = self.runtime.embed_face_batch(event.face_inputs or [])
                self.face_embed_seconds_total += time.perf_counter() - face_embed_start
                pose_embeddings_array = np.asarray(event.pose_embeddings or [], dtype=np.float32)
                self._append_embedding_sequences(
                    face_embeddings=face_embeddings,
                    left_hand_embeddings=left_hand_embeddings,
                    right_hand_embeddings=right_hand_embeddings,
                    pose_embeddings=pose_embeddings_array,
                )
            elif event.kind == "finalize":
                payload = self.finalize(reason_label=event.reason_label or "Da hoan tat cau")
            elif event.kind == "drop_false_start":
                self._reset_current_utterance(reset_detector=False)
                payload = self._payload(
                    status=event.payload_status or "listening",
                    status_label=event.payload_status_label or "Dang doi nguoi ky bat dau",
                )
            elif event.kind == "status":
                payload = self._payload(
                    status=event.payload_status or "waiting",
                    status_label=event.payload_status_label or "Dang giu ngu canh va cho tiep...",
                )

        self.gpu_chunk_seconds_total += time.perf_counter() - gpu_stage_start
        self.counters["chunks_processed"] += 1
        self.counters["prepared_chunks_processed"] += 1

        if payload is None:
            utterance_frames = self._utterance_frame_count()
            if 0 < utterance_frames < self.config.preview_initial_frames:
                remaining = self.config.preview_initial_frames - utterance_frames
                payload = self._payload(
                    status="warming_up",
                    status_label=f"Dang tich luy ngu canh... con {remaining} frame nua",
                )
            else:
                payload = self._payload(
                    status="streaming",
                    status_label="Dang nghe va tich luy ngu canh",
                )

        if self._decode_dirty:
            if self._should_defer_preview_decode():
                self.counters["decode_wait_count"] += 1
                self.counters["preview_skipped_due_to_gpu_backpressure"] += 1
                return payload
            payload = self._run_preview_decode()

        return payload

    def _should_defer_preview_decode(self) -> bool:
        with self._state_lock:
            raw_backlog = len(self._raw_chunk_queue)
            prepared_backlog = len(self._prepared_chunk_queue)
            analyzed_backlog = len(self._analyzed_chunks_by_index)
            # A small in-flight backlog is normal during realtime streaming.
            # Only defer preview decode when queues are actually saturated or
            # chunk reordering has fallen meaningfully behind.
            raw_saturated = raw_backlog >= max(1, self.config.chunk_queue_max)
            prepared_saturated = prepared_backlog >= max(1, self.config.prepared_queue_max)
            reorder_backlog = analyzed_backlog > 1
            return raw_saturated or prepared_saturated or reorder_backlog

    def _append_embedding_sequences(
        self,
        *,
        face_embeddings: np.ndarray,
        left_hand_embeddings: np.ndarray,
        right_hand_embeddings: np.ndarray,
        pose_embeddings: np.ndarray,
    ) -> None:
        for face_embedding, left_embedding, right_embedding, pose_embedding in zip(
            face_embeddings,
            left_hand_embeddings,
            right_hand_embeddings,
            pose_embeddings,
        ):
            self.recent_face_embeddings.append(face_embedding)
            self.recent_left_hand_embeddings.append(left_embedding)
            self.recent_right_hand_embeddings.append(right_embedding)
            self.recent_pose_embeddings.append(pose_embedding)

            self.utterance_face_embeddings.append(face_embedding)
            self.utterance_left_hand_embeddings.append(left_embedding)
            self.utterance_right_hand_embeddings.append(right_embedding)
            self.utterance_pose_embeddings.append(pose_embedding)

        utterance_frames = self._utterance_frame_count()
        while utterance_frames >= self._next_preview_frame_target:
            self._decode_dirty = True
            self._next_preview_frame_target += self.config.preview_stride_frames

    def _run_preview_decode(self) -> dict[str, object]:
        utterance_frames = self._utterance_frame_count()
        if utterance_frames < self.config.preview_initial_frames:
            remaining = self.config.preview_initial_frames - utterance_frames
            return self._payload(
                status="warming_up",
                status_label=f"Dang tich luy ngu canh... con {remaining} frame nua",
            )

        decode_start = time.perf_counter()
        hypothesis = self._decode_utterance(
            generation_num_beams=self.config.preview_num_beams,
            generation_max_length=self.config.preview_max_length,
        ).strip()
        self.preview_decode_seconds_total += time.perf_counter() - decode_start
        self.counters["preview_decode_count"] += 1
        self._decode_dirty = False
        self.last_decode_ts_ms = time.time() * 1000.0

        if hypothesis:
            normalized_text = " ".join(hypothesis.split())
            stabilized = self.stabilizer.update(normalized_text)
            self.current_draft_text = str(stabilized["draft_text"]).strip()
            if self.current_draft_text and self.first_draft_ts_ms is None:
                self.first_draft_ts_ms = time.time() * 1000.0
            return self._payload(
                status="live",
                status_label="Dang dich truc tiep",
            )

        return self._payload(
            status="listening",
            status_label="Chua nhan ra tu ro rang",
        )

    def _decode_utterance(
        self,
        *,
        generation_num_beams: int,
        generation_max_length: int,
    ) -> str:
        return self.runtime.decode(
            face_embeddings=np.stack(list(self.utterance_face_embeddings)),
            left_hand_embeddings=np.stack(list(self.utterance_left_hand_embeddings)),
            right_hand_embeddings=np.stack(list(self.utterance_right_hand_embeddings)),
            pose_embeddings=np.stack(list(self.utterance_pose_embeddings)),
            generation_num_beams=generation_num_beams,
            generation_max_length=generation_max_length,
        )

    def _payload_for_non_collecting_frame(self, decision: EndpointDecision) -> dict[str, object]:
        if decision.state == "collecting":
            return self._payload(
                status="streaming",
                status_label="Dang nghe va tich luy ngu canh",
            )
        if decision.state == "hold_after_pause":
            status_label = (
                "Dang giu ngu canh va cho tiep..."
                if decision.pause_ms < self.config.endpoint_resume_grace_ms
                else "Tam dung, dang cho xac nhan ket cau"
            )
            return self._payload(
                status="waiting",
                status_label=status_label,
            )
        if decision.state == "finalized_waiting":
            return self._payload(
                status="listening",
                status_label="Da chot cau, dang cho reset cau moi",
            )
        label = "San sang nghe cau tiep theo" if self.transcript_segments else "Dang doi nguoi ky bat dau"
        return self._payload(
            status="listening",
            status_label=label,
        )

    def _payload(self, *, status: str, status_label: str) -> dict[str, object]:
        final_text = self._build_final_text()
        draft_text = self._build_draft_text()
        return {
            "status": status,
            "status_label": status_label,
            "display_text": draft_text or final_text,
            "draft_text": draft_text,
            "final_text": final_text,
            "finalized_hold_countdown_ms": self._finalized_hold_countdown_ms(),
            "updated_at_ms": round(time.time() * 1000.0, 3),
        }

    def _build_final_text(self) -> str:
        return " ".join(segment for segment in self.transcript_segments if segment).strip()

    def _build_draft_text(self) -> str:
        if not self.current_draft_text:
            return ""
        return self.current_draft_text

    def _utterance_frame_count(self) -> int:
        return len(self.utterance_face_embeddings)

    def _reset_current_utterance(self, *, reset_detector: bool = True) -> None:
        self.recent_face_embeddings.clear()
        self.recent_left_hand_embeddings.clear()
        self.recent_right_hand_embeddings.clear()
        self.recent_pose_embeddings.clear()
        self.utterance_face_embeddings.clear()
        self.utterance_left_hand_embeddings.clear()
        self.utterance_right_hand_embeddings.clear()
        self.utterance_pose_embeddings.clear()
        self.current_draft_text = ""
        self.last_decode_ts_ms = 0.0
        self._decode_dirty = False
        self._clear_finalized_hold()
        self._next_preview_frame_target = self.config.preview_initial_frames
        self.stabilizer.reset()
        if reset_detector:
            self.endpoint_detector.reset()
