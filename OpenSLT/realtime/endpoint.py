from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class EndpointDecision:
    state: str
    hand_present: bool
    face_present: bool
    pose_present: bool
    body_present: bool
    scene_present: bool
    collectable_frame: bool
    signing_present: bool
    hand_motion_energy: float
    upper_body_motion_energy: float
    utterance_collect_ms: float
    pause_ms: float
    scene_absence_ms: float
    should_finalize: bool
    finalize_reason: str | None
    did_start_utterance: bool
    should_start_new_utterance: bool
    should_drop_false_start: bool


class EndpointDetector:
    """
    Heuristic state machine for realtime utterance boundaries.

    The detector distinguishes three user-facing end conditions:
    - signer pauses but remains in frame
    - signer leaves the scene
    - signer produces a short false start that should not be committed
    """

    _STABILITY_WINDOW = 3
    _STABILITY_REQUIRED = 2

    def __init__(
        self,
        *,
        target_fps: int,
        min_utterance_ms: int,
        resume_grace_ms: int,
        finalize_pause_ms: int,
        scene_absence_ms: int,
        false_start_reset_ms: int,
        hand_motion_threshold: float,
        upper_body_motion_threshold: float,
    ):
        self.frame_interval_ms = 1000.0 / max(target_fps, 1)
        self.min_utterance_ms = float(min_utterance_ms)
        self.resume_grace_ms = float(resume_grace_ms)
        self.finalize_pause_ms = float(finalize_pause_ms)
        self.scene_absence_ms_threshold = float(scene_absence_ms)
        self.false_start_reset_ms = float(false_start_reset_ms)
        self.hand_motion_threshold = float(hand_motion_threshold)
        self.upper_body_motion_threshold = float(upper_body_motion_threshold)
        self.reset()

    def reset(self) -> None:
        self.state = "idle"
        self._return_state_after_drop = "idle"
        self._last_timestamp_ms: float | None = None
        self._recent_signing_frames: deque[bool] = deque(maxlen=self._STABILITY_WINDOW)
        self._clear_runtime_state()

    def mark_finalized_waiting(self) -> None:
        self.state = "finalized_waiting"
        self._return_state_after_drop = "finalized_waiting"
        self._clear_runtime_state()

    def mark_idle(self) -> None:
        self.state = "idle"
        self._return_state_after_drop = "idle"
        self._clear_runtime_state()

    def update(
        self,
        *,
        frame_timestamp_ms: float | int | None,
        hand_present: bool,
        face_present: bool,
        pose_present: bool,
        body_present: bool,
        left_hand_points: np.ndarray | None,
        right_hand_points: np.ndarray | None,
        pose_points: np.ndarray | None,
    ) -> EndpointDecision:
        dt_ms = self._compute_dt_ms(frame_timestamp_ms)
        hand_motion_energy = self._compute_hand_motion_energy(left_hand_points, right_hand_points)
        upper_body_motion_energy = self._compute_upper_body_motion_energy(pose_points)

        scene_present = body_present or hand_present or face_present
        collectable_frame = hand_present
        signing_present = hand_present and (
            hand_motion_energy >= self.hand_motion_threshold
            or upper_body_motion_energy >= self.upper_body_motion_threshold
        )
        self._recent_signing_frames.append(signing_present)
        stable_signing_present = sum(self._recent_signing_frames) >= self._STABILITY_REQUIRED

        if collectable_frame:
            self.utterance_collect_ms += dt_ms

        should_finalize = False
        finalize_reason: str | None = None
        did_start_utterance = False
        should_start_new_utterance = False
        should_drop_false_start = False
        decision_utterance_collect_ms = self.utterance_collect_ms
        decision_pause_ms = self.pause_ms
        decision_scene_absence_ms = self.scene_absence_ms

        if self.state in {"idle", "finalized_waiting"}:
            if stable_signing_present:
                did_start_utterance = True
                should_start_new_utterance = self.state == "finalized_waiting"
                self.state = "collecting"
                self._return_state_after_drop = "finalized_waiting" if should_start_new_utterance else "idle"
                self.pause_ms = 0.0
                self.scene_absence_ms = 0.0
            elif self.utterance_collect_ms > 0.0:
                scene_absent = not body_present and not scene_present
                if scene_absent:
                    self.scene_absence_ms += dt_ms
                else:
                    self.scene_absence_ms = 0.0
                self.pause_ms += dt_ms
                # Do not throw away a partial utterance just because hands are
                # temporarily occluded while the signer still remains in frame.
                # We only drop a false start when the signer truly disappears.
                if scene_absent and self.scene_absence_ms >= self.false_start_reset_ms:
                    should_drop_false_start = True
                    decision_utterance_collect_ms = self.utterance_collect_ms
                    decision_pause_ms = self.pause_ms
                    decision_scene_absence_ms = self.scene_absence_ms
                    self._transition_after_drop()
        elif self.state == "collecting":
            if signing_present:
                self.pause_ms = 0.0
                self.scene_absence_ms = 0.0
            else:
                self.state = "hold_after_pause"
                self.pause_ms = dt_ms
                self.scene_absence_ms = dt_ms if (not body_present and not scene_present) else 0.0
        elif self.state == "hold_after_pause":
            if stable_signing_present:
                self.state = "collecting"
                self.pause_ms = 0.0
                self.scene_absence_ms = 0.0
            else:
                self.pause_ms += dt_ms
                scene_absent = not body_present and not scene_present
                if scene_absent:
                    self.scene_absence_ms += dt_ms
                else:
                    self.scene_absence_ms = 0.0

                if self.utterance_collect_ms < self.min_utterance_ms:
                    # Partial utterances stay alive while the signer is still
                    # visible; only drop them when the signer has actually left
                    # the scene for long enough.
                    if scene_absent and self.scene_absence_ms >= self.false_start_reset_ms:
                        should_drop_false_start = True
                        decision_utterance_collect_ms = self.utterance_collect_ms
                        decision_pause_ms = self.pause_ms
                        decision_scene_absence_ms = self.scene_absence_ms
                        self._transition_after_drop()
                elif scene_absent:
                    if self.scene_absence_ms >= self.scene_absence_ms_threshold:
                        should_finalize = True
                        finalize_reason = "user_left_scene"
                        decision_utterance_collect_ms = self.utterance_collect_ms
                        decision_pause_ms = self.pause_ms
                        decision_scene_absence_ms = self.scene_absence_ms
                        self.state = "finalized_waiting"
                        self._prepare_waiting_after_finalize()
                elif self.pause_ms >= self.finalize_pause_ms:
                    should_finalize = True
                    finalize_reason = "pause_end_of_sentence"
                    decision_utterance_collect_ms = self.utterance_collect_ms
                    decision_pause_ms = self.pause_ms
                    decision_scene_absence_ms = self.scene_absence_ms
                    self.state = "finalized_waiting"
                    self._prepare_waiting_after_finalize()

        return EndpointDecision(
            state=self.state,
            hand_present=hand_present,
            face_present=face_present,
            pose_present=pose_present,
            body_present=body_present,
            scene_present=scene_present,
            collectable_frame=collectable_frame,
            signing_present=signing_present,
            hand_motion_energy=hand_motion_energy,
            upper_body_motion_energy=upper_body_motion_energy,
            utterance_collect_ms=decision_utterance_collect_ms,
            pause_ms=decision_pause_ms,
            scene_absence_ms=decision_scene_absence_ms,
            should_finalize=should_finalize,
            finalize_reason=finalize_reason,
            did_start_utterance=did_start_utterance,
            should_start_new_utterance=should_start_new_utterance,
            should_drop_false_start=should_drop_false_start,
        )

    def _clear_runtime_state(self) -> None:
        self.prev_pose_points: np.ndarray | None = None
        self.prev_left_hand_points: np.ndarray | None = None
        self.prev_right_hand_points: np.ndarray | None = None
        self.utterance_collect_ms = 0.0
        self.pause_ms = 0.0
        self.scene_absence_ms = 0.0
        self._last_timestamp_ms = None
        self._recent_signing_frames.clear()

    def _transition_after_drop(self) -> None:
        self.state = self._return_state_after_drop
        self._clear_runtime_state()

    def _prepare_waiting_after_finalize(self) -> None:
        self._return_state_after_drop = "finalized_waiting"
        self._clear_runtime_state()

    def _compute_dt_ms(self, frame_timestamp_ms: float | int | None) -> float:
        if frame_timestamp_ms is None:
            return self.frame_interval_ms

        timestamp_ms = float(frame_timestamp_ms)
        if self._last_timestamp_ms is None:
            self._last_timestamp_ms = timestamp_ms
            return self.frame_interval_ms

        dt_ms = max(1.0, timestamp_ms - self._last_timestamp_ms)
        self._last_timestamp_ms = timestamp_ms
        return dt_ms

    def _compute_upper_body_motion_energy(self, pose_points: np.ndarray | None) -> float:
        if pose_points is None:
            self.prev_pose_points = None
            return 0.0
        if self.prev_pose_points is None:
            self.prev_pose_points = pose_points
            return 0.0

        motion = float(np.linalg.norm(pose_points - self.prev_pose_points, axis=1).mean())
        self.prev_pose_points = pose_points
        return motion

    def _compute_hand_motion_energy(
        self,
        left_hand_points: np.ndarray | None,
        right_hand_points: np.ndarray | None,
    ) -> float:
        energies: list[float] = []

        if left_hand_points is not None and self.prev_left_hand_points is not None:
            energies.append(
                float(np.linalg.norm(left_hand_points - self.prev_left_hand_points, axis=1).mean())
            )
        if right_hand_points is not None and self.prev_right_hand_points is not None:
            energies.append(
                float(np.linalg.norm(right_hand_points - self.prev_right_hand_points, axis=1).mean())
            )

        self.prev_left_hand_points = left_hand_points
        self.prev_right_hand_points = right_hand_points

        if not energies:
            return 0.0
        return float(np.mean(energies))
