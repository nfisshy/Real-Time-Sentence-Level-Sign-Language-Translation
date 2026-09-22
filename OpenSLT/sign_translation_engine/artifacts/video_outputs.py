import os
import subprocess
from pathlib import Path

import cv2
import numpy as np


POSE_COLOR = (41, 163, 92)
FACE_COLOR = (224, 168, 44)
HAND_COLORS = [(214, 77, 77), (64, 113, 215)]
SUBTITLE_FONT = cv2.FONT_HERSHEY_DUPLEX
SUBTITLE_TEXT_COLOR = (255, 255, 255)
SUBTITLE_BOX_COLOR = (12, 12, 12)
SUBTITLE_BOX_ALPHA = 0.62


def _to_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame


def write_video(frames, output_path: Path, fps: float = 15.0):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("Khong co frame de ghi video.")

    first_frame = frames[0]
    height, width = first_frame.shape[:2]
    temp_output_path = output_path.with_name(f"{output_path.stem}.raw{output_path.suffix}")
    writer = cv2.VideoWriter(
        str(temp_output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    try:
        for frame in frames:
            writer.write(_to_bgr(frame))
    finally:
        writer.release()

    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-i",
        str(temp_output_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    try:
        subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (FileNotFoundError, subprocess.CalledProcessError):
        os.replace(temp_output_path, output_path)
        return

    temp_output_path.unlink(missing_ok=True)


def _draw_points(frame: np.ndarray, landmarks, color, radius: int):
    height, width = frame.shape[:2]
    for x, y, _z in landmarks:
        px = int(x * width)
        py = int(y * height)
        cv2.circle(frame, (px, py), radius, color, -1)


def _subtitle_tokens(text: str) -> list[str]:
    return [token for token in text.split() if token.strip()]


def _text_width(text: str, font_scale: float, thickness: int) -> int:
    return cv2.getTextSize(text, SUBTITLE_FONT, font_scale, thickness)[0][0]


def _truncate_to_width(text: str, max_width: int, font_scale: float, thickness: int) -> str:
    candidate = text.strip()
    if not candidate:
        return ""
    if _text_width(candidate, font_scale, thickness) <= max_width:
        return candidate

    ellipsis = "..."
    truncated = candidate
    while truncated and _text_width(truncated + ellipsis, font_scale, thickness) > max_width:
        truncated = truncated[:-1].rstrip()
    return (truncated + ellipsis).strip() if truncated else ellipsis


def _wrap_subtitle_words(
    words: list[str],
    max_width: int,
    font_scale: float,
    thickness: int,
    max_lines: int = 3,
    truncate_last_line: bool = False,
) -> list[str] | None:
    if not words:
        return []

    lines: list[str] = []
    current_words: list[str] = []
    index = 0

    while index < len(words):
        word = words[index]
        candidate = " ".join(current_words + [word])
        if not current_words:
            if _text_width(candidate, font_scale, thickness) <= max_width:
                current_words.append(word)
                index += 1
                continue
            if not truncate_last_line:
                return None
            lines.append(_truncate_to_width(word, max_width, font_scale, thickness))
            index += 1
            if len(lines) == max_lines:
                return [line for line in lines if line]
            continue

        if _text_width(candidate, font_scale, thickness) <= max_width:
            current_words.append(word)
            index += 1
            continue

        lines.append(" ".join(current_words))
        current_words = []

        if len(lines) == max_lines - 1:
            if not truncate_last_line:
                return None
            remaining = " ".join(words[index:])
            lines.append(_truncate_to_width(remaining, max_width, font_scale, thickness))
            return [line for line in lines if line]

    if current_words:
        lines.append(" ".join(current_words))

    if len(lines) <= max_lines:
        return lines

    if not truncate_last_line:
        return None

    preserved = lines[: max_lines - 1]
    remaining = " ".join(lines[max_lines - 1 :])
    preserved.append(_truncate_to_width(remaining, max_width, font_scale, thickness))
    return [line for line in preserved if line]


def _fit_subtitle_layout(words: list[str], frame_width: int) -> tuple[float, int, list[str]]:
    max_width = int(frame_width * 0.88)
    thickness = 2
    scale = min(1.2, max(0.55, frame_width / 1100.0))

    while scale >= 0.45:
        lines = _wrap_subtitle_words(words, max_width, scale, thickness, max_lines=3)
        if lines is not None:
            return scale, thickness, lines
        scale = round(scale - 0.05, 2)

    fallback_scale = 0.45
    fallback_lines = _wrap_subtitle_words(
        words,
        max_width,
        fallback_scale,
        1,
        max_lines=3,
        truncate_last_line=True,
    )
    return fallback_scale, 1, fallback_lines or []


def _visible_word_count_for_frame(
    frame_index: int,
    total_frames: int,
    total_words: int,
    reveal_fraction: float,
) -> int:
    if total_words <= 0:
        return 0
    if total_frames <= 1:
        return total_words

    reveal_frames = max(1, int(np.ceil(total_frames * reveal_fraction)))
    if frame_index >= reveal_frames - 1:
        return total_words

    progress = (frame_index + 1) / reveal_frames
    return max(1, min(total_words, int(np.ceil(progress * total_words))))


def render_subtitle_frames(
    frames,
    text: str,
    reveal_fraction: float = 0.75,
    bottom_margin_ratio: float = 0.06,
):
    words = _subtitle_tokens(text)
    if len(frames) == 0 or not words:
        return []

    frame_height, frame_width = frames[0].shape[:2]
    font_scale, thickness, _full_lines = _fit_subtitle_layout(words, frame_width)
    max_text_width = int(frame_width * 0.88)
    line_height = cv2.getTextSize("Ag", SUBTITLE_FONT, font_scale, thickness)[0][1] + 12
    padding_x = 22
    padding_y = 18
    baseline_pad = 10
    line_cache: dict[int, list[str]] = {}
    rendered = []

    for frame_index, frame in enumerate(frames):
        visible_words = _visible_word_count_for_frame(frame_index, len(frames), len(words), reveal_fraction)
        if visible_words not in line_cache:
            line_cache[visible_words] = _wrap_subtitle_words(
                words[:visible_words],
                max_text_width,
                font_scale,
                thickness,
                max_lines=3,
                truncate_last_line=True,
            ) or []

        lines = line_cache[visible_words]
        canvas = frame.copy()
        if not lines:
            rendered.append(canvas)
            continue

        text_sizes = [cv2.getTextSize(line, SUBTITLE_FONT, font_scale, thickness)[0] for line in lines]
        box_width = max(size[0] for size in text_sizes) + padding_x * 2
        box_height = len(lines) * line_height + padding_y * 2
        box_x = max(0, (frame_width - box_width) // 2)
        box_y = max(0, frame_height - box_height - int(frame_height * bottom_margin_ratio))

        overlay = canvas.copy()
        cv2.rectangle(
            overlay,
            (box_x, box_y),
            (min(frame_width - 1, box_x + box_width), min(frame_height - 1, box_y + box_height)),
            SUBTITLE_BOX_COLOR,
            -1,
        )
        cv2.addWeighted(overlay, SUBTITLE_BOX_ALPHA, canvas, 1.0 - SUBTITLE_BOX_ALPHA, 0, canvas)

        y = box_y + padding_y + line_height - baseline_pad
        for line, (text_width, _text_height) in zip(lines, text_sizes):
            text_x = box_x + max(0, (box_width - text_width) // 2)
            cv2.putText(
                canvas,
                line,
                (text_x, y),
                SUBTITLE_FONT,
                font_scale,
                SUBTITLE_TEXT_COLOR,
                thickness,
                cv2.LINE_AA,
            )
            y += line_height

        rendered.append(canvas)

    return rendered


def render_landmarks_frames(frames, landmarks_by_frame):
    rendered = []
    for index, frame in enumerate(frames):
        canvas = frame.copy()
        landmarks = landmarks_by_frame.get(index) or {}

        for pose in landmarks.get("pose_landmarks") or []:
            _draw_points(canvas, pose, POSE_COLOR, radius=3)

        for face in landmarks.get("face_landmarks") or []:
            _draw_points(canvas, face, FACE_COLOR, radius=1)

        for hand_index, hand in enumerate(landmarks.get("hand_landmarks") or []):
            color = HAND_COLORS[hand_index % len(HAND_COLORS)]
            _draw_points(canvas, hand, color, radius=3)

        rendered.append(canvas)
    return rendered
