from __future__ import annotations

import argparse
import concurrent.futures
import difflib
import json
import os
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import cv2


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_PATH = ROOT_DIR / "media" / "jobs" / "input" / "2026" / "03" / "31" / "L5hUxT5YbnY_crop_2_FSr2w8Y.mp4"
DEFAULT_REPORT_DIR = ROOT_DIR / ".runtime_cache" / "realtime_benchmarks"
DEFAULT_APP_FILE = ROOT_DIR / "modal_realtime_app.py"
COMPONENT_TOTAL_KEYS = [
    "ingest_parse_seconds_total",
    "ingest_enqueue_seconds_total",
    "ingest_endpoint_seconds_total",
    "raw_queue_wait_seconds_total",
    "prepared_queue_wait_seconds_total",
    "prepared_enqueue_wait_seconds_total",
    "jpeg_decode_seconds_total",
    "rgb_convert_seconds_total",
    "crop_pose_seconds_total",
    "mediapipe_chunk_seconds_total",
    "hand_embed_seconds_total",
    "face_embed_seconds_total",
    "preview_decode_seconds_total",
    "final_decode_seconds_total",
    "gpu_chunk_seconds_total",
]
QUEUE_PEAK_KEYS = [
    "raw_queue_depth_peak",
    "prepared_queue_depth_peak",
]
CLIENT_METRIC_KEYS = [
    "client_capture_encode_seconds_total",
    "client_pack_seconds_total",
    "inflight_upload_peak",
]


@dataclass(slots=True)
class SampledFrame:
    ts_ms: float
    frame_bgr: Any


@dataclass(slots=True)
class EncodedChunk:
    chunk_index: int
    start_ts_ms: float
    end_ts_ms: float
    frames: list[SampledFrame]


@dataclass(slots=True)
class RunMetrics:
    label: str
    cold: bool
    session_id: str
    pipeline_mode: str
    harness_mode: str
    time_to_runtime_ready: float
    time_to_first_draft: float | None
    time_to_final_text: float | None
    total_session_wall_time: float
    chunk_ack_latency_p50: float
    chunk_ack_latency_p95: float
    chunks_sent: int
    chunks_coalesced: int
    chunks_processed: int
    prepared_chunks_enqueued: int
    prepared_chunks_processed: int
    frames_received: int
    frames_processed: int
    frames_collectable: int
    preview_decode_count: int
    final_decode_count: int
    raw_queue_depth: int
    prepared_queue_depth: int
    mediapipe_chunk_seconds_total: float
    gpu_chunk_seconds_total: float
    decode_wait_count: int
    preview_skipped_due_to_backpressure: int
    preview_skipped_due_to_gpu_backpressure: int
    final_text: str
    client_metrics: dict[str, float]
    component_totals: dict[str, float]
    queue_peaks: dict[str, float]


METHODS: dict[str, dict[str, Any]] = {
    "split_sota_default": {
        "app_name": "openslt-realtime-sota-default",
        "env": {
            "SIGN_TRANSLATION_REALTIME_PRESET": "split_sota_default",
        },
        "unset_env": [],
    },
    "split_mp2_cpu10": {
        "app_name": "openslt-realtime-mp2-cpu10",
        "env": {
            "SIGN_TRANSLATION_REALTIME_PRESET": "split_mp2_cpu10",
        },
        "unset_env": [],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare split_sota_default vs split_mp2_cpu10 on realtime browser-equivalent harness.")
    parser.add_argument("--video-path", default=str(DEFAULT_VIDEO_PATH))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--target-fps", type=int, default=15)
    parser.add_argument("--chunk-frames", type=int, default=15)
    parser.add_argument("--frame-width", type=int, default=640)
    parser.add_argument("--frame-height", type=int, default=480)
    parser.add_argument("--jpeg-quality", type=int, default=72)
    parser.add_argument("--warm-runs", type=int, default=3)
    parser.add_argument("--max-inflight-uploads", type=int, default=3)
    parser.add_argument("--stop-apps", action="store_true")
    return parser.parse_args()


def run_modal_command(
    args: list[str],
    *,
    extra_env: dict[str, str] | None = None,
    unset_env: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    for key in unset_env or []:
        env.pop(key, None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        args,
        cwd=ROOT_DIR.parent,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )


def current_modal_workspace() -> str:
    result = run_modal_command(["modal", "profile", "current"])
    return result.stdout.strip()


def deploy_app(app_name: str, *, env_overrides: dict[str, str], unset_env: list[str]) -> str:
    env = {"SIGN_TRANSLATION_MODAL_REALTIME_APP_NAME": app_name, **env_overrides}
    result = run_modal_command(["modal", "deploy", str(DEFAULT_APP_FILE)], extra_env=env, unset_env=unset_env)
    output = "\n".join(part for part in [result.stdout, result.stderr] if part)
    workspace = current_modal_workspace()
    marker = ".modal.run"
    if marker in output:
        start = output.find("https://")
        if start >= 0:
            end = output.find(marker, start)
            if end >= 0:
                return output[start : end + len(marker)]
    return f"https://{workspace}--{app_name}-realtime-app.modal.run"


def stop_app(app_name: str) -> None:
    try:
        run_modal_command(["modal", "app", "stop", app_name])
    except subprocess.CalledProcessError:
        pass


def encode_video_to_chunks(
    video_path: Path,
    *,
    target_fps: int,
    chunk_frames: int,
    frame_width: int,
    frame_height: int,
) -> list[EncodedChunk]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Khong mo duoc video benchmark: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    sample_period_s = 1.0 / float(target_fps)
    next_sample_ts_s = 0.0
    source_frame_index = 0
    sampled_frames: list[SampledFrame] = []

    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        frame_ts_s = source_frame_index / source_fps
        source_frame_index += 1
        if frame_ts_s + 1e-9 < next_sample_ts_s:
            continue
        resized = cv2.resize(frame_bgr, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
        sampled_frames.append(SampledFrame(ts_ms=round(next_sample_ts_s * 1000.0, 3), frame_bgr=resized))
        next_sample_ts_s += sample_period_s

    capture.release()

    chunks: list[EncodedChunk] = []
    for chunk_index, start in enumerate(range(0, len(sampled_frames), chunk_frames)):
        frames = sampled_frames[start : start + chunk_frames]
        if not frames:
            continue
        chunks.append(
            EncodedChunk(
                chunk_index=chunk_index,
                start_ts_ms=frames[0].ts_ms,
                end_ts_ms=frames[-1].ts_ms,
                frames=frames,
            )
        )
    return chunks


def encode_multipart_formdata(fields: dict[str, str], files: list[tuple[str, str, bytes, str]]) -> tuple[bytes, str]:
    boundary = f"----OpenSLTBenchmark{int(time.time() * 1000)}"
    body_parts: list[bytes] = []
    for name, value in fields.items():
        body_parts.append(f"--{boundary}\r\n".encode("utf-8"))
        body_parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        body_parts.append(value.encode("utf-8"))
        body_parts.append(b"\r\n")
    for field_name, filename, file_bytes, content_type in files:
        body_parts.append(f"--{boundary}\r\n".encode("utf-8"))
        body_parts.append(
            f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode("utf-8")
        )
        body_parts.append(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body_parts.append(file_bytes)
        body_parts.append(b"\r\n")
    body_parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(body_parts), f"multipart/form-data; boundary={boundary}"


def http_json(
    method: str,
    url: str,
    *,
    timeout: int = 60,
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
) -> tuple[int, dict[str, Any]]:
    req = urllib_request.Request(url, method=method, data=body, headers=headers or {})
    try:
        with urllib_request.urlopen(req, timeout=timeout) as response:
            return int(response.status), json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except Exception:
            payload = {"detail": str(exc)}
        return int(exc.code), payload


def fetch_json(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
    status_code, payload = http_json(method, url, **kwargs)
    if status_code >= 400:
        raise RuntimeError(f"HTTP {status_code} for {url}: {payload}")
    return payload


def wait_for_runtime_ready(base_url: str, *, timeout_s: int = 600) -> float:
    start = time.perf_counter()
    while time.perf_counter() - start < timeout_s:
        payload = fetch_json("GET", f"{base_url}/healthz", timeout=60)
        if payload.get("runtime", {}).get("ready"):
            return round(time.perf_counter() - start, 3)
        time.sleep(5.0)
    raise TimeoutError(f"Runtime khong ready sau {timeout_s}s: {base_url}")


def wait_for_session_id(base_url: str, *, timeout_s: int = 600) -> str:
    start = time.perf_counter()
    while time.perf_counter() - start < timeout_s:
        status_code, payload = http_json("POST", f"{base_url}/api/session", timeout=60)
        if status_code == 200 and payload.get("session_id"):
            return str(payload["session_id"])
        if status_code >= 500:
            raise RuntimeError(f"Loi tao session: {payload}")
        time.sleep(2.0)
    raise TimeoutError(f"Khong tao duoc session sau {timeout_s}s: {base_url}")


def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return round(values[0], 3)
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return round(ordered[lower] * (1.0 - weight) + ordered[upper] * weight, 3)


def _build_run_metrics(
    *,
    label: str,
    cold: bool,
    session_id: str,
    time_to_runtime_ready: float,
    time_to_first_draft: float | None,
    time_to_final_text: float | None,
    total_session_wall_time: float,
    ack_latencies: list[float],
    final_stats: dict[str, Any],
    final_text: str,
    client_metrics: dict[str, float],
) -> RunMetrics:
    component_totals = {key: float(final_stats.get(key, 0.0)) for key in COMPONENT_TOTAL_KEYS}
    queue_peaks = {key: float(final_stats.get(key, 0.0)) for key in QUEUE_PEAK_KEYS}
    normalized_client_metrics = {key: float(client_metrics.get(key, 0.0)) for key in CLIENT_METRIC_KEYS}
    return RunMetrics(
        label=label,
        cold=cold,
        session_id=session_id,
        pipeline_mode=str(final_stats.get("pipeline_mode", "")),
        harness_mode="realtime_browser_e2e",
        time_to_runtime_ready=time_to_runtime_ready,
        time_to_first_draft=time_to_first_draft,
        time_to_final_text=time_to_final_text,
        total_session_wall_time=total_session_wall_time,
        chunk_ack_latency_p50=percentile(ack_latencies, 0.50),
        chunk_ack_latency_p95=percentile(ack_latencies, 0.95),
        chunks_sent=int(final_stats.get("chunks_received", 0)),
        chunks_coalesced=int(final_stats.get("chunks_coalesced", 0)),
        chunks_processed=int(final_stats.get("chunks_processed", 0)),
        prepared_chunks_enqueued=int(final_stats.get("prepared_chunks_enqueued", 0)),
        prepared_chunks_processed=int(final_stats.get("prepared_chunks_processed", 0)),
        frames_received=int(final_stats.get("frames_received_total", 0)),
        frames_processed=int(final_stats.get("frames_processed_total", 0)),
        frames_collectable=int(final_stats.get("frames_collectable_total", 0)),
        preview_decode_count=int(final_stats.get("preview_decode_count", 0)),
        final_decode_count=int(final_stats.get("final_decode_count", 0)),
        raw_queue_depth=int(final_stats.get("raw_queue_depth", 0)),
        prepared_queue_depth=int(final_stats.get("prepared_queue_depth", 0)),
        mediapipe_chunk_seconds_total=float(final_stats.get("mediapipe_chunk_seconds_total", 0.0)),
        gpu_chunk_seconds_total=float(final_stats.get("gpu_chunk_seconds_total", 0.0)),
        decode_wait_count=int(final_stats.get("decode_wait_count", 0)),
        preview_skipped_due_to_backpressure=int(final_stats.get("preview_skipped_due_to_backpressure", 0)),
        preview_skipped_due_to_gpu_backpressure=int(final_stats.get("preview_skipped_due_to_gpu_backpressure", 0)),
        final_text=final_text,
        client_metrics=normalized_client_metrics,
        component_totals=component_totals,
        queue_peaks=queue_peaks,
    )


def _encode_frame_like_browser(frame: SampledFrame, *, jpeg_quality: int) -> tuple[bytes, float]:
    encode_start = time.perf_counter()
    ok, encoded = cv2.imencode(".jpg", frame.frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError("Khong ma hoa duoc frame benchmark.")
    return encoded.tobytes(), time.perf_counter() - encode_start


def _send_chunk_request(
    base_url: str,
    session_id: str,
    *,
    chunk_index: int,
    start_ts_ms: float,
    end_ts_ms: float,
    files: list[tuple[str, str, bytes, str]],
) -> tuple[float, dict[str, Any]]:
    pack_start = time.perf_counter()
    body, content_type = encode_multipart_formdata(
        fields={
            "chunk_index": str(chunk_index),
            "start_ts_ms": str(start_ts_ms),
            "end_ts_ms": str(end_ts_ms),
        },
        files=files,
    )
    pack_seconds = time.perf_counter() - pack_start
    request_start = time.perf_counter()
    status_code, payload = http_json(
        "POST",
        f"{base_url}/api/chunk/{session_id}",
        timeout=180,
        headers={"Content-Type": content_type},
        body=body,
    )
    if status_code >= 400:
        raise RuntimeError(f"Loi gui chunk: {payload}")
    return time.perf_counter() - request_start, {"payload": payload, "pack_seconds": pack_seconds}


def run_single_session_browser(
    base_url: str,
    chunks: list[EncodedChunk],
    *,
    cold: bool,
    label: str,
    jpeg_quality: int,
    max_inflight_uploads: int,
) -> RunMetrics:
    time_to_runtime_ready = wait_for_runtime_ready(base_url)
    session_id = wait_for_session_id(base_url)
    session_start_perf = time.perf_counter()
    ack_latencies: list[float] = []
    first_draft_seconds: float | None = None
    first_final_seconds: float | None = None
    client_capture_encode_seconds_total = 0.0
    client_pack_seconds_total = 0.0
    inflight_upload_peak = 0
    pending: set[concurrent.futures.Future[tuple[float, dict[str, Any]]]] = set()

    def collect_completed(*, wait_for_one: bool) -> None:
        nonlocal first_draft_seconds, first_final_seconds, client_pack_seconds_total
        if not pending:
            return
        if wait_for_one:
            done, _ = concurrent.futures.wait(
                pending,
                return_when=concurrent.futures.FIRST_COMPLETED,
                timeout=60,
            )
        else:
            done = {future for future in pending if future.done()}
        for future in list(done):
            pending.discard(future)
            ack_latency, metadata = future.result()
            ack_latencies.append(ack_latency)
            client_pack_seconds_total += float(metadata["pack_seconds"])
            payload = metadata["payload"]
            elapsed = round(time.perf_counter() - session_start_perf, 3)
            if first_draft_seconds is None and payload.get("draft_text"):
                first_draft_seconds = elapsed
            if first_final_seconds is None and payload.get("final_text"):
                first_final_seconds = elapsed

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_inflight_uploads) as executor:
        for chunk in chunks:
            release_perf = session_start_perf + (chunk.end_ts_ms / 1000.0)
            while time.perf_counter() < release_perf:
                collect_completed(wait_for_one=False)
                time.sleep(0.01)

            files: list[tuple[str, str, bytes, str]] = []
            for idx, frame in enumerate(chunk.frames):
                encoded_bytes, encode_seconds = _encode_frame_like_browser(frame, jpeg_quality=jpeg_quality)
                client_capture_encode_seconds_total += encode_seconds
                files.append(("frames", f"chunk_{chunk.chunk_index}_{idx}.jpg", encoded_bytes, "image/jpeg"))

            while len(pending) >= max_inflight_uploads:
                collect_completed(wait_for_one=True)

            pending.add(
                executor.submit(
                    _send_chunk_request,
                    base_url,
                    session_id,
                    chunk_index=chunk.chunk_index,
                    start_ts_ms=chunk.start_ts_ms,
                    end_ts_ms=chunk.end_ts_ms,
                    files=files,
                )
            )
            inflight_upload_peak = max(inflight_upload_peak, len(pending))
            collect_completed(wait_for_one=False)

        while pending:
            collect_completed(wait_for_one=True)

    stop_status_code, stop_payload = http_json("POST", f"{base_url}/api/session/{session_id}/stop", timeout=240)
    if stop_status_code >= 400:
        raise RuntimeError(f"Loi stop session: {stop_payload}")
    total_session_wall_time = round(time.perf_counter() - session_start_perf, 3)
    if first_final_seconds is None and stop_payload.get("final_text"):
        first_final_seconds = round(time.perf_counter() - session_start_perf, 3)

    final_stats = stop_payload.get("stats") or {}
    return _build_run_metrics(
        label=label,
        cold=cold,
        session_id=session_id,
        time_to_runtime_ready=time_to_runtime_ready,
        time_to_first_draft=first_draft_seconds,
        time_to_final_text=first_final_seconds,
        total_session_wall_time=total_session_wall_time,
        ack_latencies=ack_latencies,
        final_stats=final_stats,
        final_text=normalize_text(str(stop_payload.get("final_text") or "")),
        client_metrics={
            "client_capture_encode_seconds_total": round(client_capture_encode_seconds_total, 3),
            "client_pack_seconds_total": round(client_pack_seconds_total, 3),
            "inflight_upload_peak": float(inflight_upload_peak),
        },
    )


def summarize_runs(runs: list[RunMetrics]) -> dict[str, Any]:
    metric_keys = [
        "time_to_runtime_ready",
        "time_to_first_draft",
        "time_to_final_text",
        "total_session_wall_time",
        "chunk_ack_latency_p50",
        "chunk_ack_latency_p95",
        "chunks_sent",
        "chunks_coalesced",
        "chunks_processed",
        "prepared_chunks_enqueued",
        "prepared_chunks_processed",
        "frames_received",
        "frames_processed",
        "frames_collectable",
        "preview_decode_count",
        "final_decode_count",
        "raw_queue_depth",
        "prepared_queue_depth",
        "mediapipe_chunk_seconds_total",
        "gpu_chunk_seconds_total",
        "decode_wait_count",
        "preview_skipped_due_to_backpressure",
        "preview_skipped_due_to_gpu_backpressure",
    ]
    summary: dict[str, Any] = {}
    for key in metric_keys:
        values = [getattr(run, key) for run in runs if getattr(run, key) is not None]
        summary[f"{key}_mean"] = round(statistics.mean(values), 3) if values else None
    summary["client_metrics_mean"] = {
        key: round(statistics.mean([run.client_metrics[key] for run in runs]), 3)
        for key in CLIENT_METRIC_KEYS
    }
    summary["component_totals_mean"] = {
        key: round(statistics.mean([run.component_totals[key] for run in runs]), 3)
        for key in COMPONENT_TOTAL_KEYS
    }
    summary["queue_peaks_mean"] = {
        key: round(statistics.mean([run.queue_peaks[key] for run in runs]), 3)
        for key in QUEUE_PEAK_KEYS
    }
    summary["final_text"] = runs[-1].final_text if runs else ""
    return summary


def improvement_pct(reference: float | None, candidate: float | None) -> float | None:
    if reference in (None, 0) or candidate is None:
        return None
    return round(((reference - candidate) / reference) * 100.0, 2)


def compare_methods(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reference_text = normalize_text(reference["warm_summary"]["final_text"])
    candidate_text = normalize_text(candidate["warm_summary"]["final_text"])
    return {
        "time_to_first_draft": improvement_pct(
            reference["warm_summary"]["time_to_first_draft_mean"],
            candidate["warm_summary"]["time_to_first_draft_mean"],
        ),
        "time_to_final_text": improvement_pct(
            reference["warm_summary"]["time_to_final_text_mean"],
            candidate["warm_summary"]["time_to_final_text_mean"],
        ),
        "chunk_ack_latency_p50": improvement_pct(
            reference["warm_summary"]["chunk_ack_latency_p50_mean"],
            candidate["warm_summary"]["chunk_ack_latency_p50_mean"],
        ),
        "normalized_text_match": reference_text == candidate_text,
        "normalized_text_diff": list(difflib.ndiff([reference_text], [candidate_text])),
    }


def write_report(report_dir: Path, payload: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    sota = payload["results"]["split_sota_default"]["warm_summary"]
    mp2 = payload["results"]["split_mp2_cpu10"]["warm_summary"]
    comparison = payload["comparison"]
    lines = [
        "# Realtime Benchmark",
        "",
        "## Methods",
        "",
        "- `split_sota_default`: `pipeline_mode=split`, `raw_worker_count=1`, khong set CPU.",
        "- `split_mp2_cpu10`: `pipeline_mode=split`, `raw_worker_count=2`, `cpu=10`, `parallel_detectors=0`, `parallel_workers=1`.",
        "",
        "## Warm Summary",
        "",
        "| Metric | split_sota_default | split_mp2_cpu10 | Improvement vs sota |",
        "| --- | ---: | ---: | ---: |",
        f"| Time to first draft (s) | {sota['time_to_first_draft_mean']} | {mp2['time_to_first_draft_mean']} | {comparison['time_to_first_draft']}% |",
        f"| Time to final text (s) | {sota['time_to_final_text_mean']} | {mp2['time_to_final_text_mean']} | {comparison['time_to_final_text']}% |",
        f"| Chunk ack p50 (s) | {sota['chunk_ack_latency_p50_mean']} | {mp2['chunk_ack_latency_p50_mean']} | {comparison['chunk_ack_latency_p50']}% |",
        f"| Chunks processed | {sota['chunks_processed_mean']} | {mp2['chunks_processed_mean']} | - |",
        f"| Chunks coalesced | {sota['chunks_coalesced_mean']} | {mp2['chunks_coalesced_mean']} | - |",
        f"| Raw queue wait total (s) | {sota['component_totals_mean']['raw_queue_wait_seconds_total']} | {mp2['component_totals_mean']['raw_queue_wait_seconds_total']} | - |",
        f"| Prepared queue wait total (s) | {sota['component_totals_mean']['prepared_queue_wait_seconds_total']} | {mp2['component_totals_mean']['prepared_queue_wait_seconds_total']} | - |",
        "",
        "## Text Check",
        "",
        f"- Normalized text match: `{comparison['normalized_text_match']}`",
        f"- split_sota_default final text: {payload['results']['split_sota_default']['warm_summary']['final_text']}",
        f"- split_mp2_cpu10 final text: {payload['results']['split_mp2_cpu10']['warm_summary']['final_text']}",
    ]
    (report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark_suite(
    *,
    method_name: str,
    base_url: str,
    chunks: list[EncodedChunk],
    warm_runs: int,
    jpeg_quality: int,
    max_inflight_uploads: int,
) -> dict[str, Any]:
    cold_run = run_single_session_browser(
        base_url,
        chunks,
        cold=True,
        label=f"{method_name}_cold",
        jpeg_quality=jpeg_quality,
        max_inflight_uploads=max_inflight_uploads,
    )
    warm_results = [
        run_single_session_browser(
            base_url,
            chunks,
            cold=False,
            label=f"{method_name}_warm_{index + 1}",
            jpeg_quality=jpeg_quality,
            max_inflight_uploads=max_inflight_uploads,
        )
        for index in range(warm_runs)
    ]
    return {
        "cold_run": asdict(cold_run),
        "warm_runs": [asdict(run) for run in warm_results],
        "warm_summary": summarize_runs(warm_results),
    }


def main() -> None:
    args = parse_args()
    video_path = Path(args.video_path).resolve()
    chunks = encode_video_to_chunks(
        video_path,
        target_fps=args.target_fps,
        chunk_frames=args.chunk_frames,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
    )
    if not chunks:
        raise RuntimeError("Khong tao duoc chunk benchmark nao.")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_dir = Path(args.report_dir).resolve() / f"{timestamp}_sota_vs_mp2"
    results: dict[str, Any] = {}

    try:
        for method_name, method in METHODS.items():
            base_url = deploy_app(method["app_name"], env_overrides=method["env"], unset_env=method["unset_env"])
            results[method_name] = run_benchmark_suite(
                method_name=method_name,
                base_url=base_url,
                chunks=chunks,
                warm_runs=args.warm_runs,
                jpeg_quality=args.jpeg_quality,
                max_inflight_uploads=args.max_inflight_uploads,
            )
            if args.stop_apps:
                stop_app(method["app_name"])
    finally:
        if args.stop_apps:
            for method in METHODS.values():
                stop_app(method["app_name"])

    payload = {
        "harness_mode": "realtime_browser_e2e",
        "video_path": str(video_path),
        "target_fps": args.target_fps,
        "chunk_frames": args.chunk_frames,
        "frame_width": args.frame_width,
        "frame_height": args.frame_height,
        "jpeg_quality": args.jpeg_quality,
        "results": results,
        "comparison": compare_methods(results["split_sota_default"], results["split_mp2_cpu10"]),
    }
    write_report(report_dir, payload)
    print(json.dumps({"report_dir": str(report_dir), "report_json": str(report_dir / "report.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
