from __future__ import annotations

import asyncio
from dataclasses import asdict
import logging
import threading
import time
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from .config import RealtimeConfig

logger = logging.getLogger(__name__)


HTML_PAGE = """<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenSLT Live Translate</title>
  <style>
    :root {
      --ink: #1c1f19;
      --muted: #5f6755;
      --line: rgba(76, 88, 62, 0.18);
      --bg1: #f4f0e6;
      --bg2: #dbe6cf;
      --accent: #294f2d;
      --panel: rgba(255,255,255,0.82);
    }
    * { box-sizing: border-box; }
    body { font-family: Georgia, serif; margin: 0; color: var(--ink); background: radial-gradient(circle at top left, rgba(255,255,255,0.9), transparent 30%), linear-gradient(135deg, var(--bg1), var(--bg2)); min-height: 100vh; }
    .wrap { max-width: 1180px; margin: 0 auto; padding: 28px 20px 40px; }
    .topbar { display: flex; justify-content: space-between; gap: 20px; align-items: center; margin-bottom: 20px; }
    .brand { max-width: 620px; }
    .eyebrow { font-size: 12px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); margin-bottom: 10px; }
    h1 { margin: 0 0 8px; font-size: clamp(36px, 5vw, 64px); line-height: 0.95; }
    .sub { margin: 0; color: var(--muted); font-size: 17px; line-height: 1.5; }
    .actions { display: flex; gap: 12px; flex-wrap: wrap; }
    .hero { display: grid; gap: 18px; grid-template-columns: 0.95fr 1.05fr; align-items: stretch; }
    .card { background: var(--panel); border: 1px solid var(--line); border-radius: 28px; padding: 20px; backdrop-filter: blur(12px); box-shadow: 0 18px 48px rgba(47, 53, 37, 0.08); }
    video, canvas { width: 100%; border-radius: 22px; background: #0f1110; aspect-ratio: 4 / 3; object-fit: cover; }
    .hidden { display: none; }
    .stage { display: flex; align-items: center; gap: 10px; margin-bottom: 18px; color: var(--muted); }
    .stage-wrap { display: grid; gap: 8px; margin-bottom: 18px; }
    .dot { width: 12px; height: 12px; border-radius: 999px; background: #7d8970; box-shadow: 0 0 0 0 rgba(41,79,45,0.4); animation: pulse 1.6s infinite; }
    .countdown { font-size: 14px; color: var(--muted); min-height: 18px; }
    .label { font-size: 12px; text-transform: uppercase; letter-spacing: 0.12em; color: var(--muted); margin-bottom: 10px; }
    .translation { display: grid; gap: 16px; }
    .translation-panel { border: 1px solid var(--line); border-radius: 24px; padding: 22px; background: rgba(255,255,255,0.55); min-height: 132px; }
    .primary-panel { min-height: 260px; }
    .primary-text { font-size: clamp(30px, 4vw, 54px); line-height: 1.08; color: var(--accent); margin: 0; white-space: pre-wrap; }
    .secondary-text { font-size: clamp(20px, 2.2vw, 28px); line-height: 1.25; color: #49644b; margin: 0; white-space: pre-wrap; }
    .empty-text { color: #77806f; }
    .helper { margin-top: 16px; color: var(--muted); font-size: 15px; line-height: 1.6; }
    button { border: none; background: #294f2d; color: #f7fbf5; padding: 14px 22px; border-radius: 999px; cursor: pointer; font-size: 16px; font-weight: 600; }
    .secondary { background: rgba(41,79,45,0.08); color: var(--accent); }
    button:disabled { opacity: 0.4; cursor: default; }
    @keyframes pulse {
      0% { box-shadow: 0 0 0 0 rgba(41,79,45,0.3); }
      70% { box-shadow: 0 0 0 10px rgba(41,79,45,0); }
      100% { box-shadow: 0 0 0 0 rgba(41,79,45,0); }
    }
    @media (max-width: 920px) {
      .topbar { flex-direction: column; align-items: flex-start; }
      .hero { grid-template-columns: 1fr; }
      .primary-text { font-size: 34px; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <div class="brand">
        <div class="eyebrow">OpenSLT Live Translate</div>
        <h1>Dich ngon ngu ky hieu thanh van ban theo tung cau</h1>
        <p class="sub">Camera chay 30 fps, trinh duyet lay mau 15 fps, gom 15 frame thanh 1 chunk roi gui len server de giu ngu canh cau va giam drop vo ich.</p>
      </div>
      <div class="actions">
        <button id="startBtn">Bat dau demo</button>
        <button id="stopBtn" class="secondary" disabled>Dung va chot cau</button>
      </div>
    </div>
    <div class="hero">
      <div class="card">
        <video id="video" autoplay playsinline muted></video>
        <canvas id="canvas" class="hidden"></canvas>
        <p class="helper">He thong gom 15 frame thanh 1 chunk, xu ly theo lo va chi cap nhat van ban khi ngu canh da du de giam mat nghia.</p>
      </div>
      <div class="card">
        <div class="stage-wrap">
          <div class="stage">
            <div class="dot"></div>
            <div id="statusText">San sang bat dau demo</div>
          </div>
          <div class="countdown" id="countdownText"></div>
        </div>
        <div class="translation">
          <div class="translation-panel primary-panel">
            <div class="label">Ban Da Chot</div>
            <p class="primary-text empty-text" id="finalText">Cho bat camera de bat dau.</p>
          </div>
          <div class="translation-panel">
            <div class="label">Ban Nhap Dang Nghe</div>
            <p class="secondary-text empty-text" id="draftText">Ban nhap se hien o day khi he thong dang ghep cau.</p>
          </div>
        </div>
      </div>
    </div>
  </div>
  <script>
    const startBtn = document.getElementById("startBtn");
    const stopBtn = document.getElementById("stopBtn");
    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const finalText = document.getElementById("finalText");
    const draftText = document.getElementById("draftText");
    const statusText = document.getElementById("statusText");
    const countdownText = document.getElementById("countdownText");
    const ctx = canvas.getContext("2d", { willReadFrequently: true });

    const CAMERA_INTERVAL_MS = 1000 / 30;
    const SAMPLE_INTERVAL_MS = 1000 / 15;
    const CHUNK_FRAMES = 15;

    let stream = null;
    let timer = null;
    let sessionId = null;
    let stopping = false;
    let latestPayloadTs = 0;
    let pendingRequests = 0;
    let captureInProgress = false;
    let lastSampleTs = 0;
    let currentChunk = [];
    let currentChunkStartTs = 0;
    let chunkIndex = 0;
    let countdownRenderTimer = null;
    let countdownDeadlineClientMs = null;

    function renderCountdown() {
      if (!countdownDeadlineClientMs) {
        countdownText.textContent = "";
        return;
      }
      const remainingMs = Math.max(0, countdownDeadlineClientMs - Date.now());
      const remainingSeconds = (remainingMs / 1000).toFixed(1);
      countdownText.textContent = `Con ${remainingSeconds}s de reset va qua cau moi`;
      if (remainingMs <= 0) {
        countdownDeadlineClientMs = null;
        finalText.textContent = "He thong se dua cau da chot vao day.";
        finalText.classList.add("empty-text");
        draftText.textContent = "Ban nhap se hien o day khi he thong dang ghep cau.";
        draftText.classList.add("empty-text");
        statusText.textContent = "Dang doi nguoi ky bat dau";
      }
    }

    function syncCountdown(payload) {
      const countdownMs = payload.finalized_hold_countdown_ms;
      if (typeof countdownMs === "number" && countdownMs > 0) {
        countdownDeadlineClientMs = Date.now() + countdownMs;
        if (!countdownRenderTimer) {
          countdownRenderTimer = setInterval(renderCountdown, 100);
        }
        renderCountdown();
        return;
      }
      countdownDeadlineClientMs = null;
      renderCountdown();
    }

    function cameraErrorMessage(error) {
      const name = error && error.name ? error.name : "";
      if (name === "NotAllowedError" || name === "PermissionDeniedError") {
        return "Trinh duyet dang chan camera. Hay bam cho phep camera cho trang nay.";
      }
      if (name === "NotFoundError" || name === "DevicesNotFoundError") {
        return "Khong tim thay camera tren may nay.";
      }
      if (name === "NotReadableError" || name === "TrackStartError") {
        return "Camera dang duoc ung dung khac su dung. Hay dong cac app dang chiem camera.";
      }
      if (name === "OverconstrainedError" || name === "ConstraintNotSatisfiedError") {
        return "Camera khong ho tro cau hinh hien tai. Se can doi cau hinh camera.";
      }
      if (name === "SecurityError") {
        return "Trinh duyet khong cho phep truy cap camera trong boi canh hien tai.";
      }
      return "Khong mo duoc camera. Hay kiem tra quyen camera trong browser va Windows.";
    }

    async function openCameraStream() {
      try {
        return await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, frameRate: { ideal: 30, max: 30 } },
          audio: false
        });
      } catch (error) {
        const name = error && error.name ? error.name : "";
        if (name === "OverconstrainedError" || name === "ConstraintNotSatisfiedError") {
          return navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        }
        throw error;
      }
    }

    function resetChunkBuilder() {
      currentChunk = [];
      currentChunkStartTs = 0;
      lastSampleTs = 0;
      chunkIndex = 0;
    }

    function sleep(ms) {
      return new Promise((resolve) => setTimeout(resolve, ms));
    }

    async function createSessionWithWarmup() {
      while (true) {
        const sessionResponse = await fetch("/api/session", { method: "POST" });
        const sessionPayload = await sessionResponse.json();
        if (sessionPayload.session_id) {
          return sessionPayload.session_id;
        }

        if (sessionPayload.status_label) {
          statusText.textContent = sessionPayload.status_label;
        }

        if (sessionResponse.status >= 500) {
          throw new Error(sessionPayload.status_label || "Khong the khoi dong runtime realtime.");
        }

        await sleep(3000);
      }
    }

    function setDisplay(payload) {
      const payloadTs = payload.updated_at_ms || 0;
      if (payloadTs && payloadTs < latestPayloadTs) return;
      latestPayloadTs = payloadTs || latestPayloadTs;
      const committedText = payload.final_text || "";
      const previewText = payload.draft_text || "";
      finalText.textContent = committedText || "He thong se dua cau da chot vao day.";
      finalText.classList.toggle("empty-text", !committedText);
      draftText.textContent = previewText || "Ban nhap se hien o day khi he thong dang ghep cau.";
      draftText.classList.toggle("empty-text", !previewText);
      statusText.textContent = payload.status_label || payload.detail || "Dang xu ly...";
      syncCountdown(payload);
    }

    async function sendChunk(frames, startTsMs, endTsMs) {
      if (!sessionId || !frames.length) return;
      pendingRequests += 1;
      try {
        const form = new FormData();
        form.append("chunk_index", String(chunkIndex));
        form.append("start_ts_ms", String(startTsMs));
        form.append("end_ts_ms", String(endTsMs));
        frames.forEach((frame, idx) => {
          form.append("frames", frame, `chunk_${chunkIndex}_${idx}.jpg`);
        });
        chunkIndex += 1;
        const response = await fetch(`/api/chunk/${sessionId}`, {
          method: "POST",
          body: form,
        });
        const payload = await response.json();
        if (!response.ok) {
          statusText.textContent = payload.detail || "Loi gui chunk";
          return;
        }
        setDisplay(payload);
      } catch (error) {
        statusText.textContent = "Mat ket noi toi dich vu demo";
      } finally {
        pendingRequests -= 1;
      }
    }

    async function flushCurrentChunk() {
      if (!currentChunk.length) return;
      const frames = currentChunk;
      const startTsMs = currentChunkStartTs;
      const endTsMs = performance.now();
      currentChunk = [];
      currentChunkStartTs = 0;
      await sendChunk(frames, startTsMs, endTsMs);
    }

    async function captureSample() {
      if (!stream || !sessionId || stopping || captureInProgress) return;
      const now = performance.now();
      if (now - lastSampleTs < SAMPLE_INTERVAL_MS - 2) return;
      captureInProgress = true;
      lastSampleTs = now;

      try {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.72));
        if (!blob) return;

        if (!currentChunk.length) {
          currentChunkStartTs = now;
        }
        currentChunk.push(blob);

        if (currentChunk.length >= CHUNK_FRAMES) {
          const frames = currentChunk;
          const startTsMs = currentChunkStartTs;
          currentChunk = [];
          currentChunkStartTs = 0;
          void sendChunk(frames, startTsMs, now);
        }
      } finally {
        captureInProgress = false;
      }
    }

    async function start() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        statusText.textContent = "Trinh duyet nay khong ho tro mo camera.";
        finalText.textContent = "Hay thu Chrome, Edge hoac Firefox ban moi.";
        finalText.classList.add("empty-text");
        draftText.textContent = "Ban nhap se hien o day khi camera hoat dong.";
        draftText.classList.add("empty-text");
        return;
      }

      stopping = false;
      startBtn.disabled = true;
      statusText.textContent = "Dang mo camera...";
      try {
        stream = await openCameraStream();
        video.srcObject = stream;
        await video.play();
        statusText.textContent = "Dang khoi dong phien dich...";
        latestPayloadTs = 0;
        pendingRequests = 0;
        resetChunkBuilder();
        countdownDeadlineClientMs = null;
        renderCountdown();
        canvas.width = 640;
        canvas.height = 480;
        finalText.textContent = "He thong se dua cau da chot vao day.";
        finalText.classList.add("empty-text");
        draftText.textContent = "Dang lang nghe...";
        draftText.classList.add("empty-text");
        sessionId = await createSessionWithWarmup();
        statusText.textContent = "San sang nhan ngon ngu ky hieu";
        timer = setInterval(() => { void captureSample(); }, CAMERA_INTERVAL_MS);
        stopBtn.disabled = false;
      } catch (error) {
        const message = error && error.message ? error.message : cameraErrorMessage(error);
        statusText.textContent = message;
        finalText.textContent = "Khong the bat camera tren trinh duyet nay.";
        finalText.classList.add("empty-text");
        draftText.textContent = "Hay kiem tra quyen camera hoac thu dong cac app dang chiem camera.";
        draftText.classList.add("empty-text");
        if (sessionId) {
          fetch(`/api/session/${sessionId}`, { method: "DELETE" }).catch(() => {});
        }
        sessionId = null;
        if (stream) {
          for (const track of stream.getTracks()) track.stop();
        }
        stream = null;
        stopBtn.disabled = true;
        startBtn.disabled = false;
      }
    }

    async function stop() {
      if (stopping) return;
      stopping = true;
      stopBtn.disabled = true;
      if (timer) clearInterval(timer);
      timer = null;

      statusText.textContent = "Dang hoan tat cau cuoi...";
      await flushCurrentChunk();
      for (let i = 0; i < 20 && pendingRequests > 0; i += 1) {
        await new Promise((resolve) => setTimeout(resolve, 50));
      }

      const currentSessionId = sessionId;
      sessionId = null;
      if (currentSessionId) {
        try {
          const response = await fetch(`/api/session/${currentSessionId}/stop`, { method: "POST" });
          const payload = await response.json();
          setDisplay(payload);
        } catch (error) {
          statusText.textContent = "Khong the hoan tat cau cuoi";
        }
      }

      if (stream) {
        for (const track of stream.getTracks()) track.stop();
      }
      stream = null;
      pendingRequests = 0;
      captureInProgress = false;
      resetChunkBuilder();
      countdownDeadlineClientMs = null;
      renderCountdown();
      draftText.textContent = "Ban nhap se hien o day khi bat dau phien moi.";
      draftText.classList.add("empty-text");
      startBtn.disabled = false;
      stopBtn.disabled = true;
      stopping = false;
    }

    startBtn.addEventListener("click", start);
    stopBtn.addEventListener("click", () => { void stop(); });
  </script>
</body>
</html>
"""


def create_realtime_app(config: RealtimeConfig | None = None):
    from .runtime import get_runtime_status, get_shared_realtime_runtime, start_runtime_warmup
    from .session import RealtimeSession

    config = config or RealtimeConfig.from_env()
    app = FastAPI(title="OpenSLT Realtime")
    sessions: dict[str, RealtimeSession] = {}
    sessions_lock = threading.Lock()

    @app.on_event("startup")
    async def start_runtime_background_warmup() -> None:
        start_runtime_warmup()

    def _cleanup_idle_sessions(now_ms: float) -> None:
        stale_ids: list[str] = []
        for session_id, session in sessions.items():
            last_seen = session.last_activity_ts_ms or session.session_started_ts_ms
            if now_ms - last_seen > config.session_idle_timeout_ms:
                stale_ids.append(session_id)
        for session_id in stale_ids:
            sessions.pop(session_id).close()

    def _get_session(session_id: str) -> RealtimeSession:
        with sessions_lock:
            session = sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session khong ton tai hoac da het han.")
        return session

    @app.get("/")
    async def index() -> HTMLResponse:
        return HTMLResponse(HTML_PAGE)

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        runtime_status = get_runtime_status()
        return JSONResponse(
            {
                "ok": True,
                "mode": "realtime",
                "config": asdict(config),
                "runtime": runtime_status,
            }
        )

    @app.post("/api/session")
    async def create_session() -> JSONResponse:
        runtime_status = get_runtime_status()
        if runtime_status["status"] != "ready":
            start_runtime_warmup()
            status_code = 202 if runtime_status["status"] != "error" else 503
            return JSONResponse(
                {
                    "warming": runtime_status["status"] != "error",
                    "session_id": None,
                    "status": runtime_status["status"],
                    "status_label": runtime_status["message"],
                    "runtime": runtime_status,
                },
                status_code=status_code,
            )

        runtime = await asyncio.to_thread(get_shared_realtime_runtime)
        now_ms = time.time() * 1000.0
        with sessions_lock:
            _cleanup_idle_sessions(now_ms)
            session_id = uuid4().hex
            sessions[session_id] = RealtimeSession(runtime=runtime, config=config)
        logger.info("Realtime HTTP | session_created | session_id=%s", session_id)
        return JSONResponse({"session_id": session_id})

    @app.post("/api/session/{session_id}/stop")
    async def stop_session(session_id: str) -> JSONResponse:
        with sessions_lock:
            session = sessions.pop(session_id, None)
        if session is None:
            raise HTTPException(status_code=404, detail="Session khong ton tai hoac da het han.")

        try:
            payload = await asyncio.to_thread(session.stop)
            stats = await asyncio.to_thread(session.snapshot_stats)
        finally:
            session.close()

        logger.info("Realtime HTTP | session_stopped | session_id=%s", session_id)
        return JSONResponse({**payload, "session_closed": True, "session_id": session_id, "stats": stats})

    @app.get("/api/session/{session_id}/stats")
    async def get_session_stats(session_id: str) -> JSONResponse:
        session = _get_session(session_id)
        stats = await asyncio.to_thread(session.snapshot_stats)
        return JSONResponse({"session_id": session_id, **stats})

    @app.delete("/api/session/{session_id}")
    async def delete_session(session_id: str) -> JSONResponse:
        with sessions_lock:
            session = sessions.pop(session_id, None)
        if session is not None:
            session.close()
        logger.info("Realtime HTTP | session_closed | session_id=%s", session_id)
        return JSONResponse({"ok": True, "session_id": session_id})

    @app.post("/api/chunk/{session_id}")
    async def process_chunk(session_id: str, request: Request) -> JSONResponse:
        session = _get_session(session_id)
        endpoint_start = time.perf_counter()
        parse_start = time.perf_counter()
        form = await request.form()
        try:
            chunk_index = int(str(form["chunk_index"]))
            start_ts_ms = float(str(form["start_ts_ms"]))
            end_ts_ms = float(str(form["end_ts_ms"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Chunk metadata khong hop le: {exc}") from exc

        frame_bytes_list: list[bytes] = []
        for key, value in form.multi_items():
            if key != "frames":
                continue
            if hasattr(value, "read"):
                frame_bytes_list.append(await value.read())

        if not frame_bytes_list:
            raise HTTPException(status_code=400, detail="Chunk khong co frame nao.")

        parse_seconds = time.perf_counter() - parse_start
        enqueue_start = time.perf_counter()
        payload = await asyncio.to_thread(
            session.process_chunk_bytes,
            frame_bytes_list=frame_bytes_list,
            start_ts_ms=start_ts_ms,
            end_ts_ms=end_ts_ms,
            chunk_index=chunk_index,
        )
        enqueue_seconds = time.perf_counter() - enqueue_start
        endpoint_seconds = time.perf_counter() - endpoint_start
        await asyncio.to_thread(
            session.record_ingest_metrics,
            parse_seconds=parse_seconds,
            enqueue_seconds=enqueue_seconds,
            endpoint_seconds=endpoint_seconds,
        )
        return JSONResponse(payload)

    return app
