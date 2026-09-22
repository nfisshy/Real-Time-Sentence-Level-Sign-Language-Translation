# Kiến Trúc End-to-End: SHuBERT ASL → English Realtime

> [!IMPORTANT]
> Tài liệu này mô tả kiến trúc hoạt động **toàn bộ hệ thống từ đầu đến cuối**, focus vào preset mặc định **`split_mp2_cpu10`** với **2 MediaPipe raw worker**.

---

## 1. Tổng Quan Hệ Thống

```mermaid
graph LR
    subgraph "Browser (Client)"
        A["📷 Webcam 30fps"] --> B["🎬 Sample 15fps"]
        B --> C["📦 Gom 15 frame = 1 Chunk"]
        C --> D["📤 POST /api/chunk"]
    end

    subgraph "Modal Cloud (T4 GPU + 10 vCPU)"
        D --> E["🌐 FastAPI Server"]
        E --> F["📋 Raw Queue"]
        F --> G["⚙️ Raw Worker 0\n(MediaPipe)"]
        F --> H["⚙️ Raw Worker 1\n(MediaPipe)"]
        G --> I["🔄 Reorder Buffer"]
        H --> I
        I --> J["📋 Prepared Queue"]
        J --> K["🎮 GPU Worker\n(DINOv2 + SHuBERT + T5)"]
        K --> L["📝 Text Output"]
    end

    L --> M["🖥️ Browser UI\nBản Đã Chốt / Bản Nháp"]

    style A fill:#4CAF50,color:white
    style K fill:#FF5722,color:white
    style G fill:#2196F3,color:white
    style H fill:#2196F3,color:white
```

---

## 2. Kiến Trúc Deployment

### 2.1 Modal Container

```mermaid
graph TB
    subgraph "Modal Function: realtime_app"
        direction TB
        M1["modal_realtime_app.py"] --> M2["server_bootstrap.py"]
        M2 --> M3["create_realtime_app()"]
        M3 --> M4["FastAPI App"]

        subgraph "Resources"
            R1["GPU: T4"]
            R2["CPU: 10 vCPU"]
            R3["Volume: /cache\n(model weights)"]
            R4["Secret: HF Token"]
        end
    end

    subgraph "Preset: split_mp2_cpu10"
        P1["pipeline_mode = split"]
        P2["raw_worker_count = 2"]
        P3["MEDIAPIPE_PARALLEL_DETECTORS = 0\n(sequential per frame)"]
        P4["MEDIAPIPE_PARALLEL_WORKERS = 1"]
    end
```

| Thông số | Giá trị |
|---|---|
| **Image** | `debian_slim` + Python 3.10 + PyTorch 2.2.0 (cu118) |
| **GPU** | NVIDIA T4 (16GB VRAM) |
| **CPU** | 10 vCPU |
| **Min/Max Containers** | 1/1 |
| **Timeout** | 900s |
| **Startup Timeout** | 1200s |
| **Scaledown Window** | 900s |

### 2.2 Startup Warmup Sequence

Khi container khởi động, hệ thống chạy **background warmup** để load tất cả model trước khi nhận request:

```mermaid
sequenceDiagram
    participant App as FastAPI Startup
    participant W as Warmup Thread
    participant HF as HuggingFace Hub
    participant GPU as GPU Memory

    App->>W: start_runtime_warmup()
    Note over W: Status: "warming"

    W->>HF: snapshot_download(models/*)
    HF-->>W: Model artifacts downloaded

    W->>W: Django setup()
    W->>W: get_model_config()

    W->>W: Warm MediaPipe detectors
    Note over W: Face Landmarker + Hand Landmarker + Holistic (Pose)

    W->>GPU: Load DINOv2 Hand Embedder
    Note over GPU: dinov2_vits14_reg + dinov2hand.pth

    W->>GPU: Load DINOv2 Face Embedder
    Note over GPU: dinov2_vits14_reg + dinov2face.pth

    W->>GPU: Load SLT Translation Model
    Note over GPU: SignLanguageByT5 + ByT5Tokenizer

    Note over W: Status: "ready" ✅
```

**Model artifacts trên Volume `/cache`:**

| Model | File | Device | Mô tả |
|---|---|---|---|
| MediaPipe Face | `face_landmarker_v2_with_blendshapes.task` | CPU | Phát hiện 478 face landmarks |
| MediaPipe Hand | `hand_landmarker.task` | CPU | Phát hiện 21 hand landmarks × 2 tay |
| MediaPipe Pose | Built-in Holistic | CPU | 33 pose landmarks |
| DINOv2 Hand | `dinov2hand.pth` | GPU | ViT-S/14 fine-tuned, output 384-dim |
| DINOv2 Face | `dinov2face.pth` | GPU | ViT-S/14 fine-tuned, output 384-dim |
| SHuBERT | `checkpoint_836_400000.pt` | GPU | SignHuBERT encoder (768-dim repr) |
| SLT ByT5 | `checkpoint-11625/` | GPU | ByT5-base conditional generation |
| Tokenizer | `byt5_base/` | CPU | ByT5Tokenizer (byte-level) |

---

## 3. Browser Client Flow

### 3.1 Capture Pipeline

```mermaid
sequenceDiagram
    participant Cam as Camera (30fps)
    participant Timer as setInterval (33ms)
    participant Canvas as Canvas 640×480
    participant Chunk as Chunk Builder
    participant Net as fetch() POST

    Timer->>Timer: Check SAMPLE_INTERVAL (66ms = 15fps)
    Timer->>Canvas: drawImage(video, 640, 480)
    Canvas->>Canvas: toBlob("image/jpeg", 0.72)
    Canvas->>Chunk: push(blob)

    alt Chunk đủ 15 frame
        Chunk->>Net: POST /api/chunk/{session_id}
        Note over Net: FormData: chunk_index, start_ts_ms,<br/>end_ts_ms, frames[0..14]
        Net-->>Chunk: JSON payload (status, texts)
    end
```

| Thông số Browser | Giá trị |
|---|---|
| Camera FPS | 30 |
| Sample FPS | 15 (lấy mẫu cách 66ms) |
| Frame size | 640 × 480 |
| JPEG quality | 0.72 |
| Frames per chunk | 15 |
| Chunk interval | ~1 giây |

### 3.2 Session Lifecycle

```mermaid
stateDiagram-v2
    [*] --> CreateSession: POST /api/session
    CreateSession --> Streaming: session_id returned

    state Streaming {
        [*] --> Idle
        Idle --> Receiving: POST /api/chunk
        Receiving --> Processing: Chunks queued
        Processing --> LiveTranslation: Preview decode triggered
        LiveTranslation --> Processing: More chunks
        Processing --> Finalizing: Endpoint detected
        Finalizing --> Listening: Reset utterance
        Listening --> Receiving: New signing
    }

    Streaming --> Stopped: POST /api/session/{id}/stop
    Streaming --> Deleted: DELETE /api/session/{id}
    Streaming --> Timeout: 120s idle

    Stopped --> [*]
    Deleted --> [*]
    Timeout --> [*]
```

---

## 4. Split Pipeline Architecture (Core)

> [!NOTE]
> Đây là phần **quan trọng nhất** của kiến trúc. Preset `split_mp2_cpu10` tách pipeline thành **2 giai đoạn song song**: CPU-bound MediaPipe (2 worker) và GPU-bound inference (1 worker).

### 4.1 Thread Architecture

```mermaid
graph TB
    subgraph "FastAPI Async Thread Pool"
        AT["asyncio.to_thread\nprocess_chunk_bytes()"]
    end

    subgraph "Session Threads (per session)"
        direction TB

        subgraph "Raw Workers (CPU-bound)"
            RW0["🔵 Raw Worker 0\nThread: realtime-raw-0\nMediaPipe Context 0"]
            RW1["🔵 Raw Worker 1\nThread: realtime-raw-1\nMediaPipe Context 1"]
        end

        subgraph "GPU Worker"
            GW["🔴 GPU Worker\nThread: realtime-gpu\nDINOv2 + T5 Decode"]
        end
    end

    AT -->|"enqueue"| RQ["📋 Raw Queue\ndeque, max=2"]
    RQ -->|"pop"| RW0
    RQ -->|"pop"| RW1
    RW0 -->|"submit analyzed"| RB["🔄 Reorder Buffer\ndict by chunk_index"]
    RW1 -->|"submit analyzed"| RB
    RB -->|"in-order"| PQ["📋 Prepared Queue\ndeque, max=2"]
    PQ -->|"pop"| GW
    GW -->|"update"| LP["last_payload"]

    style RW0 fill:#2196F3,color:white
    style RW1 fill:#2196F3,color:white
    style GW fill:#FF5722,color:white
    style RQ fill:#FFC107,color:black
    style PQ fill:#FFC107,color:black
    style RB fill:#9C27B0,color:white
```

### 4.2 Tại sao 2 Worker + Reorder Buffer?

| Vấn đề | Giải pháp |
|---|---|
| MediaPipe chạy sequential trên CPU, mỗi frame ~25-40ms | 2 worker xử lý 2 chunk **song song** trên 2 CPU thread |
| GPU worker nhanh hơn CPU worker → GPU đói data | 2 worker cung cấp đủ data cho GPU liên tục |
| Out-of-order processing phá hỏng endpoint detection | **Reorder Buffer** (`_analyzed_chunks_by_index`) đảm bảo chunk vào prepared queue **đúng thứ tự** |
| Quá nhiều chunk backlog → latency tăng | Bounded queues (`chunk_queue_max=2`, `prepared_queue_max=2`) + coalesce khi đầy |

### 4.3 Chi Tiết Hoạt Động Từng Thread

#### Thread: FastAPI Ingest (asyncio)

```python
# ws_server.py - process_chunk endpoint
POST /api/chunk/{session_id}
  ├── Parse FormData (chunk_index, timestamps, JPEG frames)
  ├── asyncio.to_thread(session.process_chunk_bytes)
  │   ├── Normalize frame bytes
  │   ├── Lock _raw_queue_condition
  │   ├── if raw_queue >= chunk_queue_max:  # Backpressure!
  │   │   └── raw_queue[-1].extend(pending_chunk)  # Coalesce
  │   ├── else:
  │   │   └── raw_queue.append(pending_chunk)
  │   ├── notify() → wake raw worker
  │   └── return snapshot of last_payload
  └── record_ingest_metrics()
```

#### Thread: Raw Worker 0 & 1 (CPU-bound MediaPipe)

```mermaid
flowchart TD
    A["Wait on raw_queue\n(Condition.wait)"] --> B{"Queue empty\n& !stop?"}
    B -->|"yes"| A
    B -->|"chunk available"| C["Pop PendingChunk"]
    C --> D["_analyze_chunk_sync()"]

    subgraph "Analyze Chunk (per frame)"
        D --> E["JPEG decode\nnp.frombuffer + cv2.imdecode"]
        E --> F["BGR → RGB\ncv2.cvtColor"]
        F --> G["MediaPipe detect_frame_landmarks()"]

        subgraph "MediaPipe Detection (sequential)"
            G --> G1["Pose: mp.solutions.holistic.process()"]
            G --> G2["Face: FaceLandmarker.detect()"]
            G --> G3["Hand: HandLandmarker.detect()"]
        end

        G --> H["HandExtractor.extract_hand_crops_from_frame()"]
        H --> I["FaceExtractor.extract_face_crop_from_frame()"]
        I --> J["PoseProcessor.process_frame_landmarks()"]
        J --> K["→ AnalyzedFrame"]
    end

    K --> L["→ AnalyzedChunk\n(chunk_index, frames[])"]
    L --> M["_submit_analyzed_chunk()"]

    subgraph "Reorder & Enqueue"
        M --> N["Insert into _analyzed_chunks_by_index"]
        N --> O{"next_index\nin buffer?"}
        O -->|"yes"| P["Build PreparedChunk\n(endpoint detection here!)"]
        P --> Q["Enqueue to prepared_queue"]
        Q --> O
        O -->|"no"| R["Wait for missing chunk"]
    end

    R --> A
    Q --> A
```

> [!WARNING]
> **Endpoint detection** (`EndpointDetector.update()`) chạy **TRONG reorder phase** (`_build_prepared_chunk_from_analyzed`), KHÔNG phải trong raw worker. Điều này đảm bảo endpoint detection luôn nhận frame theo **đúng thứ tự thời gian**, dù 2 worker xử lý chunk nào trước.

#### Thread: GPU Worker (inference-bound)

```mermaid
flowchart TD
    A["Wait on prepared_queue\n(Condition.wait)"] --> B{"Queue empty\n& all workers done\n& stop?"}
    B -->|"yes, shutdown"| Z["Return (thread exit)"]
    B -->|"chunk available"| C["Pop PreparedChunk"]
    C --> D["_process_prepared_chunk_sync()"]

    subgraph "Process Events"
        D --> E{"Event kind?"}

        E -->|"collect"| F["embed_hands_batch()\nDINOv2 GPU"]
        F --> F2["embed_face_batch()\nDINOv2 GPU"]
        F2 --> F3["_append_embedding_sequences()\nupdate utterance deques"]
        F3 --> F4{"Đủ frame\ncho preview?"}
        F4 -->|"yes"| F5["Set _decode_dirty = True"]

        E -->|"finalize"| G["finalize()\nFinal decode beam=2"]
        G --> G2["transcript_segments.append(text)"]
        G2 --> G3["_reset_current_utterance()"]

        E -->|"reset"| H["_reset_current_utterance()\nClear all deques"]

        E -->|"status"| I["Update last_payload"]
    end

    F5 --> J{"Has backpressure?"}
    J -->|"yes"| K["Skip decode\n(catch up first)"]
    J -->|"no"| L["_run_preview_decode()"]

    subgraph "Preview Decode"
        L --> L1["Stack utterance embeddings"]
        L1 --> L2["SignLanguageByT5.generate()\nbeam=1, max_length=96"]
        L2 --> L3["ByT5Tokenizer.decode()"]
        L3 --> L4["TextStabilizer.update()"]
        L4 --> L5["Update draft_text"]
    end

    subgraph "Final Decode"
        G --> G4["Stack utterance embeddings"]
        G4 --> G5["SignLanguageByT5.generate()\nbeam=2, max_length=128"]
        G5 --> G6["ByT5Tokenizer.decode()"]
        G6 --> G7["transcript_segments.append()"]
    end

    K --> A
    L5 --> A
    G3 --> A
    H --> A
    I --> A
```

---

## 5. Data Transformations (Chi Tiết)

### 5.1 Pipeline Dữ Liệu Theo Từng Stage

```mermaid
graph LR
    subgraph "Stage 1: Browser"
        S1["JPEG bytes\n640×480\nquality=0.72"]
    end

    subgraph "Stage 2: Raw Worker"
        S2a["np.ndarray\nBGR 640×480"]
        S2b["RGB 640×480"]
        S2c["MediaPipe Landmarks\nface[478], hand[21×2], pose[33]"]
        S2d["Hand Crops\n224×224×3 RGB"]
        S2e["Face Crop\n224×224×3 Grey BG"]
        S2f["Pose Vector\n[7×2] = 14-dim float"]
    end

    subgraph "Stage 3: GPU Worker"
        S3a["Hand Embeddings\n384-dim float32\n(per frame)"]
        S3b["Face Embedding\n384-dim float32\n(per frame)"]
    end

    subgraph "Stage 4: SHuBERT + T5"
        S4a["SignHuBERT\n4-channel input\n→ 768-dim repr"]
        S4b["LinearAdapter\n→ d_model=512"]
        S4c["ByT5 Encoder\n6 layers"]
        S4d["ByT5 Decoder\n6 layers"]
        S4e["English Text\nbyte-level tokens"]
    end

    S1 --> S2a --> S2b --> S2c
    S2c --> S2d
    S2c --> S2e
    S2c --> S2f
    S2d --> S3a
    S2e --> S3b
    S3a --> S4a
    S3b --> S4a
    S2f --> S4a
    S4a --> S4b --> S4c --> S4d --> S4e
```

### 5.2 Feature Dimensions

| Stage | Data | Dimension | Notes |
|---|---|---|---|
| JPEG Input | Raw frame | 640×480×3 uint8 | JPEG quality 72% |
| MediaPipe Face | Landmarks | 478 × [x,y,z] | Normalized [0,1] |
| MediaPipe Hand | Landmarks | 21 × [x,y,z] × 2 hands | Wrist matching by pose |
| MediaPipe Pose | Landmarks | 33 × [x,y,z] | Used indices: [0,11,12,13,14,15,16] |
| Left Hand Crop | RGB image | 224×224×3 uint8 | Bounding box + scale 1.5 |
| Right Hand Crop | RGB image | 224×224×3 uint8 | Bounding box + scale 1.5 |
| Face Crop | Grey BG image | 224×224×3 uint8 | Eyes + mouth only on grey |
| Pose Embedding | Float vector | 14 float32 | 7 keypoints × 2 (x,y), normalized |
| DINOv2 Hand Emb | Float vector | 384 float32 | ViT-S/14 dinov2_vits14_reg |
| DINOv2 Face Emb | Float vector | 384 float32 | ViT-S/14 dinov2_vits14_reg |
| SHuBERT Input | 4-channel | [T, 384+384+384+14] | Concatenated channels |
| SHuBERT Output | Representations | [T, 768] × 12 layers | Weighted sum → 768-dim |
| LinearAdapter | Projected | [T, 512] | Match ByT5 d_model |
| ByT5 Output | Token IDs | Variable length | Byte-level tokenization |

---

## 6. Queue Mechanics

### 6.1 Raw Queue

```
┌─────────────────────────────────────────────────────────┐
│  Raw Queue  (deque, max=2)                              │
│                                                         │
│  Producer: FastAPI ingest thread                        │
│  Consumer: Raw Worker 0, Raw Worker 1                   │
│  Lock: _raw_queue_condition (RLock + Condition)         │
│                                                         │
│  ┌──────────┐    ┌──────────┐                          │
│  │ Chunk #N │    │ Chunk #M │    (← slots)             │
│  └──────────┘    └──────────┘                          │
│                                                         │
│  Backpressure: Khi queue đầy (≥2 chunks),              │
│  chunk mới được COALESCE vào chunk cuối                 │
│  (extend frame_bytes_list)                              │
│                                                         │
│  Kết quả: chunk bị gộp sẽ có nhiều frame hơn 15,      │
│  nhưng tránh mất data hoàn toàn                        │
└─────────────────────────────────────────────────────────┘
```

### 6.2 Reorder Buffer (Multi-worker Only)

```
┌─────────────────────────────────────────────────────────┐
│  Reorder Buffer  (dict: chunk_index → AnalyzedChunk)    │
│                                                         │
│  Vấn đề: Worker 0 xử lý chunk #5, Worker 1 xử lý     │
│  chunk #4. Worker 1 xong trước → chunk #5 đến trước #4 │
│                                                         │
│  Giải pháp: _next_analyzed_chunk_index = 4              │
│                                                         │
│  Buffer state: { 5: AnalyzedChunk#5 }                  │
│  → Chunk #5 chờ, vì #4 chưa có                        │
│                                                         │
│  Worker 0 xong chunk #4:                                │
│  Buffer state: { 4: AnalyzedChunk#4, 5: AC#5 }        │
│  → Pop #4, build PreparedChunk, enqueue                │
│  → Pop #5, build PreparedChunk, enqueue                │
│  → _next_analyzed_chunk_index = 6                       │
│                                                         │
│  KEY: Endpoint detection chạy ở đây, trong             │
│  _build_prepared_chunk_from_analyzed(), đảm bảo         │
│  frame order chính xác                                  │
└─────────────────────────────────────────────────────────┘
```

### 6.3 Prepared Queue

```
┌─────────────────────────────────────────────────────────┐
│  Prepared Queue  (deque, max=2)                         │
│                                                         │
│  Producer: Raw Workers (sau reorder)                    │
│  Consumer: GPU Worker (duy nhất 1 thread)               │
│  Lock: _prepared_queue_condition                        │
│                                                         │
│  ┌────────────────┐    ┌────────────────┐              │
│  │ PreparedChunk  │    │ PreparedChunk  │              │
│  │ events: [      │    │ events: [      │              │
│  │   collect,     │    │   collect,     │              │
│  │   status,      │    │   finalize,    │              │
│  │   collect      │    │   reset        │              │
│  │ ]              │    │ ]              │              │
│  └────────────────┘    └────────────────┘              │
│                                                         │
│  Backpressure: Raw worker BLOCK (wait) khi             │
│  prepared queue đầy (≥2). Không coalesce.              │
│                                                         │
│  GPU Worker exit condition:                             │
│  prepared_queue empty AND worker_should_stop            │
│  AND raw_queue empty AND raw_workers_active==0          │
│  AND analyzed_chunks_by_index empty                     │
└─────────────────────────────────────────────────────────┘
```

---

## 7. Endpoint Detection

Endpoint detection quyết định khi nào một câu ký hiệu **kết thúc** để trigger final decode.

```mermaid
stateDiagram-v2
    [*] --> idle: Chưa thấy hand

    idle --> active: hand_visible = True
    active --> active: hand_visible = True\n(active_frames++)

    active --> uncertain: hand_visible = False\nAND seen_activity\n(idle_frames++)

    uncertain --> active: hand_visible = True\nOR (motion > 0.018 AND pose_visible)

    uncertain --> endpoint: idle_frames >= 4\nAND active_frames >= 4
    Note right of endpoint: should_finalize = True\n→ Final decode (beam=2)

    uncertain --> reset: idle_frames >= 8
    Note right of reset: Clear utterance\n→ Listening

    endpoint --> [*]
    reset --> idle
```

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `endpoint_idle_frames` | 4 | Số frame liên tiếp không thấy hand → finalize |
| `endpoint_reset_frames` | 8 | Số frame không thấy hand → reset hoàn toàn |
| `endpoint_min_active_frames` | 4 | Tối thiểu frames có hand trước khi finalize |
| `endpoint_motion_threshold` | 0.018 | Ngưỡng motion energy để duy trì continuity |

---

## 8. Preview vs Final Decode

```mermaid
graph TB
    subgraph "Preview Decode"
        PA["Trigger:\nutterance_frames >= 30\nthen every +15 frames"]
        PB["Params:\nnum_beams = 1\nmax_length = 96"]
        PC["Stabilizer:\nCommon prefix tracking"]
        PD["Output → draft_text\n(Bản Nháp Đang Nghe)"]
        PA --> PB --> PC --> PD
    end

    subgraph "Final Decode"
        FA["Trigger:\nEndpoint detected\nOR session stop"]
        FB["Params:\nnum_beams = 2\nmax_length = 128"]
        FC["Output → transcript_segments[]\n(Bản Đã Chốt)"]
        FA --> FB --> FC
    end

    subgraph "Backpressure Guard"
        BG["Nếu raw_queue hoặc prepared_queue\ncòn chunk chờ → SKIP preview decode\n(ưu tiên catch up)"]
    end

    style PA fill:#FF9800,color:white
    style FA fill:#4CAF50,color:white
    style BG fill:#f44336,color:white
```

### Text Stabilizer

Class `TextStabilizer` giữ 2 buffer:
- **`_previous_tokens`**: Hypothesis lần trước
- **`_stable_tokens`**: Common prefix ổn định qua nhiều decode

Logic: Chỉ hiển thị text khi nó lặp lại ổn định → tránh flickering trên UI.

---

## 9. Shared Runtime & Lock Strategy

### 9.1 SharedRealtimeRuntime

```mermaid
graph TB
    subgraph "SharedRealtimeRuntime (Singleton)"
        direction LR
        HL["hand_lock\n(threading.Lock)"]
        FL["face_lock\n(threading.Lock)"]
        DL["decode_lock\n(threading.Lock)"]
    end

    subgraph "GPU Operations"
        HE["embed_hands_batch()\nDINOv2 Hand"]
        FE["embed_face_batch()\nDINOv2 Face"]
        DE["decode()\nSHuBERT + T5"]
    end

    HL -->|"protects"| HE
    FL -->|"protects"| FE
    DL -->|"protects"| DE

    style HL fill:#FF5722,color:white
    style FL fill:#FF5722,color:white
    style DL fill:#FF5722,color:white
```

> [!TIP]
> Mỗi GPU operation có **lock riêng**. Trong thực tế, chỉ có **1 GPU worker** nên không có contention. Nhưng lock vẫn cần thiết vì `decode()` được gọi từ cả GPU worker thread lẫn `finalize()` thread.

### 9.2 Session State Lock

```
_state_lock (RLock)
├── _raw_queue_condition (Condition using _state_lock)
│   ├── Producer: FastAPI ingest → notify()
│   └── Consumer: Raw Workers → wait()
│
└── _prepared_queue_condition (Condition using _state_lock)
    ├── Producer: Raw Workers (sau reorder) → notify_all()
    └── Consumer: GPU Worker → wait()
```

---

## 10. Sequence Diagram End-to-End

```mermaid
sequenceDiagram
    participant B as Browser
    participant F as FastAPI
    participant RQ as Raw Queue
    participant RW0 as Raw Worker 0
    participant RW1 as Raw Worker 1
    participant ROB as Reorder Buffer
    participant PQ as Prepared Queue
    participant GPU as GPU Worker
    participant UI as Browser UI

    B->>F: POST /api/chunk (15 JPEG frames)
    F->>RQ: enqueue PendingChunk(chunk_index=0)

    B->>F: POST /api/chunk (chunk_index=1)
    F->>RQ: enqueue PendingChunk(chunk_index=1)

    RW0->>RQ: pop chunk_index=0
    RW1->>RQ: pop chunk_index=1

    par Worker 0 processes chunk 0
        RW0->>RW0: JPEG decode × 15 frames
        RW0->>RW0: MediaPipe detect × 15 frames
        RW0->>RW0: Crop hands, face × 15 frames
        RW0->>RW0: Pose embedding × 15 frames
    and Worker 1 processes chunk 1
        RW1->>RW1: JPEG decode × 15 frames
        RW1->>RW1: MediaPipe detect × 15 frames
        RW1->>RW1: Crop hands, face × 15 frames
        RW1->>RW1: Pose embedding × 15 frames
    end

    Note over RW1: Worker 1 finishes first
    RW1->>ROB: submit AnalyzedChunk(index=1)
    Note over ROB: Waiting for index=0...

    RW0->>ROB: submit AnalyzedChunk(index=0)
    Note over ROB: Index 0 ready → process in order

    ROB->>ROB: _build_prepared_chunk(chunk 0)
    Note over ROB: EndpointDetector.update() per frame
    ROB->>PQ: enqueue PreparedChunk(0)

    ROB->>ROB: _build_prepared_chunk(chunk 1)
    ROB->>PQ: enqueue PreparedChunk(1)

    GPU->>PQ: pop PreparedChunk(0)
    GPU->>GPU: DINOv2 embed_hands_batch()
    GPU->>GPU: DINOv2 embed_face_batch()
    GPU->>GPU: append to utterance deques
    GPU->>GPU: Check: utterance >= 30 frames?
    GPU->>GPU: Preview decode (beam=1)
    GPU->>GPU: TextStabilizer.update()
    GPU-->>UI: last_payload (draft_text)

    GPU->>PQ: pop PreparedChunk(1)
    GPU->>GPU: Process events...
    Note over GPU: Endpoint detected! (finalize event)
    GPU->>GPU: Final decode (beam=2)
    GPU->>GPU: transcript_segments.append(text)
    GPU-->>UI: last_payload (final_text)
```

---

## 11. Cấu Trúc File Quan Trọng

```
OpenSLT/
├── modal_realtime_app.py          # Modal deployment entry point
│                                   # Preset selection, image build, T4 GPU
│
├── realtime/
│   ├── server_bootstrap.py        # create_modal_web_app() wrapper
│   ├── config.py                  # RealtimeConfig dataclass (all tuning params)
│   ├── ws_server.py               # FastAPI app, HTTP endpoints, browser HTML/JS
│   ├── session.py                 # RealtimeSession: queues, workers, pipeline
│   ├── runtime.py                 # SharedRealtimeRuntime: model warmup & locks
│   ├── endpoint.py                # EndpointDetector: utterance boundary FSM
│   ├── text_stabilizer.py         # TextStabilizer: draft text flickering guard
│   └── benchmark_modal_compare.py # Official benchmark: mp2 vs sota
│
├── ai_core/
│   ├── kpe_mediapipe.py           # HolisticDetector: Face+Hand+Pose detection
│   ├── crop_hands.py              # HandExtractor: hand crop from landmarks
│   ├── crop_face.py               # FaceExtractor: face cue crop (eyes+mouth)
│   ├── body_features.py           # PoseProcessor: pose keypoint normalization
│   ├── dinov2_features.py         # DINOEmbedder: ViT-S/14 embedding extraction
│   ├── inference.py               # SignLanguageByT5: SHuBERT + T5 generation
│   └── shubert.py                 # SignHubertModel: multi-channel encoder
│
├── sign_translation_engine/
│   └── runtime/
│       └── bootstrap.py           # HF model download, cache setup, config
│
└── config/
    └── settings.py                # Django settings (DB, cache, HF token)
```

---

## 12. Tham Số Cấu Hình Chính

### Preset: `split_mp2_cpu10`

| Nhóm | Tham số | Giá trị | Mô tả |
|---|---|---|---|
| **Pipeline** | `pipeline_mode` | `split` | Tách CPU/GPU workers |
| **Pipeline** | `raw_worker_count` | `2` | ← **Core: 2 MP workers** |
| **MediaPipe** | `PARALLEL_DETECTORS` | `0` (off) | Pose/Face/Hand chạy sequential per frame |
| **MediaPipe** | `PARALLEL_WORKERS` | `1` | ThreadPool per detector = 1 |
| **Camera** | `camera_input_fps` | `30` | Camera hardware FPS |
| **Camera** | `target_fps` | `15` | Browser sampling FPS |
| **Camera** | `frame_width × height` | `640 × 480` | Frame resolution |
| **Chunk** | `chunk_frames` | `15` | Frames per chunk (~1s) |
| **Queue** | `chunk_queue_max` | `2` | Raw queue bound |
| **Queue** | `prepared_queue_max` | `2` | Prepared queue bound |
| **Utterance** | `recent_window_frames` | `16` | Sliding window for recent embeddings |
| **Utterance** | `max_utterance_frames` | `128` | Max frames per utterance (~8.5s) |
| **Preview** | `preview_initial_frames` | `30` | First decode at 30 frames (~2s) |
| **Preview** | `preview_stride_frames` | `15` | Decode every +15 frames |
| **Preview** | `preview_num_beams` | `1` | Greedy decode for speed |
| **Final** | `final_num_beams` | `2` | Beam search for quality |
| **Final** | `final_max_length` | `128` | Max output tokens |
| **Endpoint** | `endpoint_idle_frames` | `4` | Frames without hand → finalize |
| **Endpoint** | `endpoint_reset_frames` | `8` | Frames without hand → reset |
| **Session** | `session_idle_timeout_ms` | `120000` | Auto-cleanup after 2 min idle |

---

## 13. Bottleneck Analysis

```mermaid
pie title CPU Time Distribution (per chunk ~15 frames)
    "MediaPipe Pose" : 35
    "MediaPipe Hand" : 25
    "MediaPipe Face" : 20
    "JPEG Decode" : 8
    "Crop & Resize" : 7
    "BGR→RGB Convert" : 5
```

```mermaid
pie title GPU Time Distribution (per chunk)
    "DINOv2 Hand Embed (×2)" : 30
    "DINOv2 Face Embed" : 15
    "SHuBERT Encode" : 20
    "T5 Decode (preview)" : 25
    "T5 Decode (final)" : 10
```

> [!IMPORTANT]
> **Bottleneck chính** nằm ở CPU MediaPipe detection (~80% CPU time). Đó là lý do **2 raw workers** hiệu quả: trong khi Worker 0 đang detect frame, Worker 1 detect chunk khác → tổng throughput tăng gần **2x** (giới hạn bởi GIL release trong C++ MediaPipe backend).

---

## 14. Tóm Tắt Flow Hoàn Chỉnh

```
📷 Camera 30fps
    ↓ (browser sample 15fps)
📦 15 frames → 1 chunk JPEG
    ↓ (HTTP POST)
🌐 FastAPI parse FormData
    ↓ (enqueue)
📋 Raw Queue [max=2, coalesce khi đầy]
    ↓ (2 workers cạnh tranh pop)
⚙️ Raw Worker 0 ──┬── JPEG decode
⚙️ Raw Worker 1 ──┘   BGR→RGB
                       MediaPipe (Pose+Face+Hand sequential)
                       Crop hands 224×224
                       Crop face 224×224 (eyes+mouth on grey)
                       Pose vector 14-dim
    ↓ (submit to reorder buffer)
🔄 Reorder Buffer [dict by chunk_index]
    ↓ (drain in-order, run endpoint detection)
📋 Prepared Queue [max=2, block khi đầy]
    ↓ (GPU worker pop)
🎮 GPU Worker
    ├── DINOv2 embed hands → 384-dim × 2
    ├── DINOv2 embed face → 384-dim
    ├── Append to utterance deques
    ├── [Preview] SHuBERT → T5 beam=1 → draft_text
    └── [Final]   SHuBERT → T5 beam=2 → final_text
    ↓
📝 last_payload {status, draft_text, final_text}
    ↓ (HTTP response / next chunk response)
🖥️ Browser UI
    ├── "Bản Đã Chốt" ← final_text
    └── "Bản Nháp Đang Nghe" ← draft_text
```
