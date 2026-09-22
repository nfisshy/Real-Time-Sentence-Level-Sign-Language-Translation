# OpenSLT Realtime

## Muc tieu

Repo nay gio chi giu **realtime path** va 2 method chinh thuc:

1. `split_mp2_cpu10` (mac dinh)
- `pipeline_mode=split`
- `raw_worker_count=2`
- `SIGN_TRANSLATION_MODAL_CPU=10`
- `SIGN_TRANSLATION_MEDIAPIPE_PARALLEL_DETECTORS=0`
- `SIGN_TRANSLATION_MEDIAPIPE_PARALLEL_WORKERS=1`

2. `split_sota_default`
- `pipeline_mode=split`
- `raw_worker_count=1`
- khong set `SIGN_TRANSLATION_MODAL_CPU`

Khong con giu cac nhanh thu nghiem `baseline`, `downscale`, `video_mode`, `cpu sweep`, `method matrix` hay telemetry UI debug.
Khong con giu dashboard upload video, job history UI, study library hay `SignStudy`.

## Thanh phan chinh

- [modal_realtime_app.py](/C:/Users/hoanghiworld/Downloads/TTIC-SHuBERT-ASLVideo-to-EnglishText/OpenSLT/modal_realtime_app.py): Modal ASGI entrypoint
- [realtime/ws_server.py](/C:/Users/hoanghiworld/Downloads/TTIC-SHuBERT-ASLVideo-to-EnglishText/OpenSLT/realtime/ws_server.py): FastAPI app va UI browser
- [realtime/runtime.py](/C:/Users/hoanghiworld/Downloads/TTIC-SHuBERT-ASLVideo-to-EnglishText/OpenSLT/realtime/runtime.py): shared runtime cho MediaPipe, DINO, T5
- [realtime/session.py](/C:/Users/hoanghiworld/Downloads/TTIC-SHuBERT-ASLVideo-to-EnglishText/OpenSLT/realtime/session.py): session chunk pipeline va queue
- [ai_core/kpe_mediapipe.py](/C:/Users/hoanghiworld/Downloads/TTIC-SHuBERT-ASLVideo-to-EnglishText/OpenSLT/ai_core/kpe_mediapipe.py): MediaPipe detector path duy nhat
- [realtime/benchmark_modal_compare.py](/C:/Users/hoanghiworld/Downloads/TTIC-SHuBERT-ASLVideo-to-EnglishText/OpenSLT/realtime/benchmark_modal_compare.py): benchmark chinh thuc so 2 method

## Thu muc con lai

- `OpenSLT/realtime`: web app va benchmark realtime
- `OpenSLT/ai_core`: model helpers can thiet cho realtime
- `OpenSLT/sign_translation_engine`: runtime bootstrap va pipeline model
- `OpenSLT/config`: minimal Django settings package chi de bootstrap runtime
- `OpenSLT/media`: clip benchmark/local media can thiet

Tat ca dashboard upload video, job history, study library, SignStudy, docs batch cu va scripts web cu da bi loai bo.

## Browser flow

- camera input `30 fps`
- browser sample `15 fps`
- `15 frame = 1 chunk`
- GUI chunk len server qua `POST /api/chunk/{session_id}`
- session xu ly theo split pipeline
- preview text hien o `Ban Nhap Dang Nghe`
- final text hien o `Ban Da Chot`

## Split pipeline

Ca 2 method deu dung:
- `raw queue`
- `prepared queue`
- `GPU worker`

Khac nhau duy nhat:
- `split_sota_default`: `raw_worker_count=1`
- `split_mp2_cpu10`: `raw_worker_count=2`

## Deploy presets

### Mac dinh: split_mp2_cpu10

```powershell
$env:SIGN_TRANSLATION_MODAL_REALTIME_APP_NAME="openslt-realtime"
$env:SIGN_TRANSLATION_REALTIME_PRESET="split_mp2_cpu10"
modal deploy OpenSLT\modal_realtime_app.py
```

### split_sota_default

```powershell
$env:SIGN_TRANSLATION_MODAL_REALTIME_APP_NAME="openslt-realtime"
$env:SIGN_TRANSLATION_REALTIME_PRESET="split_sota_default"
modal deploy OpenSLT\modal_realtime_app.py
```

Luu y:
- `SIGN_TRANSLATION_MEDIAPIPE_METHOD="sota"` khong con can nua.
- Repo hien tai chi con `sota` path trong MediaPipe, nen bien nay neu co set cung se bi bo qua.

## Benchmark chinh thuc

Chi con 1 benchmark chinh thuc:

```powershell
python OpenSLT\realtime\benchmark_modal_compare.py --stop-apps
```

Benchmark nay:
- dung `realtime_browser_e2e`
- replay clip `job_35`
- so sanh dung 2 method:
  - `split_sota_default`
  - `split_mp2_cpu10`

## Metrics can doc

- `time_to_first_draft`
- `time_to_final_text`
- `chunk_ack_latency_p50/p95`
- `chunks_processed`
- `chunks_coalesced`
- `raw_queue_wait_seconds_total`
- `prepared_queue_wait_seconds_total`
- `final_text`
- `normalized_text_match`

## Scripts giu lai

- `start_openslt_modal_realtime.ps1`
- `stop_openslt.ps1`
