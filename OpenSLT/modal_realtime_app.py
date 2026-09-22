from __future__ import annotations

import os
from pathlib import Path

import modal

ROOT_DIR = Path(__file__).resolve().parent
APP_NAME = os.environ.get("SIGN_TRANSLATION_MODAL_REALTIME_APP_NAME", "openslt-realtime")
MODEL_CACHE_VOLUME_NAME = os.environ.get("SIGN_TRANSLATION_MODAL_CACHE_VOLUME", "openslt-model-cache")
HF_SECRET_NAME = os.environ.get("SIGN_TRANSLATION_MODAL_HF_SECRET_NAME", "openslt-hf")
REALTIME_PRESET = os.environ.get("SIGN_TRANSLATION_REALTIME_PRESET", "split_mp2_cpu10").strip().lower()
PRESET_DEFAULTS: dict[str, dict[str, str]] = {
    "split_sota_default": {
        "SIGN_TRANSLATION_REALTIME_PIPELINE_MODE": "split",
        "SIGN_TRANSLATION_REALTIME_RAW_WORKERS": "1",
    },
    "split_mp2_cpu10": {
        "SIGN_TRANSLATION_REALTIME_PIPELINE_MODE": "split",
        "SIGN_TRANSLATION_REALTIME_RAW_WORKERS": "2",
        "SIGN_TRANSLATION_MEDIAPIPE_PARALLEL_DETECTORS": "0",
        "SIGN_TRANSLATION_MEDIAPIPE_PARALLEL_WORKERS": "1",
    },
}
PRESET_CPU_DEFAULTS = {
    "split_sota_default": "",
    "split_mp2_cpu10": "10",
}
if REALTIME_PRESET not in PRESET_DEFAULTS:
    REALTIME_PRESET = "split_mp2_cpu10"
CPU_REQUEST_RAW = os.environ.get("SIGN_TRANSLATION_MODAL_CPU", PRESET_CPU_DEFAULTS[REALTIME_PRESET]).strip()
FORWARDED_REALTIME_ENV = {
    **PRESET_DEFAULTS[REALTIME_PRESET],
    **{
        key: value
        for key, value in os.environ.items()
        if (key.startswith("SIGN_TRANSLATION_REALTIME_") or key.startswith("SIGN_TRANSLATION_MEDIAPIPE_")) and value.strip()
    },
}
LOCAL_SOURCE_IGNORE = [
    ".git",
    ".runtime_cache",
    "__pycache__",
    "*.pyc",
    "*.sqlite3",
    "db.sqlite3",
    "runserver*.log",
    "*.log",
    "media",
    "staticfiles",
]

app = modal.App(APP_NAME)
model_cache_volume = modal.Volume.from_name(MODEL_CACHE_VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "ffmpeg",
        "git",
        "libglib2.0-0",
        "libgl1",
        "libsm6",
        "libxext6",
        "libxrender1",
        "libgomp1",
    )
    .run_commands(
        "python -m pip install \"pip<24.1\" setuptools wheel",
        "python -m pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118",
    )
    .pip_install_from_requirements(str(ROOT_DIR / "modal_requirements.txt"))
    .env(
        {
            "PYTHONPATH": "/root/OpenSLT",
            "DJANGO_SETTINGS_MODULE": "config.settings",
            "DJANGO_RUNTIME_ROOT": "/cache/runtime",
            "DJANGO_DB_PATH": "/cache/runtime/db.sqlite3",
            "DJANGO_MEDIA_ROOT": "/cache/runtime/media",
            "DJANGO_STATIC_ROOT": "/cache/runtime/staticfiles",
            "SIGN_TRANSLATION_CACHE_ROOT": "/cache",
            "MPLBACKEND": "agg",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            **FORWARDED_REALTIME_ENV,
        }
    )
    .add_local_dir(str(ROOT_DIR), remote_path="/root/OpenSLT", ignore=LOCAL_SOURCE_IGNORE)
)

FUNCTION_KWARGS = {
    "image": image,
    "gpu": "T4",
    "timeout": 900,
    "startup_timeout": 1200,
    "min_containers": 1,
    "max_containers": 1,
    "scaledown_window": 900,
    "volumes": {"/cache": model_cache_volume},
    "secrets": [hf_secret],
}
if CPU_REQUEST_RAW:
    FUNCTION_KWARGS["cpu"] = float(CPU_REQUEST_RAW)


@app.function(**FUNCTION_KWARGS)
@modal.asgi_app()
def realtime_app():
    from realtime.server_bootstrap import create_modal_web_app

    return create_modal_web_app()
