from __future__ import annotations

import os
from pathlib import Path


def create_modal_web_app():
    from .config import RealtimeConfig
    from .ws_server import create_realtime_app

    runtime_root = Path(os.environ["DJANGO_RUNTIME_ROOT"])
    (runtime_root / "media").mkdir(parents=True, exist_ok=True)
    (runtime_root / "staticfiles").mkdir(parents=True, exist_ok=True)
    return create_realtime_app(RealtimeConfig.from_env())
