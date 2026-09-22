import os
import shutil
import threading
import logging
from pathlib import Path

import huggingface_hub
import torch
from django.conf import settings

logger = logging.getLogger(__name__)

_MODEL_CONFIG = None
_MODEL_CONFIG_LOCK = threading.Lock()
_ORIGINAL_TORCH_HUB_LOAD = None
_TORCH_HUB_PATCH_INSTALLED = False


def _ai_core_root() -> Path:
    return Path(__file__).resolve().parents[2] / "ai_core"


def setup_cache_directories():
    cache_root = Path(settings.SIGN_TRANSLATION_CACHE_ROOT)
    cache_root.mkdir(parents=True, exist_ok=True)

    cache_dirs = {
        "MPLCONFIGDIR": cache_root / "matplotlib",
        "TRANSFORMERS_CACHE": cache_root / "huggingface",
        "HF_HOME": cache_root / "huggingface",
        "FONTCONFIG_PATH": cache_root / "fontconfig",
        "TORCH_HOME": cache_root / "torch",
        "XDG_CACHE_HOME": cache_root / "cache",
    }

    for env_var, path in cache_dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[env_var] = str(path)

    torch_hub_dir = cache_dirs["TORCH_HOME"] / "hub"
    torch_hub_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Runtime cache dirs ready under %s", cache_root)
    return cache_dirs


def copy_dinov2_files_if_needed():
    src_dir = _ai_core_root()
    target_dir = Path(os.environ["TORCH_HOME"]) / "hub" / "facebookresearch_dinov2_main" / "dinov2" / "layers"
    if not target_dir.exists():
        return False

    copied = False
    for filename in ("attention.py", "block.py"):
        src_path = src_dir / filename
        target_path = target_dir / filename
        if src_path.exists():
            shutil.copy2(src_path, target_path)
            copied = True
    return copied


def install_torch_hub_patch_once():
    global _ORIGINAL_TORCH_HUB_LOAD, _TORCH_HUB_PATCH_INSTALLED

    if _TORCH_HUB_PATCH_INSTALLED:
        return

    _ORIGINAL_TORCH_HUB_LOAD = torch.hub.load

    def patched_hub_load(repo_or_dir, model, *args, **kwargs):
        loaded = _ORIGINAL_TORCH_HUB_LOAD(repo_or_dir, model, *args, **kwargs)
        repo_repr = str(repo_or_dir).lower()
        if "facebookresearch/dinov2" in repo_repr or "dinov2" in repo_repr:
            copy_dinov2_files_if_needed()
        return loaded

    torch.hub.load = patched_hub_load
    _TORCH_HUB_PATCH_INSTALLED = True


def _download_models():
    token = settings.SIGN_TRANSLATION_HF_TOKEN
    if not token:
        raise RuntimeError("HF_TOKEN chua duoc cau hinh cho Django runtime.")

    logger.info("Starting Hugging Face snapshot download for %s", settings.SIGN_TRANSLATION_MODEL_REPO)
    models_path = huggingface_hub.snapshot_download(
        repo_id=settings.SIGN_TRANSLATION_MODEL_REPO,
        allow_patterns="models/*",
        ignore_patterns=[
            "models/checkpoint-11625/optimizer.pt",
            "models/checkpoint-11625/scheduler.pt",
            "models/checkpoint-11625/trainer_state.json",
            "models/checkpoint-11625/training_args.bin",
            "models/checkpoint-11625/rng_state_*",
        ],
        token=token,
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
    )
    logger.info("Finished Hugging Face snapshot download into %s", models_path)

    config = {
        "dino_face_model_path": os.path.join(models_path, "models/dinov2face.pth"),
        "dino_hands_model_path": os.path.join(models_path, "models/dinov2hand.pth"),
        "mediapipe_face_model_path": os.path.join(models_path, "models/face_landmarker_v2_with_blendshapes.task"),
        "mediapipe_hands_model_path": os.path.join(models_path, "models/hand_landmarker.task"),
        "shubert_model_path": os.path.join(models_path, "models/checkpoint_836_400000.pt"),
        "slt_model_config": os.path.join(models_path, "models/byt5_base/config.json"),
        "slt_model_checkpoint": os.path.join(models_path, "models/checkpoint-11625"),
        "slt_tokenizer_checkpoint": os.path.join(models_path, "models/byt5_base"),
        "temp_dir": str(settings.SIGN_TRANSLATION_CACHE_ROOT),
    }

    missing = [path for path in config.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Khong tim thay model artifacts: {missing}")

    return config


def get_model_config():
    global _MODEL_CONFIG
    if _MODEL_CONFIG is not None:
        return _MODEL_CONFIG

    with _MODEL_CONFIG_LOCK:
        if _MODEL_CONFIG is not None:
            return _MODEL_CONFIG
        logger.info("Initializing model config")
        setup_cache_directories()
        install_torch_hub_patch_once()
        copy_dinov2_files_if_needed()
        _MODEL_CONFIG = _download_models()
        logger.info("Model config initialized")
        return _MODEL_CONFIG
