import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent
RUNTIME_ROOT = Path(os.environ.get("DJANGO_RUNTIME_ROOT", str(BASE_DIR)))

SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "django-insecure-sign-translation-dev-key-change-me",
)
DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = [host.strip() for host in os.environ.get("DJANGO_ALLOWED_HOSTS", "*").split(",") if host.strip()]

_csrf_trusted_origins = [
    origin.strip()
    for origin in os.environ.get("DJANGO_CSRF_TRUSTED_ORIGINS", "").split(",")
    if origin.strip()
]
if DEBUG:
    _csrf_trusted_origins.extend(
        [
            "https://*.trycloudflare.com",
            "http://127.0.0.1:8000",
            "http://localhost:8000",
        ]
    )
CSRF_TRUSTED_ORIGINS = list(dict.fromkeys(_csrf_trusted_origins))

INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.contenttypes",
]

MIDDLEWARE: list[str] = []

ROOT_URLCONF = "config.urls"

TEMPLATES: list[dict] = []

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": Path(os.environ.get("DJANGO_DB_PATH", str(RUNTIME_ROOT / "db.sqlite3"))),
    }
}

AUTH_PASSWORD_VALIDATORS = []

LANGUAGE_CODE = "vi"
TIME_ZONE = "Asia/Ho_Chi_Minh"
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATIC_ROOT = Path(os.environ.get("DJANGO_STATIC_ROOT", str(RUNTIME_ROOT / "staticfiles")))
MEDIA_URL = "/media/"
MEDIA_ROOT = Path(os.environ.get("DJANGO_MEDIA_ROOT", str(RUNTIME_ROOT / "media")))

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
    "loggers": {
        "django": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "sign_translation_engine": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

SIGN_TRANSLATION_MODEL_REPO = os.environ.get("SIGN_TRANSLATION_MODEL_REPO", "ShesterG/SHuBERT")
SIGN_TRANSLATION_HF_TOKEN = os.environ.get("HF_TOKEN", "")
SIGN_TRANSLATION_CACHE_ROOT = Path(
    os.environ.get("SIGN_TRANSLATION_CACHE_ROOT", str(RUNTIME_ROOT / ".runtime_cache"))
)
