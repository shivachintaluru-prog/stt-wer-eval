"""Atomic writes, secret-safe logging, disk guards, structured progress output."""
from __future__ import annotations
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from config import MIN_FREE_DISK_GB


# ── Atomic writes ────────────────────────────────────────────────────

def atomic_write_bytes(dst: Path, data: bytes) -> None:
    """Write bytes to dst via tmp+fsync+rename. Safe against partial writes."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst)


def atomic_write_text(dst: Path, text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(dst, text.encode(encoding))


def atomic_write_json(dst: Path, obj: Any) -> None:
    atomic_write_text(dst, json.dumps(obj, indent=2, ensure_ascii=False))


def validate_json_on_disk(path: Path) -> bool:
    """Return True iff path exists and parses as JSON. Used for skip-on-exist checks."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, OSError):
        return False


def validate_audio_on_disk(path: Path) -> bool:
    """Return True iff path looks like a valid audio file.

    For formats soundfile can decode natively (wav/flac/ogg), checks frames > 0.
    For compressed formats (m4a/mp3/webm/opus) where native decoding would
    require ffmpeg or extra codecs, we only verify the file exists with a
    non-trivial size — Azure handles the decoding server-side.
    """
    if not path.exists() or path.stat().st_size < 1024:  # min 1 KB
        return False
    ext = path.suffix.lower()
    if ext in (".wav", ".flac", ".ogg"):
        try:
            import soundfile as sf
            return sf.info(str(path)).frames > 0
        except Exception:
            return False
    # Compressed formats: trust size + presence.
    return True


# ── Disk guard ───────────────────────────────────────────────────────

def assert_free_disk(path: Path, min_gb: float = MIN_FREE_DISK_GB) -> None:
    """Raise RuntimeError if free space on path's drive is below min_gb."""
    path.parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path.parent)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < min_gb:
        raise RuntimeError(
            f"Not enough disk: {free_gb:.2f} GB free on {path.parent}, need {min_gb} GB"
        )


# ── Secret-safe logging ──────────────────────────────────────────────

_SECRET_HEADER_KEYS = {"ocp-apim-subscription-key", "authorization", "x-api-key"}


def redact_headers(headers: dict[str, str] | None) -> dict[str, str]:
    """Return a copy of headers with secret values replaced by '<redacted>'."""
    if not headers:
        return {}
    return {
        k: ("<redacted>" if k.lower() in _SECRET_HEADER_KEYS else v)
        for k, v in dict(headers).items()
    }


def redacted_error(label: str, exc: BaseException, **ctx) -> str:
    """Format an exception for logging with no secret leakage.

    We deliberately do NOT format `repr(exc)` on requests exceptions because
    they can embed the request headers. We format only the class name + the
    first line of str(exc), plus caller-provided ctx (already safe).
    """
    msg = str(exc).splitlines()[0] if str(exc) else ""
    redacted_ctx = {k: v for k, v in ctx.items() if k.lower() not in _SECRET_HEADER_KEYS}
    parts = [f"{label}: {type(exc).__name__}"]
    if msg:
        parts.append(msg)
    for k, v in redacted_ctx.items():
        parts.append(f"{k}={v}")
    return " | ".join(parts)


# ── Progress / structured logging ────────────────────────────────────

_LOG_FILE: Path | None = None
_LOG_FH = None


def set_log_file(path: Path) -> None:
    """Redirect structured logs to `path` in addition to stdout."""
    global _LOG_FILE, _LOG_FH
    path.parent.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = path
    if _LOG_FH is not None:
        _LOG_FH.close()
    _LOG_FH = open(path, "a", encoding="utf-8")


def log(event: str, **fields) -> None:
    """Emit a one-line structured event to stdout (and log file if set)."""
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event, **fields}
    line = json.dumps(record, ensure_ascii=False)
    print(line, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(line + "\n")
        _LOG_FH.flush()


def progress(stage: str, i: int, n: int, item: str, **extra) -> None:
    """Human-friendly progress line for a stage.

    Example output: [transcribe 4/30] self_notes/user_01.m4a (2.3s) -> 202
    """
    suffix = " ".join(f"{k}={v}" for k, v in extra.items())
    msg = f"[{stage} {i}/{n}] {item}"
    if suffix:
        msg += f"  {suffix}"
    print(msg, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(msg + "\n")
        _LOG_FH.flush()


# ── Filename safety (Windows NTFS) ───────────────────────────────────

_WIN_ILLEGAL = '<>:"/\\|?*'


def safe_filename(name: str, max_len: int = 100) -> str:
    """Sanitize a string to be NTFS-safe and reasonable-length."""
    cleaned = "".join("_" if c in _WIN_ILLEGAL else c for c in name).strip()
    cleaned = cleaned.replace(" ", "_")
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len]
    return cleaned or "unnamed"


# ── Env helpers ──────────────────────────────────────────────────────

def require_env(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        sys.exit(f"ERROR: environment variable {name} is not set.")
    return v


def env_or(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default
