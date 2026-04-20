"""Audio format conversion via the bundled imageio-ffmpeg binary.

Two public functions: `to_wav` and `to_opus`. Both write atomically and are
idempotent (skip if the target file already validates).
"""
from __future__ import annotations
import os
import subprocess
from functools import lru_cache
from pathlib import Path

from io_utils import log, validate_audio_on_disk


class TranscodeError(RuntimeError):
    """Raised when ffmpeg exits with a non-zero status."""


@lru_cache(maxsize=1)
def _ffmpeg_exe() -> str:
    import imageio_ffmpeg  # lazy import
    return imageio_ffmpeg.get_ffmpeg_exe()


def _run_ffmpeg(args: list[str], label: str) -> None:
    cmd = [_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        stderr_head = (e.stderr or "").splitlines()[-1] if e.stderr else ""
        log("transcode.error", label=label, stderr=stderr_head)
        raise TranscodeError(f"{label}: ffmpeg failed ({stderr_head})") from None


def to_wav(
    src_path: Path,
    dst_path: Path,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """Decode `src_path` to PCM_16 WAV at `sample_rate` Hz, `channels` channels.

    Idempotent: returns immediately if `dst_path` already exists and is readable.
    """
    if validate_audio_on_disk(dst_path):
        return dst_path
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_path.with_suffix(dst_path.suffix + ".tmp")
    _run_ffmpeg([
        "-i", str(src_path),
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-sample_fmt", "s16",
        "-f", "wav",
        str(tmp),
    ], label=f"to_wav:{src_path.name}")
    os.replace(tmp, dst_path)
    return dst_path


def to_opus(
    src_path: Path,
    dst_path: Path,
    sample_rate: int = 16000,
    channels: int = 1,
    bitrate_kbps: int = 32,
    application: str = "voip",
) -> Path:
    """Encode `src_path` to Opus (in OGG container).

    `application=voip` is tuned for speech at low bitrates — typical voice-
    capture target for mobile STT pipelines. `sample_rate` of 16 kHz is the
    narrow-band voice target; Opus internally upsamples to 48 kHz but accepts
    16 kHz input cleanly.
    """
    if validate_audio_on_disk(dst_path):
        return dst_path
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_path.with_suffix(dst_path.suffix + ".tmp")
    _run_ffmpeg([
        "-i", str(src_path),
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-c:a", "libopus",
        "-b:a", f"{bitrate_kbps}k",
        "-vbr", "on",
        "-application", application,
        "-f", "ogg",
        str(tmp),
    ], label=f"to_opus:{src_path.name}")
    os.replace(tmp, dst_path)
    return dst_path
