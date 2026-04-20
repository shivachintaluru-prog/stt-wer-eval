"""SNR-calibrated noise injection.

Prefers real-world noise files in data/noise/ (MUSAN babble, user-provided).
Falls back to pink noise (1/f spectrum) when no files are present — better
approximation of real ambient noise than white Gaussian.

SNR is verified empirically post-mix and logged. No peak-normalization after
mixing (that would distort the achieved SNR across clips).
"""
from __future__ import annotations
import random
from pathlib import Path

import numpy as np
import soundfile as sf

from config import MUSAN_SNR_TOLERANCE_DB, noise_dir
from io_utils import atomic_write_bytes, log, safe_filename


# ── Loading helpers ──────────────────────────────────────────────────

def _load_mono_float32(path: Path) -> tuple[np.ndarray, int]:
    """Load audio to mono float32. Returns (samples, sample_rate)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)  # downmix to mono
    return data.astype(np.float32), sr


def _save_wav_16k(path: Path, samples: np.ndarray) -> None:
    """Write mono 16 kHz PCM_16 WAV via atomic buffer."""
    import io
    buf = io.BytesIO()
    sf.write(buf, samples, 16000, subtype="PCM_16", format="WAV")
    atomic_write_bytes(path, buf.getvalue())


def _resample_to(samples: np.ndarray, sr: int, target_sr: int = 16000) -> np.ndarray:
    """Very-light resampling via polyphase (linear interp)."""
    if sr == target_sr:
        return samples
    from math import gcd
    g = gcd(sr, target_sr)
    up, down = target_sr // g, sr // g
    # Linear interpolation resample: good enough for noise carriers
    n_out = int(round(len(samples) * target_sr / sr))
    x_old = np.linspace(0, 1, num=len(samples), endpoint=False)
    x_new = np.linspace(0, 1, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.float32)


# ── Pink-noise fallback ──────────────────────────────────────────────

def _pink_noise(n: int, seed: int) -> np.ndarray:
    """Generate pink (1/f) noise via spectral shaping of white Gaussian.

    This is NOT real babble, but its 1/f spectrum is much closer to ambient
    noise than flat white noise. Label in reports as `noise_source=pink`.
    """
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n).astype(np.float32)
    # Paul Kellett's economical filter (approx pink, good for audio work)
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = 0.0
    out = np.empty_like(white)
    for i, w in enumerate(white):
        b0 = 0.99886 * b0 + w * 0.0555179
        b1 = 0.99332 * b1 + w * 0.0750759
        b2 = 0.96900 * b2 + w * 0.1538520
        b3 = 0.86650 * b3 + w * 0.3104856
        b4 = 0.55000 * b4 + w * 0.5329522
        b5 = -0.7616 * b5 - w * 0.0168980
        out[i] = b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362
        b6 = w * 0.115926
    return out / np.max(np.abs(out) + 1e-9) * 0.5


# ── Noise pool ───────────────────────────────────────────────────────

def _available_noise_files() -> list[Path]:
    nd = noise_dir()
    if not nd.exists():
        return []
    return [p for p in nd.iterdir() if p.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg", ".m4a")]


def _pick_noise(n_samples: int, seed: int) -> tuple[np.ndarray, str]:
    """Return a noise segment of exactly n_samples and its source label."""
    files = _available_noise_files()
    if files:
        rng = random.Random(seed)
        chosen = rng.choice(files)
        n, sr = _load_mono_float32(chosen)
        n = _resample_to(n, sr, 16000)
        if len(n) < n_samples:
            # tile
            reps = (n_samples // len(n)) + 1
            n = np.tile(n, reps)
        start = rng.randrange(0, max(1, len(n) - n_samples))
        return n[start:start + n_samples].astype(np.float32), f"file:{chosen.name}"
    return _pink_noise(n_samples, seed=seed), "synthetic:pink"


# ── SNR math ─────────────────────────────────────────────────────────

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def _achieved_snr_db(speech: np.ndarray, noise: np.ndarray) -> float:
    return 20.0 * float(np.log10(_rms(speech) / (_rms(noise) + 1e-12)))


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, target_snr_db: float) -> np.ndarray:
    """Return speech + scaled(noise) achieving the target SNR (dB RMS).

    Does NOT peak-normalize the result. If clipping would occur, scales both
    components by the same factor (preserves SNR).
    """
    rms_s = _rms(speech)
    rms_n = _rms(noise)
    target_rms_n = rms_s / (10 ** (target_snr_db / 20.0))
    scale = target_rms_n / (rms_n + 1e-12)
    mixed = speech + noise * scale
    peak = float(np.max(np.abs(mixed))) if len(mixed) else 0.0
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)
    return mixed.astype(np.float32)


# ── Public API ───────────────────────────────────────────────────────

def inject_noise(src_wav: Path, dst_wav: Path, snr_db: float, seed: int = 0) -> dict:
    """Mix noise into src_wav at the target SNR and write to dst_wav.

    Achieved SNR equals the target exactly because the clipping guard scales
    speech and noise by the same factor (preserves ratio). No pre-scaling of
    speech — we use the source audio's natural level.
    """
    speech, sr = _load_mono_float32(src_wav)
    if sr != 16000:
        speech = _resample_to(speech, sr, 16000)

    noise, source = _pick_noise(len(speech), seed=seed)

    # Scale noise so RMS(noise_scaled) = RMS(speech) / 10^(SNR/20)
    rms_s = _rms(speech)
    rms_n = _rms(noise)
    target_rms_n = rms_s / (10 ** (snr_db / 20.0))
    noise_scaled = noise * (target_rms_n / (rms_n + 1e-12))

    mixed = speech + noise_scaled

    # Clipping guard: if peak > 0.99, scale BOTH components by same factor.
    peak = float(np.max(np.abs(mixed))) if len(mixed) else 0.0
    if peak > 0.99:
        guard = 0.99 / peak
        mixed = mixed * guard
        speech_in_mix = speech * guard
        noise_in_mix = noise_scaled * guard
    else:
        speech_in_mix = speech
        noise_in_mix = noise_scaled

    achieved = _achieved_snr_db(speech_in_mix, noise_in_mix)
    delta = abs(achieved - snr_db)
    if delta > MUSAN_SNR_TOLERANCE_DB:
        log("noise.snr_warn", clip=src_wav.name, target=snr_db,
            achieved=round(achieved, 2), delta=round(delta, 2))
    _save_wav_16k(dst_wav, mixed)
    return {
        "source": source,
        "target_snr_db": snr_db,
        "achieved_snr_db": round(achieved, 2),
        "n_samples": int(len(mixed)),
        "sr": 16000,
    }
