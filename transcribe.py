"""Azure Fast Transcription REST client with atomic writes, redacted errors,
cost cap, and Retry-After honoring."""
from __future__ import annotations
import json
import time
from pathlib import Path

import requests

from config import (
    AZURE_API_VERSION, AZURE_ENDPOINT_TEMPLATE, AZURE_MAX_FILE_MB,
    AZURE_MAX_DURATION_SEC, AZURE_RETRY_MAX, AZURE_RETRY_BACKOFF_SEC,
    MAX_TRANSCRIPTIONS_PER_RUN, CONFIGS,
)
from io_utils import (
    atomic_write_json, redact_headers, redacted_error, log, progress,
    require_env, env_or, validate_json_on_disk,
)


class TranscriptionCapExceeded(RuntimeError):
    """Raised when the MAX_TRANSCRIPTIONS_PER_RUN cap is hit."""


class _Counter:
    """Process-wide counter of Azure calls this run (cost guard)."""
    n = 0


def _endpoint() -> str:
    region = env_or("AZURE_SPEECH_REGION", "eastus")
    return AZURE_ENDPOINT_TEMPLATE.format(region=region, api_version=AZURE_API_VERSION)


def _definition_for(scenario: str) -> dict:
    cfg = CONFIGS[scenario]
    d: dict = {"locales": ["en-US"], "profanityFilterMode": "None"}
    if cfg.enable_diarization:
        d["diarization"] = {"enabled": True, "maxSpeakers": cfg.max_speakers}
    return d


def _extract_hypothesis(response_json: dict, enable_diarization: bool) -> str:
    """Get the plain-text hypothesis for WER scoring.

    When diarization is OFF: concat combinedPhrases[].text (single-channel merge).
    When diarization is ON: sort phrases[] by offsetMilliseconds, concat .text
      (avoids phantom reordering that combinedPhrases can introduce for overlaps).
    """
    if enable_diarization:
        phrases = response_json.get("phrases") or []
        phrases_sorted = sorted(phrases, key=lambda p: p.get("offsetMilliseconds", 0))
        return " ".join(p.get("text", "") for p in phrases_sorted if p.get("text"))
    combined = response_json.get("combinedPhrases") or []
    return " ".join(p.get("text", "") for p in combined if p.get("text"))


def _pre_flight(audio_path: Path) -> None:
    """Raise if the audio would be rejected by Azure (size / duration basic checks)."""
    size_mb = audio_path.stat().st_size / (1024 * 1024)
    if size_mb > AZURE_MAX_FILE_MB:
        raise ValueError(f"{audio_path.name}: {size_mb:.1f} MB exceeds Azure limit {AZURE_MAX_FILE_MB} MB")
    try:
        import soundfile as sf
        info = sf.info(str(audio_path))
        duration = info.frames / info.samplerate if info.samplerate else 0
        if duration > AZURE_MAX_DURATION_SEC:
            raise ValueError(f"{audio_path.name}: {duration:.0f}s exceeds Azure limit {AZURE_MAX_DURATION_SEC}s")
    except RuntimeError:
        # non-WAV formats that soundfile can't probe — Azure will handle, skip preflight duration
        pass


def _post_with_retry(url: str, headers: dict, files, data, timeout_s: int):
    """POST with up to AZURE_RETRY_MAX retries. Honors Retry-After on 429."""
    last_exc: BaseException | None = None
    for attempt in range(AZURE_RETRY_MAX + 1):
        try:
            resp = requests.post(url, headers=headers, files=files, data=data, timeout=timeout_s)
            # Retry on 429 (rate limit), 5xx (server), and 422 (occasional transient
            # "empty audio" that goes away on retry).
            if resp.status_code < 500 and resp.status_code not in (429, 422):
                return resp
            # Retryable status
            if attempt == AZURE_RETRY_MAX:
                return resp
            retry_after = resp.headers.get("Retry-After")
            sleep_s = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else AZURE_RETRY_BACKOFF_SEC
            log("transcribe.retry", status=resp.status_code, sleep_s=sleep_s, attempt=attempt + 1)
            time.sleep(sleep_s)
        except requests.RequestException as e:
            last_exc = e
            if attempt == AZURE_RETRY_MAX:
                raise
            time.sleep(AZURE_RETRY_BACKOFF_SEC)
    # unreachable, but for type-checkers:
    if last_exc:
        raise last_exc
    raise RuntimeError("retry loop exhausted without a response")


def transcribe(
    audio_path: Path,
    scenario: str,
    hypothesis_json_path: Path,
    *,
    dry_run: bool = False,
) -> dict:
    """Transcribe one audio file. Returns summary dict. Writes full JSON to disk.

    Idempotent: if hypothesis_json_path already exists and parses, skip API call.
    """
    # Skip if already done (validated)
    if validate_json_on_disk(hypothesis_json_path):
        with open(hypothesis_json_path, "r", encoding="utf-8") as f:
            resp_json = json.load(f)
        hypothesis = _extract_hypothesis(resp_json, CONFIGS[scenario].enable_diarization)
        return {"status": "cached", "hypothesis": hypothesis, "duration_ms": resp_json.get("durationMilliseconds", 0)}

    _pre_flight(audio_path)

    if dry_run:
        size_mb = audio_path.stat().st_size / (1024 * 1024)
        try:
            import soundfile as sf
            info = sf.info(str(audio_path))
            sec = info.frames / max(info.samplerate, 1)
        except Exception:
            sec = 0.0
        return {"status": "dry_run", "estimated_seconds": sec, "size_mb": round(size_mb, 2)}

    # Cost cap
    if _Counter.n >= MAX_TRANSCRIPTIONS_PER_RUN:
        raise TranscriptionCapExceeded(
            f"Hit MAX_TRANSCRIPTIONS_PER_RUN={MAX_TRANSCRIPTIONS_PER_RUN}. Raise it explicitly if intentional."
        )

    key = require_env("AZURE_SPEECH_KEY")
    url = _endpoint()
    headers = {"Ocp-Apim-Subscription-Key": key}
    definition = _definition_for(scenario)

    t0 = time.time()
    try:
        with open(audio_path, "rb") as f:
            files = {"audio": (audio_path.name, f, "application/octet-stream")}
            data = {"definition": json.dumps(definition)}
            # Timeout must cover upload of large files + server-side processing.
            # 1800s handles up to ~500MB / multi-hour audio comfortably.
            resp = _post_with_retry(url, headers, files, data, timeout_s=1800)
    except Exception as e:
        # Redact: never let headers dict reach logs
        log("transcribe.error",
            clip=audio_path.name,
            err=redacted_error("transcribe_post_failed", e, headers=redact_headers(headers)))
        raise

    elapsed_ms = int((time.time() - t0) * 1000)
    _Counter.n += 1

    if resp.status_code >= 400:
        # Log body but *not* our request headers
        body = resp.text[:400] if resp.text else ""
        log("transcribe.http_error",
            clip=audio_path.name, status=resp.status_code,
            body_head=body, response_headers=redact_headers(dict(resp.headers)))
        resp.raise_for_status()

    resp_json = resp.json()
    atomic_write_json(hypothesis_json_path, resp_json)

    hypothesis = _extract_hypothesis(resp_json, CONFIGS[scenario].enable_diarization)
    duration_ms = resp_json.get("durationMilliseconds", 0)

    return {
        "status": "ok",
        "hypothesis": hypothesis,
        "duration_ms": duration_ms,
        "elapsed_ms": elapsed_ms,
        "n_phrases": len(resp_json.get("phrases", [])),
    }


def total_calls_this_run() -> int:
    return _Counter.n
