"""Per-source dataset loaders. Each returns a list of Clip records, each
materialized as audio+ref on disk under data/<scenario>/clips|refs/.

Sources (all resolve-or-skip for resilience):
  - user_clips          : data/user_clips/<scenario>/*.{wav,m4a,...} + *.txt
  - azure_tts_samples   : Azure Speech TTS round-trip (pipeline sanity check —
                          near-zero WER expected, NOT representative of real
                          human speech)
  - youtube_captions    : yt-dlp + creator-uploaded captions (real human speech
                          with time-aligned ground truth, format-comparison
                          ready via transcode.py)
  - lennys              : Optional podcast loader (disabled by default)

Future V2 (blocked on Python 3.14 / datasets / torchcodec compatibility):
  - tedlium, ami, librispeech_dummy — HuggingFace-streamed research datasets.
"""
from __future__ import annotations
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import soundfile as sf

from config import (
    USER_CLIP_EXTS, LENNYS_REPO_API, LENNYS_RAW_BASE, LENNYS_AUDIO_FORMAT,
    YOUTUBE_CAPTION_VIDEOS, AUDIO_FORMATS_TO_TEST, WAV_SAMPLE_RATE,
    OPUS_BITRATE_KBPS, scenario_dirs, user_clip_dir,
)
from io_utils import (
    atomic_write_bytes, atomic_write_text, safe_filename, log,
    validate_audio_on_disk,
)


@dataclass
class Clip:
    id: str                # safe filename-root, unique per scenario
    scenario: str
    source: str            # "user_clips" | "youtube_captions:creator" | ...
    audio_path: Path
    ref_path: Path
    ref_text: str
    edited_reference: bool = False  # True when ref is lossy (auto-caps, human-edited)
    audio_format: str = ""          # "wav" | "opus" | "" (native source, e.g. user_clips)
    video_id: str = ""              # stable grouping key when a video spawns multi-format variants


# ── Helpers ──────────────────────────────────────────────────────────

def _write_ref(clip_id: str, scenario: str, text: str) -> Path:
    d = scenario_dirs(scenario)["refs"]
    d.mkdir(parents=True, exist_ok=True)
    dst = d / f"{clip_id}.txt"
    atomic_write_text(dst, text)
    return dst


def _save_np_as_wav_16k(dst: Path, samples, sr: int) -> None:
    """Save a numpy array as mono 16 kHz PCM_16 WAV."""
    import numpy as np
    if samples is None:
        raise ValueError("empty audio array")
    arr = np.asarray(samples, dtype="float32")
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    if sr != 16000:
        # simple linear resample
        import math
        n_out = int(round(len(arr) * 16000 / sr))
        x_old = np.linspace(0, 1, num=len(arr), endpoint=False)
        x_new = np.linspace(0, 1, num=n_out, endpoint=False)
        arr = np.interp(x_new, x_old, arr).astype("float32")
    buf = io.BytesIO()
    sf.write(buf, arr, 16000, subtype="PCM_16", format="WAV")
    atomic_write_bytes(dst, buf.getvalue())


# ── user_clips loader ────────────────────────────────────────────────

def load_user_clips(scenario: str) -> list[Clip]:
    root = user_clip_dir(scenario)
    if not root.exists():
        return []
    out: list[Clip] = []
    for audio in sorted(root.iterdir()):
        if audio.suffix.lower() not in USER_CLIP_EXTS:
            continue
        ref_file = audio.with_suffix(".txt")
        if not ref_file.exists():
            log("dataset.skip", source="user_clips", reason="missing_ref_txt", audio=str(audio.name))
            continue
        ref_text = ref_file.read_text(encoding="utf-8").strip()
        if not ref_text:
            log("dataset.skip", source="user_clips", reason="empty_ref", audio=str(audio.name))
            continue
        clip_id = f"user_{safe_filename(audio.stem)}"
        # For uniformity copy audio to data/<scenario>/clips/ — preserves original for user
        dst_audio = scenario_dirs(scenario)["clips"] / f"{clip_id}{audio.suffix.lower()}"
        if not validate_audio_on_disk(dst_audio):
            dst_audio.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(dst_audio, audio.read_bytes())
        dst_ref = _write_ref(clip_id, scenario, ref_text)
        out.append(Clip(
            id=clip_id, scenario=scenario, source="user_clips",
            audio_path=dst_audio, ref_path=dst_ref, ref_text=ref_text,
        ))
    return out


# ── Azure TTS sample generator (pipeline sanity check) ──────────────

def load_azure_tts_samples(scenario: str, n: int) -> list[Clip]:
    """Synthesize N short self_notes via Azure Speech TTS for pipeline sanity.

    Round-tripping TTS speech through STT should give near-zero WER. High WER
    here indicates a pipeline bug (normalization, extraction, etc.), NOT a
    model-quality issue. These samples are NOT representative of real speech.
    """
    from config import AZURE_TTS_SAMPLES, AZURE_TTS_VOICE, AZURE_TTS_ENDPOINT_TEMPLATE
    from io_utils import require_env, env_or
    import requests

    try:
        key = require_env("AZURE_SPEECH_KEY")
    except SystemExit:
        # require_env exits on missing key; we want to skip gracefully instead
        log("dataset.skip", source="azure_tts_samples", reason="no_azure_key")
        return []

    region = env_or("AZURE_SPEECH_REGION", "eastus")
    url = AZURE_TTS_ENDPOINT_TEMPLATE.format(region=region)
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "riff-16khz-16bit-mono-pcm",
        "User-Agent": "stt-wer-eval",
    }
    out: list[Clip] = []
    clips_dir = scenario_dirs(scenario)["clips"]
    for i, text in enumerate(AZURE_TTS_SAMPLES[:n]):
        clip_id = f"tts_{i:02d}"
        audio_path = clips_dir / f"{clip_id}.wav"
        if not validate_audio_on_disk(audio_path):
            ssml = (
                f'<speak version="1.0" xml:lang="en-US">'
                f'<voice name="{AZURE_TTS_VOICE}">{text}</voice></speak>'
            )
            try:
                resp = requests.post(url, headers=headers, data=ssml.encode("utf-8"), timeout=60)
            except Exception as e:
                log("dataset.error", source="azure_tts_samples", stage="tts_post",
                    err=str(e).splitlines()[0])
                continue
            if resp.status_code != 200:
                log("dataset.error", source="azure_tts_samples",
                    stage="tts_http", status=resp.status_code, body_head=resp.text[:200])
                continue
            atomic_write_bytes(audio_path, resp.content)
        ref_path = _write_ref(clip_id, scenario, text)
        out.append(Clip(
            id=clip_id, scenario=scenario, source="azure_tts_samples",
            audio_path=audio_path, ref_path=ref_path, ref_text=text,
        ))
    return out


# ── YouTube videos with captions loader ─────────────────────────────

_VTT_TIMESTAMP_LINE_RE = re.compile(
    r"^\s*\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}.*$",
    flags=re.MULTILINE,
)
_VTT_HEADER_RE = re.compile(
    r"^(WEBVTT.*|NOTE .*|STYLE|REGION|X-TIMESTAMP-MAP.*|Kind:.*|Language:.*)$",
    flags=re.MULTILINE,
)
_VTT_TAG_RE = re.compile(r"<[^>]+>")


def _parse_vtt_text(vtt: str) -> str:
    """Extract plain text from a .vtt caption file.

    Strips headers, timestamp lines, cue settings, HTML-like tags. Dedupes
    consecutive identical lines (common in rolling auto-captions where each
    new line repeats the previous line plus one new word).
    """
    # Drop headers + timestamp lines
    text = _VTT_HEADER_RE.sub("", vtt)
    text = _VTT_TIMESTAMP_LINE_RE.sub("", text)
    # Strip cue HTML tags like <c>, <00:00:01.000>
    text = _VTT_TAG_RE.sub("", text)
    # Dedupe consecutive identical non-empty lines
    out_lines: list[str] = []
    prev = None
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        # Skip bare cue-sequence numbers
        if s.isdigit():
            continue
        if s != prev:
            out_lines.append(s)
            prev = s
    return " ".join(out_lines)


def _write_youtube_meta(stem: Path, meta: dict) -> None:
    """Persist small metadata sidecar so cached lookups know the caption source."""
    from io_utils import atomic_write_json
    atomic_write_json(stem.with_suffix(".meta.json"), meta)


def _read_youtube_meta(stem: Path) -> dict:
    meta_path = stem.with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    try:
        import json
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _download_youtube_audio_with_captions(
    video_id: str, audio_dst_stem: Path
) -> tuple[Path | None, Path | None, str]:
    """Download audio + captions for a YouTube video. Returns (audio_path,
    vtt_path, caption_source) where caption_source is 'creator' or 'auto' or
    'none'. `audio_dst_stem` is the path prefix without extension."""
    try:
        from yt_dlp import YoutubeDL
    except Exception as e:
        log("dataset.error", source="youtube_captions", stage="yt_dlp_import",
            err=str(e).splitlines()[0])
        return None, None, "none"

    audio_dst_stem.parent.mkdir(parents=True, exist_ok=True)
    outtmpl = str(audio_dst_stem) + ".%(ext)s"
    opts = {
        "format": LENNYS_AUDIO_FORMAT,
        "outtmpl": outtmpl,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "en-US", "en-GB"],
        "subtitlesformat": "vtt",
        "quiet": True,
        "noplaylist": True,
        "no_warnings": True,
        "retries": 5,
        "fragment_retries": 5,
    }
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as e:
        log("dataset.error", source="youtube_captions", stage="download",
            video_id=video_id, err=str(e).splitlines()[0])
        return None, None, "none"

    # Locate the audio file
    audio_path = next(
        (p for p in audio_dst_stem.parent.glob(f"{audio_dst_stem.name}.*")
         if p.suffix.lower() in (".m4a", ".webm", ".mp3", ".opus", ".ogg")),
        None,
    )
    if audio_path is None:
        return None, None, "none"

    # Locate the caption file. yt-dlp saves: <stem>.<lang>.vtt
    vtt_candidates = list(audio_dst_stem.parent.glob(f"{audio_dst_stem.name}*.vtt"))
    if not vtt_candidates:
        return audio_path, None, "none"

    # Determine creator vs auto. yt-dlp's info dict has 'subtitles' (creator)
    # and 'automatic_captions' (auto).
    had_manual = bool(info.get("subtitles") and any(
        lang in info["subtitles"] for lang in ("en", "en-US", "en-GB")
    ))
    caption_source = "creator" if had_manual else "auto"
    _write_youtube_meta(audio_dst_stem, {
        "video_id": video_id,
        "caption_source": caption_source,
        "duration_s": info.get("duration"),
    })
    return audio_path, vtt_candidates[0], caption_source


def _transcode_variants(source_audio: Path, clip_stem: Path) -> dict[str, Path]:
    """Produce WAV + Opus copies of the source. Returns {format: path}.

    Missing formats are skipped with a log entry so partial failures don't
    block the rest of the pipeline.
    """
    from transcode import to_wav, to_opus, TranscodeError
    out: dict[str, Path] = {}
    for fmt in AUDIO_FORMATS_TO_TEST:
        if fmt == "wav":
            dst = clip_stem.parent / f"{clip_stem.name}_wav.wav"
            try:
                to_wav(source_audio, dst, sample_rate=WAV_SAMPLE_RATE, channels=1)
                out["wav"] = dst
            except TranscodeError as e:
                log("dataset.error", source="youtube_captions", stage="to_wav",
                    video=source_audio.name, err=str(e))
        elif fmt == "opus":
            dst = clip_stem.parent / f"{clip_stem.name}_opus.ogg"
            try:
                to_opus(source_audio, dst, sample_rate=WAV_SAMPLE_RATE,
                        channels=1, bitrate_kbps=OPUS_BITRATE_KBPS)
                out["opus"] = dst
            except TranscodeError as e:
                log("dataset.error", source="youtube_captions", stage="to_opus",
                    video=source_audio.name, err=str(e))
    return out


def load_youtube_captions(scenario: str, n: int) -> list[Clip]:
    """Download + transcode each curated YouTube video into WAV and Opus clips.

    `n` is the number of *videos* to load. Each video yields one Clip per
    format in AUDIO_FORMATS_TO_TEST (default: 2 — wav + opus).
    """
    clips_dir = scenario_dirs(scenario)["clips"]
    clips_dir.mkdir(parents=True, exist_ok=True)
    out: list[Clip] = []
    for entry in YOUTUBE_CAPTION_VIDEOS[:n]:
        video_id = entry["video_id"]
        title_hint = safe_filename(entry.get("title_hint") or video_id, max_len=40)
        video_stem = f"yt_{title_hint}"
        audio_stem = clips_dir / video_stem

        # Locate (or fetch) the source audio + captions.
        source_audio = next(
            (p for p in clips_dir.glob(f"{video_stem}.*")
             if p.suffix.lower() in (".m4a", ".webm", ".mp3", ".opus", ".ogg")
             and validate_audio_on_disk(p)),
            None,
        )
        vtt_path = next(iter(clips_dir.glob(f"{video_stem}*.vtt")), None)
        # Pull caption source from sidecar metadata if we previously downloaded.
        meta = _read_youtube_meta(audio_stem)
        caption_source = meta.get("caption_source", "")
        if source_audio is None or vtt_path is None or not caption_source:
            source_audio, vtt_path, caption_source = _download_youtube_audio_with_captions(
                video_id, audio_stem
            )
            if source_audio is None or vtt_path is None:
                log("dataset.skip", source="youtube_captions", video_id=video_id,
                    reason="missing_audio_or_captions")
                continue

        ref_text = _parse_vtt_text(vtt_path.read_text(encoding="utf-8", errors="ignore"))
        if not ref_text or len(ref_text.split()) < 10:
            log("dataset.skip", source="youtube_captions", video_id=video_id,
                reason="empty_vtt")
            continue

        # Transcode to each target format and emit one Clip per variant.
        variants = _transcode_variants(source_audio, audio_stem)
        if not variants:
            log("dataset.skip", source="youtube_captions", video_id=video_id,
                reason="transcode_failed")
            continue

        # Write the single shared reference file once per video.
        _write_ref(video_stem, scenario, ref_text)

        for fmt, audio_path in variants.items():
            clip_id = f"{video_stem}_{fmt}"
            ref_path = _write_ref(clip_id, scenario, ref_text)
            out.append(Clip(
                id=clip_id,
                scenario=scenario,
                source=f"youtube_captions:{caption_source}",
                audio_path=audio_path,
                ref_path=ref_path,
                ref_text=ref_text,
                edited_reference=(caption_source != "creator"),
                audio_format=fmt,
                video_id=video_stem,
            ))
    return out


# ── Podcast loader (optional, disabled by default) ─────────────────
# Fetches transcript + YouTube audio from ChatPRD/lennys-podcast-transcripts,
# a public GitHub repo of podcast transcripts with per-episode YAML frontmatter
# that includes the YouTube URL. Kept as an optional source for experimentation
# — enable by adding "lennys" to a scenario's `sources` list in config.py.
#
# Caveat: transcripts cover full episodes but youtube_url often points to a
# short clip. We truncate the transcript to the clip duration via timestamps.

_LENNYS_YAML_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", flags=re.DOTALL)

# Matches Lenny's utterance headers. Timestamps appear as either HH:MM:SS or MM:SS.
# Speaker name (optional) precedes the parenthesized timestamp.
# Examples: "Lenny (00:00:00):"  /  "Casey Winters (00:00):"  /  "(00:01:21):"
_LENNYS_TIMESTAMP_RE = re.compile(
    r"^(?:[A-Z][\w'\-. ]*)?\s*\(((?:\d{1,2}:){1,2}\d{2})\):\s*$",
    flags=re.MULTILINE,
)


def _parse_timestamp(ts: str) -> int:
    """Convert a 'HH:MM:SS' or 'MM:SS' string to seconds."""
    parts = [int(p) for p in ts.split(":")]
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    else:
        return 0
    return h * 3600 + m * 60 + s


def _truncate_lennys_transcript(body: str, clip_duration_s: float) -> str:
    """The transcripts in the repo cover the FULL episode, but `youtube_url`
    often points to a short clip (e.g., 3:50). We keep only utterances whose
    timestamp is within the clip duration.
    """
    if clip_duration_s <= 0:
        return body
    positions: list[tuple[int, int]] = []
    for m in _LENNYS_TIMESTAMP_RE.finditer(body):
        t_sec = _parse_timestamp(m.group(1))
        positions.append((m.start(), t_sec))
    if not positions:
        return body
    cut_at = None
    for (pos, t_sec) in positions:
        if t_sec > clip_duration_s:
            cut_at = pos
            break
    truncated = body[:cut_at] if cut_at is not None else body
    cleaned = _LENNYS_TIMESTAMP_RE.sub(" ", truncated)
    return " ".join(cleaned.split())


def _parse_lennys_yaml(md: str) -> dict:
    """Lightweight YAML frontmatter parser (no pyyaml dep needed for flat fields)."""
    import yaml
    m = _LENNYS_YAML_RE.match(md)
    if not m:
        return {}
    try:
        return yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return {}


def _fetch_lennys_episode_list() -> list[dict]:
    """List all episode folders under /episodes via GitHub Contents API."""
    import requests
    r = requests.get(f"{LENNYS_REPO_API}/contents/episodes?per_page=300", timeout=30)
    if r.status_code != 200:
        log("dataset.error", source="lennys", stage="list", status=r.status_code)
        return []
    return [item for item in r.json() if item.get("type") == "dir"]


def _fetch_lennys_transcript(slug: str) -> tuple[dict, str]:
    """Return (frontmatter, body_text) for an episode slug."""
    import requests
    url = f"{LENNYS_RAW_BASE}/episodes/{slug}/transcript.md"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return {}, ""
    md = r.text
    fm = _parse_lennys_yaml(md)
    body = _LENNYS_YAML_RE.sub("", md, count=1).strip()
    return fm, body


def _download_youtube_audio(video_url: str, dst: Path) -> bool:
    """Download audio via yt-dlp forcing an Azure-compatible format.

    Returns True on success. On failure, logs and returns False (caller skips).
    """
    try:
        from yt_dlp import YoutubeDL
    except Exception as e:
        log("dataset.error", source="lennys", stage="yt_dlp_import", err=str(e).splitlines()[0])
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    outtmpl = str(dst.with_suffix("")) + ".%(ext)s"
    opts = {
        "format": LENNYS_AUDIO_FORMAT,   # bestaudio[ext=m4a]/bestaudio
        "outtmpl": outtmpl,
        "quiet": True,
        "noplaylist": True,
        "no_warnings": True,
        "retries": 5,
        "fragment_retries": 5,
        # Note: nopart=True triggers HTTP 416 on YouTube segmented downloads.
        # We accept occasional Windows rename failures from AV scanning and
        # retry at the caller level instead.
    }
    try:
        with YoutubeDL(opts) as ydl:
            ydl.extract_info(video_url, download=True)
    except Exception as e:
        log("dataset.error", source="lennys", stage="download", url=video_url,
            err=str(e).splitlines()[0])
        return False
    # yt-dlp picks the extension; find the actual saved file.
    for cand in dst.parent.glob(f"{dst.stem}.*"):
        if cand.suffix.lower() in (".m4a", ".webm", ".mp3", ".opus", ".ogg"):
            return True
    return False


def load_lennys(scenario: str, n: int, seed: int = 42) -> list[Clip]:
    """Pick N Lenny's episodes. Prefers already-cached audio files; downloads
    more only if the cache doesn't have enough.
    """
    import random
    clips_dir = scenario_dirs(scenario)["clips"]
    clips_dir.mkdir(parents=True, exist_ok=True)

    out: list[Clip] = []

    # Step 1: reuse already-downloaded Lenny's audio files.
    cached = sorted(p for p in clips_dir.glob("lenny_*.*")
                    if p.suffix.lower() in (".m4a", ".webm", ".mp3", ".opus", ".ogg")
                    and validate_audio_on_disk(p))
    for p in cached:
        if len(out) >= n:
            return out
        slug = p.stem.replace("lenny_", "", 1)
        fm, body = _fetch_lennys_transcript(slug)
        if not body:
            log("dataset.skip", source="lennys", reason="transcript_fetch_failed",
                slug=slug)
            continue
        dur_s = float(fm.get("duration_seconds") or 0)
        ref_text = _truncate_lennys_transcript(body, dur_s)
        if not ref_text:
            log("dataset.skip", source="lennys", reason="empty_after_truncation",
                slug=slug, duration_s=dur_s)
            continue
        clip_id = p.stem
        ref_path = _write_ref(clip_id, scenario, ref_text)
        out.append(Clip(
            id=clip_id, scenario=scenario, source="lennys",
            audio_path=p, ref_path=ref_path, ref_text=ref_text,
            edited_reference=True,
        ))

    if len(out) >= n:
        return out

    # Step 2: download additional episodes to fill the quota.
    episodes = _fetch_lennys_episode_list()
    if not episodes:
        return out
    rng = random.Random(seed)
    rng.shuffle(episodes)
    have_slugs = {c.id.replace("lenny_", "", 1) for c in out}

    for ep in episodes:
        if len(out) >= n:
            break
        slug = ep["name"]
        safe_slug = safe_filename(slug, max_len=40)
        if safe_slug in have_slugs:
            continue
        fm, body = _fetch_lennys_transcript(slug)
        if not fm or not body:
            continue
        video_url = fm.get("youtube_url")
        duration_s = fm.get("duration_seconds") or 0
        if not video_url or duration_s <= 0 or duration_s > 900:
            continue
        clip_id = f"lenny_{safe_slug}"
        audio_path = clips_dir / f"{clip_id}.m4a"
        if not any(validate_audio_on_disk(p) for p in clips_dir.glob(f"{clip_id}.*")):
            if not _download_youtube_audio(video_url, audio_path):
                continue
        actual = next((p for p in clips_dir.glob(f"{clip_id}.*")
                       if p.suffix.lower() in (".m4a", ".webm", ".mp3", ".opus", ".ogg")
                       and validate_audio_on_disk(p)), None)
        if actual is None:
            continue
        ref_text = _truncate_lennys_transcript(body, duration_s)
        if not ref_text:
            log("dataset.skip", source="lennys", reason="empty_after_truncation",
                slug=slug, duration_s=duration_s)
            continue
        ref_path = _write_ref(clip_id, scenario, ref_text)
        out.append(Clip(
            id=clip_id, scenario=scenario, source="lennys",
            audio_path=actual, ref_path=ref_path, ref_text=ref_text,
            edited_reference=True,
        ))
    return out


# ── Dispatcher ───────────────────────────────────────────────────────

LOADERS = {
    "user_clips": load_user_clips,
    "azure_tts_samples": load_azure_tts_samples,
    "youtube_captions": load_youtube_captions,
    "lennys": load_lennys,
}


def load_for_scenario(scenario: str, sources: list[str], n_target: int) -> list[Clip]:
    """Collect clips from each source. `n_target` is the loader input (often
    interpreted as number of source items like videos, not Azure calls).

    For sources that emit multiple clips per source item (e.g., youtube_captions
    emits one clip per format), the returned list can exceed n_target — that is
    intentional and is bounded downstream by MAX_TRANSCRIPTIONS_PER_RUN.
    """
    collected: list[Clip] = []
    videos_or_items_done = 0
    for src in sources:
        if videos_or_items_done >= n_target:
            break
        loader = LOADERS.get(src)
        if loader is None:
            log("dataset.error", source=src, err="unknown_loader")
            continue
        remaining = n_target - videos_or_items_done
        try:
            if src == "user_clips":
                clips = loader(scenario)
            else:
                clips = loader(scenario, remaining)
        except Exception as e:
            log("dataset.error", source=src, err=str(e).splitlines()[0])
            continue
        if not clips:
            log("dataset.empty", source=src, scenario=scenario)
            continue
        # Count unique source items (by video_id when present, else by clip_id).
        items_added = len({(c.video_id or c.id) for c in clips})
        videos_or_items_done += items_added
        collected.extend(clips)
        log("dataset.loaded", source=src, scenario=scenario,
            clips=len(clips), items=items_added)
    return collected
