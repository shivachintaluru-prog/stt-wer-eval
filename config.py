"""Single source of truth for scenario configs, targets, and dataset knobs."""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# ── Scenarios ────────────────────────────────────────────────────────
# Two tiers that mirror real voice-note / voice-assistant usage:
#   self_notes    = single-speaker dictation (you talking to your phone)
#   multi_speaker = 2-3 person conversations (interviews, meetings, calls)
SELF_NOTES = "self_notes"
MULTI_SPEAKER = "multi_speaker"
SCENARIOS = [SELF_NOTES, MULTI_SPEAKER]


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    target_wer: float                 # pass/fail gate for this scenario
    target_wer_noisy: float | None    # pass/fail gate with noise overlay
    enable_diarization: bool          # Azure: diarization on?
    max_speakers: int                 # Azure: maxSpeakers (if diarization)
    sources: list[str]                # which dataset.py loaders to try, in order


CONFIGS = {
    SELF_NOTES: ScenarioConfig(
        name=SELF_NOTES,
        target_wer=0.05,
        target_wer_noisy=0.10,
        enable_diarization=False,
        max_speakers=1,
        sources=["user_clips", "azure_tts_samples"],
    ),
    MULTI_SPEAKER: ScenarioConfig(
        name=MULTI_SPEAKER,
        target_wer=0.20,
        target_wer_noisy=0.25,
        enable_diarization=True,
        max_speakers=3,
        sources=["user_clips", "youtube_captions"],
    ),
}

# ── YouTube videos with captions (curated list) ──────────────────────
# Each entry is a YouTube video we download audio + captions for.
# Prefers creator-uploaded captions; falls back to auto-generated with a
# clear label in the report.
#
# To verify a candidate has creator (not auto) captions before adding:
#   yt-dlp --list-subs --skip-download <url>
# Look under "Available subtitles" (manual) vs "Available automatic captions".
YOUTUBE_CAPTION_VIDEOS: list[dict] = [
    # Replace with your own list. Defaults below are public creator-captioned
    # interviews + panel discussions, good for multi-speaker evaluation.
    {"video_id": "pMX2cQdPubk", "title_hint": "interview-sample-1",  "duration_s_hint": 992},
    {"video_id": "SOq05_6w0ig", "title_hint": "interview-sample-2",  "duration_s_hint": 1051},
    {"video_id": "7jaMJGtAV9M", "title_hint": "interview-sample-3",  "duration_s_hint": 1600},
    {"video_id": "us4AR-YcZd4", "title_hint": "panel-sample-1",      "duration_s_hint": 950},
    {"video_id": "C27RVio2rOs", "title_hint": "lecture-sample-1",    "duration_s_hint": 3546},
]

# ── Azure TTS sample generator (pipeline sanity check) ───────────────
# Short sentences that exercise the pipeline end-to-end without any real
# audio input. Synthesized via Azure Speech TTS, then round-tripped through
# Fast Transcription. Near-zero WER is expected — high WER indicates a
# pipeline bug (normalization, hypothesis extraction), NOT a model issue.
AZURE_TTS_VOICE = "en-US-AriaNeural"
AZURE_TTS_ENDPOINT_TEMPLATE = (
    "https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
)
AZURE_TTS_SAMPLES = [
    "Remind me to follow up with the design team about the new onboarding flow tomorrow.",
    "Key points from the product review: we need to prioritize the mobile experience and deprecate the legacy dashboard.",
    "I want to capture three ideas for the offsite. One, more cross team demos. Two, a half day of deep work. Three, no status updates.",
    "Open question for Alex: are we still on track for the February launch or should we push to March?",
    "Action items from the standup. Alice will draft the spec. Bob will check telemetry. Sam will review the API contract.",
    "Reflection on the quarter. We shipped faster than expected but customer feedback on the reliability side is still a concern.",
    "Note to self: the backend integration should land before the broad rollout, not after.",
    "Summary of my one on one with my manager. She wants more clarity on roadmap tradeoffs and faster decisions on open questions.",
    "Meeting prep for next Tuesday. Agenda is partner sync, the format decision, and the eval pipeline demo.",
    "Random thought. If we treat voice capture as a sensor instead of an agent, the integration surface becomes much simpler.",
]

# ── Clip budget per run ──────────────────────────────────────────────
DEFAULT_N_PER_SCENARIO = 10
# Hard cap on total cloud transcription calls in a single run (cost guard).
# Raised to 50 to accommodate the WAV+Opus format comparison (2 calls per video).
MAX_TRANSCRIPTIONS_PER_RUN = 50

# ── Format comparison ────────────────────────────────────────────────
# When a video comes in through the YouTube loader we transcode it to each of
# these formats and send every variant to Azure Fast Transcription separately.
# Each variant becomes its own Clip record with a format suffix in the clip_id.
AUDIO_FORMATS_TO_TEST = ["wav", "opus"]
WAV_SAMPLE_RATE = 16000            # mono, PCM_16 — typical voice-memo WAV format
OPUS_BITRATE_KBPS = 32             # voice-grade Opus; typical target for mobile STT

# ── Noise injection ──────────────────────────────────────────────────
# Default moderate-noise SNR and acceptable tolerance when we measure achieved
# SNR on the mixed signal.
MUSAN_SNR_DB_MODERATE = 15
MUSAN_SNR_TOLERANCE_DB = 0.5

# ── Azure Fast Transcription ─────────────────────────────────────────
AZURE_API_VERSION = "2025-10-15"
AZURE_ENDPOINT_TEMPLATE = (
    "https://{region}.api.cognitive.microsoft.com/speechtotext/transcriptions:transcribe"
    "?api-version={api_version}"
)
AZURE_MAX_FILE_MB = 500
AZURE_MAX_DURATION_SEC = 5 * 60 * 60
AZURE_RETRY_MAX = 1
AZURE_RETRY_BACKOFF_SEC = 2.0

# ── Disk / safety ────────────────────────────────────────────────────
MIN_FREE_DISK_GB = 2
USER_CLIP_EXTS = {".wav", ".m4a", ".mp3", ".ogg", ".flac", ".opus", ".webm"}

# ── Lenny's Podcast loader (optional additional source) ──────────────
# The repo at ChatPRD/lennys-podcast-transcripts hosts public podcast episode
# transcripts with matching YouTube URLs. The loader is included for users
# who want podcast-style multi-speaker data, but is NOT enabled in default
# scenario sources (see CONFIGS above) because transcript/audio alignment
# varies. Enable by adding "lennys" to a scenario's `sources` list.
LENNYS_REPO_API = "https://api.github.com/repos/ChatPRD/lennys-podcast-transcripts"
LENNYS_RAW_BASE = "https://raw.githubusercontent.com/ChatPRD/lennys-podcast-transcripts/main"
LENNYS_AUDIO_FORMAT = "bestaudio[ext=m4a]/bestaudio"

# ── Disfluencies (symmetric stripping for WER) ───────────────────────
DISFLUENCIES = {
    "um", "uh", "uhm", "ummm", "hmm", "mhm", "ah", "er", "eh",
    # Conservative — we do NOT strip "like", "you know", "I mean" since those
    # can be real content in conversational speech.
}


def scenario_dirs(scenario: str) -> dict[str, Path]:
    """Return the per-scenario working directories."""
    base = DATA / scenario
    return {
        "clips": base / "clips",
        "refs": base / "refs",
        "prepared": base / "prepared",
        "hypotheses": base / "hypotheses",
        "reports": base / "reports",
    }


def user_clip_dir(scenario: str) -> Path:
    return DATA / "user_clips" / scenario


def noise_dir() -> Path:
    return DATA / "noise"


def combined_reports_dir() -> Path:
    return DATA / "reports"
