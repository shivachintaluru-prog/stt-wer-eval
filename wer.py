"""WER scoring with symmetric text normalization and bootstrap CI aggregation."""
from __future__ import annotations
import random
import re
from dataclasses import dataclass
from typing import Iterable

import jiwer

from config import DISFLUENCIES


# ── Normalization ────────────────────────────────────────────────────

# Matches speaker label lines like "Speaker Name (00:00:00):" common in podcast
# transcripts and some creator-uploaded captions.
_SPEAKER_LINE_RE = re.compile(r"^\s*[A-Z][\w'\-. ]*\s*\(\d{2}:\d{2}:\d{2}\):\s*", flags=re.MULTILINE)

# Matches "[inaudible 00:00:42]" and similar "[something]" annotations
_BRACKET_ANNOTATION_RE = re.compile(r"\[[^\]]*\]")

# Common contraction map (small, high-value — Whisper-style normalizer lite)
_CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "won't": "will not", "wouldn't": "would not", "shouldn't": "should not",
    "can't": "cannot", "couldn't": "could not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "i'm": "i am", "i'd": "i would", "i'll": "i will", "i've": "i have",
    "you're": "you are", "you'd": "you would", "you'll": "you will", "you've": "you have",
    "we're": "we are", "we'd": "we would", "we'll": "we will", "we've": "we have",
    "they're": "they are", "they'd": "they would", "they'll": "they will", "they've": "they have",
    "he's": "he is", "she's": "she is", "it's": "it is",
    "that's": "that is", "there's": "there is", "here's": "here is", "what's": "what is",
    "let's": "let us", "who's": "who is", "how's": "how is",
    "gonna": "going to", "wanna": "want to", "gotta": "got to",
}

_SPACE_RE = re.compile(r"\s+")
_NONWORD_RE = re.compile(r"[^\w\s']")   # keep apostrophes initially so contractions resolve

# Symmetric digit<->word normalization. We normalize BOTH ref and hyp to the
# WORD form so "1" and "one" compare as equal.  Lightweight coverage —
# single digits + 10-19 + tens + hundred/thousand/million. For richer cases,
# swap in the `num2words` package later.
_DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
    "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
    "18": "eighteen", "19": "nineteen",
    "20": "twenty", "30": "thirty", "40": "forty", "50": "fifty",
    "60": "sixty", "70": "seventy", "80": "eighty", "90": "ninety",
    "100": "one hundred", "1000": "one thousand", "1000000": "one million",
}
_STANDALONE_DIGIT_RE = re.compile(r"\b\d+\b")


def _digits_to_words(text: str) -> str:
    """Replace standalone numeric tokens with their word form where we have a mapping.
    Unknown numbers (e.g. "2026") are left as-is so both ref and hyp at least
    fail symmetrically, not asymmetrically.
    """
    def repl(m: re.Match) -> str:
        return _DIGIT_WORDS.get(m.group(0), m.group(0))
    return _STANDALONE_DIGIT_RE.sub(repl, text)


def normalize(text: str, strip_disfluencies: bool = True) -> str:
    """Apply the symmetric normalization used on both ref and hyp before WER.

    Steps (in order):
      1. Remove speaker-label lines  (e.g. "Speaker Name (00:00:00):")
      2. Remove bracketed annotations (e.g. "[inaudible 00:00:42]")
      3. Lowercase
      4. Expand common contractions
      5. Strip punctuation (keep letters/digits/apostrophes until step 4 done, then drop all non-word)
      6. Optionally strip a conservative set of disfluencies
      7. Collapse whitespace
    """
    if not text:
        return ""
    t = _SPEAKER_LINE_RE.sub(" ", text)
    t = _BRACKET_ANNOTATION_RE.sub(" ", t)
    t = t.lower()
    for src, dst in _CONTRACTIONS.items():
        t = t.replace(src, dst)
    t = _NONWORD_RE.sub(" ", t)
    t = t.replace("'", " ")
    t = _digits_to_words(t)
    if strip_disfluencies:
        words = [w for w in t.split() if w not in DISFLUENCIES]
        t = " ".join(words)
    t = _SPACE_RE.sub(" ", t).strip()
    return t


# ── Per-clip scoring ─────────────────────────────────────────────────

@dataclass
class ClipScore:
    clip_id: str
    scenario: str
    source: str
    wer: float
    substitutions: int
    insertions: int
    deletions: int
    ref_words: int
    hyp_words: int
    ref_normalized: str
    hyp_normalized: str
    edited_reference: bool        # True when ref is lossy (human-edited, auto-caps); False for verbatim
    audio_format: str = ""        # "wav" | "opus" | "" when format comparison isn't applicable
    video_id: str = ""            # stable grouping key across format variants of the same video


def score_clip(
    clip_id: str,
    scenario: str,
    source: str,
    ref_text: str,
    hyp_text: str,
    edited_reference: bool = False,
    audio_format: str = "",
    video_id: str = "",
) -> ClipScore:
    """Normalize both sides and compute WER + S/I/D breakdown."""
    ref_n = normalize(ref_text)
    hyp_n = normalize(hyp_text)
    if not ref_n:
        # Edge case: empty reference after normalization. Return inf WER to flag.
        return ClipScore(
            clip_id=clip_id, scenario=scenario, source=source,
            wer=float("inf"), substitutions=0, insertions=len(hyp_n.split()),
            deletions=0, ref_words=0, hyp_words=len(hyp_n.split()),
            ref_normalized=ref_n, hyp_normalized=hyp_n,
            edited_reference=edited_reference,
            audio_format=audio_format, video_id=video_id,
        )

    out = jiwer.process_words(ref_n, hyp_n)
    return ClipScore(
        clip_id=clip_id,
        scenario=scenario,
        source=source,
        wer=out.wer,
        substitutions=out.substitutions,
        insertions=out.insertions,
        deletions=out.deletions,
        ref_words=len(ref_n.split()),
        hyp_words=len(hyp_n.split()),
        ref_normalized=ref_n,
        hyp_normalized=hyp_n,
        edited_reference=edited_reference,
        audio_format=audio_format, video_id=video_id,
    )


# ── Aggregation: mean, median, IQR, bootstrap CI ─────────────────────

def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def bootstrap_mean_ci(
    values: list[float],
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap CI on the mean."""
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    alpha = (1 - confidence) / 2 * 100
    return (_percentile(means, alpha), _percentile(means, 100 - alpha))


@dataclass
class ScenarioAggregate:
    scenario: str
    n: int
    mean: float
    median: float
    iqr_lo: float          # 25th percentile
    iqr_hi: float          # 75th percentile
    ci_lo: float           # bootstrap 95% CI lower
    ci_hi: float           # bootstrap 95% CI upper
    target: float
    pass_fail: str         # "pass" | "fail" | "n/a" — pass iff ci_hi <= target


def aggregate(values: Iterable[float], target: float, scenario: str) -> ScenarioAggregate:
    vals = [v for v in values if v != float("inf")]
    n = len(vals)
    if n == 0:
        return ScenarioAggregate(
            scenario=scenario, n=0, mean=0, median=0, iqr_lo=0, iqr_hi=0,
            ci_lo=float("nan"), ci_hi=float("nan"), target=target, pass_fail="n/a",
        )
    mean = sum(vals) / n
    median = _percentile(vals, 50)
    iqr_lo = _percentile(vals, 25)
    iqr_hi = _percentile(vals, 75)
    ci_lo, ci_hi = bootstrap_mean_ci(vals)
    # Pass iff the *upper* CI bound is within target — conservative gating.
    if n < 2:
        pf = "n/a"
    else:
        pf = "pass" if ci_hi <= target else "fail"
    return ScenarioAggregate(
        scenario=scenario, n=n, mean=mean, median=median,
        iqr_lo=iqr_lo, iqr_hi=iqr_hi, ci_lo=ci_lo, ci_hi=ci_hi,
        target=target, pass_fail=pf,
    )
