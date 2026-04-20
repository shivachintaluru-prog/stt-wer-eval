"""JSON + Markdown reports per scenario, plus a combined Markdown summary."""
from __future__ import annotations
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from config import CONFIGS, scenario_dirs, combined_reports_dir
from io_utils import atomic_write_json, atomic_write_text
from wer import ClipScore, ScenarioAggregate, aggregate, bootstrap_mean_ci


def _now_slug() -> str:
    return time.strftime("%Y-%m-%d_%H%M%S", time.gmtime())


def _pct(x: float) -> str:
    if x != x:  # NaN
        return "—"
    return f"{x * 100:.2f}%"


# ── Per-scenario writer ──────────────────────────────────────────────

def write_scenario_report(
    scenario: str,
    scores: list[ClipScore],
    agg: ScenarioAggregate,
    noise_agg: ScenarioAggregate | None = None,
    noise_meta: dict | None = None,
) -> tuple[Path, Path]:
    """Write both JSON and Markdown per scenario. Returns (json_path, md_path)."""
    slug = _now_slug()
    d = scenario_dirs(scenario)["reports"]
    d.mkdir(parents=True, exist_ok=True)
    json_path = d / f"results_{slug}.json"
    md_path = d / f"report_{slug}.md"

    payload = {
        "run_id": slug,
        "scenario": scenario,
        "target_wer": agg.target,
        "aggregate": asdict(agg),
        "aggregate_noisy": asdict(noise_agg) if noise_agg else None,
        "noise_meta": noise_meta or {},
        "clips": [asdict(s) for s in scores],
    }
    atomic_write_json(json_path, payload)

    # Markdown
    md = []
    md.append(f"# WER Report — `{scenario}`")
    md.append(f"\n_Run `{slug}` · backend `azure-fast-transcription` · {len(scores)} clips_\n")

    md.append("## Headline\n")
    md.append(
        f"| Metric | Value |\n|---|---|\n"
        f"| Clips (N) | {agg.n} |\n"
        f"| Mean WER | {_pct(agg.mean)} |\n"
        f"| Median WER | {_pct(agg.median)} |\n"
        f"| IQR (25th–75th) | {_pct(agg.iqr_lo)} – {_pct(agg.iqr_hi)} |\n"
        f"| 95% CI on mean | {_pct(agg.ci_lo)} – {_pct(agg.ci_hi)} |\n"
        f"| Target | {_pct(agg.target)} |\n"
        f"| Pass / Fail (CI upper ≤ target) | **{agg.pass_fail.upper()}** |\n"
    )

    if noise_agg is not None:
        md.append(f"\n### Noise variant (synthetic/MUSAN babble at SNR)\n")
        md.append(
            f"| Metric | Value |\n|---|---|\n"
            f"| Clips | {noise_agg.n} |\n"
            f"| Mean WER | {_pct(noise_agg.mean)} |\n"
            f"| 95% CI on mean | {_pct(noise_agg.ci_lo)} – {_pct(noise_agg.ci_hi)} |\n"
            f"| Target (noisy) | {_pct(noise_agg.target)} |\n"
            f"| Pass / Fail | **{noise_agg.pass_fail.upper()}** |\n"
        )
        if noise_meta:
            md.append(f"\n*Noise meta:* {noise_meta}\n")

    # ── Format comparison (WAV vs Opus), split by clean/noisy ──────
    # Key: (condition, video_id, format) — condition distinguishes
    # clean ("+noise" not in source) from noisy.
    by_cond_video: dict[str, dict[str, dict[str, ClipScore]]] = {}
    has_any_fmt = False
    for s in scores:
        if not (s.audio_format and s.video_id):
            continue
        has_any_fmt = True
        cond = "noisy" if "+noise" in s.source else "clean"
        by_cond_video.setdefault(cond, {}).setdefault(s.video_id, {})[s.audio_format] = s

    if has_any_fmt:
        for cond in ("clean", "noisy"):
            by_video = by_cond_video.get(cond, {})
            if not by_video:
                continue
            label = "Format comparison (WAV vs Opus)"
            if cond == "noisy":
                label += " — with office babble @ 15 dB SNR"
            md.append(f"\n## {label}\n")
            md.append("| Video | Ref source | WAV WER | Opus WER | Δ (Opus − WAV) |")
            md.append("|---|---|---:|---:|---:|")
            wavs: list[float] = []
            opuses: list[float] = []
            deltas: list[float] = []
            for video_id, per_fmt in sorted(by_video.items()):
                if "wav" not in per_fmt or "opus" not in per_fmt:
                    continue
                wav_s = per_fmt["wav"]
                opus_s = per_fmt["opus"]
                wavs.append(wav_s.wer)
                opuses.append(opus_s.wer)
                delta = opus_s.wer - wav_s.wer
                deltas.append(delta)
                ref_label = "creator" if not wav_s.edited_reference else "auto"
                md.append(
                    f"| `{video_id}` | {ref_label} | {_pct(wav_s.wer)} | {_pct(opus_s.wer)} | "
                    f"{'+' if delta >= 0 else ''}{_pct(delta)} |"
                )
            if wavs and opuses:
                wav_mean = sum(wavs) / len(wavs)
                opus_mean = sum(opuses) / len(opuses)
                delta_mean = sum(deltas) / len(deltas)
                wav_ci = bootstrap_mean_ci(wavs) if len(wavs) > 1 else (float("nan"), float("nan"))
                opus_ci = bootstrap_mean_ci(opuses) if len(opuses) > 1 else (float("nan"), float("nan"))
                md.append(
                    f"| **Mean (N={len(wavs)})** | — | "
                    f"**{_pct(wav_mean)}** (CI {_pct(wav_ci[0])}–{_pct(wav_ci[1])}) | "
                    f"**{_pct(opus_mean)}** (CI {_pct(opus_ci[0])}–{_pct(opus_ci[1])}) | "
                    f"**{'+' if delta_mean >= 0 else ''}{_pct(delta_mean)}** |"
                )

        # Combined clean-vs-noisy delta per format
        clean_v = by_cond_video.get("clean", {})
        noisy_v = by_cond_video.get("noisy", {})
        if clean_v and noisy_v:
            md.append("\n## Noise impact (clean vs noisy, per format)\n")
            md.append("| Video | WAV clean | WAV noisy | WAV Δ | Opus clean | Opus noisy | Opus Δ |")
            md.append("|---|---:|---:|---:|---:|---:|---:|")
            shared = sorted(set(clean_v) & set(noisy_v))
            for vid in shared:
                c = clean_v[vid]
                n = noisy_v[vid]
                if "wav" in c and "wav" in n and "opus" in c and "opus" in n:
                    cw, nw = c["wav"].wer, n["wav"].wer
                    co, no_ = c["opus"].wer, n["opus"].wer
                    md.append(
                        f"| `{vid}` | {_pct(cw)} | {_pct(nw)} | "
                        f"{'+' if nw - cw >= 0 else ''}{_pct(nw - cw)} | "
                        f"{_pct(co)} | {_pct(no_)} | "
                        f"{'+' if no_ - co >= 0 else ''}{_pct(no_ - co)} |"
                    )

    md.append("\n## Per-clip details\n")
    md.append("| Clip | Format | Source | Ref words | WER | S | I | D | Edited ref? |")
    md.append("|---|---|---|---:|---:|---:|---:|---:|---|")
    for s in sorted(scores, key=lambda x: x.wer):
        fmt = s.audio_format or "—"
        md.append(
            f"| `{s.clip_id}` | {fmt} | {s.source} | {s.ref_words} | "
            f"{_pct(s.wer)} | {s.substitutions} | {s.insertions} | {s.deletions} | "
            f"{'yes' if s.edited_reference else 'no'} |"
        )

    md.append("\n## Worst 3 clips (for debugging)\n")
    worst = sorted(scores, key=lambda x: -x.wer)[:3]
    for s in worst:
        md.append(f"\n### `{s.clip_id}` — WER {_pct(s.wer)}  (source: {s.source})")
        md.append("\n**Reference (normalized):**\n\n```")
        md.append(s.ref_normalized[:1500] + ("…" if len(s.ref_normalized) > 1500 else ""))
        md.append("```\n")
        md.append("**Hypothesis (normalized):**\n\n```")
        md.append(s.hyp_normalized[:1500] + ("…" if len(s.hyp_normalized) > 1500 else ""))
        md.append("```")

    md.append("\n---\n")
    md.append("_**Caveats:** Small-sample results (n<30) are directional, not statistically significant. "
              "Pass/fail uses the conservative bootstrap CI upper bound vs target. "
              "Clips flagged `edited ref = yes` (Lenny's Podcast) use human-edited transcripts where "
              "fillers and stammers were removed — WER on these clips may be inflated relative to verbatim refs._\n")

    atomic_write_text(md_path, "\n".join(md))
    return json_path, md_path


# ── Combined report across both scenarios ────────────────────────────

def write_combined_report(per_scenario: dict[str, dict]) -> Path:
    """per_scenario: {scenario: {agg: ScenarioAggregate, md_path: Path, ...}}"""
    slug = _now_slug()
    d = combined_reports_dir()
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"combined_{slug}.md"

    md = []
    md.append("# STT WER Evaluation — Run Summary")
    md.append(f"\n_Run `{slug}` · backend `azure-fast-transcription`_\n")

    md.append("## Headline across scenarios\n")
    md.append("| Scenario | N | Mean WER | 95% CI on mean | Target | Pass / Fail |")
    md.append("|---|---:|---:|---|---:|---|")
    for scenario, blob in per_scenario.items():
        a: ScenarioAggregate = blob["agg"]
        md.append(
            f"| `{scenario}` | {a.n} | {_pct(a.mean)} | "
            f"{_pct(a.ci_lo)} – {_pct(a.ci_hi)} | {_pct(a.target)} | **{a.pass_fail.upper()}** |"
        )
        noise_a: ScenarioAggregate | None = blob.get("noise_agg")
        if noise_a is not None:
            md.append(
                f"| `{scenario}` (+noise) | {noise_a.n} | {_pct(noise_a.mean)} | "
                f"{_pct(noise_a.ci_lo)} – {_pct(noise_a.ci_hi)} | {_pct(noise_a.target)} | **{noise_a.pass_fail.upper()}** |"
            )

    md.append("\n## Per-scenario reports\n")
    for scenario, blob in per_scenario.items():
        md_path = blob["md_path"]
        md.append(f"- `{scenario}` → [{md_path.name}]({md_path.as_posix()})")

    md.append("\n---\n")
    md.append("_This run uses small samples. Treat numbers as directional. "
              "For production-grade targets, scale each scenario to 50+ clips and compare CI lower bound to target._\n")

    atomic_write_text(out, "\n".join(md))
    return out
