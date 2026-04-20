"""Orchestrator and CLI for the WER pipeline.

Usage:
    py run.py --tier all                    # runs self_notes and multi_speaker
    py run.py --tier self_notes             # one scenario
    py run.py --tier all --max-clips 2      # smoke test with 2 clips per scenario
    py run.py --tier all --dry-run          # preview work, zero Azure calls
    py run.py --tier self_notes --noise-snr 15   # include noise variant
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

from config import (
    SCENARIOS, CONFIGS, DATA, DEFAULT_N_PER_SCENARIO,
    MUSAN_SNR_DB_MODERATE, scenario_dirs, combined_reports_dir,
)
from io_utils import (
    assert_free_disk, log, progress, set_log_file, env_or,
)
from dataset import load_for_scenario, Clip
from noise import inject_noise
from transcribe import transcribe, total_calls_this_run
from wer import score_clip, aggregate, ClipScore
from report import write_scenario_report, write_combined_report


# ── Stage: prepare ───────────────────────────────────────────────────

def stage_prepare(scenario: str, n: int) -> list[Clip]:
    cfg = CONFIGS[scenario]
    dirs = scenario_dirs(scenario)
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    clips = load_for_scenario(scenario, cfg.sources, n_target=n)
    log("stage.prepare.done", scenario=scenario, n=len(clips),
        sources=list({c.source for c in clips}))
    return clips


# ── Stage: inject noise (optional) ───────────────────────────────────

def stage_inject_noise(clips: list[Clip], snr_db: float) -> list[tuple[Clip, Path, str, dict]]:
    """Create noise-injected copies of each WAV clip.

    For each input WAV clip we produce:
      - A noisy WAV  (direct noise mix)
      - A noisy Opus (transcode the noisy WAV → Opus so the noise is baked in
        before compression, matching how real office Opus recordings behave)

    Incoming Opus clips are skipped — we never re-encode lossy → lossy.

    Returns list of (original_clip, noisy_audio_path, output_format, meta).
    """
    if not clips:
        return []
    out = []
    prepared_dir = scenario_dirs(clips[0].scenario)["prepared"]
    prepared_dir.mkdir(parents=True, exist_ok=True)
    wav_inputs = [c for c in clips if c.audio_path.suffix.lower() == ".wav"]
    skipped = len(clips) - len(wav_inputs)
    if skipped:
        log("noise.skip", reason="non_wav_source", skipped=skipped)
    for i, c in enumerate(wav_inputs, 1):
        # 1) Noisy WAV
        noisy_wav = prepared_dir / f"{c.id}_noise{snr_db:.0f}db.wav"
        meta = inject_noise(c.audio_path, noisy_wav, snr_db=snr_db,
                            seed=hash(c.id) & 0xFFFFFFFF)
        out.append((c, noisy_wav, "wav", meta))
        progress("noise", i, len(wav_inputs), c.id,
                 snr=meta["achieved_snr_db"], src=meta["source"])

        # 2) Transcode noisy WAV → Opus (so Opus variant also has noise)
        from transcode import to_opus, TranscodeError
        from config import OPUS_BITRATE_KBPS
        noisy_opus = prepared_dir / f"{c.id}_noise{snr_db:.0f}db.ogg"
        try:
            to_opus(noisy_wav, noisy_opus, sample_rate=16000, channels=1,
                    bitrate_kbps=OPUS_BITRATE_KBPS)
            out.append((c, noisy_opus, "opus", meta))
        except TranscodeError as e:
            log("noise.opus_skip", clip=c.id, err=str(e))
    return out


# ── Stage: transcribe ────────────────────────────────────────────────

def stage_transcribe(
    scenario: str,
    audio_paths: list[tuple[str, Path]],      # list of (clip_id, audio_path)
    *,
    dry_run: bool,
) -> dict[str, str]:
    """Returns {clip_id: hypothesis_text}. Uses on-disk cache via transcribe()."""
    cfg = CONFIGS[scenario]
    hyp_dir = scenario_dirs(scenario)["hypotheses"]
    hyp_dir.mkdir(parents=True, exist_ok=True)
    hyps: dict[str, str] = {}
    total = len(audio_paths)
    for i, (clip_id, audio_path) in enumerate(audio_paths, 1):
        hyp_json = hyp_dir / f"{clip_id}.json"
        try:
            result = transcribe(audio_path, scenario, hyp_json, dry_run=dry_run)
        except Exception as e:
            log("stage.transcribe.error", scenario=scenario, clip=clip_id,
                err=str(e).splitlines()[0])
            continue
        if dry_run:
            progress("dry_run", i, total, clip_id, seconds=result.get("estimated_seconds", 0))
            continue
        hyps[clip_id] = result.get("hypothesis", "")
        progress("transcribe", i, total, clip_id,
                 status=result["status"], ms=result.get("elapsed_ms", 0))
    return hyps


# ── Stage: score ─────────────────────────────────────────────────────

def stage_score(clips: list[Clip], hypotheses: dict[str, str]) -> list[ClipScore]:
    scores: list[ClipScore] = []
    for c in clips:
        hyp = hypotheses.get(c.id, "")
        if not hyp:
            continue
        scores.append(score_clip(
            clip_id=c.id, scenario=c.scenario, source=c.source,
            ref_text=c.ref_text, hyp_text=hyp,
            edited_reference=c.edited_reference,
            audio_format=c.audio_format, video_id=c.video_id,
        ))
    return scores


# ── Pipeline per scenario ────────────────────────────────────────────

def run_scenario(
    scenario: str,
    n: int,
    *,
    dry_run: bool,
    noise_snr: float | None,
) -> dict:
    log("scenario.start", scenario=scenario, n=n, dry_run=dry_run, noise_snr=noise_snr)
    clips = stage_prepare(scenario, n)
    if not clips:
        log("scenario.abort", scenario=scenario, reason="no_clips_loaded")
        return {"agg": aggregate([], CONFIGS[scenario].target_wer, scenario), "md_path": None}

    # Clean pass
    audio_list = [(c.id, c.audio_path) for c in clips]
    hyps = stage_transcribe(scenario, audio_list, dry_run=dry_run)
    scores = stage_score(clips, hyps)
    agg = aggregate([s.wer for s in scores], CONFIGS[scenario].target_wer, scenario)

    # Noise pass (self_notes only, and only if requested)
    noise_agg = None
    noise_meta_summary = None
    noise_scores: list[ClipScore] = []
    if noise_snr is not None and CONFIGS[scenario].target_wer_noisy:
        noise_variants = stage_inject_noise(clips, noise_snr)
        if noise_variants:
            # Each variant is uniquely identified by (source_clip.id, out_format)
            def _nid(clip_id: str, fmt: str) -> str:
                return f"{clip_id}_noise_{fmt}"
            noisy_audio_list = [(_nid(c.id, fmt), p) for (c, p, fmt, _) in noise_variants]
            noisy_by_id = {_nid(c.id, fmt): (c, fmt, meta)
                           for (c, _, fmt, meta) in noise_variants}
            noisy_hyps = stage_transcribe(scenario, noisy_audio_list, dry_run=dry_run)
            for nid, hyp in noisy_hyps.items():
                c, fmt, _meta = noisy_by_id[nid]
                noise_scores.append(score_clip(
                    clip_id=nid, scenario=scenario,
                    source=f"{c.source}+noise",
                    ref_text=c.ref_text, hyp_text=hyp,
                    edited_reference=c.edited_reference,
                    audio_format=fmt, video_id=c.video_id,
                ))
            noise_agg = aggregate(
                [s.wer for s in noise_scores],
                CONFIGS[scenario].target_wer_noisy,
                scenario,
            )
            srcs = [meta["source"] for (_, _, _, meta) in noise_variants]
            noise_meta_summary = {
                "target_snr_db": noise_snr,
                "n_variants": len(noise_variants),
                "sources": sorted(set(srcs)),
            }

    # Write reports
    all_scores = scores + noise_scores
    json_p, md_p = (None, None)
    if all_scores:
        json_p, md_p = write_scenario_report(
            scenario, all_scores, agg,
            noise_agg=noise_agg, noise_meta=noise_meta_summary,
        )
        log("scenario.report", scenario=scenario, md=str(md_p.as_posix()))

    return {"agg": agg, "noise_agg": noise_agg, "md_path": md_p, "scores": all_scores}


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="WER evaluation pipeline for Azure Fast Transcription")
    p.add_argument("--tier", choices=["all"] + SCENARIOS, default="all")
    p.add_argument("--max-clips", type=int, default=DEFAULT_N_PER_SCENARIO,
                   help=f"clips per scenario (default {DEFAULT_N_PER_SCENARIO})")
    p.add_argument("--noise-snr", type=float, default=None,
                   help=f"also run a noise-injected variant for self_notes at this SNR dB (e.g. {MUSAN_SNR_DB_MODERATE})")
    p.add_argument("--dry-run", action="store_true",
                   help="skip Azure calls; preview work and estimated audio minutes")
    p.add_argument("--log-file", type=str, default=str(DATA / "run.log"))
    args = p.parse_args(argv)

    assert_free_disk(DATA)
    set_log_file(Path(args.log_file))
    log("run.start", tier=args.tier, max_clips=args.max_clips, dry_run=args.dry_run,
        noise_snr=args.noise_snr, region=env_or("AZURE_SPEECH_REGION", "eastus"))

    scenarios = SCENARIOS if args.tier == "all" else [args.tier]
    per_scenario: dict[str, dict] = {}
    for scn in scenarios:
        per_scenario[scn] = run_scenario(
            scn, args.max_clips, dry_run=args.dry_run, noise_snr=args.noise_snr
        )

    combined_path = None
    if any(v.get("md_path") for v in per_scenario.values()):
        combined_path = write_combined_report(per_scenario)
        log("run.combined_report", path=str(combined_path.as_posix()))

    log("run.done", azure_calls=total_calls_this_run())
    if combined_path:
        print(f"\nCombined report: {combined_path.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
