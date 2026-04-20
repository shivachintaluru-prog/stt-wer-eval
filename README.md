# stt-wer-eval

**A reproducible Word Error Rate (WER) harness for speech-to-text services —
built by a PM who codes, because no one else was going to build it for me.**

📖 **[Read the full report →](https://shivachintaluru-prog.github.io/stt-wer-eval/)**

---

## What this is

A small, honest Python pipeline to answer questions like:

- *Does our STT service actually work on real-world speech, or just on audiobook benchmarks?*
- *Does Opus compression degrade transcription accuracy vs WAV?*
- *How much does office noise at 15 dB SNR hurt WER?*
- *Which clips break the model, and why?*

It targets **Azure AI Speech Fast Transcription** out of the box, but the
scoring and reporting modules are provider-agnostic — swap `transcribe.py`
for any other STT backend and the rest works unchanged.

## What's new here vs. rolling your own jiwer script

1. **Real ground truth from YouTube creator captions.** Creators like MKBHD
   upload their own captions for accessibility; those are time-aligned to the
   audio and near-verbatim. The pipeline auto-detects creator-uploaded vs.
   auto-generated captions and labels every clip accordingly.
2. **Clean WAV vs Opus format comparison.** Bundled ffmpeg (via
   `imageio-ffmpeg` — no admin install needed) transcodes each source into
   both formats so you can run an apples-to-apples codec test.
3. **SNR-calibrated noise injection** that actually measures what it says it
   does. No peak-normalization trap. Works with real noise files (MUSAN, your
   own recordings) or a pink-noise fallback.
4. **Conservative statistical gating.** Per-clip bootstrap 95% CIs; pass/fail
   compares the CI upper bound to the target, not the point estimate. Small
   sample sizes don't get to claim wins that aren't there.
5. **Cost + secret discipline by default.** Hard cap on transcription calls
   per run, `Retry-After` honored on 429s, subscription keys redacted in every
   logging path, atomic writes so Ctrl-C never corrupts a cached hypothesis.

## Example results

Five YouTube interviews (multi-speaker, 15-60 min each), two formats each,
with and without 15 dB office babble noise injected:

| Condition | Mean WER | 95% CI | Target | Pass? |
|---|---:|---|---:|---|
| Clean (WAV + Opus) | **7.23%** | 6.05% – 8.47% | 20% | ✅ |
| + 15 dB office babble | **8.54%** | 7.03% – 10.09% | 25% | ✅ |

Format comparison, clean: WAV 7.19% vs Opus 7.27% (**+0.09pp**, statistically tied).
Format comparison, noisy: WAV 8.61% vs Opus 8.47% (**-0.14pp** — Opus actually
slightly better under noise, likely because its voice-tuned codec quietly
de-emphasizes non-speech spectral content before the server ever sees it).

See [docs/index.html](docs/index.html) (or the [hosted version](https://shivachintaluru-prog.github.io/stt-wer-eval/)) for the full walkthrough
with charts, methodology, and per-clip breakdowns.

## Quickstart

```bash
# 1. Clone + install
git clone https://github.com/shivachintaluru-prog/stt-wer-eval.git
cd stt-wer-eval
pip install -r requirements.txt

# 2. Provision an Azure Speech resource (free tier F0 includes 5 audio-hrs/mo)
#    https://ai.azure.com/  or  portal.azure.com → Create → Speech
export AZURE_SPEECH_KEY='your-32-char-key'
export AZURE_SPEECH_REGION='eastus'

# 3. Dry-run — previews clips + estimated audio minutes, no API calls
python run.py --tier multi_speaker --max-clips 2 --dry-run

# 4. Real run — WAV + Opus variants for each video
python run.py --tier multi_speaker --max-clips 2

# 5. Add noise (15 dB office babble)
python run.py --tier multi_speaker --max-clips 2 --noise-snr 15
```

Reports land in `data/<scenario>/reports/report_<timestamp>.md` with headline,
per-clip breakdown, and WAV-vs-Opus tables.

## Bring your own audio + ground truth

```
data/user_clips/
├── self_notes/
│   ├── my_memo_01.wav       # the audio
│   └── my_memo_01.txt       # matching verbatim ground truth
└── multi_speaker/
    ├── meeting_recap.m4a
    └── meeting_recap.txt
```

The pipeline picks these up automatically and they take priority over the
YouTube-based default sources.

Supported audio containers: WAV, MP3, M4A, OGG, FLAC, Opus, WebM (Azure's
server-side decoder handles all of these).

## Bring your own YouTube list

Edit `config.YOUTUBE_CAPTION_VIDEOS` with video IDs you want tested. To
pre-check whether a video has creator captions (not just auto-gen):

```bash
yt-dlp --list-subs --skip-download https://www.youtube.com/watch?v=<id>
```

Look under **"Available subtitles"** (creator-uploaded = gold-tier ground
truth) vs. **"Available automatic captions"** (Google's own ASR, useful as a
directional pseudo-reference).

## Architecture

```
run.py                  Orchestrator + CLI
├── config.py           Scenarios, targets, YouTube list, Azure endpoint
├── io_utils.py         Atomic writes, secret redaction, disk guard
├── dataset.py          Loaders: user_clips, YouTube, Azure TTS, podcast
├── transcode.py        ffmpeg wrapper (WAV + Opus)
├── noise.py            SNR-calibrated noise injection
├── transcribe.py       Azure Fast Transcription REST client (swap me!)
├── wer.py              Normalization + jiwer scoring + bootstrap CI
└── report.py           Per-scenario + combined Markdown reports
```

Each stage is **idempotent** — it writes to disk and skips on re-run. You can
kill a batch mid-flight and restart without paying for duplicate API calls.

## Design choices worth flagging

- **Pass/fail uses CI upper bound.** At N=5, a point estimate of 4.9% vs a 5%
  target means nothing. The CI might cover 3%–7%. Gating on CI-upper
  eliminates false wins from small samples.
- **Text normalization is symmetric.** Lowercase → contractions expanded →
  punctuation stripped → speaker labels/`[inaudible]` removed → digits
  converted to words → conservative disfluency stripping. Same transform
  applied to both reference and hypothesis so you're not penalizing Azure for
  "I'm" vs "I am" or "1" vs "one".
- **Diarization uses `phrases[]` sorted by offset, not `combinedPhrases`.**
  Azure's `combinedPhrases` groups text by speaker and can reorder overlapping
  segments. Sorting `phrases[]` by timestamp gives you the linear transcript
  the reference actually encodes.
- **Noise is pre-encoding, not post-encoding.** When testing Opus-under-noise,
  we mix noise into the raw WAV *before* Opus encoding so the codec sees the
  noise (just like in production). We never double-encode.

## Why this exists

I'm a product manager on a voice/STT feature. I needed to run real WER
evaluations — the kind with confidence intervals and format comparisons and
honest "here's where it breaks" analysis. I wasn't going to get that by
filing a request and waiting a month. So I built it myself, with Claude Code
as my pair programmer.

The goal of open-sourcing this: if you're a PM, researcher, or engineer who
needs to evaluate an STT service on real speech, you shouldn't have to
reinvent ground-truth sourcing, format comparison, SNR math, and
bootstrap-CI gating from scratch. Fork this, swap `transcribe.py` for your
backend, and you'll have directional signal in an afternoon.

## Contributing

Issues and PRs welcome. Small asks:

- **New STT backends** — drop in a `transcribe_<backend>.py` and a thin
  dispatcher. Happy to merge.
- **Better noise sources** — MUSAN subset downloader, real cafe/street
  recordings with CC licenses, etc.
- **Statistical upgrades** — paired bootstrap for format deltas, per-speaker
  breakdowns, etc.

## License

MIT — see [LICENSE](LICENSE).
