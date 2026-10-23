# Validation record

Build date: 23 October 2026.

Completed in this build environment:

- Python 3.12.14 on Linux; real FFmpeg/ffprobe installed.
- Python compile check passed.
- Ruff check passed.
- 14 unittest checks passed, including real H.264/AAC output and encoded loudness QC,
  exact stem duration, protected music silence, local-asset provider comparison,
  resume behavior, output lock, budget checks and pending-generation retry protection.
- Hinglish script-only example ran successfully without ASR/provider credentials.
- The offline 12-second demonstration rendered with measured AAC loudness -14.98 LUFS
  and true peak -5.96 dBTP, within the configured delivery check.
- An AAC-padding timing discrepancy found by tests was fixed by trimming extracted audio
  to the video stream clock.

The demo audio is synthetic and cannot validate musical taste, narration intelligibility,
Hindi pronunciation or emotional/narrative choices.

Not verified in this build:

- Live ElevenLabs credentials, generation, billing or plan entitlements.
- Google authentication, current UI selectors, generation and downloads.
- LLM director against a live endpoint.
- faster-whisper model download/ASR quality, optional librosa and OTIO installation.
- Premiere/Resolve timeline import or round-trip.
- Docker build or Windows/macOS bootstrap.
- GitHub Actions execution before publication.

The CI workflow is provided for Python 3.10 and 3.12. A local test pass is not a claim
that the remote workflow has run.
