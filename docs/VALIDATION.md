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

## Narrative/loop update — 24 October 2026

- The supplied review clip was inspected locally. The luminance detector independently
  located an abrupt black-card start at 68.00 seconds. No user media was committed.
- A synthetic three-act regression checks the reflective LPF change, black-card protection,
  return at a spoken CTA while still black, and unchanged narration during music silence.
- The expanded 26-test suite includes beat-spacing/phase continuity, explicit fallback
  reporting, short-flash/startup/night rejection, word-level anchors and LLM output guards.
- Music/SFX generation is skipped for fully protected cues. Transient tails stay before the
  protected boundary. Tests verify exact timeline length after loop stitching.
- Source-separated narration/music stems for the reviewed clip were not available in the
  supplied attachment, so this is not a claim of a newly rendered, corrected version of it.
- Live generation, optional MIR dependencies, artistic/harmonic loop quality and NLE imports
  retain the verification limits listed above.
