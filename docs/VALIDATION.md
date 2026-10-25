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

## Instrumental/vocal gate and outro update — 24 October 2026

- The existing ElevenLabs adapter already set force_instrumental=true. The actual cause
  of the reported Render #2 singing cannot be attributed without its raw cue/request records.
- Added canonical prompt enforcement at planner/store/provider boundaries.
- Added a default fail-closed Demucs gate covering generated, cached and imported music.
  Controlled separator fixtures verify rejection, strip/recheck, content reuse, scale invariance,
  short bursts, error reporting, and blocking before paid generation/rendering.
- Added a pre-boundary transient preference, main-cue outro reuse, and regression coverage
  for broad LLM silence covering a CTA.
- Real FFmpeg tests still verify narration preservation and music/SFX silence.
- Demucs, PyTorch and model weights are not installed in this build environment. Acoustic
  separation/detection and the heavy optional-dependency installation were not exercised.
  The tests validate gate logic with deterministic synthetic separator outputs.
- No second-render music stem was supplied in this turn, so no claim is made of removing
  its reported singer or producing Render #3.
- All 42 local unit/integration tests and Ruff checks passed. The real-render tests use
  procedural audio; the vocal-gate tests use explicitly controlled separation fixtures.

## SFX/pacing extension — 25 October 2026

- Added event contracts, an inspectable deterministic scheduler, three optional SFX layers,
  sample-clock crest placement, bounded spectral sidechain control, true-peak checks,
  separate editor buses, and a final-video punch-in renderer.
- The earlier supplied review clip was analyzed locally. The frame-PTS detector located
  the black-card entry at 68.000 seconds and candidate bright-mask changes at 69.167,
  73.900 and 75.800 seconds. This is neither OCR nor proof of complete card detection.
  No user media or generated inspection outputs were committed.
- The 58-test suite includes real three-tier FFmpeg/AAC rendering, unchanged raw effect
  assets, protected stem silence, negative pre-roll handling, crest timing, fixed-window
  metering, cooldowns, location hold/crossfade policy, import/budget gates, stable mask
  transitions, RANSAC local/global motion, crop geometry and actual encoded image changes.
  VFR detection is tested; VFR visual rendering deliberately rejects before generation.
- The standalone six-second procedural pacing demo completed with encoded -15.12 LUFS
  and -3.24 dBTP. Its prepared tick/Foley landmarks had 0ms requested-target error.
  These are technical fixtures, not evidence of artistic quality or narration quality.
- Haar stability/geometry gates are covered, but real human eye/face accuracy is not
  calibrated. Supplied reviewed eye anchors are more dependable than the heuristic.
- No live SFX generation, Google SFX interface, VLM/OCR integration, NLE import, automatic
  breath trimming or speech J/L editing was validated or claimed. The source-separation
  caveats from the earlier update remain. No new breakfast-vlog render was produced.

## Render #4 audibility/continuity regression — 25 October 2026

- The supplied mixed 43f2955 MP4 was inspected locally. Updated mask analysis detects
  black entry at 68s and text changes at 69.167, 71, 73.9, 75.8, 79.067, 81.667 and 84.2s.
  Six requested accents use refined frame times; no OCR or source-stem inference is claimed.
- Code inspection confirms NumPy summation and normalize=0 were already present. Actual
  Render #4 config/stems/ledger were not supplied, so its cause is not conclusively known.
- Added explicit audible peak/active-RMS profile, required-event/empty-plan blocks, louder
  sub bus headroom, post-control level checks, nonzero combined-bus QC, and an actual
  FFmpeg premix-minus-reference routing test before mastering.
- Added SHA-verified music-only timeline locks with no music regeneration, preserved vocal
  checking, automatic silent-intro avoidance and unexpected music-gap QC before mux.
- All 65 tests passed locally, including an 86-second synthetic bus with one sub/six ticks,
  real intentionally divided/undivided FFmpeg mixers, locked-bed byte equality, tamper
  rejection, compact vertical text, and a controlled SFX transient surviving encoded AAC.
- No live provider/model test or actual Render #5 is claimed. The mixed attachment cannot
  recover the isolated Render #3 bed or raw narration; those are needed for that render.
