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

## Social commentary preset — 25 October 2026

- Added --preset social_commentary with an optional FFmpeg narration chain, a documentary
  accent palette, long-form chapter variation, rhetorical/closing holds and archival protection.
- Real FFmpeg tests verify the narration target and unchanged original PCM samples inside
  reviewed archival windows. An end-to-end controlled final video verifies original insert
  preservation, zero added music there, evidence accents and SFX routing.
- An 11:18 transcript fixture receives five 90–150 second chapters, distinct palettes and
  one normally 60-second generated-source request per chapter, reused across narrative splits.
- Optional LLM chapter instructions are checked using a controlled endpoint response.
  No live LLM/editorial quality validation was performed.
- Optional Resemblyzer reference matching is exercised with controlled embeddings/scores
  and dependency-failure tests. Real model installation, weights loading and speaker/archival
  accuracy were not tested. Cosine mismatch remains a candidate signal, not identity proof.
- The referenced YouTube video's full content could not be independently retrieved; the
  preset implements the supplied brief and does not claim matching that creator's exact mix.
- The suite contains 75 tests, including the previous 65 regression tests and rejection of
  an LLM collapsing all chapters into one long loop. Local tests and Ruff checks passed.

## Documentary picture editor — 25 October 2026

- Implemented one source-clock picture/audio/music plan and a rational frame/sample map.
  Actual CFR video cuts, 20ms room-tone splice bridges, both eye-anchored zoom factors,
  three graphic styles and mapped word/archive/SFX intervals are exercised.
- The full local 88-test run passed. A subsequent source/derived music-lock regression
  also passed, bringing the suite to 89 tests. Ruff, diff whitespace and environment checks
  passed. Hindi card coverage/shaping was exercised with the official Noto Sans Devanagari
  font and Pillow RAQM; actual portrait and landscape frames were inspected.
- A new standalone offline demo rendered a 16s source to 15s/450 frames. The sub-drop's
  measured crest moved to 8.010s; data/paper effects moved to 0.606s and 4.608s. Original
  archival samples moved from 6–7s to 5–6s unchanged before mastering, with zero added
  music there and throughout the black card. Real FFmpeg SFX routing and encoded frame
  count/duration checks passed. Resume did not repeat the director request.
- Tests exposed and fixed a quiet-breath deletion risk: the noise-floor estimate now uses
  the quietest 1% of 20ms blocks, avoiding a speech-heavy estimate on short recordings.
  This remains a conservative quiet-gap heuristic, not a trained breath classifier.
- One unified LLM response is tested through a controlled endpoint; automatic Haar eye
  accuracy and real ASR alignment remain uncalibrated. The synthetic demo uses explicit
  eye anchors and disables the music vocal gate explicitly. No live LLM, paid provider,
  Demucs/Resemblyzer model or real room-recording render was validated in this update.
- Source code and reproducible demo generator are committed; synthetic output media,
  downloaded fonts and user files are excluded. Graphics are baked into the output picture
  track; no editable Premiere/Resolve title round-trip is claimed.
