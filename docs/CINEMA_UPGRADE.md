# HTML documentary graphics and acoustic-only production audio

The `social_commentary` picture editor now renders real footage with Chromium HTML/CSS
overlays. The previous Pillow rounded-box compositor and procedural oscillator source
are removed from the production path. `procedural_sfx=true` and the old synthetic demo
entry points fail explicitly rather than silently falling back to generated tones.

## Setup

Use Python 3.10–3.12 and install `.[dev,edit,vocals]`. The edit extra contains Playwright
and MediaPipe FaceMesh. Bootstrap installs Chromium for the edit extra. On Windows,
install `soundfile` if torchaudio lacks a WAV output backend. Direct Playwright, Torch,
HuggingFace, pip and temporary caches to the user's work drive.

Licensed Plus Jakarta Sans, JetBrains Mono and Noto Sans Devanagari font files and their
OFL licenses live under `assets/fonts`. Browser shaping handles Devanagari; Pillow/RAQM
is no longer the production card renderer.

## Graphics

`src/html_overlay_renderer.py` supports evidence cards with a 120–380ms gold highlighter
sweep, frosted source lower thirds at left 6% / bottom 10%, and radial-charcoal chapter
cards with film grain and ASR-word-timed reveals. Text is escaped; cards do not invent
source verification. Metadata labels distinguish reviewed source context from narration.

WAAPI animations pause and seek to the exact frame clock. Each event has a hashed PNG
sequence. Settled, unchanged browser frames are hard-linked; transitions retain distinct
real Chromium screenshots. A completed marker is written only after the whole sequence
exists. Straight PNG RGB is explicitly premultiplied before applying the alpha-over
operator. Real source pixels supply the glass backdrop and evidence blur/vignette.

Evidence treatment smoothly blends over 180ms, uses a source-equivalent sigma 18 blur,
dark radial vignette, and slow 1.00–1.04 zoom. All cards remain on the shared AV clock.
The real video render cache verifies source/plan/code/word timestamps, frame count and
the encoded file's SHA-256 before reuse.

## Eye-anchored picture

MediaPipe FaceMesh refined iris centres (landmarks 468/473) are measured on actual frames
and temporally smoothed. The 1.14 and 1.20 zooms use the 180ms Bezier entrance. An affine
transform fixes the measured eye midpoint in place without integer-crop rounding;
`cv2.INTER_LANCZOS4` resamples the source. Missing measurements skip/count a frame, and a
required zoom must cover at least 80% of its interval. No default face-centre or dummy
avatar is substituted. Landmark accuracy remains dependent on the source image.

## Persistent acoustic SFX

`SFXAssetLibrary` first checks `assets/sfx_cache/<name>.wav`. Supported families are
`sub_drop`, `paper_slide_pop`, `ui_data_tick`, `whoosh_soft`, and `camera_shutter`, with
aliases for scheduler categories. A cache miss uses the official ElevenLabs SFX endpoint,
`eleven_text_to_sound_v2`, a 1.0 second request and prompt influence 0.45. Every result
retains its request, source hash, provenance, raw and shaped dominant crest timestamp.
No key is written to disk or logs. A persisted pending marker prevents automatic paid
retries after an uncertain request. Cache tampering blocks use.

Generation caps count unique missing acoustic families. Imported assets retain priority.
Actual event shaping and post-speech-carve metering run on copies. Documentary accents
align their remeasured acoustic crest to visual time +8ms; chapter subs use +10ms. Sample
peak dBFS and reconstructed/measured true peak dBTP remain separate. The sub ceiling is
−3.5 dBTP; smaller accents target −7 dBFS with independent true-peak headroom.

Black chapter cards create music silence with a transient exit and bounded 200ms tail
before the boundary. Only the card's explicitly requested accent is permitted inside;
source narration continues and archival protection always takes priority.

## Verification without synthetic AV demos

Prepare a real source with `scripts/create_documentary_demo.py --video REAL.mp4
--transcript transcript.json --documentary-plan reviewed.json --dir NEW_PROJECT`.
Review generation_plan.json before authorizing any provider call.

Run `python scripts/run_real_checks.py` and `ruff check .`. The supported runner excludes
the legacy oscillator/dummy-AV fixture tests; it covers browser state isolation, Hindi
word reveals, safe areas, transparency, cache integrity and pending-spend protection,
plus shared-clock and scheduling contracts. CI runs these checks and does not claim
actual-footage integration when no real project is supplied.

Set `DUVERA_REAL_DEMO` to an independently verified real render project to include its
delivery integration check. A real delivery must decode, pass final FFmpeg ebur128
integrated loudness (−15±1 LUFS) / true peak (≤−1 dBTP), and measure each scheduled accent
at its actual sample-clock target. The isolated routed SFX bus must be nonzero and peak
between −9.5 and −3 dBFS at the crest. Report that scope separately from the final program,
which also contains speech and whole-program mastering. No synthetic source video or
tone generation is part of this verification.
