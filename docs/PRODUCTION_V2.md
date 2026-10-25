# Production V2: real speech, evidence, audibility and room grade

The `social_commentary` raw-footage workflow now defaults to `retake_autocut: true`
and `studio_commentary_grade: true`. Use a fresh output directory after changing
the source, transcript or plan. Original footage is never overwritten.

## ASR and retakes before the director

Ingestion uses faster-whisper with word timestamps. A supplied transcript must
contain actual `timing_source: asr` words; estimated script clocks cannot drive
the auto-cutter. The optional local ASR model is installed with `.[asr]`.
Multi-second words receive separate decoding passes to recover collapsed retakes
when both passes independently identify a repeated prefix. Review ASR repairs;
no speech recognizer guarantees complete/correct transcription.

`src/retake_cutter.py` finds repeated prefixes within four seconds at token/phrase
similarity >=0.75, removing the first take up to the retained restart. It excludes
completed intervening sentences sharing common prefixes and short rhyming phrases
(e.g. दर्द है / मर्द है). These safeguards prevent destructive cuts of valid slogans.
Non-speech gaps over 450ms are measured for compact impulsive activity; quiet dead
air over 550ms can be shortened to 180ms. Acoustic classifications are proxies, not
medical cough recognition. Archive windows remain protected. Every accepted interval
and score is written to `retake_analysis.json`.

All cuts snap once to the source frame clock and create one TimelineMap used by
picture, recorded dialogue, music, ASR words, quotes and effects. A 20ms equal-power
audio bridge retains the picture duration. Every discontinuity toggles wide/1.18x
on the exact splice frame, using measured real iris midpoints and Lanczos4.

## Evidence and complete timed quotations

Generic `NARRATION // EXCERPT`, `SPOKEN IN THE SOURCE` and `OPINION & COMMENTARY`
labels fail schema validation. The rules planner never invents research citations
or makes cards from conversational narration. Supply reviewed publisher URLs for
news cards. `EvidenceFetcher` downloads real OpenGraph thumbnails and retains the
publisher URL, image URL and content hashes. Missing or changed assets fail closed.
Downloaded media is project-local and is not committed or included in source patches.

`evidence_card` combines the real image with an attributed headline excerpt,
masthead and highlighter. Source context does not validate unrelated narration claims.
`slogan_cards` takes complete `quote_lines` with one recorded ASR timestamp per
displayed word. Each later line appears when it is spoken, using `stacked` or
`side_by_side` layout. Quote times are source seconds in the EDL, then remapped.
No short quote is stretched across unrelated spoken sentences.

## Effects and final encoded-program gate

`build_documentary_sfx_stem` is called for the documentary branch. Required missing
effects fail the render. Bundles deliberately bypass the ordinary accent cooldown
only for their explicit card group:

| Visual | Acoustic layers and pre-master targets |
|---|---|
| Citation | UI tick −7dBFS |
| Chapter / slogan card | Sub −2.5dBFS + soft whoosh −6dBFS |
| Evidence | Paper −5dBFS + marker sweep at +300ms, −8dBFS |
| Thesis creep | Subtle sub −10dBFS |

These are isolated prepared-effect targets. Summation, bus headroom, final limiting
and AAC can change program peaks; they are reported separately. Acoustic assets
are permanently cached ElevenLabs SFX v2 files, never oscillator substitutes.

`mux_final_documentary_master` normalizes dialogue/music alone, then adds the actual
SFX stem at +3dB relative to that base. After summation it uses only peak limiting
and constant gain; effects never pass through a second loudnorm operation.

The gate decodes the **final MP4**, measures 35–150Hz for subs and high-pass 1800Hz
for paper/UI effects (300Hz high-pass for soft low-pitched whooshes) over
target−50ms to target+250ms. It also subtracts a matched AAC
counterfactual with only SFX omitted, to prevent speech alone satisfying the gate.
Both program and SFX-contribution band energy must exceed a conservative surrounding
music-bed upper bound by >=8dB. A missing/silent effect or failed margin fails the build.
The marker layer has its own +300ms target window. Measurements are RMS energy,
not peak dBFS, and do not substitute for a listening judgement about speech masking.

## Thesis and room treatment

A `thesis_creep` event remaps its source-clock boundary to output time, adds a
protected music silence, a subtle sub and a slow 1.00–1.16x eye-anchored creep.
The last 150ms of recorded music receives decaying early reflections **before**
the exact silence boundary; the music bus is sample-zero throughout the protected
line. This resolves the competing requirements of a tail and pure vocal silence.

`apply_studio_commentary_grade` selectively reduces pink/magenta wall saturation
by up to 35%, outside a tracked feathered face/neck ellipse. A gentle S curve,
25% radial edge vignette and face-only 5x5/0.6 luma unsharp preserve natural skin
and avoid sharpening room noise. It is a deterministic room preset, not semantic
segmentation or a substitute for controlled studio lighting.

## Real-source verification

Run `python scripts/run_real_checks.py`, `ruff check .`, then render the real input.
Run `python scripts/verify_documentary_delivery.py --config ABSOLUTE_CONFIG.json`.
The verifier fully decodes the real MP4, repeats the final effects gate, writes the
FFmpeg `ebur128=peak=true` log and requires −15±1LUFS / <=−1dBTP, exact splice-frame
switches, picture frame-count QC and sample-zero thesis music. No synthetic demo
video or synthetic music is generated. For local delivery tests, set
`DUVERA_REAL_V2` to the verified project folder; the optional V1 regression uses
`DUVERA_REAL_DEMO` pointing to its preserved earlier render.
