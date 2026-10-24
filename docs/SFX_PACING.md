# Sound design and visual pacing

The optional scheduler connects evidence to reviewable decisions, generated/imported
assets, prepared effects, separate stems, and the existing FFmpeg master. It does not
call a cloud VLM or OCR service. All event times are seconds on the source video clock.

## Modes and evidence

`--sfx-mode none` keeps the existing music/bridge workflow. `accents_only` enables
editorial effects. `full_3_tier` additionally enables acoustic ambience and object Foley.

Automatic video analysis requires `.[cv]`. It decodes each frame at a reduced analysis
resolution and pairs it with ffprobe presentation timestamps, rather than assuming frame
index divided by nominal fps. Black-card candidates require mostly dark pixels and sparse
bright marks; title changes require three mutually agreeing pending masks. This is
**not text recognition**. Cut evidence combines histogram/luma and edge changes.
Camera effects require fast, spatially distributed RANSAC motion, high inlier coherence,
multi-frame persistence and a rolling percentile salience prior. Local hand motion should
not suffice. Motion sequence peaks are candidate landmarks, not ground-truth causes.

Locations and object causes require supplied evidence. Export this contract from a
reviewed shot list, a separately configured VLM, or your editor, then use `--event-graph`:

```json
{
  "schema_version": "1.0",
  "events": [
    {
      "event_id": "courtyard",
      "time_sec": 0,
      "kind": "location_change",
      "confidence": 0.95,
      "location": "courtyard",
      "prompt": "Distant morning breeze and soft birds, no close events",
      "evidence": "Human-reviewed establishing shot"
    },
    {
      "event_id": "auto",
      "time_sec": 5.5,
      "kind": "location_change",
      "confidence": 0.95,
      "location": "rickshaw",
      "prompt": "Soft distant engine and traffic hum",
      "evidence": "Human-reviewed location change"
    },
    {
      "event_id": "phone",
      "time_sec": 31.2,
      "kind": "foley",
      "confidence": 0.95,
      "prompt": "Restrained generic phone connection chime, no speech or music",
      "evidence": "Human-reviewed on-screen connection action"
    },
    {
      "event_id": "reflection",
      "time_sec": 43,
      "end_sec": 45,
      "kind": "punch_in",
      "confidence": 0.95,
      "eye_anchor": [0.48, 0.36],
      "zoom_factor": 1.14,
      "evidence": "EXAMPLE ONLY: replace with measured eyes and verified shot interval"
    }
  ]
}
```

Those times illustrate the supplied vlog audit; they are not independently aligned media
annotations. Do not use the example eye coordinates as detected facts. Events beyond source
duration, duplicate IDs, invalid geometry and missing location/Foley prompts are rejected.

Other event kinds: `cut`, `black_card`, `text_enter`, `whip`, `zoom`, `impact`, `ui_tick`,
and `shutter`. Set confidence/intensity/semantic_salience in [0,1] from evidence.
The scheduler logs every decision in pacing_plan.json. Ordinary cuts get no sound.
Defaults are a 0.72 accent score, 600ms accent cooldown, and 1.8s major-impact cooldown.
These weights/thresholds are conservative creative policies, not scientific constants.
Repeated location labels hold the ambience through angle changes. Incoming location
ambience starts 450ms early; a 1s equal-power overlap straddles the cut. This affects
ambience only, never reorders narration or creates lip-sync contradictions.

## Generation and imports

Run plan-only first and review generation_plan.json. Every accepted effect appears with
its requested provider duration, prompt and loop flag. A 60–80ms tick requests at least
0.5s from ElevenLabs and is then locally shaped. Ambience requests at most 30s with
`loop=true` on the v2 SFX endpoint; long regions loop the asset with boundary crossfades.
Prompt-based looping cannot guarantee musical/ambient continuity; listen to seams.

Music and SFX jobs share the existing per-provider request/duration budget and pending
ledger. Effects do not bypass generation authorization. Cached plans retain event IDs
across resume so detector variation does not create new paid jobs. Failed/uncertain calls
are never automatically retried. Use a fresh output directory after changing inputs.

Import manifests keep music cue mappings and add the accepted effect asset IDs:

```json
{
  "music": {"cue_001": "approved_music.wav"},
  "sfx": {"fx_courtyard": "courtyard.wav", "fx_auto": "rickshaw.wav", "fx_phone": "phone.wav"}
}
```

Include **every** job listed for your actual sheet, including any legacy cue bridge SFX.
Manifest paths resolve relative to the manifest. Both comparison branches are validated
before any provider generation. Google pacing SFX currently require imported assets:
the configured music-browser UI is not a verified SFX synthesis interface.

## Audio placement, mixing and QC

The raw asset is preserved. Shaping works on a copy: restrained HPF, or a 35–90Hz sub
bandpass, dominant-crest crop, and click-free fades. The crest is a centered 2ms energy
maximum refined to a sample peak. It is remeasured after fades and spectral ducking.
This is an acoustic landmark proxy, not a proven model of subjective perceptual salience.

Placement uses integer 48kHz samples:

`start_sample = round(visual_time * 48000) + round(offset_ms * 48) - crest_sample`

Offsets are editable editorial presets: shutter 0ms, text/UI 6ms, other transients 8ms,
sub 10ms. Negative starts trim the pre-roll source rather than moving the crest late.
No video-frame quantization occurs in audio. Each prepared file is checked after
quantization, with its actual crest/target samples and error recorded. Timing QC requires
the measured crest to remain within 5ms of its requested target, including after
sidechain iteration and timeline trimming. A failed timing/peak gate returns qc_failed.

The mixer uses maximum fixed-window **K-weighted 400ms momentary LUFS** for Foley/accents,
and **3s short-term LUFS** for ambience. A shorter asset zero-pads to the full window;
its loudness is therefore window-dependent. It is not an RMS-to-LUFS conversion.
Peak-limited normalization may leave an effect below target, which the reports expose.
Three 10ms-hop STFT bands from 600Hz to 5kHz apply speech-keyed attenuation with bounded
attack/release. This is a masking-control proxy, not measured human intelligibility.

| Tier | Default loudness | Pre-master true-peak ceiling | Spectral attenuation limit |
|---|---:|---:|---:|
| Ambience | −32 S-LUFS | −10 dBTP | 4 dB |
| Foley | −21 M-LUFS | −6 dBTP | 5 dB |
| Small accents | −20 to −21 M-LUFS | −7 dBTP | 6 dB |
| Black-title sub | −18 M-LUFS | −7 dBTP | 6 dB |

Peak analysis uses a 4x reconstruction estimate to set gain and FFmpeg to check the
written file/bus. Tier summation receives constant attenuation if needed, preserving
timing. Effects mix into sfx.wav alongside bridge_sfx.wav; separate ambience.wav,
foley.wav and accent.wav remain available. XML/OTIO use separate full-length buses
so editor frame quantization cannot move individual sounds. NLE UI import is unverified.

Protected windows mute all new layers by default. `--sfx-silence-policy accents`
explicitly permits only editorial accents inside them. It does **not** unmute music,
ambience or Foley. This option can realize the supplied black-card sub/tick proposal;
the default retains the previous fully silent comedic performance. Some effects may be
suppressed/truncated by protection and are reported accordingly.

All event/tier levels are **pre-master**. Existing final loudnorm changes every layer
together; final MP4 AAC loudness and true peak are independently checked as before.

## Visuals and pacing boundaries

`--visual-pacing off` is default. `plan` emits instructions; `render` applies crops during
the final video encode. Automatic contrast/emphasis punch-ins require a single face,
two plausible Haar eye boxes, sufficient samples throughout the bounded shot, and
stable normalized geometry. Haar boxes can fail or be wrong; this is not a modern
pupil/face-landmark model. A reviewed event with a stable normalized eye_anchor and
end_sec is the more dependable workflow. Low confidence, missing eyes, overlapping
instructions and intervals overlapping protected silence are rejected.

The default zoom is 1.14× over 180ms with cubic-bezier(0.22,1,0.36,1), solved by monotonic
inversion. Crop origins follow the supplied eye midpoint; source rounding/chroma grid
can introduce a few pixels of error. The stable anchor is held, not a moving-face
stabilizer. At the reviewed end, framing returns to source, so choose an actual shot
boundary. XML exports visual markers and JSON instructions rather than portable NLE
crop automation. Only the final MP4 render bakes the zoom.

Current visual rendering requires CFR input starting at zero and preserves frame rate,
frame count and source order. VFR analysis/planning is supported, rendering is rejected
before billable generation. There is no automatic shot trimming, breath deletion,
dialogue J/L editing, OCR string parsing, VLM location inference, or kinetic-text rendering.
Pause candidates remain review-only because an ASR gap is not proof of disposable time.

## Offline verification and source contracts

`scripts/create_pacing_demo.py` creates a six-second full pipeline fixture. It uses
procedural audio/test imagery and explicit events/eye anchor. Vocal policy off is
deliberate and reported as unchecked; it is not a live Demucs or artistic quality test.

Primary implementation references checked during this update:

- [ElevenLabs SFX endpoint](https://elevenlabs.io/docs/api-reference/text-to-sound-effects/convert)
- [OpenCV tracking](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html)
- [OpenCV affine estimation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- [FFmpeg audio measurement and zoompan](https://ffmpeg.org/ffmpeg-filters.html)

The attached research's own distinction governs implementation: meter windows/API bounds
are contracts; sync/zoom/loudness/cooldown numbers are configurable production priors.
