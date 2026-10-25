# Documentary picture and audio editor

The social_commentary preset now includes actual picture editing. It uses one validated
source-clock plan, then maps every layer onto the same edited frame/sample clock. The
existing instrumental gate, narration chain, chapter music, protected inserts, SFX routing
checks and encoded-output QC remain in the pipeline.

## Install and preview

```sh
python scripts/bootstrap.py --extras edit,vocals
python -m src.cli --preset social_commentary --video raw.mp4 --transcript transcript.json --archival-windows archive.json --mode final_video --plan-only --output-dir documentary_plan
```

Run with the virtual-environment Python. Omit --archival-windows if there are no reviewed
inserts. Recorded input must have constant frame rate and a zero video start time. The
editor validates this before paid generation; it does not silently convert VFR footage.
The first audio stream should be clean narration, not an already scored master.

Review documentary_edit_plan.json, timeline_map.json, documentary_visual_plan.json,
documentary_render_qc.json, spotting_sheet.json, event_graph.json and generation_plan.json.
The edit plan is on the **original source clock**; visual_plan, final spotting/events and
archival_windows_edited.json use the edited clock. source_transcript.json retains original
timestamps. Never feed an edited plan back as if it used source timestamps.

Render a reviewed plan in a fresh directory, with local assets or an authorized provider:

```sh
python -m src.cli --preset social_commentary --video raw.mp4 --transcript transcript.json --archival-windows archive.json --documentary-plan documentary_plan/documentary_edit_plan.json --provider local --local-assets assets.json --mode final_video --output-dir documentary_render
```

Manifest music IDs refer to source cue IDs, which survive retiming. Imported SFX take
priority over procedural assets. Keep source music_sheet in a reviewed edit plan, or supply
--spotting-sheet with a sheet matching the untrimmed source. When --planner llm is selected,
one opt-in director request returns both picture edits and music_sheet. Returned schema,
timing and reading holds are validated; this is not a second independent music pass.

## Controls and behavior

| Control | Behavior |
|---|---|
| --documentary-edit-mode off | Existing audio-only workflow and optional older pacing renderer |
| --documentary-edit-mode plan | Writes picture proposals and shared map; use script_only or --plan-only |
| --documentary-edit-mode render | Applies cuts, zooms and graphics before preparing the final mix |
| --documentary-plan file.json | Uses reviewed source-clock edits instead of automatic picture proposals |
| --documentary-font font.ttf | Explicit font; preflight checks glyph coverage, shaping and card layout |
| --procedural-sfx | Deterministic local editorial ticks, paper accents, sub-drops and noise whooshes |
| --provider-sfx | Uses imported/provider effects through the existing spend/asset gates |

The preset chooses plan for script_only and render for asset/final modes, and enables
procedural accents. The full editor chooses the audible SFX profile unless explicitly
overridden: data ticks target −9.5dBFS sample peak, paper accents −8dBFS, sub-drops −3.5dBFS,
subject to separate true-peak/headroom checks. These are not LUFS or guaranteed perceived
loudness. --sfx-mode none disables effects. Other workflows retain their previous defaults.
Procedural effects consume no provider requests or credits. They are synthetic engineering
sounds, not licensed recordings or a substitute for generated music/semantic ambience.
There is no automatic paid retry or silent procedural fallback after a provider failure.

### Pause edits

Rules examine aligned word gaps, or conservative segment gaps when words are unavailable.
Explanatory gaps longer than 400ms can retain 180ms; topic gaps retain 320ms. Dramatic
contexts retain about 650ms and short dramatic holds survive intact. Only a quiet middle
is removable. Speech overlap, estimated timing, audible activity, archival overlap and
graphic reading intervals block removal. End breaths remain. A 20ms equal-power room-tone
bridge replaces samples around each splice without removing another 20ms from the clock.

This is conservative audio/timestamp validation, not a trained semantic breath classifier.
Misaligned ASR, quiet consonants or breaths can still need review. No speech speeding,
invented room tone, automatic word deletion, or independent audio-only timing edits occur.
Long pauses can be intentional: review the proposed plan before a production render.

### Zooms and graphics

Punch-ins use 1.14x or 1.20x with a 180ms cubic-bezier entrance/exit. Automatic anchors use
smoothed two-eye Haar boxes from a single face; missed or ambiguous detections skip the
zoom and are counted. This is not a pupil-landmark model. Supply a reviewed normalized
eye_anchor for reliable geometry; a required zoom must execute on at least 80% of its
frames. Ordinary punch-ins receive no effect by default.

Evidence cards use a dark blurred source backdrop; citations use a lower third; chapter
questions use a black picture with original dialogue continuing. Black cards add a music
silence interval. An explicitly requested card accent can sound inside that interval;
other effects remain protected and archival protection always wins.

Cards wrap and size text inside portrait/landscape safe areas. Fully readable duration is
max(1.5, visible_graphemes/14 + 0.3), plus 180ms entrance and 120ms exit. Intersections and
insufficient reading holds fail validation. Fonts must cover the text; Hindi also requires
Pillow RAQM shaping. Install Noto Sans Devanagari through your OS, or supply its font file.
Missing fonts/layout failures block before music generation rather than producing tofu.

Automatic evidence text is a short verbatim narration excerpt labeled **Narration excerpt**.
It is not fact-checked evidence. Transcript-only LLM planning cannot invent citation URLs
or face coordinates. Supply reviewed headline_text, subtext_source and optional source_url;
the URL is metadata, not an automatically fetched article screenshot. Review claim wording,
source provenance and chapter timing yourself.

### Existing music locks

In audio-only mode a locked music-only WAV remains unchanged. With picture trims enabled,
the original hash is checked and a derived locked_music_edited.wav follows the same cuts;
new card/archive silence is applied. Both source and derived hashes are reported. The
original file is never overwritten. This derived bed is not byte-identical to its source.

## Reproducible offline verification

```sh
python scripts/create_documentary_demo.py --dir documentary_demo
python -m src.cli --json-config documentary_demo/config.json
```

This controlled 16s source becomes a 15s/450-frame H.264/AAC video: one second of quiet
pause is removed, both zoom levels use explicit fixture anchors, all three card styles
render, a protected insert moves from 6–7s to 5–6s, and three effects follow the new clock.
The sub-drop crest lands at 8.010s. Reports verify exact pre-master insert samples,
music silence, real FFmpeg SFX routing and encoded picture duration/frame count.

The demo explicitly disables the music vocal gate for synthetic tones and makes no API
calls. It does not validate artistic quality, real face/ASR/speaker accuracy, live LLM or
Demucs models, external provider generation, or a real room recording. Long-form rendering
is sequential CPU/OpenCV/Pillow work and may be slow. Graphics are baked into the editor
picture track; the JSON plan/map remains reviewable, not native editable NLE title layers.
