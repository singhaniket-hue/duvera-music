# Social commentary / documentary preset

This preset implements the supplied commentary brief as reviewable production policy.
It is not a claim of matching the referenced creator's exact mix. The linked reference's
full audiovisual content could not be independently retrieved during this build.

```powershell
git pull
python scripts/bootstrap.py --extras edit,vocals
.venv\Scripts\python -m src.cli --preset social_commentary --video raw.mp4 --transcript transcript.json --archival-windows archive-windows.json --event-graph evidence-events.json --mode script_only --output-dir commentary_plan
```

Use .venv/bin/python on macOS/Linux. Review spotting_sheet.json, chapter_plan.json,
archival_analysis.json, dialogue_polish_qc.json, pacing_plan.json and generation_plan.json.
For assets/final_video, choose local/imported assets or explicitly authorize a configured
provider after reviewing the plan. Existing budgets, vocal rejection, SFX audibility,
unity-routing, music-continuity and encoded-output QC remain active. Use a new output
directory when inputs/provider choices change.

## What the preset enables

| Component | Behavior |
|---|---|
| Narration | 75Hz 12dB/oct HPF; −2.5dB at 280Hz/Q1.2; +1.5dB at 3.2kHz/Q1; 2:1 compressor, 15ms attack, 120ms release; 6.5kHz crossover with high-band compression |
| Loudness | Two-pass narration-only integrated target −14 LUFS / −2dBTP; measured afterwards; final combined program still uses its configured mastering target |
| Evidence accents | headline → paper_slide; data_reveal → ui_tick; chapter_transition → sub_drop; riser → restrained non-tonal rise; ordinary cuts remain silent |
| Environment | Supplied location ambience and object Foley events are excluded by documentary palette policy |
| Chapters | Distinct mood/instrument/BPM suggestions; narrative splits reuse their chapter source |
| Questions/ending | Suggested rhetorical-question holds and solo source voice for the closing sentence, requiring timing review |
| Archival inserts | Added music and SFX mute with a 150ms entry/return fade; original source insert remains |

Explicit --no-dialogue-polish and --sfx-mode none overrides retain priority over defaults.
--outro-music-return is off by preset default, leaving the closing sentence solo.
The full picture editor now defaults to the audible peak/active-RMS SFX profile and local
procedural editorial accents. Explicit profile/provider overrides retain priority.
See [DOCUMENTARY_EDITOR.md](DOCUMENTARY_EDITOR.md) for picture cuts, eye anchors, all three
graphic styles, shared-clock retiming and font requirements. --documentary-edit-mode off
retains the previous audio-only preset and its original research profile by default.

Compression and EQ cannot promise a studio recording, remove room echo, repair source
clipping, or hold every word at −14 LUFS. The reported target is integrated narration
loudness. Audition de-essing and EQ on the actual speaker; severe room noise/reverb needs
separate recording/repair work. Filters do not reorder samples or shorten the source.

The chain analyzes narration with archival spans masked out. The final narration output
crossfades into original source audio outside the protected spans. Samples **inside** a
reviewed insert are verified unchanged before final program mastering. The original
dialogue.wav is retained; dialogue_polished.wav is the mix/editor input. The final master
will still change the level of the whole program, including inserts.

## Reviewed archival windows: dependable default

```json
{
  "windows": [
    {
      "start_sec": 17.0,
      "end_sec": 28.5,
      "confidence": 1.0,
      "source": "reviewed",
      "evidence": "Human-verified external news insert on this source edit"
    }
  ]
}
```

These numbers illustrate the contract; the example end is not an annotation of the
linked reference. Replace them with actual intervals. Reviewed windows require no
speaker model. Endpoints beyond media duration are rejected. Archival protection also
overrides --sfx-silence-policy accents: editorial ticks cannot bleed into archival audio.
It does not erase the original broadcaster/guest/music contained in that source clip.

## Optional automatic reference-voice candidates

```powershell
python scripts/bootstrap.py --extras speaker,cv,vocals
.venv\Scripts\python -m src.cli --preset social_commentary --video raw.mp4 --transcript transcript.json --narrator-reference clean-own-voice.wav --mode script_only --output-dir commentary_voice_plan
```

Supplying a reference in this preset selects reference_voice analysis unless explicitly
overridden by --archival-detection off. The reference must contain at least three seconds
of clean detected narration. Choose a representative 10–15 second sample when possible.
The pipeline does **not** assume the first 15 seconds of a cold open are the narrator:
that region might itself be a news insert. Explicit --archival-detection reference_voice
requires both recorded source media and a reference.

The optional local Resemblyzer adapter compares 2s windows at 0.5s hops with the reference.
Insufficient detected speech abstains. Two cosine scores below 0.60 enter a candidate;
two scores at or above 0.75 signal a narrator return. Thresholds are configurable with
--speaker-mismatch-threshold / --speaker-return-threshold. These are uncalibrated policy
defaults. Candidate windows conservatively mute added music/SFX and protect original
source audio from narration processing. Scores/intervals are saved; embeddings are not.

A mismatch is **not proof of identity or archival origin**. Another host, dramatic delivery,
different microphones, noisy overlays, music and code-switching can confuse matching.
Purely instrumental/visual inserts and short clips may escape detection. Its 2s context
does not provide frame-accurate cut boundaries. Review candidates and supply corrected
--archival-windows for exact editorial control. No acoustic-room classifier, pyannote
diarization or automatic source separation of the original program is claimed.

Missing/broken model dependencies fail before provider initialization. The speaker extra
can install substantial PyTorch/audio dependencies. Its real model/weights installation
and accuracy were not exercised here; tests use controlled embeddings/scores. This
remains distinct from the separate Demucs music vocal-content gate.

## Chapter variation and the LLM

For long commentary, the deterministic preset proposes 90–150s chapters near transcript
segment boundaries. A 7.5–12.5 minute piece uses the supplied five-act pattern when its
duration permits: hook, escalation, analysis, psychology, synthesis. Other runtimes use
comparable spans. If no segment boundary satisfies the timing budget, a duration-grid
fallback is explicitly warned, rather than presented as semantic understanding.

Each chapter requests one distinct instrumental source, normally up to 60s of generated
music, and loops it as necessary with the existing beat/loop policy. It does **not** play
one 30s asset across every chapter. Shorter silence/mood splits reuse that chapter asset;
source/loop quality remains an audition task. Budgets count actual requested durations.
Change generation_seconds in a reviewed cue sheet for longer stems if your budget allows.

--planner llm uses the existing opt-in endpoint and sends transcript text there. Its
prompt requests chapter IDs, distinct palettes and rhetorical/closing holds, with a
150s cue ceiling. Returned schema/timing is checked. This does not certify semantic
chapter quality; no live LLM call was tested here. Supplied --spotting-sheet decisions
retain priority over automatic chapter planning.

In the audio-only preset, automatic chapter changes can propose restrained sub accents,
except in protected windows. A reviewed event graph can describe existing evidence entrances:

```json
{
  "events": [
    {
      "event_id": "article",
      "time_sec": 92.0,
      "kind": "headline",
      "confidence": 0.95,
      "intensity": 0.9,
      "semantic_salience": 0.95,
      "evidence": "Human-reviewed article entrance on this edit"
    },
    {
      "event_id": "statistic",
      "time_sec": 471.0,
      "kind": "data_reveal",
      "confidence": 0.95,
      "intensity": 0.9,
      "semantic_salience": 0.95,
      "evidence": "Human-reviewed statistic reveal on this edit"
    }
  ]
}
```

The times are contract examples inspired by the brief, not verified source annotations.
The picture editor instead renders its own source-clock visual events and derives their
SFX events. Rules use narration excerpts, not verified article screenshots or OCR/VLM.
If a short commentary plan has no
evidenced effects, the SFX empty-plan gate blocks asset rendering; supply actual events
or explicitly use --sfx-mode none. Do not fabricate graphics to satisfy a detector.

## Verification boundaries

Tests cover real vocal-chain audio, archival sample preservation, exact music muting,
documentary-only palette behavior, question/ending holds, five chapter requests on an
11:18 transcript, optional LLM contract validation, controlled reference matching, and
dependency failure before any provider call. No real speaker-identification accuracy,
reference-video score replication, live generation or artistically approved documentary
render is asserted.

Primary API references: [FFmpeg filters](https://ffmpeg.org/ffmpeg-filters.html),
[Resemblyzer voice encoder](https://github.com/resemble-ai/Resemblyzer/blob/master/resemblyzer/voice_encoder.py),
and [audio preprocessing](https://github.com/resemble-ai/Resemblyzer/blob/master/resemblyzer/audio.py).
