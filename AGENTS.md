# Agent setup and operating protocol

This is the Duvera Music Director Python CLI. Read README.md before changing workflows.

## Setup

Run python scripts/bootstrap.py and then the virtual-environment Python with
-m src.cli --check-env. Install extras only when needed. System package installation
requires the explicit bootstrap flag; do not assume administrator privileges.

## Understand the user's requested output

Determine mode (script_only, assets_and_script, final_video), spoken language
(english, hinglish, hindi), source media/reference script, and music provider.
Use answers already supplied by the user. Ask only for information genuinely missing.
For planning/building this repository, do not confuse the research brief with a media input.
Use JSON configs/flags in agent subprocesses; never start an input() wizard without a TTY.

## Credentials, providers and spend

Never ask for passwords or API keys in chat or put them into commits, CLI arguments or logs.
Use secure environment configuration or download assets through an authorized connector.
Python cannot silently inherit a ChatGPT plugin session.
A Google profile directory is not evidence of authentication. Use the visible login helper
on the user's own computer or use imported assets. Do not bypass provider security checks.

Review generation_plan.json before credit-consuming generation. Respect user authorization
and max duration/request limits. Never retry a pending ledger entry without checking whether
the provider already charged/generated it. Never claim live provider testing from mocks.

Music requests must retain the canonical instrumental contract at every provider boundary.
The default vocal_policy=reject requires the vocals extra and a working Demucs model/device.
Never disable it implicitly after a dependency/model failure. Stripping applies only to music,
never narration. Review vocal_checks/vocal_qc.json; a separation-energy proxy is not proof
that singing is absent. Do not regenerate paid assets automatically after rejection.
Synthetic test fixtures explicitly use off and report disabled_unchecked.

## Implementation

Contracts: src/models.py. CLI: src/cli.py. Pipeline: src/pipeline.py.
Audio: src/dsp.py and src/media.py. Providers: src/providers/.
Editorial inputs use source seconds. The documentary editor creates one explicit rational
frame/sample TimelineMap; every output layer must follow it. XML export also uses frames.
Preserve dialogue and remove music/SFX during protected silence by default.
The explicit sfx_silence_policy=accents exception permits only editorial accents, never
ambience/Foley/music. Never turn it on implicitly to fill a punchline's silence.
Never label inferred onsets as verified downbeats, estimated script timing as aligned timing,
or source band energy as measured human intelligibility.

The optional SFX/pacing extension uses EventGraph evidence and PacingPlan decisions.
Read docs/SFX_PACING.md. Review event_graph.json, pacing_plan.json and generation_plan.json.
Use sfx_mode none/accents_only/full_3_tier. Location ambience and object Foley need explicit
semantic evidence; bright masks are not OCR, and Haar eye-box midpoints are not pupil landmarks.
Do not invent scene descriptions from the transcript or delete breaths from ASR gaps.
Render impulse crests on the 48 kHz clock, not XML frames; inspect sfx_qc.json.
For the Render #4 regression, read docs/RENDER5.md. Use the explicit audible profile and
accents silence exception when requested; required effects must never silently disappear.
Keep sample-peak dBFS, active RMS dBFS, windowed LUFS and true-peak dBTP distinct.
Locked music means an existing timeline music-only PCM16 WAV plus its SHA-256, not a git
commit seed or a mixed MP4/master. Recheck the vocal gate; never auto-strip a locked bed.
Check music_continuity and sfx_routing in qc_report.json before calling a render complete.
For --preset social_commentary read docs/SOCIAL_COMMENTARY.md. Reviewed archive ranges
need no model; automatic reference-voice candidates require an explicit clean narrator
sample and the speaker extra. Do not assume a cold open is the narrator or treat cosine
similarity as verified identity. Preserve original insert samples and mute added music/SFX.
Vocal EQ/compression is not dereverberation or fixed per-word LUFS. Chapter labels and
question timing remain proposals; do not promise reference-video quality from fixtures.
Read docs/DOCUMENTARY_EDITOR.md for the full preset picture editor. It supports CFR only;
validated quiet speech gaps can be removed from picture/audio together. Never delete
timestamped speech, audible breaths or archival samples. The older --visual-pacing path
still treats pauses as review notes. Do not invent facts/citations or eye coordinates.
Check font coverage/RAQM, reading holds, shared-clock remapping and frame-count QC.
Use a fresh output directory after changing a plan.

Install .[dev,cv,edit], then run python -m unittest discover -s tests -v and ruff check . for implementation changes.
Tests must not make billable external calls. Use the offline demo for full render validation.
Report artistic/provider/NLE limitations honestly.

## Publication

Do not commit .env files, browser profiles, user transcripts/media, generated outputs or
credential artifacts. Include code, text examples, schemas, CI and documentation.
Use a fresh output directory when inputs/parameters change.
