# Duvera Music Director

Dialogue-aware music direction and audio post-production for long-form YouTube videos.
Built for English, Hindi and Hinglish, including Duvera-style reflective narration.

Turn a script or recorded video into:

1. **Script only** — a validated spotting sheet, an annotated production script, and cue CSV.
2. **Editor pack** — those notes plus prepared cue WAVs, dialogue/music/SFX stems, timeline XML,
   visual-accent markers, and measurements.
3. **Final video** — a mastered WAV and H.264/AAC MP4, measured again after AAC encoding.

Choose local/downloaded audio, official ElevenLabs music/SFX generation, or a configurable
Google Labs browser adapter. Compare ElevenLabs and Google with one shared spotting sheet.

This is a Python CLI project. No hosted service or account dashboard is required.

## Start here

Requires Python 3.10+ and FFmpeg/ffprobe on PATH for media work. A text-only spotting run
does not require FFmpeg, a GPU, a provider account, an LLM, or an ASR model.

From the repository root:

    python scripts/bootstrap.py

On macOS/Linux:

    .venv/bin/python -m src.cli --check-env
    .venv/bin/python -m src.cli --json-config examples/script_only.json

On Windows:

    .venv\Scripts\python.exe -m src.cli --check-env
    .venv\Scripts\python.exe -m src.cli --json-config examples/script_only.json

Or install into an existing environment:

    python -m pip install -e .
    duvera-music --help

JSON configuration paths resolve relative to the configuration file. CLI paths resolve
relative to the current directory. No hidden .env loading occurs; export environment variables
in your shell or supply them through your agent's secret manager.

The setup script can install optional features:

    python scripts/bootstrap.py --extras asr,mir,browser,timeline,dev

ASR downloads model weights on first use. CPU/int8 is the default. GPU configuration is
optional and depends on faster-whisper/CTranslate2 support for your computer.
The default small model and automatic code-switching may need correction for Hinglish.

System packages are only installed with the explicit --install-ffmpeg option; administrator
access may be needed. RHEL/offline installations should provide FFmpeg and Python wheels
through their normal package process.

## Run a real offline end-to-end demo

    python scripts/create_demo.py --dir demo_media
    python -m src.cli --json-config demo_media/config.json

This creates a 12-second video with synthetic speech-like tones and a synthetic music bed.
It makes **no paid calls** and is a technical fixture, not an example of musical or voice quality.
The final MP4 is in demo_media/output/local/.

## Use your own script

    python -m src.cli --script examples/duvera_hinglish.txt --lang hinglish --mode script_only --output-dir agent_output/my_script

Script-only timestamps are **estimates from words per minute**, never claimed as aligned
to a recording. Syllable counts, narrative pivots and the BPM mapping are editorial heuristics.
The rules planner is deterministic and intentionally conservative.

Use --interactive for a terminal wizard. When an agent runs without a terminal, the program
never blocks on input: use flags or --json-config. Missing data produces JSON errors.

## Recorded video/audio and accurate timing

Supply --video with either a video or audio file. For a final_video output it must have a
video stream. The first audio stream is assumed to be isolated dialogue. Existing background
music is **not separated** from a baked source mix.

Either supply a transcript with actual seconds:

    python -m src.cli --video raw.mp4 --transcript examples/transcript.json --mode script_only --output-dir agent_output/plan

Or install the ASR extra and let local faster-whisper transcribe:

    python -m pip install -e ".[asr]"
    python -m src.cli --video raw.mp4 --script reference.txt --lang hinglish --mode script_only --output-dir agent_output/asr_plan

A reference script is an ASR hint; it is not evenly stretched across the video.
The transcript JSON contract is shown in examples/transcript.json. ASR additionally writes
word_timestamps.json. Review timestamps and spelling before generating a long score.

## Editor pack using your existing music

1. Produce a plan.
2. Review spotting_sheet.json and generation_plan.json.
3. Create an asset manifest mapping every cue ID to downloaded or locally licensed audio.
4. Run the pack or final render in a **new output directory**.

    python -m src.cli --video raw.mp4 --transcript transcript.json --mode assets_and_script --provider local --local-assets assets.json --output-dir agent_output/editor_pack

    python -m src.cli --video raw.mp4 --transcript transcript.json --mode final_video --provider local --local-assets assets.json --output-dir agent_output/render

Manifest paths resolve relative to the manifest file. Optional downbeats are user-verified
source offsets in seconds. Supply SFX for cues that have bridge_sfx_prompt, or remove that
prompt in an edited spotting sheet. --spotting-sheet accepts a reviewed sheet.
Automatic onset entry detection is **not true bar/downbeat inference**. Optional librosa
adds beat estimates, still not bar-phase verification. Short assets loop with crossfades.

## ElevenLabs

Music uses POST /v1/music with music_v2 and force_instrumental=true.
SFX uses the separate POST /v1/sound-generation endpoint.
Do not use an SFX generator as a substitute for a full music generator.

Export ELEVENLABS_API_KEY securely, then review the generation plan:

    python -m src.cli --video raw.mp4 --transcript transcript.json --mode final_video --provider elevenlabs --plan-only --output-dir agent_output/eleven_run

After reviewing prompts and account usage, run the same command without --plan-only and
with explicit generation authorization:

    python -m src.cli --video raw.mp4 --transcript transcript.json --mode final_video --provider elevenlabs --allow-paid-generation --max-generated-seconds 600 --max-requests 30 --output-dir agent_output/eleven_run

Limits are per provider, in requested generated seconds and request count; they are
**not monetary price estimates**. No API key goes into arguments, JSON outputs or the repository.
Plugin/CLI presence is reported but never treated as proof that a standalone Python process
can use a ChatGPT connector. To use music generated through a connector, export/download it
and provide --elevenlabs-assets instead.

Successful requests are cached in stems/generation_ledger.json. Pending/uncertain requests
are not retried automatically. Check provider history and billing before authorizing a retry.
The ledger prevents silent duplicate spend but cannot resolve a request whose server result
was lost. A .run.lock prevents concurrent jobs from sharing an output directory.

## Google Flow / MusicFX

**Reliable default:** generate/download music through your authorized Google session,
then supply --google-flow-assets assets.json. This avoids depending on changing UI selectors
and works with a ChatGPT/browser-agent workflow.

An opt-in **experimental browser adapter** is implemented for local use:

    python -m pip install -e ".[browser]"
    python -m playwright install chromium
    python scripts/auth_google_flow.py

Sign in manually in the dedicated visible browser on your own computer. The script does not
read credentials and never calls private Google endpoints. A profile directory does not prove
authentication, access, or entitlement. Google can restrict automated browser sign-in.

Inspect your current Google music UI and create a configuration JSON with:

    {
      "url": "https://labs.google/fx/tools/music-fx",
      "prompt": {"role": "textbox", "name": "ACTUAL observed accessible name"},
      "generate": {"role": "button", "name": "ACTUAL observed accessible name"},
      "download": {"role": "button", "name": "ACTUAL observed accessible name"}
    }

Those names are placeholders, not claims about Google's current interface.
Pass --google-flow-browser-config your-config.json --provider google_flow
and --allow-paid-generation. The adapter requires each generation to append a new download
control. It fails clearly on an incompatible/replaced-result UI and never silently downloads an
old track or clicks Generate repeatedly. Use imported assets when that contract does not fit.
Browser generation, sign-in and third-party account access need validation on your own device.

## A/B comparison

    python -m src.cli --video raw.mp4 --transcript transcript.json --mode final_video --provider both_compare --elevenlabs-assets eleven-assets.json --google-flow-assets google-assets.json --output-dir agent_output/compare

This imported-assets version needs no keys or generation authorization. Omit the ElevenLabs
manifest for live API generation; omit the Google manifest only when you supply a working
browser configuration. Both branches are checked before any generation starts.

Outputs are isolated under elevenlabs/ and google_flow/, sharing a single spotting sheet.
The Markdown comparison includes actual loudness/peaks, a 1–4 kHz music/voice energy ratio,
and the availability of verified entry offsets. It does not invent downbeat accuracy or
claim the band-energy proxy is a validated intelligibility score.

## Optional LLM music director

--planner llm sends the transcript, deterministic baseline, and strict spotting schema to
DIRECTOR_LLM_URL, using DIRECTOR_LLM_MODEL and optional DIRECTOR_LLM_API_KEY.
This is a chat-completions-compatible endpoint, which can be local.

This selection sends transcript content to the configured endpoint. It is never enabled
implicitly. Returned plans are validated; bad JSON/timing fails before music generation.
The default rules planner does not claim semantic or emotional understanding.

## Mixing and quality control

- 48 kHz stereo PCM processing with exact cue/silence boundaries.
- Music HPF, depth-dependent LPF, and speech-keyed center presence carve.
- Bounded music-level control using 400 ms K-weighted windows at 100 ms hops,
  20 ms attack and 250 ms release. This is momentary-loudness control, not an
  integrated-LUFS compressor or a promise of a fixed gap for every syllable.
- Per-cue gap deviations are reported instead of hidden. The band proxy and gap
  are measured before final silence and master normalization.
- 5 ms entry and 15 ms exit fades; transient-stop cues never extend into protected speech.
- Protected silence removes music **and SFX** while leaving dialogue intact.
- Two-pass FFmpeg loudnorm, followed by measurement of the actual encoded AAC video.
- Default -15 LUFS / -1.5 dBTP target with a -1 dBTP delivery QC ceiling.
  Targets are presets, not YouTube rules. Deviations return status qc_failed and exit code 3.

Listen before publication. These measurements do not assess clipping already present in a
source, lyrical leakage, musical coherence, artistic quality, or narrative appropriateness.

## Files you get

| File | Purpose |
|---|---|
| spotting_sheet.json | Editable, schema-validated cue/silence plan |
| annotated_music_script.md | Dialogue, prompts, timing and mix instructions |
| cue_sheet.csv | Cue list for editors |
| generation_plan.json | Review requests and duration before generation |
| transcript.json | Timestamp provenance and dialogue |
| dialogue.wav | Extracted original dialogue, when media was supplied |
| provider/stems/*_music.wav | Decoded raw music per cue |
| provider/stems/*_prepared.wav | Prepared cue with gain/silence processing |
| provider/music.wav, sfx.wav | Timeline-length mix stems |
| provider/timeline.xml | FCP7 XML, separate dialogue/music/SFX tracks |
| provider/markers.json | Editorial and suggested visual accents |
| provider/timeline.otio | Optional with the timeline extra |
| provider/master.wav | 24-bit mastered audio, final_video mode |
| provider/final_video_PROVIDER.mp4 | H.264/AAC output |
| provider/qc_report.json | Cue metrics and actual output loudness |
| ab_comparison_report.md | Provider comparison when requested |

XML contains absolute file references and prepared cue clips; DSP is baked into prepared
clips, not falsely represented as portable plugin automation. Raw stems and spotting JSON
let you revise processing. XML/OTIO import behavior varies across NLEs: **Premiere/Resolve UI
round-trip has not been verified**. Preserve folder structure or relink after moving a pack.
Use --fps for the intended XML timebase; XML markers are frame-quantized.

## Docker

    docker build -t duvera-music .
    docker run --rm -v "/absolute/project-work:/work" duvera-music --json-config /work/config.json

The image includes FFmpeg and core dependencies, runs as a non-root user, and is suitable
for imported audio/supplied transcripts. Browser and ASR extras are not installed in it.

## Development and validation

    python -m pip install -e ".[dev]"
    ruff check .
    python -m unittest discover -s tests -v

Tests run real FFmpeg video rendering, encoded output measurements, stem silence,
local-assets comparison, schema errors, budget gates, pending-request protection and
non-TTY behavior. They do not charge provider accounts or pretend to test live Google access.
GitHub Actions runs the same suite on Python 3.10 and 3.12 once this repository is pushed.

See docs/VALIDATION.md for the actual build-session results and docs/RESEARCH_NOTES.md
for evidence boundaries and implementation choices.
