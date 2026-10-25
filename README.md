# Duvera Music Director

Dialogue-aware music direction, optional three-tier sound design, and visual pacing for YouTube videos.
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

Narration/music rendering now defaults to the vocal-check gate. Install its extra:

    python scripts/bootstrap.py --extras vocals

This installs Demucs and a compatible PyTorch/torchaudio pair. It is a substantial optional
download, not a lightweight frequency-band detector. Model weights load/download when the
gate prepares, before any paid music generation. The text-only workflow does not need it.
Use Python 3.10–3.12 for the pinned legacy Demucs extra; an isolated environment is recommended.

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
The procedural-tone demo explicitly sets vocal_policy=off. Its reports say
disabled_unchecked; it does not claim a real vocal-detection test.

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
Word timestamps are also embedded in transcript.json when ASR supplies them; the narrative
planner uses them to anchor a pivot inside an ASR segment instead of always using its start.

## Narrative direction and visual boundaries

The rules and LLM planners now share deterministic narrative guards:

- Contrast markers create a sectional change. Reflective Hinglish/Hindi/English phrases
  change the music color; the reflective preset uses a **4.5 kHz LPF**.
- Strong realizations, punchlines and idioms create protected music/SFX silence.
  Merely saying "lekin" does not automatically mute a sentence.
- Video analysis samples luminance and the proportion of dark/bright pixels. An abrupt
  sustained drop to a mostly black frame with sparse highlights is treated as a
  **black-card candidate**, not proof that text was read.
- Black-card sequences remain protected until their end or a later transcript CTA.
  A spoken CTA can bring music back even while the screen stays black.
- Pre-silence cues use a late-transient exit with a bounded short decay/tail.
  The tail finishes before the protected line; it never spills into music silence.
- Fully muted cues do not request music or SFX generation.

The outro return reuses the first audible cue's approved music rather than generating a
new closing song. It starts from that asset's selected loop/beat entry. Automatic beat
estimates remain distinct from verified downbeats.
Use --outro-return-sec 82 to anchor a reviewed return at 01:22, or
--no-outro-music-return to keep a black-card closing sequence silent.
CTA callbacks such as a repeated punchline do not create a second mute inside the same CTA.
Before silence, the exit prefers a real transient around 0.5 seconds before the boundary;
if there is no usable transient, it reports a microfade fallback instead of inventing a pluck.

The 4 fps default gives quarter-second visual timing resolution. Tune
--visual-sample-fps, --black-luma-threshold, --black-pixel-ratio and
--black-min-duration-sec, or use --no-visual-analysis. Thresholds are engineering
defaults, not validated universal text detectors. Startup black frames, short flashes,
and gradual dim/night transitions are deliberately not treated as abrupt black-card events.

Inspect visual_analysis.json and spotting_sheet.json before generation. With a supplied
--spotting-sheet, your reviewed editorial decisions take priority over automatic guards.
Use a fresh output directory after this update; old cached plans are not migrated.

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

Loop stitching now selects a stable whole-beat region, in multiples of
--beats-per-bar (default **assumed 4/4**). Crossfades retain the exact loop period:
they no longer subtract 100 ms from each repeat. Adjacent cues using the same imported
asset and loop region continue its phase through mood/LPF changes; after protected
silence, the resumed cue starts afresh.

A beat grid does not establish the true first beat of a bar. Automatic loops report
bar-phase uncertainty. Supply verified source endpoints to remove the assumption:

    {
      "music": {"cue_001": "track.wav", "cue_002": "track.wav"},
      "loops": {
        "cue_001": {"start_sec": 0.5, "end_sec": 12.5},
        "cue_002": {"start_sec": 0.5, "end_sec": 12.5}
      }
    }

--loop-crossfade-ms controls the boundary blend (default 40 ms). When no stable grid
is available, a fixed sample-period fallback is explicitly reported in qc_report.json.
Use verified endpoints for rubato, irregular meters, harmonic mismatches or perceptible
seams; quantization alone cannot guarantee an inaudible harmonic edit.

## Optional SFX and visual pacing

Install the local CV extra for frame-level detection:

    python scripts/bootstrap.py --extras cv,vocals

Choose --sfx-mode none (default), accents_only, or full_3_tier. The extension adds
stable black-card/text-mask accents, coherent camera-motion candidates, evidence-driven
ambience/Foley, per-tier stems, and sample-clock acoustic-crest placement. Ordinary cuts
do not automatically receive a sound. The existing music workflow remains the default.

    python -m src.cli --video raw.mp4 --transcript transcript.json --sfx-mode accents_only --mode script_only --output-dir agent_output/sfx_plan

Inspect event_graph.json, pacing_plan.json and generation_plan.json. For full_3_tier,
supply reviewed acoustic location/object events with --event-graph events.json.
The CV detector uses stable bright masks, **not OCR**, and does not infer café/hotel/traffic
semantics. Missing semantic events produce a warning rather than invented ambience.
Use --no-pacing-detection to work exclusively from supplied events without OpenCV.

Protected silence still removes every effects layer by default. The explicit
--sfx-silence-policy accents option permits title sub-drops/text ticks during those
windows while music, ambience and Foley remain silent. This changes the editorial intent:
leave it at protect to retain the fully silent deadpan delivery from the earlier patch.

--visual-pacing plan writes punch-in instructions; render applies them in final_video mode.
Automatic punch-ins require a transcript emphasis/contrast AND stable local Haar eye-box
geometry. Reviewed eye anchors can be supplied in events instead. Rendering currently
requires constant frame rate; analysis can use VFR presentation timestamps. Pause/cut
candidates are review notes; no breath deletion, dialogue J/L retiming, or shot removal occurs.

Run a complete six-second offline three-tier/zoom fixture:

    python scripts/create_pacing_demo.py --dir pacing_demo
    python -m src.cli --json-config pacing_demo/config.json

It uses synthetic tones/noise and explicitly supplied fixture events/anchor. It makes
no API calls, bypasses the music vocal check explicitly, and demonstrates engineering
behavior rather than sound-design taste or real face recognition. It also works without cv.

See [docs/SFX_PACING.md](docs/SFX_PACING.md) for the event contract, asset imports,
generation limits, measurement/QC definitions and current detector limitations.

For the breakfast-vlog Render #4 audit, see [docs/RENDER5.md](docs/RENDER5.md).
The new recipe pins an existing music-only bed by SHA-256, enables the requested audible
effects, and requires its sub-drop/six ticks rather than silently skipping them.
Enabled SFX with zero accepted events now blocks before generation. Every final render
also verifies that the actual premix contains the SFX waveform at unity gain.

--sfx-mix-profile audible uses sample-peak targets for accents and active-block RMS for
ambience, retaining independent true-peak ceilings. It is an explicit alternative to the
original research loudness profile, not a new definition of LUFS or true peak.

## Social commentary preset

--preset social_commentary enables a narration chain, a documentary accent palette, and
chapter-based music variation. Long scripts use 90–150 second editorial chapters; an
11:18 transcript receives five chapter stems in the rules preset. Chapter moods/titles
are proposals, and --planner llm can refine them through your explicitly configured endpoint.
Chapter splits reuse their chapter music instead of requesting a fresh song per sentence.

Supply reviewed third-party insert ranges with --archival-windows, or explicitly enable
local reference-voice candidates with --narrator-reference and the optional speaker extra.
No reference/ranges means archival detection is disabled and reported. An unfamiliar
voice does not establish that a clip is archival. Original insert audio bypasses the
narration chain, and added music/SFX mute across its protected interval.

    python -m src.cli --preset social_commentary --video raw.mp4 --transcript transcript.json --archival-windows archive-windows.json --event-graph evidence-events.json --mode script_only --output-dir agent_output/commentary_plan

Read [docs/SOCIAL_COMMENTARY.md](docs/SOCIAL_COMMENTARY.md) for the full setup, JSON
contracts, vocal-chain measurements and current model/editorial limits.

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

## Instrumental enforcement and the vocal-check gate

The instrumental contract is enforced at the planner, asset store, and both provider
adapters. Conflicting vocal-request clauses are removed; every music request receives
the same strict no-vocals/no-singing/no-choir/no-vocal-chops suffix. ElevenLabs also keeps
force_instrumental=true. A prompt or API flag alone is not audio-content validation.

Generated, cached, and imported music all pass through the gate before stem mixing:

- **--vocal-policy reject** (default): separate a vocal estimate, measure it, and block
  a suspect asset. A missing/broken checker blocks the job before paid provider generation.
- **--vocal-policy strip**: if suspect, use Demucs's no_vocals backing (drums+bass+other),
  restore its source level and duration, and run a second separation/proxy check.
  Residual failure blocks mixing. Only the music asset is separated; narration stays intact.
- **--vocal-policy off**: explicitly bypass the gate. Outputs report disabled_unchecked.
  This option is used by the synthetic offline fixture, not presented as a passing check.

Example using an existing local asset manifest:

    python -m src.cli --video raw.mp4 --transcript transcript.json --mode final_video --provider local --local-assets assets.json --vocal-policy strip --outro-return-sec 82 --output-dir agent_output/render3

The default threshold is -35 dBFS **referenced to a -18 dBFS RMS mixture**, measured
over active 400 ms windows. This level-normalized proxy catches short loud vocal
estimates that a whole-track average might hide. It is not the arbitrary raw input
fader level, a formant-band heuristic, or a validated singing classifier.
Tune --vocal-threshold-dbfs only after listening/calibration. Demucs can leak instruments
into its vocal estimate or miss quiet vocals; passing the threshold cannot prove vocal absence.

Use --vocal-model and --vocal-device (cpu default, cuda/mps optional) for your setup.
The selected model/device is checked before account credits can be consumed.
Quality checks never automatically regenerate paid music. Rejected raw assets are retained:
import those saved files into a new strip run to avoid generating/paying again.
Review vocal_checks/vocal_qc.json for original/residual measurements and the selected file.
Use prepared or approved files in your editor; retained raw generations can contain rejected vocals.

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
- Protected silence removes music **and SFX** while leaving dialogue intact;
  the optional explicit accents policy is documented above.
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
| visual_analysis.json | Luminance-based boundaries and analysis resolution |
| event_graph.json, pacing_plan.json | Evidence, accepted/rejected SFX decisions, visual/pause review |
| pacing_analysis.json | Optional frame-PTS CV evidence, without OCR/VLM claims |
| provider/ambience.wav, foley.wav, accent.wav | Optional separate three-tier effects stems |
| provider/sfx_qc.json | Actual placement samples, crest errors, pre-master levels and true peaks |
| vocal_checks/vocal_qc.json | Per-asset vocal proxy decisions and approved file paths |
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

The core image includes FFmpeg and runs as a non-root user. Its default vocal gate requires
the vocals extra; build with --build-arg INSTALL_VOCALS=1 for narration/music processing.
Model weights still need to be cached or downloaded. The core image supports text-only
planning and explicitly unchecked synthetic fixtures. Browser and ASR extras are not installed.

## Development and validation

    python -m pip install -e ".[dev,cv]"
    ruff check .
    python -m unittest discover -s tests -v

Tests run real FFmpeg video rendering, encoded output measurements, stem silence,
local-assets comparison, schema errors, budget gates, pending-request protection and
non-TTY behavior. They do not charge provider accounts or pretend to test live Google access.
Vocal-gate tests use controlled separator outputs; they verify reject/strip/residual,
missing-model and no-paid-call behavior, not Demucs's acoustic detection accuracy.
GitHub Actions runs the same suite on Python 3.10 and 3.12 once this repository is pushed.

See docs/VALIDATION.md for the actual build-session results and docs/RESEARCH_NOTES.md
for evidence boundaries and implementation choices.
