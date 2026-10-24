# Evidence and implementation boundaries

The supplied research brief informed this implementation. Its embedded citation tokens
cannot be resolved from the attachment alone and were not copied as verified references.

Primary implementation references checked during the build:

- https://elevenlabs.io/docs/api-reference/music/compose — official music endpoint,
  duration contract and instrumental flag.
- https://elevenlabs.io/docs/api-reference/text-to-sound-effects/convert — separate SFX API.
- https://ffmpeg.org/ffmpeg-filters.html — loudnorm, audio mixing and filtering.
- https://www.itu.int/rec/R-REC-BS.1770 — framework for program loudness measurement.
- https://librosa.org/doc/0.11.0/generated/librosa.beat.beat_track.html — beat tracking,
  not meter/downbeat inference.

Important corrections to the sample implementation in the brief:

1. Full music is not generated through the SFX endpoint.
2. Audio responses are decoded with FFmpeg, not renamed to WAV.
3. Syllable-to-BPM estimates are heuristics and Hindi schwa deletion/code-switching
   cannot be solved accurately with a few regexes.
4. No language-specific intelligibility band is asserted as a validated law.
5. Onset detection does not establish meter or a true downbeat.
6. Transient exits must remain inside the cue; fades prevent clicks after filtering too.
7. Narration and silence remain intact through stem assembly.
8. Provider comparison reuses the same sheet and reports only measurements available.
9. A stored browser profile/CLI binary is not an authenticated provider session.
10. AAC output is measured after encoding; a successful render alone does not pass QC.

Implemented choices:

- Deterministic rules first, optional JSON-constrained LLM planning.
- Local supplied-asset workflow is fully usable without any provider.
- Manual verified downbeat offsets are supported. Optional librosa estimates beats.
  Madmom/learned bar-phase models and full visual scene classification are not bundled.
- Update: streaming luminance boundary detection now supplies observed black-card candidates.
  This uses no OCR; it does not understand text or classify narrative emotion from images.
  Beat-quantized loops preserve their exact period and report bar-phase uncertainty.
  Verified loop endpoints remain the reliable option for irregular/ambiguous music.
- K-weighting is implemented at the fixed internal 48 kHz sample rate. 400 ms control
  windows are momentary energy estimates without integrated gating. Final integrated
  loudness and true peak come from FFmpeg loudnorm.
- Presence carving acts on the music center channel. Fullband music control is bounded
  and reports unmet gap windows; changing the arrangement may be preferable to deeper ducking.
- XML uses portable source references and prepared clips. It does not fabricate portable
  per-plugin EQ/volume automation across NLEs. The JSON plan is the editable source of truth.

Not included as validated capabilities: autonomous emotional understanding, precise
Hinglish forced alignment, source separation, automatic soundtrack licensing checks, verified
bar-level downbeats, NLE UI round-trip, perceptual intelligibility or retention predictions.
