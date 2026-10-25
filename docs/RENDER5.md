# Render #5: preserve the music, make effects explicit, verify the mux

The Render #4 audit describes an audible problem. Its three suggested causes are hypotheses:
the inspected code already summed effects in NumPy and used FFmpeg amix normalize=0.
It also already used zero-padded 400ms momentary loudness for micro-effects rather than
normalizing an 86-second sparse bus to integrated LUFS. No run config, generation ledger,
spotting sheet or isolated Render #4 stems accompanied the mixed MP4, so its exact run
path, actual effect levels and source of the reported vocal texture cannot be established.

Confirmed code/config risks addressed here:

- SFX defaulted to none, protected black-card silence suppressed accents, and location/
  object events were not invented without supplied evidence. The recipe explicitly sets
  full_3_tier, accents and the new audible profile and supplies the audited events.
- Enabled SFX could finish with zero accepted effects. It now blocks before providers.
  A supplied required event that is rejected by confidence/cooldown/silence rules also
  blocks. Planning still writes decisions and blocking_issues for review.
- The old bright-mask area floor missed compact text, and low IoU sensitivity missed
  changes at the same text position. The area floor is now 0.03% BEFORE dilation;
  connected components suppress isolated noise and three-frame stability remains.
- A source onset beyond 3s fell back to file time zero. Automatic music entry now finds
  sustained audio, avoiding an entirely silent generated intro. Supplied verified entries
  stay authoritative; unexpected music gaps >750ms fail premux QC outside protected
  silence. Intentional pauses belong in SilenceWindow, including their exit tail.
- The fixed accent-bus ceiling would have attenuated a louder sub-drop back to −7dBTP.
  Bus headroom now respects its planned event ceilings; any bus attenuation is reported.

## Use the already-approved Render #3 music

A git commit such as 31e1549 identifies code, not generated audio or a reproducible music
seed. You need its existing **timeline music.wav** and original spotting_sheet.json.
Do not use master.wav or a final MP4: those include narration and cannot preserve an
isolated bed. The source video in your base config must still contain raw dialogue with
the same edit clock; adding music to a baked music+voice mix would double the soundtrack.

From the repository root on Windows (use .venv/bin/python on macOS/Linux):

```powershell
git pull
python scripts/bootstrap.py --extras cv,vocals
.venv\Scripts\python scripts/prepare_breakfast_v5.py --base-config path/to/raw-config.json --music-bed path/to/render3/elevenlabs/music.wav --spotting-sheet path/to/render3/spotting_sheet.json --dir breakfast_v5
.venv\Scripts\python -m src.cli --json-config breakfast_v5/config.json --plan-only
```

The helper creates a config and an event graph; it does not generate sound or render.
It disables live generation until separately authorized and sets vocal_policy=reject.
The copied original spotting sheet remains authoritative for music/silence decisions.
The bed must be timeline-length stereo 48kHz PCM16, match its SHA-256, and respect those
silences. It is rechecked by the vocal gate before any SFX generation; if stripping would
change its content, the locked run blocks. Resolve/review that bed before making a new lock.
No music or legacy bridge-generation jobs are scheduled in locked mode. New pacing
effects are the only asset jobs. The music.wav output remains byte-identical to the lock.

After reviewing generation_plan.json, either import all SFX in its asset IDs:

```powershell
.venv\Scripts\python -m src.cli --json-config breakfast_v5/config.json --provider local --local-assets path/to/sfx-assets.json
```

Or use ElevenLabs generation with your securely configured local credentials and explicit
authorization. Use a fresh output directory if provider/input flags differ from the plan:

```powershell
.venv\Scripts\python -m src.cli --json-config breakfast_v5/config.json --provider elevenlabs --allow-paid-generation --output-dir breakfast_v5/elevenlabs_render
```

Budget caps/pending ledger protection still apply. No generated assets or secrets ship in
the repository. There is no promise of reobtaining the same music through regeneration.

## Recipe timing and levels

The location/action times below come from the user's audit. Black-card/text boundaries
were refined against frame presentation timestamps in the attached
Breakfast-Vlog-With-Music-43f2955.mp4. They are correct only for that source edit.

| Event | Visual time | Audio policy |
|---|---:|---|
| Courtyard | 0s | ambience |
| Outbound auto | 6s | start 5.55s, 450ms ambience J lead |
| Lobby | 16s | ambience change |
| Buffet | 22s | ambience change |
| Four selected plate cuts | 24.1, 25.3, 27, 29s | soft shutters, cooldown retained |
| Phone action | 31.2s | generic connection Foley |
| Hotel room | 39s | room tone |
| Return auto | 47s | start 46.55s, 450ms ambience J lead |
| Black-card reveal | 68s | sub crest at 68.010s |
| Six requested card changes | 69.166667, 71, 73.9, 75.8, 79.066667, 81.666667s | tick crest +6ms |

The extra detected text change at 84.2s remains unaccented to keep the six-tick brief.
The quoted audit rounded some card changes: placing its last tick at exactly 82s would
lag the observed 81.667s visual by about one third of a second.

The audible profile uses −3.5dBFS sample peak for the sub (−3dBTP ceiling), −8dBFS sample
peak for small transients, and −28dBFS RMS over active 20ms ambience blocks. Quiet blocks
more than 30dB below the strongest active block are excluded from that RMS statistic.
It is not LUFS. The existing research profile remains available and reports 400ms/3s
K-weighted loudness. Sample peaks and reconstructed true peaks are separately measured.
Actual ceilings can keep an asset below target; losses over the bounded 6.5dB allowance
fail rather than disappear silently. Listen and revise levels for the intended playback.

Audible accents explicitly permitted inside a protected black-card window bypass
spectral attenuation there. Music, ambience and Foley remain silent. Elsewhere, speech-
band control remains active. A nonzero low-frequency sub does not guarantee reproduction
on every phone speaker, and LF effects still consume bus/master headroom.

## Checks before a final MP4

- generation_plan.json and run_summary.json identify the chosen mode/profile/silence
  policy, accepted effect count and any blocking issues.
- sfx_qc.json records every effect's start/crest samples, prepared sample peak, active
  RMS, windowed loudness, FFmpeg true peak, bus attenuation and routed contribution peak.
  These are technical measures, not proof of subjective audibility or appropriateness.
- sfx_stem.wav is the actual combined effects input, identical to sfx.wav. A scheduled
  all-zero bus fails. No whole-program integrated normalization or music gain is applied
  to it. A constant safety attenuation only occurs if summation requires headroom and
  is reported/baked into the editor buses too.
- The FFmpeg premix uses normalize=0 and produces an actual dialogue+music reference.
  Subtracting that reference from the premix must recover the exact SFX waveform within
  floating-point tolerance. Dropped/muted/divided effects fail this routing check.
- music_continuity reports unplanned gaps and the pin's digest. Failures in effect level,
  continuity or nonzero checks prevent final muxing and return qc_failed.
- Final mastering acts on the combined program; therefore pre-master sample-peak targets
  need not equal encoded AAC peaks. Existing final loudness/true-peak QC still applies.

The regression suite uses a real 86-second effects bus with one sub and six ticks, a
controlled final AAC render, a deliberately divided FFmpeg mixer, compact vertical card
masks, and exact locked-bed reuse. These tests make no paid calls and do not establish
real Demucs vocal detection accuracy or an artistically approved Render #5.
