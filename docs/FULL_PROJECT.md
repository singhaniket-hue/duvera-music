# Full multi-part documentary projects

The production runner edits numbered real camera parts, composes attributed local
B-roll and document pages, then inserts original-sound references by pausing and
resuming the edited camera clock. Private project JSON, transcripts, source media,
generation receipts and output remain outside this repository.

```powershell
python -m src.production_project --project D:/Videos/MyProject/project.json --jobs 2
python -m src.project_score --project D:/Videos/MyProject/project.json --music-manifest D:/Videos/MyProject/score/manifest.json
```

The current assembly contract is 1920x1080, constant 60fps camera parts and stereo
48kHz audio. Source-sound inserts can use other dimensions/frame rates; they are
normalized to the assembly clock with source-attribution overlays. Every archive
insert receives an explicit music/SFX protection window. No original insert is
presented as newly recorded footage or used to manufacture evidence for unrelated claims.

A project contains an `output_dir`, ordered `parts` and optional `insertions`.
Each part needs `id`, `source`, an actual ASR `transcript`, and a source-clock `plan`
conforming to DocumentaryEditPlan. Use absolute paths. An insertion declares
`part_id`, `at_source_sec`, its `source`, `source_sha256`, `source_in_sec`,
`duration_sec`, and a visible `credit`. Its boundary snaps to a source video frame
after the retake map, and all subsequent ASR, effects and silence times shift by
the inserted frame count. `assembly_plan.json` records every interval.

`media_cutaway` events hold actual video or photo pixels over the real narration.
They require a file, hash and attribution; videos cannot be looped or padded to
fake missing coverage. Vertical sources use a blurred version of their own real
pixels as a background. `document_card` events frame supplied pages in the HTML
paper template and require their source URL and image hash. `context_note` is a
small HTML attribution/qualification bar, useful for historical dates, accused
status, illustrative examples and limitations explicitly supplied by the editor.
Context notes are separate from headline/evidence cards and never claim external
verification for the speaker's opinion.

Part receipts verify source and rendered-picture hashes. The renderer, overlay,
grade, cutaway and retake code fingerprints participate in the render signature.
A deliberately reused prior render also needs its independently verified source
hash and original picture hash. A changed plan/source requires a new project
output folder; do not discard unrelated existing outputs.

The four-theme score runner is intended for 6–10 minute films, with 90–150 second
chapters and four checked instrumental source tracks. The manifest's `music`
entries identify cue_001 through cue_004, and `vocal_qc_path` identifies the
VocalGuard receipt. All source hashes must match selected, successfully checked
assets. Generation is a separate explicit provider/budget step; mastering cannot
generate or automatically regenerate a paid asset.

The grade feathers its skin-protection mask at quarter resolution and caches
stable anchors. It retains the source-pixel feather width while avoiding an
expensive full-HD large-kernel mask blur. Camera grading and iris crops remain
full resolution. Thread limits prevent nested OpenCV encoders from consuming an
unbounded thread pool during multi-part rendering.

Full-project QC checks frame count, original source-sound samples before programme
mastering, sample-zero added music/effects through archive windows and a voice-only
closing. Every accent passes the actual encoded-MP4 audibility gate; filters use
a bounded neighbourhood with a 250ms warm-up. Subs use 35–150Hz, soft low-pitched
whooshes use a 300Hz high-pass, and paper/UI transients use an 1800Hz high-pass.
The final ebur128 log and delivery-qc.json are kept with the project. Artistic
delivery and the ASR/acoustic classification still require listening judgement.
