# Organizer workflow

Keep real releases outside Git. Preserve established train/validation/test
assignments and distribute training/validation normally. Additional valid data
can be shared in a separate code-`4` supplement. If some additional data is
reserved privately, it is tracked as `extra_test`, separate from the official
three splits. Every team receives the same released supplemental labels and
chooses how to use them.

## Prepare a private wave

Use corrected COCO labels sampled at 30 FPS. Numeric fish identities must remain
consistent within a source scene. Continuous-video fish labels are valid inputs;
the release step assigns fresh fish IDs to every chunk. Keep each full video's
recording prefix distinct and stable across waves.

```bash
python3 split_dataset.py path/to/annotated_video --output data/wave_001/prepared
python3 release_dataset.py build data/wave_001/prepared --release-id wave_001 --output data/wave_001/private
python3 release_dataset.py participants data/wave_001/private --output data/wave_001/participants
```

The splitter's default minimum is 30 frames. `--min-frames` may lower it to 20,
the minimum for a 10-image/10-target window. Default internal assignments target
70:15:15 by frame count, keeping whole chunks in chronological source order.
Ratios are approximate and need at least three usable chunks. Review the
manifest, then freeze these assignments.

Crop images and their labels together before chunking. Keep only visible boxes,
clip boxes at the image edge, and recompute their centers from the clipped boxes.
If cropping changes which fish are visible, split chunks at those membership
changes. Match images and labels by the physical source frame: a renamed or
reencoded crop can have a different frame-number offset. Cropped and full-frame
copies of the same moment retain one source identity and split assignment.
Apply the same crop when checking overlapping frames in a later wave. Changing
the crop of an existing release requires a new reviewed revision.

To assign an entire new wave explicitly, including a single chunk:

```bash
python3 split_dataset.py path/to/new_labels --split train --output data/wave_002/prepared
```

`--split` takes `train`, `validate`, or `test`. Mixed additions can use the
automatic split or an organizer-reviewed intermediate manifest.

## Add the remaining valid frames

Short runs can fall outside the original prediction minimum whenever a fish
enters or leaves. Retain those frames as additional chunks:

```bash
python3 release_dataset.py extra path/to/annotated_video --previous data/wave_001/private --release-id wave_001_full --output data/wave_001_full/private
```

This preserves previous chunks and adds unassigned source frames in consecutive
runs with fixed annotated membership. Runs may be shorter than 20 frames. They
receive new chunk numbers and globally distinct fish IDs; their section is
`extra`, using filename code `4`. Their `extra_counts` do not alter the original
three `split_counts`.

Use the previously reviewed source export. Generic fish identities are not
invented for unassigned annotations. Review conflicts and duplicate footage
before release. To omit a known bad or duplicate recording, use
`--exclude-recording RECORDING_NAME`; repeat for more exclusions. The command
does not automatically detect perceptual duplicates or fix timing alignment.

Python API:
`add_extra_release(source_dir, output_dir, release_id, previous, excluded_recordings=())`.

## Reserve a supplemental holdout

Select whole `extra` chunks for the desired held-out frame proportion, then
reserve their IDs in a new private release:

```bash
python3 release_dataset.py reserve-extra data/wave_001_full/private --release-id wave_001_reserved --output data/wave_001_reserved/private --chunk-id 0104 --chunk-id 0203
```

Replace the example IDs with reviewed `extra` chunk IDs from your manifest.
Selection is explicit; the command does not choose a percentage automatically.
For a target such as 15%, compare frame totals rather than chunk counts and
record the selected IDs. Whole chunks may make the achieved percentage
approximate.

Reserved chunks move from `extra` to `extra_test`, and their filename digit
changes from `4` to `3`. Chunk IDs, fish IDs, bytes, boxes, and centers remain
unchanged; annotation filename references are updated. The manifest distinguishes
`extra_test` from the official `test` set. `extra_test_counts` are recorded
separately, and the original train/validation/test entries remain fixed.

Python API: `reserve_extra_chunks(private_dir, output_dir, release_id, chunk_ids)`.

Keep a supplemental holdout private from its creation. Previously distributed
supplemental images cannot become an unseen test set merely by changing their
section code.

## Export only the released supplement

```bash
python3 release_dataset.py public data/wave_001_reserved/private --section extra --output data/wave_001_reserved/supplement
```

Distribute this supplemental folder or its ZIP separately from the existing
training/validation package. It includes only the remaining unreserved `extra`
chunks; neither the original train/validation data nor either private holdout
is repeated in this export. Every image has code `4`.

The standalone package has `distribution="public"`, paths `public/<chunk_id>`,
and `split="public"`. Its manifest omits original filenames, source identity
maps, private entries, and private counts. Chunk IDs, fish IDs, image bytes,
resolution, and geometry are preserved.

The broader `public` command without `--section extra` is also available. It
combines all non-held sections into one code-`4` package, including training and
validation copies. Use it only when that larger combined pool is intended.
The `participants` command preserves separate train/validate folders plus any
unreserved `extra/`; it omits both `test` and `extra_test`.

## Add later waves

Build from newly annotated frames, preserve earlier private assignments, then
repeat additional-data review and any reservation before export:

```bash
python3 release_dataset.py build data/wave_002/prepared --previous data/wave_001_reserved/private --release-id wave_002 --output data/wave_002/private
```

Prior files, IDs, chunk numbers, and original split assignments stay fixed.
New chunks use the next available number within their video. `build` rejects
already released source frames rather than re-splitting them. Corrections to
released labels require a reviewed revision, not this append-only path.

Use immutable release names and tell teams which version to use. The prediction
standardizer includes `release_id` and rejects another version's tag. Photo
fields support 99 videos, 99 chunks per video, and 999 frames/fish per chunk.
Local frames start at 001. See the [format specification](DATA_FORMAT.md).

## Validate and evaluate

```bash
python3 prediction_loader.py data/wave_001_reserved/supplement --split public
python3 prediction_format.py team_predictions.json --dataset data/wave_001_reserved/supplement --split public --output submission.json
```

The loader creates 10-input/10-target windows only from chunks with at least 20
frames. All shorter chunks remain available for detection/tracking via their
annotation files. Use `PredictionDataset(private_release, "test")` for the
original private test set, or `PredictionDataset(private_release, "extra_test")`
for the separately reserved additional data. Standardize those private results
with the matching split name.

This repository does not supply final metrics, a scoring server, or a blind
test-input release. The standardizer checks format and alignment, not scores.
All exporters refuse existing outputs and validate staged files before making
the completed folder visible. Source files are preserved. Nearby same-video
chunks can remain visually similar; these are held-out moments unless whole
recordings are reserved separately.

## Synthetic demo and optional audit

The README builds the bundled synthetic train/validation example. To regenerate
its intermediate source:

```bash
python3 examples/make_example.py --output data/new_demo_source
python3 split_dataset.py data/new_demo_source --output data/new_demo_prepared
```

For decoding, dimensions, box bounds, and trajectory review:

```bash
python3 -m pip install -r tools/requirements-audit.txt
python3 tools/audit_annotations.py path/to/coco_folder data/audit
```

The optional audit needs Pillow and does not certify FPS or every identity swap.
Preserve source attribution and licensing when distributing images; those
obligations are separate from the helper code's license.

```bash
python3 -m unittest discover -s tests -v
```
