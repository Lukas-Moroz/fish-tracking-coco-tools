# Organizer workflow

Keep real releases outside Git. Participants receive one public pool, marked
with filename code `4`, and may choose their own training/validation usage.
Keep reserved test images and labels private. The private release preserves the
organizer's internal train/validation/test assignments for audit and evaluation.

## Prepare a private wave

Use corrected COCO annotations sampled at 30 FPS. Numeric fish labels must stay
consistent within each scene. Existing continuous-video labels are valid inputs;
the release step assigns fresh fish IDs for every chunk. Keep each full video's
recording prefix distinct and stable across waves.

```bash
python3 split_dataset.py path/to/annotated_video --output data/wave_001/prepared
python3 release_dataset.py build data/wave_001/prepared --release-id wave_001 --output data/wave_001/private
```

The splitter's default minimum is 30 frames. `--min-frames` may lower it to 20,
the minimum for a 10-image/10-target window. Default internal splits target
70:15:15 by frame count, keeping whole chunks in chronological source order.
Ratios are approximate and require at least three usable chunks. Review the
manifest, then freeze the assignments.

To assign a new wave explicitly to one internal split, including a single chunk:

```bash
python3 split_dataset.py path/to/new_labels --split train --output data/wave_002/prepared
```

`--split` takes `train`, `validate`, or `test`. For mixed additions, use the
automatic split or an organizer-reviewed intermediate manifest.

## Preserve remaining valid frames

The private prediction split can omit short runs whenever a fish enters or
leaves. Add those remaining valid frames as `extra/` chunks before making the
public package:

```bash
python3 release_dataset.py extra path/to/annotated_video --previous data/wave_001/private --release-id wave_001_full --output data/wave_001_full/private
```

This preserves previous chunks and assignments, then adds unassigned source
frames in consecutive runs with fixed annotated membership. Runs may contain
fewer than 20 frames. They receive new chunk numbers and globally distinct fish
IDs, and their photos use section code `4`. Their `extra_counts` are separate
from the three official `split_counts`.

The source should match the previously reviewed export. No generic identity is
invented for an unassigned annotation. Audit any conflicts or duplicate footage
before building the release. To omit an explicitly reviewed bad or duplicate
recording, pass `--exclude-recording RECORDING_NAME`; repeat the option for more
than one recording. This is a deliberate organizer exclusion, not automatic
perceptual duplicate detection.

Python API:
`add_extra_release(source_dir, output_dir, release_id, previous, excluded_recordings=())`.

## Export the public pool

```bash
python3 release_dataset.py public data/wave_001_full/private --output data/wave_001_full/participants
```

Distribute **only** the resulting participant folder or its ZIP. The public
export combines all non-test chunks under `public/`. Every public photo uses
split digit `4`; internal train/validation filenames are renamed in the public
copy, and corresponding annotation/manifest references are updated. Chunk IDs,
fish IDs, image bytes, and geometry are preserved.

The public manifest has `distribution="public"`, `split="public"` for every
chunk, and only public counts. It omits test entries, original filenames, and
source identity maps. Earlier public training/validation labels are not imposed
on participants. Each team decides how to use the shared data.

Python API: `export_public_pool(private_dir, output_dir)`.

The legacy `participants` command remains available for releases that explicitly
need separate train/validate folders. New shared-pool releases use `public`.

## Add later waves

Prepare only newly annotated source frames, then extend the private release:

```bash
python3 release_dataset.py build data/wave_002/prepared --previous data/wave_001_full/private --release-id wave_002 --output data/wave_002/private
python3 release_dataset.py extra path/to/updated_source --previous data/wave_002/private --release-id wave_002_full --output data/wave_002_full/private
python3 release_dataset.py public data/wave_002_full/private --output data/wave_002_full/participants
```

Prior image files, labels, chunk numbers, fish IDs, and official assignments stay
fixed. Existing recordings retain their video IDs; new recordings get new IDs.
New chunks use the next unused number within the video. The `build` command
rejects already released source frames rather than re-splitting them. Label
corrections require a reviewed revision, not this append-only path.

Use immutable release names and tell all teams which release to use. The JSON
standardizer includes `release_id` and rejects records tagged for another release.
All evaluation participants must use the same release and window selection.

Photo fields support 99 videos, 99 chunks per video, and 999 frames/fish per
chunk. Local frames start at 001. See the [format specification](DATA_FORMAT.md).

## Validate and evaluate

```bash
python3 prediction_loader.py data/wave_002_full/participants --split public
python3 prediction_format.py team_predictions.json --dataset data/wave_002_full/participants --split public --output submission.json
```

The public loader indexes 10-input/10-target windows only where a chunk has at
least 20 frames. All shorter chunks remain useful through their annotation files
for detection and tracking. The private test split is never part of the public
export; run official evaluation on organizer infrastructure using the private
release. This repository does not supply a scoring server, blind test-input
release, or final metric definitions.

All exporters refuse existing output paths. They validate staged outputs before
publishing the completed folder and preserve source files. Unique chunk IDs do
not remove visual similarity between neighboring footage; these splits measure
held-out moments unless organizers reserve whole recordings separately.

## Synthetic demo and optional audit

The README builds a public release from the bundled synthetic example. To
regenerate its intermediate source:

```bash
python3 examples/make_example.py --output data/new_demo_source
python3 split_dataset.py data/new_demo_source --output data/new_demo_prepared
```

For source decoding, dimensions, box bounds, and trajectory review:

```bash
python3 -m pip install -r tools/requirements-audit.txt
python3 tools/audit_annotations.py path/to/coco_folder data/audit
```

The optional audit needs Pillow. It does not certify FPS or every identity swap.
Preserve source attribution and dataset licensing when distributing images; these
obligations are separate from the helper code's license.

```bash
python3 -m unittest discover -s tests -v
```
