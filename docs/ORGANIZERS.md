# Organizer workflow

The GitHub repository contains competition helpers and a synthetic example.
Keep real data releases outside Git. Distribute training and validation images
and labels to participants; retain all test images and labels privately.

## Prepare the first wave

Use corrected COCO annotations sampled at 30 FPS. Numeric fish labels must stay
consistent through each source scene. Existing continuous-video fish labels are
valid inputs: the release step assigns fresh fish IDs to every chunk.
Use a distinct recording filename prefix for each full video, and keep it stable
when annotating more of the same video in later waves.

```bash
python3 split_dataset.py path/to/annotated_video --output data/wave_001/prepared
python3 release_dataset.py build data/wave_001/prepared --release-id wave_001 --output data/wave_001/private
python3 release_dataset.py participants data/wave_001/private --output data/wave_001/participants
```

Distribute **only** `data/wave_001/participants/`. It contains training and
validation folders, per-chunk annotation JSON, and a manifest with no test entries.
The private folder contains all three splits and audit mappings; keep it private.

The splitter retains maximal stable chunks of at least 30 frames. `--min-frames`
may lower the cutoff to 20, the minimum needed for a 10-image/10-target window.
Default splits target 70:15:15 by frame count, using whole chunks in chronological
order within each input. Ratios are approximate and require at least three
usable chunks. The resulting manifest fixes the assignments.

To assign a wave explicitly to one split, including a wave with a single chunk:

```bash
python3 split_dataset.py path/to/new_labels --split train --output data/wave_002/prepared
```

`--split` takes `train`, `validate`, or `test`; it assigns the entire new input
wave to that split. For mixed additions, use the default multi-chunk split or
prepare an organizer-reviewed intermediate manifest with fixed assignments.

## Add later waves

Build from **only newly annotated source frames**, then name a new release and
provide the prior private release:

```bash
python3 release_dataset.py build data/wave_002/prepared --previous data/wave_001/private --release-id wave_002 --output data/wave_002/private
python3 release_dataset.py participants data/wave_002/private --output data/wave_002/participants
```

Wave 002 is a cumulative snapshot containing wave 001 plus new chunks. Earlier
image names, bytes, fish IDs, annotations, chunk numbers, and split assignments
are preserved. Existing recordings retain their video IDs; new recordings get
new IDs. Newly added chunks get the next unused chunk number within their video.

Already released source frames are rejected rather than re-split or duplicated.
If later annotation extends an existing scene, prepare only its unreleased frame
range as a new chunk. Corrections to released labels require an explicit reviewed
revision, rather than using this append-only path. Do not rerun the ratio splitter
over the accumulated released data.

Use immutable release names, such as `wave_001` and `wave_002`, and tell teams which
one to use. The JSON standardizer includes `release_id` and rejects submissions
tagged for another release. All teams in an evaluation round must use the same
release, split, window stride, and scoring rules.

Video/chunk fields support 99 videos and 99 chunks per video; each chunk supports
999 frames and 999 fish. Photos use `VVCCSFFF.ext`, with local frames starting at
001 and split codes 1/2/3. See the [format specification](DATA_FORMAT.md).

## Validate and evaluate

```bash
python3 prediction_loader.py data/wave_002/participants --split validate
python3 prediction_format.py team_predictions.json --dataset data/wave_002/participants --split validate --output submission.json
```

For withheld test evaluation, run the team's model and standardizer against the
private test split on organizer infrastructure. This repository does not implement
a scoring server, blind test-input distribution, or final metric definitions.
The format standardizer validates predictions without computing scores.

All exporters refuse existing output paths. They validate data before making the
completed staged output visible. Original source files are not modified.
Fresh fish IDs prevent accidental cross-chunk identity reuse; they do not remove
visual similarity between chunks of the same video. Use held-out-video splits if
the evaluation must measure generalization to new videos.

## Synthetic demo and optional audit

The README commands build a release directly from the bundled synthetic data.
To regenerate its intermediate source from scratch:

```bash
python3 examples/make_example.py --output data/new_demo_source
python3 split_dataset.py data/new_demo_source --output data/new_demo_prepared
```

For source image decoding, dimensions, box bounds, and trajectory review:

```bash
python3 -m pip install -r tools/requirements-audit.txt
python3 tools/audit_annotations.py path/to/coco_folder data/audit
```

The optional audit needs Pillow. It does not certify FPS or every identity swap.
Preserve source attribution and dataset licensing when distributing real images;
those obligations are separate from the helper code's license.

```bash
python3 -m unittest discover -s tests -v
```
