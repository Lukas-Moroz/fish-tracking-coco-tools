# Organizer workflow

## Prepare a dataset

Use corrected, individually identified COCO annotations sampled at 30 FPS.
Put each continuous shot in its own input folder. Numeric labels must represent
consistent fish identities within a recording; same membership alone cannot
detect every identity swap.

```bash
python3 split_dataset.py "path/to/shot_a" "path/to/shot_b" --output data/official_v1 --ratios 70 15 15
```

The utility retains maximal stable chunks of at least 30 frames. A missing or
added fish box, duplicate identity, unassigned label, frame gap, or sequence
change breaks a chunk. Short runs are reported and omitted.

Whole chunks are assigned in chronological order within each input, preserving
the supplied input-folder order. It chooses the pair of boundaries closest to
the requested train/validate/test frame proportions, with at least one chunk
in each split. It is deterministic and does not shuffle shots. At least three
qualifying chunks are required. Use `--min-frames` to change the cutoff (minimum
20 for the default prediction task).

The current corrected recording yields chunks of **105, 73, and 75 frames**:
105 train, 73 validation, and 75 test. These yield **86, 54, and 56** joint
10-observed/10-future windows. Ratios are approximate, not promises of exact
70/15/15 percentages. All three chunks contain 253 frames in total.

Same-video chunks may occupy different splits for evaluation on unseen moments
in the same environment. The utility imposes no extra time buffer. Before a
release, inspect the manifest and scene/lighting/fish-density coverage. Reorder
source arguments if the result does not represent the collected data, then
export to a new versioned directory. More footage can support held-out-camera
or held-out-recording evaluation if that later becomes a competition objective.

Once released, publish the fixed manifest and instruct participants to use those
assignments. Participants should not rerun the splitter on the official package.

## Regenerate the synthetic example

The repository includes a prepared example at `examples/dataset/`. To reproduce
it in a new local directory:

```bash
python3 examples/make_example.py --output data/example_source
python3 split_dataset.py data/example_source --output data/example_dataset
python3 prediction_loader.py data/example_dataset --split train
```

The generator produces 120 small PNG frames and COCO annotations for three
40-frame shots. It uses only the standard library. Paths above must not already
exist; choose new names when rerunning. Each prepared split has 21 windows.

## Optional annotation audit

The core utilities have no third-party dependencies. The audit alone needs Pillow:

```bash
python3 -m pip install -r tools/requirements-audit.txt
python3 tools/audit_annotations.py "path/to/shot/train" data/audit
```

The audit accepts one COCO folder with one sequence. It checks decoding,
dimensions, identities, box bounds, frame gaps, movements, size changes, and
presence runs. Large changes are review candidates, not automatic labeling
errors. It does not certify FPS or identity through every occlusion.

## Distribution and evaluation

This repository supplies preparation and loading tools plus a synthetic demo.
Real competition data is distributed separately unless the officials explicitly
choose to publish it. Generated data, local audits, and JSONL targets are ignored
by Git. The repo's demo is not an official training or benchmark dataset.

The prepared output contains annotations for all three splits. The prediction
loader's test samples include future ground truth. An open-label release can
ship the whole output; a blind leaderboard needs separate test inputs and
private ground truth. Do not assume the current exporter automatically hides
test labels.

The competition's final scoring protocol and submission format are not fixed by
these utilities. Candidate measures discussed are detection AP/recall, tracking
IDF1/identity switches, and prediction average/final displacement error. Publish
the chosen definitions and scoring code before judging submissions.

Code and dataset licensing are separate decisions. Preserve the source dataset's
attribution and licensing when distributing real images; the current Roboflow
export's README identifies it as CC BY 4.0. A code license does not replace that
dataset license.

## Run checks

```bash
python3 -m unittest discover -s tests -v
```

Tests exercise source ordering, membership changes and visibility gaps,
whole-chunk split proportions, source preservation, identity mapping, sample
alignment, and rejection of reused frames or malformed input. Run the README's bundled-example command as an end-to-end check before distributing a release.
