# Fish tracking competition helpers

Prepare annotated video chunks, load **10 input images**, and standardize the
**next 10 predicted centerpoints per fish** for evaluation. Teams supply their
own models. The core helpers require Python 3.8+ and no third-party packages.

This repository contains helper code, documentation, tests, and a small synthetic
example. Real competition images, annotations, and private holdouts are separate
releases and excluded from Git.

## Dataset sections

The established training and validation data keep their assignments. Additional
valid frames can be released as a separate **supplemental package**, marked with
code `4`; every team receives the same supplemental labels and may decide how
to use them. Organizers keep both the official test set and any additional
supplemental holdout private.

Photo names use **`VVCCSFFF.ext`**:

| Field | Meaning |
|---|---|
| `VV` | Full video ID, `01`–`99` |
| `CC` | Chunk within that video, `01`–`99` |
| `S` | `1` train, `2` validation, `3` private test/holdout, `4` supplemental/shared public |
| `FFF` | Frame within the chunk, starting at `001`, up to `999` |

For example, `01044001.jpg` is video 01, chunk 04, supplemental data, frame 001.
Codes `3` and `4` describe the data section; they do not replace individual fish
identities. Fish IDs remain constant within a chunk and are never reused in
another chunk. Numeric `track_id` values identify fish; `class_id=0` is the shared
detection class.

Each chunk has one `annotations.json`. A manifest indexes its images, identities,
and section. A standalone supplemental export looks like this:

```text
supplement/
  manifest.json
  README.txt
  public/
    0104/
      images/01044001.jpg ...
      annotations.json
```

The export uses `distribution="public"` and `split="public"` for its selected
shared data. Exports preserve the prepared image bytes, resolution, boxes, and
centers. Coordinates refer to the released images, including organizer crops.
The existing train/validate package remains usable with its original split names.

## Try the synthetic example

```bash
python3 release_dataset.py build examples/dataset --release-id demo_001 --output data/demo_001/private
python3 release_dataset.py participants data/demo_001/private --output data/demo_001/participants
python3 prediction_loader.py data/demo_001/participants --split train
```

The example contains colored shapes, not real fish. Its training and validation
splits each provide 21 windows. Outputs must not already exist.

## Use a team's model

```python
from prediction_loader import PredictionDataset, predict_sample

dataset = PredictionDataset("data/demo_001/participants", "train")
# For a standalone supplemental package instead:
# dataset = PredictionDataset("path/to/supplement", "public")

sample = dataset[0]
images = sample["input_images"]  # Ten image paths in time order
targets = sample["future"]       # Labels: [fish_count][10][2]
fish_ids = sample["track_ids"]   # Required trajectory order

# Connect your team's predictor(images) before calling:
# record = predict_sample(dataset, 0, predictor)
```

`predict_sample` passes exactly ten encoded image byte strings to
`predictor(images)`. Your adapter decodes them and returns Python lists shaped
`[fish_count][10][2]`, in the window's `track_ids` order. Its output is the next
ten `[x, y]` centers per fish. The call supplies no center labels or future
images. `prediction_input(index)` provides paths and alignment metadata.

A chunk with `N` frames provides `max(0, N - 19)` prediction windows at stride 1.
Windows never cross chunks. Chunks shorter than 20 frames remain available for
detection/tracking through their images and annotation JSON, but supply no
10-input/10-target windows. Fish counts may vary between chunks. Coordinates
always use pixels of the released images.

## Standardize prediction JSON

Save your prediction records as a JSON list, then validate the matching package
and split. For a supplemental-only package:

```bash
python3 prediction_format.py team_predictions.json --dataset path/to/supplement --split public --output submission.json
```

The standardizer checks complete window coverage, frames, fish IDs, trajectory
shape, and finite points. It includes the dataset release ID. See the
[prediction contract](docs/PREDICTION_FORMAT.md) for the API. Organizers define
metrics and evaluate reserved data separately.

## Prepare and grow the dataset

Organizers prepare stable chunks with `split_dataset.py` and build private
releases with `release_dataset.py`. The `extra` command retains valid source
frames omitted by the minimum chunk length. `reserve-extra` can move selected
additional chunks into a separate private holdout without changing the original
three split counts. Export the remaining supplement with `public --section extra`.

See [organizer commands](docs/ORGANIZERS.md) and the
[annotation and manifest format](docs/DATA_FORMAT.md).

## Run checks

```bash
python3 -m unittest discover -s tests -v
```

The optional source annotation audit in `tools/` needs Pillow; the competition
helpers do not. Code and dataset licenses are separate; a code license has not
yet been selected for this repository.
