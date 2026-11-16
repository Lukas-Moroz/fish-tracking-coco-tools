# Fish tracking competition helpers

Prepare annotated video chunks, load **10 input images**, and standardize the
**next 10 predicted centerpoints per fish** for evaluation. Teams supply their
own models. The core helpers require Python 3.8+ and no third-party packages.

This repository contains helper code, documentation, tests, and a small synthetic
example. Real competition images, annotations, and private test data are
separate releases and excluded from Git.

## Public dataset

Everyone receives the same **public pool** and may choose how to use it for
training and validation. The organizers reserve test images and labels privately.
The public pool includes valid short chunks as well as longer prediction chunks.
Every chunk has its own `annotations.json`; `manifest.json` indexes the data.

```text
participants/
  manifest.json
  README.txt
  public/
    0101/
      images/01014001.jpg ...
      annotations.json
    0102/
      images/01024001.jpg ...
      annotations.json
```

Photo names use **`VVCCSFFF.ext`**:

| Field | Meaning |
|---|---|
| `VV` | Full video ID, `01`–`99` |
| `CC` | Chunk within that video, `01`–`99` |
| `S` | `4` = public data; internal codes are `1` train, `2` validation, `3` private test |
| `FFF` | Frame within the chunk, starting at `001`, up to `999` |

For example, `01014001.jpg` is video 01, chunk 01, public data, frame 001.
All public photos use code `4`, including photos that had an internal training
or validation assignment. Their chunk IDs and fish IDs stay the same.

Fish IDs are constant within a chunk and **never reused in another chunk**.
Numeric `track_id` values identify fish; `class_id=0` is the shared detection
class. Images retain their original bytes and resolution. All boxes and
centerpoints use original-image pixels.

## Try the synthetic example

```bash
python3 release_dataset.py build examples/dataset --release-id demo_001 --output data/demo_001/private
python3 release_dataset.py public data/demo_001/private --output data/demo_001/participants
python3 prediction_loader.py data/demo_001/participants --split public
```

The example contains colored shapes, not real fish. Its public pool provides
42 prediction windows. Outputs must not already exist.

## Use a team's model

```python
from prediction_loader import PredictionDataset, predict_sample

dataset = PredictionDataset("data/demo_001/participants", "public")
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
ten `[x, y]` centerpoints per fish. The call supplies no center labels or future
images. `prediction_input(index)` provides paths and alignment metadata.

A chunk with `N` frames provides `max(0, N - 19)` prediction windows at stride 1.
Windows never cross chunks. Chunks shorter than 20 frames remain available for
detection and tracking through their images and annotation JSON, but supply no
10-input/10-target windows. Fish counts may vary between chunks.

## Standardize prediction JSON

Save your `predict_sample` records as a JSON list, then run:

```bash
python3 prediction_format.py team_predictions.json --dataset data/demo_001/participants --split public --output submission.json
```

The standardizer checks complete window coverage, frame alignment, fish IDs,
ten points per fish, and finite coordinates. It emits the shared submission
format with the dataset's release ID. See the
[prediction contract](docs/PREDICTION_FORMAT.md) for the API. Organizers define
scoring rules and run private test evaluation separately.

## Prepare and grow the dataset

Organizers prepare stable chunks with `split_dataset.py` and build a private
release with `release_dataset.py`. Existing internal assignments remain fixed.
The optional `extra` step adds valid source frames omitted by the minimum chunk
length rule. The `public` export combines every non-test chunk under code `4`.

See [organizer commands](docs/ORGANIZERS.md) and the
[annotation and manifest format](docs/DATA_FORMAT.md).

## Run checks

```bash
python3 -m unittest discover -s tests -v
```

The optional source annotation audit in `tools/` needs Pillow; the competition
helpers do not. Code and dataset licenses are separate; a code license has not
yet been selected for this repository.
