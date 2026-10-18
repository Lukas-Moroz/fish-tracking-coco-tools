# Fish tracking competition helpers

Prepare annotated video chunks, load **10 input images**, and standardize the
**next 10 predicted centerpoints per fish** for evaluation. Teams supply their
own models. The core helpers require Python 3.8+ and no third-party packages.

This repository contains helper code, documentation, tests, and a small synthetic
example. Real competition images, annotations, and withheld test data are
distributed separately and excluded from Git.

## Dataset layout

Participants receive training and validation data. Test images and labels stay
with the organizers. Every chunk has its own `annotations.json`; `manifest.json`
indexes the chunks and fixed split assignments without combining their labels.

```text
participants/
  manifest.json
  train/
    0101/
      images/01011001.jpg ...
      annotations.json
  validate/
    0102/
      images/01022001.jpg ...
      annotations.json
```

Photo names use **`VVCCSFFF.ext`**:

| Field | Meaning |
|---|---|
| `VV` | Full video ID, `01`–`99` |
| `CC` | Chunk within that video, `01`–`99` |
| `S` | `1` = train, `2` = validation, `3` = private test |
| `FFF` | Frame within the chunk, starting at `001`, up to `999` |

For example, `01011001.jpg` is video 01, chunk 01, training, frame 001.
The validation folder and loader split are named `validate`.

Fish IDs stay constant within a chunk and are **never reused in another chunk**,
including chunks of the same video. The exporter remaps existing continuous-video
annotations automatically. Images retain their original bytes and resolution;
all boxes and centerpoints use original-image pixels.

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

# Use the training or validation split supplied by the organizers.
dataset = PredictionDataset("data/demo_001/participants", "train")
sample = dataset[0]
images = sample["input_images"]  # Ten image paths in time order
targets = sample["future"]       # Training labels: [fish_count][10][2]
fish_ids = sample["track_ids"]   # Required trajectory order

# Connect your team's predictor(images) before calling:
# record = predict_sample(dataset, 0, predictor)
```

`predict_sample` passes exactly ten encoded image byte strings to
`predictor(images)`. Your adapter decodes them and returns Python lists shaped
`[fish_count][10][2]`, in the window's `track_ids` order. Its output is the next
ten `[x, y]` centerpoints per fish. The call supplies no center labels or future
images. `prediction_input(index)` provides the input paths and alignment metadata.

Windows advance by one frame and never cross chunk or split boundaries. A chunk
of `N` frames gives `max(0, N - 19)` windows. Fish counts may vary between chunks.

## Standardize prediction JSON

Save your `predict_sample` records as a JSON list, then validate and standardize:

```bash
python3 prediction_format.py team_predictions.json --dataset data/demo_001/participants --split validate --output submission.json
```

Use records from the same split named in the command. The standardizer checks
complete window coverage, frame alignment, fish identities, ten points per fish,
and finite coordinates. It emits the shared submission format with the dataset's
release ID. The [prediction contract](docs/PREDICTION_FORMAT.md) gives the full API.
Scoring policy is defined separately by the organizers.

## Prepare and grow the dataset

Organizers first prepare stable chunks from COCO exports with `split_dataset.py`,
then build a named release with `release_dataset.py`. New waves extend the prior
private release while preserving all existing chunk files, names, IDs, and split
assignments. The participant export copies only training and validation chunks.

See [organizer commands](docs/ORGANIZERS.md) and the
[annotation and manifest format](docs/DATA_FORMAT.md).

## Run checks

```bash
python3 -m unittest discover -s tests -v
```

The optional source annotation audit in `tools/` needs Pillow; the competition
helpers do not. Code and dataset licenses are separate; a code license has not
yet been selected for this repository.
