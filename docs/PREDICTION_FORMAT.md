# Team prediction contract (version 1)

The task takes **10 consecutive input images** and produces the **next 10
centerpoints per fish**. Frames are sampled at 30 FPS. A centerpoint is `[x, y]`
in original-image pixels, measured from the top-left corner, with x rightward
and y downward. Targets are bounding-box midpoints. Convert resized or cropped
model outputs back to original-image coordinates before submission.

Everyone receives the same public pool and decides how to use it for training
and validation. Public photos use section code `4`. The private benchmark test
set is reserved for organizer evaluation.

## Model interface

```python
import json
from prediction_loader import PredictionDataset, predict_sample
from prediction_format import standardize_predictions

dataset = PredictionDataset("data/demo_001/participants", "public")

def predictor(images):
    # Exactly 10 encoded image byte strings, oldest first.
    # Decode/preprocess and run your team's model here.
    # Return Python lists [fish_count][10][2], in the window's track_ids order.
    # Convert tensor/array outputs to CPU Python lists as needed.
    raise NotImplementedError("Connect your team's model here")

# Your adapter must associate model trajectories with the required fish IDs.
# Inspect dataset.prediction_input(index) for IDs and frame metadata.
records = [predict_sample(dataset, index, predictor) for index in range(len(dataset))]
submission = standardize_predictions(records, dataset)
with open("submission.json", "x", encoding="utf-8") as stream:
    json.dump(submission, stream, indent=2, allow_nan=False)
```

`predictor(images)` receives image bytes only. The helper supplies no past/future
center labels or future images to that call. Output order is fish in `track_ids`
order, future frame number, then `[x, y]`. A two-fish window returns `[2][10][2]`.
Fish counts may vary between chunks.

`prediction_input(index)` returns input paths, chunk ID, fish IDs, and observed/
future frame metadata without coordinate labels. The dataset constructor still
requires prepared annotations to validate windows. `dataset[index]` also exposes
`observed` and `future` labels for training or analysis.

The public pool can include short supplemental chunks. All chunks validate, but
only those with at least 20 frames supply 10-input/10-target windows. Each chunk
supplies `max(0, N - 19)` windows at stride 1, and windows never cross chunks.
Shorter chunks remain available for detection/tracking through their annotation
files. Public-window formatting is not an official public scoring split.

## Prediction JSON

The standardized public prediction file has these fields:

```json
{
  "schema_version": 1,
  "release_id": "wave_001_full",
  "task": "fish_center_prediction",
  "split": "public",
  "fps": 30,
  "coordinate_system": "pixels",
  "predictions": [
    {
      "chunk_id": "0101",
      "observed_frames": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      "future_frames": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
      "track_ids": [101001],
      "predicted_centers": [
        [[50, 25], [51, 25], [52, 25], [53, 25], [54, 25],
         [55, 25], [56, 25], [57, 25], [58, 25], [59, 25]]
      ]
    }
  ]
}
```

This illustrates one window. Use actual IDs/frame numbers from the chosen
release and include every eligible window in the selected dataset. The CLI uses
stride 1. Python callers may use an organizer-specified stride; a validation run
must use the same fixed window selection as its prediction records.

The window key is `(chunk_id, observed_frames)`. Fish IDs remain constant within
a chunk and are not reused in other chunks. Frame numbers are chunk-local,
starting at 1. Future frames must be the exact ten immediately after the inputs.
`release_id` binds predictions to a version; another release's tag is rejected.
For legacy schema-version-1 datasets only, the envelope omits `release_id` and
uses original frame numbers.

The standardizer accepts a raw list or the complete envelope, orders windows
and fish like the dataset, converts coordinates to JSON floats, and writes the
shared object:

```bash
python3 prediction_format.py team_predictions.json --dataset path/to/participant_package --split public --output submission.json
```

File API: `standardize_prediction_file(input_path, output_path, dataset)`.
In-memory API: `standardize_predictions(records_or_submission, dataset)`.

Missing/extra/duplicate windows, wrong frames or identities, extra fields,
incorrect trajectory shapes, and nonfinite/nonnumeric points are errors.
Predictions are not padded, clipped, or filled from ground truth. Finite points
outside image bounds are retained. Existing outputs are refused and validation
finishes before writing the file.

## Private evaluation and legacy packages

Organizers use `PredictionDataset(private_release, "test")` for reserved test
windows. Public packages contain no test data and cannot be opened with that
split. Legacy packages with separate train/validate folders remain supported
with their corresponding split names; a public-pool package uses `public`.

The standardizer checks format and alignment rather than calculating scores.
Official metrics and submission-portal rules are defined separately. Dataset
annotations retain the [per-chunk format](DATA_FORMAT.md).
