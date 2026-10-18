# Team prediction contract (version 1)

The prediction task takes **10 consecutive input images** and produces the
**next 10 centerpoints per fish**. All frames are sampled at 30 FPS. A centerpoint
is `[x, y]` in original-image pixels, measured from the top-left corner, with x
increasing rightward and y downward. Training targets are bounding-box midpoints.
If a model resizes or crops images, convert its output back to original-image
coordinates before submission.

## Model interface

```python
import json
from prediction_loader import PredictionDataset, predict_sample
from prediction_format import standardize_predictions

dataset = PredictionDataset("data/demo_001/participants", "validate")

def predictor(images):
    # images is a list of exactly 10 encoded image byte strings, oldest first.
    # Decode them using your image library and run your team's model here.
    # Return Python lists [fish_count][10][2], in the window's track_ids order.
    # For tensor/array outputs, convert with .tolist() after moving to CPU.
    raise NotImplementedError("Connect your team's model here")

# Dataset IDs specify the required output alignment. Your adapter must associate
# model trajectories with these identities; the utility does not infer identity.
# Inspect dataset.prediction_input(index) for track_ids and frame metadata.
records = [predict_sample(dataset, index, predictor) for index in range(len(dataset))]
submission = standardize_predictions(records, dataset)
with open("submission.json", "x", encoding="utf-8") as stream:
    json.dump(submission, stream, indent=2, allow_nan=False)
```

The helper calls `predictor(images)` with image bytes only. The model receives
neither past/future ground-truth centers nor future images from this call. The
output is ordered first by the window's `track_ids`, then by future frame number,
then by `[x, y]`. A window with two fish therefore returns shape `[2][10][2]`.
Different chunks can have different fish counts.

`dataset.prediction_input(index)` returns image paths, `chunk_id`, `track_ids`,
`observed_frames` and `future_frames`, with no coordinate labels. The dataset
constructor still requires prepared annotations to validate and index windows.
Training samples from `dataset[index]` also include `observed` and `future`
labels. Participants receive training and validation only. Test evaluation runs
on the organizer's private release; selecting test on a participant package is
an error.

## Submission JSON

Every team submits a single JSON object with these exact top-level fields:

```json
{
  "schema_version": 1,
  "release_id": "wave_001",
  "task": "fish_center_prediction",
  "split": "validate",
  "fps": 30,
  "coordinate_system": "pixels",
  "predictions": [
    {
      "chunk_id": "0102",
      "observed_frames": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      "future_frames": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
      "track_ids": [102001],
      "predicted_centers": [
        [[50, 25], [51, 25], [52, 25], [53, 25], [54, 25],
         [55, 25], [56, 25], [57, 25], [58, 25], [59, 25]]
      ]
    }
  ]
}
```

This illustrates one window with one fish; use actual chunk IDs, frame numbers
and fish IDs from your dataset and include **every window** in the chosen split.
The standardizer uses the default stride of 1 through the CLI. Python callers
may use the organizers' chosen stride via `PredictionDataset(..., stride=...)`;
all teams must use the same fixed window selection.

The window key is `(chunk_id, observed_frames)`. Fish IDs are constant within a
chunk and never reused in a different chunk. Frame numbers are local to the
chunk, starting at 1. `release_id` binds the submission to the dataset release;
submissions tagged for another wave are rejected. For legacy schema-version-1
datasets only, the envelope omits `release_id` and uses original frame numbers.
Future frames must be the exact 10 frames immediately after the input window.
The standardizer accepts either a raw JSON list of these records or a complete
version-1 object. It puts windows in dataset order, puts fish in manifest order,
converts coordinates to JSON floats, and emits the complete object above.

```bash
python3 prediction_format.py team_predictions.json --dataset path/to/participant_package --split validate --output submission.json
```

Python file API: `standardize_prediction_file(input_path, output_path, dataset)`.
In-memory API: `standardize_predictions(records_or_submission, dataset)`.

Missing, extra or duplicate windows; wrong frame numbers or identities; extra
fields; wrong trajectory shapes; and nonfinite/nonnumeric coordinates are errors.
Predictions are never padded, clipped, or filled from ground truth. Finite points
outside image bounds are preserved for evaluation. Existing output files are
refused, and validation finishes before any output file is created.

This contract covers centerpoint predictions. Dataset annotation JSON retains
its [per-chunk format](DATA_FORMAT.md). The final scoring policy is defined
separately by organizers.
