# Team prediction contract (version 1)

The task takes **10 consecutive input images** and produces the **next 10
centerpoints per fish**. Frames are sampled at 30 FPS. Points are `[x, y]` in
pixels of the released dataset images, with x rightward and y downward from the
top-left corner. Targets are bounding-box midpoints. Organizer crops apply to
official prediction images and labels as well as supplemental data. If a model
resizes or further crops an input, convert its outputs back to the released
image's coordinates before submission. The manifest records organizer image
transforms when present.

The established train/validation package retains its original split names.
Teams can also use a supplemental-only package of code-`4` data. Both official
test and separately reserved additional holdout data stay private. Code `4`
describes shared data, not a generic fish identity.

## Model interface

```python
import json
from prediction_loader import PredictionDataset, predict_sample
from prediction_format import standardize_predictions

# A standalone supplemental export uses the public split.
dataset = PredictionDataset("path/to/supplement", "public")
# An existing train/validation package uses its corresponding split instead:
# dataset = PredictionDataset("path/to/participants", "train")

def predictor(images):
    # Exactly 10 encoded image byte strings, oldest first.
    # Decode/preprocess and run your team's model here.
    # Return lists [fish_count][10][2], in the window's track_ids order.
    raise NotImplementedError("Connect your team's model here")

# The team's adapter must associate trajectories with the required IDs.
records = [predict_sample(dataset, i, predictor) for i in range(len(dataset))]
submission = standardize_predictions(records, dataset)
with open("submission.json", "x", encoding="utf-8") as stream:
    json.dump(submission, stream, indent=2, allow_nan=False)
```

`predictor(images)` receives image bytes only. The helper supplies no center
labels or future images to the call. Output order is fish in `track_ids` order,
future frame number, then `[x, y]`. A two-fish window returns `[2][10][2]`.
Fish counts may vary between chunks. Convert tensor/array outputs to CPU Python
lists as needed.

`prediction_input(index)` supplies input paths, chunk ID, fish IDs, and frame
metadata without coordinate labels. The constructor requires prepared labels
to validate windows. `dataset[index]` also exposes `observed` / `future` labels
for training and analysis.

Supplemental chunks can be short. All chunks validate, but only those with at
least 20 frames supply 10-input/10-target windows: `max(0, N - 19)` at stride 1.
Windows never cross chunks. Shorter chunks remain available for detection and
tracking through their images and annotation files.

## Prediction JSON

A standalone supplemental package produces this public-format envelope:

```json
{
  "schema_version": 1,
  "release_id": "wave_001_reserved",
  "task": "fish_center_prediction",
  "split": "public",
  "fps": 30,
  "coordinate_system": "pixels",
  "predictions": [
    {
      "chunk_id": "0104",
      "observed_frames": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      "future_frames": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
      "track_ids": [104001],
      "predicted_centers": [
        [[50, 25], [51, 25], [52, 25], [53, 25], [54, 25],
         [55, 25], [56, 25], [57, 25], [58, 25], [59, 25]]
      ]
    }
  ]
}
```

This is one illustrative window. Include all eligible windows in the selected
package and split, using their actual IDs/frame numbers. The CLI defaults to
stride 1; Python callers may use an organizer-specified stride consistently.
The window key is `(chunk_id, observed_frames)`.

Fish IDs stay constant within a chunk and are not reused elsewhere. Local frame
numbers start at 1, and future frames must immediately follow the ten inputs.
`release_id` binds predictions to a version; mismatched versions are rejected.
Legacy schema-version-1 datasets omit the release ID and use source frame
numbers.

The standardizer accepts a raw record list or the complete envelope, orders
windows/fish like the dataset, converts points to JSON floats, and writes the
shared object:

```bash
python3 prediction_format.py team_predictions.json --dataset path/to/supplement --split public --output submission.json
```

File API: `standardize_prediction_file(input_path, output_path, dataset)`.
In-memory API: `standardize_predictions(records_or_submission, dataset)`.
Missing, extra, or duplicate windows; wrong frames or IDs; extra fields; wrong
shapes; and nonfinite/nonnumeric points are errors. Predictions are not padded,
clipped, or filled from ground truth. Finite points outside the image are
retained. Existing outputs are refused; validation finishes before writing.

## Private evaluation

Organizers can load the original official test set with
`PredictionDataset(private_release, "test")` and separately reserved additional
data with `PredictionDataset(private_release, "extra_test")`. The envelope's
`split` must match (`test` or `extra_test`). Those sections have filename code
`3`, but remain distinct in the private manifest and its counts.

Shared exports omit both private sections. A supplemental-only package uses
`public`, and an existing train/validation package uses its corresponding names.
Code `4` and public prediction formatting do not define a new official scored
split or combine the supplemental holdout with the original test set.

The standardizer validates format/alignment and does not calculate scores.
Organizers define metrics and submission-portal rules separately. Dataset labels
retain the [per-chunk format](DATA_FORMAT.md).
