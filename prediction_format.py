#!/usr/bin/env python3
"""Standardize team prediction JSON against a fixed 10-image/10-center split."""
import argparse
import json
import sys
from pathlib import Path

from dataset_common import DatasetError, SPLIT_NAMES, is_number, load_json


def validate_centers(centers, fish_count):
    """Validate and copy [fish][10][x,y] finite pixel coordinates."""
    if not isinstance(centers, list) or len(centers) != fish_count:
        raise DatasetError("predicted_centers must contain one trajectory per track_id.")
    result = []
    for trajectory in centers:
        if not isinstance(trajectory, list) or len(trajectory) != 10:
            raise DatasetError("Each fish must have exactly 10 predicted centerpoints.")
        points = []
        for point in trajectory:
            if (not isinstance(point, list) or len(point) != 2
                    or not all(is_number(value) for value in point)):
                raise DatasetError("Each centerpoint must be [x, y] with finite numbers.")
            points.append([float(point[0]), float(point[1])])
        result.append(points)
    return result


def _integers(values, label, count=None):
    if (not isinstance(values, list) or not values
            or (count is not None and len(values) != count)
            or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in values)
            or len(set(values)) != len(values)):
        raise DatasetError(f"{label} must contain distinct nonnegative integers"
                           + (f" (exactly {count})." if count else "."))
    return values


def standardize_predictions(data, dataset):
    """Return canonical submission JSON, requiring every window exactly once.

    Accept either a list of prediction records or the canonical envelope. Record
    and track order are normalized to the dataset; no missing prediction is
    inferred or filled. Extra fields are rejected to catch typos and label leaks.
    Center coordinates are never scaled, clipped, or replaced with ground truth.
    """
    if dataset.observed != 10 or dataset.future != 10:
        raise DatasetError("Submissions require observed=10 and future=10.")
    expected_header = {"schema_version": 1, "task": "fish_center_prediction",
                       "split": dataset.split, "fps": 30, "coordinate_system": "pixels"}
    if dataset.manifest.get("schema_version") == 2:
        expected_header["release_id"] = dataset.manifest["release_id"]
    if isinstance(data, dict):
        if set(data) != set(expected_header) | {"predictions"}:
            raise DatasetError("Invalid submission envelope fields.")
        if any(type(data[k]) is not type(v) or data[k] != v for k, v in expected_header.items()):
            raise DatasetError("Submission metadata does not match the task or dataset split.")
        records = data["predictions"]
    else:
        records = data
    if not isinstance(records, list):
        raise DatasetError("Predictions must be a list of records.")
    expected = {}
    for index in range(len(dataset)):
        sample = dataset.prediction_input(index)
        sample.pop("input_images")
        key = (sample["chunk_id"], tuple(sample["observed_frames"]))
        expected[key] = sample
    submitted = {}
    fields = {"chunk_id", "observed_frames", "future_frames", "track_ids", "predicted_centers"}
    for record in records:
        if not isinstance(record, dict) or set(record) != fields:
            raise DatasetError("Each prediction requires exactly: " + ", ".join(sorted(fields)))
        if not isinstance(record["chunk_id"], str):
            raise DatasetError("chunk_id must be a string.")
        observed = _integers(record["observed_frames"], "observed_frames", 10)
        future = _integers(record["future_frames"], "future_frames", 10)
        track_ids = _integers(record["track_ids"], "track_ids")
        key = (record["chunk_id"], tuple(observed))
        if key not in expected:
            raise DatasetError(f"Unknown prediction window: {key}")
        if key in submitted:
            raise DatasetError(f"Duplicate prediction window: {key}")
        sample = expected[key]
        if future != sample["future_frames"]:
            raise DatasetError(f"Future frames do not match the next 10 frames: {key}")
        if set(track_ids) != set(sample["track_ids"]):
            raise DatasetError(f"track_ids do not match dataset fish: {key}")
        centers = validate_centers(record["predicted_centers"], len(track_ids))
        by_track = dict(zip(track_ids, centers))
        submitted[key] = dict(sample, predicted_centers=[by_track[t] for t in sample["track_ids"]])
    if len(submitted) != len(expected):
        raise DatasetError(f"Missing prediction windows: expected {len(expected)}, got {len(submitted)}.")
    return dict(expected_header, predictions=[submitted[key] for key in expected])


def standardize_prediction_file(input_path, output_path, dataset):
    """Validate a JSON file and write standardized JSON; refuse existing output."""
    result = standardize_predictions(load_json(input_path), dataset)
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    with Path(output_path).open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    return result


def main():
    from prediction_loader import PredictionDataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=SPLIT_NAMES)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = standardize_prediction_file(
            args.input, args.output, PredictionDataset(args.dataset, args.split))
        print(f"Validated {len(result['predictions'])} prediction windows: {args.output}")
    except (DatasetError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
