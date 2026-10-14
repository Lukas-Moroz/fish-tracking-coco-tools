#!/usr/bin/env python3
"""Load 10-observed/10-future center trajectories inside one fixed dataset split."""
import argparse
import json
import math
import sys
from pathlib import Path

from dataset_common import DatasetError, SPLIT_NAMES, child_path, is_number, load_json, positive_integer


class PredictionDataset:
    """Indexable, dependency-free joint trajectory samples.

    observed/future have shape [fish_count][time_steps][2], ordered by track_ids.
    Samples are built on demand, in chunk/frame order. Coordinates are pixels.
    Split names are train, validate, test. Defaults: 10 past, 10 future, stride 1.
    This class also works as a map-style dataset for a PyTorch DataLoader with
    an appropriate collate function for variable fish counts.
    """

    def __init__(self, dataset_dir, split, observed=10, future=10, stride=1):
        if split not in SPLIT_NAMES:
            raise DatasetError(f"Unknown split: {split}")
        for value, label in ((observed, "Observed length"), (future, "Future length"), (stride, "Stride")):
            positive_integer(value, label)
        self.root = Path(dataset_dir).resolve()
        self.split, self.observed, self.future, self.stride = split, observed, future, stride
        self.manifest = load_json(self.root / "manifest.json")
        if not isinstance(self.manifest, dict) or not isinstance(self.manifest.get("chunks"), list):
            raise DatasetError("Manifest must be an object containing a chunks list.")
        if self.manifest.get("schema_version") != 1 or self.manifest.get("fps") != 30:
            raise DatasetError("Expected schema_version=1 and fps=30.")
        self.chunks, self.index = [], []
        seen_paths, seen_frames = set(), set()
        # Validate all manifest chunks so a split-specific loader also catches
        # frame reuse across the other split assignments.
        for entry in self.manifest["chunks"]:
            if entry["split"] not in SPLIT_NAMES:
                raise DatasetError(f"Invalid split assignment: {entry['split']}")
            folder = child_path(self.root, entry["path"])
            if Path(entry["path"]).parts[0] != entry["split"]:
                raise DatasetError(f"Chunk path does not match its split: {entry['path']}")
            if folder in seen_paths:
                raise DatasetError(f"Repeated chunk path: {folder}")
            seen_paths.add(folder)
            frames = entry["frames"]
            if not frames or entry["frame_count"] != len(frames):
                raise DatasetError(f"Incorrect frame count: {entry['chunk_id']}")
            numbers = [f["number"] for f in frames]
            if not all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in numbers):
                raise DatasetError(f"Invalid frame numbers: {entry['chunk_id']}")
            if numbers != list(range(numbers[0], numbers[0] + len(frames))):
                raise DatasetError(f"Nonconsecutive frames: {entry['chunk_id']}")
            for frame in frames:
                key = (entry["source"], entry["recording_id"], frame["number"])
                if key in seen_frames:
                    raise DatasetError(f"Frame reused across chunks: {key}")
                seen_frames.add(key)
            if entry["split"] != split:
                continue
            track_ids = entry["track_ids"]
            if (not track_ids or not all(isinstance(t, int) and not isinstance(t, bool) and t >= 0 for t in track_ids)
                    or len(track_ids) != len(set(track_ids))):
                raise DatasetError(f"Invalid track IDs: {entry['chunk_id']}")
            names = [frame["image"] for frame in frames]
            if len(names) != len(set(names)):
                raise DatasetError(f"Repeated image name: {entry['chunk_id']}")
            positions = {name: {} for name in names}
            records = load_json(folder / "annotations.json")
            if not isinstance(records, list):
                raise DatasetError(f"Annotations must be a list: {entry['chunk_id']}")
            for record in records:
                name, track_id = record["image"], record["track_id"]
                if (not isinstance(track_id, int) or isinstance(track_id, bool) or track_id < 0
                        or name not in positions or track_id in positions[name]
                        or not isinstance(record["class_id"], int) or isinstance(record["class_id"], bool)
                        or record["class_id"] != 0):
                    raise DatasetError(f"Invalid image, duplicate identity or detection class in {entry['chunk_id']}")
                point = record["center"]
                xy = [point["x"], point["y"]]
                if not all(is_number(v) for v in xy):
                    raise DatasetError(f"Invalid center: {name}")
                bbox = record["bbox"]
                box_values = [bbox[k] for k in ("x", "y", "width", "height")]
                if not all(is_number(v) for v in box_values):
                    raise DatasetError(f"Invalid bounding box: {name}")
                x, y, width, height = box_values
                if width <= 0 or height <= 0 or x < 0 or y < 0:
                    raise DatasetError(f"Invalid bounding box: {name}")
                if not (math.isclose(xy[0], x + width/2, abs_tol=1e-9)
                        and math.isclose(xy[1], y + height/2, abs_tol=1e-9)):
                    raise DatasetError(f"Center does not match box midpoint: {name}")
                positions[name][track_id] = xy
            for name in names:
                if set(positions[name]) != set(track_ids):
                    raise DatasetError(f"Fish membership changes inside chunk: {entry['chunk_id']}, {name}")
                image = child_path(folder / "images", name)
                if not image.is_file():
                    raise DatasetError(f"Missing photo: {image}")
            trajectories = [[positions[name][track_id] for name in names] for track_id in track_ids]
            chunk_index = len(self.chunks)
            self.chunks.append({"entry": entry, "trajectories": trajectories})
            for start in range(0, len(frames) - observed - future + 1, stride):
                self.index.append((chunk_index, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        chunk_index, start = self.index[index]
        chunk = self.chunks[chunk_index]
        entry = chunk["entry"]
        middle, end = start + self.observed, start + self.observed + self.future
        # Copy coordinates so callers cannot mutate future samples in the dataset.
        observed = [[point[:] for point in trajectory[start:middle]] for trajectory in chunk["trajectories"]]
        future = [[point[:] for point in trajectory[middle:end]] for trajectory in chunk["trajectories"]]
        return {"chunk_id": entry["chunk_id"], "recording_id": entry["recording_id"], "source": entry["source"],
                "split": self.split, "fps": 30, "track_ids": entry["track_ids"][:],
                "observed_frames": [f["number"] for f in entry["frames"][start:middle]],
                "future_frames": [f["number"] for f in entry["frames"][middle:end]],
                "observed": observed, "future": future}

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--split", required=True, choices=SPLIT_NAMES)
    parser.add_argument("--observed", type=int, default=10)
    parser.add_argument("--future", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--output", type=Path, help="Optionally write every sample as JSONL (refuses existing files)")
    args = parser.parse_args()
    try:
        dataset = PredictionDataset(args.dataset_dir, args.split, args.observed, args.future, args.stride)
        if args.output:
            with args.output.open("x", encoding="utf-8") as f:
                for sample in dataset:
                    f.write(json.dumps(sample) + "\n")
        print(f"{args.split}: {len(dataset)} joint windows, "
              f"{sum(len(dataset.chunks[i]['entry']['track_ids']) for i, _ in dataset.index)} per-fish trajectories")
        if not args.output and len(dataset):
            print("First sample:")
            print(json.dumps(dataset[0], indent=2))
    except (DatasetError, OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
