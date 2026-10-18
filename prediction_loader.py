#!/usr/bin/env python3
"""Load 10-observed/10-future center trajectories inside one fixed dataset split."""
import argparse
import json
import math
import sys
from pathlib import Path

from dataset_common import DatasetError, SPLIT_NAMES, child_path, is_number, load_json, photo_name, positive_integer


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
        version = self.manifest.get("schema_version")
        if type(version) is not int or version not in (1, 2) or self.manifest.get("fps") != 30:
            raise DatasetError("Expected schema_version=1 or 2 and fps=30.")
        if version == 2 and (self.manifest.get("filename_format") != "VVCCSFFF"
                             or self.manifest.get("track_id_scope") != "globally_unique_per_chunk"
                             or self.manifest.get("distribution") not in ("private", "participants")
                             or not isinstance(self.manifest.get("release_id"), str)):
            raise DatasetError("Invalid version-2 release metadata.")
        video_registry = {}
        if version == 2:
            videos, waves = self.manifest.get("videos"), self.manifest.get("waves")
            if (not isinstance(videos, list) or not isinstance(waves, list) or not waves
                    or any(not isinstance(w, str) or not w for w in waves)
                    or len(set(waves)) != len(waves) or waves[-1] != self.manifest["release_id"]):
                raise DatasetError("Invalid video registry or release history.")
            used_video_ids = set()
            for video in videos:
                if (not isinstance(video, dict) or not isinstance(video.get("recording_id"), str)
                        or not video["recording_id"] or type(video.get("video_id")) is not int
                        or not 1 <= video["video_id"] <= 99
                        or video["video_id"] in used_video_ids or video["recording_id"] in video_registry):
                    raise DatasetError("Video IDs and recording names must be distinct in the registry.")
                video_registry[video["recording_id"]] = video["video_id"]
                used_video_ids.add(video["video_id"])
        if self.manifest.get("distribution") == "participants" and split == "test":
            raise DatasetError("Test data is withheld from the participant package.")
        self.chunks, self.index = [], []
        seen_paths, seen_frames, seen_tracks, seen_chunks = set(), set(), set(), set()
        # Validate all manifest chunks so a split-specific loader also catches
        # frame reuse across the other split assignments.
        for entry in self.manifest["chunks"]:
            if entry["chunk_id"] in seen_chunks:
                raise DatasetError(f"Repeated chunk ID: {entry['chunk_id']}")
            seen_chunks.add(entry["chunk_id"])
            if entry["split"] not in SPLIT_NAMES:
                raise DatasetError(f"Invalid split assignment: {entry['split']}")
            if self.manifest.get("distribution") == "participants" and entry["split"] == "test":
                raise DatasetError("Participant manifest must not include test chunks.")
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
            if version == 2:
                video, chunk = entry["video_id"], entry["chunk_number"]
                if (type(video) is not int or type(chunk) is not int
                        or not 1 <= video <= 99 or not 1 <= chunk <= 99
                        or video_registry.get(entry["recording_id"]) != video
                        or entry.get("wave") not in self.manifest["waves"]
                        or entry["path"] != f"{entry['split']}/{entry['chunk_id']}"
                        or entry["chunk_id"] != f"{video:02d}{chunk:02d}"
                        or numbers != list(range(1, len(frames) + 1))):
                    raise DatasetError(f"Invalid chunk identity or local frame numbers: {entry['chunk_id']}")
                source_numbers = [f["source_number"] for f in frames]
                if (any(type(n) is not int or n < 0 for n in source_numbers)
                        or source_numbers != list(range(source_numbers[0], source_numbers[0] + len(frames)))):
                    raise DatasetError(f"Invalid source frame numbers: {entry['chunk_id']}")
                for frame in frames:
                    expected = photo_name(video, chunk, entry["split"], frame["number"], Path(frame["image"]).suffix)
                    if frame["image"] != expected:
                        raise DatasetError(f"Image name does not match manifest: {frame['image']}")
            for frame in frames:
                key = ((entry["recording_id"], frame["source_number"]) if version == 2 else
                       (entry["source"], entry["recording_id"], frame["number"]))
                if key in seen_frames:
                    raise DatasetError(f"Frame reused across chunks: {key}")
                seen_frames.add(key)
            track_ids = entry["track_ids"]
            if (not track_ids or not all(isinstance(t, int) and not isinstance(t, bool) and t >= 0 for t in track_ids)
                    or len(track_ids) != len(set(track_ids))):
                raise DatasetError(f"Invalid track IDs: {entry['chunk_id']}")
            if version == 2:
                expected_ids = [(entry["video_id"] * 100 + entry["chunk_number"]) * 1000 + local
                                for local in range(1, len(track_ids) + 1)]
                if len(track_ids) > 999 or track_ids != expected_ids or seen_tracks.intersection(track_ids):
                    raise DatasetError(f"Fish IDs must be globally distinct per chunk: {entry['chunk_id']}")
                seen_tracks.update(track_ids)
            if entry["split"] != split:
                continue
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

    def prediction_input(self, index):
        """Return ten ordered image paths and alignment metadata, without labels.

        The competition interface is fixed at 10 input and 10 output frames.
        The configurable trajectory loader remains available for research.
        """
        if self.observed != 10 or self.future != 10:
            raise DatasetError("Prediction inputs require observed=10 and future=10.")
        chunk_index, start = self.index[index]
        entry = self.chunks[chunk_index]["entry"]
        frames = entry["frames"]
        return {
            "chunk_id": entry["chunk_id"],
            "observed_frames": [f["number"] for f in frames[start:start + 10]],
            "future_frames": [f["number"] for f in frames[start + 10:start + 20]],
            "track_ids": entry["track_ids"][:],
            "input_images": [str(child_path(self.root / entry["path"] / "images", f["image"]))
                             for f in frames[start:start + 10]],
        }

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
                "input_images": [str(child_path(self.root / entry["path"] / "images", f["image"]))
                                 for f in entry["frames"][start:middle]],
                "observed": observed, "future": future}

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


def predict_sample(dataset, index, predictor):
    """Run a team's predictor on exactly ten images and validate its output.

    predictor(images) receives ten encoded image byte strings in time order.
    Decode/preprocess these inside the team's adapter as needed. Return Python
    lists shaped [fish_count][10][2], in prediction_input(index)['track_ids']
    order, in original-image pixels. No centers or future images reach predictor.
    This utility supplies the interface and validation, not a trained model.
    """
    from prediction_format import validate_centers

    inputs = dataset.prediction_input(index)
    images = [Path(path).read_bytes() for path in inputs.pop("input_images")]
    centers = validate_centers(predictor(images), len(inputs["track_ids"]))
    return dict(inputs, predicted_centers=centers)


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
