#!/usr/bin/env python3
"""Create versioned competition releases with VVCCSFFF photos and chunk fish IDs."""
import argparse
import copy
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

from dataset_common import DatasetError, SPLIT_NAMES, child_path, load_json, photo_name
from prediction_loader import PredictionDataset

FILENAME_FORMAT = "VVCCSFFF"


def _validate_dataset(root):
    # Validate every split before copying anything, including manifest frame reuse.
    manifest = load_json(Path(root) / "manifest.json")
    splits = ("train", "validate") if manifest.get("distribution") == "participants" else SPLIT_NAMES
    for split in splits:
        PredictionDataset(root, split)
    return manifest


def _copy_chunk(root, entry, target):
    source = child_path(root, entry["path"])
    folder = target / entry["path"]
    (folder / "images").mkdir(parents=True)
    shutil.copy2(source / "annotations.json", folder / "annotations.json")
    for frame in entry["frames"]:
        shutil.copy2(child_path(source / "images", frame["image"]), folder / "images" / frame["image"])


def _write_manifest(root, manifest):
    manifest["split_counts"] = {
        split: {"chunks": sum(c["split"] == split for c in manifest["chunks"]),
                "frames": sum(c["frame_count"] for c in manifest["chunks"] if c["split"] == split),
                "prediction_windows_10_10_stride_1": sum(max(0, c["frame_count"] - 19)
                                                       for c in manifest["chunks"] if c["split"] == split)}
        for split in SPLIT_NAMES if manifest["distribution"] == "private" or split != "test"
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (root / "README.txt").write_text(
        f"Competition release: {manifest['release_id']} ({manifest['distribution']}).\n"
        "Photos: VVCCSFFF.ext; video/chunk use two digits, split is 1=train, 2=validate,\n"
        "3=test, and frame is 001–999 within the chunk. Image bytes are preserved.\n"
        "One annotations.json per chunk; manifest.json is the index. Coordinates are pixels.\n"
        "Fish IDs are constant inside a chunk and never reused in another chunk.\n"
        "Use the fixed splits. Model input: 10 images; output: next 10 centers per fish.\n"
        + ("Private organizer data includes withheld test images and labels.\n"
           if manifest["distribution"] == "private" else "Participant package: training and validation only. Test data is withheld.\n"),
        encoding="utf-8")


def _stage_output(output, writer):
    output = Path(output).resolve()
    if output.exists():
        raise DatasetError(f"Output already exists; choose a new directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        staged = Path(temporary) / "dataset"
        staged.mkdir()
        result = writer(staged)
        _validate_dataset(staged)
        if output.exists():
            raise DatasetError(f"Output appeared during export: {output}")
        staged.rename(output)
    return result


def build_release(dataset_dir, output_dir, release_id, previous=None):
    """Convert a prepared wave and optionally append it to a private release.

    Inputs remain unchanged. Earlier chunk files, IDs, names and splits are copied
    unchanged. New waves must contain disjoint source frames. Recording names
    must identify the same full video across waves and be globally unambiguous.
    """
    if not isinstance(release_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", release_id):
        raise DatasetError("release_id must be a nonempty name using letters, numbers, _, . or -.")
    root = Path(dataset_dir).resolve()
    incoming = _validate_dataset(root)
    if incoming.get("schema_version") != 1:
        raise DatasetError("New waves must be prepared schema-version-1 datasets from split_dataset.py.")
    if not incoming["chunks"]:
        raise DatasetError("The new wave has no chunks.")
    prior = None
    if previous is not None:
        prior = _validate_dataset(previous)
        if prior.get("schema_version") != 2 or prior.get("distribution") != "private":
            raise DatasetError("--previous must be a private version-2 release.")
        if release_id in prior["waves"]:
            raise DatasetError(f"Release name already used: {release_id}")
    manifest = copy.deepcopy(prior) if prior else {
        "schema_version": 2, "fps": 30, "filename_format": FILENAME_FORMAT,
        "track_id_scope": "globally_unique_per_chunk", "distribution": "private",
        "videos": [], "waves": [], "chunks": []}
    manifest["release_id"] = release_id
    manifest["waves"].append(release_id)
    videos = {v["recording_id"]: v["video_id"] for v in manifest["videos"]}
    next_chunk = {}
    used_frames = set()
    for entry in manifest["chunks"]:
        video = entry["video_id"]
        next_chunk[video] = max(next_chunk.get(video, 0), entry["chunk_number"])
        used_frames.update((entry["recording_id"], f["source_number"]) for f in entry["frames"])
    # A recording name cannot refer to different source folders in the same wave.
    sources = {}
    plans = []
    for entry in incoming["chunks"]:
        recording = entry["recording_id"]
        if recording in sources and sources[recording] != entry["source"]:
            raise DatasetError(f"Ambiguous recording name across inputs: {recording}")
        sources[recording] = entry["source"]
        if recording not in videos:
            videos[recording] = max(videos.values(), default=0) + 1
            manifest["videos"].append({"recording_id": recording, "video_id": videos[recording]})
        video = videos[recording]
        chunk = next_chunk.get(video, 0) + 1
        next_chunk[video] = chunk
        if len(entry["frames"]) > 999:
            raise DatasetError("A chunk exceeds 999 frames; divide it explicitly before release.")
        if len(entry["track_ids"]) > 999:
            raise DatasetError("A chunk exceeds 999 fish IDs.")
        mapping = {old: (video * 100 + chunk) * 1000 + local
                   for local, old in enumerate(sorted(entry["track_ids"]), 1)}
        frames = []
        for number, frame in enumerate(entry["frames"], 1):
            key = (recording, frame["number"])
            if key in used_frames:
                raise DatasetError(f"Source frame already released: {key}; earlier releases are immutable.")
            used_frames.add(key)
            image = photo_name(video, chunk, entry["split"], number, Path(frame["image"]).suffix)
            frames.append({"number": number, "image": image, "source_number": frame["number"],
                           "source_image": frame["image"]})
        chunk_id = f"{video:02d}{chunk:02d}"
        new_entry = {"chunk_id": chunk_id, "recording_id": recording, "source": f"video_{video:02d}",
                     "video_id": video, "chunk_number": chunk, "split": entry["split"],
                     "path": f"{entry['split']}/{chunk_id}", "frame_count": len(frames),
                     "track_ids": list(mapping.values()), "frames": frames, "wave": release_id,
                     "source_track_id_map": {str(old): new for old, new in mapping.items()}}
        manifest["chunks"].append(new_entry)
        plans.append((entry, new_entry, mapping))

    def write(staged):
        if prior:
            for entry in prior["chunks"]:
                _copy_chunk(previous, entry, staged)
        for original, entry, mapping in plans:
            source = child_path(root, original["path"])
            folder = staged / entry["path"]
            (folder / "images").mkdir(parents=True)
            names = {}
            for frame in entry["frames"]:
                names[frame["source_image"]] = frame["image"]
                shutil.copy2(child_path(source / "images", frame["source_image"]), folder / "images" / frame["image"])
            records = load_json(source / "annotations.json")
            for record in records:
                record["image"] = names[record["image"]]
                record["track_id"] = mapping[record["track_id"]]
            (folder / "annotations.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        _write_manifest(staged, manifest)
        return manifest

    return _stage_output(output_dir, write)


def export_participant_package(private_dir, output_dir):
    """Copy only train/validate chunks; omit all test files and test metadata."""
    manifest = _validate_dataset(private_dir)
    if manifest.get("schema_version") != 2 or manifest.get("distribution") != "private":
        raise DatasetError("Participant export requires a private version-2 release.")
    manifest["distribution"] = "participants"
    manifest["chunks"] = [c for c in manifest["chunks"] if c["split"] != "test"]
    if not manifest["chunks"]:
        raise DatasetError("No training or validation chunks to distribute.")
    for entry in manifest["chunks"]:
        entry.pop("source_track_id_map", None)
        for frame in entry["frames"]:
            frame.pop("source_image", None)
    used = {c["video_id"] for c in manifest["chunks"]}
    manifest["videos"] = [v for v in manifest["videos"] if v["video_id"] in used]

    def write(staged):
        for entry in manifest["chunks"]:
            _copy_chunk(private_dir, entry, staged)
        _write_manifest(staged, manifest)
        return manifest

    return _stage_output(output_dir, write)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Create a private named release from a prepared wave")
    build.add_argument("dataset", type=Path)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--release-id", required=True)
    build.add_argument("--previous", type=Path)
    public = subparsers.add_parser("participants", help="Export training/validation only")
    public.add_argument("dataset", type=Path)
    public.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "build":
            result = build_release(args.dataset, args.output, args.release_id, args.previous)
        else:
            result = export_participant_package(args.dataset, args.output)
        print(f"{result['release_id']}: {len(result['chunks'])} chunks ({result['distribution']}) -> {args.output}")
    except (DatasetError, OSError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
