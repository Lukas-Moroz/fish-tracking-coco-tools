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

from dataset_common import (DatasetError, SPLIT_NAMES, child_path, load_json,
                            photo_name, read_frames, stable_runs)
from prediction_loader import PredictionDataset

FILENAME_FORMAT = "VVCCSFFF"


def _same_file_bytes(first, second):
    if Path(first).stat().st_size != Path(second).stat().st_size:
        return False
    with Path(first).open("rb") as left, Path(second).open("rb") as right:
        while True:
            block = left.read(8 * 1024 * 1024)
            if block != right.read(8 * 1024 * 1024):
                return False
            if not block:
                return True


def _validate_dataset(root):
    # Validate every split before copying anything, including manifest frame reuse.
    manifest = load_json(Path(root) / "manifest.json")
    if manifest.get("distribution") == "public":
        splits = ("public",)
    else:
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
    sections = (("public",) if manifest["distribution"] == "public" else
                SPLIT_NAMES if manifest["distribution"] == "private" else ("train", "validate"))
    manifest["split_counts"] = {
        split: {"chunks": sum(c["split"] == split for c in manifest["chunks"]),
                "frames": sum(c["frame_count"] for c in manifest["chunks"] if c["split"] == split),
                "prediction_windows_10_10_stride_1": sum(max(0, c["frame_count"] - 19)
                                                       for c in manifest["chunks"] if c["split"] == split)}
        for split in sections
    }
    extras = [c for c in manifest["chunks"] if c["split"] == "extra"]
    if extras:
        manifest["extra_counts"] = {"chunks": len(extras), "frames": sum(c["frame_count"] for c in extras)}
    else:
        manifest.pop("extra_counts", None)
    reserved = [c for c in manifest["chunks"] if c["split"] == "extra_test"]
    if reserved:
        manifest["extra_test_counts"] = {
            "chunks": len(reserved), "frames": sum(c["frame_count"] for c in reserved),
            "prediction_windows_10_10_stride_1": sum(max(0, c["frame_count"] - 19) for c in reserved)}
    else:
        manifest.pop("extra_test_counts", None)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    public_pool = manifest["distribution"] == "public"
    (root / "README.txt").write_text(
        f"Competition release: {manifest['release_id']} ({manifest['distribution']}).\n"
        "Photos: VVCCSFFF.ext; video/chunk use two digits, split is 1=train, 2=validate,\n"
        "3=private test, 4=public/additional data; frame is 001–999 within the chunk.\n"
        "Image bytes and original pixel coordinates are preserved.\n"
        "One annotations.json per chunk; manifest.json is the index. Coordinates are pixels.\n"
        "Fish IDs are constant inside a chunk and never reused in another chunk.\n"
        "Model input: 10 images; output: next 10 centers per fish.\n"
        + ("All supplied data uses code4 in public/. Teams choose their own training/validation usage.\n"
           "Short chunks remain labeled data; only chunks of at least 20 frames supply 10+10 windows.\n"
           "Use PredictionDataset(package, 'public'). Private test images and labels are withheld.\n"
           if public_pool else
           "Use the fixed official splits. Additional code4 frames, when present, live in extra/.\n"
           + ("Private organizer data includes withheld test images and labels.\n"
              if manifest["distribution"] == "private" else "Participant package: all private holdouts are withheld.\n")),
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
    manifest["chunks"] = [c for c in manifest["chunks"] if c["split"] not in ("test", "extra_test")]
    manifest.pop("extra_test_counts", None)
    manifest.pop("extra_test_policy", None)
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


def add_extra_release(dataset_dir, output_dir, release_id, previous, excluded_recordings=()):
    """Preserve a private release and add all unassigned labeled source frames.

    Additional frames use section 'extra' / code4. They retain every stable run,
    including short runs, without entering the official train/validate/test counts.
    Exclusions must name organizer-reviewed duplicate or invalid recordings.
    """
    if not isinstance(release_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", release_id):
        raise DatasetError("Invalid release_id.")
    prior = _validate_dataset(previous)
    if prior.get("schema_version") != 2 or prior.get("distribution") != "private":
        raise DatasetError("--previous must be a private version-2 release.")
    if release_id in prior["waves"]:
        raise DatasetError(f"Release name already used: {release_id}")
    frames, _ = read_frames(dataset_dir)
    excluded = set(excluded_recordings)
    available = {f["sequence"] for f in frames}
    if excluded - available:
        raise DatasetError(f"Unknown excluded recordings: {sorted(excluded - available)}")
    accepted = [f for f in frames if f["sequence"] not in excluded]
    if any(f["reason"] for f in accepted):
        raise DatasetError("Additional data contains missing, ambiguous, or nonnumeric fish identities.")
    source_index = {(f["sequence"], f["number"]): f for f in accepted}
    for entry in prior["chunks"]:
        overlap = [(frame, source_index[(entry["recording_id"], frame["source_number"])])
                   for frame in entry["frames"]
                   if (entry["recording_id"], frame["source_number"]) in source_index]
        if not overlap:
            continue
        mapping = {int(old): new for old, new in entry.get("source_track_id_map", {}).items()}
        folder = child_path(previous, entry["path"])
        by_image = {}
        for record in load_json(folder / "annotations.json"):
            by_image.setdefault(record["image"], []).append(record)
        for frame, source_frame in overlap:
            if any(r["track_id"] not in mapping for r in source_frame["records"]):
                raise DatasetError(f"Source identity changed for previously released frame: {source_frame['name']}")
            expected = [dict(r, image=frame["image"], track_id=mapping[r["track_id"]])
                        for r in source_frame["records"]]
            if (sorted(expected, key=lambda r: r["track_id"])
                    != sorted(by_image.get(frame["image"], []), key=lambda r: r["track_id"])
                    or not _same_file_bytes(source_frame["source"], folder / "images" / frame["image"])):
                raise DatasetError(f"Source image or labels changed for previously released frame: {source_frame['name']}")
    used = {(c["recording_id"], f["source_number"]) for c in prior["chunks"] for f in c["frames"]}
    remaining = [f for f in accepted if (f["sequence"], f["number"]) not in used]
    if not remaining:
        raise DatasetError("No unassigned source frames remain.")
    manifest = copy.deepcopy(prior)
    manifest["release_id"] = release_id
    manifest["waves"].append(release_id)
    manifest["extra_section"] = {"name": "extra", "filename_code": 4,
                                  "purpose": "additional labeled data outside official splits"}
    manifest["extra_excluded_recordings"] = sorted(excluded)
    videos = {v["recording_id"]: v["video_id"] for v in manifest["videos"]}
    next_chunk = {}
    for c in manifest["chunks"]:
        next_chunk[c["video_id"]] = max(next_chunk.get(c["video_id"], 0), c["chunk_number"])
    # Honor the existing registry order; new recordings are appended deterministically.
    order = [v["recording_id"] for v in manifest["videos"]]
    order += sorted({f["sequence"] for f in remaining} - set(order))
    plans = []
    for recording in order:
        selected = [f for f in remaining if f["sequence"] == recording]
        if not selected:
            continue
        if recording not in videos:
            videos[recording] = max(videos.values(), default=0) + 1
            manifest["videos"].append({"recording_id": recording, "video_id": videos[recording]})
        video = videos[recording]
        for run in stable_runs(selected):
            for start in range(0, len(run), 999):
                part = run[start:start + 999]
                number = next_chunk.get(video, 0) + 1
                next_chunk[video] = number
                ids = sorted(part[0]["signature"])
                if len(ids) > 999:
                    raise DatasetError("A chunk exceeds 999 fish IDs.")
                mapping = {old: (video * 100 + number) * 1000 + local
                           for local, old in enumerate(ids, 1)}
                output_frames = [{"number": local,
                                  "image": photo_name(video, number, "extra", local, Path(f["name"]).suffix),
                                  "source_number": f["number"], "source_image": f["name"]}
                                 for local, f in enumerate(part, 1)]
                chunk_id = f"{video:02d}{number:02d}"
                entry = {"chunk_id": chunk_id, "recording_id": recording, "source": f"video_{video:02d}",
                         "video_id": video, "chunk_number": number, "split": "extra",
                         "path": f"extra/{chunk_id}", "frame_count": len(part),
                         "track_ids": list(mapping.values()), "frames": output_frames, "wave": release_id,
                         "source_track_id_map": {str(old): new for old, new in mapping.items()}}
                manifest["chunks"].append(entry)
                plans.append((part, entry, mapping))

    def write(staged):
        for entry in prior["chunks"]:
            _copy_chunk(previous, entry, staged)
        for part, entry, mapping in plans:
            folder = staged / entry["path"]
            (folder / "images").mkdir(parents=True)
            records = []
            for source_frame, target_frame in zip(part, entry["frames"]):
                shutil.copy2(source_frame["source"], folder / "images" / target_frame["image"])
                records.extend(dict(r, image=target_frame["image"], track_id=mapping[r["track_id"]])
                               for r in source_frame["records"])
            (folder / "annotations.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        _write_manifest(staged, manifest)
        return manifest

    return _stage_output(output_dir, write)


def reserve_extra_chunks(private_dir, output_dir, release_id, chunk_ids):
    """Reserve selected whole supplemental chunks without altering official splits.

    Reserved supplemental images use section 'extra_test' and filename code3.
    Their counts remain separate from the official test benchmark.
    """
    if not isinstance(release_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", release_id):
        raise DatasetError("Invalid release_id.")
    manifest = _validate_dataset(private_dir)
    if manifest.get("schema_version") != 2 or manifest.get("distribution") != "private":
        raise DatasetError("Supplemental reserve requires a private version-2 release.")
    if release_id in manifest["waves"]:
        raise DatasetError(f"Release name already used: {release_id}")
    selected_list = list(chunk_ids)
    if (not selected_list or any(not isinstance(c, str) for c in selected_list)
            or len(set(selected_list)) != len(selected_list)):
        raise DatasetError("Select distinct supplemental chunk IDs.")
    selected = set(selected_list)
    eligible = {c["chunk_id"] for c in manifest["chunks"] if c["split"] == "extra"}
    if selected - eligible:
        raise DatasetError(f"Only existing extra chunks can be reserved: {sorted(selected - eligible)}")
    originals = copy.deepcopy(manifest["chunks"])
    source_release = manifest["release_id"]
    source_extra_frames = sum(c["frame_count"] for c in originals if c["split"] == "extra")
    reserved_frames = sum(c["frame_count"] for c in originals if c["chunk_id"] in selected)
    manifest["release_id"] = release_id
    manifest["waves"].append(release_id)
    manifest["extra_test_policy"] = {
        "source_release": source_release,
        "reserved_chunk_ids": sorted(selected), "reserved_frames": reserved_frames,
        "source_extra_frames": source_extra_frames,
        "fraction_of_source_extra_frames": reserved_frames / source_extra_frames,
        "purpose": "additional supplemental holdout; official test unchanged"}
    for entry in manifest["chunks"]:
        if entry["chunk_id"] not in selected:
            continue
        entry["split"] = "extra_test"
        entry["path"] = f"extra_test/{entry['chunk_id']}"
        for frame in entry["frames"]:
            frame["image"] = photo_name(entry["video_id"], entry["chunk_number"], "extra_test",
                                        frame["number"], Path(frame["image"]).suffix)

    def write(staged):
        for original, entry in zip(originals, manifest["chunks"]):
            if entry["chunk_id"] not in selected:
                _copy_chunk(private_dir, original, staged)
                continue
            source = child_path(private_dir, original["path"])
            folder = staged / entry["path"]
            (folder / "images").mkdir(parents=True)
            names = {}
            for before, after in zip(original["frames"], entry["frames"]):
                names[before["image"]] = after["image"]
                shutil.copy2(child_path(source / "images", before["image"]), folder / "images" / after["image"])
            records = load_json(source / "annotations.json")
            for record in records:
                record["image"] = names[record["image"]]
            (folder / "annotations.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        _write_manifest(staged, manifest)
        return manifest

    return _stage_output(output_dir, write)


def export_public_pool(private_dir, output_dir, sections=None):
    """Export non-reserved frames to code4; optionally choose source sections."""
    manifest = _validate_dataset(private_dir)
    if manifest.get("schema_version") != 2 or manifest.get("distribution") != "private":
        raise DatasetError("Public export requires a private version-2 release.")
    chosen_sections = {"train", "validate", "extra"} if sections is None else set(sections)
    if not chosen_sections or chosen_sections - {"train", "validate", "extra"}:
        raise DatasetError("Public export sections may be train, validate, or extra; test data is withheld.")
    originals = [copy.deepcopy(c) for c in manifest["chunks"] if c["split"] in chosen_sections]
    if not originals:
        raise DatasetError("No non-test chunks to distribute.")
    manifest["distribution"] = "public"
    for key in ("target_ratios", "requested_ratios", "min_chunk_frames", "extra_section",
                "extra_excluded_recordings", "extra_counts", "extra_test_counts", "extra_test_policy"):
        manifest.pop(key, None)
    manifest["split_method"] = "single public pool; teams choose usage; private test withheld"
    manifest["chunks"] = []
    for original in originals:
        entry = copy.deepcopy(original)
        entry["split"] = "public"
        entry["path"] = f"public/{entry['chunk_id']}"
        entry.pop("source_track_id_map", None)
        for frame in entry["frames"]:
            frame["image"] = photo_name(entry["video_id"], entry["chunk_number"], "public",
                                        frame["number"], Path(frame["image"]).suffix)
            frame.pop("source_image", None)
        manifest["chunks"].append(entry)
    visible = {c["video_id"] for c in manifest["chunks"]}
    manifest["videos"] = [v for v in manifest["videos"] if v["video_id"] in visible]

    def write(staged):
        for original, entry in zip(originals, manifest["chunks"]):
            source = child_path(private_dir, original["path"])
            folder = staged / entry["path"]
            (folder / "images").mkdir(parents=True)
            names = {}
            for before, after in zip(original["frames"], entry["frames"]):
                names[before["image"]] = after["image"]
                shutil.copy2(child_path(source / "images", before["image"]), folder / "images" / after["image"])
            records = load_json(source / "annotations.json")
            for record in records:
                record["image"] = names[record["image"]]
            (folder / "annotations.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
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
    public = subparsers.add_parser("participants", help="Export training, validation, and released supplements")
    public.add_argument("dataset", type=Path)
    public.add_argument("--output", required=True, type=Path)
    extra = subparsers.add_parser("extra", help="Add previously unassigned frames with code4")
    extra.add_argument("dataset", type=Path)
    extra.add_argument("--previous", required=True, type=Path)
    extra.add_argument("--release-id", required=True)
    extra.add_argument("--output", required=True, type=Path)
    extra.add_argument("--exclude-recording", action="append", default=[])
    pool = subparsers.add_parser("public", help="Export one code4 public pool, withholding test")
    pool.add_argument("dataset", type=Path)
    pool.add_argument("--output", required=True, type=Path)
    pool.add_argument("--section", choices=("train", "validate", "extra"), action="append",
                      help="Optionally export only selected sections; use --section extra for a supplemental-only pool")
    reserve = subparsers.add_parser("reserve-extra", help="Reserve selected supplemental chunks as private code3 data")
    reserve.add_argument("dataset", type=Path)
    reserve.add_argument("--output", required=True, type=Path)
    reserve.add_argument("--release-id", required=True)
    reserve.add_argument("--chunk-id", action="append", required=True)
    args = parser.parse_args()
    try:
        if args.command == "build":
            result = build_release(args.dataset, args.output, args.release_id, args.previous)
        elif args.command == "extra":
            result = add_extra_release(args.dataset, args.output, args.release_id, args.previous,
                                       args.exclude_recording)
        elif args.command == "public":
            result = export_public_pool(args.dataset, args.output, args.section)
        elif args.command == "reserve-extra":
            result = reserve_extra_chunks(args.dataset, args.output, args.release_id, args.chunk_id)
        else:
            result = export_participant_package(args.dataset, args.output)
        print(f"{result['release_id']}: {len(result['chunks'])} chunks ({result['distribution']}) -> {args.output}")
    except (DatasetError, OSError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
