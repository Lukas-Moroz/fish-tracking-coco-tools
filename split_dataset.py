#!/usr/bin/env python3
"""Export maximal stable fish sequences, assigning whole chunks to fixed splits."""
import argparse
import json
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path

from dataset_common import DatasetError, SPLIT_NAMES, positive_integer, read_frames, stable_runs

def assign_splits(lengths, ratios=(70, 15, 15)):
    """Find chronological whole-chunk partitions nearest the requested frame ratios.

    Each split gets at least one chunk. The squared frame-count error is minimized;
    ties choose earlier boundaries. Input order is retained, never shuffled.
    """
    if len(lengths) < 3:
        raise DatasetError("At least three qualifying chunks are needed to populate all splits. "
                              "Add footage or explicitly divide a long source shot before exporting.")
    if len(ratios) != 3 or not all(math.isfinite(r) and r > 0 for r in ratios):
        raise DatasetError("Provide three positive finite split ratios.")
    prefix = [0]
    for length in lengths:
        prefix.append(prefix[-1] + length)
    targets = [prefix[-1] * r / sum(ratios) for r in ratios]
    # For a fixed validation/test boundary j, the best earlier boundary i is
    # nearest to this ideal cumulative train length. Find it with binary search.
    from bisect import bisect_left
    best = None
    for j in range(2, len(lengths)):
        ideal = (targets[0] + prefix[j] - targets[1]) / 2
        position = bisect_left(prefix, ideal, 1, j)
        for i in {max(1, min(j - 1, position)), max(1, min(j - 1, position - 1))}:
            counts = (prefix[i], prefix[j] - prefix[i], prefix[-1] - prefix[j])
            candidate = (sum((count - target) ** 2 for count, target in zip(counts, targets)), i, j)
            if best is None or candidate < best:
                best = candidate
    _, i, j = best
    return ["train"] * i + ["validate"] * (j - i) + ["test"] * (len(lengths) - j)


def plan_chunks(dataset_dirs, min_frames=30, ratios=(70, 15, 15)):
    positive_integer(min_frames, "Minimum chunk length")
    if min_frames < 20:
        raise DatasetError("Minimum chunk length must be at least 20 frames.")
    chunks, sources = [], []
    resolved = [Path(p).resolve() for p in dataset_dirs]
    if len(resolved) != len(set(resolved)):
        raise DatasetError("The same source directory was provided more than once.")
    used_photos = set()
    for source_index, source in enumerate(resolved, 1):
        frames, _ = read_frames(source)
        source_id = f"source_{source_index:02d}"
        for frame in frames:
            photo = frame["source"].resolve()
            if photo in used_photos:
                raise DatasetError(f"Source image provided in multiple inputs: {photo}")
            used_photos.add(photo)
        runs = stable_runs(frames)
        source_chunks = [run for run in runs if len(run) >= min_frames]
        sources.append({"source": source_id, "source_frames": len(frames),
                        "retained_frames": sum(len(run) for run in source_chunks),
                        "omitted_runs": [{"sequence": run[0]["sequence"], "start": run[0]["number"],
                                          "end": run[-1]["number"], "frame_count": len(run)}
                                         for run in runs if len(run) < min_frames],
                        "ambiguous_frames": [{"image": f["name"], "reason": f["reason"]}
                                             for f in frames if f["reason"]]})
        for run in source_chunks:
            first, last = run[0], run[-1]
            safe_sequence = re.sub(r"[^A-Za-z0-9_.-]", "_", first["sequence"])
            chunk_id = (f"chunk_{len(chunks)+1:03d}_source_{source_index:02d}_"
                        f"{safe_sequence}_{first['number']:04d}-{last['number']:04d}")
            chunks.append({"chunk_id": chunk_id, "recording_id": first["sequence"],
                           "source": source_id, "frames": run})
    assignments = assign_splits([len(chunk["frames"]) for chunk in chunks], ratios)
    for chunk, split in zip(chunks, assignments):
        chunk["split"] = split
    return chunks, sources


def export_chunks(dataset_dirs, output_dir, min_frames=30, ratios=(70, 15, 15)):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise DatasetError(f"Output already exists; choose a new directory: {output_dir}")
    chunks, sources = plan_chunks(dataset_dirs, min_frames, ratios)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.", dir=output_dir.parent) as temporary:
        staged = Path(temporary) / "dataset"
        manifest = write_export(chunks, sources, staged, min_frames, ratios)
        if output_dir.exists():
            raise DatasetError(f"Output appeared during export: {output_dir}")
        staged.rename(output_dir)
    return manifest


def write_export(chunks, sources, output_dir, min_frames, ratios):
    """Write an already validated plan into a temporary directory."""
    manifest = {"schema_version": 1, "fps": 30, "min_chunk_frames": min_frames,
                "detection_class": {"id": 0, "name": "fish"},
                "requested_ratios": dict(zip(SPLIT_NAMES, ratios)),
                "split_method": "chronological whole chunks in supplied source order",
                "sources": sources, "chunks": [], "split_counts": {}}
    output_dir.mkdir(parents=True)
    for split in SPLIT_NAMES:
        (output_dir / split).mkdir()
    for chunk in chunks:
        relative_dir = Path(chunk["split"]) / chunk["chunk_id"]
        folder = output_dir / relative_dir
        (folder / "images").mkdir(parents=True)
        records = []
        for frame in chunk["frames"]:
            shutil.copy2(frame["source"], folder / "images" / frame["name"])
            for record in frame["records"]:
                records.append({"image": record["image"], "class_id": 0,
                                "track_id": record["track_id"],
                                "bbox": record["bbox"], "center": record["center"]})
        (folder / "annotations.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        frames = chunk["frames"]
        manifest["chunks"].append({"chunk_id": chunk["chunk_id"], "recording_id": chunk["recording_id"],
                                   "source": chunk["source"], "split": chunk["split"],
                                   "path": relative_dir.as_posix(), "frame_count": len(frames),
                                   "track_ids": sorted(frames[0]["signature"]),
                                   "frames": [{"number": f["number"], "image": f["name"]} for f in frames]})
    total = sum(len(chunk["frames"]) for chunk in chunks)
    for split in SPLIT_NAMES:
        selected = [chunk for chunk in chunks if chunk["split"] == split]
        count = sum(len(chunk["frames"]) for chunk in selected)
        manifest["split_counts"][split] = {"chunks": len(selected), "frames": count,
                                          "fraction": count / total,
                                          "prediction_windows_10_10_stride_1": sum(len(c["frames"]) - 19 for c in selected)}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output_dir / "README.txt").write_text(
        "Stable fish dataset, sampled at 30 FPS.\n"
        "Splits are fixed in manifest.json. Whole chunks are assigned in chronological order\n"
        "within each source; supplied source-folder order is preserved. Same-recording chunks\n"
        "may be in different splits. Requested ratios are approximate.\n"
        "A chunk requires consecutive frame numbers and identical annotated fish IDs throughout.\n"
        "Any missing/added fish box or ambiguous identity breaks the chunk.\n"
        "Annotations: image, class_id=0 (fish), track_id, bbox, center. Coordinates are pixels.\n"
        "Track identity is scoped to the source recording; IDs can repeat in other recordings.\n"
        "Centers are bounding-box midpoints. Source files are preserved.\n"
        "Only stable chunks meeting the minimum length are included in this package.\n"
        "Use prediction_loader.py within one split; it never crosses chunk boundaries.\n",
        encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dirs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument("--ratios", nargs=3, type=float, default=(70, 15, 15), metavar=("TRAIN", "VALIDATE", "TEST"))
    args = parser.parse_args()
    try:
        manifest = export_chunks(args.dataset_dirs, args.output, args.min_frames, args.ratios)
    except (DatasetError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Exported {len(manifest['chunks'])} stable chunks to {args.output}")
    for split, counts in manifest["split_counts"].items():
        print(f"  {split}: {counts['frames']} frames ({counts['fraction']:.1%}), "
              f"{counts['chunks']} chunks, {counts['prediction_windows_10_10_stride_1']} prediction windows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
