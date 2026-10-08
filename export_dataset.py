#!/usr/bin/env python3
"""Export a Roboflow COCO dataset as a zip of images plus cleaned fish-tracking JSON.

Every split folder (a directory containing _annotations.coco.json, e.g. train/,
valid/, test/) is converted with convert_annotations.convert(). Images are
renamed to their original filename (images[].extra.name) so they match the
"image" field in the cleaned JSON.

Zip layout:
    <export_name>/
        train/
            fish_tracking_annotations.json
            images/fish_sample2_mp4-0000.jpg
            ...
        valid/...
        test/...

Usage:
    python export_dataset.py dataset_dir output.zip
"""

import json
import sys
import zipfile
from pathlib import Path

from convert_annotations import ConversionError, convert, load_coco

COCO_FILENAME = "_annotations.coco.json"
OUTPUT_JSON_NAME = "fish_tracking_annotations.json"


def find_splits(dataset_dir):
    """Return split directories that contain a COCO annotation file."""
    if (dataset_dir / COCO_FILENAME).is_file():
        return [dataset_dir]
    splits = sorted(p.parent for p in dataset_dir.glob(f"*/{COCO_FILENAME}"))
    if not splits:
        raise ConversionError(f"No {COCO_FILENAME} found in {dataset_dir} or its subfolders.")
    return splits


def plan_images(data, split_dir):
    """Return a sorted list of (source_path, output_name) for every image in the split."""
    planned = {}
    for img in data["images"]:
        extra = img.get("extra") or {}
        output_name = extra.get("name") or img.get("file_name")
        source = split_dir / img["file_name"]
        if not source.is_file():
            raise ConversionError(f"Image file not found: {source}")
        if output_name in planned:
            raise ConversionError(
                f"Two images in {split_dir.name}/ would both be exported as '{output_name}' "
                f"({planned[output_name].name} and {source.name})."
            )
        planned[output_name] = source
    return sorted((src, name) for name, src in planned.items())


def export(dataset_dir, output_zip):
    splits = find_splits(dataset_dir)
    export_name = output_zip.stem
    summary = []

    # Do all validation before touching the output file.
    prepared = []
    for split_dir in splits:
        data = load_coco(split_dir / COCO_FILENAME)
        records, skipped = convert(data)
        images = plan_images(data, split_dir)
        prefix = export_name if split_dir == dataset_dir else f"{export_name}/{split_dir.name}"
        prepared.append((prefix, records, images))
        summary.append((split_dir.name, len(data["annotations"]), len(records), skipped, len(images)))

    with zipfile.ZipFile(output_zip, "w") as zf:
        for prefix, records, images in prepared:
            zf.writestr(
                f"{prefix}/{OUTPUT_JSON_NAME}",
                json.dumps(records, indent=2) + "\n",
                compress_type=zipfile.ZIP_DEFLATED,
            )
            for source, output_name in images:
                # JPEGs are already compressed; store them as-is.
                zf.write(source, f"{prefix}/images/{output_name}", compress_type=zipfile.ZIP_STORED)

    return summary


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("Usage: python export_dataset.py dataset_dir output.zip", file=sys.stderr)
        return 2

    dataset_dir, output_zip = Path(argv[0]), Path(argv[1])
    if not dataset_dir.is_dir():
        print(f"ERROR: dataset directory not found: {dataset_dir}", file=sys.stderr)
        return 2
    if output_zip.suffix.lower() != ".zip":
        print("ERROR: output path must end in .zip", file=sys.stderr)
        return 2

    try:
        summary = export(dataset_dir, output_zip)
    except ConversionError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"Wrote {output_zip} ({output_zip.stat().st_size / 1_048_576:.1f} MB)")
    print(f"{'split':<8}{'images':>8}{'annotations':>13}{'converted':>11}{'skipped':>9}")
    for name, n_ann, n_conv, n_skip, n_img in summary:
        print(f"{name:<8}{n_img:>8}{n_ann:>13}{n_conv:>11}{n_skip:>9}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
