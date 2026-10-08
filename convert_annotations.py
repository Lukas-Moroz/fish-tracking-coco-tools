#!/usr/bin/env python3
"""Convert a Roboflow COCO annotation file into a simplified fish-tracking JSON.

Each output record contains only the original image name, the persistent fish ID
(taken from the numeric category *name*, not the COCO category_id), and the
pixel-space bounding box.

Usage:
    python convert_annotations.py input_coco.json output.json
"""

import json
import math
import re
import sys
from pathlib import Path

REQUIRED_SECTIONS = ("images", "categories", "annotations")

# Category names that look like numbers. Anything matching this must parse to an
# integer fish ID; anything not matching (e.g. "fish") is treated as unassigned.
NUMERIC_NAME = re.compile(r"^[+-]?\d+(\.\d+)?$")


class ConversionError(Exception):
    """Raised when the source file is malformed."""


def load_coco(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise ConversionError(f"Input file not found: {path}")
    except json.JSONDecodeError as e:
        raise ConversionError(f"Input file is not valid JSON: {e}")

    if not isinstance(data, dict):
        raise ConversionError("Top-level JSON must be an object.")
    missing = [s for s in REQUIRED_SECTIONS if s not in data]
    if missing:
        raise ConversionError(f"Missing required section(s): {', '.join(missing)}")
    for section in REQUIRED_SECTIONS:
        if not isinstance(data[section], list):
            raise ConversionError(f"Section '{section}' must be a list.")
    return data


def build_image_lookup(images):
    """Map image id -> original filename (extra.name, falling back to file_name)."""
    lookup = {}
    for img in images:
        if "id" not in img:
            raise ConversionError(f"Image entry has no 'id': {img}")
        extra = img.get("extra") or {}
        name = extra.get("name") or img.get("file_name")
        if not name:
            raise ConversionError(f"Image {img['id']} has neither extra.name nor file_name.")
        lookup[img["id"]] = name
    return lookup


def build_category_lookup(categories):
    """Map category id -> category name."""
    lookup = {}
    for cat in categories:
        if "id" not in cat or "name" not in cat:
            raise ConversionError(f"Category entry missing 'id' or 'name': {cat}")
        lookup[cat["id"]] = str(cat["name"])
    return lookup


def parse_fish_id(category_name):
    """Return the integer fish ID, or None if the name is not numeric (e.g. "fish").

    Raises ConversionError if the name looks numeric but is not a whole number.
    """
    name = category_name.strip()
    if not NUMERIC_NAME.match(name):
        return None
    value = float(name)
    if not value.is_integer():
        raise ConversionError(f"Cannot parse fish ID from category name '{category_name}'.")
    return int(value)


def is_number(value):
    # bool is a subclass of int, so exclude it explicitly.
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def clean_number(value):
    """Keep decimals as-is; drop a trailing .0 from whole-number floats."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def parse_bbox(bbox, ann_id):
    if not isinstance(bbox, list) or len(bbox) != 4 or not all(is_number(v) for v in bbox):
        raise ConversionError(
            f"Annotation {ann_id}: bbox must be a list of exactly four numbers, got {bbox!r}"
        )
    x, y, w, h = (clean_number(v) for v in bbox)
    return {"x": x, "y": y, "width": w, "height": h}


def convert(data):
    images = build_image_lookup(data["images"])
    categories = build_category_lookup(data["categories"])

    records = []
    skipped = 0
    for ann in data["annotations"]:
        ann_id = ann.get("id", "<no id>")

        image_id = ann.get("image_id")
        if image_id not in images:
            raise ConversionError(f"Annotation {ann_id} references unknown image_id {image_id!r}.")
        image_name = images[image_id]

        category_id = ann.get("category_id")
        if category_id not in categories:
            raise ConversionError(
                f"Annotation {ann_id} references unknown category_id {category_id!r}."
            )
        category_name = categories[category_id]

        bbox = parse_bbox(ann.get("bbox"), ann_id)

        fish_id = parse_fish_id(category_name)
        if fish_id is None:
            print(
                f"WARNING: skipping annotation without individual fish ID "
                f"(image={image_name}, annotation_id={ann_id}, category='{category_name}')",
                file=sys.stderr,
            )
            skipped += 1
            continue

        records.append({"image": image_name, "fish_id": fish_id, "bbox": bbox})

    # Bbox coordinates break ties when the same fish ID appears twice in one image.
    records.sort(
        key=lambda r: (
            r["image"],
            r["fish_id"],
            r["bbox"]["x"],
            r["bbox"]["y"],
            r["bbox"]["width"],
            r["bbox"]["height"],
        )
    )
    return records, skipped


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("Usage: python convert_annotations.py input_coco.json output.json", file=sys.stderr)
        return 2

    input_path, output_path = Path(argv[0]), Path(argv[1])
    if input_path.resolve() == output_path.resolve():
        print("ERROR: output path must differ from the input COCO file.", file=sys.stderr)
        return 2

    try:
        data = load_coco(input_path)
        records, skipped = convert(data)
    except ConversionError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
        f.write("\n")

    fish_ids = sorted({r["fish_id"] for r in records})
    print(f"Wrote {output_path}")
    print(f"Source annotations:           {len(data['annotations'])}")
    print(f"Converted:                    {len(records)}")
    print(f"Skipped (no individual ID):   {skipped}")
    print(f"Unique fish IDs ({len(fish_ids)}):          {fish_ids}")
    print("\nFirst 5 records:")
    print(json.dumps(records[:5], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
