"""Shared validation for 30 FPS COCO fish sequences."""
import json
import math
import re
from decimal import Decimal
from pathlib import Path

COCO_FILENAME = "_annotations.coco.json"
SPLIT_NAMES = ("train", "validate", "test")
FRAME_NAME = re.compile(r"^(.*?)-(\d+)\.[^.]+$")
HASHED_NAME = re.compile(r"^(.*?-\d+)_\w+\.rf\.[^.]+\.(\w+)$")


class DatasetError(ValueError):
    """The dataset does not meet the documented input contract."""


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def positive_integer(value, label):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise DatasetError(f"{label} must be a positive integer.")


def child_path(root, relative):
    root = Path(root).resolve()
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise DatasetError(f"Expected a relative dataset path: {relative!r}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise DatasetError(f"Path leaves dataset directory: {relative}")
    return path


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DatasetError(f"Cannot read JSON {path}: {error}")


def load_coco(path):
    data = load_json(path)
    if not isinstance(data, dict):
        raise DatasetError(f"COCO JSON must be an object: {path}")
    for section in ("images", "categories", "annotations"):
        if not isinstance(data.get(section), list):
            raise DatasetError(f"COCO '{section}' must be a list: {path}")
        ids = set()
        for entry in data[section]:
            if not isinstance(entry, dict):
                raise DatasetError(f"COCO '{section}' contains a non-object entry.")
            identifier = entry.get("id")
            if not isinstance(identifier, int) or isinstance(identifier, bool) or identifier < 0:
                raise DatasetError(f"Invalid {section} ID: {identifier!r}")
            if identifier in ids:
                raise DatasetError(f"Duplicate {section} ID: {identifier}")
            ids.add(identifier)
    return data


def find_splits(dataset_dir):
    dataset_dir = Path(dataset_dir)
    if (dataset_dir / COCO_FILENAME).is_file():
        return [dataset_dir]
    splits = sorted(p.parent for p in dataset_dir.glob(f"*/{COCO_FILENAME}"))
    if not splits:
        raise DatasetError(f"No {COCO_FILENAME} found in {dataset_dir} or its immediate subfolders.")
    return splits


def parse_fish_id(name):
    name = str(name).strip()
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", name):
        return None
    number = Decimal(name)
    if number != number.to_integral_value() or number < 0:
        raise DatasetError(f"Fish ID must be a nonnegative whole number: {name}")
    return int(number)


def read_frames(dataset_dir):
    """Read source frames, rejecting malformed data and marking ambiguous identities."""
    frames, categories, seen = [], {}, set()
    for source_dir in find_splits(dataset_dir):
        data = load_coco(source_dir / COCO_FILENAME)
        local_categories = {}
        for category in data["categories"]:
            if "name" not in category:
                raise DatasetError(f"Category {category['id']} has no name.")
            cid, name = category["id"], str(category["name"])
            if cid in categories and categories[cid] != name:
                raise DatasetError(f"Conflicting category ID across source splits: {cid}")
            local_categories[cid] = name
            categories[cid] = name
        images = {}
        for image in data["images"]:
            extra = image.get("extra") or {}
            if not isinstance(extra, dict):
                raise DatasetError(f"Image {image['id']}: extra must be an object.")
            filename = image.get("file_name")
            source = child_path(source_dir, filename)
            if not source.is_file():
                raise DatasetError(f"Missing image: {source}")
            name = extra.get("name") or Path(filename).name
            if not isinstance(name, str) or Path(name).name != name or "\\" in name:
                raise DatasetError(f"Original image name must be a basename: {name!r}")
            hashed = HASHED_NAME.fullmatch(name)
            if hashed:
                name = f"{hashed.group(1)}.{hashed.group(2)}"
            match = FRAME_NAME.fullmatch(name)
            if not match:
                raise DatasetError(f"Expected sequence-NNNN.ext original image name: {name}")
            sequence, number = match.group(1), int(match.group(2))
            if not sequence:
                raise DatasetError(f"Image has no sequence prefix: {name}")
            if (sequence, number) in seen:
                raise DatasetError(f"Duplicate sequence/frame: {sequence}, {number}")
            seen.add((sequence, number))
            positive_integer(image.get("width"), "Image width")
            positive_integer(image.get("height"), "Image height")
            frame = dict(sequence=sequence, number=number, name=name, source=source,
                         width=image["width"], height=image["height"],
                         records=[], fish_ids=[], reason=None)
            images[image["id"]] = frame
            frames.append(frame)
        for annotation in data["annotations"]:
            image_id, cid = annotation.get("image_id"), annotation.get("category_id")
            if any(not isinstance(v, int) or isinstance(v, bool) for v in (image_id, cid)):
                raise DatasetError(f"Invalid image/category reference in annotation {annotation['id']}")
            if image_id not in images or cid not in local_categories:
                raise DatasetError(f"Invalid image/category reference in annotation {annotation['id']}")
            frame = images[image_id]
            values = annotation.get("bbox")
            if not isinstance(values, list) or len(values) != 4 or not all(is_number(v) for v in values):
                raise DatasetError(f"Annotation {annotation['id']}: bbox must be four finite numbers.")
            x, y, w, h = values
            if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > frame["width"] or y + h > frame["height"]:
                raise DatasetError(f"Invalid/out-of-bounds box in {frame['name']}: {values}")
            fish_id = parse_fish_id(local_categories[cid])
            if fish_id is None:
                frame["reason"] = "annotation without an individual fish ID"
            frame["fish_ids"].append(fish_id)
            frame["records"].append({"image": frame["name"], "class_id": 0, "track_id": fish_id,
                                     "bbox": dict(x=x, y=y, width=w, height=h),
                                     "center": dict(x=x+w/2, y=y+h/2)})
        for frame in images.values():
            if not frame["fish_ids"]:
                frame["reason"] = "no annotated fish"
            elif len(frame["fish_ids"]) != len(set(frame["fish_ids"])):
                frame["reason"] = "duplicate individual fish ID"
            frame["signature"] = frozenset(frame["fish_ids"])
            frame["records"].sort(key=lambda record: (record["track_id"] is None, record["track_id"] or 0))
    if not frames:
        raise DatasetError("No source images found.")
    frames.sort(key=lambda frame: (frame["sequence"], frame["number"]))
    return frames, categories


def stable_runs(frames):
    runs, active = [], []
    for frame in frames:
        if frame["reason"]:
            if active:
                runs.append(active)
            active = []
            continue
        if active:
            previous = active[-1]
            if (frame["sequence"] != previous["sequence"]
                    or frame["number"] != previous["number"] + 1
                    or frame["signature"] != previous["signature"]):
                runs.append(active)
                active = []
        active.append(frame)
    if active:
        runs.append(active)
    return runs
