# fish-tracking-coco-tools

Scripts for turning a Roboflow COCO export of a fish-tracking dataset into a simpler format: one record per bounding box, keyed by the original image filename and a persistent fish ID.

Requires Python 3.8+ and nothing outside the standard library.

## Output format

```json
[
  {
    "image": "fish_sample2_mp4-0003.jpg",
    "fish_id": 100,
    "bbox": { "x": 654.45, "y": 501.11, "width": 24.2, "height": 33.49 }
  }
]
```

- `image` is the original filename (`images[].extra.name`), not Roboflow's hashed `file_name`. If `extra.name` is missing, `file_name` is used.
- `fish_id` is the numeric category **name** (e.g. `"100"`), not the COCO `category_id`.
- `bbox` is COCO `[x, y, width, height]` in pixels, measured from the top-left corner. Values are not normalized or rounded.
- Records are sorted by image, then by fish ID.

Annotations whose category name isn't a number (e.g. the placeholder `"fish"` for fish that don't have an individual ID yet) are skipped with a warning.

## Usage

### Convert one COCO file

```bash
python convert_annotations.py path/to/_annotations.coco.json fish_tracking_annotations.json
```

The script prints how many annotations were converted and skipped, the unique fish IDs, and the first five records.

### Export images and cleaned JSON as a zip

```bash
python export_dataset.py "path/to/fish tracking 2.coco" fish_tracking_export.zip
```

It processes every split folder (`train/`, `valid/`, `test/`) that contains `_annotations.coco.json` and renames each image to its original filename, so the images match the `image` field in the JSON:

```
fish_tracking_export/
  train/fish_tracking_annotations.json
  train/images/fish_sample2_mp4-0000.jpg
  valid/...
  test/...
```

`export_dataset.py` imports from `convert_annotations.py`, so keep both files in the same folder.

## Validation

Both scripts stop with an error (exit code 1) if:

- `images`, `categories`, or `annotations` is missing
- an annotation references an unknown `image_id` or `category_id`
- a `bbox` isn't exactly four numbers
- a numeric category name isn't a whole number (e.g. `"100.5"`)

`export_dataset.py` also stops before writing the zip if an image file is missing, or if two images in the same split would end up with the same filename.

## Dataset

The dataset is not included in this repo. It comes from [fish-tracking-2 on Roboflow Universe](https://universe.roboflow.com/new-workspace-tqctf/fish-tracking-2-yr4n1) and is licensed CC BY 4.0. To use it, export it in COCO format and pass the extracted folder to the scripts.
