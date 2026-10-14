# Dataset format

All source frames represent consecutive samples from a **30 FPS video**.
Coordinates are in pixels. A missing box means that fish is absent for dataset
construction; the tools never invent a position during an occlusion.

## Source COCO exports

Each input folder is one continuous shot or contains explicitly named sequences.
Accepted layouts are a folder containing `_annotations.coco.json` plus images,
or a Roboflow export with immediate `train/`, `valid/`, `test/` subfolders.
The existing Roboflow assignments are reunited before the organizers create
the new official splits.

Original filenames must identify a sequence and a frame number, for example
`tank_a-0013.jpg`. `images[].extra.name` takes precedence over `file_name`;
Roboflow's hashed filename is used to locate the source image. Hashed names are
also recognized when original-name metadata is absent.

Numeric COCO category **names**, such as `100` and `101`, identify individual
fish. COCO category IDs are export-specific and do not become tracking IDs.
Generic/unassigned category names break a usable sequence.

Frame numbers must be consecutive within a chunk. Different shot prefixes and
different input folders are never joined. Assign distinct sequence prefixes to
different recordings. The full identity namespace is `(source, recording_id,
track_id)`; within one chunk, `track_id` is sufficient.

## Prepared folder layout

```text
dataset/
  manifest.json
  README.txt
  train/
    chunk_001_.../
      images/
      annotations.json
  validate/
    chunk_002_.../
      images/
      annotations.json
  test/
    chunk_003_.../
      images/
      annotations.json
```

`annotations.json` is a list with one record per fish per frame:

```json
{
  "image": "tank_a-0013.jpg",
  "class_id": 0,
  "track_id": 100,
  "bbox": {"x": 10, "y": 20, "width": 5, "height": 7},
  "center": {"x": 12.5, "y": 23.5}
}
```

| Field | Meaning |
|---|---|
| `image` | Filename relative to the chunk's `images/` directory |
| `class_id` | Always `0`: the single detection class, `fish` |
| `track_id` | Nonnegative integer identifying a fish within its recording |
| `bbox` | Top-left `x`, `y`, plus positive `width`, `height`, all in pixels |
| `center` | Bounding-box midpoint: `x + width/2`, `y + height/2` |

Center is the box midpoint, not a segmentation centroid. Boxes must fit within
COCO image dimensions. The preparation utility does not decode image pixels;
organizers can use the optional image audit to verify dimensions and decoding.

## Manifest

`manifest.json` has schema version `1`, FPS `30`, the requested split ratios,
actual split counts, source summaries, and a `chunks` list. A chunk includes:

- `chunk_id`, `source`, `recording_id`, `split`, and a relative `path`.
- `frame_count` and ordered `frames` containing `number` and `image`.
- A sorted list of `track_ids`, constant throughout the chunk.

Source IDs such as `source_01` preserve provenance without publishing the
organizer's absolute computer paths. Sources are numbered in command-line order.
Source summaries list short omitted runs and ambiguous frames. Existing outputs
are refused so generating a new dataset does not silently overwrite a release.

## Prediction samples

```python
from prediction_loader import PredictionDataset

dataset = PredictionDataset("dataset", "train")
sample = dataset[0]
```

Each sample contains `chunk_id`, `recording_id`, `source`, `split`, `fps`,
`track_ids`, `observed_frames`, `future_frames`, `observed`, and `future`.

`observed` has shape **[fish count, observed steps, 2]** and `future` has shape
**[fish count, future steps, 2]**. Both are Python lists of `[x, y]` coordinates,
with fish in `track_ids` order. Defaults are 10 observed steps, 10 future steps,
and stride 1. A 30-frame chunk supplies **11 joint windows**. Each joint window
contains one trajectory per fish, so it is not equivalent to one per-fish sample.

For a chunk of `N` frames, the window count is:

```python
max(0, (N - observed_steps - future_steps) // stride + 1)
```

The loader never crosses a chunk or split boundary. It rejects missing centers,
changed membership, duplicate IDs, reused source frames, invalid paths, and
nonconsecutive frame numbers. Different chunks can contain different numbers
of fish, so neural-network batching requires custom collation or padding.

These samples contain future ground truth. For a blind competition, test targets
and test annotations must remain with the organizers; this repo does not
currently implement an unlabeled test-input loader or a scoring server.
