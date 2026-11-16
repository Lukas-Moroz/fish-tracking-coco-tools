# Competition data format

Frames are consecutive samples from 30 FPS video. Coordinates use original-image
pixels: x increases rightward and y downward from the top-left corner. Centers
are bounding-box midpoints. Images are not resized.

## Public layout (schema version 2)

```text
participants/
  manifest.json
  README.txt
  public/
    0101/
      images/
        01014001.jpg
        01014002.jpg
      annotations.json
    0102/
      images/
        01024001.jpg
      annotations.json
```

All participants receive the same public pool and choose their own training and
validation usage. Private test files and test manifest entries are absent.
One JSON per chunk keeps labels independent; the manifest is an index rather
than a combined master file of every box and centerpoint.

The organizer's private release retains `train/`, `validate/`, and `test/`
assignments. It may also contain `extra/` chunks for valid frames omitted by the
minimum length rule. The public exporter merges every non-test section into
`public/`, renaming the split digit to `4`. Internal chunk files and assignments
are preserved in the private release.

## Photo names

Names use `VVCCSFFF.ext`, preserving the image encoding:

| Field | Width | Values |
|---|---:|---|
| Video ID | 2 | `01`–`99`; identifies the full source video |
| Chunk number | 2 | `01`–`99` within that video, retained across waves |
| Section code | 1 | `4` public/supplemental; internal `1` train, `2` validation, `3` private test |
| Local frame | 3 | `001`–`999`, restarting for each chunk |

`01024017.jpg` means video 01, chunk 02, public data, local frame 017.
Chunk IDs are `VVCC`, such as `0102`, and do not encode the section. Changing a
public filename's section digit does not change its chunk or fish IDs.
Capacity overflows are rejected rather than silently widening fields. Chunks
over 999 frames must be divided explicitly before release.

## Per-chunk annotations

`annotations.json` is a list with one record per fish per image:

```json
{
  "image": "01014001.jpg",
  "class_id": 0,
  "track_id": 101001,
  "bbox": {"x": 10, "y": 20, "width": 5, "height": 7},
  "center": {"x": 12.5, "y": 23.5}
}
```

`class_id` is always `0`, the shared detection class. Each fish has its own
positive numeric `track_id`. `bbox` uses top-left x/y plus positive width/height.
`center` is the midpoint, stored as an x/y object.

Every fish has one identity throughout its chunk and a different ID in every
other chunk. The allocation formula is
`(video_id * 100 + chunk_number) * 1000 + local_fish`, with `local_fish` from
1–999 in sorted source-ID order. Video 01 / chunk 01 uses 101001, 101002, etc.;
chunk 02 uses 102001, 102002, etc. These are integers. Cross-chunk biological
identity is not implied. Source-to-release identity maps remain private.

Supplemental frames are grouped at each membership change or source-frame gap,
including runs shorter than 20 frames. Each resulting chunk still has fixed
annotated fish membership. Short chunks retain all valid labels without padding
or interpolating missing observations.

## Manifest

Version-2 manifests contain `schema_version=2`, `fps=30`, `release_id`, `waves`,
`filename_format="VVCCSFFF"`, `track_id_scope="globally_unique_per_chunk"`,
`distribution`, a video registry, counts, and a `chunks` list.

| Distribution | Chunk sections | Counts |
|---|---|---|
| `public` | `public` only | `split_counts.public` |
| `private` | `train`, `validate`, `test`, optional `extra` | Three official `split_counts`; supplemental `extra_counts` separately |
| `participants` (legacy) | `train`, `validate`, optional `extra` | Legacy available split counts, with no test entries |

Each chunk includes:

- `chunk_id`, `video_id`, `chunk_number`, `recording_id`, `source`, and `wave`.
- Its section in `split`, relative `path`, `frame_count`, and ordered `track_ids`.
- Ordered `frames` with local `number`, `image`, and original `source_number`.

Public chunks use `split="public"` and `path="public/<chunk_id>"`. The public
manifest omits original filenames, source fish-ID maps, and private test entries.
`recording_id` must identify one full video consistently across waves. Absolute
computer paths are not included.

Source frame numbers detect accidental reuse across chunks and waves. Local
frame numbers start at 1 and are used for prediction windows and JSON records.
New supplemental chunks use the next unused number within their video; earlier
chunks are never renumbered. Supplemental counts do not alter the organizer's
three official split totals or ratios.

## Organizer inputs (schema version 1)

`split_dataset.py` accepts COCO folders with `_annotations.coco.json` and images,
including Roboflow exports with immediate train/valid/test folders. Source
assignments are reunited before whole-chunk splitting. Original filenames must
identify a recording and source frame, such as `tank_a-0013.jpg`. Roboflow
`extra.name` takes precedence; hashed exports are recognized. Numeric category
names identify individual fish; COCO category IDs are export-specific.

The splitter finds maximal consecutive runs with fixed annotated membership.
A missing/added box, frame gap, recording change, or ambiguous ID ends a run.
Its intermediate format keeps source filenames, frame numbers, and fish IDs.
`release_dataset.py build` applies release names and chunk-specific identities.
`release_dataset.py extra` can later add remaining valid source frames as short
supplemental chunks. Review duplicates and annotation alignment before release;
the export does not automatically establish that two transformed videos are
independent footage.

The bundled `examples/dataset/` is an intermediate synthetic dataset. The loader
supports schema versions 1 and 2; legacy train/validation packages remain usable.

## Prediction windows

```python
from prediction_loader import PredictionDataset

dataset = PredictionDataset("participants", "public")
```

A sample includes metadata, `track_ids`, ten `input_images` paths,
`observed_frames`, `future_frames`, and labels `observed` / `future` shaped
`[fish_count][10][2]`. `prediction_input(index)` omits coordinate labels;
`predict_sample` calls a team's predictor with ten encoded images and validates
ten future centers per fish. See the [prediction contract](PREDICTION_FORMAT.md).

At stride 1, each chunk supplies `max(0, N - 19)` windows. Public chunks shorter
than 20 frames validate and remain available for detection/tracking, but supply
no prediction windows. Windows never cross chunk boundaries. The public pool is
not an additional official scoring split; the private test set is evaluated by
organizers using their reserved data.
