# Competition data format

Frames are consecutive samples from 30 FPS video. Coordinates use original-image
pixels: x increases rightward and y downward from the top-left corner. Centers
are bounding-box midpoints. Images are not resized.

## Sections and packages (schema version 2)

| Section | Filename code | Purpose |
|---|---:|---|
| `train` | 1 | Established training assignment |
| `validate` | 2 | Established validation assignment |
| `test` | 3 | Official private test assignment |
| `extra` | 4 | Additional valid source frames available for shared use |
| `extra_test` | 3 | Separately reserved private holdout from additional data |
| `public` | 4 | Selected non-held data in a standalone shared export |

The official train/validate/test assignments stay fixed. Additional frames have
separate `extra` and `extra_test` accounting; reserving some of them does not
alter the original three split totals or ratios. Every team receives the same
released supplemental labels and may choose how to use them.

The private organizer package can contain all five internal sections. The
`participants` export preserves train/validate folders and optional `extra/`,
excluding both private test sections. A supplemental-only export contains just
the unreserved additional data:

```text
supplement/
  manifest.json
  README.txt
  public/
    0104/
      images/
        01044001.jpg
        01044002.jpg
      annotations.json
```

One JSON per chunk keeps labels independent. The manifest is an index rather
than a combined master file of every box and centerpoint. The standalone export
uses `public/` so it can be loaded with `split="public"` without depending on an
existing training/validation folder.

## Photo names

Names use `VVCCSFFF.ext`, preserving the image encoding:

| Field | Width | Values |
|---|---:|---|
| Video ID | 2 | `01`–`99`; identifies the full source video |
| Chunk number | 2 | `01`–`99` within that video, retained across waves |
| Section code | 1 | `1` train, `2` validation, `3` private test/holdout, `4` supplemental/shared public |
| Local frame | 3 | `001`–`999`, restarting for each chunk |

`01044017.jpg` means video 01, chunk 04, supplemental/shared data, frame 017.
Chunk IDs are `VVCC`, such as `0104`, and do not encode the section. Reserving
an additional chunk changes its private section digit from `4` to `3`, but its
chunk and fish IDs stay the same. Manifest section names distinguish `test`
from `extra_test`, which share the private code `3`.

Capacity overflows are rejected rather than silently widening fields. Chunks
over 999 frames must be divided explicitly before release.

## Per-chunk annotations

`annotations.json` is a list with one record per fish per image:

```json
{
  "image": "01044001.jpg",
  "class_id": 0,
  "track_id": 104001,
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
chunk 04 uses 104001, 104002, etc. These are integers. Cross-chunk biological
identity is not implied. Source-to-release identity maps remain private.

Additional frames are grouped at membership changes and source-frame gaps,
including runs shorter than 20 frames. Each chunk still has fixed annotated
membership. Short chunks retain valid labels without padding or interpolating
missing observations. They need not meet the original prediction minimum.

## Manifest

Version-2 manifests contain `schema_version=2`, `fps=30`, `release_id`, `waves`,
`filename_format="VVCCSFFF"`, `track_id_scope="globally_unique_per_chunk"`,
`distribution`, a video registry, counts, and a `chunks` list.

| Distribution | Sections | Counts |
|---|---|---|
| `private` | Official splits, optional `extra` and `extra_test` | Original three `split_counts`; `extra_counts` and `extra_test_counts` separately |
| `participants` | `train`, `validate`, optional `extra` | Available official counts and unreserved additional counts; no holdout entries |
| `public` | `public` only | `split_counts.public` for the selected shared export |

Each chunk includes:

- `chunk_id`, `video_id`, `chunk_number`, `recording_id`, `source`, and `wave`.
- Its section in `split`, relative `path`, `frame_count`, and ordered `track_ids`.
- Ordered `frames` with local `number`, `image`, and original `source_number`.

A supplemental-only public manifest has `split="public"` and
`path="public/<chunk_id>"` for every chunk. It contains only selected unreserved
`extra` chunks, omitting original filenames, source fish-ID maps, and both
private test sections. `recording_id` must consistently identify one full video
across waves. Absolute computer paths are not included.

Source frame numbers detect reuse across chunks and waves. Local numbers start
at 1 and identify prediction windows and records. Newly added chunks use the
next unused number within their video; earlier chunks are never renumbered.
A manifest's section names and counts determine its contents; code `4` alone
does not imply that all training and validation data were repackaged.

## Organizer inputs (schema version 1)

`split_dataset.py` accepts COCO folders with `_annotations.coco.json` and images,
including Roboflow exports with immediate train/valid/test folders. Source
assignments are reunited before whole-chunk splitting. Original filenames must
identify a recording and source frame, such as `tank_a-0013.jpg`. Roboflow
`extra.name` takes precedence; hashed exports are recognized. Numeric category
names identify individual fish; category IDs are export-specific.

The splitter finds maximal consecutive runs with fixed annotated membership.
A missing/added box, gap, recording change, or ambiguous ID ends a run. Its
intermediate format keeps source names, frame numbers, and fish IDs. The release
step applies release names and chunk-specific identities. `extra` adds remaining
valid frames; `reserve-extra` can withhold selected additional chunks.

Review duplicates and alignment before release. Excluding a reviewed recording
with `--exclude-recording` is an organizer decision; the exporter does not
automatically prove that two transformed videos are independent footage. The
bundled synthetic dataset is an intermediate format. The loader supports both
schema versions and existing train/validation packages.

## Prediction windows

```python
from prediction_loader import PredictionDataset

dataset = PredictionDataset("path/to/supplement", "public")
```

A sample includes metadata, fish IDs, ten image paths, observed/future frame
numbers, and labels shaped `[fish_count][10][2]`. `prediction_input(index)` omits
coordinate labels; `predict_sample` passes ten encoded images to a team's model
and validates ten future centers per fish. See the
[prediction contract](PREDICTION_FORMAT.md).

At stride 1, a chunk provides `max(0, N - 19)` windows. Chunks shorter than 20
frames validate and remain available for detection/tracking but provide no
prediction windows. Windows never cross chunks. Organizers can load private
`test` and `extra_test` separately; neither section is accessible in a shared
package.
