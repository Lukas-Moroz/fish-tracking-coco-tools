# Competition data format

Frames are consecutive samples from 30 FPS video. Coordinates use original-image
pixels: x increases rightward and y downward from the top-left corner. A target
centerpoint is the bounding-box midpoint. No image resizing is performed.

## Release layout (schema version 2)

```text
release/
  manifest.json
  README.txt
  train/
    0101/
      images/
        01011001.jpg
        01011002.jpg
      annotations.json
  validate/
    0102/
      images/
        01022001.jpg
      annotations.json
```

The organizer's **private** release additionally contains `test/<chunk>/`.
The participant package omits test directories, test annotations, and test
entries from its manifest. The folder name for validation is `validate`.

One JSON per chunk keeps each batch of labels independent. The manifest is the
master index, not a master file of all boxes and centerpoints. This supports new
waves without replacing earlier chunk annotation files.

## Photo names

Names use `VVCCSFFF.ext`, keeping the original image encoding:

| Field | Width | Values |
|---|---:|---|
| Video ID | 2 | `01`–`99`; identifies the full source video |
| Chunk number | 2 | `01`–`99` within that video, retained across waves |
| Split | 1 | `1` train, `2` validation, `3` private test |
| Local frame | 3 | `001`–`999`, restarting for each chunk |

`01022017.jpg` means video 01, chunk 02, validation, local frame 017.
Chunk IDs are `VVCC`, such as `0102`; they do not encode the split.
The exporter rejects capacity overflows rather than silently widening fields.
Chunks exceeding 999 frames must be divided explicitly before release.

## Per-chunk annotations

`annotations.json` is a list with one record per fish per image:

```json
{
  "image": "01011001.jpg",
  "class_id": 0,
  "track_id": 101001,
  "bbox": {"x": 10, "y": 20, "width": 5, "height": 7},
  "center": {"x": 12.5, "y": 23.5}
}
```

`class_id` is always `0` for fish. `bbox` uses top-left x/y and positive width/
height. `center` equals `[x + width/2, y + height/2]`, stored as an x/y object.

Every fish has one identity throughout its chunk. It gets a **different ID in
every other chunk**, even if the original annotation used one continuous video
identity. IDs are globally distinct within the release and across its waves.

The allocation formula is `(video_id * 100 + chunk_number) * 1000 + local_fish`.
`local_fish` is 1–999, assigned in sorted source-ID order. Video 01 / chunk 01
therefore uses 101001, 101002, etc.; chunk 02 uses 102001, 102002, etc. These are
integers, not fixed-width strings. Cross-chunk biological identity is not implied.
The original-to-release ID mapping is retained only in the private manifest.

## Manifest

Release manifests contain `schema_version=2`, `fps=30`, `release_id`, `waves`,
`distribution` (`private` or `participants`), `filename_format="VVCCSFFF"`,
`track_id_scope="globally_unique_per_chunk"`, a video registry, split counts,
and a `chunks` list. Each chunk includes:

- `chunk_id`, `video_id`, `chunk_number`, `recording_id`, `source`, `wave`.
- Its fixed `split`, relative `path`, `frame_count`, and ordered `track_ids`.
- Ordered `frames` with local `number`, new `image`, and original `source_number`.

`recording_id` must identify one full video unambiguously across all waves.
The private manifest additionally retains original filenames and source fish-ID
mappings for audit. Absolute computer paths are not included.

Original source frame numbers detect accidental reuse across chunks or waves.
Local frame numbers, starting at 1, are used by prediction windows and submissions.
New chunks get the next available number within their video; late annotation of
earlier footage does not renumber released chunks.

## Organizer inputs (schema version 1)

`split_dataset.py` accepts COCO folders with `_annotations.coco.json` and images,
including Roboflow exports with immediate train/valid/test subfolders. Those
source assignments are reunited before the official whole-chunk split is made.
Original filenames must identify the recording and source frame, for example
`tank_a-0013.jpg`. Roboflow `extra.name` takes precedence; hashed exports are
also recognized. Numeric category **names** identify fish; category IDs are
export-specific. Distinct full videos must have distinct recording names.

It finds maximal consecutive sequences with the same annotated fish membership.
A missing/added box, frame gap, recording change, or ambiguous ID ends a chunk.
This intermediate format retains original names, source frame numbers, and
continuous-video fish IDs. `release_dataset.py build` applies the final naming
and chunk-ID rules. The bundled `examples/dataset/` is an intermediate synthetic
dataset. The prediction loader supports both schema versions.

## Prediction windows

Each sample includes chunk/recording/split metadata, `track_ids`, ten
`input_images` paths, `observed_frames`, `future_frames`, and training labels
`observed` / `future` shaped `[fish_count][10][2]`. Labels are Python x/y lists.
`prediction_input(index)` omits coordinate labels; `predict_sample` calls a
team's model with exactly ten encoded images and checks its ten future centers
per fish. See the [prediction contract](PREDICTION_FORMAT.md).

Stride defaults to 1. For `N` frames, a chunk supplies `max(0, N - 19)` windows.
Windows never cross chunks or splits. Participant test access is refused;
withheld test evaluation must run using the organizer's private data.
