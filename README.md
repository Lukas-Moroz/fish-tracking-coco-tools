# Fish tracking competition helpers

This repo contains the helper functions for preparing fish-video data and loading
motion-prediction samples. **The real competition dataset will be distributed
separately.** A small synthetic dataset is included so you can try everything now.

## What's in this repo?

| File or folder | What it does |
|---|---|
| `prediction_loader.py` | Loads 10 past center positions and the next 10 positions for each fish |
| `split_dataset.py` | Organizer tool: finds stable sequences and creates train/validation/test folders |
| `examples/dataset/` | Ready-to-run synthetic data: three 40-frame shots with two moving shapes each |
| `examples/make_example.py` | Recreates synthetic source data without extra dependencies |
| `dataset_common.py` | Validation shared by the helper functions |
| `docs/` | Detailed data format and instructions for organizers |
| `tests/` | Automated checks |
| `tools/` | Optional annotation audit for organizers |

## 1. Get the repo

You need **Python 3.8 or newer**. The main helpers need no additional packages.

```bash
git clone https://github.com/Lukas-Moroz/fish-tracking-coco-tools.git
cd fish-tracking-coco-tools
```

You can also download the repo as a ZIP and open a terminal in its extracted folder.

## 2. Try the included example

```bash
python3 prediction_loader.py examples/dataset --split train
```

This prints the number of prediction samples and the first sample. The example
has **21 samples in each split**. It uses colored shapes, not real fish footage.

## 3. Use the loader in your code

```python
from prediction_loader import PredictionDataset

dataset = PredictionDataset("examples/dataset", "train")
sample = dataset[0]

past = sample["observed"]      # [fish_count][10][2], x/y in pixels
target = sample["future"]     # [fish_count][10][2], x/y in pixels
fish_ids = sample["track_ids"] # Same fish order in past and target

print("Samples:", len(dataset))
print("Fish IDs:", fish_ids)
```

Replace `"examples/dataset"` with the path to your official dataset when provided.
Use `"train"`, `"validate"`, or `"test"` to select a split. Keep the organizers'
assignments unchanged.

Each sample uses **10 observed frames to predict the next 10 frames**, then the
loader slides forward by one frame. It never crosses a chunk or split boundary.
A 30-frame chunk gives **11 samples**, each containing all fish in that chunk.
All videos are sampled at **30 FPS**; centers are bounding-box midpoints.

Different chunks can have different fish counts. If batching samples for a neural
network, use custom collation or padding. Samples contain future labels; a blind
competition's test inputs/targets require a separate organizer distribution.

## Preparing new data (organizers)

```bash
python3 split_dataset.py "path/to/roboflow_export" --output data/official_v1
```

This keeps the **largest consecutive sequences with the same annotated fish IDs
in every frame**, with a minimum of **30 frames**. A missing or added fish box,
frame gap, shot change, or ambiguous identity ends a sequence.

Whole chunks are assigned to `train`, `validate`, and `test`, targeting **70:15:15
by photo count**. At least three usable chunks are required. Ratios are approximate;
actual counts and fixed assignments are recorded in `manifest.json`. Original
files are preserved. Existing output directories are refused.

Detection has one class, `fish` (`class_id=0`), plus separate `track_id` values.
Each annotation contains only `image`, `class_id`, `track_id`, `bbox`, and `center`.

See [organizer instructions](docs/ORGANIZERS.md) for multiple-shot preparation,
example regeneration, and distribution. See [data format](docs/DATA_FORMAT.md)
for the annotation schema and prediction sample details.

## Run the tests

```bash
python3 -m unittest discover -s tests -v
```

Real footage, generated datasets, local audit results, and exported JSONL samples
are excluded from Git. The bundled synthetic example is the only committed image
data. The utility code is currently unlicensed; no open-source license has been
selected. Official scoring rules and submission formats will be supplied separately.
