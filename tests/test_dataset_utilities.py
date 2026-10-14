import itertools
import json
import random
import tempfile
import unittest
from pathlib import Path

from dataset_common import DatasetError
from prediction_loader import PredictionDataset
from split_dataset import assign_splits, export_chunks


def make_source(root, lengths=(40, 35, 30)):
    source = root / "source"
    source.mkdir()
    images, annotations = [], []
    number = 0
    for run_index, length in enumerate(lengths):
        for _ in range(length):
            name = f"video-{number:04d}.jpg"
            (source / name).write_bytes(f"photo {number}".encode())
            images.append(dict(id=number, file_name=name, width=1000, height=1000))
            # Category IDs deliberately differ from individual fish labels.
            for cid in (1, run_index + 2):
                annotations.append(dict(id=len(annotations), image_id=number, category_id=cid,
                                        bbox=[number, cid * 20, 10, 8]))
            number += 1
    categories = [dict(id=cid, name=str(100 + cid)) for cid in range(1, len(lengths) + 2)]
    # COCO image order is not chronological, as in Roboflow exports.
    data = dict(images=list(reversed(images)), annotations=list(reversed(annotations)), categories=categories)
    (source / "_annotations.coco.json").write_text(json.dumps(data))
    return source


class UtilityTests(unittest.TestCase):
    def test_partition_matches_exhaustive_frame_ratio_search(self):
        generator = random.Random(4)
        for n in range(3, 12):
            lengths = [generator.randint(30, 200) for _ in range(n)]
            for ratios in ((70, 15, 15), (3, 2, 1), (1, 1, 1)):
                targets = [sum(lengths) * r / sum(ratios) for r in ratios]
                possibilities = []
                for i, j in itertools.combinations(range(1, n), 2):
                    counts = (sum(lengths[:i]), sum(lengths[i:j]), sum(lengths[j:]))
                    possibilities.append((sum((a-b)**2 for a, b in zip(counts, targets)), i, j))
                _, i, j = min(possibilities)
                expected = ["train"] * i + ["validate"] * (j-i) + ["test"] * (n-j)
                self.assertEqual(assign_splits(lengths, ratios), expected)

    def test_export_and_loader_targets_have_correct_order_and_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_source(root)
            original = (source / "_annotations.coco.json").read_bytes()
            output = root / "export"
            manifest = export_chunks([source], output)
            self.assertEqual([c["frame_count"] for c in manifest["chunks"]], [40, 35, 30])
            self.assertEqual([c["split"] for c in manifest["chunks"]], ["train", "validate", "test"])
            all_numbers = set()
            for split, expected_windows in (("train", 21), ("validate", 16), ("test", 11)):
                dataset = PredictionDataset(output, split)
                self.assertEqual(len(dataset), expected_windows)
                entry = dataset.chunks[0]["entry"]
                for f in entry["frames"]:
                    self.assertNotIn(f["number"], all_numbers)
                    all_numbers.add(f["number"])
                    self.assertEqual((output / entry["path"] / "images" / f["image"]).read_bytes(),
                                     (source / f["image"]).read_bytes())
                annotations = json.loads((output / entry["path"] / "annotations.json").read_text())
                self.assertTrue(all(set(a) == {"image", "class_id", "track_id", "bbox", "center"} for a in annotations))
                self.assertTrue(all(a["class_id"] == 0 and a["track_id"] >= 101 for a in annotations))
                for sample in dataset:
                    self.assertEqual(len(sample["observed_frames"]), 10)
                    self.assertEqual(len(sample["future_frames"]), 10)
                    self.assertEqual(sample["future_frames"][0], sample["observed_frames"][-1] + 1)
                    for index, track_id in enumerate(sample["track_ids"]):
                        cid = track_id - 100
                        self.assertEqual(sample["observed"][index], [[n + 5, cid * 20 + 4] for n in sample["observed_frames"]])
                        self.assertEqual(sample["future"][index], [[n + 5, cid * 20 + 4] for n in sample["future_frames"]])
                dataset[0]["observed"][0][0][0] = -100
                self.assertGreaterEqual(dataset[0]["observed"][0][0][0], 5)
                self.assertEqual(len(PredictionDataset(output, split, stride=2)), (expected_windows + 1)//2)
            self.assertEqual(len(all_numbers), 105)
            self.assertEqual((source / "_annotations.coco.json").read_bytes(), original)

    def test_missing_box_breaks_runs_before_window_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_source(root, (70, 35, 30))
            file = source / "_annotations.coco.json"
            data = json.loads(file.read_text())
            data["annotations"] = [a for a in data["annotations"] if not (a["image_id"] == 35 and a["category_id"] == 2)]
            file.write_text(json.dumps(data))
            manifest = export_chunks([source], root / "export")
            self.assertEqual([c["frame_count"] for c in manifest["chunks"]], [35, 34, 35, 30])
            for split in ("train", "validate", "test"):
                for sample in PredictionDataset(root / "export", split):
                    self.assertNotIn(35, sample["observed_frames"] + sample["future_frames"])

    def test_loader_rejects_tampered_membership_and_frame_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_source(root)
            output = root / "export"
            manifest = export_chunks([source], output)
            annotation_file = output / manifest["chunks"][0]["path"] / "annotations.json"
            records = json.loads(annotation_file.read_text())
            annotation_file.write_text(json.dumps(records[1:]))
            with self.assertRaisesRegex(DatasetError, "membership changes"):
                PredictionDataset(output, "train")
            annotation_file.write_text(json.dumps(records))
            for number, frame in enumerate(manifest["chunks"][1]["frames"]):
                frame["number"] = number
            with self.assertRaises(DatasetError):
                (output / "manifest.json").write_text(json.dumps(manifest))
                PredictionDataset(output, "validate")

    def test_insufficient_chunks_and_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_source(root, (30,))
            output = root / "export"
            with self.assertRaisesRegex(DatasetError, "three qualifying"):
                export_chunks([source], output)
            self.assertFalse(output.exists())
            output.mkdir()
            with self.assertRaisesRegex(DatasetError, "already exists"):
                export_chunks([source], output)


if __name__ == "__main__":
    unittest.main()
