import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dataset_common import DatasetError, parse_fish_id, read_frames, stable_runs
from examples.make_example import make_example
from prediction_loader import PredictionDataset
from split_dataset import export_chunks
from tests.test_dataset_utilities import make_source


class ValidationTests(unittest.TestCase):
    def test_committed_example_is_complete_and_loadable(self):
        root = Path(__file__).resolve().parents[1] / "examples" / "dataset"
        for split in ("train", "validate", "test"):
            dataset = PredictionDataset(root, split)
            self.assertEqual(len(dataset), 21)
            self.assertEqual(dataset[0]["track_ids"], [100, 101])
            self.assertEqual(dataset[-1]["future_frames"], list(range(30, 40)))

    def test_sequence_gap_and_equal_count_id_replacement_break_runs(self):
        def frame(n, ids=(100, 101), sequence="video", reason=None):
            return dict(number=n, signature=frozenset(ids), sequence=sequence, reason=reason)
        frames = [frame(0), frame(1), frame(2, (100, 102)), frame(4, (100, 102)),
                  frame(5, reason="duplicate"), frame(6), frame(7, sequence="other")]
        self.assertEqual([len(r) for r in stable_runs(frames)], [2, 1, 1, 1, 1])

    def test_numeric_identity_is_exact(self):
        self.assertEqual(parse_fish_id("9007199254740993"), 9007199254740993)
        self.assertEqual(parse_fish_id("100.0"), 100)
        self.assertIsNone(parse_fish_id("fish"))
        for name in ("100.5", "-1"):
            with self.assertRaises(DatasetError):
                parse_fish_id(name)

    def test_bad_source_boxes_references_and_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = make_source(Path(directory))
            file = source / "_annotations.coco.json"
            original = json.loads(file.read_text())
            for mode in ("duplicate_id", "nan", "negative", "bounds", "reference", "unsafe_path"):
                data = json.loads(json.dumps(original))
                if mode == "duplicate_id":
                    data["annotations"].append(data["annotations"][0])
                elif mode == "nan":
                    data["annotations"][0]["bbox"][0] = float("nan")
                elif mode == "negative":
                    data["annotations"][0]["bbox"][2] = -1
                elif mode == "bounds":
                    data["annotations"][0]["bbox"][0] = 999
                elif mode == "reference":
                    data["annotations"][0]["image_id"] = []
                else:
                    data["images"][0]["file_name"] = "../outside.jpg"
                file.write_text(json.dumps(data))
                with self.subTest(mode=mode), self.assertRaises(DatasetError):
                    read_frames(source)

    def test_duplicate_identity_marks_frame_and_breaks_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            source = make_source(Path(directory))
            file = source / "_annotations.coco.json"
            data = json.loads(file.read_text())
            extra = dict(data["annotations"][-1], id=9999)
            data["annotations"].append(extra)
            file.write_text(json.dumps(data))
            frames, _ = read_frames(source)
            self.assertEqual(frames[0]["reason"], "duplicate individual fish ID")

    def test_failed_export_does_not_leave_partial_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_source(root)
            output = root / "export"
            with patch("split_dataset.shutil.copy2", side_effect=OSError("copy failed")):
                with self.assertRaises(OSError):
                    export_chunks([source], output)
            self.assertFalse(output.exists())
            self.assertFalse(list(root.glob(".export.*")))

    def test_synthetic_example_runs_end_to_end_without_local_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_example(root / "source")
            output = root / "dataset"
            manifest = export_chunks([source], output)
            self.assertNotIn(str(root), json.dumps(manifest))
            for split in ("train", "validate", "test"):
                dataset = PredictionDataset(output, split)
                self.assertEqual(len(dataset), 21)
                self.assertEqual(dataset[0]["track_ids"], [100, 101])
                self.assertEqual(dataset[0]["observed_frames"], list(range(10)))
                self.assertEqual(dataset[-1]["future_frames"], list(range(30, 40)))

    def test_invalid_midpoints_and_split_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_source(root)
            output = root / "export"
            manifest = export_chunks([source], output)
            file = output / manifest["chunks"][0]["path"] / "annotations.json"
            records = json.loads(file.read_text())
            records[0]["center"]["x"] += 10
            file.write_text(json.dumps(records))
            with self.assertRaisesRegex(DatasetError, "midpoint"):
                PredictionDataset(output, "train")
            manifest["chunks"][1]["split"] = "test"
            (output / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(DatasetError, "does not match"):
                PredictionDataset(output, "test")


if __name__ == "__main__":
    unittest.main()
