import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dataset_common import DatasetError, photo_name
from prediction_format import standardize_predictions
from prediction_loader import PredictionDataset, predict_sample
from release_dataset import build_release, export_participant_package
from split_dataset import export_chunks
from tests.test_dataset_utilities import make_source


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = make_source(self.root)
        self.prepared = self.root / "prepared"
        export_chunks([self.source], self.prepared)
        self.private = self.root / "private"
        self.manifest = build_release(self.prepared, self.private, "wave_001")

    def test_names_ids_alignment_and_source_preservation(self):
        all_ids = set()
        self.assertEqual([c["chunk_id"] for c in self.manifest["chunks"]], ["0101", "0102", "0103"])
        for chunk, code in zip(self.manifest["chunks"], (1, 2, 3)):
            self.assertFalse(all_ids.intersection(chunk["track_ids"]))
            all_ids.update(chunk["track_ids"])
            self.assertEqual(chunk["frames"][0]["image"], f"01{chunk['chunk_number']:02}{code}001.jpg")
            self.assertEqual(chunk["frames"][0]["number"], 1)
            mapping = {new: int(old) for old, new in chunk["source_track_id_map"].items()}
            for frame in chunk["frames"]:
                self.assertEqual((self.private / chunk["path"] / "images" / frame["image"]).read_bytes(),
                                 (self.source / frame["source_image"]).read_bytes())
            dataset = PredictionDataset(self.private, chunk["split"])
            for sample in dataset:
                for i, track in enumerate(sample["track_ids"]):
                    # Original x is source frame number + half box width.
                    offset = chunk["frames"][0]["source_number"] - 1
                    expected = [[n + offset + 5, (mapping[track] - 100) * 20 + 4]
                                for n in sample["future_frames"]]
                    self.assertEqual(sample["future"][i], expected)
        self.assertTrue((self.source / "video-0000.jpg").is_file())

    def test_participant_package_contains_no_test_data_or_identity_mapping(self):
        output = self.root / "participants"
        result = export_participant_package(self.private, output)
        self.assertFalse((output / "test").exists())
        self.assertEqual(set(result["split_counts"]), {"train", "validate"})
        self.assertEqual([c["split"] for c in result["chunks"]], ["train", "validate"])
        self.assertNotIn('"0103"', (output / "manifest.json").read_text())
        for chunk in result["chunks"]:
            self.assertNotIn("source_track_id_map", chunk)
            self.assertNotIn("source_image", chunk["frames"][0])
            self.assertEqual((output / chunk["path"] / "annotations.json").read_bytes(),
                             (self.private / chunk["path"] / "annotations.json").read_bytes())
        self.assertEqual(len(PredictionDataset(output, "train")), 21)
        with self.assertRaisesRegex(DatasetError, "withheld"):
            PredictionDataset(output, "test")

    def test_new_wave_extends_same_video_without_reassigning_previous_chunks(self):
        # The same video receives new annotated source frames in a later wave.
        wave_root = self.root / "wave2_source"
        wave_root.mkdir()
        source = make_source(wave_root, (30,))
        path = source / "_annotations.coco.json"
        data = json.loads(path.read_text())
        for image in data["images"]:
            old = image["file_name"]
            name = f"video-{image['id'] + 200:04}.jpg"
            (source / old).rename(source / name)
            image["file_name"] = name
        path.write_text(json.dumps(data))
        prepared = self.root / "wave2_prepared"
        export_chunks([source], prepared, split="train")
        output = self.root / "wave2"
        result = build_release(prepared, output, "wave_002", previous=self.private)
        self.assertEqual(result["waves"], ["wave_001", "wave_002"])
        self.assertEqual(result["chunks"][:3], self.manifest["chunks"])
        self.assertEqual(result["chunks"][3]["chunk_id"], "0104")
        self.assertEqual(result["chunks"][3]["frames"][0]["image"], "01041001.jpg")
        for path in self.private.rglob("*"):
            if path.is_file() and path.name not in ("manifest.json", "README.txt"):
                self.assertEqual(path.read_bytes(), (output / path.relative_to(self.private)).read_bytes())
        self.assertEqual(len(PredictionDataset(output, "train")), 32)

    def test_submission_is_bound_to_release_and_uses_only_ten_images(self):
        dataset = PredictionDataset(self.private, "train")
        records = [predict_sample(dataset, i, lambda images: [[[0, 0]] * 10 for _ in range(2)])
                   for i in range(len(dataset))]
        result = standardize_predictions(records, dataset)
        self.assertEqual(result["release_id"], "wave_001")
        self.assertEqual(standardize_predictions(result, dataset), result)
        result["release_id"] = "wave_002"
        with self.assertRaises(DatasetError):
            standardize_predictions(result, dataset)

    def test_new_video_gets_fresh_registry_and_fish_ids(self):
        wave_root = self.root / "other_video"
        wave_root.mkdir()
        source = make_source(wave_root, (30,))
        path = source / "_annotations.coco.json"
        data = json.loads(path.read_text())
        for image in data["images"]:
            old = image["file_name"]
            image["file_name"] = old.replace("video-", "other-")
            (source / old).rename(source / image["file_name"])
        path.write_text(json.dumps(data))
        prepared = self.root / "new_video_prepared"
        export_chunks([source], prepared, split="validate")
        result = build_release(prepared, self.root / "new_video_release", "wave_002", self.private)
        self.assertEqual(result["videos"], [{"recording_id": "video", "video_id": 1},
                                             {"recording_id": "other", "video_id": 2}])
        chunk = result["chunks"][-1]
        self.assertEqual(chunk["chunk_id"], "0201")
        self.assertEqual(chunk["frames"][0]["image"], "02012001.jpg")
        self.assertEqual(chunk["track_ids"], [201001, 201002])

    def test_test_only_videos_are_not_listed_in_participant_registry(self):
        # Each bundled synthetic scene is a separate video.
        demo = Path(__file__).resolve().parents[1] / "examples" / "dataset"
        private = self.root / "demo_private"
        build_release(demo, private, "demo")
        result = export_participant_package(private, self.root / "demo_participants")
        self.assertEqual([v["recording_id"] for v in result["videos"]], ["demo_a", "demo_b"])
        self.assertNotIn("demo_c", json.dumps(result))

    def test_failures_do_not_leave_partial_releases(self):
        with self.assertRaisesRegex(DatasetError, "already released"):
            build_release(self.prepared, self.root / "overlap", "wave_002", previous=self.private)
        self.assertFalse((self.root / "overlap").exists())
        with self.assertRaisesRegex(DatasetError, "already used"):
            build_release(self.prepared, self.root / "same_name", "wave_001", previous=self.private)
        with self.assertRaisesRegex(DatasetError, "already exists"):
            build_release(self.prepared, self.private, "other")
        with patch("release_dataset.shutil.copy2", side_effect=OSError("copy failed")):
            with self.assertRaises(OSError):
                build_release(self.prepared, self.root / "failed", "other")
        self.assertFalse((self.root / "failed").exists())
        self.assertFalse(list(self.root.glob(".failed.*")))

    def test_invalid_release_identity_names_and_source_overlap_are_rejected(self):
        original = copy.deepcopy(self.manifest)
        path = self.private / "manifest.json"
        cases = []
        data = copy.deepcopy(original)
        data["chunks"][0]["frames"][0]["image"] = "01012001.jpg"
        cases.append(data)
        data = copy.deepcopy(original)
        data["chunks"][1]["track_ids"] = data["chunks"][0]["track_ids"]
        cases.append(data)
        data = copy.deepcopy(original)
        for i, frame in enumerate(data["chunks"][1]["frames"]):
            frame["source_number"] = i
        cases.append(data)
        for data in cases:
            path.write_text(json.dumps(data))
            with self.assertRaises(DatasetError):
                PredictionDataset(self.private, "train")
        path.write_text(json.dumps(original))

    def test_filename_capacity_and_extension_are_explicit(self):
        self.assertEqual(photo_name(99, 99, "test", 999, ".PNG"), "99993999.png")
        for args in ((100, 1, "train", 1, ".jpg"), (1, 100, "train", 1, ".jpg"),
                     (1, 1, "train", 1000, ".jpg"), (True, 1, "train", 1, ".jpg"),
                     (1, 1, "train", 0, ".jpg"), (1, 1, "train", 1, ".gif")):
            with self.assertRaises(DatasetError):
                photo_name(*args)


if __name__ == "__main__":
    unittest.main()
