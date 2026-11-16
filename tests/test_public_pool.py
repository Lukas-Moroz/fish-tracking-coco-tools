import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from dataset_common import DatasetError, SPLIT_NAMES, photo_name
from prediction_format import standardize_predictions
from prediction_loader import PredictionDataset, predict_sample
from release_dataset import build_release
from split_dataset import export_chunks
from tests.test_dataset_utilities import make_source


def make_package(root, distribution, sections):
    """Small labeled files, including short chunks that cannot form a window."""
    manifest = {"schema_version": 2, "fps": 30, "filename_format": "VVCCSFFF",
                "track_id_scope": "globally_unique_per_chunk", "distribution": distribution,
                "release_id": "demo", "waves": ["demo"],
                "videos": [{"recording_id": "video", "video_id": 1}], "chunks": []}
    source_number = 0
    for chunk_number, (section, count) in enumerate(sections, 1):
        chunk_id = f"01{chunk_number:02d}"
        folder = root / section / chunk_id
        (folder / "images").mkdir(parents=True)
        track_id = (100 + chunk_number) * 1000 + 1
        entry = {"chunk_id": chunk_id, "recording_id": "video", "source": "video_01",
                 "video_id": 1, "chunk_number": chunk_number, "split": section,
                 "path": f"{section}/{chunk_id}", "frame_count": count,
                 "track_ids": [track_id], "wave": "demo", "frames": [],
                 "source_track_id_map": {"100": track_id}}
        records = []
        for number in range(1, count + 1):
            name = photo_name(1, chunk_number, section, number, ".jpg")
            (folder / "images" / name).write_bytes(f"image {source_number}".encode())
            entry["frames"].append({"number": number, "image": name,
                                     "source_number": source_number,
                                     "source_image": f"video-{source_number:04d}.jpg"})
            records.append({"image": name, "class_id": 0, "track_id": track_id,
                            "bbox": {"x": 10, "y": 20, "width": 8, "height": 6},
                            "center": {"x": 14, "y": 23}})
            source_number += 1
        (folder / "annotations.json").write_text(json.dumps(records))
        manifest["chunks"].append(entry)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


class PublicPoolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_code_four_does_not_add_an_official_evaluation_split(self):
        self.assertEqual(SPLIT_NAMES, ("train", "validate", "test"))
        self.assertEqual(photo_name(5, 8, "public", 1, ".jpg"), "05084001.jpg")
        self.assertEqual(photo_name(5, 8, "extra", 1, ".jpg"), "05084001.jpg")

    def test_public_windows_do_not_bridge_short_chunks_or_other_chunks(self):
        folder = self.root / "public"
        make_package(folder, "public", [("public", 23), ("public", 1),
                                       ("public", 19), ("public", 30)])
        dataset = PredictionDataset(folder, "public")
        self.assertEqual(len(dataset.chunks), 4)
        self.assertEqual(len(dataset), 4 + 11)
        self.assertEqual({s["chunk_id"] for s in dataset}, {"0101", "0104"})
        for sample in dataset:
            self.assertEqual(len(sample["input_images"]), 10)
            self.assertEqual(len(sample["future"][0]), 10)
            self.assertEqual(sample["future_frames"][0], sample["observed_frames"][-1] + 1)
            self.assertTrue(all(Path(p).name[4] == "4" for p in sample["input_images"]))
        received = []

        def predictor(images):
            received.extend(images)
            return [[[0, 0] for _ in range(10)]]

        record = predict_sample(dataset, 0, predictor)
        self.assertEqual(len(received), 10)
        self.assertTrue(all(isinstance(image, bytes) for image in received))
        self.assertEqual(record["chunk_id"], "0101")

    def test_public_prediction_json_and_both_cli_entry_points(self):
        folder = self.root / "public"
        make_package(folder, "public", [("public", 22), ("public", 2)])
        dataset = PredictionDataset(folder, "public")
        records = []
        for index in range(len(dataset)):
            inputs = dataset.prediction_input(index)
            inputs.pop("input_images")
            records.append(dict(inputs, predicted_centers=[[[1, 2]] * 10]))
        result = standardize_predictions(records, dataset)
        self.assertEqual(result["split"], "public")
        self.assertEqual(result["release_id"], "demo")
        self.assertEqual(len(result["predictions"]), 3)
        self.assertEqual(standardize_predictions(result, dataset), result)

        repo = Path(__file__).resolve().parents[1]
        source = self.root / "raw.json"
        source.write_text(json.dumps(records))
        target = self.root / "submission.json"
        windows = self.root / "windows.jsonl"
        for command in (
                ["prediction_loader.py", str(folder), "--split", "public", "--output", str(windows)],
                ["prediction_format.py", str(source), "--dataset", str(folder), "--split", "public",
                 "--output", str(target)]):
            completed = subprocess.run([sys.executable] + command, cwd=repo, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(target.read_text()), result)
        self.assertEqual(len(windows.read_text().splitlines()), 3)

    def test_private_extras_are_validated_but_do_not_supply_official_windows(self):
        folder = self.root / "private"
        make_package(folder, "private", [("train", 30), ("validate", 30),
                                        ("test", 30), ("extra", 24)])
        for split in SPLIT_NAMES:
            dataset = PredictionDataset(folder, split)
            self.assertEqual(len(dataset), 11)
            self.assertEqual({s["split"] for s in dataset}, {split})
            self.assertNotIn("0104", {s["chunk_id"] for s in dataset})
        with self.assertRaises(DatasetError):
            PredictionDataset(folder, "extra")
        annotation_file = folder / "extra/0104/annotations.json"
        records = json.loads(annotation_file.read_text())
        records[0]["center"]["x"] = 999
        annotation_file.write_text(json.dumps(records))
        with self.assertRaisesRegex(DatasetError, "midpoint"):
            PredictionDataset(folder, "train")

    def test_missing_supplemental_images_are_rejected(self):
        folder = self.root / "private"
        manifest = make_package(folder, "private", [("train", 30), ("extra", 1)])
        frame = manifest["chunks"][1]["frames"][0]
        (folder / "extra/0102/images" / frame["image"]).unlink()
        with self.assertRaisesRegex(DatasetError, "Missing photo"):
            PredictionDataset(folder, "train")

    def test_distributions_cannot_silently_mix_public_and_official_sections(self):
        public = self.root / "public"
        make_package(public, "public", [("public", 20)])
        for split in ("train", "validate"):
            with self.subTest(split=split), self.assertRaisesRegex(DatasetError, "public pool"):
                PredictionDataset(public, split)
        with self.assertRaisesRegex(DatasetError, "withheld"):
            PredictionDataset(public, "test")
        for distribution, sections, split in (
                ("private", [("train", 30), ("public", 1)], "train"),
                ("public", [("public", 30), ("train", 1)], "public"),
                ("participants", [("train", 30), ("test", 1)], "train")):
            folder = self.root / f"mixed_{distribution}"
            make_package(folder, distribution, sections)
            with self.subTest(distribution=distribution), self.assertRaisesRegex(DatasetError, "Invalid section"):
                PredictionDataset(folder, split)
        private = self.root / "normal_private"
        make_package(private, "private", [("train", 30)])
        with self.assertRaisesRegex(DatasetError, "public-pool"):
            PredictionDataset(private, "public")

    def test_short_public_chunk_labels_are_still_checked(self):
        folder = self.root / "public"
        make_package(folder, "public", [("public", 1)])
        self.assertEqual(len(PredictionDataset(folder, "public")), 0)
        path = folder / "public/0101/annotations.json"
        records = json.loads(path.read_text())
        records[0]["track_id"] = "generic fish"
        path.write_text(json.dumps(records))
        with self.assertRaises(DatasetError):
            PredictionDataset(folder, "public")

    def test_extra_source_frames_cannot_duplicate_any_official_section(self):
        folder = self.root / "private"
        manifest = make_package(folder, "private", [("train", 30), ("extra", 1)])
        manifest["chunks"][1]["frames"][0]["source_number"] = 0
        (folder / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(DatasetError, "Frame reused"):
            PredictionDataset(folder, "train")


class PublicPoolExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = make_source(self.root)
        path = self.source / "_annotations.coco.json"
        data = json.loads(path.read_text())
        # A one-frame disappearance leaves two short, previously unused runs.
        data["annotations"] = [a for a in data["annotations"]
                               if not (a["image_id"] == 35 and a["category_id"] == 2)]
        path.write_text(json.dumps(data))
        self.prepared = self.root / "prepared"
        export_chunks([self.source], self.prepared)
        self.private = self.root / "private"
        self.prior = build_release(self.prepared, self.private, "initial")

    def test_adds_all_unused_frames_and_exports_every_non_test_frame_with_code_four(self):
        from release_dataset import add_extra_release, export_public_pool

        expanded = self.root / "expanded"
        manifest = add_extra_release(self.source, expanded, "full", self.private)
        self.assertEqual(manifest["chunks"][:3], self.prior["chunks"])
        extras = [c for c in manifest["chunks"] if c["split"] == "extra"]
        self.assertEqual([c["frame_count"] for c in extras], [1, 4])
        self.assertEqual(sum(c["frame_count"] for c in manifest["chunks"]), 105)
        for split in SPLIT_NAMES:
            self.assertEqual(len(PredictionDataset(expanded, split)), len(PredictionDataset(self.private, split)))
        for entry in self.prior["chunks"]:
            for path in (self.private / entry["path"]).rglob("*"):
                if path.is_file():
                    self.assertEqual(path.read_bytes(), (expanded / path.relative_to(self.private)).read_bytes())

        output = self.root / "public"
        public = export_public_pool(expanded, output)
        self.assertEqual(public["distribution"], "public")
        self.assertEqual({c["split"] for c in public["chunks"]}, {"public"})
        self.assertEqual(sum(c["frame_count"] for c in public["chunks"]), 75)
        self.assertFalse((output / "test").exists())
        self.assertFalse((output / "train").exists())
        self.assertFalse((output / "validate").exists())
        self.assertNotIn("0103", {c["chunk_id"] for c in public["chunks"]})
        self.assertEqual(len(PredictionDataset(output, "public")), 32)
        track_ids = []
        for entry in public["chunks"]:
            self.assertNotIn("source_track_id_map", entry)
            track_ids.extend(entry["track_ids"])
            records = json.loads((output / entry["path"] / "annotations.json").read_text())
            names = {frame["image"] for frame in entry["frames"]}
            self.assertTrue(all(record["image"] in names for record in records))
            for frame in entry["frames"]:
                self.assertNotIn("source_image", frame)
                self.assertEqual(frame["image"][4], "4")
                self.assertEqual((output / entry["path"] / "images" / frame["image"]).read_bytes(),
                                 (self.source / f"video-{frame['source_number']:04d}.jpg").read_bytes())
        self.assertEqual(len(track_ids), len(set(track_ids)))

    def test_generic_unused_frame_labels_are_rejected_without_creating_an_output(self):
        from release_dataset import add_extra_release

        path = self.source / "_annotations.coco.json"
        data = json.loads(path.read_text())
        data["categories"].append({"id": 99, "name": "fish"})
        data["annotations"].append({"id": 99999, "image_id": 35, "category_id": 99,
                                    "bbox": [1, 2, 3, 4]})
        path.write_text(json.dumps(data))
        output = self.root / "bad_extra"
        with self.assertRaises(DatasetError):
            add_extra_release(self.source, output, "bad", self.private)
        self.assertFalse(output.exists())

    def test_changed_released_images_or_labels_cannot_be_silently_skipped(self):
        from release_dataset import add_extra_release

        annotation_file = self.source / "_annotations.coco.json"
        original_json = annotation_file.read_text()
        image = self.source / "video-0000.jpg"
        original_image = image.read_bytes()
        for mode in ("labels", "image"):
            annotation_file.write_text(original_json)
            image.write_bytes(original_image)
            if mode == "labels":
                data = json.loads(original_json)
                record = next(a for a in data["annotations"]
                              if a["image_id"] == 0 and a["category_id"] == 1)
                record["bbox"][0] += 1
                annotation_file.write_text(json.dumps(data))
            else:
                image.write_bytes(original_image + b" changed")
            output = self.root / f"changed_{mode}"
            with self.subTest(mode=mode), self.assertRaisesRegex(DatasetError, "previously released"):
                add_extra_release(self.source, output, "changed", self.private)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
