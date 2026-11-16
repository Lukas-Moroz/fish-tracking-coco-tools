import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from dataset_common import DatasetError, EXTRA_TEST_SECTION, SPLIT_NAMES, photo_name
from prediction_format import standardize_predictions
from prediction_loader import PredictionDataset, predict_sample
from tests.test_public_pool import make_package


class SupplementalReserveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.previous = self.root / "private"
        self.prior = make_package(
            self.previous, "private", [("train", 30), ("validate", 30),
                                       ("test", 30), ("extra", 22), ("extra", 5)])

    def reserve(self):
        from release_dataset import reserve_extra_chunks

        output = self.root / "reserved"
        manifest = reserve_extra_chunks(self.previous, output, "reserved", ["0104"])
        return output, manifest

    def test_supplemental_holdout_uses_code_three_without_adding_an_official_split(self):
        self.assertEqual(SPLIT_NAMES, ("train", "validate", "test"))
        self.assertEqual(EXTRA_TEST_SECTION, "extra_test")
        self.assertEqual(photo_name(1, 4, EXTRA_TEST_SECTION, 1, ".jpg"), "01043001.jpg")
        reserved, manifest = self.reserve()
        held = next(c for c in manifest["chunks"] if c["chunk_id"] == "0104")
        original = self.prior["chunks"][3]
        self.assertEqual(held["split"], EXTRA_TEST_SECTION)
        self.assertEqual(held["path"], "extra_test/0104")
        self.assertEqual(held["track_ids"], original["track_ids"])
        self.assertEqual(held["source_track_id_map"], original["source_track_id_map"])
        self.assertEqual(held["frame_count"], 22)
        self.assertFalse((reserved / "extra/0104").exists())
        before = json.loads((self.previous / original["path"] / "annotations.json").read_text())
        after = json.loads((reserved / held["path"] / "annotations.json").read_text())
        for a, b in zip(before, after):
            self.assertEqual({k: v for k, v in a.items() if k != "image"},
                             {k: v for k, v in b.items() if k != "image"})
            self.assertEqual(b["image"][4], "3")
            self.assertEqual((self.previous / original["path"] / "images" / a["image"]).read_bytes(),
                             (reserved / held["path"] / "images" / b["image"]).read_bytes())
        for entry in self.prior["chunks"][:3]:
            self.assertEqual(next(c for c in manifest["chunks"] if c["chunk_id"] == entry["chunk_id"]), entry)
            for path in (self.previous / entry["path"]).rglob("*"):
                if path.is_file():
                    self.assertEqual(path.read_bytes(), (reserved / path.relative_to(self.previous)).read_bytes())
        for split in SPLIT_NAMES:
            self.assertEqual(len(PredictionDataset(reserved, split)), 11)

    def test_only_private_organizer_data_can_load_and_standardize_extra_test(self):
        reserved, manifest = self.reserve()
        dataset = PredictionDataset(reserved, EXTRA_TEST_SECTION)
        self.assertEqual(len(dataset), 3)
        self.assertEqual({s["chunk_id"] for s in dataset}, {"0104"})
        self.assertEqual(dataset[0]["observed_frames"], list(range(1, 11)))
        self.assertEqual(dataset[-1]["future_frames"], list(range(13, 23)))
        records = [predict_sample(dataset, index, lambda images: [[[0, 0]] * 10])
                   for index in range(len(dataset))]
        result = standardize_predictions(records, dataset)
        self.assertEqual(result["split"], EXTRA_TEST_SECTION)
        self.assertEqual(result["release_id"], "reserved")
        source = self.root / "raw.json"
        target = self.root / "standardized.json"
        source.write_text(json.dumps(records))
        completed = subprocess.run(
            [sys.executable, "prediction_format.py", str(source), "--dataset", str(reserved),
             "--split", EXTRA_TEST_SECTION, "--output", str(target)],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(target.read_text()), result)

    def test_all_public_and_supplement_only_exports_exclude_both_private_holdouts(self):
        from release_dataset import export_participant_package, export_public_pool

        reserved, _ = self.reserve()
        all_public = self.root / "public"
        public = export_public_pool(reserved, all_public)
        self.assertEqual(sum(c["frame_count"] for c in public["chunks"]), 65)
        self.assertEqual({c["chunk_id"] for c in public["chunks"]}, {"0101", "0102", "0105"})
        self.assertNotIn("extra_test_counts", public)
        self.assertFalse((all_public / "extra_test").exists())
        self.assertFalse((all_public / "test").exists())
        for section in ("test", EXTRA_TEST_SECTION):
            with self.subTest(section=section), self.assertRaisesRegex(DatasetError, "withheld"):
                PredictionDataset(all_public, section)
        extras_only = self.root / "public_extras"
        extras = export_public_pool(reserved, extras_only, sections=("extra",))
        self.assertEqual([c["chunk_id"] for c in extras["chunks"]], ["0105"])
        self.assertEqual(extras["chunks"][0]["frame_count"], 5)
        self.assertTrue(all(f["image"][4] == "4" for f in extras["chunks"][0]["frames"]))
        self.assertEqual(len(PredictionDataset(extras_only, "public")), 0)
        legacy = self.root / "participants"
        manifest = export_participant_package(reserved, legacy)
        self.assertEqual({c["chunk_id"] for c in manifest["chunks"]}, {"0101", "0102", "0105"})
        self.assertNotIn("extra_test_counts", manifest)
        with self.assertRaisesRegex(DatasetError, "withheld"):
            PredictionDataset(legacy, EXTRA_TEST_SECTION)

    def test_reserved_short_chunk_annotations_are_validated_even_for_training(self):
        reserved, manifest = self.reserve()
        entry = next(c for c in manifest["chunks"] if c["split"] == EXTRA_TEST_SECTION)
        path = reserved / entry["path"] / "annotations.json"
        records = json.loads(path.read_text())
        records[0]["center"]["x"] = 999
        path.write_text(json.dumps(records))
        with self.assertRaisesRegex(DatasetError, "midpoint"):
            PredictionDataset(reserved, "train")

    def test_reserved_source_frames_cannot_overlap_an_official_split(self):
        reserved, manifest = self.reserve()
        entry = next(c for c in manifest["chunks"] if c["split"] == EXTRA_TEST_SECTION)
        for i, frame in enumerate(entry["frames"]):
            frame["source_number"] = i
        (reserved / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(DatasetError, "Frame reused"):
            PredictionDataset(reserved, "train")

    def test_core_and_unknown_chunk_ids_cannot_be_reserved(self):
        from release_dataset import reserve_extra_chunks

        for selection in (["0101"], ["0103"], ["9999"]):
            output = self.root / f"invalid_{selection[0]}"
            with self.subTest(selection=selection), self.assertRaises(DatasetError):
                reserve_extra_chunks(self.previous, output, "invalid", selection)
            self.assertFalse(output.exists())

    def test_legacy_schema_cannot_be_used_as_a_private_supplemental_holdout(self):
        folder = self.root / "legacy"
        folder.mkdir()
        (folder / "manifest.json").write_text(json.dumps({"schema_version": 1, "fps": 30, "chunks": []}))
        with self.assertRaisesRegex(DatasetError, "private organizer"):
            PredictionDataset(folder, EXTRA_TEST_SECTION)


if __name__ == "__main__":
    unittest.main()
