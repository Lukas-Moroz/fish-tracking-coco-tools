import copy
import json
import tempfile
import unittest
from pathlib import Path

from dataset_common import DatasetError
from prediction_format import standardize_prediction_file, standardize_predictions
from prediction_loader import PredictionDataset, predict_sample


class PredictionInterfaceTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / "examples" / "dataset"
        self.dataset = PredictionDataset(root, "train")
        self.records = []
        for index in range(len(self.dataset)):
            sample = self.dataset.prediction_input(index)
            sample.pop("input_images")
            # Distinct tracks and steps catch accidental reordering.
            sample["predicted_centers"] = [
                [[track + step / 10, 25 + step] for step in range(10)]
                for track in sample["track_ids"]]
            self.records.append(sample)

    def test_predictor_receives_only_ten_images_and_output_is_aligned(self):
        expected = self.dataset.prediction_input(1)
        self.assertNotIn("observed", expected)
        self.assertNotIn("future", expected)
        self.assertEqual(self.dataset[1]["input_images"], expected["input_images"])
        calls = []

        def predictor(images):
            calls.append(images)
            return self.records[1]["predicted_centers"]

        result = predict_sample(self.dataset, 1, predictor)
        self.assertEqual(calls, [[Path(path).read_bytes() for path in expected["input_images"]]])
        self.assertEqual(len(calls[0]), 10)
        self.assertEqual(result, self.records[1])
        result["predicted_centers"][0][0][0] = -1
        self.assertNotEqual(self.records[1]["predicted_centers"][0][0][0], -1)

    def test_standardizes_order_and_roundtrips_without_changing_centers(self):
        shuffled = copy.deepcopy(self.records[::-1])
        for record in shuffled:
            record["track_ids"].reverse()
            record["predicted_centers"].reverse()
        result = standardize_predictions(shuffled, self.dataset)
        self.assertEqual(result["predictions"], self.records)
        self.assertEqual(result["coordinate_system"], "pixels")
        self.assertEqual(standardize_predictions(result, self.dataset), result)
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "raw.json", Path(directory) / "submission.json"
            source.write_text(json.dumps(shuffled))
            standardize_prediction_file(source, target, self.dataset)
            self.assertEqual(json.loads(target.read_text()), result)
            with self.assertRaises(FileExistsError):
                standardize_prediction_file(source, target, self.dataset)

    def test_invalid_predictions_cannot_reach_evaluation(self):
        def change(field, value):
            records = copy.deepcopy(self.records)
            records[0][field] = value
            return records

        centers = self.records[0]["predicted_centers"]
        cases = {
            "missing window": self.records[:-1],
            "duplicate window": self.records + self.records[:1],
            "wrong chunk": change("chunk_id", "unknown"),
            "wrong observation order": change("observed_frames", list(range(9, -1, -1))),
            "wrong future": change("future_frames", list(range(11, 21))),
            "boolean frame": change("observed_frames", [False] + list(range(1, 10))),
            "duplicate fish": change("track_ids", [100, 100]),
            "unknown fish": change("track_ids", [100, 999]),
            "missing fish": change("predicted_centers", centers[:1]),
            "short horizon": change("predicted_centers", [centers[0][:9], centers[1]]),
            "extra horizon": change("predicted_centers", [centers[0] + [[1, 2]], centers[1]]),
            "nan": change("predicted_centers", [[[float("nan"), 2]] * 10, centers[1]]),
            "infinity": change("predicted_centers", [[[1, float("inf")]] * 10, centers[1]]),
            "boolean coordinate": change("predicted_centers", [[[True, 2]] * 10, centers[1]]),
            "extra field": [dict(self.records[0], future=[]) ] + self.records[1:],
            "not records": "invalid",
        }
        for label, data in cases.items():
            with self.subTest(label=label), self.assertRaises(DatasetError):
                standardize_predictions(data, self.dataset)
        canonical = standardize_predictions(self.records, self.dataset)
        for key, value in (("split", "test"), ("fps", 29), ("schema_version", True),
                           ("coordinate_system", "normalized")):
            with self.subTest(key=key), self.assertRaises(DatasetError):
                standardize_predictions(dict(canonical, **{key: value}), self.dataset)

    def test_competition_interface_rejects_custom_horizons_and_bad_model_shape(self):
        dataset = PredictionDataset(self.dataset.root, "train", observed=9, future=11)
        with self.assertRaises(DatasetError):
            dataset.prediction_input(0)
        with self.assertRaises(DatasetError):
            standardize_predictions([], dataset)
        with self.assertRaisesRegex(DatasetError, "exactly 10"):
            predict_sample(self.dataset, 0, lambda images: [[[1, 2]] * 9] * 2)


if __name__ == "__main__":
    unittest.main()
