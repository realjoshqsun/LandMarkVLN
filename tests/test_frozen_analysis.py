from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from landmarkvln.frozen import MODEL_PATHS, reproduce, validate_all


ROOT = Path(__file__).resolve().parents[1]


class FrozenAnalysisTest(unittest.TestCase):
    def test_all_models_have_identical_membership(self) -> None:
        results = validate_all(ROOT)
        self.assertEqual(len(results), 8)
        self.assertTrue(all(result.rows == 937 for result in results))
        self.assertEqual(len({result.probe_ids for result in results}), 1)

    def test_expected_model_set(self) -> None:
        expected = json.loads(
            (ROOT / "artifacts/frozen/expected_results.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(MODEL_PATHS), set(expected["models"]))

    def test_registered_point_estimates_reproduce(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            summary = reproduce(ROOT, Path(temporary))
        self.assertEqual(summary["models"], 8)
        self.assertEqual(summary["policy_instance_rows"], 7496)
        self.assertEqual(len(summary["registered_vxa"]), 20)


if __name__ == "__main__":
    unittest.main()

