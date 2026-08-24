from __future__ import annotations

import unittest
from pathlib import Path

from landmarkvln.data import (
    MODEL_SPECS,
    ROWS_PER_MODEL,
    TOTAL_ROWS,
    TRAJECTORIES_PER_MODEL,
    iter_model_rows,
    validate_all,
)


ROOT = Path(__file__).resolve().parents[1]


class DataTest(unittest.TestCase):
    def test_model_specs_are_explicit_and_complete(self) -> None:
        self.assertEqual(
            tuple(MODEL_SPECS),
            (
                "onevla",
                "internvla_n1",
                "omnivla",
                "uni_navid",
                "fantasyvln",
                "streamvln",
                "vamos",
                "tic_vla",
            ),
        )
        self.assertEqual(sum(len(spec.scopes) for spec in MODEL_SPECS.values()), 20)
        for spec in MODEL_SPECS.values():
            self.assertTrue(spec.display_name)
            self.assertTrue(spec.relative_path.endswith(".jsonl.gz"))

    def test_model_rows_are_streamed_from_gzip_jsonl(self) -> None:
        first_row = next(iter_model_rows(ROOT, "onevla"))
        self.assertEqual(first_row["task_index"], 0)
        self.assertTrue(first_row["probe_id"])
        self.assertTrue(first_row["replay_id"])

    def test_frozen_files_have_shared_sample_membership(self) -> None:
        results = validate_all(ROOT)

        self.assertEqual(len(results), 8)
        self.assertEqual(sum(result.rows for result in results), TOTAL_ROWS)
        self.assertTrue(all(result.rows == ROWS_PER_MODEL for result in results))
        self.assertTrue(
            all(len(result.replay_ids) == TRAJECTORIES_PER_MODEL for result in results)
        )
        self.assertEqual(len({result.probe_ids for result in results}), 1)


if __name__ == "__main__":
    unittest.main()
