from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from landmarkvln.causal import reproduce_causal


ROOT = Path(__file__).resolve().parents[1]


class CausalAnalysisTest(unittest.TestCase):
    def test_all_registered_primary_rows_match_paper_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rows = reproduce_causal(ROOT, Path(temporary))
        self.assertEqual(len(rows), 23)
        self.assertEqual(sum(row["model"] == "omnivla" for row in rows), 1)
        self.assertEqual(sum(row["model"] == "vamos" for row in rows), 1)


if __name__ == "__main__":
    unittest.main()

