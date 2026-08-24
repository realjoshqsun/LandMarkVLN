"""Regression checks against the values reported in the current paper."""

from __future__ import annotations

import json
import math
import os
import unittest
from pathlib import Path
from typing import Any

from landmarkvln.analysis import analyze
from landmarkvln.data import MODEL_SPECS


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/frozen/paper_results.json"
FULL_TEST = os.environ.get("LANDMARKVLN_FULL_TEST") == "1"


def _paper_branch(value: str) -> str | None:
    return None if value == "native" else value


def _vxa_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (str(row["model"]), str(row["estimator"]), str(row["scope"])): row
        for row in rows
    }


def _paper_model(row: dict[str, Any]) -> str:
    return MODEL_SPECS[str(row["model"])].display_name


def _intervention_index(
    rows: list[dict[str, Any]], *, generated: bool
) -> dict[tuple[str, str | None, str], dict[str, Any]]:
    if generated:
        return {
            (_paper_model(row), _paper_branch(str(row["branch"])), str(row["scope"])): row
            for row in rows
        }
    return {
        (str(row["model"]), row.get("branch"), str(row["scope"])): row
        for row in rows
    }


class PaperCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
        repetitions = (
            {}
            if FULL_TEST
            else {
                "bootstrap_repetitions": 2,
                "randomization_repetitions": 4,
                "spearman_repetitions": 2,
            }
        )
        cls.actual = analyze(ROOT, validate_inputs=False, **repetitions)

    def assert_close(self, actual: float, expected: float) -> None:
        self.assertTrue(
            math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-14),
            f"expected {expected!r}, found {actual!r}",
        )

    def test_meta_and_grid_sizes(self) -> None:
        meta = self.reference["meta"]
        self.assertEqual(self.actual["instances"], meta["samples"])
        self.assertEqual(
            self.actual["source_trajectories"], meta["source_trajectories"]
        )
        self.assertEqual(
            self.actual["policy_instance_rows"], meta["model_sample_rows"]
        )
        for reference_key, actual_key, size in (
            ("vxa", "vxa", 60),
            ("matched", "matched_intervention", 23),
            ("broad", "broad_response", 23),
            ("triangulation", "attribution_intervention", 23),
        ):
            self.assertEqual(len(self.reference[reference_key]), size)
            self.assertEqual(len(self.actual[actual_key]), size)

    def test_all_vxa_point_statistics(self) -> None:
        expected = _vxa_index(self.reference["vxa"])
        actual = {
            (_paper_model(row), row["estimator"], row["scope"]): row
            for row in self.actual["vxa"]
        }
        self.assertEqual(set(actual), set(expected))
        for key, target in expected.items():
            row = actual[key]
            with self.subTest(panel=key):
                self.assertEqual(row["n"], target["n"])
                self.assert_close(row["localization_rate"], target["localization_rate"])
                self.assert_close(row["risk_difference"], target["risk_difference"])
                self.assert_close(row["fisher_two_sided_p"], target["fisher_p"])
                self.assert_close(row["holm_p"], target["holm_p"])

    def test_all_intervention_and_association_point_statistics(self) -> None:
        matched = _intervention_index(
            self.actual["matched_intervention"], generated=True
        )
        broad = _intervention_index(self.actual["broad_response"], generated=True)
        association = _intervention_index(
            self.actual["attribution_intervention"], generated=True
        )
        expected_matched = _intervention_index(
            self.reference["matched"], generated=False
        )
        expected_broad = _intervention_index(
            self.reference["broad"], generated=False
        )
        expected_association = _intervention_index(
            self.reference["triangulation"], generated=False
        )
        self.assertEqual(set(matched), set(expected_matched))
        self.assertEqual(set(broad), set(expected_broad))
        self.assertEqual(set(association), set(expected_association))

        for key, target in expected_matched.items():
            row = matched[key]
            with self.subTest(table="matched", panel=key):
                self.assertEqual(row["n"], target["n"])
                self.assert_close(row["mean_effect"], target["mean"])
                self.assert_close(row["positive_rate"], target["positive_rate"])
        for key, target in expected_broad.items():
            row = broad[key]
            with self.subTest(table="broad", panel=key):
                self.assertEqual(row["n"], target["n"])
                self.assert_close(row["mean_response"], target["mean"])
        for key, target in expected_association.items():
            row = association[key]
            with self.subTest(table="association", panel=key):
                self.assertEqual(row["n"], target["n"])
                self.assert_close(row["spearman_rho"], target["rho"])

    @unittest.skipUnless(
        FULL_TEST,
        "set LANDMARKVLN_FULL_TEST=1 or run make verify-paper",
    )
    def test_full_resampling_values_are_exact(self) -> None:
        expected_vxa = _vxa_index(self.reference["vxa"])
        actual_vxa = {
            (_paper_model(row), row["estimator"], row["scope"]): row
            for row in self.actual["vxa"]
        }
        for key, target in expected_vxa.items():
            row = actual_vxa[key]
            with self.subTest(table="vxa", panel=key):
                self.assertEqual(
                    [row["ci95_low"], row["ci95_high"]], target["ci95"]
                )

        expected_matched = _intervention_index(
            self.reference["matched"], generated=False
        )
        actual_matched = _intervention_index(
            self.actual["matched_intervention"], generated=True
        )
        for key, target in expected_matched.items():
            row = actual_matched[key]
            with self.subTest(table="matched", panel=key):
                self.assertEqual(
                    [row["ci95_low"], row["ci95_high"]], target["ci95"]
                )
                self.assertEqual(row["holm_p"], target["randomization_holm_p"])

        expected_broad = _intervention_index(
            self.reference["broad"], generated=False
        )
        actual_broad = _intervention_index(
            self.actual["broad_response"], generated=True
        )
        for key, target in expected_broad.items():
            with self.subTest(table="broad", panel=key):
                self.assertEqual(actual_broad[key]["mean_response"], target["mean"])

        expected_association = _intervention_index(
            self.reference["triangulation"], generated=False
        )
        actual_association = _intervention_index(
            self.actual["attribution_intervention"], generated=True
        )
        for key, target in expected_association.items():
            row = actual_association[key]
            with self.subTest(table="association", panel=key):
                self.assertEqual(
                    [row["ci95_low"], row["ci95_high"]], target["ci95"]
                )


if __name__ == "__main__":
    unittest.main()
