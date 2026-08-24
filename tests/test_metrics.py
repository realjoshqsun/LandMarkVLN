from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

from landmarkvln.data import MODEL_SPECS, iter_model_rows
from landmarkvln.metrics import (
    BRANCH_SPECS,
    ESTIMATORS,
    action_value,
    attribution_cell,
    broad_effect,
    matched_effect,
)


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SCOPES = {
    "onevla": ("C", "M", "CM"),
    "internvla_n1": ("C", "M", "CM"),
    "omnivla": ("C",),
    "uni_navid": ("C", "M", "CM"),
    "fantasyvln": ("C", "M", "CM"),
    "streamvln": ("C", "M", "CM"),
    "vamos": ("C",),
    "tic_vla": ("C", "M", "CM"),
}

EXPECTED_MATCHED_COUNTS = {
    ("onevla", "native", "C"): 7_228,
    ("onevla", "native", "M"): 7_228,
    ("onevla", "native", "CM"): 7_228,
    ("internvla_n1", "continuous", "C"): 4_171,
    ("internvla_n1", "continuous", "M"): 4_150,
    ("internvla_n1", "continuous", "CM"): 3_995,
    ("internvla_n1", "discrete", "C"): 2_622,
    ("internvla_n1", "discrete", "M"): 2_545,
    ("internvla_n1", "discrete", "CM"): 2_480,
    ("omnivla", "native", "C"): 7_228,
    ("uni_navid", "native", "C"): 7_228,
    ("uni_navid", "native", "M"): 7_228,
    ("uni_navid", "native", "CM"): 7_228,
    ("fantasyvln", "native", "C"): 7_225,
    ("fantasyvln", "native", "M"): 7_225,
    ("fantasyvln", "native", "CM"): 7_225,
    ("streamvln", "native", "C"): 7_228,
    ("streamvln", "native", "M"): 7_228,
    ("streamvln", "native", "CM"): 7_228,
    ("vamos", "native", "C"): 7_228,
    ("tic_vla", "native", "C"): 7_228,
    ("tic_vla", "native", "M"): 7_228,
    ("tic_vla", "native", "CM"): 7_228,
}

EXPECTED_BROAD_COUNTS = {
    **{
        key: count
        for key, count in EXPECTED_MATCHED_COUNTS.items()
        if key[0] != "internvla_n1"
    },
    ("internvla_n1", "continuous", "C"): 2_217,
    ("internvla_n1", "continuous", "M"): 2_815,
    ("internvla_n1", "continuous", "CM"): 31,
    ("internvla_n1", "discrete", "C"): 2_677,
    ("internvla_n1", "discrete", "M"): 2_606,
    ("internvla_n1", "discrete", "CM"): 2_783,
}

EXPECTED_TRIANGULATION_COUNTS = {
    **EXPECTED_MATCHED_COUNTS,
    ("fantasyvln", "native", "C"): 7_219,
    ("fantasyvln", "native", "M"): 7_219,
    ("fantasyvln", "native", "CM"): 7_219,
}


class MetricsAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vxa_counts: Counter[tuple[str, str, str]] = Counter()
        cls.matched_counts: Counter[tuple[str, str, str]] = Counter()
        cls.broad_counts: Counter[tuple[str, str, str]] = Counter()
        cls.triangulation_counts: Counter[tuple[str, str, str]] = Counter()
        cls.fantasy_action_differences = 0
        cls.fantasy_action_adapter_errors = 0

        for model, spec in MODEL_SPECS.items():
            branches = tuple(item["branch"] for item in BRANCH_SPECS[model])
            for row in iter_model_rows(ROOT, model):
                if model == "fantasyvln":
                    local_action = action_value(model, row)
                    cls.fantasy_action_adapter_errors += (
                        local_action != row["A_by_context"]["sub"]
                    )
                    cls.fantasy_action_differences += local_action != row["A"]

                for scope in spec.scopes:
                    cells = {
                        estimator: attribution_cell(model, row, estimator, scope)
                        for estimator in ESTIMATORS
                    }
                    for estimator, cell in cells.items():
                        cls.vxa_counts[(model, estimator, scope)] += cell is not None

                    for branch in branches:
                        key = (model, branch, scope)
                        matched = matched_effect(model, branch, scope, row)
                        broad = broad_effect(model, branch, scope, row)
                        cls.matched_counts[key] += matched is not None
                        cls.broad_counts[key] += broad is not None
                        cls.triangulation_counts[key] += (
                            cells["gradient_times_attention"] is not None
                            and matched is not None
                        )

    def test_all_60_vxa_panel_counts_match_the_paper(self) -> None:
        expected = {
            (model, estimator, scope): 7_222 if model == "fantasyvln" else 7_228
            for model, scopes in EXPECTED_SCOPES.items()
            for estimator in ESTIMATORS
            for scope in scopes
        }

        self.assertEqual(len(expected), 60)
        self.assertEqual(dict(self.vxa_counts), expected)

    def test_all_23_matched_intervention_counts_match_the_paper(self) -> None:
        self.assertEqual(len(EXPECTED_MATCHED_COUNTS), 23)
        self.assertEqual(dict(self.matched_counts), EXPECTED_MATCHED_COUNTS)

    def test_all_23_broad_intervention_counts_match_the_paper(self) -> None:
        self.assertEqual(len(EXPECTED_BROAD_COUNTS), 23)
        self.assertEqual(dict(self.broad_counts), EXPECTED_BROAD_COUNTS)

    def test_all_23_triangulation_counts_match_the_paper(self) -> None:
        self.assertEqual(len(EXPECTED_TRIANGULATION_COUNTS), 23)
        self.assertEqual(
            dict(self.triangulation_counts), EXPECTED_TRIANGULATION_COUNTS
        )

    def test_fantasy_uses_local_context_action_diagnostic(self) -> None:
        self.assertEqual(self.fantasy_action_adapter_errors, 0)
        self.assertEqual(self.fantasy_action_differences, 1_747)


if __name__ == "__main__":
    unittest.main()
