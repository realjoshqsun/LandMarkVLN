from __future__ import annotations

import unittest

import numpy as np
from scipy import stats

from landmarkvln.statistics import (
    DEFAULT_BOOTSTRAP_REPETITIONS,
    DEFAULT_RANDOMIZATION_REPETITIONS,
    DEFAULT_SPEARMAN_BOOTSTRAP_REPETITIONS,
    draw_trajectory_weights,
    holm_adjust,
    matched_effect,
    risk_difference,
    spearman_correlation,
    trajectory_bootstrap_mean,
    trajectory_sign_randomization,
    vxa_association,
)


def _indices_from_weights(
    rows: list[np.ndarray], weights: np.ndarray
) -> np.ndarray:
    return np.concatenate(
        [
            np.tile(rows[index], int(count))
            for index, count in enumerate(weights)
            if count
        ]
    )


class StatisticsTest(unittest.TestCase):
    def test_protocol_defaults(self) -> None:
        self.assertEqual(DEFAULT_BOOTSTRAP_REPETITIONS, 5_000)
        self.assertEqual(DEFAULT_SPEARMAN_BOOTSTRAP_REPETITIONS, 2_000)
        self.assertEqual(DEFAULT_RANDOMIZATION_REPETITIONS, 10_000)

    def test_trajectory_weights_are_seeded_multinomial_counts(self) -> None:
        expected_rng = np.random.default_rng(17)
        expected = expected_rng.multinomial(3, np.full(3, 1.0 / 3.0), size=8)
        actual = draw_trajectory_weights(3, 8, 17)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual.sum(axis=1), np.full(8, 3))

    def test_grouped_mean_bootstrap_retains_whole_trajectories(self) -> None:
        values = np.asarray([0.0, 0.0, 9.0])
        trajectories = ["long", "long", "short"]
        repetitions = 40
        seed = 7

        result = trajectory_bootstrap_mean(
            values, trajectories, repetitions=repetitions, random_state=seed
        )

        reference_rng = np.random.default_rng(seed)
        rows = [np.asarray([0, 1]), np.asarray([2])]
        weights = reference_rng.multinomial(
            2, np.full(2, 0.5), size=repetitions
        )
        draws = [
            float(np.mean(values[_indices_from_weights(rows, row)]))
            for row in weights
        ]
        expected_ci = np.quantile(draws, [0.025, 0.975])

        self.assertEqual(result.estimate, 3.0)
        np.testing.assert_allclose(result.ci95, expected_ci)
        self.assertEqual(result.trajectory_count, 2)
        self.assertEqual(result.valid_repetitions, repetitions)

    def test_vxa_uses_grouped_ci_but_sample_level_fisher(self) -> None:
        action = np.asarray([1, 1, 1, 0, 0, 0], dtype=bool)
        localized = np.asarray([1, 1, 0, 1, 0, 0], dtype=bool)
        trajectories = ["a", "a", "b", "b", "c", "c"]
        repetitions = 80
        seed = 13

        result = vxa_association(
            action,
            localized,
            trajectories,
            repetitions=repetitions,
            random_state=seed,
        )

        self.assertEqual(result.table, ((2, 1), (1, 2)))
        self.assertAlmostEqual(result.risk_difference.estimate, 1.0 / 3.0)
        fisher = stats.fisher_exact(result.table, alternative="two-sided")
        self.assertEqual(result.fisher_odds_ratio, float(fisher.statistic))
        self.assertEqual(result.fisher_two_sided_p, float(fisher.pvalue))

        reference_rng = np.random.default_rng(seed)
        rows = [np.asarray([0, 1]), np.asarray([2, 3]), np.asarray([4, 5])]
        weights = reference_rng.multinomial(
            3, np.full(3, 1.0 / 3.0), size=repetitions
        )
        draws = []
        for row in weights:
            indices = _indices_from_weights(rows, row)
            if np.any(action[indices]) and np.any(~action[indices]):
                draws.append(risk_difference(action[indices], localized[indices]))
        expected_ci = np.quantile(draws, [0.025, 0.975])
        np.testing.assert_allclose(result.risk_difference.ci95, expected_ci)
        self.assertEqual(result.risk_difference.valid_repetitions, len(draws))

    def test_holm_adjustment_is_monotone_in_rank(self) -> None:
        adjusted = holm_adjust({"first": 0.01, "third": 0.04, "second": 0.03})
        self.assertEqual(adjusted, {"first": 0.03, "second": 0.06, "third": 0.06})
        self.assertEqual(holm_adjust({}), {})

    def test_sign_randomization_flips_trajectory_sums(self) -> None:
        effects = np.asarray([1.0, -0.5, 2.0, 0.0])
        trajectories = ["a", "a", "b", "c"]
        repetitions = 25
        seed = 11

        result = trajectory_sign_randomization(
            effects,
            trajectories,
            repetitions=repetitions,
            random_state=seed,
        )

        group_sums = np.asarray([0.5, 2.0, 0.0])
        reference_rng = np.random.default_rng(seed)
        signs = reference_rng.integers(
            0, 2, size=(repetitions, 3), dtype=np.int8
        )
        signs = signs.astype(np.float64) * 2.0 - 1.0
        null = (signs @ group_sums) / len(effects)
        expected = (1 + np.sum(null >= np.mean(effects))) / (repetitions + 1)

        self.assertAlmostEqual(result.observed_mean, float(np.mean(effects)))
        self.assertEqual(result.p_value, expected)
        self.assertEqual(result.trajectory_count, 3)

    def test_spearman_bootstrap_resamples_trajectories(self) -> None:
        left = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        right = np.asarray([0.0, 2.0, 1.0, 3.0, 5.0, 4.0])
        trajectories = ["a", "a", "b", "b", "c", "c"]
        repetitions = 30
        seed = 19

        result = spearman_correlation(
            left,
            right,
            trajectories,
            repetitions=repetitions,
            random_state=seed,
        )

        observed = stats.spearmanr(left, right, alternative="two-sided")
        self.assertEqual(result.rho, float(observed.statistic))
        self.assertEqual(result.two_sided_p, float(observed.pvalue))

        reference_rng = np.random.default_rng(seed)
        rows = [np.asarray([0, 1]), np.asarray([2, 3]), np.asarray([4, 5])]
        draws = []
        for _ in range(repetitions):
            selected = reference_rng.integers(0, 3, size=3)
            indices = np.concatenate([rows[index] for index in selected])
            draws.append(float(stats.spearmanr(left[indices], right[indices]).statistic))
        expected_ci = np.quantile(draws, [0.025, 0.975])
        np.testing.assert_allclose(result.ci95, expected_ci)
        self.assertEqual(result.valid_repetitions, repetitions)

    def test_matched_effect_runs_both_registered_procedures(self) -> None:
        result = matched_effect(
            [1.0, -0.25, 0.5],
            ["a", "a", "b"],
            bootstrap_repetitions=10,
            randomization_repetitions=12,
            random_state=5,
        )
        self.assertAlmostEqual(result.mean_effect.estimate, 1.25 / 3.0)
        self.assertEqual(result.mean_effect.repetitions, 10)
        self.assertEqual(result.sign_randomization.repetitions, 12)

    def test_undefined_and_invalid_inputs_fail_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "both A=1 and A=0"):
            risk_difference([1, 1], [0, 1])
        with self.assertRaisesRegex(ValueError, "finite"):
            trajectory_bootstrap_mean([1.0, np.nan], ["a", "b"], repetitions=2)
        with self.assertRaisesRegex(ValueError, "two distinct"):
            spearman_correlation(
                [1.0, 1.0, 1.0],
                [1.0, 2.0, 3.0],
                ["a", "b", "c"],
                repetitions=2,
            )
        with self.assertRaisesRegex(ValueError, "invalid p-value"):
            holm_adjust({"bad": 1.1})


if __name__ == "__main__":
    unittest.main()
