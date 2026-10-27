"""Statistical procedures used by the LandMarkVLN paper tables.

All uncertainty estimates resample source trajectories and retain every
sample belonging to a selected trajectory. A seeded NumPy generator may be
shared across calls when an analysis needs to reproduce a registered panel
order.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats


DEFAULT_BOOTSTRAP_REPETITIONS = 5_000
DEFAULT_SPEARMAN_BOOTSTRAP_REPETITIONS = 2_000
DEFAULT_RANDOMIZATION_REPETITIONS = 10_000
DEFAULT_RANDOM_SEED = 20_260_808

RandomState = int | np.random.Generator
Key = TypeVar("Key", bound=Hashable)


@dataclass(frozen=True)
class BootstrapEstimate:
    estimate: float
    ci95: tuple[float, float]
    sample_count: int
    trajectory_count: int
    repetitions: int
    valid_repetitions: int


@dataclass(frozen=True)
class VxAResult:
    """Binary visual-localization by action-diagnostic association.

    The table has rows A=1, A=0 and columns V=1, V=0.
    """

    table: tuple[tuple[int, int], tuple[int, int]]
    localization_rate: float
    localization_rate_given_action_true: float
    localization_rate_given_action_false: float
    risk_difference: BootstrapEstimate
    fisher_odds_ratio: float
    fisher_two_sided_p: float


@dataclass(frozen=True)
class RandomizationResult:
    observed_mean: float
    p_value: float
    sample_count: int
    trajectory_count: int
    repetitions: int


@dataclass(frozen=True)
class MatchedEffectResult:
    mean_effect: BootstrapEstimate
    sign_randomization: RandomizationResult


@dataclass(frozen=True)
class SpearmanResult:
    rho: float
    two_sided_p: float
    ci95: tuple[float, float]
    sample_count: int
    trajectory_count: int
    repetitions: int
    valid_repetitions: int


def _generator(random_state: RandomState) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, (int, np.integer)) and not isinstance(
        random_state, (bool, np.bool_)
    ):
        return np.random.default_rng(int(random_state))
    raise TypeError("random_state must be an integer seed or numpy Generator")


def _positive_integer(value: int, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _continuous(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(array) == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _binary(values: ArrayLike, name: str) -> NDArray[np.bool_]:
    array = _continuous(values, name)
    if not np.all((array == 0.0) | (array == 1.0)):
        raise ValueError(f"{name} must contain only 0/1 or boolean values")
    return array.astype(np.bool_)


def _trajectory_rows(
    trajectory_ids: Sequence[Hashable], sample_count: int
) -> list[NDArray[np.int64]]:
    identifiers = list(trajectory_ids)
    if len(identifiers) != sample_count:
        raise ValueError("trajectory_ids must have the same length as the samples")

    grouped: dict[Hashable, list[int]] = {}
    for index, identifier in enumerate(identifiers):
        if identifier is None:
            raise ValueError("trajectory_ids must not contain missing values")
        if isinstance(identifier, (float, np.floating)) and not np.isfinite(identifier):
            raise ValueError("trajectory_ids must not contain missing values")
        try:
            hash(identifier)
        except TypeError as error:
            raise TypeError("each trajectory ID must be hashable") from error
        grouped.setdefault(identifier, []).append(index)

    ordered_ids = sorted(grouped, key=lambda value: (type(value).__name__, repr(value)))
    return [np.asarray(grouped[value], dtype=np.int64) for value in ordered_ids]


def draw_trajectory_weights(
    trajectory_count: int,
    repetitions: int,
    random_state: RandomState = DEFAULT_RANDOM_SEED,
) -> NDArray[np.int64]:
    """Draw multinomial trajectory counts for bootstrap replicates."""

    trajectory_count = _positive_integer(trajectory_count, "trajectory_count")
    repetitions = _positive_integer(repetitions, "repetitions")
    rng = _generator(random_state)
    probabilities = np.full(trajectory_count, 1.0 / trajectory_count)
    return rng.multinomial(trajectory_count, probabilities, size=repetitions)


def _weights(
    trajectory_count: int,
    repetitions: int,
    random_state: RandomState,
    supplied: ArrayLike | None,
) -> NDArray[np.int64]:
    if supplied is None:
        return draw_trajectory_weights(trajectory_count, repetitions, random_state)
    weights = np.asarray(supplied, dtype=np.int64)
    if weights.shape != (repetitions, trajectory_count):
        raise ValueError(
            "bootstrap_weights must have shape "
            f"({repetitions}, {trajectory_count})"
        )
    return weights


def _percentile_ci(draws: NDArray[np.float64]) -> tuple[tuple[float, float], int]:
    finite = draws[np.isfinite(draws)]
    if len(finite) == 0:
        raise ValueError("the statistic was undefined in every bootstrap replicate")
    low, high = np.quantile(finite, [0.025, 0.975])
    return (float(low), float(high)), int(len(finite))


def trajectory_bootstrap_mean(
    values: ArrayLike,
    trajectory_ids: Sequence[Hashable],
    *,
    repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    random_state: RandomState = DEFAULT_RANDOM_SEED,
    bootstrap_weights: ArrayLike | None = None,
) -> BootstrapEstimate:
    """Mean and 95% percentile CI from source-trajectory resampling."""

    sample_values = _continuous(values, "values")
    rows = _trajectory_rows(trajectory_ids, len(sample_values))
    repetitions = _positive_integer(repetitions, "repetitions")
    weights = _weights(len(rows), repetitions, random_state, bootstrap_weights)
    group_sums = np.asarray([sample_values[index].sum() for index in rows])
    group_counts = np.asarray([len(index) for index in rows], dtype=np.int64)
    draws = (weights @ group_sums) / (weights @ group_counts)
    ci95, valid_repetitions = _percentile_ci(draws)
    return BootstrapEstimate(
        estimate=float(np.mean(sample_values)),
        ci95=ci95,
        sample_count=len(sample_values),
        trajectory_count=len(rows),
        repetitions=repetitions,
        valid_repetitions=valid_repetitions,
    )


def _vxa_cells(
    action_values: NDArray[np.bool_], localized: NDArray[np.bool_]
) -> NDArray[np.int64]:
    return np.asarray(
        [
            np.sum(action_values & localized),
            np.sum(action_values & ~localized),
            np.sum(~action_values & localized),
            np.sum(~action_values & ~localized),
        ],
        dtype=np.int64,
    )


def _risk_difference_from_cells(cells: NDArray[np.int64]) -> float:
    a1_v1, a1_v0, a0_v1, a0_v0 = cells
    a1_count = int(a1_v1 + a1_v0)
    a0_count = int(a0_v1 + a0_v0)
    if a1_count == 0 or a0_count == 0:
        return float("nan")
    return float(a1_v1 / a1_count - a0_v1 / a0_count)


def risk_difference(action_values: ArrayLike, localized: ArrayLike) -> float:
    """Return P(V=1 | A=1) - P(V=1 | A=0) at sample level."""

    action = _binary(action_values, "action_values")
    visual = _binary(localized, "localized")
    if len(action) != len(visual):
        raise ValueError("action_values and localized must have the same length")
    result = _risk_difference_from_cells(_vxa_cells(action, visual))
    if not np.isfinite(result):
        raise ValueError("risk difference requires samples with both A=1 and A=0")
    return result


def vxa_association(
    action_values: ArrayLike,
    localized: ArrayLike,
    trajectory_ids: Sequence[Hashable],
    *,
    repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    random_state: RandomState = DEFAULT_RANDOM_SEED,
    bootstrap_weights: ArrayLike | None = None,
) -> VxAResult:
    """Compute VxA risk difference, grouped CI, and sample-level Fisher test."""

    action = _binary(action_values, "action_values")
    visual = _binary(localized, "localized")
    if len(action) != len(visual):
        raise ValueError("action_values and localized must have the same length")
    rows = _trajectory_rows(trajectory_ids, len(action))
    repetitions = _positive_integer(repetitions, "repetitions")
    weights = _weights(len(rows), repetitions, random_state, bootstrap_weights)

    cells = _vxa_cells(action, visual)
    observed = _risk_difference_from_cells(cells)
    if not np.isfinite(observed):
        raise ValueError("VxA analysis requires samples with both A=1 and A=0")
    group_cells = np.asarray([_vxa_cells(action[index], visual[index]) for index in rows])
    sampled_cells = weights @ group_cells
    draws = np.asarray(
        [_risk_difference_from_cells(sampled) for sampled in sampled_cells],
        dtype=np.float64,
    )
    ci95, valid_repetitions = _percentile_ci(draws)

    table = ((int(cells[0]), int(cells[1])), (int(cells[2]), int(cells[3])))
    fisher = stats.fisher_exact(table, alternative="two-sided")
    a1_count = int(cells[0] + cells[1])
    a0_count = int(cells[2] + cells[3])
    estimate = BootstrapEstimate(
        estimate=observed,
        ci95=ci95,
        sample_count=len(action),
        trajectory_count=len(rows),
        repetitions=repetitions,
        valid_repetitions=valid_repetitions,
    )
    return VxAResult(
        table=table,
        localization_rate=float(np.mean(visual)),
        localization_rate_given_action_true=float(cells[0] / a1_count),
        localization_rate_given_action_false=float(cells[2] / a0_count),
        risk_difference=estimate,
        fisher_odds_ratio=float(fisher.statistic),
        fisher_two_sided_p=float(fisher.pvalue),
    )


def holm_adjust(p_values: Mapping[Key, float]) -> dict[Key, float]:
    """Return Holm-adjusted p-values for one explicitly supplied family."""

    checked: list[tuple[Key, float]] = []
    for key, raw_value in p_values.items():
        value = float(raw_value)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid p-value for {key!r}: {raw_value!r}")
        checked.append((key, value))
    checked.sort(key=lambda item: item[1])
    adjusted: dict[Key, float] = {}
    running_maximum = 0.0
    family_size = len(checked)
    for rank, (key, value) in enumerate(checked):
        running_maximum = max(running_maximum, (family_size - rank) * value)
        adjusted[key] = min(1.0, running_maximum)
    return adjusted


def trajectory_sign_randomization(
    effects: ArrayLike,
    trajectory_ids: Sequence[Hashable],
    *,
    repetitions: int = DEFAULT_RANDOMIZATION_REPETITIONS,
    random_state: RandomState = DEFAULT_RANDOM_SEED,
) -> RandomizationResult:
    """Test a positive mean using one random sign per trajectory."""

    sample_effects = _continuous(effects, "effects")
    rows = _trajectory_rows(trajectory_ids, len(sample_effects))
    repetitions = _positive_integer(repetitions, "repetitions")
    rng = _generator(random_state)
    group_sums = np.asarray([sample_effects[index].sum() for index in rows])
    observed = float(np.mean(sample_effects))
    signs = rng.integers(0, 2, size=(repetitions, len(rows)), dtype=np.int8)
    signs = signs.astype(np.float64) * 2.0 - 1.0
    randomized = (signs @ group_sums) / len(sample_effects)
    exceedances = int(np.sum(randomized >= observed))
    return RandomizationResult(
        observed_mean=observed,
        p_value=(1.0 + exceedances) / (repetitions + 1.0),
        sample_count=len(sample_effects),
        trajectory_count=len(rows),
        repetitions=repetitions,
    )


def matched_effect(
    effects: ArrayLike,
    trajectory_ids: Sequence[Hashable],
    *,
    bootstrap_repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    randomization_repetitions: int = DEFAULT_RANDOMIZATION_REPETITIONS,
    random_state: RandomState = DEFAULT_RANDOM_SEED,
    bootstrap_weights: ArrayLike | None = None,
) -> MatchedEffectResult:
    """Compute a matched mean-effect interval and one-sided sign test."""

    rng = _generator(random_state)
    mean_effect = trajectory_bootstrap_mean(
        effects,
        trajectory_ids,
        repetitions=bootstrap_repetitions,
        random_state=rng,
        bootstrap_weights=bootstrap_weights,
    )
    randomization = trajectory_sign_randomization(
        effects,
        trajectory_ids,
        repetitions=randomization_repetitions,
        random_state=rng,
    )
    return MatchedEffectResult(mean_effect, randomization)


def _spearman(
    left: NDArray[np.float64], right: NDArray[np.float64]
) -> tuple[float, float] | None:
    if len(left) < 3 or len(np.unique(left)) < 2 or len(np.unique(right)) < 2:
        return None
    result = stats.spearmanr(left, right, alternative="two-sided")
    rho = float(result.statistic)
    p_value = float(result.pvalue)
    if not np.isfinite(rho) or not np.isfinite(p_value):
        return None
    return rho, p_value


def spearman_correlation(
    left: ArrayLike,
    right: ArrayLike,
    trajectory_ids: Sequence[Hashable],
    *,
    repetitions: int = DEFAULT_SPEARMAN_BOOTSTRAP_REPETITIONS,
    random_state: RandomState = DEFAULT_RANDOM_SEED,
) -> SpearmanResult:
    """Spearman rho, two-sided p-value, and trajectory-bootstrap CI."""

    left_values = _continuous(left, "left")
    right_values = _continuous(right, "right")
    if len(left_values) != len(right_values):
        raise ValueError("left and right must have the same length")
    rows = _trajectory_rows(trajectory_ids, len(left_values))
    repetitions = _positive_integer(repetitions, "repetitions")
    rng = _generator(random_state)
    observed = _spearman(left_values, right_values)
    if observed is None:
        raise ValueError(
            "Spearman correlation requires at least three samples and two "
            "distinct values in each variable"
        )

    draws = np.full(repetitions, np.nan, dtype=np.float64)
    for draw_index in range(repetitions):
        selected = rng.integers(0, len(rows), size=len(rows))
        indices = np.concatenate([rows[index] for index in selected])
        bootstrap = _spearman(left_values[indices], right_values[indices])
        if bootstrap is not None:
            draws[draw_index] = bootstrap[0]
    ci95, valid_repetitions = _percentile_ci(draws)
    return SpearmanResult(
        rho=observed[0],
        two_sided_p=observed[1],
        ci95=ci95,
        sample_count=len(left_values),
        trajectory_count=len(rows),
        repetitions=repetitions,
        valid_repetitions=valid_repetitions,
    )
