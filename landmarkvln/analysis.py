from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from landmarkvln.data import MODEL_SPECS, iter_model_rows, validate_all
from landmarkvln.metrics import (
    BRANCH_SPECS,
    CONTEXTS,
    ESTIMATORS,
    action_value,
    attribution_cell,
    broad_effect,
    matched_effect,
)
from landmarkvln.statistics import (
    DEFAULT_BOOTSTRAP_REPETITIONS,
    DEFAULT_RANDOMIZATION_REPETITIONS,
    DEFAULT_SPEARMAN_BOOTSTRAP_REPETITIONS,
    draw_trajectory_weights,
    holm_adjust,
    spearman_correlation,
    trajectory_bootstrap_mean,
    trajectory_sign_randomization,
    vxa_association,
)


ANALYSIS_SEEDS = {
    "onevla": 20_260_808,
    "internvla_n1": 20_260_808,
    "omnivla": 20_260_808,
    "uni_navid": 20_260_808,
    "fantasyvln": 20_260_808,
    "streamvln": 20_260_808,
    "vamos": 20_260_809,
    "tic_vla": 20_260_809,
}

PRIMARY_ESTIMATOR = "gradient_times_attention"
PRIMARY_CONTEXT = "sub"

UNI_SECONDARY_EFFECTS = (
    "target_minus_control_clean_chunk_NLL",
    "target_minus_control_categorical_KL",
    "target_minus_control_normalized_edit_distance",
)
TIC_SECONDARY_EFFECTS = (
    "raw_chunk_l2",
    "mean_position_l2",
    "first_waypoint_l2",
    "endpoint_l2",
    "max_position_l2",
)


def _trajectory_count(trajectory_ids: list[str]) -> int:
    return len(set(trajectory_ids))


def _vxa_inputs(
    model: str,
    records: list[dict[str, Any]],
    context: str,
    estimator: str,
    scope: str,
) -> tuple[list[bool], list[bool], list[str]]:
    actions: list[bool] = []
    detected: list[bool] = []
    trajectories: list[str] = []
    for record in records:
        cell = attribution_cell(model, record, estimator, scope, context)
        if cell is None:
            continue
        actions.append(action_value(model, record, context))
        detected.append(cell["detected"])
        trajectories.append(record["replay_id"])
    return actions, detected, trajectories


def _vxa_row(
    model: str,
    estimator: str,
    scope: str,
    actions: list[bool],
    detected: list[bool],
    trajectories: list[str],
    weights: np.ndarray,
    bootstrap_repetitions: int,
) -> dict[str, Any]:
    result = vxa_association(
        actions,
        detected,
        trajectories,
        repetitions=bootstrap_repetitions,
        bootstrap_weights=weights,
    )
    return {
        "model": model,
        "display_name": MODEL_SPECS[model].display_name,
        "estimator": estimator,
        "scope": scope,
        "n": result.risk_difference.sample_count,
        "trajectories": result.risk_difference.trajectory_count,
        "action_rate": sum(actions) / len(actions),
        "localization_rate": result.localization_rate,
        "risk_difference": result.risk_difference.estimate,
        "ci95_low": result.risk_difference.ci95[0],
        "ci95_high": result.risk_difference.ci95[1],
        "fisher_two_sided_p": result.fisher_two_sided_p,
        "holm_p": None,
    }


def _vxa_rows(
    model: str,
    records: list[dict[str, Any]],
    global_weights: np.ndarray,
    rng: np.random.Generator,
    bootstrap_repetitions: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    if model == "fantasyvln":
        # Context-specific validity requires per-panel trajectory sets; draw all
        # contexts to preserve the paper's RNG sequence.
        for context in CONTEXTS:
            for estimator in ESTIMATORS:
                for scope in MODEL_SPECS[model].scopes:
                    actions, detected, trajectories = _vxa_inputs(
                        model, records, context, estimator, scope
                    )
                    weights = draw_trajectory_weights(
                        _trajectory_count(trajectories),
                        bootstrap_repetitions,
                        rng,
                    )
                    if context == PRIMARY_CONTEXT:
                        output.append(
                            _vxa_row(
                                model,
                                estimator,
                                scope,
                                actions,
                                detected,
                                trajectories,
                                weights,
                                bootstrap_repetitions,
                            )
                        )
    else:
        for estimator in ESTIMATORS:
            for scope in MODEL_SPECS[model].scopes:
                actions, detected, trajectories = _vxa_inputs(
                    model, records, PRIMARY_CONTEXT, estimator, scope
                )
                output.append(
                    _vxa_row(
                        model,
                        estimator,
                        scope,
                        actions,
                        detected,
                        trajectories,
                        global_weights,
                        bootstrap_repetitions,
                    )
                )

    adjusted = holm_adjust(
        {
            (row["estimator"], row["scope"]): row["fisher_two_sided_p"]
            for row in output
        }
    )
    for row in output:
        row["holm_p"] = adjusted[(row["estimator"], row["scope"])]
    return output


def _effect_inputs(
    model: str,
    branch: str,
    scope: str,
    records: list[dict[str, Any]],
) -> tuple[list[float], list[str], list[float], list[str]]:
    effects: list[float] = []
    trajectories: list[str] = []
    broad_values: list[float] = []
    broad_trajectories: list[str] = []
    for record in records:
        effect = matched_effect(model, branch, scope, record)
        if effect is not None:
            effects.append(effect)
            trajectories.append(record["replay_id"])
        response = broad_effect(model, branch, scope, record)
        if response is not None:
            broad_values.append(response)
            broad_trajectories.append(record["replay_id"])
    return effects, trajectories, broad_values, broad_trajectories


def _effect_rows(
    model: str,
    branch_spec: dict[str, Any],
    scope: str,
    records: list[dict[str, Any]],
    weights: np.ndarray,
    rng: np.random.Generator,
    bootstrap_repetitions: int,
    randomization_repetitions: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    branch = branch_spec["branch"]
    effects, trajectories, broad_values, _ = _effect_inputs(
        model, branch, scope, records
    )
    mean_effect = trajectory_bootstrap_mean(
        effects,
        trajectories,
        repetitions=bootstrap_repetitions,
        bootstrap_weights=weights,
    )
    randomization = trajectory_sign_randomization(
        effects,
        trajectories,
        repetitions=randomization_repetitions,
        random_state=rng,
    )
    matched_row = {
        "model": model,
        "branch": branch,
        "display_name": branch_spec["display_name"],
        "native_metric": branch_spec["metric"],
        "scope": scope,
        "n": mean_effect.sample_count,
        "trajectories": mean_effect.trajectory_count,
        "mean_effect": mean_effect.estimate,
        "ci95_low": mean_effect.ci95[0],
        "ci95_high": mean_effect.ci95[1],
        "positive_rate": sum(value > 0 for value in effects) / len(effects),
        "randomization_p": randomization.p_value,
        "holm_p": None,
    }
    broad_row = {
        "model": model,
        "branch": branch,
        "display_name": branch_spec["display_name"],
        "native_metric": branch_spec["metric"],
        "scope": scope,
        "n": len(broad_values),
        "mean_response": float(np.mean(np.asarray(broad_values, dtype=np.float64))),
    }
    return matched_row, broad_row


def _secondary_values(
    model: str,
    scope: str,
    metric: str,
    records: list[dict[str, Any]],
) -> tuple[list[float], list[str]]:
    values: list[float] = []
    trajectories: list[str] = []
    for record in records:
        causal = record.get("causal")
        if causal is None:
            continue
        if model in {"uni_navid", "fantasyvln"}:
            value = causal[scope]["numeric"][metric]
        elif model == "tic_vla":
            value = causal[scope]["target_minus_control"][metric]
        else:
            raise KeyError(f"{model} has no registered secondary effect sequence")
        values.append(float(value))
        trajectories.append(record["replay_id"])
    return values, trajectories


def _consume_secondary_randomizations(
    model: str,
    scope: str,
    records: list[dict[str, Any]],
    rng: np.random.Generator,
    randomization_repetitions: int,
) -> None:
    metrics = (
        TIC_SECONDARY_EFFECTS
        if model == "tic_vla"
        else UNI_SECONDARY_EFFECTS
    )
    for metric in metrics:
        values, trajectories = _secondary_values(model, scope, metric, records)
        trajectory_sign_randomization(
            values,
            trajectories,
            repetitions=randomization_repetitions,
            random_state=rng,
        )


def _association_inputs(
    model: str,
    branch: str,
    scope: str,
    records: list[dict[str, Any]],
    context: str,
    estimator: str,
) -> tuple[list[float], list[float], list[str]]:
    attribution_scores: list[float] = []
    effects: list[float] = []
    trajectories: list[str] = []
    for record in records:
        effect = matched_effect(model, branch, scope, record)
        if effect is None:
            continue
        cell = attribution_cell(model, record, estimator, scope, context)
        if cell is None:
            continue
        attribution_scores.append(float(cell["effect_size"]))
        effects.append(effect)
        trajectories.append(record["replay_id"])
    return attribution_scores, effects, trajectories


def _association_row(
    model: str,
    branch_spec: dict[str, Any],
    scope: str,
    records: list[dict[str, Any]],
    context: str,
    estimator: str,
    rng: np.random.Generator,
    spearman_repetitions: int,
) -> dict[str, Any]:
    left, right, trajectories = _association_inputs(
        model,
        branch_spec["branch"],
        scope,
        records,
        context,
        estimator,
    )
    result = spearman_correlation(
        left,
        right,
        trajectories,
        repetitions=spearman_repetitions,
        random_state=rng,
    )
    return {
        "model": model,
        "branch": branch_spec["branch"],
        "display_name": branch_spec["display_name"],
        "native_metric": branch_spec["metric"],
        "scope": scope,
        "n": result.sample_count,
        "trajectories": result.trajectory_count,
        "spearman_rho": result.rho,
        "ci95_low": result.ci95[0],
        "ci95_high": result.ci95[1],
        "spearman_two_sided_p": result.two_sided_p,
    }


def _intervention_rows(
    model: str,
    records: list[dict[str, Any]],
    global_weights: np.ndarray,
    rng: np.random.Generator,
    bootstrap_repetitions: int,
    randomization_repetitions: int,
    spearman_repetitions: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    matched_rows: list[dict[str, Any]] = []
    broad_rows: list[dict[str, Any]] = []
    stream_associations: list[dict[str, Any]] = []

    if model == "internvla_n1":
        for branch_spec in BRANCH_SPECS[model]:
            for scope in MODEL_SPECS[model].scopes:
                _, trajectories, _, _ = _effect_inputs(
                    model, branch_spec["branch"], scope, records
                )
                local_weights = draw_trajectory_weights(
                    _trajectory_count(trajectories),
                    bootstrap_repetitions,
                    rng,
                )
                matched_row, broad_row = _effect_rows(
                    model,
                    branch_spec,
                    scope,
                    records,
                    local_weights,
                    rng,
                    bootstrap_repetitions,
                    randomization_repetitions,
                )
                matched_rows.append(matched_row)
                broad_rows.append(broad_row)
    elif model == "vamos":
        branch_spec = BRANCH_SPECS[model][0]
        scope = MODEL_SPECS[model].scopes[0]
        _, trajectories, _, broad_trajectories = _effect_inputs(
            model, branch_spec["branch"], scope, records
        )
        paired_weights = draw_trajectory_weights(
            _trajectory_count(trajectories), bootstrap_repetitions, rng
        )
        # Preserve the paper's RNG sequence before the primary sign test.
        draw_trajectory_weights(
            _trajectory_count(broad_trajectories), bootstrap_repetitions, rng
        )
        matched_row, broad_row = _effect_rows(
            model,
            branch_spec,
            scope,
            records,
            paired_weights,
            rng,
            bootstrap_repetitions,
            randomization_repetitions,
        )
        matched_rows.append(matched_row)
        broad_rows.append(broad_row)
    else:
        branch_spec = BRANCH_SPECS[model][0]
        panel_weights = global_weights
        if model == "fantasyvln":
            _, causal_trajectories, _, _ = _effect_inputs(
                model,
                branch_spec["branch"],
                MODEL_SPECS[model].scopes[0],
                records,
            )
            panel_weights = draw_trajectory_weights(
                _trajectory_count(causal_trajectories),
                bootstrap_repetitions,
                rng,
            )

        for scope in MODEL_SPECS[model].scopes:
            matched_row, broad_row = _effect_rows(
                model,
                branch_spec,
                scope,
                records,
                panel_weights,
                rng,
                bootstrap_repetitions,
                randomization_repetitions,
            )
            matched_rows.append(matched_row)
            broad_rows.append(broad_row)

            if model in {"uni_navid", "fantasyvln", "tic_vla"}:
                _consume_secondary_randomizations(
                    model,
                    scope,
                    records,
                    rng,
                    randomization_repetitions,
                )
            if model == "streamvln":
                stream_associations.append(
                    _association_row(
                        model,
                        branch_spec,
                        scope,
                        records,
                        PRIMARY_CONTEXT,
                        PRIMARY_ESTIMATOR,
                        rng,
                        spearman_repetitions,
                    )
                )

    for branch_spec in BRANCH_SPECS[model]:
        family = [
            row for row in matched_rows if row["branch"] == branch_spec["branch"]
        ]
        adjusted = holm_adjust(
            {row["scope"]: row["randomization_p"] for row in family}
        )
        for row in family:
            row["holm_p"] = adjusted[row["scope"]]

    return matched_rows, broad_rows, stream_associations


def _association_rows(
    model: str,
    records: list[dict[str, Any]],
    spearman_repetitions: int,
) -> list[dict[str, Any]]:
    if model == "streamvln":
        return []

    rng = np.random.default_rng(ANALYSIS_SEEDS[model] + 1)
    output: list[dict[str, Any]] = []

    def run(
        branch_spec: dict[str, Any],
        context: str,
        estimator: str,
        scope: str,
        publish: bool,
    ) -> None:
        row = _association_row(
            model,
            branch_spec,
            scope,
            records,
            context,
            estimator,
            rng,
            spearman_repetitions,
        )
        if publish:
            output.append(row)

    if model == "onevla":
        branch_spec = BRANCH_SPECS[model][0]
        for scope in MODEL_SPECS[model].scopes:
            run(
                branch_spec,
                PRIMARY_CONTEXT,
                PRIMARY_ESTIMATOR,
                scope,
                True,
            )
    elif model == "internvla_n1":
        for branch_spec in BRANCH_SPECS[model]:
            for scope in MODEL_SPECS[model].scopes:
                run(
                    branch_spec,
                    PRIMARY_CONTEXT,
                    PRIMARY_ESTIMATOR,
                    scope,
                    True,
                )
    else:
        branch_spec = BRANCH_SPECS[model][0]
        for context in CONTEXTS:
            for estimator in ESTIMATORS:
                for scope in MODEL_SPECS[model].scopes:
                    publish = (
                        context == PRIMARY_CONTEXT
                        and estimator == PRIMARY_ESTIMATOR
                    )
                    run(branch_spec, context, estimator, scope, publish)
    return output


def analyze(
    root: Path,
    *,
    bootstrap_repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    randomization_repetitions: int = DEFAULT_RANDOMIZATION_REPETITIONS,
    spearman_repetitions: int = DEFAULT_SPEARMAN_BOOTSTRAP_REPETITIONS,
    validate_inputs: bool = True,
) -> dict[str, Any]:
    if validate_inputs:
        validate_all(root)

    policies: list[dict[str, Any]] = []
    vxa_rows: list[dict[str, Any]] = []
    matched_rows: list[dict[str, Any]] = []
    broad_rows: list[dict[str, Any]] = []
    association_rows: list[dict[str, Any]] = []

    for model, model_spec in MODEL_SPECS.items():
        records = list(iter_model_rows(root, model))
        actions = [action_value(model, record) for record in records]
        trajectories = [record["replay_id"] for record in records]
        policies.append(
            {
                "model": model,
                "display_name": model_spec.display_name,
                "instances": len(records),
                "trajectories": _trajectory_count(trajectories),
                "action_true": sum(actions),
                "action_rate": sum(actions) / len(actions),
            }
        )

        rng = np.random.default_rng(ANALYSIS_SEEDS[model])
        global_weights = draw_trajectory_weights(
            _trajectory_count(trajectories),
            bootstrap_repetitions,
            rng,
        )
        vxa_rows.extend(
            _vxa_rows(
                model,
                records,
                global_weights,
                rng,
                bootstrap_repetitions,
            )
        )
        matched, broad, stream_association = _intervention_rows(
            model,
            records,
            global_weights,
            rng,
            bootstrap_repetitions,
            randomization_repetitions,
            spearman_repetitions,
        )
        matched_rows.extend(matched)
        broad_rows.extend(broad)
        association_rows.extend(stream_association)
        association_rows.extend(
            _association_rows(model, records, spearman_repetitions)
        )

    expected_sizes = {
        "vxa": 60,
        "matched_intervention": 23,
        "broad_response": 23,
        "attribution_intervention": 23,
    }
    actual_sizes = {
        "vxa": len(vxa_rows),
        "matched_intervention": len(matched_rows),
        "broad_response": len(broad_rows),
        "attribution_intervention": len(association_rows),
    }
    if actual_sizes != expected_sizes:
        raise ValueError(
            f"analysis grid mismatch: expected {expected_sizes}, found {actual_sizes}"
        )

    return {
        "schema_version": 2,
        "instances": 7_228,
        "source_trajectories": 4_025,
        "models": 8,
        "policy_instance_rows": 57_824,
        "settings": {
            "context": PRIMARY_CONTEXT,
            "bootstrap_repetitions": bootstrap_repetitions,
            "randomization_repetitions": randomization_repetitions,
            "spearman_bootstrap_repetitions": spearman_repetitions,
            "group_by": "replay_id",
        },
        "policy_summary": policies,
        "vxa": vxa_rows,
        "matched_intervention": matched_rows,
        "broad_response": broad_rows,
        "attribution_intervention": association_rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.12g}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def write_results(output_dir: Path, results: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")
    for key, filename in (
        ("policy_summary", "policy_summary.csv"),
        ("vxa", "vxa.csv"),
        ("matched_intervention", "matched_intervention.csv"),
        ("broad_response", "broad_response.csv"),
        ("attribution_intervention", "attribution_intervention.csv"),
    ):
        _write_csv(output_dir / filename, results[key])


def reproduce(
    root: Path,
    output_dir: Path,
    *,
    bootstrap_repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    randomization_repetitions: int = DEFAULT_RANDOMIZATION_REPETITIONS,
    spearman_repetitions: int = DEFAULT_SPEARMAN_BOOTSTRAP_REPETITIONS,
    validate_inputs: bool = True,
) -> dict[str, Any]:
    results = analyze(
        root,
        bootstrap_repetitions=bootstrap_repetitions,
        randomization_repetitions=randomization_repetitions,
        spearman_repetitions=spearman_repetitions,
        validate_inputs=validate_inputs,
    )
    write_results(output_dir, results)
    return results
