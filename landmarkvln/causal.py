from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Iterable

from landmarkvln.frozen import MODEL_PATHS, iter_jsonl


def _summary(
    model: str,
    display_name: str,
    metric: str,
    scope: str,
    values: Iterable[float],
) -> dict[str, Any]:
    kept = [float(value) for value in values]
    if not kept:
        raise ValueError(f"no causal values for {model}/{scope}")
    return {
        "model": model,
        "display_name": display_name,
        "native_metric": metric,
        "scope": scope,
        "n": len(kept),
        "mean_effect": fmean(kept),
        "positive_rate": sum(value > 0 for value in kept) / len(kept),
    }


def _rows(root: Path, model: str) -> list[dict[str, Any]]:
    return list(iter_jsonl(root / MODEL_PATHS[model]))


def primary_causal_rows(root: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    scopes = ("C", "M", "CM")

    rows = _rows(root, "onevla")
    for scope in scopes:
        output.append(
            _summary(
                "onevla",
                "OneVLA",
                "navigation_l2",
                scope,
                (row["causal"][scope]["target_minus_control_navigation_l2"] for row in rows),
            )
        )

    rows = _rows(root, "internvla_n1")
    intern_branches = (
        (
            "internvla_n1_cont",
            "InternVLA-N1 (cont.)",
            "continuous_xy_trajectory",
            "trajectory_waypoint_rms_m",
            "position_rms_m",
        ),
        (
            "internvla_n1_disc",
            "InternVLA-N1 (disc.)",
            "discrete_action_chunk",
            "normalized_discrete_edit_distance",
            "normalized_edit_distance",
        ),
    )
    for key, display, branch, distance_kind, metric in intern_branches:
        for scope in scopes:
            values = (
                row["causal"][scope]["target_minus_control_distance"]
                for row in rows
                if row["clean_native_branch"] == branch
                and row["causal"][scope]["pair_distance_valid"]
                and row["causal"][scope]["pair_distance_kind"] == distance_kind
                and row["causal"][scope]["target_minus_control_distance"] is not None
            )
            output.append(_summary(key, display, metric, scope, values))

    rows = _rows(root, "omnivla")
    output.append(
        _summary(
            "omnivla",
            "OmniVLA",
            "position_rms_m",
            "C",
            (row["causal"]["delta"]["position_rms_m"] for row in rows),
        )
    )

    rows = _rows(root, "uni_navid")
    for scope in scopes:
        output.append(
            _summary(
                "uni_navid",
                "Uni-NaVid",
                "clean_action_log_probability",
                scope,
                (
                    row["causal"][scope]["numeric"][
                        "control_minus_target_clean_chunk_log_probability"
                    ]
                    for row in rows
                ),
            )
        )

    rows = _rows(root, "fantasyvln")
    for scope in scopes:
        output.append(
            _summary(
                "fantasyvln",
                "FantasyVLN",
                "clean_action_log_probability",
                scope,
                (
                    row["causal"][scope]["numeric"][
                        "control_minus_target_clean_action_chunk_log_probability"
                    ]
                    for row in rows
                    if row["causal"] is not None
                ),
            )
        )

    rows = _rows(root, "streamvln")
    for scope in scopes:
        output.append(
            _summary(
                "streamvln",
                "StreamVLN",
                "clean_action_log_probability",
                scope,
                (
                    row["causal"][scope][
                        "primary_control_minus_target_log_probability_delta"
                    ]
                    for row in rows
                ),
            )
        )

    rows = _rows(root, "vamos")
    output.append(
        _summary(
            "vamos",
            "VAMOS",
            "clean_path_log_probability",
            "C",
            (row["causal"]["delta"]["clean_path_log_probability_harm"] for row in rows),
        )
    )

    rows = _rows(root, "tic_vla")
    for scope in scopes:
        output.append(
            _summary(
                "tic_vla",
                "TIC-VLA",
                "position_rms",
                scope,
                (
                    row["causal"][scope]["target_minus_control"]["position_rms"]
                    for row in rows
                ),
            )
        )
    return output


def validate_expected(root: Path, rows: list[dict[str, Any]]) -> None:
    expected_path = root / "artifacts/frozen/expected_causal.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))["rows"]
    actual = {(row["model"], row["scope"]): row for row in rows}
    if set(actual) != {(row["model"], row["scope"]) for row in expected}:
        raise ValueError("registered causal row set does not match expected results")
    for target in expected:
        row = actual[(target["model"], target["scope"])]
        if row["n"] != target["n"]:
            raise ValueError(f"causal N mismatch: {target['model']}/{target['scope']}")
        decimals = int(target["mean_decimals"])
        if round(row["mean_effect"], decimals) != target["mean_effect"]:
            raise ValueError(f"causal mean mismatch: {target['model']}/{target['scope']}")
        if round(100 * row["positive_rate"], 1) != target["positive_percent"]:
            raise ValueError(f"causal positive-rate mismatch: {target['model']}/{target['scope']}")


def reproduce_causal(root: Path, output_dir: Path) -> list[dict[str, Any]]:
    rows = primary_causal_rows(root)
    validate_expected(root, rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "registered_causal.csv"
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "mean_effect": f"{row['mean_effect']:.10f}",
                    "positive_rate": f"{row['positive_rate']:.10f}",
                }
            )
    return rows

