from __future__ import annotations

import math
from typing import Any


CONTEXTS = ("full", "prefix", "sub")
ESTIMATORS = ("attention_only", "gradient_only", "gradient_times_attention")

# A branch is a separate native output family. Only InternVLA-N1 has two.
BRANCH_SPECS: dict[str, tuple[dict[str, Any], ...]] = {
    "onevla": (
        {"branch": "native", "display_name": "OneVLA", "metric": "navigation_l2"},
    ),
    "internvla_n1": (
        {
            "branch": "continuous",
            "display_name": "InternVLA-N1 (cont.)",
            "metric": "position_rms_m",
        },
        {
            "branch": "discrete",
            "display_name": "InternVLA-N1 (disc.)",
            "metric": "normalized_edit_distance",
        },
    ),
    "omnivla": (
        {"branch": "native", "display_name": "OmniVLA", "metric": "position_rms_m"},
    ),
    "uni_navid": (
        {
            "branch": "native",
            "display_name": "Uni-NaVid",
            "metric": "clean_action_log_probability",
        },
    ),
    "fantasyvln": (
        {
            "branch": "native",
            "display_name": "FantasyVLN",
            "metric": "clean_action_log_probability",
        },
    ),
    "streamvln": (
        {
            "branch": "native",
            "display_name": "StreamVLN",
            "metric": "clean_action_log_probability",
        },
    ),
    "vamos": (
        {
            "branch": "native",
            "display_name": "VAMOS",
            "metric": "clean_path_log_probability",
        },
    ),
    "tic_vla": (
        {"branch": "native", "display_name": "TIC-VLA", "metric": "position_rms"},
    ),
}


def action_value(model: str, row: dict[str, Any], context: str = "sub") -> bool:
    if context not in CONTEXTS:
        raise KeyError(f"unknown context: {context}")
    value = row["A_by_context"][context] if model == "fantasyvln" else row["A"]
    if not isinstance(value, bool):
        raise ValueError(f"{model}: A is not boolean for {row.get('probe_id')}")
    return value


def attribution_cell(
    model: str,
    row: dict[str, Any],
    estimator: str,
    scope: str,
    context: str = "sub",
) -> dict[str, Any] | None:
    """Return one attribution cell, or None when that cell is invalid."""

    if context not in CONTEXTS:
        raise KeyError(f"unknown context: {context}")
    if estimator not in ESTIMATORS:
        raise KeyError(f"unknown estimator: {estimator}")
    try:
        value = row["axv"][context][estimator]
        cell = value if model in {"omnivla", "vamos"} else value[scope]
    except (KeyError, TypeError):
        return None
    if not isinstance(cell, dict):
        return None
    if cell.get("valid") is False:
        return None
    if cell.get("parse_abstention") is True:
        return None
    if cell.get("abstention_reason") not in (None, ""):
        return None
    if not isinstance(cell.get("detected"), bool):
        return None
    effect_size = cell.get("effect_size")
    if not isinstance(effect_size, (int, float)) or not math.isfinite(effect_size):
        return None
    return cell


def _intern_kind(branch: str) -> tuple[str, str]:
    if branch == "continuous":
        return "continuous_xy_trajectory", "trajectory_waypoint_rms_m"
    if branch == "discrete":
        return "discrete_action_chunk", "normalized_discrete_edit_distance"
    raise KeyError(f"unknown InternVLA-N1 branch: {branch}")


def matched_effect(
    model: str, branch: str, scope: str, row: dict[str, Any]
) -> float | None:
    """Extract the paper's matched target-control effect in native units."""

    causal = row.get("causal")
    if causal is None:
        return None
    try:
        if model == "onevla":
            value = causal[scope]["target_minus_control_navigation_l2"]
        elif model == "internvla_n1":
            clean_kind, distance_kind = _intern_kind(branch)
            cell = causal[scope]
            if row.get("clean_native_branch") != clean_kind:
                return None
            if not cell.get("pair_distance_valid"):
                return None
            if cell.get("pair_distance_kind") != distance_kind:
                return None
            value = cell["target_minus_control_distance"]
        elif model == "omnivla":
            value = causal["delta"]["position_rms_m"]
        elif model == "uni_navid":
            value = causal[scope]["numeric"][
                "control_minus_target_clean_chunk_log_probability"
            ]
        elif model == "fantasyvln":
            value = causal[scope]["numeric"][
                "control_minus_target_clean_action_chunk_log_probability"
            ]
        elif model == "streamvln":
            value = causal[scope]["primary_control_minus_target_log_probability_delta"]
        elif model == "vamos":
            value = causal["delta"]["clean_path_log_probability_harm"]
        elif model == "tic_vla":
            value = causal[scope]["target_minus_control"]["position_rms"]
        else:
            raise KeyError(f"unknown model: {model}")
    except (KeyError, TypeError):
        return None
    return _finite_float(value)


def broad_effect(
    model: str, branch: str, scope: str, row: dict[str, Any]
) -> float | None:
    """Extract the response to editing every visual input in a scope."""

    causal = row.get("causal")
    if causal is None:
        return None
    try:
        if model == "onevla":
            value = causal[scope]["full_navigation_l2"]
        elif model == "internvla_n1":
            clean_kind, distance_kind = _intern_kind(branch)
            cell = causal[scope]["full"]
            if row.get("clean_native_branch") != clean_kind:
                return None
            if not cell.get("distance_valid") or cell.get("distance_kind") != distance_kind:
                return None
            value = cell["distance"]
        elif model == "omnivla":
            value = causal["full"]["position_rms_m"]
        elif model in {"uni_navid", "fantasyvln", "streamvln"}:
            value = causal[scope]["broad_clean_minus_full_log_probability"]
        elif model == "vamos":
            value = causal["full"]["clean_path_log_probability_harm"]
        elif model == "tic_vla":
            value = causal[scope]["broad_full_metrics"]["position_rms"]
        else:
            raise KeyError(f"unknown model: {model}")
    except (KeyError, TypeError):
        return None
    return _finite_float(value)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None
