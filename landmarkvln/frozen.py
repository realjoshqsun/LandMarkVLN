from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


MODEL_PATHS = {
    "onevla": "artifacts/frozen/models/onevla/per_probe_metrics.jsonl",
    "internvla_n1": "artifacts/frozen/models/internvla_n1/per_probe_metrics.jsonl",
    "omnivla": "artifacts/frozen/models/omnivla/per_probe_metrics.jsonl",
    "uni_navid": "artifacts/frozen/models/uni_navid/per_probe_metrics.jsonl",
    "fantasyvln": "artifacts/frozen/models/fantasyvln/per_probe_metrics.jsonl",
    "streamvln": "artifacts/frozen/models/streamvln/per_probe_metrics.jsonl",
    "vamos": "artifacts/frozen/models/vamos/per_probe_metrics.jsonl",
    "tic_vla": "artifacts/frozen/models/tic_vla/per_probe_metrics.jsonl",
}


@dataclass(frozen=True)
class ValidationResult:
    model: str
    rows: int
    a_true: int
    probe_ids: frozenset[str]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield row


def expected_results(root: Path) -> dict[str, Any]:
    return read_json(root / "artifacts/frozen/expected_results.json")


def validate_model(root: Path, model: str, expected: dict[str, Any]) -> ValidationResult:
    path = root / MODEL_PATHS[model]
    if not path.is_file():
        raise FileNotFoundError(path)

    rows = 0
    a_true = 0
    probe_ids: set[str] = set()
    task_indices: set[int] = set()

    for row in iter_jsonl(path):
        rows += 1
        probe_id = row.get("probe_id")
        task_index = row.get("task_index")
        a_value = row.get("A")
        if not isinstance(probe_id, str) or not probe_id:
            raise ValueError(f"{model}: invalid probe_id in row {rows}")
        if not isinstance(task_index, int):
            raise ValueError(f"{model}: invalid task_index in row {rows}")
        if not isinstance(a_value, bool):
            raise ValueError(f"{model}: A must be boolean in row {rows}")
        if probe_id in probe_ids:
            raise ValueError(f"{model}: duplicate probe_id {probe_id}")
        if task_index in task_indices:
            raise ValueError(f"{model}: duplicate task_index {task_index}")
        probe_ids.add(probe_id)
        task_indices.add(task_index)
        a_true += int(a_value)

    expected_rows = int(expected["instances_per_model"])
    expected_a = int(expected["models"][model]["a_true"])
    if rows != expected_rows:
        raise ValueError(f"{model}: expected {expected_rows} rows, found {rows}")
    if task_indices != set(range(expected_rows)):
        raise ValueError(f"{model}: task_index does not cover 0..{expected_rows - 1}")
    if a_true != expected_a:
        raise ValueError(f"{model}: expected {expected_a} A=true rows, found {a_true}")

    return ValidationResult(model, rows, a_true, frozenset(probe_ids))


def validate_all(root: Path) -> list[ValidationResult]:
    expected = expected_results(root)
    results = [validate_model(root, model, expected) for model in MODEL_PATHS]
    reference = results[0].probe_ids
    for result in results[1:]:
        if result.probe_ids != reference:
            missing = len(reference - result.probe_ids)
            extra = len(result.probe_ids - reference)
            raise ValueError(
                f"{result.model}: cross-model membership mismatch "
                f"({missing} missing, {extra} extra)"
            )
    expected_total = int(expected["policy_instance_rows"])
    actual_total = sum(result.rows for result in results)
    if actual_total != expected_total:
        raise ValueError(f"expected {expected_total} total rows, found {actual_total}")
    return results


def registered_axv_cell(row: dict[str, Any], scope: str) -> dict[str, Any] | None:
    try:
        registered = row["axv"]["sub"]["gradient_times_attention"]
    except (KeyError, TypeError):
        return None
    if not isinstance(registered, dict):
        return None
    if "detected" in registered:
        return registered if scope == "C" else None
    cell = registered.get(scope)
    return cell if isinstance(cell, dict) else None


def usable_detection(cell: dict[str, Any] | None) -> bool | None:
    if cell is None:
        return None
    if cell.get("valid") is False:
        return None
    if cell.get("parse_abstention") is True:
        return None
    if cell.get("abstention_reason") not in (None, ""):
        return None
    detected = cell.get("detected")
    return detected if isinstance(detected, bool) else None


def summarize_registered_vxa(
    root: Path, model: str, scopes: list[str]
) -> list[dict[str, Any]]:
    path = root / MODEL_PATHS[model]
    counters = {
        scope: {"a1_n": 0, "a1_v": 0, "a0_n": 0, "a0_v": 0, "abstained": 0}
        for scope in scopes
    }
    for row in iter_jsonl(path):
        # FantasyVLN has context-matched common-A definitions. The cross-model
        # headline uses full context (row["A"]), while the registered sub-
        # instruction A-by-V panel prospectively uses A_by_context["sub"].
        if model == "fantasyvln":
            a_value = bool(row["A_by_context"]["sub"])
        else:
            a_value = bool(row["A"])
        for scope in scopes:
            detected = usable_detection(registered_axv_cell(row, scope))
            if detected is None:
                counters[scope]["abstained"] += 1
                continue
            prefix = "a1" if a_value else "a0"
            counters[scope][f"{prefix}_n"] += 1
            counters[scope][f"{prefix}_v"] += int(detected)

    output: list[dict[str, Any]] = []
    for scope in scopes:
        cell = counters[scope]
        a1_rate = cell["a1_v"] / cell["a1_n"] if cell["a1_n"] else None
        a0_rate = cell["a0_v"] / cell["a0_n"] if cell["a0_n"] else None
        rd = a1_rate - a0_rate if a1_rate is not None and a0_rate is not None else None
        output.append(
            {
                "model": model,
                "scope": scope,
                "valid_rows": cell["a1_n"] + cell["a0_n"],
                "abstained_rows": cell["abstained"],
                "a1_n": cell["a1_n"],
                "a1_v": cell["a1_v"],
                "a1_v_rate": a1_rate,
                "a0_n": cell["a0_n"],
                "a0_v": cell["a0_v"],
                "a0_v_rate": a0_rate,
                "risk_difference": rd,
            }
        )
    return output


def reproduce(root: Path, output_dir: Path) -> dict[str, Any]:
    validation = validate_all(root)
    expected = expected_results(root)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_rows: list[dict[str, Any]] = []
    vxa_rows: list[dict[str, Any]] = []
    for result in validation:
        spec = expected["models"][result.model]
        policy_rows.append(
            {
                "model": result.model,
                "display_name": spec["display_name"],
                "scopes": "/".join(spec["scopes"]),
                "instances": result.rows,
                "a_true": result.a_true,
                "a_rate": result.a_true / result.rows,
            }
        )
        vxa_rows.extend(summarize_registered_vxa(root, result.model, list(spec["scopes"])))

    policy_path = output_dir / "policy_summary.csv"
    with policy_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(policy_rows[0]))
        writer.writeheader()
        for row in policy_rows:
            writer.writerow({**row, "a_rate": f"{row['a_rate']:.10f}"})

    vxa_path = output_dir / "registered_vxa.csv"
    fieldnames = list(vxa_rows[0])
    with vxa_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in vxa_rows:
            formatted = dict(row)
            for key in ("a1_v_rate", "a0_v_rate", "risk_difference"):
                value = formatted[key]
                formatted[key] = "" if value is None else f"{value:.10f}"
            writer.writerow(formatted)

    summary = {
        "schema_version": 1,
        "instances_per_model": expected["instances_per_model"],
        "models": len(policy_rows),
        "policy_instance_rows": sum(row["instances"] for row in policy_rows),
        "registered_context": "sub",
        "registered_estimator": "gradient_times_attention",
        "policy_summary": policy_rows,
        "registered_vxa": vxa_rows,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary

