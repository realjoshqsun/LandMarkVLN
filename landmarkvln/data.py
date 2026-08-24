"""Read and validate the frozen sample-level analysis records."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROWS_PER_MODEL = 7_228
TOTAL_ROWS = 57_824
TRAJECTORIES_PER_MODEL = 4_025


@dataclass(frozen=True)
class ModelSpec:
    """Human-readable metadata and the frozen file for one policy."""

    display_name: str
    scopes: tuple[str, ...]
    relative_path: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "onevla": ModelSpec(
        "OneVLA",
        ("C", "M", "CM"),
        "artifacts/frozen/models/onevla/per_probe_metrics.jsonl.gz",
    ),
    "internvla_n1": ModelSpec(
        "InternVLA-N1",
        ("C", "M", "CM"),
        "artifacts/frozen/models/internvla_n1/per_probe_metrics.jsonl.gz",
    ),
    "omnivla": ModelSpec(
        "OmniVLA",
        ("C",),
        "artifacts/frozen/models/omnivla/per_probe_metrics.jsonl.gz",
    ),
    "uni_navid": ModelSpec(
        "Uni-NaVid",
        ("C", "M", "CM"),
        "artifacts/frozen/models/uni_navid/per_probe_metrics.jsonl.gz",
    ),
    "fantasyvln": ModelSpec(
        "FantasyVLN",
        ("C", "M", "CM"),
        "artifacts/frozen/models/fantasyvln/per_probe_metrics.jsonl.gz",
    ),
    "streamvln": ModelSpec(
        "StreamVLN",
        ("C", "M", "CM"),
        "artifacts/frozen/models/streamvln/per_probe_metrics.jsonl.gz",
    ),
    "vamos": ModelSpec(
        "VAMOS",
        ("C",),
        "artifacts/frozen/models/vamos/per_probe_metrics.jsonl.gz",
    ),
    "tic_vla": ModelSpec(
        "TIC-VLA",
        ("C", "M", "CM"),
        "artifacts/frozen/models/tic_vla/per_probe_metrics.jsonl.gz",
    ),
}


@dataclass(frozen=True)
class ValidationResult:
    """Identifiers collected while validating one policy's records."""

    model: str
    rows: int
    probe_ids: frozenset[str]
    replay_ids: frozenset[str]


def model_path(root: Path, model: str) -> Path:
    """Return the frozen JSONL path for ``model`` below ``root``."""

    try:
        relative_path = MODEL_SPECS[model].relative_path
    except KeyError:
        raise KeyError(f"unknown model: {model}") from None
    return root / relative_path


def iter_gzip_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield objects from a gzip-compressed JSON Lines file."""

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            yield row


def iter_model_rows(root: Path, model: str) -> Iterator[dict[str, Any]]:
    """Stream all frozen records for one model."""

    path = model_path(root, model)
    if not path.is_file():
        raise FileNotFoundError(path)
    yield from iter_gzip_jsonl(path)


def validate_model(root: Path, model: str) -> ValidationResult:
    """Validate row count and sample identifiers for one model."""

    probe_ids: set[str] = set()
    replay_ids: set[str] = set()
    task_indices: set[int] = set()
    rows = 0

    for rows, row in enumerate(iter_model_rows(root, model), start=1):
        probe_id = row.get("probe_id")
        if not isinstance(probe_id, str) or not probe_id:
            raise ValueError(f"{model}: invalid probe_id in row {rows}")
        if probe_id in probe_ids:
            raise ValueError(f"{model}: duplicate probe_id {probe_id}")
        probe_ids.add(probe_id)

        task_index = row.get("task_index")
        if not isinstance(task_index, int) or isinstance(task_index, bool):
            raise ValueError(f"{model}: invalid task_index in row {rows}")
        if task_index in task_indices:
            raise ValueError(f"{model}: duplicate task_index {task_index}")
        task_indices.add(task_index)

        replay_id = row.get("replay_id")
        if not isinstance(replay_id, str) or not replay_id:
            raise ValueError(f"{model}: invalid replay_id in row {rows}")
        replay_ids.add(replay_id)

    if rows != ROWS_PER_MODEL:
        raise ValueError(f"{model}: expected {ROWS_PER_MODEL} rows, found {rows}")
    expected_indices = set(range(ROWS_PER_MODEL))
    if task_indices != expected_indices:
        raise ValueError(f"{model}: task_index does not cover 0..{ROWS_PER_MODEL - 1}")

    return ValidationResult(
        model=model,
        rows=rows,
        probe_ids=frozenset(probe_ids),
        replay_ids=frozenset(replay_ids),
    )


def validate_all(root: Path) -> list[ValidationResult]:
    """Validate the eight files and their shared sample membership."""

    results = [validate_model(root, model) for model in MODEL_SPECS]
    reference = results[0].probe_ids
    for result in results:
        if result.probe_ids != reference:
            missing = len(reference - result.probe_ids)
            extra = len(result.probe_ids - reference)
            raise ValueError(
                f"{result.model}: cross-model probe membership mismatch "
                f"({missing} missing, {extra} extra)"
            )
        if len(result.replay_ids) != TRAJECTORIES_PER_MODEL:
            raise ValueError(
                f"{result.model}: expected {TRAJECTORIES_PER_MODEL} unique replay_id "
                f"values, found {len(result.replay_ids)}"
            )

    total_rows = sum(result.rows for result in results)
    if total_rows != TOTAL_ROWS:
        raise ValueError(f"expected {TOTAL_ROWS} total rows, found {total_rows}")
    return results
