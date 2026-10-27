#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from landmarkvln.data import validate_all


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    results = validate_all(root)
    print(
        f"validated {len(results)} models, "
        f"{results[0].rows} samples per model, "
        f"{len(results[0].replay_ids)} source trajectories, and "
        f"{sum(result.rows for result in results)} model-sample rows"
    )


if __name__ == "__main__":
    main()
