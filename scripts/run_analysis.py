#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from landmarkvln.analysis import reproduce
from landmarkvln.statistics import (
    DEFAULT_BOOTSTRAP_REPETITIONS,
    DEFAULT_RANDOMIZATION_REPETITIONS,
    DEFAULT_SPEARMAN_BOOTSTRAP_REPETITIONS,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPETITIONS,
        help="trajectory bootstrap repetitions for VxA and matched effects",
    )
    parser.add_argument(
        "--randomization-repetitions",
        type=int,
        default=DEFAULT_RANDOMIZATION_REPETITIONS,
        help="trajectory sign-randomization repetitions",
    )
    parser.add_argument(
        "--spearman-repetitions",
        type=int,
        default=DEFAULT_SPEARMAN_BOOTSTRAP_REPETITIONS,
        help="trajectory bootstrap repetitions for Spearman intervals",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir or root / "artifacts/reproduced"
    results = reproduce(
        root,
        output_dir,
        bootstrap_repetitions=args.bootstrap_repetitions,
        randomization_repetitions=args.randomization_repetitions,
        spearman_repetitions=args.spearman_repetitions,
    )
    print(
        f"reproduced {len(results['vxa'])} VxA panels, "
        f"{len(results['matched_intervention'])} matched interventions, "
        f"{len(results['broad_response'])} broad responses, and "
        f"{len(results['attribution_intervention'])} correlations in {output_dir}"
    )


if __name__ == "__main__":
    main()
