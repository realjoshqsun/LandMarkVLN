#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from landmarkvln.causal import reproduce_causal
from landmarkvln.frozen import reproduce


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir or root / "artifacts/reproduced"
    summary = reproduce(root, output)
    causal_rows = reproduce_causal(root, output)
    print(
        f"reproduced {len(summary['policy_summary'])} model summaries and "
        f"{len(summary['registered_vxa'])} registered A-by-V cells, plus "
        f"{len(causal_rows)} native causal rows in {output}"
    )


if __name__ == "__main__":
    main()

