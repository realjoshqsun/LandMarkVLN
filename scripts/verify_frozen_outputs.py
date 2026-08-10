#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from landmarkvln.frozen import validate_all


def verify_checksums(root: Path) -> int:
    manifest = root / "artifacts/frozen/checksums.sha256"
    checked = 0
    for line_number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            expected, relative = line.split(maxsplit=1)
        except ValueError as error:
            raise ValueError(f"invalid checksum line {line_number}") from error
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"checksum mismatch: {relative}")
        checked += 1
    return checked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    checksums = verify_checksums(root)
    models = validate_all(root)
    print(
        f"verified {checksums} frozen files; "
        f"{len(models)} models x {models[0].rows} instances"
    )


if __name__ == "__main__":
    main()

