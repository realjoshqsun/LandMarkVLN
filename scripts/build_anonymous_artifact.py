#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path


FIXED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
PREFIX = "LandMarkVLN-WACV2027"
SKIP_PARTS = {".git", ".pytest_cache", "__pycache__", "dist", ".venv"}


def selected_files(root: Path, manifest: Path) -> list[Path]:
    selected: set[Path] = set()
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        entry = raw.strip()
        if not entry or entry.startswith("#"):
            continue
        path = root / entry
        if not path.exists():
            raise FileNotFoundError(f"allowlisted path does not exist: {entry}")
        candidates = path.rglob("*") if path.is_dir() else [path]
        for candidate in candidates:
            if not candidate.is_file():
                continue
            relative = candidate.relative_to(root)
            if any(part in SKIP_PARTS for part in relative.parts):
                continue
            if candidate.is_symlink():
                raise ValueError(f"symlink is not allowed: {relative}")
            selected.add(relative)
    return sorted(selected, key=lambda path: path.as_posix())


def zip_info(name: str, mode: int = 0o644) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = mode << 16
    info.create_system = 3
    return info


def build(root: Path, output: Path) -> str:
    scripts = root / "scripts"
    subprocess.run(
        [sys.executable, str(scripts / "verify_frozen_outputs.py"), "--root", str(root)],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(scripts / "audit_anonymity.py"), "--root", str(root)],
        check=True,
    )
    files = selected_files(root, root / "artifact-manifest.txt")

    output.parent.mkdir(parents=True, exist_ok=True)
    content_hashes: list[str] = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in files:
            data = (root / relative).read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            content_hashes.append(f"{digest}  {relative.as_posix()}")
            archive.writestr(zip_info(f"{PREFIX}/{relative.as_posix()}"), data)
        manifest_bytes = ("\n".join(content_hashes) + "\n").encode("utf-8")
        archive.writestr(zip_info(f"{PREFIX}/ARTIFACT_CONTENTS.sha256"), manifest_bytes)

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    (output.parent / "SHA256SUMS").write_text(
        f"{digest}  {output.name}\n", encoding="utf-8"
    )
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    root_default = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "dist/LandMarkVLN-WACV2027-anonymous.zip"
    digest = build(root, output)
    print(f"built {output} ({output.stat().st_size} bytes)")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()

