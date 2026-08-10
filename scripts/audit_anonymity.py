#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path
from typing import Iterable


TEXT_SUFFIXES = {
    "", ".cfg", ".csv", ".html", ".ini", ".json", ".jsonl", ".md",
    ".py", ".rst", ".sh", ".svg", ".toml", ".txt", ".yaml", ".yml",
}
SKIP_PARTS = {".git", ".pytest_cache", "__pycache__", "dist", ".venv"}
FORBIDDEN = {
    "workspace username": re.compile("q84" + "sun", re.IGNORECASE),
    "private allocation": re.compile(r"def[-_]mcrowley", re.IGNORECASE),
    "private absolute path": re.compile(
        r"/(?:home|scratch|project|gpfs|lustre|cluster)/[A-Za-z0-9._-]+",
        re.IGNORECASE,
    ),
    "internal cluster name": re.compile(r"\b(?:" + "|".join(("f" + "ir", "ni" + "bi", "ror" + "qual", "tril" + "lium")) + r")\b", re.IGNORECASE),
    "email address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "GitHub token": re.compile(r"\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    "AWS access key": re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "credential assignment": re.compile(
        r"\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[:=]\s*['\"]?[^\s'\"]{8,}",
        re.IGNORECASE,
    ),
}


def candidate_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in SKIP_PARTS for part in relative.parts):
            continue
        if path.is_symlink():
            yield path
        elif path.is_file() and (path.suffix.lower() in TEXT_SUFFIXES or path.suffix == ".gz"):
            yield path


def iter_lines(path: Path) -> Iterable[tuple[int, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                yield line_number, line
    except (OSError, EOFError) as error:
        yield 0, f"UNREADABLE_ARCHIVE:{error}"


def audit(root: Path) -> list[str]:
    findings: list[str] = []
    for path in candidate_files(root):
        relative = path.relative_to(root)
        if path.is_symlink():
            findings.append(f"{relative}: symlink is not allowed in the review artifact")
            continue
        name = relative.as_posix()
        for label, pattern in FORBIDDEN.items():
            if pattern.search(name):
                findings.append(f"{relative}: {label} in filename")
        for line_number, line in iter_lines(path):
            for label, pattern in FORBIDDEN.items():
                match = pattern.search(line)
                if match:
                    excerpt = match.group(0)[:100]
                    findings.append(f"{relative}:{line_number}: {label}: {excerpt!r}")
                    if len(findings) >= 100:
                        return findings

    return findings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    findings = audit(root)
    if findings:
        for finding in findings:
            print(f"FAIL {finding}")
        raise SystemExit(f"anonymity audit failed with {len(findings)} finding(s)")
    print("anonymity audit passed")


if __name__ == "__main__":
    main()

