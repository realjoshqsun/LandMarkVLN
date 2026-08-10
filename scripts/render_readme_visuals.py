#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


def render(policy_csv: Path, output: Path) -> None:
    with policy_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("policy summary is empty")

    width = 1120
    height = 120 + 48 * len(rows)
    label_x = 190
    bar_x = 240
    bar_width = 720
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Common action diagnostic by navigation VLA model</title>',
        '<desc id="desc">Horizontal bars show the percentage of the 937 frozen instances satisfying the local common action diagnostic for each model. This is not closed-loop success.</desc>',
        '<rect width="100%" height="100%" rx="18" fill="#f7f9fc"/>',
        '<text x="40" y="42" font-family="system-ui,sans-serif" font-size="24" font-weight="700" fill="#102a43">Common action diagnostic</text>',
        '<text x="40" y="70" font-family="system-ui,sans-serif" font-size="15" fill="#486581">Local horizon-matched agreement; not closed-loop success or a leaderboard</text>',
    ]
    for tick in range(0, 101, 20):
        x = bar_x + bar_width * tick / 100
        svg.append(
            f'<line x1="{x:.1f}" y1="88" x2="{x:.1f}" y2="{height - 30}" stroke="#d9e2ec" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{x:.1f}" y="104" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#627d98">{tick}%</text>'
        )

    colors = ["#4c78a8", "#5b8db8", "#72a6c7", "#59a14f", "#76b76d", "#f28e2b", "#e6a13c", "#b279a2"]
    for index, row in enumerate(rows):
        y = 126 + 48 * index
        rate = float(row["a_rate"])
        pixels = bar_width * rate
        display = html.escape(row["display_name"])
        svg.append(
            f'<text x="{label_x}" y="{y + 20}" text-anchor="end" font-family="system-ui,sans-serif" font-size="16" fill="#243b53">{display}</text>'
        )
        svg.append(
            f'<rect x="{bar_x}" y="{y}" width="{bar_width}" height="28" rx="6" fill="#e7eef6"/>'
        )
        svg.append(
            f'<rect x="{bar_x}" y="{y}" width="{pixels:.2f}" height="28" rx="6" fill="{colors[index % len(colors)]}"/>'
        )
        svg.append(
            f'<text x="{bar_x + pixels + 10:.2f}" y="{y + 20}" font-family="system-ui,sans-serif" font-size="14" font-weight="700" fill="#243b53">{100 * rate:.1f}%</text>'
        )
    svg.append("</svg>\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    root_default = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=root_default)
    args = parser.parse_args()
    root = args.root.resolve()
    render(
        root / "artifacts/reproduced/policy_summary.csv",
        root / "assets/readme/model_overview.svg",
    )
    print("regenerated assets/readme/model_overview.svg")


if __name__ == "__main__":
    main()

