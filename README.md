# LandMarkVLN

This repository reproduces the main LandMarkVLN analyses from analysis-ready,
sample-level outputs for eight navigation VLA models. The complete path runs on
CPU with Python, NumPy, and SciPy.

## Release scope

| Component | Count |
|---|---:|
| LandMarkVLN samples | 7,228 |
| Source navigation trajectories | 4,025 |
| Navigation VLA models | 8 |
| Model-sample rows | 57,824 |
| VxA panels | 60 |
| Matched intervention configurations | 23 |
| Broad-response configurations | 23 |
| Attribution-intervention correlations | 23 |

The eight gzip files occupy about 77 MiB. Every model is evaluated on the same
7,228-sample membership.

## Quick start

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

make validate
make test
make quick
```

`make quick` runs the complete analysis grid with fewer resampling repetitions
and writes results to `artifacts/reproduced/quick/`.

Run the paper-scale statistical protocol with:

```bash
make reproduce
```

Full results are written to `artifacts/reproduced/`. Run `make verify-paper` to execute the full protocol and check every published confidence interval and Monte Carlo value against `artifacts/frozen/paper_results.json`.

## Reproduced analyses

| Output | Analysis |
|---|---|
| `policy_summary.csv` | Local action diagnostic for each of the eight models |
| `vxa.csv` | 60 VxA panels across attention, gradient, and attention × gradient |
| `matched_intervention.csv` | 23 matched landmark-versus-control effects |
| `broad_response.csv` | 23 full-frame broad responses |
| `attribution_intervention.csv` | 23 Spearman attribution-intervention associations |
| `summary.json` | Complete machine-readable results and analysis settings |

VxA uses the landmark-bearing local instruction segment (`sub`). Temporal
models expose current (`C`), memory (`M`), and combined (`CM`) scopes.
Current-only models expose `C`, giving 20 model-scope combinations and 60
panels across the three attribution estimators.

All uncertainty resampling is grouped by `replay_id`, the source navigation
trajectory. The full analysis uses:

- 5,000 trajectory-group bootstrap repetitions for VxA risk differences;
- 5,000 trajectory-group bootstrap repetitions and 10,000 trajectory sign
  randomizations for matched effects;
- 2,000 trajectory-group bootstrap repetitions for Spearman correlations.

## Scientific interpretation

`A` is a local, horizon-matched action diagnostic. It measures agreement with
the local expert action criterion, not closed-loop navigation success or SPL.

Matched causal effects retain each model's native response scale. Their signs
and within-model uncertainty are comparable to the paper definitions; their
raw magnitudes do not support cross-model ranking.

Memory and combined scopes are structurally unavailable for current-only
models. They are represented as not applicable, not as zero.

Invalid, malformed, or abstained measurements are excluded by explicit adapter
rules. They are not converted to negative detections or zero effects.

## Release boundary

This repository is the analysis-ready CPU reproduction path. It contains the
sample-level model outputs and the code required to validate them and reproduce
the paper analyses.

GPU model inference uses upstream model repositories, checkpoints,
Matterport3D imagery, and R2R/RxR assets. Obtain those resources from their
original providers under their licenses and access terms; they are not
redistributed in this repository.

## Repository layout

```text
artifacts/frozen/models/   eight analysis-ready gzip JSONL files
artifacts/frozen/paper_results.json  paper-value regression reference
artifacts/reproduced/      generated CSV and JSON results
landmarkvln/data.py        streaming readers and membership validation
landmarkvln/metrics.py     explicit model-schema adapters
landmarkvln/statistics.py  shared statistical procedures
landmarkvln/analysis.py    complete 60/23/23/23 analysis grid
scripts/                   command-line entry points
tests/                     focused data, adapter, and numerical tests
```

See `REPRODUCIBILITY.md` for the full command sequence and `DATA_CARD.md` for
dataset composition, fields, validity rules, and licensing.
