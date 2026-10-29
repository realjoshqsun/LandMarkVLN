# LandMarkVLN

[![Project Homepage](https://img.shields.io/badge/Project-Homepage-245bc5?logo=googlechrome&logoColor=white)](https://realjoshqsun.github.io/landmark-vln/)
[![WACV 2027](https://img.shields.io/badge/WACV-2027-334155)](https://realjoshqsun.github.io/landmark-vln/)

Official repository for **Do Navigation VLA Models Ground What They Are Told?
An Empirical Audit of Landmark-Vision Alignment in VLN**, accepted to
**WACV 2027, Datasets Track**.

**Josh Qixuan Sun, [Xiaoying Xing](https://sites.google.com/view/xiaoyingxing/home), [Yurun Chen](https://euron-zc.github.io/), [Huaiyuan Weng](https://scholar.google.com/citations?user=e1hV8IgAAAAJ&hl=en), [Wentao Zhu](https://wentao.live/), and Mark Crowley**

LandMarkVLN studies whether navigation vision-language-action models localize
and rely on the landmarks named in their instructions. Its 7,228-sample
evaluation dataset connects instructions with current and historical
observations, same-instance landmark masks, and local reference actions.
The audit separates local action agreement, landmark localization through
visual attribution, and landmark-specific output sensitivity through matched
visual interventions.

This release provides the **CPU analysis code and analysis-ready model
measurements** used to reproduce the main paper analyses across eight navigation
VLA models. It runs with Python, NumPy, and SciPy. Source imagery, raw dataset
assets, model checkpoints, and GPU inference environments are obtained separately
from their original providers; see [Release boundary](#release-boundary).

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
git clone https://github.com/realjoshqsun/LandMarkVLN.git
cd LandMarkVLN
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
sample-level model measurements and the code required to validate them and
reproduce the paper analyses. It does not include the raw LandMarkVLN input
bundle, dataset-construction pipeline, or end-to-end GPU inference pipeline. RGB frames, instruction or
landmark text, pixel masks, and full attribution maps are not distributed here.

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

## Citation

```bibtex
@inproceedings{sun2027landmark,
  title = {Do Navigation VLA Models Ground What They Are Told? An Empirical Audit of Landmark-Vision Alignment in VLN},
  author = {Josh Qixuan Sun and Xiaoying Xing and Yurun Chen and Huaiyuan Weng and Wentao Zhu and Mark Crowley},
  booktitle = {Winter Conference on Applications of Computer Vision},
  year = {2027}
}
```

## License

The analysis code is released under the [MIT License](LICENSE). Third-party
source assets remain subject to their original licenses and access terms; see
[Third-party notices](THIRD_PARTY_NOTICES.md) and the [data card](DATA_CARD.md).
