# Reproducibility guide

This guide reproduces the paper analyses from the eight analysis-ready gzip
files. The workflow is CPU-only and uses Python, NumPy, and SciPy.

## Environment

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On clusters that provide the Alliance software stack, NumPy and SciPy can be
loaded from the system module:

```bash
module load scipy-stack/2026a
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --no-deps -e .
```

## Validate and test

Run from the repository root:

```bash
make validate
make test
```

`make validate` streams all eight gzip files and checks:

- 7,228 rows for each model and 57,824 rows in total;
- unique `probe_id` and `task_index` values within each model;
- identical sample membership across all eight models;
- complete `task_index` coverage;
- 4,025 distinct source trajectories per model.

`make test` exercises the streaming reader, explicit model adapters, grouped
bootstrap, Fisher test, Holm adjustment, trajectory sign randomization,
Spearman implementation, and all current-paper point estimates.

## Quick analysis

```bash
make quick
```

The quick run writes to `artifacts/reproduced/quick/` and uses 10 bootstrap
repetitions for VxA and matched effects, 20 trajectory sign randomizations,
and 10 bootstrap repetitions for Spearman correlations. It covers the complete
60/23/23/23 grid and is intended for an end-to-end execution check. Sample
counts and observed point estimates match the full run; its resampling
intervals and Monte Carlo values are intentionally coarse.

## Paper-scale analysis

```bash
make reproduce
```

The full run writes to `artifacts/reproduced/` and uses the paper protocol:

- 5,000 trajectory-group bootstrap repetitions for VxA risk differences;
- 5,000 trajectory-group bootstrap repetitions for matched mean effects;
- 10,000 trajectory-level sign randomizations for matched effects;
- 2,000 trajectory-group bootstrap repetitions for Spearman correlations.

Fixed analysis seeds make repeated runs reproducible. For a full numerical regression against the paper values, run `make verify-paper`; this executes the paper-scale protocol and checks every published confidence interval and Monte Carlo value.

## Generated files

### `policy_summary.csv`

Eight rows, one per model, with the number of samples and trajectories and the
count and rate for the local action diagnostic `A`.

### `vxa.csv`

Sixty rows cover the local instruction context, three attribution estimators,
and every available temporal scope. Each row reports valid counts, action and
localization rates, the VxA risk difference, its grouped-bootstrap interval, a
two-sided sample-level Fisher value, and the within-model Holm-adjusted value.
Temporal models form Holm families of nine panels; current-only models form
families of three.

### `matched_intervention.csv`

Twenty-three model-branch-scope rows report the mean landmark-versus-control
effect in native units, its grouped-bootstrap interval, the positive-effect
rate, a one-sided trajectory sign-randomization value, and Holm adjustment
across scopes within each model and native branch.

### `broad_response.csv`

Twenty-three model-branch-scope rows report the mean response to the full-frame
intervention in each model's native response units.

### `attribution_intervention.csv`

Twenty-three model-branch-scope rows relate the continuous attention × gradient
effect size to the matched intervention effect. Each row reports paired counts,
Spearman's rho, its two-sided value, and a grouped-bootstrap interval.

### `summary.json`

This file contains all five tables together with dataset counts, analysis
settings, repetition counts, and the grouping variable.

## Analysis definitions

VxA uses the landmark-bearing local instruction segment (`sub`) with
attention-only, gradient-only, and attention × gradient attribution. Confidence
intervals resample source trajectories identified by `replay_id`, retaining all
samples from every selected trajectory. Fisher tests use the observed
sample-level 2 × 2 table.

Matched interventions compare the landmark edit with its matched spatial
control while keeping the full instruction fixed. Sign randomization applies
one shared sign to every eligible sample from a source trajectory.

The attribution-intervention analysis uses the continuous attention × gradient
effect size, not its binary detection decision.

## Validity rules

`landmarkvln/metrics.py` contains short, explicit adapters for the eight native
result schemas. An analysis row is included when the required attribution or
causal response is finite and valid for that model and branch.

Abstentions, malformed values, invalid native distances, and unavailable
branches are excluded from the corresponding denominator. OmniVLA and VAMOS
are current-only models; memory and combined scopes are structurally not
applicable.

## Inference boundary

The eight model outputs were produced with their native checkpoints, input
formats, action decoders, and response metrics. This CPU path begins at the
analysis-ready records.

Re-running model inference requires the corresponding GPU environments,
upstream model weights, Matterport3D scans, and R2R/RxR resources. Obtain those
assets from their original providers under their licenses and access terms.
