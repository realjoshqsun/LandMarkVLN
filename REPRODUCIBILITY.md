# Reproducibility guide

## CPU frozen-output path

Run from the repository root:

```bash
make verify
make reproduce-analysis
make test
```

`make verify` checks byte-level SHA-256 identities, JSONL structure, exact
937-row coverage per model, unique instance identities, and identical
cross-model membership.

`make reproduce-analysis` then recomputes:

- common action-diagnostic counts and rates;
- registered `sub`-instruction, gradient-times-attention A-by-V point
  estimates for every available temporal scope;
- registered model-native causal sample means and positive-effect fractions;
- exact valid and abstention denominators;
- the README model summary visualization.

Generated files are written to `artifacts/reproduced/`. They are derived files
and are intentionally ignored by Git.

## Statistical boundary

The compact CPU path reconstructs deterministic point estimates directly from
the frozen per-instance records. Paper confidence intervals and tests use
replay-grouped resampling and the prospectively frozen model-specific outcome
definitions. The full resampling implementation is not declared portable until
its source/configuration provenance and clean-environment tests pass.

## Full inference boundary

Each evaluated policy uses its released native interface, checkpoint, decoding
rules, action representation, and response metric. Checkpoints, simulator RGB,
and third-party repositories are deliberately excluded from this artifact.
They must be obtained from their official sources under the corresponding
licenses.

The clean export will expose GPU smoke and full-inference commands only after
every adapter has:

1. a fixed upstream repository revision and checkpoint digest;
2. a standalone environment definition;
3. a one-instance structural regression test;
4. portable relative paths and resume behavior;
5. an anonymity and third-party-license audit.

## Anonymous archive

```bash
make artifact
```

The archive builder performs verification and anonymity checks before writing
a deterministic ZIP. It does not include `.git`, local build products, caches,
model weights, or non-allowlisted files.

