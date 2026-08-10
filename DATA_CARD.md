# LandMarkVLN data card

## Summary

LandMarkVLN is a fixed diagnostic analysis set for studying instruction--vision
alignment in vision-and-language navigation. The frozen release contains 937
landmark-centered instances evaluated with eight navigation VLA models.

## Intended uses

- Audit whether a model's attribution concentrates on an instruction-named
  landmark.
- Compare attribution evidence with localized target-versus-control
  interventions.
- Study differences across current-view, memory, and combined temporal scopes
  when those scopes exist in a model interface.
- Reproduce the reported model-specific point estimates and statistical tests.

## Out-of-scope uses

- Estimating landmark prevalence in the complete R2R or RxR populations.
- Treating automatic landmark masks as human pixel-level ground truth.
- Ranking models by raw causal-effect magnitude across incompatible native
  action spaces.
- Treating the common action diagnostic as closed-loop success, SPL, or a
  deployment-safety measure.

## Source assets and redistribution

The analysis derives from R2R/RxR navigation records and Matterport3D visual
assets. Raw RGB scans, simulator assets, and model checkpoints are not
redistributed here. Users must obtain upstream assets separately and comply
with their licenses and access conditions.

The frozen model-output tables contain deterministic instance identifiers,
dataset/split labels, registered attribution measurements, model-native causal
responses, and replay identities used for grouped inference.

## Construction summary

The construction pipeline applies text screening, current-view visual
grounding, temporal same-instance verification, matched-control generation,
deduplication, and final integrity checks. Evaluated-model outputs are not used
to select analysis-set membership.

## Limitations

- The set is diagnostic rather than a prevalence sample.
- Automatic open-source grounding consensus is not equivalent to human pixel
  annotation.
- Architectures expose different visual histories and action spaces.
- A null registered attribution or causal result does not prove that a model
  never uses the landmark.
- Conclusions are bounded by the released checkpoints, decoding rules, source
  datasets, and frozen analysis definitions.

## Versioning

The review artifact is immutable and hash-bound. Any change to membership,
model checkpoint, registered estimator, or causal outcome requires a new
version rather than silent replacement.

