# Provenance inventory

This inventory separates artifacts already safe for the clean export from
components that still require portability or licensing work. Paths below are
repository-relative labels; private filesystem and scheduler paths are not
release metadata.

## Frozen scientific inputs included

| Component | Exported form | Integrity gate |
|---|---|---|
| Final analysis membership | Cross-model instance identities in every per-instance table | 937 unique rows and identical membership |
| OneVLA | `artifacts/frozen/models/onevla/per_probe_metrics.jsonl` | source SHA-256 plus structural checks |
| InternVLA-N1 | `artifacts/frozen/models/internvla_n1/per_probe_metrics.jsonl` | source SHA-256 plus structural checks |
| OmniVLA | `artifacts/frozen/models/omnivla/per_probe_metrics.jsonl` | source SHA-256 plus structural checks |
| Uni-NaVid | `artifacts/frozen/models/uni_navid/per_probe_metrics.jsonl` | source SHA-256 plus structural checks |
| FantasyVLN | `artifacts/frozen/models/fantasyvln/per_probe_metrics.jsonl` | source SHA-256 plus structural checks |
| StreamVLN | `artifacts/frozen/models/streamvln/per_probe_metrics.jsonl` | source SHA-256 plus structural checks |
| VAMOS | `artifacts/frozen/models/vamos/per_probe_metrics.jsonl` | corrected canonical source SHA-256 plus structural checks |
| TIC-VLA | `artifacts/frozen/models/tic_vla/per_probe_metrics.jsonl` | source SHA-256 plus structural checks |

The VAMOS input follows the operative canonical-artifact correction: the file
is `per_probe_metrics.jsonl`, not the obsolete filename recorded in the earlier
narrative document.

## Pending admission to the clean export

| Component | Current authority | Required gate before inclusion |
|---|---|---|
| Data-construction executables | Final open-source 937 construction scripts and frozen rules | remove operational paths; create portable inputs; clean-environment test |
| Eight model runners | Final model-specific run/merge/audit scripts | pin upstream revisions; isolate environments; one-instance smoke tests |
| Full registered statistics | Final model-specific analysis and independent-audit scripts | replace scratch inputs with exported tables; reproduce all intervals/tests |
| Paper tables and figures | Frozen manuscript data and render scripts | bind each output to exported inputs and expected-result tolerances |
| Dataset metadata | Source-data manifests and licenses | complete redistribution and access review |

Historical populations, discarded screening variants, scheduler recovery
records, raw job logs, and identity-bearing operational manifests are not
release candidates.

