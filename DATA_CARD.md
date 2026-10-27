# LandMarkVLN data card

## Summary

LandMarkVLN is a diagnostic analysis set for studying whether navigation VLA
models localize instruction-named landmarks and exhibit landmark-specific
output sensitivity.

The release contains 7,228 samples from 4,025 source navigation trajectories.
Eight models process the same membership, producing 57,824 model-sample rows.

The evaluated models are OneVLA, InternVLA-N1, OmniVLA, Uni-NaVid,
FantasyVLN, StreamVLN, VAMOS, and TIC-VLA.

## Composition

LandMarkVLN draws from the continuous-environment releases of R2R and
English-language RxR.

| Source split | Samples | Source trajectories |
|---|---:|---:|
| R2R train | 578 | 432 |
| R2R val-seen | 37 | 27 |
| R2R val-unseen | 111 | 77 |
| RxR train | 5,044 | 2,700 |
| RxR val-seen | 578 | 304 |
| RxR val-unseen | 880 | 485 |
| **Total** | **7,228** | **4,025** |

Each sample connects:

- a navigation instruction and landmark-bearing local instruction segment;
- one singular physical landmark;
- a current observation and recent observation history;
- temporally aligned landmark masks;
- a local expert-action horizon;
- registered landmark and matched-control interventions.

## Construction

The construction pipeline proceeds through:

1. text-only selection of singular, physical landmark referents;
2. current-view grounding and geometric verification;
3. temporal visibility resolution and same-instance verification;
4. matched-control generation, deduplication, and integrity checks.

FGR2R and Landmark-RxR provide fine-grained route supervision for the initial
alignment. Evaluated-model outputs do not determine sample membership.

## Released records

The repository distributes eight gzip-compressed JSONL files, one per model.
Together they occupy about 77 MiB. These are diagnostic measurements rather
than the raw visual-language dataset: instruction and landmark text, RGB
frames, pixel masks, and full attribution maps are not included.

Each row contains analysis-ready measurements for one model and one shared
LandMarkVLN sample. The principal fields are:

| Field | Meaning |
|---|---|
| `probe_id` | Shared sample identifier used for cross-model membership |
| `replay_id` | Source navigation trajectory used for grouped inference |
| `task_index` | Stable row index from 0 through 7,227 |
| `dataset`, `split` | R2R/RxR source and source split |
| `A` | Local, horizon-matched action diagnostic |
| `A_by_context` | Context-specific action diagnostic where required |
| `A_horizon` | Local action-evaluation horizon metadata |
| `axv.sub` | Local-instruction attribution results |
| `causal` | Model-native matched and broad intervention responses |

`axv.sub` contains attention-only, gradient-only, and attention × gradient
estimators. Temporal models record current (`C`), memory (`M`), and combined
(`CM`) scopes. OmniVLA and VAMOS record only the current scope.

The causal schema preserves each model's native response family and units.
InternVLA-N1 has separate continuous and discrete native branches.

## Validity and missingness

Validity is part of the measurement schema. Relevant indicators include
`valid`, `parse_abstention`, `abstention_reason`, and model-native distance or
branch-validity fields.

The analysis applies these rules before constructing each denominator:

- valid finite measurements contribute to the corresponding panel;
- abstained or malformed attribution cells are excluded;
- invalid native causal distances and unavailable branches are excluded;
- invalid measurements are not converted to false detections or zero effects;
- unavailable memory and combined scopes for current-only models are
  structural not-applicable values.

As a result, panel-specific valid counts can differ from 7,228.

## Intended uses

The release supports reproduction of the 60 VxA analyses, 23 matched
intervention analyses, 23 broad-response summaries, and 23
attribution-intervention Spearman analyses. It also supports inspection of
validity and temporal-scope behavior and alternative analyses that preserve the
sample-level and trajectory grouping structure.

## Interpretation

`A` evaluates local action agreement over the registered horizon. It is not an
episode-level success or SPL measure.

`V` records whether target attribution mass passes the registered
matched-translation diagnostic. Landmark-specific output sensitivity is
evaluated separately through interventions.

Causal effects remain in each model's native response units. Their direction
and within-model uncertainty follow a common analysis protocol, while raw
magnitudes do not define a cross-model leaderboard.

LandMarkVLN is selected for resolvable, intervention-ready landmarks. Its rates
do not estimate landmark prevalence in the complete R2R or RxR populations.
Grounding and temporal verification are registered machine-generated
measurements rather than human pixel annotation.

## Source assets and licensing

Construction uses R2R, English-language RxR, FGR2R, Landmark-RxR, and
Matterport3D resources. Model inference uses the evaluated models' upstream
repositories and checkpoints.

The release contains analysis-ready numerical records. Obtain imagery,
simulator assets, source datasets, model checkpoints, and GPU environments from
their original providers under their licenses, access agreements, and usage
terms.

## Privacy

The released gzip records contain numerical diagnostics, dataset labels, and
sample or trajectory identifiers. They contain no RGB frames and add no
participant data or personal attributes. The underlying Matterport3D imagery
remains governed by its original access and privacy conditions.
