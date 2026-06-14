# M3 Step78 RAND20 Selectivity Audit

## Result

`RANDOM_SELECTIVE_MATCH_EXISTS`

This exploratory audit official-decodes all 20 frozen RAND20 candidates from
the original M3 step78 canary. It does not change the preregistered v1 result:
`RANDOM_NOT_BEATEN`.

## Counts

- candidate count: `20`
- official 31744 count: `15`
- selective 31744 count, arm prefix >= 5/6: `13`
- nonselective 31744 count: `2`
- best official margin: `5.75`
- best official margin candidate: `0`

## Interpretation

The preregistered v1 RAND20 selected candidate `0`, which emitted `31744`
but only matched the clean arm prefix at `4/6`. That selected candidate remains
nonselective and the original v1 terminal result remains `RANDOM_NOT_BEATEN`.

However, the full official audit found `13/20` frozen random candidates that
both emitted `31744` and matched the clean arm prefix at least `5/6`. This
means the step78 frame is strongly random-sensitive under the v1
processor-space random control. The result is not merely a post-hoc objection
to one nonselective selected random sample.

The v1 hinge objective also saturates at the margin threshold, so high-margin
random candidates can sit in the same zero-loss set as TRUE_PGD. The next
development step should keep the target token and frame fixed but use the
pre-registered non-saturating log-ratio objective in a separate CPU-only
implementation PR before any new GPU canary.

## Provenance

- source candidate CSV: `tables/m3_step78_canary_candidate_controls_af545e1_seed80.csv`
- output directory: `/data/liuyu/outputs/m3_step78_true_pgd_20260614/rand20_selectivity_5932266_seed80`
- commit: `59322666de1aa3c228d100cfe23c1ea853a322ed`
- input hash check: `PASS`
- GPU mapping: `CUDA_VISIBLE_DEVICES=2,6`
- rollout status: no LIBERO rollout launched

## Claim Boundary

Allowed claim: the frozen RAND20 candidate set was official-decoded for
selectivity auditing with hash-verified candidate reconstruction. On this
frozen development frame, random selective 31744 matches exist in the
preregistered RAND20 candidate set.

Forbidden claim: this does not establish true-PGD superiority, closed-loop
critical-closure disruption, paired task effect, or held-out transfer.
