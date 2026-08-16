# Stage VIII R1 Relative Selector Negative Handoff

Date: 2026-08-17

## Terminal decision

`STAGE_VIII_R1_NO_GENERALIZABLE_RELATIVE_SELECTOR`

The frozen R1 study did not produce a generalizable passive relative-timing
selector. Per the owner decision, this is a valid scientific negative result.
Stop the Stage VIII passive-selector line here. Do not open a scheduler
pre-holdout, fresh M4, physical timing matrix, PGD rollout, Eval160, or
protected evaluation from this result.

R0 remains positive for descriptive identifiability: immutable S7-A/S7-B/S7-C
scores contained within-parent relative signal. R1 shows that the bounded
causal selector candidates did not generalize strongly enough to pass the
frozen promotion gate.

## Exact execution binding

- PR: [#117](https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack/pull/117)
- PR state: OPEN / DRAFT / MERGEABLE
- source commit: `c03cb32b14d38978239e53adc953cc8620b775a1`
- source tree: `6f5cff89855fbab3bbc06b92c170586db9e0091d`
- official environment: `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`
- server worktree: `/mnt/sdc/dty_user/openvla_attack_worktrees/stage-viii-r1-c03`
- CI: `cpu-b3-official-v3`, `cpu-detector-v5`, and `cpu-stageb` all SUCCESS

The run was CPU-only with `CUDA_VISIBLE_DEVICES` empty. No GPU worker,
intervention, M4, or PGD rollout was started.

## Frozen inputs and sealed output

- R1 protocol: `configs/STAGE_VIII_R1_RELATIVE_SELECTOR_PROTOCOL_V1.json`
- R0 root:
  `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_VIII_RELATIVE_TIMING_SELECTOR/STAGE_VIII_R0_RELATIVE_TIMING_IDENTIFIABILITY_20260816T154705Z`
- R1 root:
  `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_VIII_RELATIVE_TIMING_SELECTOR/STAGE_VIII_R1_RELATIVE_SELECTOR_20260816T161050Z`
- R1 summary SHA256:
  `3ace17e589529c420e59deceeada7d8dbf84efb775d1402999b53702e5f51b06`
- R1 `SHA256SUMS` SHA256:
  `552a7180b37ce337560e541ea2a264c246888bb6aa2e219694296ce99fab0ea6`
- R1 `SHA256SUMS.sha256` SHA256:
  `2c106f7784607f06466c8cf39ac231719081a46ec6dd1f5160bffca0b39bbbb1`
- R1 `ROOT_SEAL.json` SHA256:
  `cbcc55b350e59cc222e76f944a19f1f430b69e6f64363531823637530593f7a1`
- R1 `ROOT_SEAL.sha256` SHA256:
  `088ca0d4a9649ded53772aee87d0e07999fbe8bd72f27d4ad5441a8d36afa566`

Independent checks completed successfully:

```text
sha256sum -c SHA256SUMS
sha256sum -c SHA256SUMS.sha256
sha256sum -c ROOT_SEAL.sha256
```

The root seal records `candidate_training_performed=true`,
`new_m4_authorized=false`, `intervention_executed=false`, and
`pgd_authorized=false`.

## Population closure

- all rows: `4032`
- T5 rows: `1344`
- consumable T5 rows: `1191`
- parent population: `56`
- parent split: TRAIN `32`, VAL `11`, DEVTEST `13`
- abstains were masked and never converted to negatives
- pair unit: same `(stage, canonical_parent_key)` only
- model input: frozen `16x25D` causal history
- R1-B context: frozen clean language/visual embeddings plus clean policy 9D;
  no suite/task identifiers

## Frozen DEVTEST gate results

| candidate | parent-macro AUC | top-1 lift | top-3 lift | zero-regret | LOSO mean | LOSO worst | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| R1-A | 0.577615 | 1.049746 | 0.979762 | 0.555556 | 0.627747 | 0.475000 | FAIL |
| R1-B | 0.658572 | 1.049746 | 0.979762 | 0.555556 | 0.639665 | 0.477851 | FAIL |
| frozen minimum | 0.720000 | 1.500000 | 1.300000 | 0.700000 | 0.620000 | 0.550000 | — |

R1-A DEVTEST parent-macro AUC by suite:

```text
libero_10      0.871212
libero_goal    0.000000
libero_object  0.558730
libero_spatial 0.037500
```

R1-B DEVTEST parent-macro AUC by suite:

```text
libero_10      0.888037
libero_goal    0.012500
libero_object  0.766667
libero_spatial 0.062500
```

R1-B improved parent-macro AUC over R1-A by `+0.080956`, but its own frozen
gate failed. Its worst-suite improvement was only `+0.0125`, and neither
candidate met the top-k or zero-regret requirements. Therefore the frozen
context replacement rule does not authorize promotion.

## Protected boundary

```json
{
  "protected_reads": 0,
  "eval160_reads": 0,
  "attack_rollouts": 0,
  "vis_pgd_attack_rollouts": 0
}
```

`Eval160` and protected evaluation remain `UNREAD`.

## Scientific conclusion and stop condition

The current evidence supports the narrower conclusion:

> Relative timing signal is present descriptively in the immutable Stage VII
> development scores, but the two pre-registered Stage VIII R1 selectors do
> not establish a generalizable deployment-facing relative vulnerability
> selector under the frozen parent-grouped and LOSO gate.

Do not reinterpret this as a positive detector, do not tune the threshold or
features, do not replace parents, and do not rerun R1 to pass. Any new detector
architecture or new physical validation population requires a separate owner
decision and a new protocol namespace.
