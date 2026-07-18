# Official V3 Detector V5 takeover

Snapshot: 2026-07-18.  This is a new development line after the V4
proprio-only screening.  V4 code and evidence remain historical references;
they are not overwritten or promoted.

## Change of objective

V4 asked whether a current step looked like valid retention.  V5 asks which
candidate close window is most worth one attack budget, using a causal
multimodal Student stream and an explicit abstention-capable one-shot
scheduler.

The clean-only Teacher output is a criticality/utility proxy.  It is not a
counterfactual attack label and cannot authorize an attack.

## V4 line frozen

- V4 R2.2 read-only reevaluation processes started by this agent were stopped
  after the V5 switch: PIDs 1627386, 1627387, 1627388, and 1627389.
- The four processes were verified as `dty_user` jobs running the temporary
  `official_v3_v4_r2_2` evaluator overlay before termination.
- All V4 checkpoint, prediction, Teacher, and negative-result roots remain
  untouched.  No V4 root is deleted or overwritten.
- PR #86 remains the historical Draft V4 review line.

## V5 source line

- Branch: `codex/official-v3-detector-v5-20260718`
- Base: Official archive commit
  `5e27d7c4b1a188bc6a78555f94d2571222587805`
- Draft PR: [#87](https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack/pull/87)
- Development scope: FIT_TRAIN only, states 0--19.
- FIT-DEV, CAL, CHECK, final-parent data and attack results are not read.
- Formal training and attack authorization are hard-coded false in the V5
  development contracts.
- Current branch HEAD: `42dcc906f7db5f65347e4472f569b0d4990b6605`.

## V5 contracts now implemented

1. `DETECTOR_V5_DEVELOPMENT_PROTOCOL_V1` freezes four controlled variants:
   proprio-only, proprio plus policy intent, proprio plus causal visual, and
   all three streams.
2. `DETECTOR_V5_TEACHER_UTILITY_V1` separates clean privileged/future Teacher
   evidence from causal Student inputs, uses utility tiers 0--3, and requires
   XOR-safe known supervision.
3. `CausalMultimodalVulnerabilityRanker` uses mask-aware causal recurrent
   branches, gated fusion, utility/release/regrasp/support/uncertainty heads,
   and a differentiable within-episode ranking loss.
4. `V5OneShotScheduler` enforces candidate-close gating, 3-of-5 persistence,
   release/regrasp/uncertainty vetoes, abstention, and one emit per episode.
5. `audit_v5_input_availability.py` is a read-only FIT-only census.  It
   distinguishes V5-A readiness from policy-intent and causal-visual source
   availability and refuses non-FIT identity rows.

The first V5 utility root had no tier-3 windows and is retained as a
diagnostic.  After freezing the T10 proxy rule, the current V5 Teacher root
and independent audit are:

- Teacher root: `OFFICIAL_V3_DETECTOR_V5_TEACHER_UTILITY_V3_c823653_20260718`;
- Teacher root seal SHA256:
  `60a7def4ae35d760f10515af1cc134cc7aa423442538e5a7bb9d156da8fb56aa`;
- manifest SHA256:
  `ddac8c925a00b5566cc1895247f90bb703d4504ca3a57d210bc773a043824004`;
- independent audit:
  `OFFICIAL_V3_DETECTOR_V5_TEACHER_UTILITY_AUDIT_c1f955b_20260718`;
- audit PASS: 800 identities, 176,336 steps, 68,892 known steps;
- utility tiers: tier1=14,500, tier2=794, tier3=53,598;
- source V2.1.3 seal binding:
  `8dd727978e2cfeebf41a3964f27b259390ff841e1b0fcd435ee4ec7b8c87c9a7`.

Tier3 is explicitly a clean-only T10-or-longer `VALID_RETENTION` proxy.  It
must not be described as measured attack vulnerability.

## V5-A development smoke

The first smoke used only V5-A, fold 0, seed `20260717`, 32 train episodes,
2 epochs, FP32 on physical GPU4 in `openvla-official-a800`.  It is not a full
fold run.

- output root:
  `OFFICIAL_V3_DETECTOR_V5_A_SMOKE_F0_S20260717_35db39d_20260718`;
- root `SHA256SUMS` SHA256:
  `732d10a6f6fbcebb2267861c69147a42a7e1249650b07883cdf8b778ac14058b`;
- checkpoint SHA256:
  `2dc8191d6a8ef70134f5ef01bdda14fea5db44da7dead1a158324bd12a9bb0d2`;
- diagnostic top-1 highest-utility hit: 155/182 = 0.8516;
- pure-negative abstention at 0.5: 8/18 = 0.4444.

The smoke bundle is explicitly `eligible_for_model_selection=false`,
`formal_training_authorized=false`, and `formal_attack_authorized=false`.
These metrics are engineering diagnostics only and do not form a V5
viability decision.

## Not yet executed

- No V5 Teacher utility materialization beyond the sealed clean-only proxy
  root above.
- No formal V5 training or model-selection checkpoint.  The bounded V5-A
  development smoke above is the only GPU execution.
- No V5 validation or model selection.
- No FIT-DEV, CAL, CHECK, Direct-open, canary, or attack.
- No CLEAN/S1 mutation.  PR #87 is a Draft; it has not been marked Ready or
  merged.

## FIT input availability audit

The read-only audit root is:

`OFFICIAL_V3_DETECTOR_V5_INPUT_AUDIT_17509e7_20260718`

The complete 2000-row registry was filtered mechanically to states 0--19;
the audit did not inspect any other split.  Results:

- registry-derived FIT identities: 800;
- S1 episode manifests: 800/800;
- Student 25D/data-contract rows: 800/800 PASS, 0 HOLD;
- S1 tree digest: `8f679082651c54a4ac8843e406c61c7a2867eaad21e934759c2755d6c4f1a29f`;
- registry CSV SHA256:
  `09f71b3a9b8250c80735382ba5deab6dbcadfa21b645e4a981eefb114b236af5`;
- policy-intent root: NOT SUPPLIED/NOT FOUND in the current evidence tree;
- causal-visual root: NOT SUPPLIED/NOT FOUND in the current evidence tree.

The prior policy-intent export log ends with a fail-closed source-audit
failure at `libero_10/task_00/state_00`; it did not create a 9D root.  A
recursive evidence-root search found no PNG/JPG/JPEG/NPZ visual source.  This
means the current sealed corpus supports V5-A (proprio-only) development, but
V5-B/C/D are conditional and must not be trained by fabricating or silently
reusing a missing stream.

## Required next gate

The FIT-only input availability audit, sealed V5 proxy-root audit, and bounded
V5-A development smoke are complete.  The next gate is a fresh, non-overwrite
full-FIT V5-A development run plus the sealed phase/window evaluator.  The
current smoke is not a scientific pass: top-1 selection is useful as an
engineering signal, but pure-negative abstention is only 8/18 at threshold
0.5.  Policy-intent and visual variants remain blocked until their source
roots exist and pass the same audit.

The scientific gate is not a larger GRU.  A V5 candidate must improve
matched-recall window selection, mixed-episode top-1 selection, and
pure-negative abstention under the frozen one-shot budget, with all exact
numerators and denominators sealed.

```text
V4_CONTINUATION             = STOPPED
V4_EVIDENCE                 = PRESERVED
V5_CODE_CONTRACT            = DEVELOPMENT_READY
V5_INPUT_AVAILABILITY       = PASS_DATA_ONLY; V5-A READY, V5-B/C/D HOLD
V5_TEACHER                  = PASS CLEAN_ONLY_PROXY; NOT ATTACK LABELS
V5_GPU_TRAINING             = DEVELOPMENT SMOKE PASS; FORMAL NOT STARTED
FIT_DEV / CAL / CHECK       = NOT READ
ATTACK                      = NOT STARTED
```
