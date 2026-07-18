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
- Development scope: FIT_TRAIN only, states 0--19.
- FIT-DEV, CAL, CHECK, final-parent data and attack results are not read.
- Formal training and attack authorization are hard-coded false in the V5
  development contracts.

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

## Not yet executed

- No V5 Teacher utility materialization.
- No V5 GPU training or checkpoint.
- No V5 validation or model selection.
- No FIT-DEV, CAL, CHECK, Direct-open, canary, or attack.
- No CLEAN/S1 mutation and no GitHub PR mutation at this snapshot.

## Required next gate

Run the FIT-only input availability audit first.  It must establish which of
the four variants have complete causal source streams, timestamp alignment,
and sealed non-privileged inputs.  Only after that audit passes should a
synthetic V5 Teacher utility census and a small CPU/GPU smoke be considered.

The scientific gate is not a larger GRU.  A V5 candidate must improve
matched-recall window selection, mixed-episode top-1 selection, and
pure-negative abstention under the frozen one-shot budget, with all exact
numerators and denominators sealed.

```text
V4_CONTINUATION             = STOPPED
V4_EVIDENCE                 = PRESERVED
V5_CODE_CONTRACT            = DEVELOPMENT_READY
V5_INPUT_AVAILABILITY       = NOT YET RUN
V5_TEACHER                  = NOT STARTED
V5_GPU_TRAINING             = NOT STARTED
FIT_DEV / CAL / CHECK       = NOT READ
ATTACK                      = NOT STARTED
```
