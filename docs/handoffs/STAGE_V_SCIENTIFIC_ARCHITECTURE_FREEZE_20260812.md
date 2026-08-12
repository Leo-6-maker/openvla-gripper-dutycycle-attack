# Stage V scientific architecture freeze — 2026-08-12

Status: `FROZEN_PROSPECTIVE_SCIENTIFIC_ARCHITECTURE`

This is an append-only scientific correction. It does not delete or rewrite
historical M4, Counterfactual/Hybrid Teacher, or R3 Teacher/Student evidence.
Those artifacts remain historical and are not silently promoted to the primary
paper pipeline.

## Primary pipeline

```text
CLEAN rollout
  -> privileged clean Teacher C_t
  -> causal Student C_hat_t trained only on clean Teacher labels
  -> held-out matched counterfactual V_t(d)
  -> timing / VIS / defense protocols later
```

The variables are separated as `C_t != V_t(d) != E_t`:

- `C_t` is clean-derived physical criticality from clean state and clean
  telemetry only. It is not a causal vulnerability label.
- `V_t(d)` is a held-out physical counterfactual outcome under frozen
  CONTROL/T3/T5/T10 conditions. Unknown or abstained outcomes are not
  negatives.
- `E_t` is visual exploitability and requires a separate VIS protocol.

The privileged Teacher may use clean physical state. The deployment-facing
Student may use only causal, deployment-visible observation/action/history,
gripper/eef/contact/phase/timing features. It may not use privileged Teacher
fields, M4 outcomes, attack/VIS/oracle/random outcomes, future or post-treatment
variables, identity/task leakage, or M4 dose as a training feature.

## Claim boundary

The main claim is two-tiered:

1. Physical phenomenon: a completed held-out M4 can test dose/state-dependent
   physical susceptibility inside the frozen critical-opportunity corridor.
2. Clean localization: a clean Teacher and clean-supervised causal Student may
   localize that susceptibility, but this is not established until their fresh
   primary evidence is evaluated against held-out `V_phys`.

The current 24-probe design contains only contact-positive
`CONTACT_MANIPULATION`, `CARRY`, and `ENGAGED_LIFT` states. It contains no
`PRE_CONTACT` or safe/background panel. Therefore it cannot support a
high-risk-versus-low-risk or critical-versus-safe enrichment claim. A new
negative-control panel would require a separate prospective protocol and was
not launched.

## Current evidence and gates

- Formal clean corridor: `29/40` stable parents; status
  `HOLD_FORMAL_M4_CORRIDOR_INSUFFICIENT`.
- Valid replenishment additions: `1 libero_10 + 6 libero_goal`; four
  `libero_spatial` parents remain.
- M4 intervention labels/outcomes: not started and not read.
- Historical R3 Teacher/Student artifacts exist, but their reports are
  development-only or coverage-HOLD and are `HISTORICAL_PRE_FREEZE_SECONDARY_OR_UNCLASSIFIED`.
- Current live audit found no project process, no compute app, and all eight
  A800 GPUs idle. No scientific runtime was started for this correction.
- Protected counters remain zero: protected reads, Eval160 reads, attack
  rollouts, and VIS/PGD attack rollouts.

## Required firewall and sequence

The final M4 parent set must be excluded from FIT, CAL, CHECK, threshold
selection, model selection, and outcome-informed redesign. The current known
36 identities are disjoint from FIT670 and all six G1 train/validation/test
manifests. They overlap the historical G10 identity registry in `36/36`, but
the G10 protocol records `G10_READ=false`; this is quarantined identity-level
overlap, not usable outcome evidence.

The legal sequence is:

1. finish the architecture and claim freeze;
2. obtain the four remaining spatial clean qualifications;
3. freeze one exact 40-parent manifest, split, and exact clean probe manifest;
4. rerun the complete firewall audit against the final primary FIT/CAL/CHECK
   manifests;
5. build and lock the clean Teacher/Student primary evidence package;
6. only then read held-out M4 outcomes and report the predeclared analysis.

No Teacher threshold or Student feature may be revised from M4 outcomes. The
old historical designs may be reported as secondary ablations only.

## Machine-readable artifacts

- [Architecture freeze](../../configs/STAGE_V_SCIENTIFIC_ARCHITECTURE_FREEZE_V1.json)
- [Claim/evidence matrix](../../reports/STAGE_V_CLAIM_EVIDENCE_ALIGNMENT_AUDIT_V1.json)
- [Primary data firewall audit](../../reports/STAGE_V_PRIMARY_DATA_FIREWALL_OVERLAP_AUDIT_V1.json)
- [M4 probe-support audit](../../reports/STAGE_V_M4_PROBE_SUPPORT_AUDIT_V1.json)
- [Current takeover update](STAGE_V_M4_CURRENT_TAKEOVER_UPDATE_20260812.md)

The repository source binding at freeze is commit
`eb05ae20ead190e91221ede6ae6d18fca70c2b30`, tree
`092f55c3fbce82421bfa8bca67cd192f321048d7`.

Semantic artifact SHA256 bindings:

- `configs/STAGE_V_SCIENTIFIC_ARCHITECTURE_FREEZE_V1.json`: `ef601cdfe5a4eadca07c2a7e372c8b2d66710c323a3697ef8da5eeadb17b6ee4`
- `reports/STAGE_V_CLAIM_EVIDENCE_ALIGNMENT_AUDIT_V1.json`: `896fbdb48b2a206846bf1dc171271d8a13bdaad2759a3a14133672ef4ca5b486`
- `reports/STAGE_V_PRIMARY_DATA_FIREWALL_OVERLAP_AUDIT_V1.json`: `0cadb59d01b860cfd9995499bb036e39918c1e84aaf1b8f8650e2a5d80a3ba3f`
- `reports/STAGE_V_M4_PROBE_SUPPORT_AUDIT_V1.json`: `98e977d5d22f294cc71b65ca5ff87b3c0c9749c0dfdd3d9524118087af8d2e59`
