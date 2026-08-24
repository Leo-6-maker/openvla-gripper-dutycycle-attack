# Canonical main integration audit V1

Status: `MAIN_INTEGRATION_AUDIT_PASS`

Audit date: 2026-08-24

Scope: PR `#138`, canonical integration of the complete Stage X/F1/Stage Z
and repository-hygiene lineage into `main`. This is a repository-history and
software audit, not a scientific promotion.

## Exact lineage

- old `main`: `d68d4e3a3ce5b2f3b40fb69efbea9d09e0d8e14e`
- PR #135 HEAD: `7f095965df61e8065d5c76fc2be4504bed4d9ab6`
- PR #136 HEAD: `6c3c326d7dbe5bb98c05cd37cca096cd4ff96020`
- canonical integration merge: `ec2eb18eeaad8928f43b04d26f54e54955c29176`
- canonical integration tree: `0f6705a5ce6a43d19e0b7b4c67658680a75e3513`
- merge parents, in order: old `main`, exact PR #136 HEAD

Git ancestry checks pass: PR #135 is an ancestor of PR #136, and PR #136 is
an ancestor/parent of the canonical merge. No squash, rebase, cherry-pick, or
history rewrite was used.

## Merge-tree audit

The read-only merge simulation found one overlapping path: the stale root
`README.md`. The canonical merge uses the exact PR #136 authority-map README
blob `70bcdef62a7a7e34df99ce5f0c655d5ae9bda8ed`.

Relative to PR #136, the merged tree only retains the main-only historical
`reports/C2F_CODEX_HANDOFF_20260710.md`. The old-main-to-integration path audit
contains no deleted or renamed file entry. No unresolved merge entry remains.

The pre-existing branch
`codex/stage-z-cross-model-physical-replication-20260824` is intentionally not
silently mixed into this integration. It remains preserved at
`2e1e9fde5a2b9c0206f0f40ed8f7792156b5ac22`, one documentation-only commit
beyond PR #135. A new Stage-Z branch must be created from canonical `main`.

## Authority and deterministic-export audit

The following commands passed on the canonical merge:

- `CODE_R1_AUTHORITY_FIREWALL_PASS entries=88 sidecar_pairs=14 git_objects=2`
- `PAPER_V1_CLAIM_AUDIT_READ_ONLY_PASS claims=48 eval160=UNREAD protected=UNREAD`
- `PAPER_V2_EXPORT_CHECK_PASS source_head=a3a1846939df4be4c04de5c1de8bfee541b4d656 files=8`
- combined repository, Paper V2, Stage X, primary-matrix, and Stage Z static
  suite: `88 passed, 1 skipped`

The authority audit confirms that registered working-tree blobs, sidecar
digests, and historical Git-object bindings remain resolvable. Green checks do
not reinterpret or promote any scientific claim.

## Remote CPU CI

All seven pull-request CPU workflows passed on merge HEAD `ec2eb18e...`:

| Workflow | Result | Run |
| --- | --- | --- |
| `cpu-detector-v5` | PASS | `32697721598` |
| `cpu-factorized-l3` | PASS | `32697721547` |
| `cpu-factorized-phase-c` | PASS | `32697721630` |
| `cpu-pilot-analysis` | PASS | `32697721604` |
| `repository-hygiene` | PASS | `32697721548` |
| `cpu-b3-official-v3` | PASS | `32697721741` |
| `cpu-stageb` | PASS | `32697721557` |

The final PR HEAD after adding this audit record must repeat the applicable
CI checks successfully before integration into `main`.

## Frozen scientific boundary

The controlling Stage Z root remains:

`HOLD_STAGE_Z_Z0R2_OFT_CHECKPOINT_AUTHORITY_NOT_ESTABLISHED`

Its only blocker remains `M1 four-suite server byte manifests not yet sealed`.
`scientific_rollout_started=false`; model inference, GPU workers, simulator,
`env.step`, physical intervention, `V_phys`, Eval160, and protected-read
counters are all zero.

This integration does not authorize Z1, model loading, GPU execution,
simulator use, attack generation, physical intervention, or protected access.
It does not reopen Stage X/F1, F1-D, or BRIDGE.

## Disposition

The canonical merge preserves the complete #135/#136 lineage, immutable
authority paths, Paper V2 export pipeline, and current Stage Z HOLD. After the
audit-record CI passes, PR #138 may be integrated with history-preserving merge
semantics. Squash merge is prohibited.

`MAIN_INTEGRATION_READY_FOR_HISTORY_PRESERVING_MERGE`
