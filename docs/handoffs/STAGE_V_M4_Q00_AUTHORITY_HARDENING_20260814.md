# Stage V Q00 authority hardening — 2026-08-14

Status: `READY_FOR_ZERO_TREATMENT_Q00_REVIEW`.
The authority hardening is complete. No Q00 canary, formal M4 branch,
intervention, label read, V_phys generation, attack/VIS evaluation, Eval160
read, or protected evaluation was run.

## Live repository and CI

```text
PR: #112
state: OPEN / DRAFT / mergeable
runtime implementation commit: f087f4060cd6f191de25ffddf45edd8037b1eadd
runtime implementation tree:   99e8976e348145d6603470f629cfae1d4a291b5e
branch: codex/m4-corridor-replenishment-post-32-of-40-hold-20260813
```

This append-only handoff was added in docs commit
`158e1a18` on top of that implementation commit; later docs-only commits do
not change the runtime bindings above.

The three Actions for the final head are successful:

```text
cpu-stageb       run 31759998010  SUCCESS
cpu-detector-v5  run 31759998064  SUCCESS
cpu-b3-official  run 31759998076  SUCCESS
```

No unresolved inline review threads were returned. Keep the PR draft; do not
merge.

## Authority hardening

The new `STAGE_V_M4_Q00_ZERO_TREATMENT_AUTHORITY_V1` validator is in
`scripts/detector_v5/stage_v_m4_q00_authority.py` and is required by
`run_stage_v_m4_zero_treatment_auditor.audit_probe` before any frozen action
replay. It rejects:

- a missing owner authorization basis;
- any `formal_m4_authorized=true` or treatment scope;
- M4 V2 that is not `FROZEN_PROSPECTIVE_NOT_AUTHORIZED` with
  `runtime_authorized=false`;
- any Q00 identity other than
  `libero_10/task_01/state_42 / Q00 / step 46`;
- incomplete `CONTROL/T3/T5/T10` exact-plan identity closure;
- snapshot, exact-plan, auditor, runtime-diff, or provenance SHA mismatch;
- dirty or mismatched runtime provenance;
- `>=20480`, a second project worker per GPU, or foreign-process interference;
- any nonzero treatment, forced-open, label, V_phys, outcome, or protected
  counter field.

The validator is deliberately separate from formal M4 authorization. A valid
Q00 authority must have `formal_m4_authorized=false`; it cannot promote the
formal M4 protocol.

## Frozen Q00 bindings

```text
M4 V2 protocol:
  path: /mnt/sdc/dty_user/openvla_attack_worktrees/codex-stage-v-q00-auth-99e8976e/configs/STAGE_V_M4_MATCHED_ACTION_PROTOCOL_V2.json
  sha256: f21e8b689357e6269b0fe78c379d937a68c9bc0bcf3102fc085b29fabebc20a4
  status: FROZEN_PROSPECTIVE_NOT_AUTHORIZED

exact-plan manifest:
  path: /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_M4_EXACT_40X24_PLAN_ONLY_V1_20260813T064532Z/EXACT_PROBE_AND_SNAPSHOT_MANIFEST.json
  sha256: 0c6caa4dd6638a5d06dfb5f36cc3a7270000b988b3e16e97e7008dd017a3fdf0

Q00 snapshot manifest:
  path: /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_M4_EXACT_40X24_PLAN_ONLY_V1_20260813T064532Z/parents/libero_10__task_01__state_42/CAUSAL_SNAPSHOTS/Q00/CAUSAL_PROBE_SNAPSHOT_V2.json
  sha256: cc01c01cd87c5394af531db7618027e91735c2a78f2f1aee835c58bbfe729b20
  parent: libero_10/task_01/state_42
  probe: Q00
  step: 46
  snapshot source commit/tree: 3bb52975f2f48bb8250d02aa00d7f2fbcc633a65 / 7d9870d4897a6314fdb8169ea7d0b5d998ec73b4

current code source at this handoff:
  commit/tree: f087f4060cd6f191de25ffddf45edd8037b1eadd / 99e8976e348145d6603470f629cfae1d4a291b5e
```

The source-plane difference is retained as a diagnostic binding. It is not
silently reinterpreted as a causal explanation and does not authorize a
formal rebind.

Current implementation file hashes:

```text
scripts/detector_v5/stage_v_m4_q00_authority.py
  32b394bc7f53dbdf7d1c23973ca4627cd8b84d608c6777036dcfa7db22c9c88a
scripts/detector_v5/run_stage_v_m4_zero_treatment_auditor.py
  1e3614e6b2cb971216f5582ce08592d12608bd5eb7a157295833413c8d86d8da
scripts/detector_v5/stage_v_runtime_diff.py
  a33342309bb53ab245c16a561cc678604a19e781a7ab103b1a76a8cbd468bd4a
scripts/detector_v5/stage_v_gpu_resource_contract.py
  dc92b4148dc2b568aa5c5826ade97eb25db78f753a075e4fa399c82a3ccbd404
```

## Verification

```text
official environment:
  /mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python
server test worktree:
  /mnt/sdc/dty_user/openvla_attack_worktrees/codex-stage-v-q00-auth-99e8976e
server worktree HEAD/tree:
  f087f4060cd6f191de25ffddf45edd8037b1eadd / 99e8976e348145d6603470f629cfae1d4a291b5e

targeted official-environment tests at the final head: 9 passed
full tests/detector_v5 at the final head: 342 passed, 6 skipped
local Python compile: PASS
git diff --check: PASS
```

Direct SSH is `dty` (`dty_user@pm-364c0001`), with no jump host. The formal
historical worktree remains clean at
`8951731c34595bcc73295f3c9c7390b13a219154 / 8ba22918c8c3fb1db4a6586525f3ab1b08e0d259`.
No formal M4 process was observed.

Immutable historical evidence remains unchanged:

```text
GLOBAL_HOLD.json: f5e3b37db9c438e0c9219d56b8bf92e7d02d6e8c40c69f03bc6986c9bc203db5
Parent00 closure: adfec5855839c80a6a38042573567dc4ee4c041f6294a00e8dc0cb6c9bd29242
Parent01 closure: 48e51bf359e146036f83aa12e16880e76af673eca7babbcf39f0a0831c21185c
protected counters: 0
```

The latest GPU sample was telemetry only: GPUs 0, 1, 4, and 6 had more than
20480 MiB free; GPUs 2, 3, 5, and 7 did not. Foreign compute processes were
left untouched. This sample is not a canary authorization and must be
rechecked immediately before any future lease.

## Authority decision and next action

`Q00_AUTHORITY_HARDENING = PASS`.
`FORMAL_M4 = HOLD`.
`Q00_ZERO_TREATMENT_CANARY = NOT RUN`.
`V_PHYS = NOT GENERATED AS A VALID/CONSUMABLE RESULT`.
`ATTACK/VIS = NOT RUN`.
`EVAL160 = NOT READ`.
`PROTECTED_COUNTERS = 0`.

A future Q00 authority receipt must be a new machine-readable file with:

```text
schema = STAGE_V_M4_Q00_ZERO_TREATMENT_AUTHORITY_V1
status = PASS_Q00_ZERO_TREATMENT_AUTHORITY
canary_authorized = true
runtime_authorized = true
owner_authorized = true
formal_m4_authorized = false
```

It must bind the live runtime commit/tree and clean provenance receipt, the
exact files above, the fresh resource lease, and the exact Q00 identity. The
preflight must fail closed on any mismatch, release its lease, preserve the
receipt/diff, and stop. No rerun-to-pass is allowed.

This hardening does not itself constitute owner authorization and does not
launch the first GPU canary. The next legal action is explicit GPT/owner review
of that Q00 authority receipt and the clean-only launcher command.
