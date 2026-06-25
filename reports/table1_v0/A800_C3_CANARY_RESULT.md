# A800 C3 Exact-Prefix Canary Result

Date: 2026-06-25

## Decision

```text
A800_C3_MIGRATION_PARITY: FAIL_AGENTVIEW_BYTE_DRIFT
C3_EXACT_OBSERVATION_CONTRACT: FAIL
OBJECT_CLEAN_REPLAY: BLOCKED
TABLE1_ATTACKS: BLOCKED
```

The frozen Goal parent was:

```text
libero_goal|task=4|state=1|eval_seed=0|condition=CLEAN
```

## Attempts

| Attempt | Code | GPUs | Result | First divergence |
|---|---|---:|---|---|
| Production C3 | `fad471e6` | 5,6 | FAIL | post-step observation SHA, step 8 |
| Diagnostic C3 | `8bbe1dea` | 5,6 | FAIL | post-step observation SHA, step 2 |

The diagnostic attempt found that only `agentview_image` differed:

```text
shape: 256x256x3
dtype: uint8
different pixels: 2
maximum absolute difference: 1
mean absolute difference: 1.0172526041666666e-05
first differing index: [125, 195, 2]
```

All compared nonvisual observation fields were exact, including joint state,
gripper state, EEF pose, object poses, proprioception, and flattened object
state. This is consistent with renderer-level byte nondeterminism, but it does
not prove a hardware root cause.

## Evidence Integrity

The diagnostic producer wrote the recursive manifest and then mutated
`c3_prefix_replay_summary.json` to add the manifest digest. Independent
recalculation therefore found:

```text
manifest rows: 16
mismatched rows: 1
mismatched path: c3_prefix_replay_summary.json
producer seal status: FAIL
```

The failure is a tooling/provenance defect, not a change to the C3 scientific
outcome. The runner is repaired on this branch so the summary is finalized
before the recursive manifest is generated. The existing server output remains
unchanged and is explicitly classified `SEAL_MISMATCH_DISCOVERED`.

## Allowed Claim

On the tested A800 Goal parent, exact action-prefix replay reproduced all
inspected nonvisual observation fields but failed the frozen exact observation
contract because the rendered agent-view image differed at byte level.

## Forbidden Claims

- A800 migration parity passed.
- Exact replay is scientifically qualified on A800.
- VIS, RAND, TMA, or the proposed method was evaluated.
- VIS is better than random.
- Any Table 1 attack or task-effect result exists.
