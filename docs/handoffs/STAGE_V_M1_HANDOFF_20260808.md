# Stage V M1 / RB1A Handoff — 2026-08-08

## Current status

```text
RB1A                         HOLD
M1-R0                        PASS
M1-R1                        NOT_STARTED
M1-R2                        NOT_STARTED
M1 final classification     UNCLASSIFIED
current M1 verdict          HOLD_WAIT_FOR_SAFE_GPU
new qualification           NOT_AUTHORIZED
formal map                   NOT_AUTHORIZED
protected evaluation        0 reads
Eval160                     0 reads
follow-on rollout work       0
```

This handoff records the latest complete state. It does not promote the old
RB1A diagnostic into a scientific result and does not authorize any follow-on
stage.

## Source and environment

```text
repository:    Leo-6-maker/openvla-gripper-dutycycle-attack
branch:        codex/m1-visual-determinism-diagnostic-20260808
commit:        8bd74ff60b28a7a59b94c27f97e99c4e668810a9
tree:          2c942f5874630a8a75907d1ce5a347ad5b3e8d40
worktree:      /mnt/sdc/dty_user/stage-v-m1-diagnostic_8bd74ff6
status:        clean
python:        /mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python
sys.prefix:    /mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800
torch:         2.2.2+cu118
```

The branch was created from RB1 runtime-equivalence commit
`561721902fa49a162f96f8f392b286ef9f4889fc` (tree
`d603841e75be8edfa94b70891655588325694a43`). M1 implementation commit
`d5cf3e407e547b14572c16c750a184bb71a5d04d` was followed by the raw-manifest
compatibility fix `8bd74ff60b28a7a59b94c27f97e99c4e668810a9`.

## Validation completed

On the clean server worktree, using the official environment:

```text
102 tests passed
py_compile: PASS
git diff --check: PASS
```

The tested set covered the existing Stage V canonical core, replay, Dynamic-8,
supervisor, R2/Q2, RB1, runner-binding tests, and the M1 visual-determinism
tests.

## M1-R0 read-only forensic result

R0 read the prior sealed RB1A input-diagnostic root without changing it and
mechanically selected the first failing exposed identity:

```text
libero_10/task_08/state_47
```

R0 status is `M1_R0_PASS_LOCALIZATION_SUFFICIENT_FOR_REPEATABILITY_TEST`.
The old sidecars show `numeric_difference_available=false`; raw pixel
magnitude was deliberately deferred to R2.

Historical localization in the old pair was:

```text
raw observation first mismatch: step 0
policy RGB first mismatch:     step 223
model pixel input mismatch:    step 223
physical state:                 exact in old trace comparison
policy tokens/actions:          exact in old trace comparison
```

This is localization evidence only. It is not the final M1 classification.

## M1 root and sealed artifacts

```text
/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/
M1_VISUAL_DETERMINISM_DIAGNOSTIC_8bd74ff6_20260808T051639Z
```

The root contains R0 reports, source/environment/input bindings, the
re-derived diagnostic candidate and contract, and the preflight HOLD record.
The 15-file `M1_SHA256SUMS` check passed.

Selected hashes:

```text
M1_FINAL_HOLD_REPORT.json
6b86bf1214c19a8098171111cd20a592990e5278c0956f23c009b255786c0cc2

M1_READ_ONLY_FORENSIC_REPORT.json
3f1313b554096c0ac4d0356b50589415f9be845ed6eb1734d6152360e25d80f1

M1_PRELAUNCH_HOLD_REPORT.json
0523f795c0902ca22c79d7f7ff104705ec67d4145a7bf82b4e2fdb2f595a1457

M1_SHA256SUMS
f52a85393bded071cfb1638354da74f35becf9ca4b6d3fefdca9b80b3d5e3e7b

M1_SHA256SUMS.sha256
ff7ee8e25a8b8b1f1904808c3f4cd22f0bc972a64bcabddad9f5dc986678eaef
```

The prior roots remain read-only and were not reused:

```text
RB1_RUNTIME_INPUT_DIAGNOSTIC_56172190_20260808T041216Z
SHA256SUMS seal content: 6f56df57342e78595697424d5784f5e47d6c749a30ebfdfbb64546b55bcd355f

RB1_RUNTIME_EQUIVALENCE_DIAGNOSTIC_db14a92e_20260808T025339Z
SHA256SUMS seal content: d47e7f68e09831c31a55a57189ac161a5009783137b4822da94322982528bb36
```

## Resource HOLD

The latest live preflight was recorded at `2026-08-08 05:54 UTC`:

```text
GPU 0/1/2/4/5/6/7: ~10 MiB used, no compute application
GPU 3:             6715 MiB used, 74508 MiB free
GPU 3 owner:       PID 964381, Isaac GR00T, 6696 MiB
M1 runtime runs:   0
GPU5 touched:      false
foreign process:   not terminated or modified
```

Therefore the M1 root remains `HOLD_WAIT_FOR_SAFE_GPU`. No Q1/C1/Q2/C2
directory exists and no M1 runtime subprocess was started.

## GPU policy and the latest eight-GPU request

The frozen M1 protocol currently declares:

```text
GPU3-only execution
gpu5_authorized = false
fixed run order = Q1 -> C1 -> Q2 -> C2
```

A request to use all eight GPUs is recorded as a pending protocol change, not
silently applied to this root. M1-R1 has four pre-registered runs; launching
eight workers would require changing the run set, scheduling semantics, or
replicate count. Such a change requires a new protocol version, commit/tree,
clean server worktree, and new root. The current M1 root remains governed by
the frozen GPU3-only protocol.

## Next authorized continuation

After a fresh preflight shows GPU3 safe, the next steps are:

1. Verify branch, commit, tree, clean status, official `sys.prefix`, fresh
   manifest, and absence of all four R1 output directories.
2. Append a continuation receipt without overwriting the prior HOLD report or
   its checksum files.
3. Run fresh subprocesses in exactly `Q1 -> C1 -> Q2 -> C2` order on the
   frozen M1 path.
4. Produce the R1 pair matrix and independent audit.
5. Derive the raw-capture plan mechanically, then run R2.
6. Report one allowed M1 classification and stop for human review.

RB1B, new qualification, formal map, Student work, protected evaluation and
any later stages remain blocked until separately reviewed and authorized.
