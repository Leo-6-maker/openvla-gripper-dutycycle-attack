# Layer 3 D5-Triggered VIS-Temporal v1 Freeze

**Freeze date:** 2026-06-18  
**Branch:** `exp/l3-d5-vis-temporal-20260617`  
**Primary runner commit:** `a170bbc6fe0c7fd2ab1d1b60d29728692d56f0e1`  
**Initial auditor commit:** `a7ade62bf9cf23dca1e354ca3d7ca7e0780edde6`

## Frozen v1 system

```text
Frozen D5-v1 clean-only online detector
    -> first emit at step 60 on Butter_s11
    -> fixed K=10 VIS-temporal window
    -> current attacked-state RGB at every step
    -> persistent PGD attacker with prev_delta warm start
    -> target token 31744 / CLIP_MEDIATED_OPEN
    -> official raw-to-env action conversion
    -> physical gripper response and task rollout
```

### Frozen attack contract

- Parent/state: `Butter_s11`
- Trigger: online D5 first emit
- Attack window: `emit ... emit+9`
- K: 10 environment steps
- Objective: `autoregressive_prefix_gripper_target_token_logratio_arm_v3`
- Target token: 31744
- Target execution class: `CLIP_MEDIATED_OPEN`
- Epsilon: 6/255 in processor pixel space
- PGD steps per frame: 20
- Temporal initialization: `prev_delta`
- Arm preservation weight: 0.5
- Strict route: true
- Fallback: forbidden
- Seeds: 81 and 82
- Matched conditions: `CLEAN_D5`, `TRUE_SINGLE`, `TRUE_TEMPORAL_K10`, `RAND_TEMPORAL_K10`, `SHUFFLED_TEMPORAL_K10`

## Frozen v1 empirical boundary

The v1 experiments establish the following conservative result:

```text
Layer-2 online trigger: established on Butter_s11
VIS-temporal semantic OPEN duty control: established
Actual env OPEN-command duty control: established
Selective physical gripper opening: observed
Recovery delay / additional completion steps: observed
Contact-quality failure: not established
Official task failure: not established
```

All reported v1 rollouts ultimately complete the task. The temporal intervention increases recovery effort and episode length in the TRUE condition, but the clean policy recovers. Therefore the allowed scientific wording is:

> The contact-onset detector launches a bounded temporal visual intervention that selectively controls the gripper OPEN duty cycle and propagates to physical opening, while the policy retains enough closed-loop recovery capacity to complete the task.

The forbidden wording is:

- task-failure attack established;
- irreversible grasp failure established;
- general failure-critical window detection established;
- broad cross-parent task-level generalization established.

## Why v1 is frozen

v1 answers the mechanism question: a clean-only online detector can trigger a temporal visual control channel from image perturbation to semantic OPEN, executed OPEN command, and physical gripper response.

v1 also exposes the remaining gap: the detector currently targets **contact onset / attack opportunity**, not the later **failure-critical point of no return**. The policy can reclose, regrasp, or otherwise recover after the attack window.

No further tuning of K, epsilon, objective, parent, state, or thresholds is allowed inside v1. Any timing redesign belongs to v2.

## v2 scope: failure-critical phase detector

The next version changes the Layer-2 target definition from:

```text
contact onset / gripper-relevant opportunity
```

to:

```text
stable grasp established
AND object lifted / following the end effector
AND release is not yet safe
AND intervention is likely to cause low-recovery-margin contact failure
```

### v2 phase vocabulary

- approach
- grasp close
- stable grasp established
- first lift
- stable carry
- pre-release hazard
- release safe
- recovery / regrasp
- abstain / unsupported

### v2 supervision boundary

The privileged teacher may use clean simulation state to label physical phases:

- object pose and height
- object-to-EEF distance
- object following / detach status
- target distance
- gripper qpos / width
- EEF pose and velocity

The deployment student may use only causal deployment-safe history:

- gripper command, qpos, width
- EEF pose and velocity
- recent action history
- optional online visual / OpenVLA clean-forward features
- task language or mechanism metadata

The student must not use future frames, attack outcomes, task success, manual failure labels, privileged object state, or absolute timestep as a shortcut.

### v2 evaluation target

1. Teacher-label audit on clean trajectories.
2. Causal replay: trigger inside post-lift stable-carry / pre-release hazard, with low false-early-trigger rate.
3. Abstain outside supported manipulation phases.
4. Frozen detector-triggered VIS-temporal evaluation on held-out seeds.
5. Separate gates for semantic duty, command duty, physical response, contact failure, and official task failure.

## Versioning rule

- v1 artifacts and claims remain immutable.
- v2 must use a new branch, output root, configuration namespace, reports, and hash manifest.
- v2 results must never overwrite or silently reinterpret v1 artifacts.
