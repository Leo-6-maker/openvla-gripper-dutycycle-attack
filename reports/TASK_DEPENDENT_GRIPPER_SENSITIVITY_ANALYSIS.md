# Task-Dependent Gripper Sensitivity Analysis

**Generated**: 2026-05-30 02:50 CST

## Executive Summary

Oracle_open intervention physically opens the gripper on ALL 4 tested Object tasks (confirmed by qpos/width response). However, only tomato_sauce and milk s2 fail officially. Salad_dressing survives despite the largest qpos opening (-0.0143). This proves task/object-dependent vulnerability: some tasks have critical contact phases where even mild gripper disruption breaks success, while others tolerate large gripper openings.

## Sensitivity Ranking

| Task | Oracle SR | avg_qpos_delta | avg_burst | avg_clean_steps | avg_oracle_steps | Classification |
|------|-----------|---------------|-----------|-----------------|-----------------|----------------|
| tomato_sauce | **0/3** | -0.0035 | 125 | 197 | 290 (timeout) | **high_oracle_sensitive** |
| milk | 2/3 | -0.0044 | 52 | 167 | 191 | **medium_oracle_sensitive** |
| salad_dressing | 3/3 | -0.0143 | 74 | 129 | 172 | **true_oracle_robust** |
| ketchup | 3/3 | -0.0105 | 35 | 170 | 174 | **true_oracle_robust** |

## Key Findings

### 1. Gripper physically opens on ALL tasks
Oracle consistently sets attacked action to +1.0 (open). Qpos drops on every task (pre_q ≈ +0.030 → post_q ≈ +0.017). The absence of qpos response on tomato is NOT the reason for failure — tomato fails with minimal qpos change.

### 2. Tomato failure mechanism
Tomato fails despite minimal qpos opening. The attack burst (125 steps avg) coincides with the task's critical contact phase. Even a slight gripper disruption at this phase breaks the task. This suggests the detector window overlaps with a fragile grasp/lift moment unique to tomato.

### 3. Salad_dressing is genuinely robust
Qpos opens by -0.0143 (largest among all tasks) over 74 attack steps, yet task succeeds. Salad_dressing s0 reaches 260/280 steps (near-timeout) but still succeeds. The task can recover from significant gripper opening.

### 4. Ketchup is robust
Moderate qpos response (-0.0105) with moderate burst (35 steps). Task completes in 174 steps (vs 170 clean). Minimal disruption.

### 5. Milk is intermediate
Milk s0,s1 survive oracle (burst 23-28). Milk s2 fails at 290 steps with burst=105. Milk s2 has later trigger (step 186 vs 111-123 for s0,s1), suggesting the trigger window timing matters.

## Interpretation

**Detector windows represent candidate gripper-relevant phases**, but:
- Only some tasks have critical contact fragility at these windows
- Salad_dressing and ketchup tolerate forced opening
- Tomato_sauce has a task-specific fragile phase
- Milk is intermediate depending on state

**Do NOT claim**: universal Object vulnerability, all tasks gripper-sensitive, detector window universally causal.

**DO claim**: Task-dependent vulnerability confirmed. Tomato is the strongest evidence case.
