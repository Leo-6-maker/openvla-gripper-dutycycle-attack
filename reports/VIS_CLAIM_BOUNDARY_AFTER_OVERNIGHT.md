# VIS Claim Boundary — After Overnight (2026-06-03)

## Allowed Claims

1. **Action bridge**: `prefix_locked_gripper_open_margin` induces true generated gripper OPEN in rollout with armL2=0.000.

2. **Physical bridge**: prefix_margin causes qpos to transition from closed to fully open, with qpos_delta 15-54× clean baseline.

3. **Same-budget task-level evidence**: At eps6, prefix fails (16-18/18 OPEN) while random eps6 succeeds 6/6 on ketchup 10-27 with 0 OPEN.

4. **Budget-compressed evidence**: eps4 also induces the same mechanism (18/18 OPEN, qpos fully open). Source has GPU-warning; healthy-GPU reproduction pending.

5. **Window generalization**: Effect confirmed in both 10-27 and 20-37 clean-CLOSE windows.

6. **Mechanism**: Ketchup failures are early_grasp_disruption — VIS induces premature gripper OPEN during grasp formation, preventing stable grasp. NOT pre-release drop.

7. **Random specificity**: Random controls do not induce gripper OPEN and mostly/all succeed. The VIS effect is not reproduced by random perturbation of comparable magnitude.

8. **Reproducibility**: 7/7 prefix seeds across eps4/6/8, 2/2 windows, 100% failure rate.

## Forbidden Claims

1. ProprioNoStep-guided VIS attack.
2. Universal window independence ("not window-specific").
3. Broad LIBERO generalization (only ketchup fully established; cream/salad have polluted denominators or incomplete controls).
4. Stealth / low-budget attack (eps4 needs healthy-GPU reproduction).
5. Physical-world transfer.
6. Pre-release drop mechanism for ketchup.
7. Trained detector / learned selector success.
8. Full-benchmark success rate.

## Caveats

- eps4 traces come from GPU23 which had GPU3 Xid31. Need healthy-GPU reproduction before final claim.
- Cream and salad have partial results but denominators are either clean-only (cream random missing) or polluted (salad random fails at eps4/6).
- ProprioNoStep raw top windows are natural-release-confounded; shifted pre-release candidates (42-59) not yet tested.
