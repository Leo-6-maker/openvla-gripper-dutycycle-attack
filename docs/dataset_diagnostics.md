# Dataset Diagnostics

## Why Some Tasks Are Excluded From Main Evidence

A task must first have clean success and a visible contact/lift/carry/release phase before it can support a gripper-opening attack claim. We therefore treat clean-policy failures as eligibility failures rather than negative attack evidence.

## Moka Pots

`libero10_moka_pots` is a real multi-stage pick/place task with visible graspable objects after asset repair. However, current clean and oracle diagnostics show execution instability, stuck rollouts, and multi-object transport complexity. The observed oracle disruption currently arrests the task rather than demonstrating a stable transfer-phase gripper-opening slip/drop pattern. Moka is therefore Future Work / Dataset Diagnostics.

## Alphabet Soup

The evaluated alphabet-soup clean traces were dominated by `no_grasp`, so the task did not provide an eligible clean contact/lift/carry trajectory for gripper-opening attack evaluation.

## Open Middle Drawer

The drawer task is not a pick/place gripper-transfer task, so it is low priority for validating gripper-opening slip/drop mechanisms. It may be useful for other robustness questions but is excluded from the current gripper-transfer claim.
