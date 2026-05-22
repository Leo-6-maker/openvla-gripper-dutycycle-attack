# Non-BB Bowl-on-Plate Evidence Summary

This is diagnostic-level non-Black-Bowl mechanism evidence, not Table2.

- input_root: /data/liuyu/outputs/bowl_plate_state_specific_autowindow_gated_diag_20260522
- final_state: targeted_VIS_reproduced_with_state_specific_autowindows
- state_specific_autowindow_used: true
- fixed_state0_window_reuse: false
- pass_states: 4, 5
- non_pass_or_skipped_states: 3, 7, 8, 9

State4 and state5 pass the small mechanism gate: clean/random/VIS_current controls succeed, command-open reference reproduces vulnerability, and targeted VIS produces timeout/failure with gripper qpos close to the command-open reference. State9 is not interpreted as pass because targeted qpos separation is insufficient.

Human video/trace review is still pending. This result does not support LIBERO-wide generalization, 142-row full attack, or formal Table2.
