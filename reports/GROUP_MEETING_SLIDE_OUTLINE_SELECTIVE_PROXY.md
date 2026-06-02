# Group Meeting Slide Outline — Selective Sustained Proxy

## Slide 1: Problem
- OpenVLA gripper duty-cycle vulnerability
- Can we identify gripper-sensitive windows and attack them?
- Detector + command-layer attack pipeline

## Slide 2: Clean-Only Detector
- ProprioNoStep TCN (38K params, 13 proprio features)
- Trained on Object-100 teacher labels
- Coverage 0.99, miss 0, false-early 7
- Candidate window selector, NOT oracle-optimal

## Slide 3: Full10 Oracle Sensitivity
- 10 Object tasks × 5 states × clean/oracle = 100 rollouts
- Spectrum: High (cream, tomato 2/10) to Robust (ketchup, salad 10/10)
- Detector triggers on ALL tasks. Sensitivity is task/object-dependent.

## Slide 4: Why Original Proxy Failed
- burst 9-19 steps, no qpos response, SR 6/6
- attack_remaining=5 with no feedback loop
- oracle sustained via re-trigger; proxy didn't change physical state

## Slide 5: Sustained Proxy Design
- Decoupled attack_burst_steps from detector_trigger_duration
- sustained_command_open_proxy: hold gripper open for N steps
- Command-layer, NOT visual (VIS)
- Code: +11/-2 lines, 19/19 tests pass

## Slide 6: Full10 sus30 Result
- High-sensitive: 0/10 (cream_cheese + tomato_sauce collapse)
- Robust: 10/10 (ketchup + salad_dressing preserved)
- Selectivity: +100%
- sus30 stronger than oracle (longer sustained burst)

## Slide 7: Detector Ablation
- ProprioNoStep vs VisualNoStep vs VisualProprioNoStep
- Visual adds +0.004 AUROC but loses 10% coverage
- ProprioNoStep remains best practical detector

## Slide 8: Caveats
- NOT visual attack / universal attack / VIS PGD
- Command-layer sustained proxy
- Task/object-dependent sensitivity
- Detector selects candidate windows, not oracle-optimal

## Slide 9: Next Steps
- Manual audit of high-sensitive failures
- Potential detector refinement
- True VIS PGD after command-layer story solid
