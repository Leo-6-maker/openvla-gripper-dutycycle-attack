# State5 Window Phase Sensitivity

**Generated**: 2026-05-23T17:30Z

## Finding

The targeted attack's effectiveness is window-phase-dependent. Identical attack parameters
(force_gripper_open_token_ce, eps=0.25, ss=0.050, asteps=60, rho=1.0, linf=2.12, 10 active steps)
produce different outcomes depending on which trajectory window is attacked.

## Evidence

| Repeat | Window | Attack SR | Task Outcome | Phase Classification |
|--------|--------|-----------|-------------|---------------------|
| r0 | [62,71] | SR=0.0 | TASK FAILS | attack_sensitive |
| r1 | [62,71] | SR=0.0 | TASK FAILS | attack_sensitive |
| r2 | [62,71] | SR=0.0 | TASK FAILS | attack_sensitive |
| r13 | [65,74] | SR=1.0 | TASK SUCCEEDS | phase_robust |
| r18 | [93,102] | SR=0.0 | TASK FAILS | attack_sensitive |

## Window Position Analysis

| Window | Offset from done | Phase Description |
|--------|-----------------|-------------------|
| [62,71] | done ≈ 82-85 | Pre-place approach — gripper carrying bowl toward plate |
| [65,74] | done ≈ 82-85 | Pre-place approach — slightly later start, more overlap with placement |
| [93,102] | done ≈ 115-120 | Post-place / post-release — gripper already released bowl |

r0/r1/r2 windows [62,71] target the pre-place approach phase. The attack forces
gripper-open tokens during the carry-to-plate phase, causing premature release.

r13 window [65,74] is only 3 steps later but may capture a slightly different
phase of the approach. The task succeeds despite the attack, suggesting the gripper
is either already committed to a close state or the visual perturbation at this
specific window does not flip gripper tokens.

r18 window [93,102] is much later (post-release). The attack forces gripper-open
tokens during the return/retract phase. The task fails, suggesting the timing
of gripper-open tokens matters across the full trajectory — not just during grasp.

## Key Implications

1. **Attack sensitivity is not uniform across trajectory phases.**
   The same linf=2.12 perturbation can cause 0% SR or 100% SR depending on
   the 10-step window position.

2. **Autowindow position is a hidden variable.**
   Two clean_detect runs (r0 with done≈82 and r18 with done≈115) produce
   different absolute window positions, but both are "pre-place" relative
   to their own success_done. The relative phase position matters more than
   the absolute step index.

3. **r13 is not a protocol failure.** It is legitimate evidence that some
   window phases are robust to even strong targeted attacks.

4. **The 4/5 pass rate (80%) is consistent with phase-dependent attack efficacy.**
   Not every clean trajectory window is equally vulnerable.
