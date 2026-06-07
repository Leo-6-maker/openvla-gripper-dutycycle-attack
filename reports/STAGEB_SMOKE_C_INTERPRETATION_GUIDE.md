# Stage-B Smoke-C — Result Interpretation Guide

**EXPLORATORY ONLY — SELECTIVE RERUN SUBSET — NOT FINAL DETECTOR**

## Case A: VIS-specific qpos response clearly present

**Signs**:
- cmd_susceptible positives majority have qpos_delta_shifted >= 0.01
- hard negatives have near-zero qpos response
- random_confounded windows show random-induced qpos (confirms perturbation sensitivity)

**Conclusion**: command → physical bridge chain holds.
**Next**: expand to 60-100 balanced rows with v1.1 standard runner.

## Case B: Command OPEN exists but qpos response rare

**Signs**:
- VIS changes decoded gripper action (open_count >= 6)
- But qpos_delta_shifted < 0.01 for most positives
- Gripper doesn't physically open despite command change

**Conclusion**: command susceptibility ≠ physical bridging.
**Next**: need stronger opportunity/contact filter; don't blindly expand.

## Case C: VIS and random both show qpos response

**Signs**:
- Random_confounded rate > 30%
- Random-induced qpos_delta similar to VIS
- Windows are generally perturbation-sensitive, not VIS-specific

**Conclusion**: random contrast insufficient; windows too easy to perturb.
**Next**: resample harder negatives; lower random_confounded threshold; audit perturbation budget.
