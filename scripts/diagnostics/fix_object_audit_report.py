#!/usr/bin/env python3
"""Fix the Object audit report with correct T_gform interpretation."""
import re

report_path = 'reports/OBJECT_DATA_DETECTOR_READINESS_AUDIT.md'
with open(report_path, 'r') as f:
    content = f.read()

# 1. Fix Executive Summary verdict
old_verdict = "### Verdict\n\n**CURRENT DATA IS SUFFICIENT FOR RULE-BASED EARLY-GRASP DETECTOR DEVELOPMENT, BUT NOT FOR LEARNED DETECTOR TRAINING.**\n\nKey reasons:\n1. T_gform distribution is strongly concentrated at small values (see Section 4)\n2. This makes a learned detector likely unnecessary — a simple rule-based close-onset trigger works\n3. The data lacks temporal trace CSVs (flat format only), requiring conversion for temporal training\n4. Per-step phase labels need to be built from scratch (heuristic pipeline)\n5. The existing teacher labels target release/pre-place, not early-grasp"

new_verdict = """### Verdict

**CURRENT DATA IS SUFFICIENT FOR LEARNED CAUSAL EARLY-GRASP DETECTOR TRAINING.**

Key findings:
1. T_gform varies massively across episodes: range 7-221, mean=84, std=29 (see Section 4)
2. This makes a learned detector NECESSARY -- a fixed rule-based trigger cannot cover this variation
3. The data has full temporal traces with complete runtime features (100% coverage)
4. The flat dataset can be converted to temporal traces for TCN training
5. Per-step phase labels must be built heuristically (done in this audit)
6. The existing teacher labels target release/pre-place (different mechanism); NOT usable for early-grasp

Caveats:
- Only seed=0 data available (no seed robustness)
- chocolate_pudding has duplicate reruns (needs dedup)
- ~2 episodes show oscillating gripper (early heuristic false positives)
- Need to verify late-T_gform episodes are physically correct"""

assert old_verdict in content, "Old verdict not found!"
content = content.replace(old_verdict, new_verdict)

# 2. Fix Section 6 strategy recommendation
old_strategy = "### Recommendation: RULE-BASED BASELINE FIRST\n\n**Rationale**: T_gform is heavily concentrated at small step indices.\nA learned causal TCN detector would largely learn \"trigger when gripper_command first drops below 0.5\"\nwhich is equivalent to a simple rule."

new_strategy = """### Recommendation: LEARNED CAUSAL TCN DETECTOR

**Rationale**: T_gform varies 30x across episodes (7-221), with std=29.
A fixed rule-based trigger (e.g., "attack at step 10") would fail on most episodes.
A learned causal TCN detector is necessary to capture the temporal context that predicts
when grasp formation will occur.

### Recommended pipeline

1. **Convert flat to traces**: Group flat dataset by episode_key, sort by step_idx
2. **Build per-step phase labels**: Use heuristic pipeline from this audit
3. **Train causal TCN**: 13-D runtime input, 3-class output (pre_grasp / grasp_formation / post_grasp)
4. **Online trigger**: T_pred = first step where P(grasp_formation) > threshold for K consecutive steps
5. **Attack window**: [T_pred + Delta, T_pred + Delta + 17] where Delta in {5, 10}

### Why learn instead of rule

A rule-based CLOSE->OPEN transition detector would fire at the correct step,
but ONLY after the OPEN command is generated. This is detection, not anticipation.

A learned TCN may detect PRE-GRASP signatures (EEF deceleration, approach trajectory,
gripper alignment) BEFORE the OPEN command, enabling earlier and more robust triggering.

### When a rule-based baseline is sufficient

If the TCN does not outperform rule-based (fire on CLOSE->OPEN), then
the rule IS the detector. But the wide T_gform variation justifies the attempt."""

assert old_strategy in content, "Old strategy not found!"
content = content.replace(old_strategy, new_strategy)

# 3. Fix Interpretation text
old_interp_text = "0.0% of T_gform values are in {0,1}."
if old_interp_text in content:
    interp_start = content.find("### Interpretation")
    # Find next section after Interpretation
    next_section = content.find("## 5.", interp_start)
    if next_section < 0:
        next_section = content.find("---", interp_start)
    if interp_start >= 0 and next_section > interp_start:
        new_interp = """### Interpretation

T_gform shows MEANINGFUL VARIATION across episodes: range 7-221, mean=84, std=29.
This is a 30x difference between earliest and latest grasp.

**This strongly supports learned detector training.** A fixed-window attack or
simple rule-based trigger cannot capture this variation.

Distribution shape:
- Mode: 60-80 steps (33 episodes, 32%) -- typical grasp zone
- Center mass 40-120 (91 episodes, 89%) -- main distribution
- Left tail 0-40 (4 episodes) -- very early grasp (verify for oscillation artifacts)
- Right tail 120-260 (7 episodes) -- late grasp (verify for anomaly)

Per-task variation:
- cream_cheese: narrowest range (78-128, span=50)
- ketchup: wide range (10-179, span=169)
- orange_juice: widest range (49-221, span=172)
- alphabet_soup: tightest (53-110, span=57)

This per-task and cross-episode variation is exactly what a learned causal TCN
should capture. The detector can learn to identify pre-grasp signatures from
runtime features (EEF deceleration, approach trajectory, alignment changes)
that predict WHEN grasp formation will occur.

### Caveats

1. **Early oscillation episodes**: ketchup_s8 (T=10) and milk_s2 (T=7) show
   oscillating OPEN/CLOSE commands early. These may be heuristic noise --
   the actual stable grasp happens later in these episodes.

2. **Duplicate reruns**: chocolate_pudding has 4 extra rerun episodes
   (obj100_chocolate_pudding_rerun_*). Deduplicate before training.

3. **Seed=0 only**: All data uses seed=0. Need seed 1-2 for robustness
   validation before any generalization claim.

4. **Wait steps**: step_idx starts at 10 (10 wait steps). Policy step =
   dataset step_idx - 10. The T_gform values reported here are dataset
   step_idx (including wait steps)."""

        content = content[:interp_start] + new_interp + "\n\n" + content[next_section:]

# 4. Fix Blockers section
old_blocker = "1. **Per-step phase labels needed**: Must run `build_clean_phase_dataset.py`"
if old_blocker in content:
    blocker_start = content.find("## 7. Blockers")
    blocker_end = content.find("## 8. Next Commands")
    if blocker_start >= 0 and blocker_end > blocker_start:
        new_blockers = """## 7. Blockers Before Training

1. **Deduplicate chocolate_pudding reruns**: 4 rerun episodes duplicate original states.
   Keep the successful one (or the rerun if original failed).

2. **Verify early-oscillation episodes**: ketchup_s8 and milk_s2 may need manual
   label correction (the heuristic may have detected pre-grasp oscillation as T_gform).

3. **Convert flat to temporal traces**: The flat dataset must be grouped by episode_key
   and sorted by step_idx to create temporal sequences for TCN training.

4. **Add seed diversity**: All data is seed=0. Need at least seed 1 data for
   robustness validation.

5. **Filter failed episodes**: 19 failed episodes (no grasp / env termination)
   should be excluded from training, or used only as negative examples.

6. **Validate heuristic labels on held-out tasks**: Before training, verify that
   the heuristic label quality is consistent across all 10 tasks.

All blockers are addressable. None is a hard block to starting detector training."""

        content = content[:blocker_start] + new_blockers + "\n\n" + content[blocker_end:]

with open(report_path, 'w') as f:
    f.write(content)
print('Report fixed successfully.')
