# VIS Verified Contact-Frame Diagnostic — Negative Result

**Date**: 2026-05-31 | **Branch**: `exp/codex-autonomous-vis-crosssuite-20260531`

## 1. Contact-Frame Collection

| Task | State | Result | Target Frames | Notes |
|------|-------|--------|---------------|-------|
| ketchup | 0 | ✅ | 96-100 verified | 148 total frames, success=True |
| tomato_sauce | 0 | ✅ | 132-136 verified | 216 total frames, success=True |
| cream_cheese | 0 | ❌ | 141-145 missing | Early termination at 65 steps |

## 2. Frame Audit

- Gate CF-2: **PASS** (2/3 verified)
- ketchup step 98: contact/carry frame, gripper 0.996
- tomato_sauce step 134: contact/carry frame, gripper 0.996

## 3. VIS Diagnostic Results

**32 configs tested** (2 frames × 2 objectives × 2 eps × 2 steps × 2 postproc):

| Metric | Value |
|--------|-------|
| Token flips | **0/32** |
| Clean gripper | 0.996 (all) |
| Adv gripper | 0.996 (all) |
| Gripper delta | 0.000 (all) |
| CE reduction | 18-34 → 0.0-0.12 |
| Arm L2 drift | 0.63-0.93 |
| linf budget | 3.98-7.97/255 |
| Runtime | 17-34s |

**Gate VIS-1: FAIL**

## 4. Interpretation

The token-prefix PGD path works at the loss/logit level (target CE drops dramatically) but does not change the decoded gripper action under ε≤8/255. The discrete argmax over action tokens remains unchanged even though the continuous logits shift significantly.

This result was obtained on **verified contact/carry frames** (not wait/pre-policy frames), confirming the finding is not an artifact of frame selection.

## 5. Blocked Status

- Forced-window VIS micro: **BLOCKED**
- Detector-triggered VIS: **BLOCKED**
- VIS rollout: **BLOCKED**

## 6. Production Status

- ProprioNoStep: **unchanged**
- sustained_command_open_proxy_30: **unchanged**
- Success predicate: **unchanged**
