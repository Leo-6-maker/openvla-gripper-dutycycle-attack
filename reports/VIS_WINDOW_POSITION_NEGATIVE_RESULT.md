# VIS Window Position Negative Result

**Date**: 2026-06-01

## Result

72 no-rollout VIS configs across 6 frames at early/detector/late window positions:
- ε = 4/255, 8/255, 12/255
- steps = 10, 20
- objectives = gripper_open_region_ce, force_gripper_open_token_ce

**0/72 decoded gripper token flips.** CE drops 34→0 but gripper stays 0.996 at all windows.

## Conclusion

VIS token-prefix PGD failure is NOT explained by bad window selection. Across early, detector, and late contact-phase frames, decoded gripper token never flips under tested budgets. Current VIS limitation is the decoded-token boundary, not window position.

## Valid Claim

"Under tested budgets and objectives, VIS token-prefix PGD fails to flip decoded gripper tokens across early/detector/late windows; current VIS limitation is decoded-token boundary, not window selection."

## Forbidden

VIS attack successful / VIS rollout ready / forced-window VIS works
