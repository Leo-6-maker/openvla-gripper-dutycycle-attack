# VIS Decode Path Bugfix Audit

**Date**: 2026-06-01 | **Phase**: A — Decode Bug Fix

## Bug Description

Two issues in the VIS no-rollout diagnostic decode path:

1. **Missing OpenVLA prompt wrapping**: Direct `processor(text, image)` was used instead of `processor(prompt(instruction), image)`. OpenVLA expects the format: `In: What action should the robot take to {instruction}?\nOut:`

2. **Missing action prefix token 29871**: Token 29871 must be the last input token before generation. Without it, the model generates 6 action tokens + EOS instead of 7 action tokens.

### Mechanism

Without the prompt wrapper and action prefix:
- Model generates tokens: `[dim0, dim1, dim2, dim3, dim4, dim5, EOS(2)]`
- `action_dim=7`, so `sequences[0, -7:]` includes EOS as the 7th token
- EOS token (id=2) → bin = 32000 - 2 - 1 = 31997 → clipped to 254
- BC[254] = 0.996078 → ALL frames showed grip=0.996078 regardless of actual state

### Impact

All no-rollout VIS diagnostics that used direct `processor(text, image)` without prompt wrapping and action prefix produced invalid decode results. This includes:
- Old tomato/ketchup threshold sweep (both directions)
- Any report claiming "decoded gripper unchanged" under old path
- The 0/72 token-flip summaries from old path

## Fix

```python
def prompt(instruction):
    return f"In: What action should the robot take to {instruction}?\nOut:"

# Before (BROKEN):
inputs = processor(instruction, image, return_tensors='pt')

# After (FIXED):
text = prompt(instruction.lower())
inputs = processor(text, image, return_tensors='pt')
inputs.pop("attention_mask", None)
# Append action prefix token 29871 if not present
if not torch.all(inputs["input_ids"][:, -1] == 29871):
    suffix = torch.tensor([[29871]], dtype=torch.long, device=device)
    inputs["input_ids"] = torch.cat((inputs["input_ids"], suffix), dim=1)
```

## Before/After Examples

| Frame | Old Decode (broken) | New Decode (fixed) |
|-------|---------------------|---------------------|
| ketchup_s0_step0098 | grip=0.996078 (EOS-as-gripper) | grip=0.000000 (neutral, correct) |
| tomato_s0_step0134 | grip=0.996078 (EOS-as-gripper) | grip=0.000000 (neutral, correct) |
| ketchup_s0_step0050 | grip=0.996078 (EOS-as-gripper) | grip=0.996078 (actually open, coincidentally same) |

## Invalidated Results

The following old results are invalidated by this decode bug:
- Old tomato/ketchup threshold sweep CSV (pre-fix decode path)
- Any summary claiming tomato reproduced but ketchup did not (opposite with correct decode)
- Any 0/72 token-flip counts from old decode path
- The claim that ketchup did NOT reproduce on rerun (ketchup actually reproduces 5/5 with correct decode)

## Currently Trusted Results (correct decode only)

- **Gate S1 PASS**: Direction semantics confirmed (0.0→0.996 = NEUTRAL→FULL OPEN)
- **ketchup_s0_step0098**: 5/5 seeds grip 0.0→0.996 (strong reproduce, 100%)
- **tomato_s0_step0134**: 0/5 seeds no change (negative)
- **ketchup_s0_step0050**: clean grip already 0.996, PGD flips to 0.0 (wrong direction, not positive)
- **Arm L2**: ketchup contact 0.839 (elevated, needs Phase E audit)

## Smoke Checks

- [x] prompt wrapper applied: `prompt(instruction.lower())`
- [x] action prefix token 29871 appended to input_ids
- [x] decoded action has 7 dimensions
- [x] EOS token (id=2) not interpreted as gripper action token
- [x] No fallback to zeros
- [x] Clean decode stable (3x identical)
- [x] py_compile passes
