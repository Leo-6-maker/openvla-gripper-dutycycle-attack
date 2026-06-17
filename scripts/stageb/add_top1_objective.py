#!/usr/bin/env python3
"""Add prefix_locked_gripper_top1_open_vs_close objective to attack_adapter.py."""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else \
    '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/src/gripper_attack/attack_adapter.py'

with open(path) as f:
    src = f.read()

changes = 0

# 1. Add to _PREFIX_LOCKED_OBJS
old = '_PREFIX_LOCKED_OBJS = {"prefix_locked_gripper_open_region_ce", "prefix_locked_gripper_open_margin", "gripper_open_expected_action"}'
new = '_PREFIX_LOCKED_OBJS = {"prefix_locked_gripper_open_region_ce", "prefix_locked_gripper_open_margin", "gripper_open_expected_action", "prefix_locked_gripper_top1_open_vs_close"}'
if old in src:
    src = src.replace(old, new)
    changes += 1
    print('1. Added to _PREFIX_LOCKED_OBJS')

# 2. Add loss calculation
old = '                elif obj == "gripper_open_expected_action":\n                    probs = torch.softmax(gripper_row, dim=-1)\n                    open_prob_mass = probs[region_token_ids].sum()\n                    gripper_loss = -open_prob_mass\n                else:\n                    gripper_loss = logits.sum() * 0.0'
new = '                elif obj == "gripper_open_expected_action":\n                    probs = torch.softmax(gripper_row, dim=-1)\n                    open_prob_mass = probs[region_token_ids].sum()\n                    gripper_loss = -open_prob_mass\n                elif obj == "prefix_locked_gripper_top1_open_vs_close":\n                    max_open = gripper_row[region_token_ids].max()\n                    max_close = gripper_row[close_token_ids].max()\n                    gripper_loss = F.relu(max_close - max_open + float(margin))\n                else:\n                    gripper_loss = logits.sum() * 0.0'
if old in src:
    src = src.replace(old, new)
    changes += 1
    print('2. Added top-1 loss calculation')
else:
    print('2. WARNING: loss pattern not found')

# 3. Add _loss parameter for close_token_ids
# The _loss function signature needs close_token_ids parameter
old_sig = 'def _loss(self, full_input_ids, labels, pixel_values, *, objective: str = "targeted_directional_ce", region_token_ids=None, margin: float = 5.0, num_action_tokens: int = 7, loss_weights: dict = None, arm_preserve_weight: float = 0.1):'
new_sig = 'def _loss(self, full_input_ids, labels, pixel_values, *, objective: str = "targeted_directional_ce", region_token_ids=None, close_token_ids=None, margin: float = 5.0, num_action_tokens: int = 7, loss_weights: dict = None, arm_preserve_weight: float = 0.1):'
if old_sig in src:
    src = src.replace(old_sig, new_sig)
    changes += 1
    print('3. Added close_token_ids to _loss signature')

# 4. Pass close_token_ids in attack()
old = 'loss_kwargs["region_token_ids"] = region_token_ids'
new = 'loss_kwargs["region_token_ids"] = region_token_ids\n            loss_kwargs["close_token_ids"] = corrected_region_info.get("close_token_ids")'
if old in src:
    src = src.replace(old, new)
    changes += 1
    print('4. Added close_token_ids to loss_kwargs')
else:
    print('4. WARNING: loss_kwargs pattern not found')

# 5. Also check if _needs_region needs update for new objective
old_needs = 'is_prefix_locked_open_margin = objective in {"prefix_locked_gripper_open_margin"}'
new_needs = 'is_prefix_locked_open_margin = objective in {"prefix_locked_gripper_open_margin"}\n        is_prefix_locked_top1 = objective in {"prefix_locked_gripper_top1_open_vs_close"}'
if old_needs in src:
    src = src.replace(old_needs, new_needs)

old_needs2 = '_needs_region = is_gripper_region or is_prefix_locked_open_region or is_prefix_locked_open_margin or is_gripper_expected_action or is_corrected_hybrid'
new_needs2 = '_needs_region = is_gripper_region or is_prefix_locked_open_region or is_prefix_locked_open_margin or is_gripper_expected_action or is_corrected_hybrid or is_prefix_locked_top1'
if old_needs2 in src:
    src = src.replace(old_needs2, new_needs2)
    changes += 1
    print('5. Updated _needs_region for top-1 objective')

# 6. Update debug info label
old_debug = '"gripper_open_region_loss": bool(is_gripper_region or is_prefix_locked_open_region),'
new_debug = '"gripper_open_region_loss": bool(is_gripper_region or is_prefix_locked_open_region),\n                    "gripper_top1_open_vs_close_loss": bool(is_prefix_locked_top1),'
if old_debug in src:
    src = src.replace(old_debug, new_debug)
    changes += 1
    print('6. Added top-1 debug label')

with open(path, 'w') as f:
    f.write(src)

print('\nTotal changes: %d' % changes)
if changes >= 4:
    print('SUCCESS: Top-1 objective added to attack_adapter.py')
else:
    print('WARNING: Some changes failed - check manually')
