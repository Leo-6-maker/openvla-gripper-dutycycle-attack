#!/usr/bin/env python3
"""Apply RAND/VIS hardening to V6 runner."""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else \
    '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s20d_v6_online_trigger_l3_runner.py'

with open(path) as f:
    src = f.read()

changes = 0

# 1. Fix boundary
old = 'clean_close = clean_gripper_raw < 0.5\n            clean_open = not clean_close'
new = 'clean_close = clean_gripper_raw < 0.5\n            clean_open = clean_gripper_raw > 0.5'
if old in src:
    src = src.replace(old, new)
    changes += 1
    print('1. boundary fix')

# 2. RAND: use prompt() matching clean decode
old = "rand_inputs = processor(\n                    text=instruction, images=v4_image,\n                    return_tensors='pt')"
new = "rand_inputs = processor(\n                    text=prompt(str(instruction).lower()), images=v4_image,\n                    return_tensors='pt')"
if old in src:
    src = src.replace(old, new)
    changes += 1
    print('2. RAND prompt match')

# 3. RAND: epsilon projection
old = 'rand_inputs["pixel_values"] = torch.clamp(\n                    pv + noise, 0.0, 1.0)'
new = 'rand_inputs["pixel_values"] = torch.maximum(torch.minimum(pv.float() + noise, pv.float() + eps_norm), pv.float() - eps_norm).to(dtype=model_dtype)'
if old in src:
    src = src.replace(old, new)
    changes += 1
    print('3. RAND epsilon projection')

# 4. RAND: seeded generator
old = 'noise = torch.empty_like(pv).uniform_(\n                    -eps_norm, eps_norm)'
new = 'g_rand = torch.Generator(device=pv.device)\n                g_rand.manual_seed(args.attack_seed + step)\n                noise = torch.empty(pv.shape, device=pv.device, dtype=torch.float32).uniform_(-eps_norm, eps_norm, generator=g_rand)'
if old in src:
    src = src.replace(old, new)
    changes += 1
    print('4. RAND seeded generator')

# 5. VIS: method hard gate
old = "attacker_config['gripper_margin'] = 5.0"
if old not in src:
    print('5. VIS gate: config pattern changed, skipping')
else:
    # Find the attacker creation and add gate after it
    idx = src.find("print('[%s] V6 TokenPrefixPGD attacker")
    if idx > 0:
        # Find end of that line
        end = src.find('\n', idx)
        gate_code = '\n    if attacker.method not in {"token_prefix_pgd", "openvla_token_prefix_pgd", "visual_token_prefix_pgd"}:\n        raise RuntimeError("V6 HARD FAIL: attacker method=%s" % attacker.method)'
        if gate_code not in src:
            src = src[:end+1] + gate_code + src[end+1:]
            changes += 1
            print('5. VIS method hard gate added')

# 6. Qpos opening delta
old = 'gripper_qpos_after = float(\n        np.sum(gripper_phys_after'
if old in src:
    idx = src.find(old)
    end = src.find('\n', src.find(')', idx + len(old)) + idx + len(old))
    src = src[:end+1] + '    qpos_opening_delta = gripper_qpos_before - gripper_qpos_after  # positive=opening\n' + src[end+1:]
    changes += 1
    print('6. Qpos opening delta')

# 7. done init
if 'done = False\n\nwhile step < max_steps:' not in src:
    src = src.replace('while step < max_steps:', 'done = False\n\nwhile step < max_steps:')
    changes += 1
    print('7. done init')

# 8. n_v5_pgd_applied_steps in summary
old = "'n_steps': step,"
if old in src and 'n_pgd_applied_steps' not in src:
    src = src.replace(old, "'n_steps': step,\n        'n_pgd_applied_steps': sum(1 for r in trace_rows if r.get('pgd_applied', 0)),\n        'success_step_primary': success_step_primary,\n        'done_step_any': done_step_any,")
    changes += 1
    print('8. pgd count + step fields in summary')

with open(path, 'w') as f:
    f.write(src)
print('Total changes: %d' % changes)
