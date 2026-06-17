#!/usr/bin/env python3
"""Fix region computation in d0_repeat_decode.py."""
path = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/d0_repeat_decode.py'
with open(path) as f:
    lines = f.readlines()

start_idx = None; end_idx = None
for i, line in enumerate(lines):
    if 'Compute open/close token IDs' in line:
        start_idx = i
    if start_idx is not None and 'close_token_ids = region_info' in line:
        end_idx = i
        break

new_lines = [
    "# Compute open/close token IDs directly\n",
    "stats_d = model.get_action_stats(unnorm_key)\n",
    "low = np.asarray(stats_d['q01'], dtype=np.float32)\n",
    "high = np.asarray(stats_d['q99'], dtype=np.float32)\n",
    "centers = np.asarray(model.bin_centers, dtype=np.float32)\n",
    "vocab_size = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)\n",
    "gripper_dim = action_dim - 1\n",
    "open_tokens = []; close_tokens = []\n",
    "for disc in range(len(centers)):\n",
    "    norm = centers[disc]\n",
    "    decoded_action = float(0.5 * (norm + 1.0) * (high[gripper_dim] - low[gripper_dim]) + low[gripper_dim])\n",
    "    env_val = 2.0 * decoded_action - 1.0\n",
    "    tid = int(vocab_size - disc - 1)\n",
    "    if env_val < -0.5: open_tokens.append(tid)\n",
    "    elif env_val > 0.5: close_tokens.append(tid)\n",
    "open_token_ids = torch.tensor(sorted(set(open_tokens)), dtype=torch.long, device=device)\n",
    "close_token_ids = torch.tensor(sorted(set(close_tokens)), dtype=torch.long, device=device)\n",
    "\n",
]
lines[start_idx:end_idx+1] = new_lines
with open(path, 'w') as f:
    f.writelines(lines)
print('Fixed lines %d-%d' % (start_idx+1, end_idx+1))
