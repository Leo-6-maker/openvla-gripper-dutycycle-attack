#!/usr/bin/env python3
"""Hotfix Object attack bridge to support --unnorm_key and --suite_name for cross-suite."""
import sys

path = sys.argv[1]
with open(path) as f:
    content = f.read()

# Add args
old = 'ap.add_argument("--task_idx", type=int, default=6, help="LIBERO task index (default 6=butter)")'
new = '''ap.add_argument("--task_idx", type=int, default=6, help="LIBERO task index")
    ap.add_argument("--unnorm_key", type=str, default="libero_object", help="Dataset statistics key")
    ap.add_argument("--suite_name", type=str, default="libero_object", help="LIBERO benchmark suite name")'''
if old in content:
    content = content.replace(old, new)
    print("Added --unnorm_key and --suite_name args")
else:
    print("ARG SECTION NOT FOUND")

# Replace hardcoded strings
reps = [
    ('model.get_action_dim("libero_object")', 'model.get_action_dim(args.unnorm_key)'),
    ('suite = bm["libero_object"]()', 'suite = bm[args.suite_name]()'),
    ('"libero_object", 8,', 'args.unnorm_key, 8,'),
    ('unnorm_key="libero_object"', 'unnorm_key=args.unnorm_key'),
    ('model.get_action_stats("libero_object")', 'model.get_action_stats(args.unnorm_key)'),
]
for old_s, new_s in reps:
    if old_s in content:
        content = content.replace(old_s, new_s)
        print(f"Replaced: {old_s[:60]}")
    else:
        print(f"NOT FOUND: {old_s[:60]}")

with open(path, 'w') as f:
    f.write(content)
print("DONE")
