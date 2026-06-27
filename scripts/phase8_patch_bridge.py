#!/usr/bin/env python3
"""Phase 8 — Patch bridge for suite-aware object-site lookup. Targeted edits only."""
import shutil

BRIDGE = "scripts/stageb/run_v2_vis_sc5_mlp_bridge_telemetry_v2.py"
BACKUP = BRIDGE + ".phase8_backup_20260627"

# Only backup if not already backed up
import os
if not os.path.exists(BACKUP):
    shutil.copy2(BRIDGE, BACKUP)
    print(f"Backup: {BRIDGE} -> {BACKUP}")
else:
    print(f"Backup exists: {BACKUP}")

with open(BRIDGE, "r") as f:
    content = f.read()

# ── Edit 1: Add --max_env_steps and --object_site_registry after --suite_name arg ──
old1 = '''ap.add_argument("--suite_name", type=str, default="libero_object", help="LIBERO benchmark suite name")
ap.add_argument("--save_video"'''
new1 = '''ap.add_argument("--suite_name", type=str, default="libero_object", help="LIBERO benchmark suite name")
ap.add_argument("--max_env_steps", type=int, default=400, help="Max rollout steps (env + loop)")
ap.add_argument("--object_site_registry", type=str, default="configs/phase8_primary_object_sites.json",
                help="JSON registry mapping (suite,task_idx)->primary_object_site")
ap.add_argument("--save_video"'''

if old1 in content:
    content = content.replace(old1, new1, 1)
    print("  Edit 1 OK: added --max_env_steps and --object_site_registry args")
else:
    print("  Edit 1 FAIL: pattern not found!")

# ── Edit 2: Replace hardcoded 400 in build_v4_exact_env ──
old2 = "env, obs = build_v4_exact_env(bddl, args.render_gpu, 400, 10)"
new2 = "env, obs = build_v4_exact_env(bddl, args.render_gpu, args.max_env_steps, 10)"
if old2 in content:
    content = content.replace(old2, new2, 1)
    print("  Edit 2 OK: build_v4_exact_env max_steps parameterized")
else:
    print("  Edit 2 FAIL: build_v4_exact_env pattern not found!")

# ── Edit 3: Replace hardcoded object-site parsing with registry lookup ──
old3 = '''_task_name = task_obj.name
_obj_key = _task_name.replace("pick_up_the_","").replace("_and_place_it_in_the_basket","")
obj_sid = env.sim.model.site_name2id(f"{_obj_key}_1_default_site")'''

new3 = '''# Suite-aware object site lookup (registry-based, replaces hardcoded string parsing)
import json as _json
with open(args.object_site_registry, "r") as _f:
    _obj_registry = _json.load(_f)
_suite_key = args.suite_name
_task_key = str(TASK_IDX)
if _suite_key not in _obj_registry or _task_key not in _obj_registry[_suite_key]:
    _candidates = [s for s in env.sim.model.site_names if s.endswith("_default_site")]
    raise SystemExit(
        f"FATAL: No object site registry entry for suite={_suite_key} task_idx={_task_key}. "
        f"Available default_sites: {_candidates}. "
        f"Add entry to {args.object_site_registry} before running.")
OBJ_SITE_NAME = _obj_registry[_suite_key][_task_key]["primary_object_site"]
print(f"Object site: suite={_suite_key} task={_task_key} -> {OBJ_SITE_NAME}", flush=True)
obj_sid = env.sim.model.site_name2id(OBJ_SITE_NAME)'''

if old3 in content:
    content = content.replace(old3, new3, 1)
    print("  Edit 3 OK: replaced hardcoded object-site parsing with registry lookup")
else:
    print("  Edit 3 FAIL: object-site pattern not found!")
    # Debug: show what's around that area
    import re
    for m in re.finditer(r'_task_name.*\n.*_obj_key.*\n.*obj_sid.*', content):
        print(f"  Found near: {m.group()[:200]}")

# ── Edit 4: Replace hardcoded 400 in rollout loop ──
old4 = "for step in range(400):"
new4 = "for step in range(args.max_env_steps):"
if old4 in content:
    content = content.replace(old4, new4, 1)
    print("  Edit 4 OK: rollout loop parameterized")
else:
    print("  Edit 4 FAIL: rollout loop pattern not found!")

# ── Write ──
with open(BRIDGE, "w") as f:
    f.write(content)

# ── Verify ──
checks_ok = []
checks_fail = []
for pattern, desc in [
    ("args.max_env_steps", "max_env_steps used"),
    ("OBJ_SITE_NAME", "registry lookup used"),
    ("_obj_registry", "registry loaded"),
    ("phase8_primary_object_sites.json", "registry path"),
]:
    if pattern in content:
        checks_ok.append(desc)
    else:
        checks_fail.append(desc)

if '_task_name.replace(' not in content and '_obj_key = _task_name' not in content:
    checks_ok.append("NO hardcoded string parsing")
else:
    checks_fail.append("hardcoded string parsing STILL PRESENT")

print(f"\nVerification: {len(checks_ok)} OK, {len(checks_fail)} FAIL")
for c in checks_ok: print(f"  ✓ {c}")
for c in checks_fail: print(f"  ✗ {c}")

if checks_fail:
    exit(1)
else:
    print("\nBridge patched successfully.")
    print(f"Backup: {BACKUP}")
