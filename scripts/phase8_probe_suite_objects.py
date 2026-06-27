#!/usr/bin/env python3
"""
Phase 8 — Suite Object Site Probe
Generates SUITE_OBJECT_SITE_INVENTORY.csv for Spatial, Goal, LIBERO-10.
Run on server: python scripts/phase8_probe_suite_objects.py
"""
import csv, json, os, sys
from pathlib import Path

OUT_CSV = "reports/phase8_handoff/SUITE_OBJECT_SITE_INVENTORY.csv"
OUT_JSON = "configs/phase8_primary_object_sites.json"

SUITES = {
    "libero_spatial": "models/libero-spatial/spatial_c8f03f4_20260620",
    "libero_goal": "models/libero-goal",
    "libero_10": "models/libero-10/openvla-7b-finetuned-libero-10",
}

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from libero.libero import benchmark, get_libero_path

bm = benchmark.get_benchmark_dict()

FIELD_NAMES = [
    "suite", "task_idx", "task_name", "instruction", "bddl_file",
    "problem_folder", "all_sites", "default_sites", "candidate_objects",
    "primary_site", "object_key", "resolution", "verified", "error"
]
rows = []
registry = {}

for suite_name, model_path in SUITES.items():
    model_dir = Path(model_path)
    stats_path = model_dir / "dataset_statistics.json"
    if not stats_path.exists():
        print(f"SKIP {suite_name}: no dataset_statistics.json at {stats_path}")
        continue

    with open(stats_path) as f:
        stats = json.load(f)

    keys = [k for k in stats if k not in ("num_transitions", "num_trajectories")]
    print(f"\n{'='*60}")
    print(f"Suite: {suite_name}  |  Model: {model_path}")
    print(f"dataset_statistics keys: {keys}  |  action_dim: {len(stats[keys[0]]['action']['mean'])}")

    try:
        suite = bm[suite_name]()
    except KeyError:
        print(f"  ERROR: suite '{suite_name}' not in benchmark registry")
        print(f"  Available: {list(bm.keys())}")
        continue

    n_tasks = len(suite.tasks)
    print(f"Tasks: {n_tasks}")

    for task_idx in range(n_tasks):
        task = suite.tasks[task_idx]
        task_name = task.name
        instruction = task.language
        problem_folder = task.problem_folder
        bddl_file = task.bddl_file

        # Correct BDDL resolution (same as bridge line 163)
        bddl = os.path.join(get_libero_path("bddl_files"), problem_folder, bddl_file)

        def fail_row(reason):
            return {
                "suite": suite_name, "task_idx": task_idx, "task_name": task_name,
                "instruction": instruction, "bddl_file": bddl,
                "problem_folder": problem_folder,
                "all_sites": "", "default_sites": "", "candidate_objects": "",
                "primary_site": "", "object_key": "",
                "resolution": "FAILED", "verified": "false", "error": reason[:300]
            }

        try:
            from gripper_attack.libero_v4_env_factory import build_v4_exact_env
            env, _obs = build_v4_exact_env(bddl, -1, 400, 0)
        except Exception as e:
            reason = str(e)[:300]
            print(f"  [{suite_name} t{task_idx}] ENV FAIL: {reason[:100]}")
            rows.append(fail_row(reason))
            continue

        # Collect site names
        all_sites = list(env.sim.model.site_names)
        default_sites = [s for s in all_sites if s.endswith("_default_site")]
        body_names = list(env.sim.model.body_names)

        # Extract candidate objects from default_sites
        candidate_objects = []
        for ds in default_sites:
            parts = ds.rsplit("_1_default_site", 1)
            if len(parts) == 2:
                candidate_objects.append(parts[0])

        # Primary: first non-furniture default_site
        furniture_prefixes = {"table_top", "main_table", "flat_stove", "wooden_cabinet",
                              "wine_rack", "plate", "stove"}
        non_furniture = [c for c in candidate_objects
                         if not any(c == f or c.startswith(f + "_") for f in furniture_prefixes)]

        primary = non_furniture[0] if non_furniture else (candidate_objects[0] if candidate_objects else "UNKNOWN")
        resolution = "first_non_furniture" if non_furniture else "first_candidate"

        row = {
            "suite": suite_name, "task_idx": task_idx,
            "task_name": task_name, "instruction": instruction,
            "bddl_file": bddl, "problem_folder": problem_folder,
            "all_sites": "|".join(all_sites),
            "default_sites": "|".join(default_sites),
            "candidate_objects": "|".join(candidate_objects),
            "primary_site": f"{primary}_1_default_site",
            "object_key": primary,
            "resolution": resolution, "verified": "false", "error": ""
        }
        rows.append(row)

        print(f"  [{suite_name} t{task_idx}] {task_name}")
        print(f"    folder={problem_folder}  bddl={bddl_file}")
        print(f"    objects: {candidate_objects}")
        print(f"    primary: {primary}  ({resolution})")

        env.close()
        del env

# Write CSV
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELD_NAMES)
    w.writeheader()
    w.writerows(rows)

print(f"\nWrote {len(rows)} rows → {OUT_CSV}")

# Write JSON registry
registry = {}
for r in rows:
    suite = r["suite"]
    if suite not in registry:
        registry[suite] = {}
    registry[suite][str(r["task_idx"])] = {
        "task_name": r["task_name"],
        "task_instruction": r["instruction"],
        "problem_folder": r["problem_folder"],
        "bddl_file": r["bddl_file"],
        "primary_object_site": r["primary_site"],
        "object_key": r["object_key"],
        "resolution_method": r["resolution"],
        "verified": r["verified"],
        "candidate_objects": r["candidate_objects"].split("|") if r["candidate_objects"] else [],
        "error": r["error"],
    }

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(registry, f, indent=2, ensure_ascii=False)

print(f"Wrote registry → {OUT_JSON}")
