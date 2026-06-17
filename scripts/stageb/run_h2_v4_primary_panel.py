#!/usr/bin/env python3
"""H2 V4 primary panel: 3 anchor frames x 2 seeds on GPU(2,6)."""
import csv, json, os, shutil, subprocess, sys, time, yaml
from pathlib import Path

REPO = Path("/data/liuyu/worktrees/l3_deepseek_autonomous_20260617")
PKG = Path("/data/liuyu/outputs/l3_deepseek_autonomous_20260617_r1/canonical_packages_v2")
OUT = Path("/data/liuyu/outputs/l3_deepseek_autonomous_20260617_r1/h2_v4_primary")
PY = "/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python"
RUNNER = REPO / "scripts/stageb/run_m3_step78_true_pgd_fixed_frame.py"
V4_TPL = REPO / "configs/m3_butter_s11_step60_v4.yaml"

JOBS = [
    ("butter_s11", "butter", 11, 60, 82),
    ("tomato_sauce_s23", "tomato_sauce", 23, 141, 81),
    ("tomato_sauce_s23", "tomato_sauce", 23, 141, 82),
    ("salad_dressing_s11", "salad_dressing", 11, 59, 81),
    ("salad_dressing_s11", "salad_dressing", 11, 59, 82),
]

OUT.mkdir(parents=True, exist_ok=True)
results = []

for pid, task, sid, step, seed in JOBS:
    tag = "{}_step{:04d}_seed{}".format(pid, step, seed)
    job_dir = OUT / tag
    input_dir = job_dir / "input"
    output_dir = job_dir / "canary"
    config_path = job_dir / "config.yaml"

    if (output_dir / "m3_v4_selected_results.csv").exists():
        print("SKIP {} (already done)".format(tag))
        continue

    job_dir.mkdir(parents=True, exist_ok=True)

    # Prepare input
    pkg_dir = PKG / "{}_step{:04d}".format(pid, step)
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pkg_dir / "raw_frame.npy", input_dir / "raw_agentview_step78.npy")
    with open(pkg_dir / "clean_generation.json") as f:
        d = json.load(f)
    gen = {
        "instruction": d["instruction"],
        "clean_action": d["clean_action"],
        "clean_exact_7_tokens": d["exact_clean_7_tokens"],
        "official": {
            "tokens": d["exact_clean_7_tokens"],
            "arm_prefix": d["clean_arm_prefix"],
            "gripper_token": d["clean_gripper_token"],
            "score_invariant": d["official_score_invariant"],
            "target_stats": {},
        },
    }
    with open(input_dir / "clean_generation_step78.json", "w") as f:
        json.dump(gen, f)

    # Create config
    with open(V4_TPL) as f:
        cfg = yaml.safe_load(f)
    cfg["input"]["task"] = task
    cfg["input"]["state_id"] = sid
    cfg["input"]["absolute_step"] = step
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print("[{}/5] {}".format(len(results) + 1, tag))
    start = time.time()
    r = subprocess.run(
        [PY, str(RUNNER), "--config", str(config_path), "--mode", "canary_v4",
         "--input_dir", str(input_dir), "--output_dir", str(output_dir),
         "--attack_seed", str(seed)],
        cwd=str(REPO), capture_output=True, text=True, timeout=600,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "2,6"},
    )
    elapsed = time.time() - start

    if r.returncode != 0:
        print("  FAIL [{:.0f}s]: {}".format(elapsed, r.stderr[-200:]))
        results.append({"tag": tag, "status": "FAIL", "stderr": r.stderr[-200:]})
    else:
        sel_csv = output_dir / "m3_v4_selected_results.csv"
        if sel_csv.exists():
            rows = list(csv.DictReader(open(sel_csv)))
            true_row = next((x for x in rows if "TRUE" in x.get("condition", "")), None)
            if true_row:
                arm = true_row.get("arm_prefix_match_count", "?")
                margin = true_row.get("official_target31744_margin", "?")
                res = true_row.get("stage_result", "?")
                print("  {} arm={}/6 margin={} [{:.0f}s]".format(res, arm, margin, elapsed))
                results.append({"tag": tag, "status": res, "arm": arm, "margin": margin})
        else:
            print("  NO_RESULTS [{:.0f}s]".format(elapsed))
            results.append({"tag": tag, "status": "NO_RESULTS"})

    # Clear GPU cache between jobs
    subprocess.run(
        [PY, "-c", "import torch; torch.cuda.empty_cache()"],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "2,6"},
    )

# Summary
n_pass = sum(1 for x in results if "PASS" in str(x.get("status", "")))
print("\n=== H2 V4 PRIMARY: {}/{} ===".format(n_pass, len(results)))
for x in results:
    print("  {}: {} arm={} margin={}".format(
        x["tag"], x.get("status", "?"), x.get("arm", "?"), x.get("margin", "?")))

with open(OUT / "summary.json", "w") as f:
    json.dump(results, f, indent=2)

sys.exit(0 if n_pass == len(results) else 1)
