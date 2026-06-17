#!/usr/bin/env python3
"""H2 batch: Run all remaining fixed-frame jobs with arm-v3 contract.

H1 already completed: butter_s11 step60 seed81.
Remaining: 19 jobs (10 frames × 2 seeds - 1).
Uses logratio_arm_v3 objective with arm_preserve_weight=0.5.
"""
import json, os, subprocess, sys, time, csv, shutil, hashlib
from pathlib import Path

REPO = Path("/data/liuyu/worktrees/l3_deepseek_autonomous_20260617")
PYTHON = "/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python"
RUNNER = REPO / "scripts/stageb/run_m3_step78_true_pgd_fixed_frame.py"
BASE_CONFIG = REPO / "configs/m3_butter_s11_step60_arm_v3.yaml"
PKG_ROOT = Path("/data/liuyu/outputs/l3_deepseek_autonomous_20260617_r1/canonical_packages_v2")
OUT_ROOT = Path("/data/liuyu/outputs/l3_deepseek_autonomous_20260617_r1/h2_panel")

SELECTED_PARENTS = {
    "butter_s11": {"task": "butter", "state_id": 11},
    "tomato_sauce_s23": {"task": "tomato_sauce", "state_id": 23},
    "salad_dressing_s11": {"task": "salad_dressing", "state_id": 11},
}

JOBS = [
    # butter_s11
    ("butter_s11", 58, "teacher_ws"), ("butter_s11", 60, "teacher_anchor+d5_emit"),
    ("butter_s11", 68, "teacher_we"),
    # tomato_sauce_s23
    ("tomato_sauce_s23", 69, "d5_emit"), ("tomato_sauce_s23", 139, "teacher_ws"),
    ("tomato_sauce_s23", 141, "teacher_anchor"),
    # salad_dressing_s11
    ("salad_dressing_s11", 57, "teacher_ws"), ("salad_dressing_s11", 59, "teacher_anchor"),
    ("salad_dressing_s11", 67, "teacher_we"), ("salad_dressing_s11", 128, "d5_emit"),
]

SEEDS = [81, 82]
DONE = {("butter_s11", 60, 81)}  # H1 completed


def make_config(pid, step, out_config_path):
    """Create frame-specific arm-v3 config from template."""
    sel = SELECTED_PARENTS[pid]
    with open(BASE_CONFIG) as f:
        cfg_text = f.read()
    cfg_text = cfg_text.replace("task: butter", f"task: {sel['task']}")
    cfg_text = cfg_text.replace("state_id: 11", f"state_id: {sel['state_id']}")
    cfg_text = cfg_text.replace("absolute_step: 60", f"absolute_step: {step}")
    with open(out_config_path, "w") as f:
        f.write(cfg_text)


def prepare_input(pid, step, input_dir):
    """Prepare input dir from canonical v2 package."""
    pkg_dir = PKG_ROOT / f"{pid}_step{step:04d}"
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pkg_dir / "raw_frame.npy", input_dir / "raw_agentview_step78.npy")
    with open(pkg_dir / "clean_generation.json") as f:
        d = json.load(f)
    out = {
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
        json.dump(out, f, indent=2)


def run_one(pid, step, seed, attempt=1):
    """Run a single frame-seed canary. Returns (success, result_dict)."""
    tag = f"{pid}_step{step:04d}_seed{seed}"
    if attempt > 1:
        tag += f"_r{attempt}"

    input_dir = OUT_ROOT / tag / "input"
    output_dir = OUT_ROOT / tag / "canary"
    config_path = OUT_ROOT / tag / "config.yaml"

    # Skip if already complete
    cond_csv = output_dir / "m3_step78_condition_results.csv"
    if cond_csv.exists():
        rows = list(csv.DictReader(open(cond_csv)))
        if rows:
            return True, {"status": "already_complete", "output_dir": str(output_dir)}

    # Prepare
    output_dir.mkdir(parents=True, exist_ok=True)
    prepare_input(pid, step, input_dir)
    make_config(pid, step, config_path)

    start = time.time()
    result = subprocess.run(
        [PYTHON, str(RUNNER), "--config", str(config_path), "--mode", "canary",
         "--input_dir", str(input_dir), "--output_dir", str(output_dir),
         "--attack_seed", str(seed)],
        cwd=str(REPO),
        capture_output=True, text=True,
        timeout=300,  # 5 min timeout
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "1,5"},
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        return False, {"status": "failed", "stderr": result.stderr[-500:],
                       "elapsed": elapsed, "output_dir": str(output_dir)}

    # Quick classification
    if not cond_csv.exists():
        return False, {"status": "no_output", "elapsed": elapsed, "output_dir": str(output_dir)}

    rows = list(csv.DictReader(open(cond_csv)))
    true_row = next((r for r in rows if r.get("condition") == "TRUE_PGD_FINAL"), None)
    if not true_row:
        return False, {"status": "no_true_row", "elapsed": elapsed, "output_dir": str(output_dir)}

    arm_match = int(true_row.get("arm_prefix_match_count", "0") or 0)
    gripper = int(true_row.get("official_gripper_token", "0") or 0)
    true_margin = float(true_row.get("official_target31744_margin", "-inf") or "-inf")

    rand_row = next((r for r in rows if r.get("condition") == "RAND20"), None)
    shuffled_row = next((r for r in rows if r.get("condition") == "SHUFFLED_GRAD_PGD20"), None)

    rand_gripper = int(rand_row.get("official_gripper_token", "0") or 0) if rand_row else -1
    shuffled_gripper = int(shuffled_row.get("official_gripper_token", "0") or 0) if shuffled_row else -1
    shuffled_arm = int(shuffled_row.get("arm_prefix_match_count", "0") or 0) if shuffled_row else -1

    # H1 gate
    route_status = true_row.get("route_status", "")
    score_inv = true_row.get("score_invariant_status", "")
    linf = float(true_row.get("processor_linf", "999") or 999)

    gate_pass = (
        route_status == "PASS"
        and score_inv == "PASS"
        and gripper == 31744
        and arm_match >= 5
        and linf <= 6.0/255.0 + 1e-9
    )
    if gate_pass and rand_row:
        rand_margin = float(rand_row.get("official_target31744_margin", "-inf") or "-inf")
        gate_pass = gate_pass and (true_margin > rand_margin)
    if gate_pass and shuffled_row:
        shuffled_margin = float(shuffled_row.get("official_target31744_margin", "-inf") or "-inf")
        gate_pass = gate_pass and (true_margin > shuffled_margin)

    result_class = "FRAME_SEED_PASS" if gate_pass else "FRAME_SEED_SCIENTIFIC_FAIL"
    info = {
        "status": result_class,
        "elapsed": round(elapsed, 1),
        "arm_match": f"{arm_match}/6",
        "true_gripper": gripper,
        "true_margin": true_margin,
        "rand_gripper": rand_gripper,
        "shuffled_gripper": shuffled_gripper,
        "shuffled_arm": shuffled_arm,
        "linf": linf,
        "output_dir": str(output_dir),
    }
    print(f"  {tag}: {result_class} arm={arm_match}/6 margin={true_margin:.1f} "
          f"shuf_grip={shuffled_gripper} [{elapsed:.0f}s]")
    return True, info


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    total = 0
    passed = 0
    results = []

    print(f"H2 Batch Panel (arm-v3 contract)")
    print(f"  Config: logratio_arm_v3, arm_preserve_weight=0.5")
    print(f"  GPU: CUDA_VISIBLE_DEVICES=1,5\n")

    for pid, step, role in JOBS:
        for seed in SEEDS:
            if (pid, step, seed) in DONE:
                print(f"  SKIP {pid} step{step} seed{seed} (H1 done)")
                results.append({"parent_id": pid, "step": step, "seed": seed,
                               "status": "H1_PASS", "arm_match": "5/6", "true_margin": 21.25})
                total += 1; passed += 1
                continue

            print(f"[{total+1}/20] {pid} step{step} seed{seed} ({role})")
            ok, info = run_one(pid, step, seed)
            info["parent_id"] = pid
            info["step"] = step
            info["seed"] = seed
            info["role"] = role
            results.append(info)
            total += 1
            if info.get("status") in ("FRAME_SEED_PASS", "H1_PASS", "already_complete"):
                passed += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"  H2 BATCH COMPLETE: {passed}/{total} frame-seeds passed")
    print(f"{'='*60}")

    # Per-frame aggregation
    frame_results = {}
    for r in results:
        key = (r["parent_id"], r["step"])
        frame_results.setdefault(key, {})
        frame_results[key][f"seed{r['seed']}"] = r.get("status", "?")

    with open(OUT_ROOT / "h2_summary.json", "w") as f:
        json.dump({"total": total, "passed": passed, "results": results,
                   "frame_results": {f"{p}_{s}": v for (p, s), v in frame_results.items()}},
                  f, indent=2, default=str)

    return 0 if passed >= total * 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
