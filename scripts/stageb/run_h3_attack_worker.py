#!/usr/bin/env python3
"""A0: H3 V4 attack worker — eligibility audit + V4 canary on eligible frames."""
import csv, hashlib, json, os, subprocess, sys, time, yaml
from pathlib import Path

REPO = Path(os.environ.get("H3_REPO", "/data/liuyu/worktrees/l3_h3_h5_2h_20260617"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

PKG_V2 = "/data/liuyu/outputs/l3_deepseek_autonomous_20260617_r1/canonical_packages_v2"
H3_PKG = "/data/liuyu/outputs/l3_h3_h5_2h_20260617_r1/h3_packages"
H2_V4_DIR = "/data/liuyu/outputs/l3_deepseek_autonomous_20260617_r1/h2_v4_primary"
H1_V4_DIR = "/data/liuyu/outputs/l3_deepseek_autonomous_20260617_r1/h1_v4_butter_s11_step60_seed81/canary"
OUT_BASE = Path(os.environ.get("H3_OUT", "/data/liuyu/outputs/l3_h3_h5_2h_20260617_r1"))
PY = "/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python"
RUNNER = REPO / "scripts/stageb/run_m3_step78_true_pgd_fixed_frame.py"
V4_TPL = REPO / "configs/m3_butter_s11_step60_v4.yaml"

# All 21 preregistered steps with known eligibility from C0+H0
ALL_STEPS = {
    ("butter_s11", 57): "ineligible_31744", ("butter_s11", 58): "ineligible_31744",
    ("butter_s11", 59): "ineligible_31744", ("butter_s11", 60): "eligible",
    ("butter_s11", 61): "eligible_new", ("butter_s11", 62): "eligible_new",
    ("butter_s11", 63): "eligible_new",
    ("tomato_sauce_s23", 138): "ineligible_31744", ("tomato_sauce_s23", 139): "ineligible_31744",
    ("tomato_sauce_s23", 140): "ineligible_31744", ("tomato_sauce_s23", 141): "eligible",
    ("tomato_sauce_s23", 142): "ineligible_31744", ("tomato_sauce_s23", 143): "ineligible_31744",
    ("tomato_sauce_s23", 144): "ineligible_31744",
    ("salad_dressing_s11", 56): "ineligible_31744", ("salad_dressing_s11", 57): "ineligible_31744",
    ("salad_dressing_s11", 58): "ineligible_31744", ("salad_dressing_s11", 59): "eligible",
    ("salad_dressing_s11", 60): "eligible_new", ("salad_dressing_s11", 61): "eligible_new",
    ("salad_dressing_s11", 62): "eligible_new",
}

SELECTED_PARENTS = {
    "butter_s11": ("butter", 11), "tomato_sauce_s23": ("tomato_sauce", 23),
    "salad_dressing_s11": ("salad_dressing", 11),
}

# Existing anchor results that can be reused (SHA-identical)
ANCHOR_RESULTS = {
    ("butter_s11", 60, 81): H1_V4_DIR,
    ("butter_s11", 60, 82): os.path.join(H2_V4_DIR, "butter_s11_step0060_seed82", "canary"),
    ("tomato_sauce_s23", 141, 81): os.path.join(H2_V4_DIR, "tomato_sauce_s23_step0141_seed81", "canary"),
    ("tomato_sauce_s23", 141, 82): os.path.join(H2_V4_DIR, "tomato_sauce_s23_step0141_seed82", "canary"),
    ("salad_dressing_s11", 59, 81): os.path.join(H2_V4_DIR, "salad_dressing_s11_step0059_seed81", "canary"),
    ("salad_dressing_s11", 59, 82): os.path.join(H2_V4_DIR, "salad_dressing_s11_step0059_seed82", "canary"),
}


def find_pkg(pid, step):
    """Find package dir for a frame — new or existing."""
    task = SELECTED_PARENTS[pid][0]
    # Check H3 new captures
    for prefix in [f"{pid}_{task}", f"{pid}"]:
        p = os.path.join(H3_PKG, f"{prefix}_step{step:04d}")
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "clean_generation.json")):
            return p
    # Check v2 existing
    p = os.path.join(PKG_V2, f"{pid}_step{step:04d}")
    if os.path.isdir(p):
        return p
    return None


def run_one(pid, step, seed, worker_label):
    """Run V4 canary on a single frame-seed."""
    tag = f"{pid}_step{step:04d}_seed{seed}"
    job_dir = OUT_BASE / "h3_attacks" / tag
    output_dir = job_dir / "canary"

    # Check reuse
    reuse_key = (pid, step, seed)
    if reuse_key in ANCHOR_RESULTS and os.path.isdir(ANCHOR_RESULTS[reuse_key]):
        print(f"  REUSE {tag} (H2 anchor, SHA-identical)")
        return True, str(ANCHOR_RESULTS[reuse_key])

    if (output_dir / "m3_v4_selected_results.csv").exists():
        print(f"  SKIP {tag} (already done)")
        return True, str(output_dir)

    input_dir = job_dir / "input"
    config_path = job_dir / "config.yaml"
    job_dir.mkdir(parents=True, exist_ok=True)

    # Prepare input
    pkg_dir = find_pkg(pid, step)
    if not pkg_dir:
        print(f"  MISSING_PKG {tag}")
        return False, ""

    import shutil
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(os.path.join(pkg_dir, "raw_frame.npy"), input_dir / "raw_agentview_step78.npy")
    with open(os.path.join(pkg_dir, "clean_generation.json")) as f:
        d = json.load(f)
    gen = {
        "instruction": d["instruction"], "clean_action": d["clean_action"],
        "clean_exact_7_tokens": d["exact_clean_7_tokens"],
        "official": {"tokens": d["exact_clean_7_tokens"], "arm_prefix": d["clean_arm_prefix"],
                     "gripper_token": d["clean_gripper_token"],
                     "score_invariant": d.get("score_invariant", d.get("official_score_invariant", {})),
                     "target_stats": {}},
    }
    with open(input_dir / "clean_generation_step78.json", "w") as f:
        json.dump(gen, f)

    # Create config
    task, sid = SELECTED_PARENTS[pid]
    with open(V4_TPL) as f:
        cfg = yaml.safe_load(f)
    cfg["input"]["task"] = task
    cfg["input"]["state_id"] = sid
    cfg["input"]["absolute_step"] = step
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    cuda_devs = os.environ.get("CUDA_VISIBLE_DEVICES", "1,5")
    start = time.time()
    r = subprocess.run(
        [PY, str(RUNNER), "--config", str(config_path), "--mode", "canary_v4",
         "--input_dir", str(input_dir), "--output_dir", str(output_dir),
         "--attack_seed", str(seed)],
        cwd=str(REPO), capture_output=True, text=True, timeout=600,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": cuda_devs},
    )
    elapsed = time.time() - start

    if r.returncode != 0:
        print(f"  FAIL [{elapsed:.0f}s]: {r.stderr[-150:]}")
        return False, str(output_dir)

    # Quick classification
    sel_csv = output_dir / "m3_v4_selected_results.csv"
    if sel_csv.exists():
        rows = list(csv.DictReader(open(sel_csv)))
        true_row = next((x for x in rows if "TRUE" in x.get("condition", "")), None)
        if true_row:
            arm = true_row.get("arm_prefix_match_count", "?")
            margin = true_row.get("official_target31744_margin", "?")
            res = true_row.get("stage_result", "?")
            print(f"  {tag}: {res} arm={arm}/6 margin={margin} [{elapsed:.0f}s]")
    return True, str(output_dir)


def main():
    worker = os.environ.get("H3_WORKER", "A")
    # Static queue assignment: Worker A gets butter, Worker B gets salad+tomato reminders
    if worker == "A":
        my_jobs = [
            ("butter_s11", 61, 81), ("butter_s11", 61, 82),
            ("butter_s11", 62, 81), ("butter_s11", 62, 82),
            ("butter_s11", 63, 81), ("butter_s11", 63, 82),
        ]
    else:
        my_jobs = [
            ("salad_dressing_s11", 60, 81), ("salad_dressing_s11", 60, 82),
            ("salad_dressing_s11", 61, 81), ("salad_dressing_s11", 61, 82),
            ("salad_dressing_s11", 62, 81), ("salad_dressing_s11", 62, 82),
        ]

    print(f"Worker {worker}: {len(my_jobs)} attack jobs")

    results = []
    for i, (pid, step, seed) in enumerate(my_jobs):
        print(f"[{i+1}/{len(my_jobs)}] {pid} step{step} seed{seed}")
        ok, out = run_one(pid, step, seed, worker)
        results.append({"pid": pid, "step": step, "seed": seed, "ok": ok, "out": out})

    n_ok = sum(1 for r in results if r["ok"])
    print(f"\nWorker {worker}: {n_ok}/{len(my_jobs)} jobs ok")
    with open(OUT_BASE / f"a0_worker_{worker.lower()}_ledger.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
