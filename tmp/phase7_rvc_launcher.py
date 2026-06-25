#!/usr/bin/env python3
"""Phase 7 Object: RvC timing ablation launcher.

RvC = Raw-vs-Clamped: ablate whether timing precision (learned emit step)
vs fixed-anchor substitution changes attack efficacy.

Design:
  - 15 cells: top 9 clean-qualified + 3 NC FT cells + 3 TV-miss cells (if any)
    If TV-miss cells = 0, use 10 qualified + 3 NC FT + 2 extra qualified
  - 2 conditions per cell: TRUE_T10 at learned emit step vs TRUE_T10 at teacher anchor
  - 30 rollouts total
  - 1 seed per cell (seed 42, same as canary)
"""
import argparse, csv, json, os, subprocess, sys, hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SERVER_REPO = Path("/mnt/sdc/dty_user/openvla_attack")

V2_CKPT = "/mnt/sdc/dty_user/openvla_attack/outputs/sc5_v2_seed42/sc5_mlp_v2.pt"
V2_CKPT_SHA = "b679e4e072531c70511a336ed68c563cf746938f6864b3cbd14f333e4f0eb09c"

BRIDGE_SCRIPT = "/mnt/sdc/dty_user/openvla_attack/scripts/stageb/run_v2_vis_sc5_mlp_bridge.py"
PYTHON = "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python3"

# Known clean-qualified cells (from CLEAN baseline): 9 cells with emit+success
QUALIFIED_CELLS = [
    ("butter_s0", 6, 0, 76, 85),
    ("butter_s2", 6, 2, 82, 85),
    ("ketchup_s0", 8, 0, 91, 94),
    ("salad_dressing_s0", 4, 0, 74, 72),
    ("bbq_sauce_s0", 1, 0, 77, 79),
    ("milk_s4", 3, 4, 72, 68),
    ("orange_juice_s0", 2, 0, 95, 96),
    ("tomato_sauce_s0", 5, 0, 117, 120),
    ("alphabet_soup_s0", 7, 0, 108, 112),
]

# NC FT cells
NC_FT_CELLS = [
    ("census_t6_s16", 6, 16, 235, -1),  # teacher anchor TBD after census
    ("census_t7_s12", 7, 12, 146, -1),
]

# TV-miss cells (V2 doesn't emit on Teacher TV)
TV_MISS_CELLS = [
    # To be filled after attack benchmark completes
    # These are the cells where V2 emit = -1 but Teacher says TV
]

RVC_OUT = "/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/rvc"


def select_rvc_cells():
    """Select 15 cells for RvC ablation.

    Priority: 9 qualified + 3 NC FT + 3 TV-miss (if any)
    If TV-miss < 3, fill with remaining qualified cells.
    """
    cells = []

    # Top 9 qualified (use all 9)
    for name, task, state, emit, anchor in QUALIFIED_CELLS:
        cells.append({
            "cell": name, "task_idx": task, "state_id": state,
            "mlp_emit": emit, "teacher_anchor": anchor,
            "category": "TV_qualified",
        })

    # 3 NC FT cells
    for name, task, state, emit, anchor in NC_FT_CELLS:
        cells.append({
            "cell": name, "task_idx": task, "state_id": state,
            "mlp_emit": emit, "teacher_anchor": anchor,
            "category": "NC_FT",
        })

    # Add TV-miss cells if any (placeholder)
    for name, task, state, emit, anchor in TV_MISS_CELLS:
        cells.append({
            "cell": name, "task_idx": task, "state_id": state,
            "mlp_emit": emit, "teacher_anchor": anchor,
            "category": "TV_miss",
        })

    # If fewer than 15, add extra qualified cells from remaining Object cells
    # For now, target is min(15, len(cells)) or 15 if we have enough
    return cells[:15]


def generate_manifest(cells):
    """Generate RvC manifest CSV with 2 conditions per cell."""
    rows = []
    run_id = 0
    for c in cells:
        for cond_label, cond_type, use_anchor in [
            ("LEARNED_EMIT", "TRUE_T10", False),
            ("TEACHER_ANCHOR", "TRUE_T10", True),
        ]:
            run_id += 1
            rows.append({
                "run_id": f"rvc_{run_id:03d}",
                "cell": c["cell"],
                "task_idx": c["task_idx"],
                "state_id": c["state_id"],
                "category": c["category"],
                "condition": cond_type,
                "trigger_mode": cond_label,
                "mlp_emit": c["mlp_emit"],
                "teacher_anchor": c["teacher_anchor"],
                "use_anchor_override": use_anchor,
                "seed_id": 42,
                "checkpoint": V2_CKPT,
                "checkpoint_sha": V2_CKPT_SHA,
                "backend": "upstream_tf_jpeg",
                "output_dir": f"{RVC_OUT}/rvc_{run_id:03d}_{c['cell']}_{cond_label}",
            })

    manifest_path = os.path.join(RVC_OUT, "RVC_MANIFEST.csv")
    os.makedirs(RVC_OUT, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    print(f"RvC manifest: {manifest_path} ({len(rows)} runs)")
    return manifest_path, rows


def generate_launch_script(rows, gpu=5):
    """Generate bash launch script for RvC runs."""
    script_path = os.path.join(RVC_OUT, "launch_rvc.sh")
    lines = [
        "#!/bin/bash",
        "set -e",
        f"GPU={gpu}",
        "cd /mnt/sdc/dty_user/openvla_attack",
        "export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1",
        "export OPENVLA_DTYPE=bfloat16 OPENVLA_ATTN_IMPLEMENTATION=eager",
        "export OPENVLA_MODEL_PATH=/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-object",
        "export HOME=/mnt/sdc/dty_user/openvla_attack/sandbox_home TMPDIR=/mnt/sdc/dty_user/openvla_attack/tmp",
        "export TF_FORCE_GPU_ALLOW_GROWTH=true",
        "export CUDA_VISIBLE_DEVICES=$GPU",
        f"PY={PYTHON}",
        f"B={BRIDGE_SCRIPT}",
        f"COMMIT=$(git rev-parse HEAD)",
        f"OUT={RVC_OUT}",
        'mkdir -p $OUT',
        f'echo "=== RvC GPU=$GPU commit=$COMMIT $(date) ==="',
        "",
    ]

    for r in rows:
        run_dir = r["output_dir"]
        if r["use_anchor_override"]:
            # Use teacher anchor instead of MLP emit
            anchor_arg = f"--anchor {r['teacher_anchor']}"
            note = "TEACHER_ANCHOR"
        else:
            anchor_arg = f"--anchor {r['teacher_anchor']}"  # audit only
            note = "LEARNED_EMIT"

        lines.append(f"# {r['run_id']}: {r['cell']} {note}")
        lines.append(f"echo '=== {r['run_id']} {r['cell']} {note} $(date) ==='")
        lines.append(f"rm -rf {run_dir}; mkdir -p {run_dir}")
        lines.append(
            f"$PY $B --condition {r['condition']} --state_id {r['state_id']} "
            f"{anchor_arg} --seed_id {r['seed_id']} --task_idx {r['task_idx']} "
            f"--output_dir {run_dir} --render_gpu $GPU --mlp_path {V2_CKPT} "
            f"--libero_preprocess_backend upstream_tf_jpeg "
            f"> {run_dir}/stdout.log 2> {run_dir}/stderr.log"
        )
        lines.append(f"echo '{r['run_id']}: '$(tail -1 {run_dir}/stdout.log)")

    lines.append(f'echo "=== RvC DONE $(date) ==="')

    with open(script_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(script_path, 0o755)
    print(f"Launch script: {script_path}")
    return script_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=5, help="GPU device for RvC runs")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    print("=== Phase 7: RvC Timing Ablation ===")
    cells = select_rvc_cells()
    print(f"Selected {len(cells)} cells:")
    for c in cells:
        print(f"  {c['cell']}: {c['category']} emit={c['mlp_emit']} anchor={c['teacher_anchor']}")

    manifest_path, rows = generate_manifest(cells)
    script_path = generate_launch_script(rows, gpu=args.gpu)

    if not args.dry_run:
        print(f"\nTo launch on server:")
        print(f"  bash {script_path}")

    print(f"\nExpected wall time: ~{len(rows) * 2.5 / 60:.1f} minutes "
          f"({len(rows)} rollouts x ~2.5 min each)")


if __name__ == "__main__":
    main()
