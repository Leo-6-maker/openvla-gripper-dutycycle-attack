#!/usr/bin/env python3
"""V2 Layer123 online regression launcher.
Re-runs old 11-object Layer123 conditions with V2 seed42 detector.
Compares against V1 baseline from outputs/layer123_final3/.
"""
import argparse, csv, json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# ── Old V1 baseline results (11 objects, from log.txt) ──
OBJECTS = [
    {"object": "alphabet_soup",   "state_id": 0, "teacher_anchor": 86,  "task_idx": -1},  # needs task_idx
    {"object": "bbq_sauce",       "state_id": 0, "teacher_anchor": 128, "task_idx": -1},
    {"object": "butter",          "state_id": 0, "teacher_anchor": 85,  "task_idx": 6},
    {"object": "butter",          "state_id": 2, "teacher_anchor": 100, "task_idx": 6},
    {"object": "chocolate_pudding","state_id": 2, "teacher_anchor": 90,  "task_idx": -1},
    {"object": "cream_cheese",    "state_id": 0, "teacher_anchor": 116, "task_idx": -1},
    {"object": "ketchup",         "state_id": 0, "teacher_anchor": 95,  "task_idx": -1},
    {"object": "milk",            "state_id": 4, "teacher_anchor": 92,  "task_idx": -1},
    {"object": "orange_juice",    "state_id": 0, "teacher_anchor": 167, "task_idx": -1},
    {"object": "salad_dressing",  "state_id": 0, "teacher_anchor": 84,  "task_idx": -1},
    {"object": "tomato_sauce",    "state_id": 0, "teacher_anchor": 176, "task_idx": -1},
]

CONDITIONS = ["CLEAN", "TRUE_T10", "RAND_T10"]

# Server paths
SERVER_REPO = Path("/mnt/sdc/dty_user/openvla_attack")
BRIDGE = SERVER_REPO / "scripts/stageb/run_v2_vis_sc5_mlp_bridge.py"
V2_CKPT = SERVER_REPO / "outputs/sc5_v2_seed42/sc5_mlp_v2.pt"
PYTHON = SERVER_REPO / "envs/openvla-official-a800/bin/python3"
OUT_BASE = SERVER_REPO / "evidence/m1c/phase6c_v2_regression"


def write_manifest():
    """Generate manifest CSV for V2 regression runs."""
    rows = []
    for obj in OBJECTS:
        for cond in CONDITIONS:
            cell_id = f"{obj['object']}_s{obj['state_id']}_{cond.lower()}"
            out_dir = OUT_BASE / cell_id
            cmd = (
                f"{PYTHON} {BRIDGE} "
                f"--condition {cond} "
                f"--state_id {obj['state_id']} "
                f"--anchor {obj['teacher_anchor']} "
                f"--seed_id 42 "
                f"--output_dir {out_dir} "
                f"--render_gpu $GPU "
                f"--mlp_path {V2_CKPT} "
                f"--save_video --source_commit $COMMIT "
                f"--video_fps 10 --frame_stride 1"
            )
            if obj["task_idx"] >= 0:
                cmd += f" --task_idx {obj['task_idx']}"
            rows.append({
                "cell_id": cell_id,
                "object": obj["object"],
                "state_id": obj["state_id"],
                "teacher_anchor": obj["teacher_anchor"],
                "condition": cond,
                "command": cmd,
                "v1_emit_step": "",  # fill from log
                "v1_success": "",    # fill from log
            })

    manifest_path = REPO / "evidence/m1c/phase6c_v2_regression_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Manifest: {manifest_path} ({len(rows)} cells)")

    # Also write launch script
    script = "#!/bin/bash\nset -e\n"
    script += f"cd {SERVER_REPO}\n"
    script += f"export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n"
    script += f"export OPENVLA_DTYPE=bfloat16 OPENVLA_ATTN_IMPLEMENTATION=eager\n"
    script += f'export OPENVLA_MODEL_PATH={SERVER_REPO}/models/openvla-7b-finetuned-libero-object\n'
    script += f"export HOME={SERVER_REPO}/sandbox_home TMPDIR={SERVER_REPO}/tmp\n"
    script += f"PY={PYTHON}\n"
    script += f"BRIDGE={BRIDGE}\n"
    script += f"CKPT={V2_CKPT}\n"
    script += f"OUT={OUT_BASE}\n"
    script += f"COMMIT=$(git rev-parse HEAD)\n"
    script += f'GPU=${{1:-0}}\n'
    script += f'export CUDA_VISIBLE_DEVICES=$GPU\n'
    script += f'echo "GPU=$GPU COMMIT=$COMMIT start=$(date)"\n'
    script += f'mkdir -p $OUT\n\n'

    for obj in OBJECTS:
        for cond in CONDITIONS:
            cell_id = f"{obj['object']}_s{obj['state_id']}_{cond.lower()}"
            script += f"# {cell_id}\n"
            script += f'if [ -f "$OUT/{cell_id}/.done" ]; then echo "SKIP {cell_id}"; else\n'
            script += f'  rm -rf "$OUT/{cell_id}" && mkdir -p "$OUT/{cell_id}"\n'
            task_arg = f" --task_idx {obj['task_idx']}" if obj['task_idx'] >= 0 else ""
            script += f'  $PY $BRIDGE --condition {cond} --state_id {obj["state_id"]} --anchor {obj["teacher_anchor"]} --seed_id 42 --output_dir "$OUT/{cell_id}" --render_gpu $GPU --mlp_path $CKPT --save_video --source_commit $COMMIT --video_fps 10 --frame_stride 1{task_arg} > "$OUT/{cell_id}/stdout.log" 2> "$OUT/{cell_id}/stderr.log"\n'
            script += f'  touch "$OUT/{cell_id}/.done"\n'
            script += f'fi\n\n'

    script += f'echo "ALL DONE $(date)"\n'

    script_path = REPO / "tmp/launch_v2_regression.sh"
    with open(script_path, "w") as f:
        f.write(script)
    print(f"Launch script: {script_path}")

    # Count GPU hours
    n_cells = len(OBJECTS) * len(CONDITIONS)
    print(f"\nEstimated: {n_cells} rollouts, ~{n_cells * 3:.0f}-{n_cells * 6:.0f} GPU-minutes")
    print(f"Run: bash {script_path} <GPU_ID>")


if __name__ == "__main__":
    write_manifest()
