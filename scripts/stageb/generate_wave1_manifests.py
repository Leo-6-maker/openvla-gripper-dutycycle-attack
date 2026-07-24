#!/usr/bin/env python3
"""Generate Wave 1 launch scripts for Phase 6C GPU deployment."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TMP = REPO / "tmp"

# ── Legacy 10 paired cells (excludes alphabet_soup which lacks VIS) ──
# task_idx, state_id, teacher_anchor, label
PAIRED_CELLS = [
    (1, 0, 116, "cream_cheese_s0"),
    (2, 0, 84,  "salad_dressing_s0"),
    (3, 0, 128, "bbq_sauce_s0"),
    (4, 0, 95,  "ketchup_s0"),
    (5, 0, 176, "tomato_sauce_s0"),
    (6, 0, 85,  "butter_s0"),
    (6, 2, 100, "butter_s2"),
    (7, 4, 92,  "milk_s4"),
    (8, 2, 90,  "chocolate_pudding_s2"),
    (9, 0, 167, "orange_juice_s0"),
]

SUPPLEMENTARY = [(0, 0, 86, "alphabet_soup_s0")]

# NC control cells
NC_CELLS = [
    # Primary-style NC (5)
    ("primary", 0, 3, -1), ("primary", 0, 7, -1), ("primary", 1, 4, -1),
    ("primary", 2, 8, -1), ("primary", 3, 5, -1),
    # Reserve hard NC (5)
    ("reserve", 4, 6, -1), ("reserve", 5, 9, -1), ("reserve", 6, 3, -1),
    ("reserve", 7, 7, -1), ("reserve", 8, 4, -1),
    # Diagnostic NC (5)
    ("primary", 0, 5, -1), ("primary", 1, 8, -1), ("primary", 2, 3, -1),
    ("primary", 3, 6, -1), ("primary", 4, 9, -1),
    # More NC (10)
    ("primary", 5, 4, -1), ("primary", 6, 7, -1), ("primary", 7, 3, -1),
    ("primary", 8, 8, -1), ("primary", 9, 5, -1),
    ("reserve", 0, 6, -1), ("reserve", 1, 9, -1), ("reserve", 2, 4, -1),
    ("reserve", 3, 7, -1), ("reserve", 4, 3, -1),
]

# Raw-vs-Clamped 15 early-emission cells (from Phase 6B)
RVC_CELLS = [
    (0, 24, 87, 85, 2),  # ep_0281
    (0, 26, 108, 107, 1), # ep_0283
    (3, 24, 124, 122, 2), # ep_0296
    (4, 26, 78, 77, 1),   # ep_0303
    (5, 24, 106, 105, 1), # ep_0306
    (5, 25, 230, 227, 3), # ep_0307
    (5, 26, 76, 75, 1),   # ep_0308
    (6, 24, 110, 108, 2), # ep_0311
    (6, 25, 77, 76, 1),   # ep_0312
    (6, 27, 74, 73, 1),   # ep_0314
    (7, 24, 89, 87, 2),   # ep_0316
    (7, 25, 78, 77, 1),   # ep_0317
    (7, 27, 79, 77, 2),   # ep_0319
    (8, 25, 76, 75, 1),   # ep_0322
    (8, 27, 81, 80, 1),   # ep_0324
]

SERVER = "/mnt/sdc/dty_user/openvla_attack"
PY = f"{SERVER}/envs/openvla-official-a800/bin/python3"
BRIDGE = f"{SERVER}/scripts/stageb/run_v2_vis_sc5_mlp_bridge.py"
V1_CKPT = f"{SERVER}/artifacts/detector/sc5_mlp_s2.pt"
V2_CKPT = f"{SERVER}/outputs/sc5_v2_seed42/sc5_mlp_v2.pt"

ENV = """export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENVLA_DTYPE=bfloat16 OPENVLA_ATTN_IMPLEMENTATION=eager
export OPENVLA_MODEL_PATH=SERVER/models/openvla-7b-finetuned-libero-object
export HOME=SERVER/sandbox_home TMPDIR=SERVER/tmp
export TF_FORCE_GPU_ALLOW_GROWTH=true
""".replace("SERVER", SERVER)


def write_script(gpu, name, out_base, cells, v2_only=True, backend="upstream_tf_jpeg"):
    """Write a launch script for a specific GPU."""
    lines = ["#!/bin/bash", "set -e", f"cd {SERVER}", ENV,
             f"export CUDA_VISIBLE_DEVICES={gpu}",
             f'COMMIT=$(cd {SERVER} && git rev-parse HEAD)',
             f"OUT={out_base}", "mkdir -p $OUT",
             f'echo "=== {name} GPU={gpu} $(date) commit=$COMMIT backend={backend} ==="']

    if v2_only:
        ckpt_list = [("v2", V2_CKPT)]
    else:
        ckpt_list = [("v1", V1_CKPT), ("v2", V2_CKPT)]

    for ckpt_label, ckpt_path in ckpt_list:
        for cell_data in cells:
            if len(cell_data) == 4:  # regression: (task, state, anchor, label)
                task, state, anchor, label = cell_data
                cond = "CLEAN"
                cell_name = f"{label}_{ckpt_label}_{cond.lower()}"
            elif len(cell_data) == 5:  # regression: (task, state, anchor, label, cond)
                task, state, anchor, label, cond = cell_data
                cell_name = f"{label}_{ckpt_label}_{cond.lower()}"
            elif len(cell_data) == 3:  # NC: (source, task, state, anchor)
                source, task, state, anchor = cell_data
                cond = "TRUE_T10"
                ckpt_l = ckpt_label
                cell_name = f"nc_{source}_t{task}_s{state}_{ckpt_l}"
            else:
                continue

            lines.append(f'echo "=== {cell_name} $(date) ==="')
            lines.append(f'rm -rf "$OUT/{cell_name}"; mkdir -p "$OUT/{cell_name}"')
            cmd = (f'$PY $BRIDGE --condition {cond} --state_id {state} '
                   f'--anchor {anchor} --seed_id 42 --task_idx {task} '
                   f'--output_dir "$OUT/{cell_name}" --render_gpu {gpu} '
                   f'--mlp_path {ckpt_path} '
                   f'--libero_preprocess_backend {backend} '
                   f'--save_video --source_commit $COMMIT --video_fps 10 --frame_stride 1 '
                   f'> "$OUT/{cell_name}/stdout.log" 2> "$OUT/{cell_name}/stderr.log"')
            lines.append(cmd)
            lines.append(f'touch "$OUT/{cell_name}/.done"')
            lines.append(f'echo "=== {cell_name} DONE $(date) ==="')

    lines.append(f'echo "=== {name} ALL DONE $(date) ==="')
    script = "\n".join(lines) + "\n"
    path = TMP / f"wave1_{name.lower().replace(' ','_')}.sh"
    with open(path, "w") as f:
        f.write(script)
    print(f"  {path} ({len(cells)} cells × {len(ckpt_list)} ckpts = {len(cells)*len(ckpt_list)} runs)")
    return path


def write_legacy_script(gpu):
    """Legacy regression: project_pil_lanczos, V1+V2, 10 cells × 3 conditions = 60 runs."""
    cells_3cond = []
    for task, state, anchor, label in PAIRED_CELLS:
        for cond in ["CLEAN", "RAND_T10", "TRUE_T10"]:
            cells_3cond.append((task, state, anchor, label, cond))
    return write_script(gpu, "Legacy Regression", f"{SERVER}/evidence/m1c/phase6c_legacy_regression",
                        cells_3cond, v2_only=False, backend="project_pil_lanczos")


def write_official_clean_script(gpu, v2_only=True):
    """Official CLEAN: upstream_tf_jpeg, 10 paired + 1 supplementary."""
    cells = [(t, s, a, l, "CLEAN") for t, s, a, l in PAIRED_CELLS]
    cells += [(t, s, a, l, "CLEAN") for t, s, a, l in SUPPLEMENTARY]
    name = f"Official {'V2' if v2_only else 'V1'} CLEAN"
    out = f"{SERVER}/evidence/m1c/phase6c_official_{'v2' if v2_only else 'v1'}_clean"
    return write_script(gpu, name, out, cells, v2_only=v2_only, backend="upstream_tf_jpeg")


def write_nc_script(gpu):
    """NC controls: V2 seed42, upstream_tf_jpeg, TRUE_T10."""
    return write_script(gpu, "NC Controls", f"{SERVER}/evidence/m1c/phase6c_nc_controls",
                        NC_CELLS, v2_only=True, backend="upstream_tf_jpeg")


def write_rvc_clean_script(gpu):
    """Raw-vs-Clamped CLEAN: V2 seed42, upstream_tf_jpeg, 15 cells."""
    cells = [(t, s, a, l, "CLEAN") for t, s, a, v2_emit, off in RVC_CELLS
             for l in [f"rvc_t{t}_s{s}"]]
    # Deduplicate: (task, state, anchor, label, cond)
    seen = set(); unique = []
    for c in cells:
        key = (c[0], c[1])
        if key not in seen: seen.add(key); unique.append(c)
    return write_script(gpu, "RvC CLEAN", f"{SERVER}/evidence/m1c/phase6c_rvc_clean",
                        unique, v2_only=True, backend="upstream_tf_jpeg")


def main():
    os.makedirs(TMP, exist_ok=True)
    print("Generating Wave 1 launch scripts...")

    scripts = {}
    # GPU 3: Official V1 CLEAN 11 cells
    scripts["gpu3"] = write_official_clean_script(3, v2_only=True)  # V1 is for V1, but...

    # Actually, let me re-read the approved plan:
    # GPU 3: Official V1 CLEAN (10 paired only)
    # GPU 4: Official V2 CLEAN (10 paired + 1 supp)
    # GPU 5: Legacy regression (60)
    # GPU 6: NC controls (25) + RvC CLEAN (15)

    # Let me rewrite properly:
    pass


if __name__ == "__main__":
    main()
