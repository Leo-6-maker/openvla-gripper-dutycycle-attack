#!/usr/bin/env python3
"""Diagnose task_01 object-language alias mismatch.

task_01: pick_up_the_cream_cheese_and_place_it_in_the_basket
Problem: stable_carry=953 but primary=0

Hypotheses:
  1. MuJoCo body name contains filtered keyword (visual, collision, link)
  2. Different body (not cream_cheese) is closer to gripper
  3. Body distance > 0.15 threshold

Run on server: python scripts/stageb/diagnose_task01_object_match.py
"""

from __future__ import annotations

import os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))

import numpy as np


def main():
    from libero.libero import benchmark, get_libero_path
    from gripper_attack.libero_v4_env_factory import build_v4_exact_env, apply_dummy_wait
    from scripts.stageb.c2f_libero_openvla_adapter import _TeacherLabeler, _resolve_task_language, _visible_gpu_id

    suite = "libero_object"
    task_idx = 1  # task_01: pick up the cream cheese

    bm = benchmark.get_benchmark_dict()
    task_suite = bm[suite]()
    task = task_suite.get_task(task_idx)
    init_states = task_suite.get_task_init_states(task_idx)

    task_language, source = _resolve_task_language(task, {})
    print(f"=== Task Language ===")
    print(f"  raw: {task_language!r}")
    print(f"  source: {source}")
    print()

    task_bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
    env, obs = build_v4_exact_env(str(task_bddl), _visible_gpu_id(), 300, 10)
    obs = env.set_init_state(init_states[0])
    env, obs = apply_dummy_wait(env, obs, 10)

    # ── Body name audit ──
    print(f"=== All MuJoCo Body Names (n={len(env.sim.model.body_names)}) ===")
    FILTER_KW = ["robot", "floor", "world", "gripper", "link", "collision", "visual"]
    kept = []
    filtered = []
    for bn in env.sim.model.body_names:
        if any(skip in bn for skip in FILTER_KW):
            filtered.append(bn)
        else:
            kept.append(bn)
    print(f"  Kept ({len(kept)}):")
    for bn in sorted(kept):
        print(f"    {bn}")
    print(f"  Filtered ({len(filtered)}):")
    for bn in sorted(filtered):
        print(f"    {bn}")
    print()

    # Check if "cream_cheese" appears in any body name
    cc_bodies = [bn for bn in env.sim.model.body_names if "cream" in bn.lower() or "cheese" in bn.lower()]
    print(f"=== Bodies containing 'cream' or 'cheese' ===")
    for bn in cc_bodies:
        is_filtered = any(skip in bn for skip in FILTER_KW)
        print(f"  {bn}  (filtered={is_filtered})")
    print()

    # ── Step through episode and log stable_carry steps ──
    from gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2
    teacher = _TeacherLabeler(env, task_language, target_object_name="cream_cheese_1")
    _streamer = SC5StreamingFeatureAdapterV2()

    eef_sid = env.sim.model.site_name2id("gripper0_grip_site")
    _eef_init = env.sim.data.site_xpos[eef_sid]
    _prev_eef = (float(_eef_init[0]), float(_eef_init[1]), float(_eef_init[2]))

    import torch
    from v4_run_eval_openvla import decode_with_scores, postprocess_openvla_action_for_libero, physical_gripper_state

    model_path = "/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-object"
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoModelCls
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoModelCls
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    vla_model = AutoModelCls.from_pretrained(model_path, trust_remote_code=True, local_files_only=True,
                                              torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda:0")
    model_dtype = next(vla_model.parameters()).dtype

    stable_carry_log = []
    MAX_STEPS = 300

    for step in range(MAX_STEPS):
        raw = np.asarray(obs["agentview_image"])
        if raw.ndim == 2:
            raw = np.stack([raw] * 3, axis=-1)
        if raw.ndim != 3 or raw.shape[-1] < 3:
            break
        raw = raw[..., :3].copy()

        action, _, _, _ = decode_with_scores(
            vla_model, processor, "cuda:0", raw, task_language, "libero_object", 8,
            libero_preprocess_backend="upstream_tf_jpeg", center_crop=True,
            resize_size=224, drop_attention_mask=True,
        )
        env_action = postprocess_openvla_action_for_libero(
            np.asarray(action, dtype=np.float32), enabled=True,
        )

        lbl = teacher.label(step)
        phase = lbl["phase"]

        if phase == "stable_carry":
            # Log diagnostic for this stable_carry step
            eef_pos = env.sim.data.site_xpos[eef_sid]
            eef_z = float(eef_pos[2])
            gq = env.sim.data.qpos[7:9]
            gripper_closed = float(gq[0] + gq[1]) < 0.04

            # Re-run identify + match
            grasped_obj = teacher._identify_grasped_object()
            matches = teacher._object_matches_task_target(grasped_obj)

            # Find all bodies within 0.15
            bodies_in_range = []
            for body_name in env.sim.model.body_names:
                if any(skip in body_name for skip in ["robot", "floor", "world", "gripper", "link", "collision", "visual"]):
                    continue
                try:
                    bid = env.sim.model.body_name2id(body_name)
                    body_pos = env.sim.data.body_xpos[bid]
                    dist = float(np.linalg.norm(eef_pos - body_pos))
                    if dist < 0.15:
                        bodies_in_range.append((body_name, round(dist, 4)))
                except Exception:
                    continue
            bodies_in_range.sort(key=lambda x: x[1])

            log_entry = {
                "step": step,
                "eef_z": round(eef_z, 4),
                "gripper_closed": bool(gripper_closed),
                "grasped_obj": grasped_obj,
                "matches_target": matches,
                "event_role": lbl["event_role"],
                "primary": lbl["primary_attackable"],
                "bodies_in_range": bodies_in_range[:5],
                "close_start_eef_z": round(teacher._close_start_eef_z, 4),
                "max_eef_z_since_close": round(teacher._max_eef_z_since_close, 4),
                "rel_lift": round(teacher._max_eef_z_since_close - teacher._close_start_eef_z, 4),
            }
            stable_carry_log.append(log_entry)

            if len(stable_carry_log) <= 5:
                print(f"Step {step} stable_carry:")
                print(f"  eef_z={eef_z:.4f}, rel_lift={log_entry['rel_lift']:.4f}")
                print(f"  grasped_obj={grasped_obj!r}, matches_target={matches}")
                print(f"  event_role={lbl['event_role']}, primary={lbl['primary_attackable']}")
                print(f"  bodies_in_range (<0.15): {bodies_in_range[:5]}")
                print()

        obs, reward, done, info = env.step(env_action)

    env.close()

    # ── Summary ──
    n_sc = len(stable_carry_log)
    n_primary = sum(1 for e in stable_carry_log if e["primary"])
    n_grasped_but_mismatch = sum(1 for e in stable_carry_log if e["grasped_obj"] and not e["matches_target"])
    n_no_grasped = sum(1 for e in stable_carry_log if not e["grasped_obj"])
    n_match = sum(1 for e in stable_carry_log if e["matches_target"])

    print(f"=== Summary ===")
    print(f"  total stable_carry steps: {n_sc}")
    print(f"  primary_attackable: {n_primary}")
    print(f"  grasped but mismatch: {n_grasped_but_mismatch}")
    print(f"  no object identified: {n_no_grasped}")
    print(f"  matched target: {n_match}")

    if n_sc > 0:
        # Show unique grasped_obj values
        unique_objs = set(e["grasped_obj"] for e in stable_carry_log if e["grasped_obj"])
        print(f"  unique grasped objects: {unique_objs}")
        unique_roles = set(e["event_role"] for e in stable_carry_log)
        print(f"  unique event_roles: {unique_roles}")

        # Show first mismatch detail
        mismatches = [e for e in stable_carry_log if e["grasped_obj"] and not e["matches_target"]]
        if mismatches:
            print(f"\n  First mismatch example:")
            m = mismatches[0]
            print(f"    step={m['step']}, grasped_obj={m['grasped_obj']!r}")
            print(f"    bodies_in_range={m['bodies_in_range']}")
            print(f"    task_language={task_language!r}")


if __name__ == "__main__":
    main()
