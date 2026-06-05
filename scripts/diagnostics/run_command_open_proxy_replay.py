#!/usr/bin/env python3
"""run_command_open_proxy_replay.py — Command-layer forced-OPEN proxy.

Forces gripper command to OPEN during the candidate window. No image PGD.
Records qpos response, done, steps, armL2. Tests physical/task susceptibility.

This is an UPPER-BOUND proxy. It cannot be used as gold VIS label.

Usage:
    PY scripts/diagnostics/run_command_open_proxy_replay.py \
      --candidate-csv tables/fast_vis_calibration_candidates_v0.csv \
      --output-csv tables/fast_vis_command_proxy_v0.csv \
      --output-report reports/FAST_VIS_COMMAND_PROXY_V0.md
"""

from __future__ import annotations
import argparse, csv, json, os, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO = Path(os.environ.get("ATTACK_REPO",
    "/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605"))

MEASUREMENT_VERSION = "v1_obs_robot0_gripper_qpos_or_mujoco_joint_lookup_20260605"
ACTION_INJECTION_VERSION = "v1_final_env_step_action_space_after_normalize_invert_20260605"
FORCED_OPEN_RAW_GRIPPER = 0.0


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-csv", default="tables/fast_vis_calibration_candidates_v0.csv")
    ap.add_argument("--output-csv", default="tables/fast_vis_command_proxy_v0.csv")
    ap.add_argument("--output-report", default="reports/FAST_VIS_COMMAND_PROXY_V0.md")
    ap.add_argument("--gpu-pair", default="0,1")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-steps", type=int, default=300)
    return ap.parse_args()


def validate_gpu_pair(gpu_pair: str):
    ids = [x.strip() for x in gpu_pair.split(",") if x.strip()]
    if any(x in {"3", "7"} for x in ids):
        raise SystemExit("INFRA_FAILED: GPU3/GPU7 are blacklisted; requested --gpu-pair=%s" % gpu_pair)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").replace(" ", "")
    if visible == "2,6" and gpu_pair.replace(" ", "") == "2,6":
        raise SystemExit(
            "INFRA_FAILED: do not combine CUDA_VISIBLE_DEVICES=2,6 with --gpu-pair 2,6; "
            "inside a remapped visible set, --gpu-pair would need logical 0,1, but this is not recommended"
        )
    return gpu_pair


def parse_gpu_ids(gpu_pair: str):
    return [int(x.strip()) for x in validate_gpu_pair(gpu_pair).split(",") if x.strip()]


def normalize_gripper_action(action, binarize=True):
    import numpy as np
    action = np.asarray(action, dtype=np.float32).copy()
    action[..., -1] = 2.0 * action[..., -1] - 1.0
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
        action[..., -1] = 1.0 if action[..., -1] == 0 else action[..., -1]
    return action


def invert_gripper_action(action):
    import numpy as np
    action = np.asarray(action, dtype=np.float32).copy()
    action[..., -1] = -1.0 * action[..., -1]
    return action


def raw_gripper_to_env_action(raw_gripper: float) -> float:
    import numpy as np
    action = np.zeros(7, dtype=np.float32)
    action[-1] = float(raw_gripper)
    return float(invert_gripper_action(normalize_gripper_action(action, binarize=True))[-1])


def read_gripper_qpos(obs, env):
    """Return (qpos_value, source, status). Never use env._joint_positions."""
    import numpy as np

    if isinstance(obs, dict) and "robot0_gripper_qpos" in obs:
        arr = np.asarray(obs.get("robot0_gripper_qpos"), dtype=np.float32).reshape(-1)
        if arr.size > 0:
            return float(arr[0]), "obs.robot0_gripper_qpos", "ok"

    sim = getattr(env, "sim", None)
    model = getattr(sim, "model", None)
    data = getattr(sim, "data", None)
    if model is not None and data is not None and hasattr(data, "qpos"):
        joint_names = list(getattr(model, "joint_names", []) or [])
        for name in joint_names:
            lname = str(name).lower()
            if "gripper" not in lname:
                continue
            try:
                jid = model.joint_name2id(name)
                adr = int(model.jnt_qposadr[jid])
                return float(data.qpos[adr]), f"mujoco_joint:{name}", "ok"
            except Exception:
                continue
    return None, "unavailable", "missing_gripper_qpos"


def infra_status_from_error(error_text: str):
    low = str(error_text).lower()
    if any(tok in low for tok in ["xid", "out of memory", "oom", "cuda illegal", "cublas"]):
        return "INFRA_FAILED"
    return "ERROR"


def load_libero_env():
    """LIBERO environment with OpenVLA task setup."""
    import libero
    from libero.libero import benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_object"]()
    return task_suite


def run_command_proxy_episode(task_suite, task_id: int, state_id: int,
                               window_start: int, window_end: int, gpu_pair: str,
                               max_steps: int):
    """Run rollout with forced OPEN gripper during the window."""
    import numpy as np
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from PIL import Image

    gpu_ids = parse_gpu_ids(gpu_pair)
    device = f"cuda:{gpu_ids[0]}"

    # Load model
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if len(gpu_ids) >= 2:
        model_kwargs.update(
            device_map="auto",
            max_memory={gpu_ids[0]: "10500MiB", gpu_ids[1]: "10500MiB", "cpu": "64GiB"},
        )
    model = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b", **model_kwargs)
    if len(gpu_ids) < 2:
        model = model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

    # Setup environment
    task = task_suite.get_task(task_id)
    task_suite.set_task_id(task_id)
    env_args = {"bddl_file_name": os.path.expanduser(task_suite.get_task_bddl_file_path(task_id)),
                "camera_heights": 224, "camera_widths": 224}
    env = task_suite.env
    env.reset()
    obs = env.set_init_state(state_id % 50)

    instruction = task.language_instruction
    action_dim = 7
    unnorm_key = "libero_goal"
    stats = list(model.norm_stats.values())[0]
    mask = np.asarray(stats.get("mask", np.ones(action_dim, dtype=bool)), dtype=bool)
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    denom = np.maximum(q99 - q01, 1e-6)
    bins = np.asarray(model.bin_centers, dtype=np.float32)
    vocab_size = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)

    t0 = time.time()
    step = 0
    done = False
    forced_open_count = 0
    qpos_first_pre = None
    qpos_min_post = None
    qpos_source = ""
    measurement_failed = False
    measurement_failure_reason = ""
    arm_l2_sum = 0.0
    clean_gripper_action = ""
    forced_gripper_action = FORCED_OPEN_RAW_GRIPPER
    forced_open_value_used = raw_gripper_to_env_action(FORCED_OPEN_RAW_GRIPPER)
    post_transform_gripper_action = ""
    post_transform_gripper_actions = []

    while step < max_steps and not done:
        qpos_pre, qpos_source_step, qpos_status = read_gripper_qpos(obs, env)
        if qpos_status != "ok":
            measurement_failed = True
            measurement_failure_reason = qpos_status
        elif not qpos_source:
            qpos_source = qpos_source_step

        img = Image.fromarray(obs["agentview_image"][::-1])
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        inputs = processor(prompt, img, return_tensors="pt")
        inputs.pop("attention_mask", None)
        input_ids = inputs["input_ids"].to(device)
        pixel_values = inputs["pixel_values"].to(device=device, dtype=torch.bfloat16)

        with torch.no_grad():
            out = model.generate(input_ids, pixel_values=pixel_values,
                                 max_new_tokens=action_dim, do_sample=False,
                                 return_dict_in_generate=True, output_scores=True)
        tokens = out.sequences[0, -action_dim:].cpu().numpy()
        action = np.zeros(action_dim, dtype=np.float32)
        for dim in range(action_dim):
            if mask[dim]:
                tid = int(vocab_size - tokens[dim] - 1)
                action[dim] = 0.5 * (bins[tid] + 1.0) * denom[dim] + q01[dim]

        if window_start <= step < window_end and clean_gripper_action == "":
            clean_gripper_action = float(action[-1])

        env_action = invert_gripper_action(normalize_gripper_action(action, binarize=True))

        # Force OPEN in final env.step action space after normalize/invert transforms.
        if window_start <= step < window_end:
            if qpos_pre is not None and qpos_first_pre is None:
                qpos_first_pre = float(qpos_pre)
            env_action[-1] = forced_open_value_used
            post_transform_gripper_action = float(env_action[-1])
            post_transform_gripper_actions.append(post_transform_gripper_action)
            forced_open_count += 1

        obs, reward, done, info = env.step(env_action)
        qpos_post, qpos_source_post, qpos_post_status = read_gripper_qpos(obs, env)
        if qpos_post_status != "ok":
            measurement_failed = True
            measurement_failure_reason = qpos_post_status
        else:
            if not qpos_source:
                qpos_source = qpos_source_post
            if window_start <= step < window_end:
                qpos_min_post = (
                    float(qpos_post) if qpos_min_post is None
                    else min(float(qpos_min_post), float(qpos_post))
                )
        step += 1

    runtime = time.time() - t0
    env.close()

    if measurement_failed or qpos_first_pre is None or qpos_min_post is None:
        qpos_opening_delta = ""
        physical_response = "measurement_failed"
        provenance_status = f"MEASUREMENT_FAILED:{measurement_failure_reason or 'missing_window_qpos'}"
    else:
        qpos_opening_delta = round(float(qpos_first_pre) - float(qpos_min_post), 6)
        physical_response = (
            "strong" if qpos_opening_delta >= 0.03
            else ("weak" if qpos_opening_delta >= 0.01 else "none")
        )
        provenance_status = "ok"

    return {
        "forced_open_count": forced_open_count,
        "qpos_opening_delta": qpos_opening_delta,
        "task_done": int(done),
        "steps": step,
        "arm_l2": round(arm_l2_sum, 4),
        "physical_response": physical_response,
        "task_failure": int(not done),
        "runtime_sec": round(runtime, 2),
        "provenance_status": provenance_status,
        "measurement_version": MEASUREMENT_VERSION,
        "action_injection_version": ACTION_INJECTION_VERSION,
        "gripper_qpos_source": qpos_source or "unavailable",
        "clean_gripper_action": clean_gripper_action,
        "forced_gripper_action": forced_gripper_action,
        "forced_open_value_used": forced_open_value_used,
        "post_transform_gripper_action": post_transform_gripper_action,
    }


def main():
    args = parse_args()
    validate_gpu_pair(args.gpu_pair)
    if not os.path.exists(args.candidate_csv):
        print(f"ERROR: {args.candidate_csv} not found"); sys.exit(1)

    with open(args.candidate_csv, newline="") as f:
        candidates = list(csv.DictReader(f))

    if args.dry_run:
        for c in candidates:
            print(f"  {c['task_key']}_s{c['state_id']} [{c['parent_window_start']},{c['parent_window_end']}]")
        return

    print("Loading LIBERO environment...")
    task_suite = load_libero_env()
    task_names = [task.name for task in task_suite.tasks]
    print(f"Tasks: {len(task_names)}")

    results = []
    for i, c in enumerate(candidates):
        task = c["task_key"]
        sid = int(c["state_id"])
        ws = int(c["parent_window_start"])
        we = int(c["parent_window_end"])

        # Find task ID
        matches = [j for j, name in enumerate(task_names) if task.replace("_", " ") in name.lower()]
        if not matches:
            print(f"SKIP {task}_s{sid}: task not found")
            results.append(dict(task_key=task, state_id=sid, window_start=ws, window_end=we,
                                label=c.get("full_vis_label",""), label_source="full_vis_label",
                                label_confidence="silver_proxy_not_gold",
                                denominator_status="not_applicable_command_proxy",
                                gpu_pair=args.gpu_pair, runtime_sec="",
                                measurement_version=MEASUREMENT_VERSION,
                                action_injection_version=ACTION_INJECTION_VERSION,
                                gripper_qpos_source="", clean_gripper_action="",
                                forced_gripper_action=FORCED_OPEN_RAW_GRIPPER,
                                forced_open_value_used=raw_gripper_to_env_action(FORCED_OPEN_RAW_GRIPPER),
                                post_transform_gripper_action="",
                                provenance_status="ERROR:task_not_found"))
            continue
        task_id = matches[0]

        print(f"[{i+1}/{len(candidates)}] {task}_s{sid} [{ws},{we}] task_id={task_id}")

        try:
            r = run_command_proxy_episode(
                task_suite, task_id, sid, ws, we, args.gpu_pair, args.max_steps)
        except Exception as e:
            status = infra_status_from_error(str(e))
            r = {
                "provenance_status": f"{status}:{e}",
                "runtime_sec": "",
                "measurement_version": MEASUREMENT_VERSION,
                "action_injection_version": ACTION_INJECTION_VERSION,
                "gripper_qpos_source": "",
                "clean_gripper_action": "",
                "forced_gripper_action": FORCED_OPEN_RAW_GRIPPER,
                "forced_open_value_used": raw_gripper_to_env_action(FORCED_OPEN_RAW_GRIPPER),
                "post_transform_gripper_action": "",
            }

        confidence = "silver_proxy_not_gold"
        source = "full_vis_label"
        provenance = str(r.get("provenance_status", ""))
        if provenance.startswith("MEASUREMENT_FAILED"):
            confidence = "not_label_measurement_failed"
            source = "full_vis_label_reference_only"
        elif provenance.startswith("INFRA_FAILED"):
            confidence = "not_label_infra_failed"
            source = "full_vis_label_reference_only"

        r.update(dict(task_key=task, state_id=sid, window_start=ws, window_end=we,
                      label=c.get("full_vis_label",""), label_source=source,
                      label_confidence=confidence,
                      denominator_status="not_applicable_command_proxy",
                      gpu_pair=args.gpu_pair))
        results.append(r)
        print(f"  done={r.get('task_done','?')}, qpos_delta={r.get('qpos_opening_delta','?')}, "
              f"steps={r.get('steps','?')}, runtime={r.get('runtime_sec','?')}s")

    # Write CSV
    fields = ["task_key","state_id","window_start","window_end","label",
              "label_source","label_confidence","denominator_status","gpu_pair",
              "measurement_version","action_injection_version","gripper_qpos_source",
              "clean_gripper_action","forced_gripper_action","forced_open_value_used",
              "post_transform_gripper_action",
              "forced_open_count","qpos_opening_delta","task_done","steps",
              "physical_response","task_failure","runtime_sec","provenance_status"]
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(results)
    print(f"Results: {args.output_csv}")

    # Report
    pos = [r for r in results if int(r.get("label",0) or 0) == 1]
    neg = [r for r in results if int(r.get("label",0) or 0) == 0]
    report = f"""# Command-Open Proxy Replay v0

**Candidates**: {len(candidates)} ({len(pos)} pos, {len(neg)} neg)
**Method**: Force gripper OPEN during candidate window, no image PGD.
**WARNING**: This is an upper-bound proxy, NOT a gold VIS label.

## Results

| Task | State | Window | Label | Done | QposΔ | Qpos source | Clean raw g | Forced env g | Physical | Steps | Runtime | Provenance |
|------|-------|--------|-------|------|-------|-------------|-------------|--------------|----------|-------|---------|------------|
"""
    for r in results:
        report += (
            f"| {r['task_key']} | {r['state_id']} | [{r['window_start']},{r['window_end']}] | "
            f"{r.get('label','?')} | {r.get('task_done','?')} | {r.get('qpos_opening_delta','?')} | "
            f"{r.get('gripper_qpos_source','?')} | {r.get('clean_gripper_action','?')} | "
            f"{r.get('post_transform_gripper_action','?')} | {r.get('physical_response','?')} | "
            f"{r.get('steps','?')} | {r.get('runtime_sec','?')} | {r.get('provenance_status','?')} |\n"
        )

    report += f"""
## Summary

- Positives that stay done with forced OPEN: task-positive despite gripper override
- Negatives that stay done: control negative
- Large qpos delta during window: physical bridge possible
- Task failure after forced OPEN: SUSCEPTIBLE to gripper perturbation
- qpos measurement version: `{MEASUREMENT_VERSION}`
- action injection version: `{ACTION_INJECTION_VERSION}`
- forced OPEN is injected as final env-step gripper action after normalize/invert transforms.
- `env._joint_positions` is not used for gripper qpos.
- `MEASUREMENT_FAILED` rows are not usable proxy labels.

## Claim boundary

- This is PHYSICAL/TASK SUSCEPTIBILITY screening.
- It does NOT prove VIS attack success.
- It cannot be used as gold or silver label for detector training.
"""
    with open(args.output_report, "w") as f:
        f.write(report)
    print(f"Report: {args.output_report}")


if __name__ == "__main__":
    main()
