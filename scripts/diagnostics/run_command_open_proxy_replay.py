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

MODEL_PATH = "/data/aviary/models/openvla/openvla-7b-finetuned-libero-object"
UNNORM_KEY = "libero_object"
MEASUREMENT_VERSION = "v2_mujoco_gripper_qpos_primary_obs_audit_fallback_20260605"
ACTION_INJECTION_VERSION = "v1_final_env_step_action_space_after_normalize_invert_20260605"
FORCED_OPEN_RAW_GRIPPER = 0.0

TASK_CONFIGS = {
    "alphabet_soup": {"task_id": 0, "instruction": "pick up the alphabet soup and place it in the basket"},
    "cream_cheese": {"task_id": 1, "instruction": "pick up the cream cheese and place it in the basket"},
    "salad_dressing": {"task_id": 2, "instruction": "put the salad dressing in the basket"},
    "bbq_sauce": {"task_id": 3, "instruction": "pick up the bbq sauce and place it in the basket"},
    "ketchup": {"task_id": 4, "instruction": "pick up the ketchup and place it in the basket"},
    "tomato_sauce": {"task_id": 5, "instruction": "pick up the tomato sauce and place it in the basket"},
    "butter": {"task_id": 6, "instruction": "pick up the butter and place it in the basket"},
    "milk": {"task_id": 7, "instruction": "pick up the milk and place it in the basket"},
    "chocolate_pudding": {"task_id": 8, "instruction": "pick up the chocolate pudding and place it in the basket"},
    "orange_juice": {"task_id": 9, "instruction": "pick up the orange juice and place it in the basket"},
}


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


def from_pretrained_local(cls, path: str, **kwargs):
    try:
        return cls.from_pretrained(path, local_files_only=True, **kwargs)
    except TypeError:
        return cls.from_pretrained(path, **kwargs)


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


def read_mujoco_gripper_qpos(env):
    sim = getattr(env, "sim", None)
    model = getattr(sim, "model", None)
    data = getattr(sim, "data", None)
    if model is not None and data is not None and hasattr(data, "qpos"):
        joint_names = list(getattr(model, "joint_names", []) or [])
        preferred = []
        fallback = []
        for name in joint_names:
            lname = str(name).lower()
            if "gripper" in lname and "finger" in lname:
                preferred.append(name)
            elif "gripper" in lname:
                fallback.append(name)
        for name in preferred + fallback:
            try:
                jid = model.joint_name2id(name)
                adr = int(model.jnt_qposadr[jid])
                return float(data.qpos[adr]), f"mujoco_joint:{name}", "ok"
            except Exception:
                continue
    return None, "", "missing_mujoco_gripper_qpos"


def read_obs_gripper_qpos(obs):
    import numpy as np
    if isinstance(obs, dict) and "robot0_gripper_qpos" in obs:
        arr = np.asarray(obs.get("robot0_gripper_qpos"), dtype=np.float32).reshape(-1)
        if arr.size > 0:
            return float(arr[0]), "obs.robot0_gripper_qpos", "ok"
    return None, "", "missing_obs_robot0_gripper_qpos"


def read_gripper_qpos(obs, env):
    """Return gripper qpos audit fields. Never use env._joint_positions."""
    mujoco_qpos, mujoco_source, mujoco_status = read_mujoco_gripper_qpos(env)
    obs_qpos, obs_source, obs_status = read_obs_gripper_qpos(obs)

    warning = ""
    if mujoco_qpos is not None and obs_qpos is not None and abs(float(mujoco_qpos) - float(obs_qpos)) > 1e-3:
        warning = "mujoco_obs_qpos_mismatch"

    if mujoco_qpos is not None:
        used = float(mujoco_qpos)
        source = mujoco_source
        status = "ok"
    elif obs_qpos is not None:
        used = float(obs_qpos)
        source = obs_source
        status = "ok"
    else:
        used = None
        source = "unavailable"
        status = "missing_gripper_qpos"

    return {
        "used": used,
        "source": source,
        "status": status,
        "mujoco": mujoco_qpos,
        "obs": obs_qpos,
        "mujoco_status": mujoco_status,
        "obs_status": obs_status,
        "warning": warning,
        "source_priority": "mujoco_primary_obs_fallback",
    }


def infra_status_from_error(error_text: str):
    low = str(error_text).lower()
    if any(tok in low for tok in ["xid", "out of memory", "oom", "cuda illegal", "cublas"]):
        return "INFRA_FAILED"
    return "ERROR"


def load_libero_task_suite():
    from libero.libero import benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    return benchmark_dict["libero_object"]()


def run_command_proxy_episode(task_suite, task_key: str, task_id: int, state_id: int,
                               window_start: int, window_end: int, gpu_pair: str,
                               max_steps: int):
    """Run rollout with forced OPEN gripper during the window."""
    import numpy as np
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from PIL import Image

    gpu_ids = parse_gpu_ids(gpu_pair)
    device = f"cuda:{gpu_ids[0]}"

    # Load model
    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if len(gpu_ids) >= 2:
        model_kwargs.update(
            device_map="auto",
            max_memory={gpu_ids[0]: "10500MiB", gpu_ids[1]: "10500MiB", "cpu": "64GiB"},
        )
    model = from_pretrained_local(AutoModelForVision2Seq, MODEL_PATH, **model_kwargs)
    if len(gpu_ids) < 2:
        model = model.to(device)
    model.eval()
    processor = from_pretrained_local(AutoProcessor, MODEL_PATH, trust_remote_code=True)

    # Setup environment
    task = task_suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    initial_states = task_suite.get_task_init_states(task_id)
    if state_id < 0 or state_id >= len(initial_states):
        raise RuntimeError(f"invalid_state_id:{state_id}:available=0..{len(initial_states)-1}")
    env_args = {
        "bddl_file_name": bddl,
        "camera_heights": 224,
        "camera_widths": 224,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "use_camera_obs": True,
        "camera_names": ["agentview"],
        "control_freq": 20,
        "render_gpu_device_id": gpu_ids[0],
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    obs = env.reset()
    env.sim.data.qvel[:] = 0
    env.sim.forward()
    env.set_init_state(initial_states[state_id])
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))

    instruction = TASK_CONFIGS[task_key]["instruction"]
    action_dim = 7
    stats = model.get_action_stats(UNNORM_KEY)
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
    qpos_mujoco = ""
    qpos_obs = ""
    qpos_used = ""
    qpos_warning = ""
    measurement_failed = False
    measurement_failure_reason = ""
    arm_l2_sum = 0.0
    clean_gripper_action = ""
    forced_gripper_action = FORCED_OPEN_RAW_GRIPPER
    forced_open_value_used = raw_gripper_to_env_action(FORCED_OPEN_RAW_GRIPPER)
    post_transform_gripper_action = ""
    post_transform_gripper_actions = []

    while step < max_steps and not done:
        qpos_pre_audit = read_gripper_qpos(obs, env)
        qpos_pre = qpos_pre_audit["used"]
        if qpos_pre_audit["mujoco"] is not None:
            qpos_mujoco = float(qpos_pre_audit["mujoco"])
        if qpos_pre_audit["obs"] is not None:
            qpos_obs = float(qpos_pre_audit["obs"])
        if qpos_pre is not None:
            qpos_used = float(qpos_pre)
        if qpos_pre_audit["warning"]:
            qpos_warning = qpos_pre_audit["warning"]
        if qpos_pre_audit["status"] != "ok":
            measurement_failed = True
            measurement_failure_reason = qpos_pre_audit["status"]
        elif not qpos_source:
            qpos_source = qpos_pre_audit["source"]

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
        qpos_post_audit = read_gripper_qpos(obs, env)
        qpos_post = qpos_post_audit["used"]
        if qpos_post_audit["mujoco"] is not None:
            qpos_mujoco = float(qpos_post_audit["mujoco"])
        if qpos_post_audit["obs"] is not None:
            qpos_obs = float(qpos_post_audit["obs"])
        if qpos_post is not None:
            qpos_used = float(qpos_post)
        if qpos_post_audit["warning"]:
            qpos_warning = qpos_post_audit["warning"]
        if qpos_post_audit["status"] != "ok":
            measurement_failed = True
            measurement_failure_reason = qpos_post_audit["status"]
        else:
            if not qpos_source:
                qpos_source = qpos_post_audit["source"]
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
        "gripper_qpos_mujoco": qpos_mujoco,
        "gripper_qpos_obs": qpos_obs,
        "gripper_qpos_used": qpos_used,
        "gripper_qpos_source_priority": "mujoco_primary_obs_fallback",
        "gripper_qpos_warning": qpos_warning,
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
    task_suite = load_libero_task_suite()

    results = []
    for i, c in enumerate(candidates):
        task = c["task_key"]
        sid = int(c["state_id"])
        ws = int(c["parent_window_start"])
        we = int(c["parent_window_end"])

        if task not in TASK_CONFIGS:
            raise SystemExit(f"ERROR:no_task_mapping:{task}")
        task_id = int(TASK_CONFIGS[task]["task_id"])

        print(f"[{i+1}/{len(candidates)}] {task}_s{sid} [{ws},{we}] task_id={task_id}")

        try:
            r = run_command_proxy_episode(
                task_suite, task, task_id, sid, ws, we, args.gpu_pair, args.max_steps)
        except Exception as e:
            status = infra_status_from_error(str(e))
            r = {
                "provenance_status": f"{status}:{e}",
                "runtime_sec": "",
                "measurement_version": MEASUREMENT_VERSION,
                "action_injection_version": ACTION_INJECTION_VERSION,
                "gripper_qpos_source": "",
                "gripper_qpos_mujoco": "",
                "gripper_qpos_obs": "",
                "gripper_qpos_used": "",
                "gripper_qpos_source_priority": "mujoco_primary_obs_fallback",
                "gripper_qpos_warning": "",
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
              "gripper_qpos_mujoco","gripper_qpos_obs","gripper_qpos_used",
              "gripper_qpos_source_priority","gripper_qpos_warning",
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

| Task | State | Window | Label | Done | QposΔ | Qpos source | Mujoco qpos | Obs qpos | Used qpos | Warning | Clean raw g | Forced env g | Physical | Steps | Runtime | Provenance |
|------|-------|--------|-------|------|-------|-------------|-------------|----------|-----------|---------|-------------|--------------|----------|-------|---------|------------|
"""
    for r in results:
        report += (
            f"| {r['task_key']} | {r['state_id']} | [{r['window_start']},{r['window_end']}] | "
            f"{r.get('label','?')} | {r.get('task_done','?')} | {r.get('qpos_opening_delta','?')} | "
            f"{r.get('gripper_qpos_source','?')} | {r.get('gripper_qpos_mujoco','?')} | "
            f"{r.get('gripper_qpos_obs','?')} | {r.get('gripper_qpos_used','?')} | "
            f"{r.get('gripper_qpos_warning','?')} | {r.get('clean_gripper_action','?')} | "
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
- MuJoCo gripper joint qpos is primary; `obs["robot0_gripper_qpos"]` is fallback/audit comparison only.
- `env._joint_positions` is not used for gripper qpos.
- MuJoCo/obs qpos mismatch is recorded in `gripper_qpos_warning`, never silently zeroed.
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
