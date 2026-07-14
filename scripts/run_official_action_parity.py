#!/usr/bin/env python3
"""Run the official OpenVLA action-path parity gate for one LIBERO suite."""

from __future__ import annotations

import argparse
import csv
import copy
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=["libero_object", "libero_spatial", "libero_goal", "libero_10"])
    ap.add_argument("--model-path", required=True, type=Path)
    ap.add_argument("--gpu", required=True, type=int)
    ap.add_argument("--upstream-root", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    return ap.parse_args()


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OPENVLA_ATTN_IMPLEMENTATION", "flash_attention_2")
sys.path.insert(0, str(args.upstream_root))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from gripper_attack.official_libero_protocol import (
    OFFICIAL_HORIZONS,
    NUM_STEPS_WAIT,
    NUM_TRIALS_PER_TASK,
    tensor_sha256,
)
from gripper_attack.official_openvla_adapter import OfficialOpenVLAActionAdapter


def load_model() -> tuple[object, object, torch.device, str]:
    from experiments.robot.openvla_utils import get_processor
    from experiments.robot.robot_utils import get_model

    cfg = SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=str(args.model_path),
        load_in_8bit=False,
        load_in_4bit=False,
    )
    model = get_model(cfg)
    processor = get_processor(cfg)
    model.eval()
    device = next(model.parameters()).device
    norm_stats = getattr(model, "norm_stats", {})
    unnorm_key = args.suite
    if unnorm_key not in norm_stats and f"{unnorm_key}_no_noops" in norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    if unnorm_key not in norm_stats:
        raise RuntimeError(f"missing official unnorm key {args.suite}: {list(norm_stats)}")
    return model, processor, device, unnorm_key


def make_env(task, state, *, render_gpu: int):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from experiments.robot.libero.libero_utils import get_libero_dummy_action

    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    env.reset()
    # LIBERO's reset path may retain or mutate the supplied state object;
    # isolate each replay so the two P3 traces receive byte-identical state.
    obs = env.set_init_state(copy.deepcopy(state))
    for _ in range(NUM_STEPS_WAIT):
        obs, _reward, _done, _info = env.step(get_libero_dummy_action("openvla"))
    return env, obs


def image_from_obs(obs):
    from experiments.robot.libero.libero_utils import get_libero_image

    return get_libero_image(obs, 224)


def compare_case(adapter, image, instruction, *, case_type, task_idx, state_id, step_idx, cases):
    official_action, official_meta = adapter.predict_action(image, instruction, capture=True)
    instrumented_action, _generation, instrumented_meta = adapter.predict_action_with_scores(image, instruction)
    official_tokens = list(official_meta.get("tokens", []))
    instrumented_tokens = list(instrumented_meta.get("captured_action_token_ids", []))
    input_ids_equal = tensor_sha256(official_meta["inputs"]["input_ids"]) == tensor_sha256(instrumented_meta["inputs"]["input_ids"])
    pixel_equal = tensor_sha256(official_meta["inputs"]["pixel_values"]) == tensor_sha256(instrumented_meta["inputs"]["pixel_values"])
    action_error = float(np.max(np.abs(np.asarray(official_action) - np.asarray(instrumented_action))))
    postprocess_error = float(np.max(np.abs(adapter.postprocess(official_action) - adapter.postprocess(instrumented_action))))
    row = {
        "case_type": case_type,
        "suite": args.suite,
        "task_idx": task_idx,
        "state_id": state_id,
        "step_idx": step_idx,
        "prompt_equal": official_meta["prompt"] == instrumented_meta["prompt"],
        "input_ids_equal": input_ids_equal,
        "pixel_tensor_equal": pixel_equal,
        "token_equal": official_tokens == instrumented_tokens,
        "official_tokens": json.dumps(official_tokens),
        "instrumented_tokens": json.dumps(instrumented_tokens),
        "action_max_abs_error": action_error,
        "postprocessed_action_max_abs_error": postprocess_error,
        "instrumented_generation_passes": instrumented_meta.get("generation_passes_per_step"),
        "instrumented_score_count": instrumented_meta.get("captured_score_count"),
        "continuous_action_pass": action_error <= 1e-6,
        "status": "PASS" if (
            official_tokens == instrumented_tokens
            and input_ids_equal
            and pixel_equal
            and action_error <= 1e-6
            and postprocess_error <= 1e-6
            and instrumented_meta.get("generation_passes_per_step") == 1
            and instrumented_meta.get("captured_score_count") == 7
        ) else "FAIL",
    }
    cases.append(row)
    return np.asarray(official_action, dtype=np.float32), np.asarray(instrumented_action, dtype=np.float32), row


def run_short_trace_parity(adapter, task, state) -> dict[str, object]:
    """Compare uninstrumented and instrumented calls on one official trace.

    The official action is executed and both pre/post-gripper actions are
    compared at every step on the same observation.
    """
    from experiments.robot.libero.libero_utils import get_libero_dummy_action

    env, obs = make_env(task, state, render_gpu=args.gpu)
    action_errors: list[float] = []
    env_action_errors: list[float] = []
    done = False
    try:
        for _ in range(20):
            image = image_from_obs(obs)
            official_action, _official_meta = adapter.predict_action(image, str(task.language), capture=True)
            instrumented_action, _generation, _instrumented_meta = adapter.predict_action_with_scores(image, str(task.language))
            official_env_action = adapter.postprocess(official_action)
            instrumented_env_action = adapter.postprocess(instrumented_action)
            action_errors.append(float(np.max(np.abs(np.asarray(official_action) - np.asarray(instrumented_action)))))
            env_action_errors.append(float(np.max(np.abs(official_env_action - instrumented_env_action))))
            # The official action is the executed action for this trace.
            obs, _reward, done, _info = env.step(official_env_action.tolist())
            if done:
                break
    finally:
        env.close()
    return {
        "official_steps": len(action_errors),
        "instrumented_steps": len(action_errors),
        "official_done": bool(done),
        "instrumented_done": bool(done),
        "action_prefix_max_abs_error": max(action_errors, default=float("inf")),
        "postprocessed_action_prefix_max_abs_error": max(env_action_errors, default=float("inf")),
        "observation_source": "official_execution_trace",
    }


def main() -> int:
    started = time.time()
    model, processor, device, unnorm_key = load_model()
    adapter = OfficialOpenVLAActionAdapter(
        model, processor, device, unnorm_key, center_crop=True,
        base_vla_name=str(args.model_path),
    )

    from libero.libero import benchmark

    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    cases: list[dict[str, object]] = []
    first_image = None

    # P2: 10 tasks x 2 states x 2 sampled observations = 40 observations/suite.
    for task_idx in range(10):
        task = task_suite.get_task(task_idx)
        states = task_suite.get_task_init_states(task_idx)
        if len(states) < 2:
            raise RuntimeError(f"not enough init states for task {task_idx}")
        env, obs = make_env(task, states[0], render_gpu=args.gpu)
        try:
            for state_id in (0, 1):
                if state_id == 1:
                    env.close()
                    env, obs = make_env(task, states[state_id], render_gpu=args.gpu)
                for step_idx in (0, 1):
                    image = image_from_obs(obs)
                    if first_image is None:
                        first_image = np.asarray(image).copy()
                    compare_case(
                        adapter,
                        image,
                        str(task.language),
                        case_type="P2_SINGLE_STEP",
                        task_idx=task_idx,
                        state_id=state_id,
                        step_idx=step_idx,
                        cases=cases,
                    )
                    from experiments.robot.libero.libero_utils import get_libero_dummy_action

                    obs, _reward, _done, _info = env.step(get_libero_dummy_action("openvla"))
        finally:
            env.close()

    # P3: two 20-step traces, official execution vs score adapter execution.
    trace_rows = []
    for task_idx in (0, 1):
        task = task_suite.get_task(task_idx)
        state = task_suite.get_task_init_states(task_idx)[0]
        trace = run_short_trace_parity(adapter, task, state)
        trace_rows.append({
            "case_type": "P3_SHORT_TRACE",
            "suite": args.suite,
            "task_idx": task_idx,
            "state_id": 0,
            **trace,
            "status": "PASS" if trace["official_steps"] == trace["instrumented_steps"] and trace["action_prefix_max_abs_error"] <= 1e-6 and trace["postprocessed_action_prefix_max_abs_error"] <= 1e-6 else "FAIL",
        })

    # P4: one deterministic image perturbation, then official re-decode parity.
    if first_image is None:
        raise RuntimeError("no parity image captured")
    adversarial = first_image.copy()
    adversarial[0, 0, 0] = (int(adversarial[0, 0, 0]) + 1) % 256
    compare_case(
        adapter,
        adversarial,
        str(task_suite.get_task(0).language),
        case_type="P4_ADVERSARIAL_REDECODE",
        task_idx=0,
        state_id=0,
        step_idx=0,
        cases=cases,
    )

    p2 = [r for r in cases if r["case_type"] == "P2_SINGLE_STEP"]
    p4 = [r for r in cases if r["case_type"] == "P4_ADVERSARIAL_REDECODE"]
    all_p2_pass = len(p2) == 40 and all(r["status"] == "PASS" for r in p2)
    all_p4_pass = len(p4) == 1 and p4[0]["status"] == "PASS"
    all_p3_pass = len(trace_rows) == 2 and all(r["status"] == "PASS" for r in trace_rows)
    summary = {
        "status": "OFFICIAL_ACTION_PARITY_PASS" if all_p2_pass and all_p3_pass and all_p4_pass else "OFFICIAL_PARITY_FAIL",
        "suite": args.suite,
        "model_path": str(args.model_path),
        "gpu": args.gpu,
        "device": str(device),
        "unnorm_key": unnorm_key,
        "num_tasks": int(task_suite.n_tasks),
        "num_trials_per_task": NUM_TRIALS_PER_TASK,
        "num_steps_wait": NUM_STEPS_WAIT,
        "official_horizon": OFFICIAL_HORIZONS[args.suite],
        "p2_cases": len(p2),
        "p2_pass": sum(r["status"] == "PASS" for r in p2),
        "p3_cases": len(trace_rows),
        "p3_pass": sum(r["status"] == "PASS" for r in trace_rows),
        "p4_cases": len(p4),
        "p4_pass": sum(r["status"] == "PASS" for r in p4),
        "elapsed_sec": round(time.time() - started, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".cases.csv").open("w", newline="", encoding="utf-8") as f:
        fields = sorted({k for row in [*cases, *trace_rows] for k in row})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([*cases, *trace_rows])
    args.output.write_text(json.dumps({"summary": summary, "trace_cases": trace_rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "OFFICIAL_ACTION_PARITY_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
