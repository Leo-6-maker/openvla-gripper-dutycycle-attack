#!/usr/bin/env python3
"""A800 Spatial Frozen-Frame Matrix — MIG2A Static Parity.
Lane O: predict_action (official). Lane G: generate + manual decode (diagnostic).
5 tasks x 2 frames x 2 lanes x 3 repeats = 60 inference runs."""

import csv, json, hashlib, os, time, sys, argparse
import numpy as np
from pathlib import Path
from PIL import Image

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def tensor_hash_bfloat16(t):
    import torch
    return hashlib.sha256(t.float().cpu().numpy().tobytes()).hexdigest()


def run_matrix(model_path, frames_dir, output_csv, gpu_index=6):
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    model_name = Path(model_path).name

    print(f"MODEL: {model_path}")
    print(f"FRAMES: {frames_dir}")
    print(f"GPU: {gpu_index}")

    # Load once
    proc = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="eager",
        local_files_only=True, trust_remote_code=True, low_cpu_mem_usage=True,
        device_map="cuda:0",
    )
    tok = proc.tokenizer

    # Load action stats
    with open(f"{model_path}/dataset_statistics.json") as f:
        stats = json.load(f)
    action_stats = stats["libero_spatial"]["action"]
    q01 = np.array(action_stats["q01"])
    q99 = np.array(action_stats["q99"])

    # Load frame manifest
    with open(f"{frames_dir}/frame_manifest.json") as f:
        manifest = json.load(f)

    with open(output_csv, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow([
            "episode", "task_name", "step", "phase", "frame_file",
            "raw_frame_sha256", "pixel_values_sha256", "prompt_ids_sha256",
            "prompt_token_ids", "prompt_token_count",
            "lane", "run",
            "generated_token_ids", "action_raw", "action_unnormed",
            "gripper_raw", "gripper_env", "gripper_class",
            "inference_time_s", "deterministic", "lane_o_g_match",
            "vram_peak_gib",
        ])

        lane_o_results = {}
        lane_g_results = {}

        for ep_entry in manifest:
            ep_name = ep_entry["episode"]
            task_name = ep_entry["task_name"]
            prompt = f"In: What action should the robot take to {task_name.lower()}?\nOut:"

            for frame_entry in ep_entry["frames"]:
                step = frame_entry["step"]
                phase = frame_entry["phase"]
                frame_file = frame_entry["file"]
                frame_path = f"{frames_dir}/{frame_file}"

                # Load and hash raw frame
                img = Image.open(frame_path).convert("RGB")
                raw_hash = hashlib.sha256(np.array(img).tobytes()).hexdigest()

                # === LANE O ===
                inputs = proc(prompt, img, return_tensors="pt").to("cuda:0", dtype=torch.bfloat16)
                pixel_sha = tensor_hash_bfloat16(inputs.pixel_values)
                prompt_sha = tensor_hash_bfloat16(inputs.input_ids)
                prompt_ids = inputs.input_ids[0].tolist()

                o_actions = []
                for run in range(3):
                    t0 = time.time()
                    result = model.predict_action(
                        input_ids=inputs.input_ids,
                        pixel_values=inputs.pixel_values,
                        unnorm_key="libero_spatial",
                        do_sample=False,
                    )
                    dt = time.time() - t0
                    vals = np.array(result).flatten()
                    o_actions.append(vals)
                    writer.writerow([
                        ep_name, task_name, step, phase, frame_file,
                        raw_hash, pixel_sha, prompt_sha,
                        " ".join(str(x) for x in prompt_ids), len(prompt_ids),
                        "Lane_O", run + 1,
                        "N/A_predict_action_internal",
                        " ".join(f"{x:.8f}" for x in vals.tolist()),
                        "ALREADY_UNNORMED",
                        vals[6], "see_gripper_class", "see_gripper_class",
                        f"{dt:.4f}", "see_determinism",
                        "see_lane_comparison",
                        f"{torch.cuda.max_memory_allocated() / 1024**3:.2f}",
                    ])

                o_key = f"{ep_name}_step{step}"
                o_match = all(np.allclose(o_actions[0], a) for a in o_actions[1:])
                lane_o_results[o_key] = {
                    "deterministic": o_match,
                    "action": o_actions[0],
                    "prompt_ids": prompt_ids,
                    "pixel_sha": pixel_sha,
                    "prompt_sha": prompt_sha,
                }

                # === LANE G ===
                g_actions = []
                g_tokens_all = []
                for run in range(3):
                    t0 = time.time()
                    outputs = model.generate(
                        input_ids=inputs.input_ids,
                        pixel_values=inputs.pixel_values,
                        max_new_tokens=7,
                        do_sample=False,
                        pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id,
                        attention_mask=inputs.attention_mask,
                    )
                    new_tokens = outputs[0, inputs.input_ids.shape[1]:]
                    action_result = model.predict_action(new_tokens.unsqueeze(0))
                    vals = np.array(action_result).flatten()
                    dt = time.time() - t0
                    g_actions.append(vals)
                    g_tokens_all.append(new_tokens.tolist())
                    # Gripper
                    rg = vals[6]
                    ng = (rg * 2) - 1
                    bg = 1.0 if ng >= 0 else -1.0
                    ig = -bg
                    writer.writerow([
                        ep_name, task_name, step, phase, frame_file,
                        raw_hash, pixel_sha, prompt_sha,
                        " ".join(str(x) for x in prompt_ids), len(prompt_ids),
                        "Lane_G", run + 1,
                        " ".join(str(x) for x in new_tokens.tolist()),
                        " ".join(f"{x:.8f}" for x in vals.tolist()),
                        "ALREADY_UNNORMED",
                        f"{rg:.8f}", f"{ig:.8f}",
                        "OPEN" if ig < 0 else "CLOSE",
                        f"{dt:.4f}", "see_determinism",
                        "see_lane_comparison",
                        f"{torch.cuda.max_memory_allocated() / 1024**3:.2f}",
                    ])

                g_key = f"{ep_name}_step{step}"
                g_match = all(np.allclose(g_actions[0], a) for a in g_actions[1:])
                lane_g_results[g_key] = {
                    "deterministic": g_match,
                    "action": g_actions[0],
                    "tokens": g_tokens_all[0],
                }

                # Lane O vs G comparison
                o_g_match = bool(np.allclose(o_actions[0], g_actions[0]))
                print(f"{ep_name}/step{step}: O_det={o_match} G_det={g_match} O_G_match={o_g_match}")

    # Summary
    o_det_count = sum(1 for v in lane_o_results.values() if v["deterministic"])
    g_det_count = sum(1 for v in lane_g_results.values() if v["deterministic"])
    total = len(lane_o_results)
    print(f"\nLane O deterministic: {o_det_count}/{total}")
    print(f"Lane G deterministic: {g_det_count}/{total}")

    # Device gate
    devices = sorted(set(str(p.device) for p in model.parameters()))
    print(f"Device gate: {'PASS' if devices == ['cuda:0'] else 'FAIL'} ({devices})")

    return lane_o_results, lane_g_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--gpu", type=int, default=6)
    args = parser.parse_args()
    run_matrix(args.model_path, args.frames_dir, args.output_csv, args.gpu)
