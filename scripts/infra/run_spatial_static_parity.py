#!/usr/bin/env python3
"""MIG2A+B Spatial Static Parity — Lane O vs Lane M exact mirror + cross-host support."""
import csv, json, hashlib, os, sys, time, struct, argparse
import numpy as np
from pathlib import Path
from PIL import Image


# ===== PRECISE TENSOR HASHES =====

def hash_int_tensor(t):
    """Hash: dtype name + shape + raw int64 bytes."""
    import torch
    arr = t.cpu().numpy().astype(np.int64)
    h = hashlib.sha256()
    h.update(str(t.dtype).encode())
    h.update(np.array(t.shape, dtype=np.int64).tobytes())
    h.update(arr.tobytes())
    return h.hexdigest()

def hash_float32_tensor(t):
    """Hash: dtype name + shape + raw float32 bytes."""
    import torch
    arr = t.float().cpu().numpy().astype(np.float32)
    h = hashlib.sha256()
    h.update(b"float32")
    h.update(np.array(t.shape, dtype=np.int64).tobytes())
    h.update(arr.tobytes())
    return h.hexdigest()

def hash_bfloat16_tensor(t):
    """Hash: dtype + shape + raw bytes (via storage)."""
    import torch
    h = hashlib.sha256()
    h.update(b"bfloat16")
    h.update(np.array(t.shape, dtype=np.int64).tobytes())
    h.update(t.cpu().contiguous().view(torch.int16).numpy().tobytes())
    return h.hexdigest()

def tensor_hash(t):
    """Auto-dispatcher."""
    import torch
    dt = str(t.dtype)
    if dt in ("torch.int64", "torch.int32", "torch.long"):
        return hash_int_tensor(t)
    elif dt == "torch.float32":
        return hash_float32_tensor(t)
    elif dt == "torch.bfloat16":
        return hash_bfloat16_tensor(t)
    else:
        return hash_float32_tensor(t.float())


# ===== LANE M: EXACT MIRROR OF predict_action =====

def lane_m_mirror(model, input_ids, pixel_values, unnorm_key, attention_mask=None):
    """Exact mirror of predict_action() without calling predict_action.
    From source: modeling_prismatic.py, SHA c10a6d1fbb414152bb3fda9d8acd3d1a9df7b5b6f94b2a8a69c73c9adcb1b8b2
    """
    import torch

    # Step 1: Append token 29871 if not already present (matches training-time inputs)
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
        )
        if attention_mask is not None:
            attention_mask = torch.cat(
                (attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)), dim=1
            )

    # Step 2: Generate (matching predict_action kwargs exactly)
    gen_kwargs = {
        "max_new_tokens": model.get_action_dim(unnorm_key),
        "do_sample": False,
        "pad_token_id": model.pad_token_id,
    }
    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask

    generated_ids = model.generate(input_ids, pixel_values=pixel_values, **gen_kwargs)

    # Step 3: Extract action tokens and decode
    action_dim = model.get_action_dim(unnorm_key)
    predicted_action_token_ids = generated_ids[0, -action_dim:].cpu().numpy()
    discretized_actions = model.vocab_size - predicted_action_token_ids
    discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=model.bin_centers.shape[0] - 1)
    normalized_actions = model.bin_centers[discretized_actions]

    # Step 4: Unnormalize (EXACT formula from source)
    action_norm_stats = model.get_action_stats(unnorm_key)
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high = np.array(action_norm_stats["q99"])
    action_low = np.array(action_norm_stats["q01"])
    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )

    return actions, generated_ids[0], discretized_actions, normalized_actions


# ===== MAIN =====

def run_parity_matrix(model_path, frames_dir, output_dir, gpu=6, profile="A800-O"):
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq

    os.makedirs(output_dir, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print(f"MODEL: {model_path}")
    print(f"PROFILE: {profile}")
    print(f"GPU: {gpu}")

    # Load once
    proc = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="eager",
        local_files_only=True, trust_remote_code=True, low_cpu_mem_usage=True,
        device_map="cuda:0",
    )
    tok = proc.tokenizer

    # Model attributes
    model_attrs = {
        "vocab_size": model.vocab_size,
        "bin_centers_shape": list(model.bin_centers.shape),
        "bin_centers_dtype": str(model.bin_centers.dtype),
        "action_dim": model.get_action_dim("libero_spatial"),
        "action_stats": model.get_action_stats("libero_spatial"),
        "pad_token_id": model.pad_token_id,
        "predict_action_source_sha": "c10a6d1fbb414152bb3fda9d8acd3d1a9df7b5b6f94b2a8a69c73c9adcb1b8b2",
    }

    # Load frame manifest
    with open(f"{frames_dir}/frame_manifest.json") as f:
        manifest = json.load(f)

    output_csv = f"{output_dir}/parity_matrix_{profile.replace('-','_')}_{gpu}.csv"
    sum_rows = []

    with open(output_csv, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow([
            "episode", "task_name", "step", "phase", "frame_file",
            "raw_file_sha256", "decoded_rgb_sha256",
            "bfloat16_pixel_sha256", "int64_input_ids_sha256",
            "token_29871_appended", "prompt_token_ids_final",
            "lane", "run",
            "generated_token_ids", "normalized_action", "final_action",
            "gripper_raw", "gripper_env", "gripper_class",
            "inference_time_s", "token_exact_match", "action_exact_match",
            "max_abs_diff",
        ])

        lane_o_results = {}
        lane_m_results = {}

        for ep_entry in manifest:
            for frame_entry in ep_entry["frames"]:
                ep_name = ep_entry["episode"]
                task_name = ep_entry["task_name"]
                step = frame_entry["step"]
                phase = frame_entry["phase"]
                fname = frame_entry["file"]
                fpath = f"{frames_dir}/{fname}"

                # Frame SHA verification
                with open(fpath, "rb") as rb:
                    file_bytes = rb.read()
                file_sha = hashlib.sha256(file_bytes).hexdigest()
                img = Image.open(fpath).convert("RGB")
                rgb_arr = np.array(img)
                rgb_sha = hashlib.sha256(rgb_arr.tobytes()).hexdigest()

                prompt = f"In: What action should the robot take to {task_name.lower()}?\nOut:"
                inputs = proc(prompt, img, return_tensors="pt").to("cuda:0", dtype=torch.bfloat16)

                bf16_pixel_sha = hash_bfloat16_tensor(inputs.pixel_values)
                i64_ids_sha = hash_int_tensor(inputs.input_ids)

                # Check if token 29871 needs appending
                needs_29871 = (inputs.input_ids[0, -1].item() != 29871)

                # ===== LANE O: predict_action =====
                o_actions = []
                for run in range(3):
                    t0 = time.time()
                    result = model.predict_action(
                        input_ids=inputs.input_ids.clone(),
                        pixel_values=inputs.pixel_values.clone(),
                        unnorm_key="libero_spatial",
                        do_sample=False,
                    )
                    dt = time.time() - t0
                    if isinstance(result, tuple):
                        vals = np.array(result[0]).flatten()
                    else:
                        vals = np.array(result).flatten()
                    o_actions.append(vals)
                    rg = vals[6]; ng = (rg * 2) - 1; ig = - (1.0 if ng >= 0 else -1.0)
                    writer.writerow([
                        ep_name, task_name, step, phase, fname,
                        file_sha, rgb_sha, bf16_pixel_sha, i64_ids_sha,
                        needs_29871, "internal",
                        "Lane_O", run + 1,
                        "internal", "internal", " ".join(f"{x:.8f}" for x in vals.tolist()),
                        f"{rg:.8f}", f"{ig:.8f}", "OPEN" if ig < 0 else "CLOSE",
                        f"{dt:.4f}", "see_below", "see_below", "see_below",
                    ])

                o_exact = all(np.array_equal(o_actions[0], a) for a in o_actions[1:])
                o_key = f"{ep_name}_step{step}"
                lane_o_results[o_key] = {"deterministic": o_exact, "action": o_actions[0]}

                # ===== LANE M: exact mirror =====
                m_actions = []
                m_tokens = []
                m_normalized = []
                for run in range(3):
                    t0 = time.time()
                    vals_m, gen_ids, disc, norm = lane_m_mirror(
                        model,
                        inputs.input_ids.clone(),
                        inputs.pixel_values.clone(),
                        "libero_spatial",
                        attention_mask=inputs.attention_mask.clone() if hasattr(inputs, "attention_mask") else None,
                    )
                    dt = time.time() - t0
                    m_actions.append(vals_m)
                    m_tokens.append(gen_ids)
                    m_normalized.append(norm)
                    rg = vals_m[6]; ng = (rg * 2) - 1; ig = - (1.0 if ng >= 0 else -1.0)

                    # Generated tokens: last action_dim tokens
                    action_dim = model.get_action_dim("libero_spatial")
                    gen_tokens = gen_ids[-action_dim:].cpu().numpy().tolist()
                    norm_vals = norm

                    writer.writerow([
                        ep_name, task_name, step, phase, fname,
                        file_sha, rgb_sha, bf16_pixel_sha, i64_ids_sha,
                        needs_29871, "with_29871",
                        "Lane_M", run + 1,
                        " ".join(str(x) for x in gen_tokens),
                        " ".join(f"{x:.8f}" for x in norm_vals.tolist()),
                        " ".join(f"{x:.8f}" for x in vals_m.tolist()),
                        f"{rg:.8f}", f"{ig:.8f}", "OPEN" if ig < 0 else "CLOSE",
                        f"{dt:.4f}", "see_below", "see_below", "see_below",
                    ])

                m_exact = all(np.array_equal(m_actions[0], a) for a in m_actions[1:])
                m_key = f"{ep_name}_step{step}"
                lane_m_results[m_key] = {"deterministic": m_exact, "action": m_actions[0]}

                # Lane O vs Lane M comparison
                o_m_match = bool(np.array_equal(o_actions[0], m_actions[0]))
                max_diff = float(np.abs(o_actions[0] - m_actions[0]).max())

                sum_rows.append({
                    "key": o_key,
                    "O_det": o_exact, "M_det": m_exact,
                    "O_M_exact": o_m_match, "O_M_max_abs_diff": max_diff,
                })
                print(f"{ep_name}/step{step}: O_det={o_exact} M_det={m_exact} O_M_exact={o_m_match} max_diff={max_diff:.2e}")

    # Summary
    o_det_count = sum(1 for r in sum_rows if r["O_det"])
    m_det_count = sum(1 for r in sum_rows if r["M_det"])
    om_match_count = sum(1 for r in sum_rows if r["O_M_exact"])
    total = len(sum_rows)
    print(f"\n=== SUMMARY ({profile}) ===")
    print(f"Lane O deterministic: {o_det_count}/{total}")
    print(f"Lane M deterministic: {m_det_count}/{total}")
    print(f"Lane O == Lane M exact: {om_match_count}/{total}")

    if om_match_count != total:
        print("WARNING: Lane O != Lane M on some frames!")
        for r in sum_rows:
            if not r["O_M_exact"]:
                print(f"  {r['key']}: max_diff={r['O_M_max_abs_diff']:.2e}")

    # Device gate
    devices = sorted(set(str(p.device) for p in model.parameters()))
    print(f"Device gate: {'PASS' if devices == ['cuda:0'] else 'FAIL'}")

    return {
        "O_deterministic": o_det_count, "M_deterministic": m_det_count,
        "O_M_exact_match": om_match_count, "total_frames": total,
        "model_attrs": model_attrs, "rows": sum_rows,
        "device_gate_pass": devices == ["cuda:0"],
        "devices": devices,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--profile", default="A800-O")
    args = parser.parse_args()
    run_parity_matrix(args.model_path, args.frames_dir, args.output_dir, args.gpu, args.profile)
