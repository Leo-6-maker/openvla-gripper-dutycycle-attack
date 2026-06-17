#!/usr/bin/env python3
"""H0-P: Generate 10 canonical clean packages for selected L3 VIS frames.

GPU(1,5) clean-only inference. No PGD, no perturbation, no attack.
Produces: raw_frame.npy, processor_inputs_attack.pt, clean_generation.json,
          clean_action.npy, input_manifest.csv, artifact_hash_manifest.csv
"""

import csv, hashlib, json, os, sys, time
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

os.environ.setdefault("OPENVLA_ATTN_IMPLEMENTATION", "eager")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1,5")

# Frozen constants
MODEL_PATH = "/data/aviary/models/openvla/openvla-7b-finetuned-libero-object"
UNNORM_KEY = "libero_object"
FRAME_DIR = "/data/liuyu/outputs/l12_frame_handoff_v2_r1"

TASK_IDX = {
    "alphabet_soup": 0, "cream_cheese": 1, "salad_dressing": 2, "bbq_sauce": 3,
    "ketchup": 4, "tomato_sauce": 5, "butter": 6, "milk": 7,
    "chocolate_pudding": 8, "orange_juice": 9,
}

SELECTED_FRAMES = [
    {"parent_id": "butter_s11", "task": "butter", "state_id": 11, "step": 58,
     "role": "teacher_ws", "inside_window": True, "primary": True},
    {"parent_id": "butter_s11", "task": "butter", "state_id": 11, "step": 60,
     "role": "teacher_anchor+d5_emit", "inside_window": True, "primary": True},
    {"parent_id": "butter_s11", "task": "butter", "state_id": 11, "step": 68,
     "role": "teacher_we", "inside_window": False, "primary": False, "diagnostic": True},
    {"parent_id": "tomato_sauce_s23", "task": "tomato_sauce", "state_id": 23, "step": 69,
     "role": "d5_emit", "inside_window": False, "primary": False, "diagnostic": True},
    {"parent_id": "tomato_sauce_s23", "task": "tomato_sauce", "state_id": 23, "step": 139,
     "role": "teacher_ws", "inside_window": True, "primary": True},
    {"parent_id": "tomato_sauce_s23", "task": "tomato_sauce", "state_id": 23, "step": 141,
     "role": "teacher_anchor", "inside_window": True, "primary": True},
    {"parent_id": "salad_dressing_s11", "task": "salad_dressing", "state_id": 11, "step": 57,
     "role": "teacher_ws", "inside_window": True, "primary": True},
    {"parent_id": "salad_dressing_s11", "task": "salad_dressing", "state_id": 11, "step": 59,
     "role": "teacher_anchor", "inside_window": True, "primary": True},
    {"parent_id": "salad_dressing_s11", "task": "salad_dressing", "state_id": 11, "step": 67,
     "role": "teacher_we", "inside_window": False, "primary": False, "diagnostic": True},
    {"parent_id": "salad_dressing_s11", "task": "salad_dressing", "state_id": 11, "step": 128,
     "role": "d5_emit", "inside_window": False, "primary": False, "diagnostic": True},
]


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    if not os.path.isfile(path): return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


def tensor_sha256(t):
    return hashlib.sha256(t.detach().cpu().float().numpy().tobytes()).hexdigest()


def load_model():
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoModelCls
    except Exception:
        from transformers import AutoModelForVision2Seq as AutoModelCls

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    model = AutoModelCls.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map="auto", max_memory={0: "10000MiB", 1: "10000MiB", "cpu": "128GiB"},
        attn_implementation="eager",
    )
    device = "cuda:0"
    if hasattr(model, "hf_device_map"):
        for v in model.hf_device_map.values():
            if isinstance(v, (int, str)) and str(v).startswith("cuda"):
                device = str(v); break
    return model, processor, device


def model_fingerprint(model):
    cfg = getattr(model, "config", None)
    return {
        "model_type": str(getattr(cfg, "model_type", "")),
        "vocab_size": int(getattr(getattr(cfg, "text_config", cfg), "vocab_size", 0) or 0),
        "pad_to_multiple_of": int(getattr(cfg, "pad_to_multiple_of", 0) or 0),
        "action_bins": int(getattr(getattr(model, "bin_centers", []), "shape", [0])[0] or 0),
        "norm_stats_keys": sorted(list(getattr(model, "norm_stats", {}).keys())),
    }


def get_instruction(task: str) -> str:
    from libero.libero import benchmark
    bm = benchmark.get_benchmark_dict()
    suite = bm["libero_object"]()
    task_obj = suite.get_task(TASK_IDX[task])
    return task_obj.language


def preprocess_canonical(raw_image: np.ndarray, processor, instruction: str, device: str, model_dtype: torch.dtype):
    from gripper_attack.attack_adapter import prepare_openvla_image_for_attack
    from v4_run_eval_openvla import prompt

    image = prepare_openvla_image_for_attack(
        raw_image,
        libero_official_preprocess=False,
        libero_preprocess_backend="official_pil_lanczos",
        center_crop=True,
        resize_size=224,
    )
    inputs = processor(prompt(instruction), image, return_tensors="pt")
    inputs.pop("attention_mask", None)
    input_ids = inputs["input_ids"].to(device)
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat([input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)], dim=1)
    pixel_values = inputs["pixel_values"].to(device=device, dtype=model_dtype)
    return {"input_ids": input_ids, "pixel_values": pixel_values}


def clean_official_decode(model, adv_inputs, action_dim: int, tolerance: float = 1e-6):
    from gripper_attack.v3_generation_parity import extract_exact_new_tokens
    from gripper_attack.m3_controls import tensor_sha256 as tsha
    from scripts.stageb.diagnose_m3_true_pgd_fixed_frame import validate_processed_argmax_matches_emitted
    from gripper_attack.execution_target import target_token_cw_loss_and_stats
    from gripper_attack.v3_generation_parity import generation_score_audit_from_row

    input_ids = adv_inputs["input_ids"]
    pixel_values = adv_inputs["pixel_values"]
    with torch.inference_mode():
        gen = model.generate(
            input_ids=input_ids, pixel_values=pixel_values,
            max_new_tokens=int(action_dim), do_sample=False,
            return_dict_in_generate=True, output_scores=True,
        )
    tokens = extract_exact_new_tokens(gen.sequences, prompt_len=int(input_ids.shape[1]), expected_new_tokens=int(action_dim))
    score_row = gen.scores[-1][0].detach().float().cpu()
    invariant = validate_processed_argmax_matches_emitted(score_row, int(tokens[-1]), tolerance=tolerance)

    vocab_eff = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
    bin_centers = model.bin_centers
    action_stats = model.get_action_stats(UNNORM_KEY)

    audit = generation_score_audit_from_row(
        score_row, emitted_token=int(tokens[-1]), vocab_eff=vocab_eff,
        n_bins=int(bin_centers.shape[0]), bin_centers=bin_centers,
        action_stats=action_stats, surrogate_top_token=int(tokens[-1]),
    )

    return {
        "tokens": [int(t) for t in tokens],
        "arm_prefix": [int(t) for t in tokens[:6]],
        "gripper_token": int(tokens[-1]),
        "score_row_sha256": tsha(score_row),
        "score_invariant": invariant,
        "score_audit": audit,
    }


def process_one_frame(spec: Dict, model, processor, device: str, model_dtype, out_base: Path):
    task = spec["task"]
    sid = spec["state_id"]
    step = spec["step"]
    pid = spec["parent_id"]

    parent_dir = f"{task}_s{sid}_frame"
    npy_path = os.path.join(FRAME_DIR, parent_dir, f"frame_{step:04d}.npy")

    if not os.path.isfile(npy_path):
        print(f"  MISSING: {npy_path}")
        return None

    raw_image = np.load(npy_path)
    raw_sha = sha256_file(npy_path)
    instruction = get_instruction(task)

    if not instruction:
        print(f"  EMPTY_INSTRUCTION: {pid} step{step}")
        return None

    pkg_dir = out_base / f"{pid}_step{step:04d}"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    # Save raw frame
    np.save(pkg_dir / "raw_frame.npy", raw_image)
    with open(pkg_dir / "raw_frame.sha256", "w") as f:
        f.write(raw_sha)

    # Canonical preprocessing
    proc_inputs = preprocess_canonical(raw_image, processor, instruction, device, model_dtype)
    torch.save({k: v.detach().cpu() for k, v in proc_inputs.items()}, pkg_dir / "processor_inputs_attack.pt")
    pt_sha = tensor_sha256(proc_inputs["pixel_values"])
    prompt_sha = tensor_sha256(proc_inputs["input_ids"])
    with open(pkg_dir / "processor_tensor.sha256", "w") as f:
        f.write(f"{pt_sha}\n{prompt_sha}\n")

    # Clean official decode
    action_dim = int(model.get_action_dim(UNNORM_KEY))
    clean_result = clean_official_decode(model, proc_inputs, action_dim)

    clean_action = np.array(clean_result["tokens"], dtype=np.float32)
    np.save(pkg_dir / "clean_action.npy", clean_action)

    is_primary = spec.get("primary", False)
    inside_window = spec.get("inside_window", False)
    clean_grp = int(clean_result["gripper_token"])

    # Classify eligibility
    if is_primary and clean_grp != 31872:
        eligibility = "CLEAN_CONTEXT_INELIGIBLE"
    elif not is_primary and clean_grp == 31872:
        eligibility = "CLEAN_ELIGIBLE"
    elif not is_primary and clean_grp == 31744:
        eligibility = "CLEAN_ALREADY_TARGET"
    elif not is_primary:
        eligibility = "CLEAN_NOT_CLOSE"
    else:
        eligibility = "CLEAN_ELIGIBLE"

    clean_gen = {
        "parent_id": pid, "task": task, "state_id": sid,
        "frame_step": step, "frame_role": spec["role"],
        "inside_teacher_window": inside_window, "is_primary": is_primary,
        "instruction": instruction,
        "prompt": f"<|vision_start|><image><|vision_end|>In the LIBERO Object environment, {instruction}",
        "clean_action": clean_action.tolist(),
        "exact_clean_7_tokens": clean_result["tokens"],
        "clean_arm_prefix": clean_result["arm_prefix"],
        "clean_gripper_token": clean_grp,
        "official_score_invariant": {
            "tie_aware_pass": clean_result["score_invariant"]["tie_aware_pass"],
            "argmax_token": clean_result["score_invariant"]["argmax_token"],
        },
        "prompt_token_sha256": prompt_sha,
        "raw_frame_sha256": raw_sha,
        "canonical_processor_tensor_sha256": pt_sha,
        "model_fingerprint": model_fingerprint(model),
        "model_path": MODEL_PATH,
        "unnorm_key": UNNORM_KEY,
        "preprocessing_contract": {
            "libero_official_preprocess": False,
            "libero_preprocess_backend": "official_pil_lanczos",
            "center_crop": True,
            "resize_size": 224,
        },
        "source_commit": "",
        "source_timing_trace_sha": "",
        "source_action_env_identity": "ACTION_ENV_RAWFRAME_EXACT_BOUND_WITH_OBS_WAIVER",
        "clean_eligibility": eligibility,
    }

    with open(pkg_dir / "clean_generation.json", "w") as f:
        json.dump(clean_gen, f, indent=2)

    # Input manifest
    with open(pkg_dir / "input_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "parent_id", "task", "state_id", "frame_step", "frame_role",
            "raw_frame_sha256", "processor_tensor_sha256", "prompt_token_sha256",
            "clean_gripper_token", "clean_eligibility", "instruction",
        ])
        w.writeheader()
        w.writerow({
            "parent_id": pid, "task": task, "state_id": str(sid),
            "frame_step": str(step), "frame_role": spec["role"],
            "raw_frame_sha256": raw_sha, "processor_tensor_sha256": pt_sha,
            "prompt_token_sha256": prompt_sha,
            "clean_gripper_token": str(clean_grp),
            "clean_eligibility": eligibility, "instruction": instruction,
        })

    # Artifact hash manifest
    artifacts = {}
    for fname in sorted(os.listdir(pkg_dir)):
        fpath = pkg_dir / fname
        if os.path.isfile(fpath):
            artifacts[fname] = sha256_file(str(fpath))
    with open(pkg_dir / "artifact_hash_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "sha256"])
        for k, v in sorted(artifacts.items()):
            w.writerow([k, v])

    status = "PASS" if is_primary and clean_grp == 31872 else (
        "DIAGNOSTIC_" + eligibility if not is_primary else "PRIMARY_FAIL_" + eligibility)
    print(f"  {pid} step{step}: {status} gripper={clean_grp} primary={is_primary}")

    return {
        "parent_id": pid, "step": step, "role": spec["role"],
        "clean_gripper_token": clean_grp, "clean_eligibility": eligibility,
        "is_primary": is_primary, "raw_sha": raw_sha, "pt_sha": pt_sha,
        "prompt_sha": prompt_sha, "package_dir": str(pkg_dir),
    }


def main():
    out_base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/liuyu/outputs/l3_deepseek_autonomous_20260617_r1/canonical_packages")
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"H0-P: Generating 10 canonical clean packages")
    print(f"  Output: {out_base}")
    print(f"  Model: {MODEL_PATH}")

    model, processor, device = load_model()
    model_dtype = next(model.parameters()).dtype
    print(f"  Device: {device}, dtype: {model_dtype}")

    results = []
    for i, spec in enumerate(SELECTED_FRAMES):
        print(f"[{i+1}/10] {spec['parent_id']} step{spec['step']} ({spec['role']})")
        result = process_one_frame(spec, model, processor, device, model_dtype, out_base)
        if result:
            results.append(result)
        torch.cuda.empty_cache()

    # Summary
    primary_ok = sum(1 for r in results if r["is_primary"] and r["clean_eligibility"] == "CLEAN_ELIGIBLE")
    primary_total = sum(1 for r in results if r["is_primary"])
    print(f"\n=== H0-P Summary ===")
    print(f"  Packages: {len(results)}/10")
    print(f"  Primary CLOSE: {primary_ok}/{primary_total}")

    with open(out_base / "h0_package_summary.json", "w") as f:
        json.dump({
            "n_packages": len(results),
            "primary_clean_close": f"{primary_ok}/{primary_total}",
            "results": results,
        }, f, indent=2, default=str)

    return 0 if len(results) == 10 else 1


if __name__ == "__main__":
    sys.exit(main())
