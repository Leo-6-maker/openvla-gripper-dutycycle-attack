#!/usr/bin/env python3
"""H0-P v2: Generate canonical clean packages with corrected contracts.

Fixes vs v1:
- clean_action.npy: continuous 7D action (not token IDs)
- prompt text: actual prompt() output from v4_run_eval_openvla
- tensor SHA: uses M3 canonical tensor_sha256 (torch.save + sha256)
- source provenance: source_commit + timing_trace_sha non-empty
- artifact manifest with recursive SHAs
"""
import csv, hashlib, json, os, re, sys, time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "stageb"))

os.environ.setdefault("OPENVLA_ATTN_IMPLEMENTATION", "eager")

MODEL_PATH = "/data/aviary/models/openvla/openvla-7b-finetuned-libero-object"
UNNORM_KEY = "libero_object"
FRAME_DIR = "/data/liuyu/outputs/l12_frame_handoff_v2_r1"
TIMING_PANEL = "/data/liuyu/outputs/l12_timing_panel_v2"
TIMING_RESUME = "/data/liuyu/outputs/l12_timing_panel_v2_resume_r1"
SOURCE_COMMIT = "50da442c1b033a780b802c6345c376b23d4833b1"

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
     "role": "teacher_we", "inside_window": False, "primary": False},
    {"parent_id": "tomato_sauce_s23", "task": "tomato_sauce", "state_id": 23, "step": 69,
     "role": "d5_emit", "inside_window": False, "primary": False},
    {"parent_id": "tomato_sauce_s23", "task": "tomato_sauce", "state_id": 23, "step": 139,
     "role": "teacher_ws", "inside_window": True, "primary": True},
    {"parent_id": "tomato_sauce_s23", "task": "tomato_sauce", "state_id": 23, "step": 141,
     "role": "teacher_anchor", "inside_window": True, "primary": True},
    {"parent_id": "salad_dressing_s11", "task": "salad_dressing", "state_id": 11, "step": 57,
     "role": "teacher_ws", "inside_window": True, "primary": True},
    {"parent_id": "salad_dressing_s11", "task": "salad_dressing", "state_id": 11, "step": 59,
     "role": "teacher_anchor", "inside_window": True, "primary": True},
    {"parent_id": "salad_dressing_s11", "task": "salad_dressing", "state_id": 11, "step": 67,
     "role": "teacher_we", "inside_window": False, "primary": False},
    {"parent_id": "salad_dressing_s11", "task": "salad_dressing", "state_id": 11, "step": 128,
     "role": "d5_emit", "inside_window": False, "primary": False},
]


def sha256_file(path):
    if not os.path.isfile(path): return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


def tensor_sha256(t):
    """Canonical M3 tensor SHA: torch.save + sha256 of bytes."""
    import io
    buf = io.BytesIO()
    torch.save(t.detach().cpu(), buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_model():
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoModelCls
    except Exception:
        from transformers import AutoModelForVision2Seq as AutoModelCls
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    mm = os.environ.get("OPENVLA_CUDA_MAX_MEMORY", "10000MiB")
    visible = torch.cuda.device_count()
    max_memory = {idx: mm for idx in range(max(visible, 1))}
    max_memory["cpu"] = "128GiB"
    model = AutoModelCls.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map="auto", max_memory=max_memory,
        attn_implementation="eager",
    )
    device = "cuda:0"
    if hasattr(model, "hf_device_map"):
        for v in model.hf_device_map.values():
            if isinstance(v, int):
                device = f"cuda:{v}"
                break
            elif isinstance(v, str) and v.startswith("cuda"):
                device = str(v)
                break
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


def get_timing_trace_sha(task: str, state_id: int) -> str:
    tag = f"{task}_s{state_id}"
    for base in [TIMING_PANEL, TIMING_RESUME]:
        ep_dir = os.path.join(base, f"{tag}_shadow_attempt1")
        st_csv = os.path.join(ep_dir, "step_trace.csv")
        if os.path.isfile(st_csv):
            return sha256_file(st_csv)
    return ""


def process_one_frame(spec: Dict, model, processor, device: str, model_dtype, out_base: Path):
    from v4_run_eval_openvla import decode_with_scores, prompt, postprocess_openvla_action_for_libero
    from gripper_attack.attack_adapter import prepare_openvla_image_for_attack

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

    # Use decode_with_scores to get BOTH continuous action AND tokens
    action, scores, dt, gen = decode_with_scores(
        model, processor, device, raw_image, instruction,
        UNNORM_KEY, 8,
        libero_official_preprocess=False,
        libero_preprocess_backend="official_pil_lanczos",
        center_crop=True, resize_size=224, drop_attention_mask=True,
    )
    clean_action = np.asarray(action, dtype=np.float32)
    env_action = postprocess_openvla_action_for_libero(clean_action, enabled=True)
    np.save(pkg_dir / "clean_action.npy", clean_action)

    action_dim = int(model.get_action_dim(UNNORM_KEY))

    # Canonical preprocessing — same as attack runner
    preproc_inputs = {}
    raw_pil = None
    from PIL import Image
    if isinstance(raw_image, np.ndarray):
        raw_pil = Image.fromarray(raw_image)
    proc_image = prepare_openvla_image_for_attack(
        raw_image if raw_pil is None else raw_pil,
        libero_official_preprocess=False,
        libero_preprocess_backend="official_pil_lanczos",
        center_crop=True, resize_size=224,
    )
    actual_prompt = prompt(instruction)
    inputs = processor(actual_prompt, proc_image, return_tensors="pt")
    inputs.pop("attention_mask", None)
    input_ids = inputs["input_ids"].to(device)
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat([input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)], dim=1)
    pixel_values = inputs["pixel_values"].to(device=device, dtype=model_dtype)
    preproc_inputs = {"input_ids": input_ids, "pixel_values": pixel_values}
    torch.save({k: v.detach().cpu() for k, v in preproc_inputs.items()}, pkg_dir / "processor_inputs_attack.pt")

    # Canonical tensor SHA (M3 algorithm: torch.save + sha256)
    pt_sha = tensor_sha256(pixel_values)
    prompt_sha = tensor_sha256(input_ids)

    # Score invariance check
    with torch.inference_mode():
        gen_out = model.generate(
            input_ids=input_ids, pixel_values=pixel_values,
            max_new_tokens=action_dim, do_sample=False,
            return_dict_in_generate=True, output_scores=True,
        )
    from gripper_attack.v3_generation_parity import extract_exact_new_tokens as eent
    official_tokens = eent(gen_out.sequences, prompt_len=int(input_ids.shape[1]), expected_new_tokens=action_dim)
    score_row = gen_out.scores[-1][0].detach().float().cpu()
    from scripts.stageb.diagnose_m3_true_pgd_fixed_frame import validate_processed_argmax_matches_emitted
    invariant = validate_processed_argmax_matches_emitted(score_row, int(official_tokens[-1]), tolerance=1e-6)

    clean_grp = int(official_tokens[-1])
    is_primary = spec.get("primary", False)
    inside_window = spec.get("inside_window", False)

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

    timing_sha = get_timing_trace_sha(task, sid)

    clean_gen = {
        "parent_id": pid, "task": task, "state_id": sid,
        "frame_step": step, "frame_role": spec["role"],
        "inside_teacher_window": inside_window, "is_primary": is_primary,
        "instruction": instruction,
        "prompt": actual_prompt,
        "clean_action": clean_action.tolist(),
        "clean_env_action": env_action.tolist(),
        "exact_clean_7_tokens": [int(t) for t in official_tokens],
        "clean_arm_prefix": [int(t) for t in official_tokens[:6]],
        "clean_gripper_token": clean_grp,
        "official_score_invariant": {
            "tie_aware_pass": invariant["tie_aware_pass"],
            "argmax_token": invariant["argmax_token"],
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
            "center_crop": True, "resize_size": 224,
            "postprocess_gripper": True,
        },
        "source_commit": SOURCE_COMMIT,
        "source_timing_trace_sha": timing_sha,
        "source_osb_waiver": "OBS_SEQUENCE_IDENTITY_UNAVAILABLE_SOURCE_NOT_CAPTURED",
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
            "prompt", "source_commit", "source_timing_trace_sha",
        ])
        w.writeheader()
        w.writerow({
            "parent_id": pid, "task": task, "state_id": str(sid),
            "frame_step": str(step), "frame_role": spec["role"],
            "raw_frame_sha256": raw_sha, "processor_tensor_sha256": pt_sha,
            "prompt_token_sha256": prompt_sha,
            "clean_gripper_token": str(clean_grp),
            "clean_eligibility": eligibility, "instruction": instruction,
            "prompt": actual_prompt, "source_commit": SOURCE_COMMIT,
            "source_timing_trace_sha": timing_sha,
        })

    # Recursive artifact hash manifest
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
        f"DIAGNOSTIC_{eligibility}" if not is_primary else f"PRIMARY_FAIL_{eligibility}")
    print(f"  {pid} step{step}: {status} gripper={clean_grp} primary={is_primary} "
          f"action_dim={len(clean_action)}")

    return {
        "parent_id": pid, "step": step, "role": spec["role"],
        "clean_gripper_token": clean_grp, "clean_eligibility": eligibility,
        "is_primary": is_primary, "raw_sha": raw_sha, "pt_sha": pt_sha,
        "prompt_sha": prompt_sha, "package_dir": str(pkg_dir),
        "clean_action_shape": list(clean_action.shape),
    }


def main():
    out_base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/data/liuyu/outputs/l3_deepseek_autonomous_20260617_r1/canonical_packages_v2")
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"H0-P v2: Generating 10 canonical clean packages")
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

    primary_ok = sum(1 for r in results if r["is_primary"] and r["clean_eligibility"] == "CLEAN_ELIGIBLE")
    primary_total = sum(1 for r in results if r["is_primary"])
    print(f"\n=== H0-P v2 Summary ===")
    print(f"  Packages: {len(results)}/10")
    print(f"  Primary CLOSE: {primary_ok}/{primary_total}")
    for r in results:
        action_shape = r.get("clean_action_shape", "?")
        print(f"  {r['parent_id']} step{r['step']}: gripper={r['clean_gripper_token']} "
              f"action_shape={action_shape}")

    with open(out_base / "h0_package_summary_v2.json", "w") as f:
        json.dump({"n_packages": len(results), "primary_clean_close": f"{primary_ok}/{primary_total}",
                   "results": results}, f, indent=2, default=str)

    return 0 if len(results) == 10 else 1


if __name__ == "__main__":
    sys.exit(main())
