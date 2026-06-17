#!/usr/bin/env python3
"""C0: H3 targeted frame capture worker. One instance per GPU pair."""
import csv, hashlib, json, os, subprocess, sys, time, yaml
from pathlib import Path

import numpy as np
import torch

REPO = Path(os.environ.get("H3_REPO", "/data/liuyu/worktrees/l3_h3_h5_2h_20260617"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "stageb"))

MODEL_PATH = "/data/aviary/models/openvla/openvla-7b-finetuned-libero-object"
UNNORM_KEY = "libero_object"
OUT_BASE = Path(os.environ.get("H3_OUT", "/data/liuyu/outputs/l3_h3_h5_2h_20260617_r1"))
EXISTING_DIR = "/data/liuyu/outputs/l3_deepseek_autonomous_20260617_r1/canonical_packages_v2"
SOURCE_COMMIT = "bcd945f5e9ff4e4f85479032755ed770226b64e6"

TASK_IDX = {
    "alphabet_soup": 0, "cream_cheese": 1, "salad_dressing": 2, "bbq_sauce": 3,
    "ketchup": 4, "tomato_sauce": 5, "butter": 6, "milk": 7,
    "chocolate_pudding": 8, "orange_juice": 9,
}


def sha256_file(p):
    if not os.path.isfile(p): return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


def load_model():
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoModelCls
    except Exception:
        from transformers import AutoModelForVision2Seq as AutoModelCls
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    visible = torch.cuda.device_count()
    mm = os.environ.get("OPENVLA_CUDA_MAX_MEMORY", "10000MiB")
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
                device = f"cuda:{v}"; break
            elif isinstance(v, str) and v.startswith("cuda"):
                device = v; break
    return model, processor, device


def model_fingerprint(model):
    cfg = getattr(model, "config", None)
    return {
        "model_type": str(getattr(cfg, "model_type", "")),
        "vocab_size": int(getattr(getattr(cfg, "text_config", cfg), "vocab_size", 0) or 0),
        "pad_to_multiple_of": int(getattr(cfg, "pad_to_multiple_of", 0) or 0),
    }


def capture_one(task, state_id, step, parent_id, model, processor, device, model_dtype):
    from v4_run_eval_openvla import decode_with_scores, prompt, postprocess_openvla_action_for_libero
    from gripper_attack.attack_adapter import prepare_openvla_image_for_attack
    from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env
    from libero.libero import benchmark, get_libero_path

    tag = "{}_s{}_step{}".format(task, state_id, step)
    pkg_dir = OUT_BASE / "h3_packages" / "{}_{}_step{:04d}".format(parent_id, task, step)
    pkg_dir.mkdir(parents=True, exist_ok=True)

    # Check if already captured
    if (pkg_dir / "clean_generation.json").exists():
        print("  {}: already captured".format(tag))
        return True, str(pkg_dir)

    # Check if we can reuse from existing
    existing_pkg = os.path.join(EXISTING_DIR, "{}_step{:04d}".format(parent_id, step))
    if os.path.isdir(existing_pkg) and os.path.isfile(os.path.join(existing_pkg, "clean_generation.json")):
        print("  {}: reuse from existing (SHA={})".format(tag, sha256_file(os.path.join(existing_pkg, "raw_frame.npy"))[:16]))
        return True, existing_pkg

    # Run episode to capture frame
    bm = benchmark.get_benchmark_dict()
    suite = bm["libero_object"]()
    task_idx = TASK_IDX[task]
    task_obj = suite.get_task(task_idx)
    init_states = suite.get_task_init_states(task_idx)
    bddl = os.path.join(get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file)

    render_gpu = int(os.environ.get("H3_RENDER_GPU", "5"))
    env, obs = build_v4_exact_env(bddl, render_gpu, 400, 10)
    obs = env.set_init_state(init_states[state_id])
    env, obs = apply_dummy_wait(env, obs, 10)
    instruction = task_obj.language

    raw_at_step = None
    clean_action = None
    env_action = None
    obs_at_step = None
    terminated_early = False

    for s in range(step + 1):
        if "agentview_image" not in obs:
            terminated_early = True; break
        raw = np.asarray(obs["agentview_image"]).copy()

        if s == step:
            raw_at_step = raw
            obs_at_step = {k: np.asarray(v).copy() if hasattr(v, "__array__") else v
                          for k, v in obs.items() if k in ("agentview_image", "robot0_gripper_qpos")}
            action, _scores, _dt, _gen = decode_with_scores(
                model, processor, device, raw, instruction, UNNORM_KEY, 8,
                libero_official_preprocess=False, libero_preprocess_backend="official_pil_lanczos",
                center_crop=True, resize_size=224, drop_attention_mask=True,
            )
            clean_action = np.asarray(action, dtype=np.float32)
            env_action = postprocess_openvla_action_for_libero(clean_action, enabled=True)
            break

        action, _scores, _dt, _gen = decode_with_scores(
            model, processor, device, raw, instruction, UNNORM_KEY, 8,
            libero_official_preprocess=False, libero_preprocess_backend="official_pil_lanczos",
            center_crop=True, resize_size=224, drop_attention_mask=True,
        )
        obs, _reward, _done, _info = env.step(postprocess_openvla_action_for_libero(np.asarray(action, dtype=np.float32), enabled=True))

    env.close()

    if terminated_early or raw_at_step is None:
        with open(pkg_dir / "FRAME_UNAVAILABLE.txt", "w") as f:
            f.write("Episode terminated before step {}".format(step))
        print("  {}: FRAME_UNAVAILABLE_EPISODE_TERMINATED".format(tag))
        return False, str(pkg_dir)

    # Save
    np.save(pkg_dir / "raw_frame.npy", raw_at_step)
    raw_sha = sha256_file(str(pkg_dir / "raw_frame.npy"))
    with open(pkg_dir / "raw_frame.sha256", "w") as f:
        f.write(raw_sha)

    np.save(pkg_dir / "clean_action.npy", clean_action)

    # Canonical preprocessing
    from PIL import Image
    proc_image = prepare_openvla_image_for_attack(
        raw_at_step, libero_official_preprocess=False,
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
    torch.save({"input_ids": input_ids.detach().cpu(), "pixel_values": pixel_values.detach().cpu()},
               pkg_dir / "processor_inputs_attack.pt")

    # Tensor SHAs (M3 canonical)
    import io
    def tsha(t):
        buf = io.BytesIO(); torch.save(t.detach().cpu(), buf); return hashlib.sha256(buf.getvalue()).hexdigest()
    pt_sha = tsha(pixel_values)
    prompt_sha = tsha(input_ids)

    # Clean official decode
    with torch.inference_mode():
        gen_out = model.generate(
            input_ids=input_ids, pixel_values=pixel_values,
            max_new_tokens=model.get_action_dim(UNNORM_KEY), do_sample=False,
            return_dict_in_generate=True, output_scores=True,
        )
    from gripper_attack.v3_generation_parity import extract_exact_new_tokens
    tokens = extract_exact_new_tokens(gen_out.sequences, prompt_len=int(input_ids.shape[1]),
                                      expected_new_tokens=model.get_action_dim(UNNORM_KEY))
    score_row = gen_out.scores[-1][0].detach().float().cpu()
    from scripts.stageb.diagnose_m3_true_pgd_fixed_frame import validate_processed_argmax_matches_emitted
    invariant = validate_processed_argmax_matches_emitted(score_row, int(tokens[-1]), tolerance=1e-6)

    clean_gen = {
        "parent_id": parent_id, "task": task, "state_id": state_id, "frame_step": step,
        "instruction": instruction, "prompt": actual_prompt,
        "clean_action": clean_action.tolist(),
        "env_action": env_action.tolist() if env_action is not None else [],
        "exact_clean_7_tokens": [int(t) for t in tokens],
        "clean_arm_prefix": [int(t) for t in tokens[:6]],
        "clean_gripper_token": int(tokens[-1]),
        "score_invariant": invariant,
        "raw_frame_sha256": raw_sha,
        "processor_tensor_sha256": pt_sha,
        "prompt_token_sha256": prompt_sha,
        "model_fingerprint": model_fingerprint(model),
        "model_path": MODEL_PATH, "unnorm_key": UNNORM_KEY,
        "preprocessing_contract": {"libero_official_preprocess": False, "center_crop": True, "resize_size": 224},
        "source_commit": SOURCE_COMMIT,
        "obs_sha256": sha256_file(str(pkg_dir / "raw_frame.npy")),  # proxy via raw frame
        "obs_waiver": "ACTION_ENV_RAWFRAME_EXACT_BOUND_WITH_OBS_WAIVER",
    }
    with open(pkg_dir / "clean_generation.json", "w") as f:
        json.dump(clean_gen, f, indent=2)

    eligibility = "CLEAN_ELIGIBLE" if int(tokens[-1]) == 31872 else (
        "CLEAN_ALREADY_TARGET" if int(tokens[-1]) == 31744 else "CLEAN_NOT_CLOSE")
    print("  {}: captured grip={} eligibility={} action_dim={}".format(
        tag, int(tokens[-1]), eligibility, len(clean_action)))
    return True, str(pkg_dir)


def main():
    queue_csv = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "tables/l3_h3_capture_queue.csv")
    worker_label = os.environ.get("H3_WORKER", "A")

    queue = list(csv.DictReader(open(queue_csv)))
    my_jobs = [r for r in queue if r["worker"].startswith(worker_label)]
    print("Worker {}: {} jobs".format(worker_label, len(my_jobs)))

    model, processor, device = load_model()
    model_dtype = next(model.parameters()).dtype
    print("Model loaded on {}".format(device))

    ledger = []
    for i, job in enumerate(my_jobs):
        task = job["task"]
        state_id = int(job["state_id"])
        step = int(job["step"])
        pid = job["parent_id"]
        print("[{}/{}] {} step{}".format(i+1, len(my_jobs), pid, step))

        try:
            ok, pkg_dir = capture_one(task, state_id, step, pid, model, processor, device, model_dtype)
            ledger.append({"parent_id": pid, "task": task, "state_id": state_id, "step": step,
                          "status": "CAPTURED" if ok else "FAILED", "pkg_dir": pkg_dir})
        except Exception as e:
            print("  ERROR: {}".format(e))
            ledger.append({"parent_id": pid, "task": task, "state_id": state_id, "step": step,
                          "status": "ERROR", "error": str(e)})

    # Summary
    n_ok = sum(1 for r in ledger if r["status"] == "CAPTURED")
    print("\nWorker {}: {}/{} captured".format(worker_label, n_ok, len(my_jobs)))

    with open(OUT_BASE / "c0_worker_{}_ledger.json".format(worker_label.lower()), "w") as f:
        json.dump(ledger, f, indent=2)

    sys.exit(0 if n_ok == len(my_jobs) else 1)


if __name__ == "__main__":
    main()
