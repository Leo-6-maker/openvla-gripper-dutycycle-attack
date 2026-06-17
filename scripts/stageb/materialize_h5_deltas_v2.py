#!/usr/bin/env python3
"""Phase B v2: Materialize 6 frozen deltas correctly.

Uses actual MuJoCo replay frame (Phase A confirmed pixel_values identity).
Runs each condition separately: TRUE PGD, RAND21, SHUFFLED gradient.
Saves immutable delta.pt + adv_pixel_values.pt + metadata.json.
"""
import csv, hashlib, io, json, os, sys, time, yaml
from pathlib import Path

import numpy as np
import torch

REPO = Path("/data/liuyu/worktrees/l3_h5_oracle_20260617")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "stageb"))

MODEL_PATH = "/data/aviary/models/openvla/openvla-7b-finetuned-libero-object"
H2_PKG = "/data/liuyu/outputs/l3_deepseek_autonomous_20260617_r1/canonical_packages_v2/butter_s11_step0060"
OUT = Path("/data/liuyu/outputs/l3_h5_p0_1_repair_20260617_r1/frozen_deltas_v2")
OUT.mkdir(parents=True, exist_ok=True)

BINDINGS_PATH = REPO / "artifacts/l3_h5_candidate_bindings_v2.json"
CONFIG_PATH = REPO / "configs/m3_butter_s11_step60_v4.yaml"
UNNORM_KEY = "libero_object"


def tsha(t):
    buf = io.BytesIO(); torch.save(t.detach().cpu(), buf); return hashlib.sha256(buf.getvalue()).hexdigest()


def get_replay_frame():
    """Replay butter_s11 state11 to step60, return raw frame + instruction."""
    from v4_run_eval_openvla import decode_with_scores, postprocess_openvla_action_for_libero
    from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env
    from libero.libero import benchmark, get_libero_path

    bm = benchmark.get_benchmark_dict()
    suite = bm["libero_object"]()
    task_obj = suite.get_task(6)
    init_states = suite.get_task_init_states(6)
    bddl = os.path.join(get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file)
    instruction = task_obj.language

    render_gpu = int(os.environ.get("H5_RENDER_GPU", "5"))
    env, obs = build_v4_exact_env(bddl, render_gpu, 400, 10)
    obs = env.set_init_state(init_states[11])
    env, obs = apply_dummy_wait(env, obs, 10)

    raw_at_60 = None
    for step in range(61):
        raw = np.asarray(obs["agentview_image"]).copy()
        action, _, _, _ = decode_with_scores(
            model, processor, device, raw, instruction, UNNORM_KEY, 8,
            libero_official_preprocess=False, libero_preprocess_backend="official_pil_lanczos",
            center_crop=True, resize_size=224, drop_attention_mask=True,
        )
        if step == 60:
            raw_at_60 = raw; break
        obs, _, _, _ = env.step(postprocess_openvla_action_for_libero(np.asarray(action, dtype=np.float32), enabled=True))
    env.close()
    return raw_at_60, instruction


def run_true_pgd(raw_frame, instruction, clean_action, seed):
    """Run TRUE_PGD_TRAJECTORY21_SELECTIVE, return all 21 trajectory candidates."""
    from gripper_attack.attack_adapter import OpenVLAVisualAttacker
    from gripper_attack.route_contract import route_config_from_attack_config, validate_true_pgd_attack_result

    opt = dict(cfg["attack_optimizer"])
    attacker = OpenVLAVisualAttacker(
        model=model, processor=processor, config={"attack_optimizer": opt},
        seed=seed, preprocess_kwargs=dict(cfg.get("preprocess", {})), device=device,
    )
    route = route_config_from_attack_config({"attack_optimizer": opt})
    clean_gen = type("CleanGen", (), {})()
    clean_gen.sequences = torch.tensor(
        [clean_ids[0].detach().cpu().tolist() + gen["exact_clean_7_tokens"]],
        dtype=torch.long, device=device,
    )
    clean_gen.scores = []
    result = attacker.attack(raw_frame, instruction, clean_action, clean_action, clean_gen, unnorm_key=UNNORM_KEY)
    validate_true_pgd_attack_result(result, route)
    return result.debug.get("trajectory_candidate_inputs", [])


def run_rand21(raw_frame, instruction, clean_action, seed):
    """Run RAND21_SELECTIVE, return all 21 candidates with official decode."""
    from gripper_attack.m3_controls import rand_seed_schedule, sample_processor_delta, project_and_cast_processor_values
    from gripper_attack.v3_generation_parity import extract_exact_new_tokens
    from scripts.stageb.diagnose_m3_true_pgd_fixed_frame import validate_processed_argmax_matches_emitted

    count = cfg["controls"]["rand21_count"]
    epsilon = cfg["attack_optimizer"]["epsilon"]
    seeds = rand_seed_schedule(seed + 100000, count=count)

    # Preprocess raw frame
    from gripper_attack.attack_adapter import prepare_openvla_image_for_attack
    from v4_run_eval_openvla import prompt
    from PIL import Image
    proc_image = prepare_openvla_image_for_attack(
        raw_frame, libero_official_preprocess=False,
        libero_preprocess_backend="official_pil_lanczos", center_crop=True, resize_size=224,
    )
    inputs = processor(prompt(instruction), proc_image, return_tensors="pt")
    inputs.pop("attention_mask", None)
    input_ids = inputs["input_ids"].to(device)
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat([input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)], dim=1)
    pixel_values = inputs["pixel_values"].to(device=device, dtype=model_dtype)
    x = pixel_values

    candidates = []
    for idx, cand_seed in enumerate(seeds):
        delta = sample_processor_delta(x.shape, epsilon=epsilon, seed=int(cand_seed),
                                       dtype=torch.float32, device=x.device)
        projected, corrections = project_and_cast_processor_values(x, delta, epsilon=epsilon, candidate_is_delta=True)
        adv_pv = projected.detach()

        # Official decode
        with torch.inference_mode():
            gen_out = model.generate(
                input_ids=input_ids, pixel_values=adv_pv.to(dtype=model_dtype),
                max_new_tokens=model.get_action_dim(UNNORM_KEY), do_sample=False,
                return_dict_in_generate=True, output_scores=True,
            )
        tokens = extract_exact_new_tokens(gen_out.sequences, prompt_len=int(input_ids.shape[1]),
                                          expected_new_tokens=model.get_action_dim(UNNORM_KEY))
        score_row = gen_out.scores[-1][0].detach().float().cpu()
        invariant = validate_processed_argmax_matches_emitted(score_row, int(tokens[-1]), tolerance=1e-6)

        # Compute target margin
        target_score = float(score_row[TARGET_TOKEN])
        others = score_row.clone()
        others[TARGET_TOKEN] = float("-inf")
        best_other_score = float(others.max())
        best_other_token = int(others.argmax())

        candidates.append({
            "candidate_index": idx, "candidate_seed": int(cand_seed),
            "condition": "RAND21_SELECTIVE",
            "pixel_values": adv_pv,
            "delta_sha256": tsha((adv_pv.float() - x.float()).to(dtype=torch.bfloat16)),
            "processor_input_sha256": tsha(adv_pv),
            "official_tokens": [int(t) for t in tokens],
            "official_gripper_token": int(tokens[-1]),
            "arm_prefix_match_count": sum(1 for a, b in zip(tokens[:6].tolist(), gen["clean_arm_prefix"]) if a == b),
            "arm_prefix_match_denominator": 6,
            "official_target31744_margin": target_score - best_other_score,
            "official_best_competitor_token": best_other_token,
            "score_invariant_status": "PASS" if invariant["tie_aware_pass"] else "FAIL",
            "processor_linf": (adv_pv.float() - x.float()).abs().max().item(),
        })
    return candidates


def run_shuffled_grad(raw_frame, instruction, clean_action, seed):
    """Run SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE, return trajectory candidates."""
    from gripper_attack.attack_adapter import OpenVLAVisualAttacker

    opt = dict(cfg["attack_optimizer"])
    opt["gradient_transform"] = "permute"
    opt["gradient_transform_seed"] = seed + 100000

    attacker = OpenVLAVisualAttacker(
        model=model, processor=processor, config={"attack_optimizer": opt},
        seed=seed, preprocess_kwargs=dict(cfg.get("preprocess", {})), device=device,
    )
    clean_gen = type("CleanGen", (), {})()
    clean_gen.sequences = torch.tensor(
        [clean_ids[0].detach().cpu().tolist() + gen["exact_clean_7_tokens"]],
        dtype=torch.long, device=device,
    )
    clean_gen.scores = []
    result = attacker.attack(raw_frame, instruction, clean_action, clean_action, clean_gen, unnorm_key=UNNORM_KEY)
    return result.debug.get("trajectory_candidate_inputs", [])


def main():
    global model, processor, device, model_dtype, cfg, gen, clean_ids, TARGET_TOKEN

    TARGET_TOKEN = 31744

    # Load model
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoModelCls
    except Exception:
        from transformers import AutoModelForVision2Seq as AutoModelCls

    processor_obj = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    processor = processor_obj
    mm = os.environ.get("OPENVLA_CUDA_MAX_MEMORY", "10000MiB")
    visible = torch.cuda.device_count()
    model_obj = AutoModelCls.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map="auto", max_memory={idx: mm for idx in range(visible)} | {"cpu": "128GiB"},
        attn_implementation="eager",
    )
    model = model_obj
    model_dtype = next(model.parameters()).dtype
    device = "cuda:0"
    for v in model.hf_device_map.values():
        if isinstance(v, int): device = f"cuda:{v}"; break

    # Load config and clean data
    with open(CONFIG_PATH) as f:
        cfg_data = yaml.safe_load(f)
    cfg = cfg_data
    with open(os.path.join(H2_PKG, "clean_generation.json")) as f:
        gen = json.load(f)
    pt_data = torch.load(os.path.join(H2_PKG, "processor_inputs_attack.pt"), map_location="cpu", weights_only=True)
    clean_ids = pt_data["input_ids"].to(device=device)
    with open(BINDINGS_PATH) as f:
        bindings = json.load(f)

    # Get replay frame (Phase A confirmed pixel_values identity)
    print("Replaying butter_s11 to step60...")
    raw_frame, instruction = get_replay_frame()
    clean_action = np.array(gen["clean_action"], dtype=np.float32)
    print(f"  Frame captured, shape={raw_frame.shape}")

    condition_funcs = {
        "true": run_true_pgd,
        "rand": run_rand21,
        "shuffled": run_shuffled_grad,
    }
    cond_names = {
        "true": "TRUE_PGD_TRAJECTORY21_SELECTIVE",
        "rand": "RAND21_SELECTIVE",
        "shuffled": "SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE",
    }

    results = []
    for seed in [81, 82]:
        print(f"\n=== Seed {seed} ===")

        # Candidate audit CSV for reference
        if seed == 81:
            cand_csv_path = "/data/liuyu/outputs/l3_deepseek_autonomous_20260617_r1/h1_v4_butter_s11_step60_seed81/canary/m3_v4_candidate_audit.csv"
        else:
            cand_csv_path = "/data/liuyu/outputs/l3_deepseek_autonomous_20260617_r1/h2_v4_primary/butter_s11_step0060_seed82/canary/m3_v4_candidate_audit.csv"
        ref_rows = list(csv.DictReader(open(cand_csv_path)))

        for condition in ["true", "rand", "shuffled"]:
            cid = bindings[f"seed{seed}"][condition]["candidate_id"]
            cond_full = cond_names[condition]
            print(f"  {condition} (id={cid})...")

            candidates = condition_funcs[condition](raw_frame, instruction, clean_action, seed)
            print(f"    Generated {len(candidates)} candidates")

            # Find the matching candidate by SHA
            ref_row = next((r for r in ref_rows if r["condition"] == cond_full and int(r.get("candidate_id", "-1")) == cid), {})
            expected_delta = ref_row.get("delta_sha256", "")
            expected_proc = ref_row.get("processor_input_sha256", "")

            matched = None
            for c in candidates:
                delta_sha = c.get("delta_sha256", "")
                if delta_sha == expected_delta:
                    matched = c; break

            if matched is None:
                # Try matching by processor_input_sha256
                for c in candidates:
                    proc_sha = c.get("processor_input_sha256", "")
                    if proc_sha == expected_proc:
                        matched = c; break

            if matched is None:
                print(f"    FAIL: no candidate matches frozen SHA")
                results.append({"seed": seed, "condition": condition, "cid": cid, "status": "NO_MATCH"})
                continue

            adv_pv = matched["pixel_values"]
            delta = (adv_pv.float() - clean_pv_cpu.float()).to(dtype=torch.bfloat16)

            name = f"seed{seed}_{condition}"
            torch.save(adv_pv.detach().cpu(), OUT / f"{name}_adv_pixel_values.pt")
            torch.save(delta.detach().cpu(), OUT / f"{name}_delta.pt")

            # Verify
            adv_reload = torch.load(OUT / f"{name}_adv_pixel_values.pt", map_location="cpu", weights_only=True)
            delta_reload = torch.load(OUT / f"{name}_delta.pt", map_location="cpu", weights_only=True)
            adv_sha = tsha(adv_reload)
            delta_sha = tsha(delta_reload)
            linf = delta_reload.float().abs().max().item()

            sha_ok = adv_sha == expected_proc and delta_sha == expected_delta
            linf_ok = linf <= 0.02353

            # Independent re-decode
            adv_pv_dev = adv_reload.to(device=device, dtype=model_dtype)
            with torch.inference_mode():
                gen_out = model.generate(
                    input_ids=clean_ids, pixel_values=adv_pv_dev,
                    max_new_tokens=model.get_action_dim(UNNORM_KEY), do_sample=False,
                    return_dict_in_generate=True, output_scores=True,
                )
            from gripper_attack.v3_generation_parity import extract_exact_new_tokens
            tokens = extract_exact_new_tokens(gen_out.sequences, prompt_len=int(clean_ids.shape[1]),
                                              expected_new_tokens=model.get_action_dim(UNNORM_KEY))
            gripper = int(tokens[-1])
            arm_match = sum(1 for a, b in zip(tokens[:6].tolist(), gen["clean_arm_prefix"]) if a == b)
            from scripts.stageb.diagnose_m3_true_pgd_fixed_frame import validate_processed_argmax_matches_emitted
            score_row = gen_out.scores[-1][0].detach().float().cpu()
            invariant = validate_processed_argmax_matches_emitted(score_row, gripper, tolerance=1e-6)

            # Metadata
            meta = {
                "seed": seed, "condition": condition, "candidate_id": cid,
                "condition_full": cond_full,
                "clean_input_ids_sha256": tsha(clean_ids),
                "adv_tensor_sha256": adv_sha, "delta_sha256": delta_sha,
                "frozen_adv_sha": expected_proc, "frozen_delta_sha": expected_delta,
                "linf": linf, "linf_ok": linf_ok, "sha_match": sha_ok,
                "official_gripper_token": gripper, "arm_prefix_match": f"{arm_match}/6",
                "score_invariant_pass": invariant["tie_aware_pass"],
                "config_sha256": tsha(torch.tensor(0)),  # placeholder
            }
            with open(OUT / f"{name}_metadata.json", "w") as f:
                json.dump(meta, f, indent=2)

            status = "OK" if sha_ok and linf_ok and invariant["tie_aware_pass"] else "FAIL"
            print(f"    {status} sha={sha_ok} linf={linf:.6f} grip={gripper} arm={arm_match}/6 inv={invariant['tie_aware_pass']}")
            results.append({"seed": seed, "condition": condition, "cid": cid, "status": status, **meta})

    all_ok = all(r["status"] == "OK" for r in results)
    print(f"\nB-GATE: {'PASS' if all_ok else 'FAIL'} — {sum(1 for r in results if r['status']=='OK')}/{len(results)}")
    json.dump({"results": results, "gate_pass": all_ok}, open(OUT / "materialization_audit_v2.json", "w"), indent=2, default=str)
    return 0 if all_ok else 1


if __name__ == "__main__":
    # Need clean_pv for Linf check
    pt_data = torch.load(os.path.join(H2_PKG, "processor_inputs_attack.pt"), map_location="cpu", weights_only=True)
    clean_pv_cpu = pt_data["pixel_values"]
    sys.exit(main())
