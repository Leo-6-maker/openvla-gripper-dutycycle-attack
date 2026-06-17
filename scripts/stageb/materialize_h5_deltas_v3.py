#!/usr/bin/env python3
"""Phase B v3: Corrected delta materialization.

Fixes:
- Delta SHA computed on float32 (not bfloat16)
- RAND verifies frozen candidate ID matches selected
- Saves 6 immutable artifacts with metadata.json
"""
import csv, hashlib, io, json, os, sys, torch, numpy as np, yaml
from pathlib import Path

REPO = Path("/data/liuyu/worktrees/l3_h5_oracle_20260617")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "stageb"))

from run_m3_step78_true_pgd_fixed_frame import *

OUT = Path("/data/liuyu/outputs/l3_h5_p0_1_repair_20260617_r1/frozen_artifacts_v3")
OUT.mkdir(parents=True, exist_ok=True)


def tsha(t):
    """M3 canonical tensor SHA: torch.save + sha256 of bytes."""
    buf = io.BytesIO(); torch.save(t.detach().cpu(), buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def main():
    cfg = load_config(Path("configs/m3_butter_s11_step60_v4.yaml"))

    inputs = {
        81: "/data/liuyu/outputs/l3_deepseek_autonomous_20260617_r1/h1_v4_butter_s11_step60_seed81/input",
        82: "/data/liuyu/outputs/l3_deepseek_autonomous_20260617_r1/h2_v4_primary/butter_s11_step0060_seed82/input",
    }

    bindings = json.load(open(REPO / "artifacts/l3_h5_candidate_bindings_v2.json"))

    all_results = []

    for seed in [81, 82]:
        print("\n=== Seed {} ===".format(seed))
        seed_dir = OUT / "seed{}".format(seed)
        seed_dir.mkdir(parents=True, exist_ok=True)

        raw_image, clean_json = load_frozen_input(Path(inputs[seed]))
        model, processor, device = load_model(cfg["model"]["path"], -1)
        model_dtype = next(model.parameters()).dtype
        action_dim = int(model.get_action_dim(cfg["model"]["unnorm_key"]))
        instruction = str(clean_json["instruction"])
        clean_action = np.asarray(clean_json["clean_action"], dtype=np.float32)

        base_inputs = preprocess_raw_image(raw_image, processor, instruction, cfg, device, model_dtype)
        clean_pv = base_inputs["pixel_values"]

        # Save clean
        torch.save(clean_pv.detach().cpu(), seed_dir / "clean_pixel_values.pt")
        clean_pv_sha = tsha(clean_pv)
        print("  Clean PV SHA: {}".format(clean_pv_sha[:16]))

        clean_gen_obj = type("CleanGen", (), {})()
        clean_gen_obj.sequences = torch.tensor(
            [base_inputs["input_ids"][0].detach().cpu().tolist() + clean_json["clean_exact_7_tokens"]],
            dtype=torch.long, device=base_inputs["input_ids"].device,
        )
        clean_gen_obj.scores = []

        # Candidate audit for reference
        if seed == 81:
            cand_csv = "/data/liuyu/outputs/l3_deepseek_autonomous_20260617_r1/h1_v4_butter_s11_step60_seed81/canary/m3_v4_candidate_audit.csv"
        else:
            cand_csv = "/data/liuyu/outputs/l3_deepseek_autonomous_20260617_r1/h2_v4_primary/butter_s11_step0060_seed82/canary/m3_v4_candidate_audit.csv"
        ref_rows = list(csv.DictReader(open(cand_csv)))

        # ── TRUE PGD ──
        print("  TRUE_PGD_TRAJECTORY21_SELECTIVE...")
        true_info, true_inputs = run_true_pgd_condition(
            name="TRUE_PGD_TRAJECTORY21_SELECTIVE", model=model, processor=processor,
            cfg=cfg, raw_image=raw_image, instruction=instruction,
            clean_action=clean_action, clean_gen=clean_gen_obj,
            device=device, seed=seed, gradient_transform="none",
        )
        true_traj = true_info["debug"].get("trajectory_candidate_inputs", [])

        # SHUFFLED
        print("  SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE...")
        shuf_info, shuf_inputs = run_true_pgd_condition(
            name="SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE", model=model, processor=processor,
            cfg=cfg, raw_image=raw_image, instruction=instruction,
            clean_action=clean_action, clean_gen=clean_gen_obj,
            device=device, seed=seed, gradient_transform=str(cfg["controls"]["shuffled_grad_mode"]),
        )
        shuf_traj = shuf_info["debug"].get("trajectory_candidate_inputs", [])

        # RAND21 — compute locally to capture pixel_values
        print("  RAND21_SELECTIVE...")
        from gripper_attack.m3_controls import rand_seed_schedule, sample_processor_delta, project_and_cast_processor_values
        rand_count = int(cfg["controls"].get("rand21_count", 21))
        rand_seeds = rand_seed_schedule(seed + 100000, count=rand_count)
        rand_pixel_values = {}  # candidate_id -> pixel_values tensor
        rand_scores = []
        adapter = TokenPrefixPGDAttacker(
            model, processor, {"attack_optimizer": cfg["attack_optimizer"]},
            seed=seed, preprocess_kwargs=dict(cfg.get("preprocess", {})), device=device,
        )
        x = base_inputs["pixel_values"]
        for idx, cand_seed in enumerate(rand_seeds):
            delta = sample_processor_delta(x.shape, epsilon=float(cfg["attack_optimizer"]["epsilon"]),
                                           seed=int(cand_seed), dtype=torch.float32, device=x.device)
            projected, corrections = project_and_cast_processor_values(x, delta, epsilon=float(cfg["attack_optimizer"]["epsilon"]), candidate_is_delta=True)
            rand_pixel_values[idx] = projected.detach()
            cand_inputs = {"input_ids": base_inputs["input_ids"], "pixel_values": projected.detach()}
            stats = surrogate_stats_from_generated_prefix(
                adapter, cand_inputs["input_ids"], cand_inputs["pixel_values"],
                action_dim=action_dim, target_token_id=int(cfg["attack_optimizer"]["target_token_id"]),
                margin=float(cfg["attack_optimizer"]["gripper_margin"]),
            )
            score = float(stats.get("target_objective_margin", stats["target_minus_best_competitor_margin"]))
            rand_scores.append(score)
        from gripper_attack.m3_controls import select_best_surrogate_only
        selected_rand_id = select_best_surrogate_only(list(range(len(rand_scores))), rand_scores)
        print("  RAND selected: {} (scores: min={:.1f} max={:.1f})".format(selected_rand_id, min(rand_scores), max(rand_scores)))

        # ── Materialize 3 conditions ──
        for cond_key in ["true", "rand", "shuffled"]:
            cid = bindings["seed{}".format(seed)][cond_key]["candidate_id"]
            cond_data = conditions_data[cond_key]
            cond_full = {"true": "TRUE_PGD_TRAJECTORY21_SELECTIVE",
                        "rand": "RAND21_SELECTIVE",
                        "shuffled": "SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE"}[cond_key]

            ref_row = next((r for r in ref_rows if r["condition"] == cond_full
                           and int(r.get("candidate_id", "-1")) == cid), None)
            if ref_row is None:
                print("  {} id={}: NO_REF_ROW".format(cond_key, cid))
                continue

            frozen_delta_sha = ref_row["delta_sha256"]
            frozen_proc_sha = ref_row["processor_input_sha256"]

            # Get adversarial pixel_values
            adv_pv = None
            if cond_key == "true":
                tc = next((t for t in true_traj if t.get("candidate_index") == cid), None)
                if tc: adv_pv = tc.get("pixel_values")
            elif cond_key == "shuffled":
                tc = next((t for t in shuf_traj if t.get("candidate_index") == cid), None)
                if tc: adv_pv = tc.get("pixel_values")
            elif cond_key == "rand":
                adv_pv = rand_pixel_values.get(cid)

            if adv_pv is None or not isinstance(adv_pv, torch.Tensor):
                print("  {} id={}: NO_ADV_TENSOR".format(cond_key, cid))
                continue

            # Compute delta in FLOAT32, SHA on float32
            delta_fp32 = adv_pv.detach().float() - clean_pv.detach().float()

            # SHA verification
            adv_sha = tsha(adv_pv)  # original dtype
            delta_sha_fp32 = tsha(delta_fp32)
            linf = delta_fp32.abs().max().item()

            sha_ok = adv_sha == frozen_proc_sha and delta_sha_fp32 == frozen_delta_sha

            # Save artifacts
            name = "seed{}_{}".format(seed, cond_key)
            torch.save(adv_pv.detach().cpu(), seed_dir / "{}_adv_pixel_values.pt".format(name))
            torch.save(delta_fp32.cpu(), seed_dir / "{}_delta.pt".format(name))

            # Independent re-decode
            adv_pv_dev = adv_pv.to(device=device, dtype=model_dtype)
            with torch.inference_mode():
                gen_out = model.generate(
                    input_ids=base_inputs["input_ids"], pixel_values=adv_pv_dev,
                    max_new_tokens=action_dim, do_sample=False,
                    return_dict_in_generate=True, output_scores=True,
                )
            from gripper_attack.v3_generation_parity import extract_exact_new_tokens
            tokens = extract_exact_new_tokens(gen_out.sequences, prompt_len=int(base_inputs["input_ids"].shape[1]),
                                              expected_new_tokens=action_dim)
            gripper = int(tokens[-1])
            arm_match = sum(1 for a, b in zip(tokens[:6].tolist(), clean_json["clean_exact_7_tokens"][:6]) if a == b)
            from scripts.stageb.diagnose_m3_true_pgd_fixed_frame import validate_processed_argmax_matches_emitted
            score_row = gen_out.scores[-1][0].detach().float().cpu()
            invariant = validate_processed_argmax_matches_emitted(score_row, gripper, tolerance=1e-6)

            # RAND ID verification
            rand_id_ok = True
            if cond_key == "rand" and selected_rand_id != cid:
                rand_id_ok = False
                print("  RAND ID MISMATCH: selected={} frozen={}".format(selected_rand_id, cid))

            meta = {
                "seed": seed, "condition": cond_key, "candidate_id": cid,
                "condition_full": cond_full,
                "clean_pv_sha256": clean_pv_sha,
                "adv_pv_sha256": adv_sha, "delta_sha256_fp32": delta_sha_fp32,
                "frozen_adv_sha": frozen_proc_sha, "frozen_delta_sha": frozen_delta_sha,
                "sha_match": sha_ok, "rand_id_match": rand_id_ok,
                "linf": linf, "linf_ok": linf <= 0.02353,
                "official_gripper_token": gripper, "arm_prefix_match": "{}/6".format(arm_match),
                "score_invariant_pass": invariant["tie_aware_pass"],
            }
            with open(seed_dir / "{}_metadata.json".format(name), "w") as f:
                json.dump(meta, f, indent=2)

            status = "OK" if sha_ok and rand_id_ok and linf <= 0.02353 and invariant["tie_aware_pass"] else "FAIL"
            print("  {} id={}: {} sha={} linf={:.6f} grip={} arm={}/6 rand_id_ok={}".format(
                cond_key, cid, status, sha_ok, linf, gripper, arm_match, rand_id_ok))
            all_results.append(meta)

        torch.cuda.empty_cache()

    # ── B-GATE ──
    n_ok = sum(1 for r in all_results if r.get("sha_match") and r.get("rand_id_match", True)
               and r.get("linf_ok") and r.get("score_invariant_pass"))
    all_ok = n_ok == 6
    print("\n" + "=" * 60)
    print("B-GATE: {} — {}/6 artifacts PASS".format("PASS" if all_ok else "FAIL", n_ok))
    print("=" * 60)

    gate = {"phase": "B", "gate_pass": all_ok, "n_pass": n_ok, "n_total": 6, "results": all_results}
    json.dump(gate, open(OUT / "b_gate.json", "w"), indent=2, default=str)

    # Copy to standard location for C/D/E phases
    frozen_dir = Path("/data/liuyu/outputs/l3_h5_p0_1_repair_20260617_r1/frozen_artifacts")
    frozen_dir.mkdir(parents=True, exist_ok=True)
    for seed in [81, 82]:
        src = OUT / "seed{}".format(seed)
        dst = frozen_dir / "seed{}".format(seed)
        if src.is_dir():
            import shutil
            if dst.exists(): shutil.rmtree(str(dst))
            shutil.copytree(str(src), str(dst))

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
