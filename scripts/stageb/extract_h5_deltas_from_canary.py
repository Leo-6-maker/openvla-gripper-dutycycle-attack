#!/usr/bin/env python3
"""Extract deltas by re-running V4 canary on EXACT frozen inputs, save .pt files."""
import io, hashlib, json, os, sys, torch, numpy as np, csv
from pathlib import Path

REPO = Path("/data/liuyu/worktrees/l3_h5_oracle_20260617")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "stageb"))

from run_m3_step78_true_pgd_fixed_frame import *

OUT = Path("/data/liuyu/outputs/l3_h5_p0_1_repair_20260617_r1/delta_extraction_v3")
OUT.mkdir(parents=True, exist_ok=True)


def tsha(t):
    buf = io.BytesIO(); torch.save(t.detach().cpu(), buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def main():
    cfg = load_config(Path("configs/m3_butter_s11_step60_v4.yaml"))

    inputs = {
        81: "/data/liuyu/outputs/l3_deepseek_autonomous_20260617_r1/h1_v4_butter_s11_step60_seed81/input",
        82: "/data/liuyu/outputs/l3_deepseek_autonomous_20260617_r1/h2_v4_primary/butter_s11_step0060_seed82/input",
    }

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

        # Save clean
        torch.save(base_inputs["pixel_values"].detach().cpu(), seed_dir / "clean_pixel_values.pt")
        print("  Clean PV SHA: {}".format(tsha(base_inputs["pixel_values"])[:16]))

        clean_gen_obj = type("CleanGen", (), {})()
        clean_gen_obj.sequences = torch.tensor(
            [base_inputs["input_ids"][0].detach().cpu().tolist() + clean_json["clean_exact_7_tokens"]],
            dtype=torch.long, device=base_inputs["input_ids"].device,
        )
        clean_gen_obj.scores = []

        # TRUE PGD
        print("  TRUE_PGD_TRAJECTORY21_SELECTIVE...")
        true_info, true_inputs = run_true_pgd_condition(
            name="TRUE_PGD_TRAJECTORY21_SELECTIVE", model=model, processor=processor,
            cfg=cfg, raw_image=raw_image, instruction=instruction,
            clean_action=clean_action, clean_gen=clean_gen_obj,
            device=device, seed=seed, gradient_transform="none",
        )
        true_traj = true_info["debug"].get("trajectory_candidate_inputs", [])
        print("  {} TRUE trajectory candidates".format(len(true_traj)))
        for tc in true_traj:
            pv = tc.get("pixel_values")
            if pv is not None and isinstance(pv, torch.Tensor):
                cid = tc.get("candidate_index", 0)
                torch.save(pv.detach().cpu(), seed_dir / "true_cand{}_adv_pv.pt".format(cid))
                delta = (pv.float() - base_inputs["pixel_values"].float()).to(dtype=torch.bfloat16)
                torch.save(delta.detach().cpu(), seed_dir / "true_cand{}_delta.pt".format(cid))
                ds_expected = tc.get("delta_sha256", "?")
                ds_actual = tsha(delta)
                ok = "OK" if ds_expected == ds_actual else "MISMATCH"
                if cid <= 3 or ok == "MISMATCH":
                    print("    cand{}: delta_sha exp={}... act={}... {}".format(cid, ds_expected[:16], ds_actual[:16], ok))

        # SHUFFLED
        print("  SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE...")
        shuf_info, shuf_inputs = run_true_pgd_condition(
            name="SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE", model=model, processor=processor,
            cfg=cfg, raw_image=raw_image, instruction=instruction,
            clean_action=clean_action, clean_gen=clean_gen_obj,
            device=device, seed=seed,
            gradient_transform=str(cfg["controls"]["shuffled_grad_mode"]),
        )
        shuf_traj = shuf_info["debug"].get("trajectory_candidate_inputs", [])
        print("  {} SHUFFLED trajectory candidates".format(len(shuf_traj)))
        for tc in shuf_traj:
            pv = tc.get("pixel_values")
            if pv is not None and isinstance(pv, torch.Tensor):
                cid = tc.get("candidate_index", 0)
                torch.save(pv.detach().cpu(), seed_dir / "shuffled_cand{}_adv_pv.pt".format(cid))
                delta = (pv.float() - base_inputs["pixel_values"].float()).to(dtype=torch.bfloat16)
                torch.save(delta.detach().cpu(), seed_dir / "shuffled_cand{}_delta.pt".format(cid))

        # RAND21
        print("  RAND21_SELECTIVE...")
        rand_info, rand_candidates, rand_inputs = run_rand20(
            model=model, processor=processor, cfg=cfg, base_inputs=base_inputs,
            instruction=instruction, device=device, seed=seed,
            action_dim=action_dim, clean_action=clean_action,
            target_token_id=int(cfg["attack_optimizer"]["target_token_id"]),
            margin=float(cfg["attack_optimizer"]["gripper_margin"]),
        )
        selected_id = rand_info.get("selected_candidate", -1)
        print("  RAND selected: {}".format(selected_id))
        # Save all RAND candidates
        for cand in rand_candidates:
            cid = int(cand.get("candidate_id", -1))
            # RAND pixel_values are in rand_inputs_by_id dict — we need to re-access
            # The rand_inputs here is just the selected one
        # Save selected
        rand_pv = rand_inputs.get("pixel_values")
        if rand_pv is not None:
            torch.save(rand_pv.detach().cpu(), seed_dir / "rand_cand{}_adv_pv.pt".format(selected_id))
            delta = (rand_pv.float() - base_inputs["pixel_values"].float()).to(dtype=torch.bfloat16)
            torch.save(delta.detach().cpu(), seed_dir / "rand_cand{}_delta.pt".format(selected_id))

        torch.cuda.empty_cache()

    print("\nExtraction complete: {}".format(OUT))


if __name__ == "__main__":
    main()
