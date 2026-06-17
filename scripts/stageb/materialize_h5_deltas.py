#!/usr/bin/env python3
"""Phase B: Deterministically regenerate and materialize 6 frozen deltas."""
import csv, hashlib, io, json, os, sys, torch, numpy as np, yaml
from pathlib import Path

REPO = Path("/data/liuyu/worktrees/l3_h5_oracle_20260617")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "stageb"))

from v4_run_eval_openvla import prompt
from gripper_attack.attack_adapter import OpenVLAVisualAttacker
from gripper_attack.route_contract import route_config_from_attack_config, validate_true_pgd_attack_result
from gripper_attack.m3_controls import tensor_sha256 as m3_tsha
from transformers import AutoProcessor
try:
    from transformers import AutoModelForImageTextToText as AutoModelCls
except Exception:
    from transformers import AutoModelForVision2Seq as AutoModelCls

MODEL_PATH = "/data/aviary/models/openvla/openvla-7b-finetuned-libero-object"
H2_PKG = "/data/liuyu/outputs/l3_deepseek_autonomous_20260617_r1/canonical_packages_v2/butter_s11_step0060"
OUT = Path("/data/liuyu/outputs/l3_h5_p0_1_repair_20260617_r1/frozen_deltas")
OUT.mkdir(parents=True, exist_ok=True)

BINDINGS_PATH = REPO / "artifacts/l3_h5_candidate_bindings_v2.json"
CONFIG_PATH = REPO / "configs/m3_butter_s11_step60_v4.yaml"

BINDINGS = json.load(open(BINDINGS_PATH))


def tsha(t):
    buf = io.BytesIO()
    torch.save(t.detach().cpu(), buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def main():
    # Load model
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    mm = os.environ.get("OPENVLA_CUDA_MAX_MEMORY", "10000MiB")
    visible = torch.cuda.device_count()
    model = AutoModelCls.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map="auto", max_memory={idx: mm for idx in range(visible)} | {"cpu": "128GiB"},
        attn_implementation="eager",
    )
    model_dtype = next(model.parameters()).dtype
    print("Model loaded")

    # Load clean inputs
    pt_data = torch.load(os.path.join(H2_PKG, "processor_inputs_attack.pt"), map_location="cpu", weights_only=True)
    clean_pv = pt_data["pixel_values"]
    clean_ids = pt_data["input_ids"]
    device = "cuda:0"
    if hasattr(model, "hf_device_map"):
        for v in model.hf_device_map.values():
            if isinstance(v, int):
                device = f"cuda:{v}"; break

    clean_pv = clean_pv.to(device=device, dtype=model_dtype)
    clean_ids = clean_ids.to(device=device)
    print(f"Clean pixel_values on {device}")

    with open(os.path.join(H2_PKG, "clean_generation.json")) as f:
        gen = json.load(f)

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    opt = dict(cfg["attack_optimizer"])

    clean_gen = type("CleanGen", (), {})()
    clean_gen.sequences = torch.tensor(
        [clean_ids[0].detach().cpu().tolist() + gen["exact_clean_7_tokens"]],
        dtype=torch.long, device=device,
    )
    clean_gen.scores = []

    results = []
    for seed in [81, 82]:
        print(f"\n=== Seed {seed} ===")

        attacker = OpenVLAVisualAttacker(
            model=model, processor=processor,
            config={"attack_optimizer": opt}, seed=seed,
            preprocess_kwargs=dict(cfg.get("preprocess", {})), device=device,
        )
        route = route_config_from_attack_config({"attack_optimizer": opt})
        result = attacker.attack(
            np.zeros((224, 224, 3), dtype=np.uint8),
            gen["instruction"],
            np.array(gen["clean_action"], dtype=np.float32),
            np.array(gen["clean_action"], dtype=np.float32),
            clean_gen, unnorm_key="libero_object",
        )
        validate_true_pgd_attack_result(result, route)
        debug = result.debug
        traj_cands = debug.get("trajectory_candidate_inputs", [])
        print(f"  Trajectory candidates: {len(traj_cands)}")

        # Candidate audit CSV
        if seed == 81:
            cand_csv = "/data/liuyu/outputs/l3_deepseek_autonomous_20260617_r1/h1_v4_butter_s11_step60_seed81/canary/m3_v4_candidate_audit.csv"
        else:
            cand_csv = "/data/liuyu/outputs/l3_deepseek_autonomous_20260617_r1/h2_v4_primary/butter_s11_step0060_seed82/canary/m3_v4_candidate_audit.csv"
        cand_rows = list(csv.DictReader(open(cand_csv)))

        for condition in ["true", "rand", "shuffled"]:
            cid = BINDINGS[f"seed{seed}"][condition]["candidate_id"]
            cond_full = {
                "true": "TRUE_PGD_TRAJECTORY21_SELECTIVE",
                "rand": "RAND21_SELECTIVE",
                "shuffled": "SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE",
            }[condition]

            tc = next((t for t in traj_cands if t.get("candidate_index") == cid), None)
            if tc is None:
                print(f"  {condition} id={cid}: NOT_FOUND")
                continue

            adv_data = tc.get("pixel_values")
            if adv_data is None:
                print(f"  {condition} id={cid}: NO_DATA")
                continue

            adv_pv = torch.tensor(adv_data, device=device, dtype=model_dtype)
            delta = adv_pv.float() - clean_pv.float()

            name = f"seed{seed}_{condition}"
            torch.save(delta, OUT / f"{name}_delta.pt")
            torch.save(adv_pv, OUT / f"{name}_adv_pixel_values.pt")

            delta_reload = torch.load(OUT / f"{name}_delta.pt", map_location="cpu", weights_only=True)
            adv_reload = torch.load(OUT / f"{name}_adv_pixel_values.pt", map_location="cpu", weights_only=True)
            delta_sha = tsha(delta_reload)
            proc_sha = tsha(adv_reload)
            linf = delta_reload.float().abs().max().item()

            row = next((r for r in cand_rows if r["condition"] == cond_full and int(r.get("candidate_id", "-1")) == cid), {})
            exp_delta = row.get("delta_sha256", "?")
            exp_proc = row.get("processor_input_sha256", "?")

            ok = delta_sha == exp_delta and proc_sha == exp_proc and linf <= 0.02353
            print(f"  {condition} id={cid}: {'OK' if ok else 'FAIL'} "
                  f"delta_ok={delta_sha==exp_delta} proc_ok={proc_sha==exp_proc} linf={linf:.6f}")
            results.append({"seed": seed, "condition": condition, "cid": cid,
                           "delta_sha_ok": delta_sha == exp_delta,
                           "proc_sha_ok": proc_sha == exp_proc, "linf_ok": linf <= 0.02353})

    all_ok = all(r["delta_sha_ok"] and r["proc_sha_ok"] and r["linf_ok"] for r in results)
    print(f"\nB-GATE: {'PASS' if all_ok else 'FAIL'} — {sum(1 for r in results if r['delta_sha_ok'] and r['proc_sha_ok'] and r['linf_ok'])}/{len(results)}")
    json.dump({"results": results, "gate_pass": all_ok}, open(OUT / "materialization_audit.json", "w"), indent=2)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
