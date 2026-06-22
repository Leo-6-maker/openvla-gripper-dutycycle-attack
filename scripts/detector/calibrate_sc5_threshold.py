#!/usr/bin/env python3
"""T1: Calibrate shared tau_corridor/tau_release on CAL_FP32 episodes."""
import csv, json, os, sys, math
import numpy as np
from itertools import product

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from gripper_attack.sc5mlp_v1 import SC5MLPV1, SC5_FEATURES, SC5_PHASES


def main():
    ckpt = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    ckpt_path = ckpt.get("ckpt", os.path.join(REPO, "outputs/sc5_training/seed1/sc5_mlp_s1.pt"))
    cal_dir = ckpt.get("cal_dir", os.path.join(REPO, "evidence/c1_val_cal_xfer_fp32_gpu4"))
    output = ckpt.get("output", os.path.join(REPO, "outputs/sc5_calibration/threshold_search.json"))

    # Load checkpoint
    import torch
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mean = data["mean"]
    std = data["std"]
    model = SC5MLPV1()
    # Filter out confidence_head keys from old checkpoint
    state = {k: v for k, v in data["model_state"].items() if not k.startswith("confidence_head")}
    model.load_state_dict(state, strict=False)
    model.eval()

    # Load CAL episodes (init10)
    cal_eps = []
    for d in sorted(os.listdir(cal_dir)):
        ep_dir = os.path.join(cal_dir, d)
        if not os.path.isdir(ep_dir):
            continue
        tf = os.path.join(ep_dir, "trace.csv")
        if not os.path.exists(tf):
            continue
        trace = list(csv.DictReader(open(tf)))
        ii = int(trace[0].get("init_idx", -1))
        if ii != 10:
            continue
        cal_eps.append((d, trace))

    print(f"CAL episodes: {len(cal_eps)}")

    # Grid search
    taus = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
            0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    guard = 5
    K = 10

    results = []
    for tau_c, tau_r in product(taus, taus):
        metrics = {"tau_c": tau_c, "tau_r": tau_r,
                   "post_release_triggers": 0, "false_early": 0,
                   "no_corridor_abstain": 0, "k10_containment": 0,
                   "anchor_errors": [], "coverage": 0, "total_corridor_eps": 0}
        total_eps = len(cal_eps)

        for ep_name, trace in cal_eps:
            # Find teacher corridor windows
            teacher_steps = []
            for row in trace:
                phase = row.get("teacher_phase", "abstain_unsupported")
                if phase in ("stable_carry", "pre_place_unsupported"):
                    teacher_steps.append(int(row.get("step_idx", 0)))

            has_corridor = len(teacher_steps) > 0
            if not has_corridor:
                # No corridor — detector should abstain
                any_trigger = False
                # Run detector through trace to check for triggers
                state = "IDLE"
                arm_step = -1
                for row in trace:
                    X = np.array([[float(row.get(fn, 0)) for fn in SC5_FEATURES]], dtype=np.float32)
                    X = (X - mean) / (std + 1e-8)
                    with torch.no_grad():
                        out = model(torch.tensor(X, dtype=torch.float32))
                    cp = torch.sigmoid(out["corridor_logit"]).item()
                    rp = torch.sigmoid(out["release_logit"]).item()
                    pp = SC5_PHASES[out["phase_logits"][0].argmax().item()]
                    step = int(row.get("step_idx", 0))

                    if state == "IDLE":
                        if pp == "stable_carry" and cp > tau_c:
                            state = "ARMED"
                            arm_step = step
                    elif state == "ARMED":
                        if step >= arm_step + guard and cp > tau_c and rp < tau_r:
                            state = "EMITTED"
                            any_trigger = True
                            break

                if not any_trigger:
                    metrics["no_corridor_abstain"] += 1
                continue

            metrics["total_corridor_eps"] += 1

            # Run detector
            state = "IDLE"
            arm_step = -1
            emit_step = -1
            false_early_flag = False
            post_release_flag = False

            for row in trace:
                X = np.array([[float(row.get(fn, 0)) for fn in SC5_FEATURES]], dtype=np.float32)
                X = (X - mean) / (std + 1e-8)
                with torch.no_grad():
                    out = model(torch.tensor(X, dtype=torch.float32))
                cp = torch.sigmoid(out["corridor_logit"]).item()
                rp = torch.sigmoid(out["release_logit"]).item()
                pp = SC5_PHASES[out["phase_logits"][0].argmax().item()]
                step = int(row.get("step_idx", 0))

                if state == "IDLE":
                    if pp == "stable_carry" and cp > tau_c:
                        state = "ARMED"
                        arm_step = step
                elif state == "ARMED":
                    if step >= arm_step + guard and cp > tau_c and rp < tau_r:
                        state = "EMITTED"
                        emit_step = step
                        break

            if emit_step >= 0:
                # Check false-early: trigger before first teacher corridor step
                first_teacher = min(teacher_steps) if teacher_steps else 999
                if emit_step < first_teacher:
                    metrics["false_early"] += 1
                # Check post-release
                release_steps = [int(r.get("step_idx", 0)) for r in trace
                                if r.get("teacher_phase") == "release_safe"]
                if release_steps and emit_step > min(release_steps):
                    metrics["post_release_triggers"] += 1

                # K10 containment: trigger within K steps of first teacher corridor
                anchor_err = abs(emit_step - first_teacher)
                metrics["anchor_errors"].append(anchor_err)
                if anchor_err <= K:
                    metrics["k10_containment"] += 1
                metrics["coverage"] += 1

        # Compute summary
        n_corridor = metrics["total_corridor_eps"]
        if n_corridor > 0:
            k10_rate = metrics["k10_containment"] / n_corridor
            cov_rate = metrics["coverage"] / n_corridor
            median_err = np.median(metrics["anchor_errors"]) if metrics["anchor_errors"] else 999
        else:
            k10_rate = 0
            cov_rate = 0
            median_err = 999

        false_early_rate = metrics["false_early"] / max(1, n_corridor)
        post_rel_rate = metrics["post_release_triggers"] / max(1, n_corridor)
        abstain_rate = metrics["no_corridor_abstain"] / max(1, total_eps - n_corridor)

        constraints_met = (post_rel_rate == 0 and false_early_rate <= 0.10
                          and abstain_rate >= 0.90 and cov_rate >= 0.70)

        results.append({
            "tau_c": tau_c, "tau_r": tau_r,
            "k10_containment": round(k10_rate, 4),
            "coverage": round(cov_rate, 4),
            "median_anchor_error": round(median_err, 2),
            "false_early_rate": round(false_early_rate, 4),
            "post_release_rate": round(post_rel_rate, 4),
            "no_corridor_abstain": round(abstain_rate, 4),
            "constraints_met": constraints_met,
            "n_corridor_eps": n_corridor,
        })

    # Filter constraint-satisfying candidates
    valid = [r for r in results if r["constraints_met"]]
    print(f"Grid: {len(results)} combos, {len(valid)} valid")

    if valid:
        # Rank: max K10, min median error, min distance from (0.3, 0.3)
        valid.sort(key=lambda r: (-r["k10_containment"], r["median_anchor_error"],
                                   abs(r["tau_c"] - 0.3) + abs(r["tau_r"] - 0.3)))
        best = valid[0]
        print(f"Best: tau_c={best['tau_c']} tau_r={best['tau_r']} "
              f"k10={best['k10_containment']:.3f} med_err={best['median_anchor_error']}")
        status = "PASS"
    else:
        best = {"tau_c": 0.3, "tau_r": 0.3}
        print("No valid candidates! Keeping defaults (0.3, 0.3)")
        status = "THRESHOLD_CALIBRATION_FAIL"

    os.makedirs(os.path.dirname(output), exist_ok=True)
    json.dump({"status": status, "best": best, "all_results": results,
               "n_cal_eps": len(cal_eps), "guard": guard, "K": K}, open(output, "w"), indent=2)
    print(f"Saved: {output}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
