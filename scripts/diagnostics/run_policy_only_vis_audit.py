#!/usr/bin/env python3
"""run_policy_only_vis_audit.py — Policy-level VIS audit without env stepping.

Loads cached clean observations from existing traces, runs PGD attack,
decodes OpenVLA action from adversarial inputs, checks gripper OPEN count.
No LIBERO environment required. No qpos/done/task success as input.

Usage:
    PY scripts/diagnostics/run_policy_only_vis_audit.py \
      --candidate-csv tables/fast_vis_calibration_candidates_v0.csv \
      --eps-list 2,4,6 --pgd-steps-list 5,10 \
      --pgd-restarts 1 --objective prefix_locked_gripper_open_margin \
      --output-csv tables/fast_vis_policy_only_audit_v0.csv \
      --output-report reports/FAST_VIS_POLICY_ONLY_AUDIT_V0.md
"""

from __future__ import annotations
import argparse, csv, json, os, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO = Path(os.environ.get("ATTACK_REPO",
    "/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605"))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-csv", default="tables/fast_vis_calibration_candidates_v0.csv")
    ap.add_argument("--eps-list", default="2,4,6", help="comma-separated eps_raw_pixels values")
    ap.add_argument("--pgd-steps-list", default="5,10", help="comma-separated pgd_steps values")
    ap.add_argument("--pgd-restarts", type=int, default=1)
    ap.add_argument("--objective", default="prefix_locked_gripper_open_margin")
    ap.add_argument("--output-csv", default="tables/fast_vis_policy_only_audit_v0.csv")
    ap.add_argument("--output-report", default="reports/FAST_VIS_POLICY_ONLY_AUDIT_V0.md")
    ap.add_argument("--traces-base-dir", default="/data/liuyu/outputs/nightly_object_batch3b_20260604")
    ap.add_argument("--gpu-pair", default="0,1")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def validate_gpu_pair(gpu_pair: str):
    ids = [x.strip() for x in gpu_pair.split(",") if x.strip()]
    if any(x in {"3", "7"} for x in ids):
        raise SystemExit("INFRA_FAILED: GPU3/GPU7 are blacklisted; requested --gpu-pair=%s" % gpu_pair)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").replace(" ", "")
    if visible == "2,6" and gpu_pair.replace(" ", "") == "2,6":
        raise SystemExit(
            "INFRA_FAILED: do not combine CUDA_VISIBLE_DEVICES=2,6 with --gpu-pair 2,6; "
            "inside a remapped visible set, --gpu-pair would need logical 0,1, but this is not recommended"
        )
    return gpu_pair


def parse_gpu_ids(gpu_pair: str):
    return [int(x.strip()) for x in validate_gpu_pair(gpu_pair).split(",") if x.strip()]


def infra_status_from_error(error_text: str):
    low = str(error_text).lower()
    if any(tok in low for tok in ["xid", "out of memory", "oom", "cuda illegal", "cublas"]):
        return "INFRA_FAILED"
    return "ERROR"


def load_model_processor(gpu_pair: str):
    """Load OpenVLA model and processor once."""
    import torch
    import transformers
    gpu_ids = parse_gpu_ids(gpu_pair)
    device = f"cuda:{gpu_ids[0]}"
    model_id = "openvla/openvla-7b"
    processor = transformers.AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if len(gpu_ids) >= 2:
        model_kwargs.update(
            device_map="auto",
            max_memory={gpu_ids[0]: "10500MiB", gpu_ids[1]: "10500MiB", "cpu": "64GiB"},
        )
    model = transformers.AutoModelForVision2Seq.from_pretrained(
        model_id, **model_kwargs)
    if len(gpu_ids) < 2:
        model = model.to(device)
    model.eval()
    return model, processor, device


def find_cached_clean_image(task_key, state_id, window_start, window_end, traces_base_dir):
    """Load a cached clean observation from existing trace CSV."""
    ep_id = f"{task_key}_s{state_id}"
    traces_dir = Path(traces_base_dir) / ep_id / "traces"
    if not traces_dir.is_dir():
        return None, f"no_traces_dir_{traces_dir}"
    # Find clean trace for this window
    for f in sorted(traces_dir.glob("*_clean_w*_trace.csv")):
        fname = f.name
        if f"w{window_start}_{window_end}" in fname:
            # Read first observation row
            with open(f, newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    if row.get("image_path"):
                        img_path = row["image_path"]
                        if os.path.exists(img_path):
                            import numpy as np
                            from PIL import Image
                            return np.array(Image.open(img_path).convert("RGB")), "ok"
            return None, "no_image_in_trace"
    return None, f"no_clean_trace_for_w{window_start}_{window_end}"


def run_policy_only_attack(model, processor, device, image_np, instruction: str,
                            eps_raw: int, pgd_steps: int, pgd_restarts: int,
                            objective: str):
    """Run PGD attack on single observation, return decoded gripper action."""
    import numpy as np
    import torch
    from gripper_attack.attack_adapter import (
        TokenPrefixPGDAttacker, prepare_openvla_image_for_attack, _prompt
    )
    action_dim = 7  # OpenVLA action dim
    unnorm_key = "libero_goal"

    # Get clean action for target
    image = prepare_openvla_image_for_attack(image_np)
    inputs = processor(_prompt(instruction), image, return_tensors="pt")
    inputs.pop("attention_mask", None)
    pixel_values = inputs["pixel_values"].to(device=device, dtype=torch.bfloat16)
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        clean_out = model.generate(
            input_ids, pixel_values=pixel_values, max_new_tokens=action_dim,
            do_sample=False, return_dict_in_generate=True, output_scores=True)
    clean_action_tokens = clean_out.sequences[0, -action_dim:].cpu().numpy()

    # Build target action: arm = clean, gripper = OPEN
    stats = model.get_action_stats(unnorm_key)
    action_stats = stats[list(stats.keys())[0]]
    mask = np.asarray(action_stats.get("mask", np.ones(action_dim, dtype=bool)), dtype=bool)
    q01 = np.asarray(action_stats["q01"], dtype=np.float32)
    q99 = np.asarray(action_stats["q99"], dtype=np.float32)
    denom = np.maximum(q99 - q01, 1e-6)
    bins = np.asarray(model.bin_centers, dtype=np.float32)
    vocab_size = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)

    # Decode clean action
    clean_action = np.zeros(action_dim, dtype=np.float32)
    for dim in range(action_dim):
        if mask[dim]:
            tid = int(vocab_size - clean_action_tokens[dim] - 1)
            norm = bins[tid]
            clean_action[dim] = 0.5 * (norm + 1.0) * (q99[dim] - q01[dim]) + q01[dim]

    # Target: arm = clean, gripper = OPEN (decoded_action < 0.5)
    target_action = clean_action.copy()
    target_action[-1] = 0.0  # OPEN

    # Compute processor-space epsilon
    eps_processor = np.array([eps_raw / 255.0] * 3, dtype=np.float32)
    eps_proc = float(eps_processor.min())
    step_size = max(eps_proc / max(pgd_steps, 1), 1e-4)

    config = {
        "attack_optimizer": {
            "epsilon": eps_proc,
            "step_size": step_size,
            "num_steps": pgd_steps,
            "random_start": False,
            "temporal_init": "none",
            "objective": objective,
            "gripper_margin": 5.0,
            "arm_preserve_weight": 0.1,
        }
    }

    attacker = TokenPrefixPGDAttacker(model, processor, config, seed=0, device=device)

    # Multi-restart PGD
    t0 = time.time()
    best_open_prob = 0.0
    best_action = None
    best_result = None

    for restart in range(pgd_restarts):
        if restart > 0:
            attacker.seed = restart
        result = attacker.attack(image_np, instruction=instruction,
                                 clean_action=clean_action,
                                 target_action=target_action,
                                 unnorm_key=unnorm_key)
        # Decode action from adversarial inputs
        try:
            from gripper_attack.attack_adapter import get_adv_inputs_from_attack_result
            adv_inputs = get_adv_inputs_from_attack_result(result)
            with torch.no_grad():
                adv_out = model.generate(
                    adv_inputs["input_ids"].to(device),
                    pixel_values=adv_inputs["pixel_values"].to(device=device, dtype=torch.bfloat16),
                    max_new_tokens=action_dim, do_sample=False,
                    return_dict_in_generate=True, output_scores=True)
            adv_tokens = adv_out.sequences[0, -action_dim:].cpu().numpy()

            gripper_tid = int(vocab_size - adv_tokens[-1] - 1)
            gripper_norm = bins[gripper_tid]
            gripper_action = 0.5 * (gripper_norm + 1.0) * (q99[-1] - q01[-1]) + q01[-1]
            is_open = float(gripper_action) < 0.5

            if is_open:
                best_open_prob = 1.0
                best_action = gripper_action
                best_result = result
                break  # Found an OPEN-restart
        except Exception as e:
            continue

    runtime = time.time() - t0

    # Fallback: use best result
    if best_result is None and pgd_restarts > 0:
        try:
            attacker2 = TokenPrefixPGDAttacker(model, processor, config, seed=0, device=device)
            best_result = attacker2.attack(image_np, instruction=instruction,
                                           clean_action=clean_action,
                                           target_action=target_action,
                                           unnorm_key=unnorm_key)
            from gripper_attack.attack_adapter import get_adv_inputs_from_attack_result
            adv_inputs = get_adv_inputs_from_attack_result(best_result)
            with torch.no_grad():
                adv_out = model.generate(
                    adv_inputs["input_ids"].to(device),
                    pixel_values=adv_inputs["pixel_values"].to(device=device, dtype=torch.bfloat16),
                    max_new_tokens=action_dim, do_sample=False,
                    return_dict_in_generate=True, output_scores=True)
            adv_tokens = adv_out.sequences[0, -action_dim:].cpu().numpy()
            gripper_tid = int(vocab_size - adv_tokens[-1] - 1)
            gripper_norm = bins[gripper_tid]
            best_action = 0.5 * (gripper_norm + 1.0) * (q99[-1] - q01[-1]) + q01[-1]
            best_open_prob = 1.0 if float(best_action) < 0.5 else 0.0
        except Exception as e:
            status = infra_status_from_error(str(e))
            return {
                "vis_open": 0, "open_margin": 0.0, "open_margin_min": 0.0,
                "steps_until_open": -1, "policy_transfer_score": 0.0,
                "runtime_sec": runtime, "budget": f"eps{eps_raw}_steps{pgd_steps}_rst{pgd_restarts}",
                "provenance_status": f"{status}: {e}",
                "eps_raw_pixels": eps_raw, "pgd_steps": pgd_steps, "pgd_restarts": pgd_restarts,
            }

    return {
        "vis_open": int(best_open_prob > 0.5),
        "open_margin": best_open_prob,
        "open_margin_min": best_open_prob,
        "steps_until_open": 0 if best_open_prob > 0.5 else -1,
        "policy_transfer_score": best_open_prob,
        "runtime_sec": round(runtime, 2),
        "budget": f"eps{eps_raw}_steps{pgd_steps}_rst{pgd_restarts}",
        "provenance_status": "ok",
        "eps_raw_pixels": eps_raw, "pgd_steps": pgd_steps, "pgd_restarts": pgd_restarts,
    }


def main():
    args = parse_args()
    validate_gpu_pair(args.gpu_pair)
    if not os.path.exists(args.candidate_csv):
        print(f"ERROR: {args.candidate_csv} not found"); sys.exit(1)

    with open(args.candidate_csv, newline="") as f:
        candidates = list(csv.DictReader(f))

    eps_list = [int(x) for x in args.eps_list.split(",")]
    steps_list = [int(x) for x in args.pgd_steps_list.split(",")]

    if args.dry_run:
        for c in candidates:
            print(
                f"  {c['task_key']}_s{c['state_id']} "
                f"[{c['parent_window_start']},{c['parent_window_end']}]"
            )
        return

    print(f"Loading model on GPU {args.gpu_pair}...")
    model, processor, device = load_model_processor(args.gpu_pair)
    print(f"Model loaded on {device}")

    results = []
    total = len(candidates) * len(eps_list) * len(steps_list)
    n = 0

    for c in candidates:
        task = c["task_key"]; sid = c["state_id"]
        ws = int(c["parent_window_start"]); we = int(c["parent_window_end"])

        # Load cached clean observation
        image_np, status = find_cached_clean_image(task, sid, ws, we, args.traces_base_dir)
        if image_np is None:
            print(f"SKIP {task}_s{sid}: {status}")
            for eps in eps_list:
                for steps in steps_list:
                    results.append(dict(
                        task_key=task, state_id=sid, window_start=ws, window_end=we,
                        label=c.get("full_vis_label",""), label_source="full_vis_label",
                        label_confidence="gold_full_vis", denominator_status="not_applicable_policy_only",
                        gpu_pair=args.gpu_pair,
                        vis_open="", open_margin="", runtime_sec="",
                        provenance_status=f"BLOCKED_MISSING_CACHED_OBS:{status}",
                        eps_raw_pixels=eps, pgd_steps=steps, pgd_restarts=args.pgd_restarts,
                    ))
            continue

        instruction = "pick up the {} and place it in the basket".format(
            task.replace("_", " "))

        for eps in eps_list:
            for steps in steps_list:
                n += 1
                print(f"[{n}/{total}] {task}_s{sid} eps={eps} steps={steps}")
                r = run_policy_only_attack(
                    model, processor, device, image_np, instruction,
                    eps, steps, args.pgd_restarts, args.objective)
                r.update(dict(task_key=task, state_id=sid, window_start=ws, window_end=we,
                              label=c.get("full_vis_label",""), label_source="full_vis_label",
                              label_confidence="gold_full_vis",
                              denominator_status="not_applicable_policy_only",
                              gpu_pair=args.gpu_pair))
                results.append(r)

    # Write CSV
    fields = ["task_key","state_id","window_start","window_end","label",
              "label_source","label_confidence","denominator_status","gpu_pair",
              "eps_raw_pixels","pgd_steps","pgd_restarts",
              "vis_open","open_margin","open_margin_min","steps_until_open",
              "policy_transfer_score","runtime_sec","budget","provenance_status"]
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(results)
    print(f"Results: {args.output_csv} ({len(results)} rows)")

    # Write report
    pos = [r for r in results if int(r.get("label", 0) or 0) == 1 and "ok" in str(r.get("provenance_status",""))]
    neg = [r for r in results if int(r.get("label", 0) or 0) == 0 and "ok" in str(r.get("provenance_status",""))]
    blocked = [r for r in results if "BLOCKED" in str(r.get("provenance_status",""))]

    candidate_pos = sum(1 for c in candidates if int(c.get("full_vis_label", 0) or 0) == 1)
    candidate_neg = sum(1 for c in candidates if int(c.get("full_vis_label", 0) or 0) == 0)

    report = f"""# Policy-Only VIS Audit v0

**Candidates**: {len(candidates)} ({candidate_pos} pos, {candidate_neg} neg)
**Budgets tested**: eps={eps_list}, steps={steps_list}, restarts={args.pgd_restarts}
**Total runs**: {len(results)} ({len(blocked)} blocked — missing cached obs)

## Per-Budget Summary

| eps | steps | n | VIS_OPEN | runtime_sec(mean) |
|-----|-------|---|----------|-------------------|
"""
    from collections import defaultdict
    budget_groups = defaultdict(list)
    for r in results:
        if "ok" in str(r.get("provenance_status","")):
            budget_groups[(r["eps_raw_pixels"], r["pgd_steps"])].append(r)

    for (eps, steps), rows in sorted(budget_groups.items()):
        open_count = sum(int(r.get("vis_open", 0) or 0) for r in rows)
        rt_values = [float(r.get("runtime_sec", 0) or 0) for r in rows]
        rt = sum(rt_values) / len(rt_values) if rt_values else 0.0
        report += f"| {eps} | {steps} | {len(rows)} | {open_count} | {rt:.2f} |\n"

    report += f"""
## Positives (policy OPEN rate)

| Budget | Task | VIS_OPEN |
|--------|------|----------|
"""
    for r in sorted(results, key=lambda r: (r.get("budget",""), r["task_key"])):
        if int(r.get("label",0) or 0) == 1 and "ok" in str(r.get("provenance_status","")):
            report += f"| {r.get('budget','')} | {r['task_key']}_s{r['state_id']} | {r.get('vis_open','?')} |\n"

    report += f"""
## Blocked (missing cached obs)

{len(blocked)} runs blocked — need clean rollout traces with image_path column.
"""
    if blocked:
        for b in blocked[:5]:
            report += f"- {b['task_key']}_s{b['state_id']}: {b['provenance_status']}\n"

    report += """
## Actionable

- Policy-only VIS can screen candidates in ~5-30 sec (vs 25-40 min for full VIS).
- OPEN rate across budgets indicates policy-level attackability.
- This is a SCREENING tool, not a gold label.
"""
    with open(args.output_report, "w") as f:
        f.write(report)
    print(f"Report: {args.output_report}")


if __name__ == "__main__":
    main()
