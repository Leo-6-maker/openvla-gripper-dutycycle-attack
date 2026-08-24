#!/usr/bin/env python3
"""Run held-out OOF predictions for one Factorized Student checkpoint.

Each checkpoint predicts ONLY its own fold's validation identities.
Outputs step-level predictions as sealed JSONL shard.
"""
import argparse, csv, hashlib, json, os, sys, uuid, platform
from pathlib import Path
from collections import defaultdict

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from gripper_attack.v5_factorized_dataset import (
    FactorizedEpisode, load_factorized_episodes,
    verify_factorized_source_roots, SUPPORTED_ROUTES,
)
from gripper_attack.v5_factorized_student import FactorizedStudent
from gripper_attack.b3_training_protocol import load_fit_fold_bundle, sha256_file, verify_sealed_directory


def _atomic_text(p, v):
    t = p.with_name(f".{p.name}.{uuid.uuid4().hex}.tmp")
    with t.open("x") as f: f.write(v); f.flush(); os.fsync(f.fileno())
    os.replace(t, p)


def write_seal(root):
    excl = {"SHA256SUMS", "SHA256SUMS.sha256"}
    fs = sorted((p for p in root.rglob("*") if p.is_file() and p.name not in excl),
                key=lambda p: p.relative_to(root).as_posix())
    c = "".join(f"{sha256_file(p)}  {p.relative_to(root).as_posix()}\n" for p in fs)
    _atomic_text(root / "SHA256SUMS", c)
    _atomic_text(root / "SHA256SUMS.sha256", f"{sha256_file(root / 'SHA256SUMS')}  SHA256SUMS\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", type=Path, required=True,
                    help="Path to foldX_seedY output directory (contains checkpoint.pt, normalization.json, etc.)")
    ap.add_argument("--s1-root", type=Path, required=True)
    ap.add_argument("--teacher-root", type=Path, required=True)
    ap.add_argument("--fold-root", type=Path, required=True)
    ap.add_argument("--policy-intent-root", type=Path, default=None)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--registry", type=Path, required=True)
    args = ap.parse_args()

    ckpt_dir = args.checkpoint_dir.resolve()
    out = args.output_root.resolve()

    # Verify checkpoint seal
    verify_sealed_directory(ckpt_dir)

    # Load checkpoint metadata
    src_binding = json.loads((ckpt_dir / "source_binding.json").read_text())
    model_type = src_binding["model_type"]
    fold_id = src_binding["fold_id"]
    seed = src_binding["seed"]
    use_9d = (model_type == "25D9D")

    if use_9d and args.policy_intent_root is None:
        raise SystemExit("25D9D requires --policy-intent-root")

    # Load normalization
    norm = json.loads((ckpt_dir / "normalization.json").read_text())
    mean_25d = torch.tensor(norm["mean_25d"])
    std_25d = torch.tensor(norm["std_25d"])
    mean_9d = std_9d = None
    if use_9d:
        mean_9d = torch.tensor(norm["mean_9d"])
        std_9d = torch.tensor(norm["std_9d"])

    # Verify source roots
    verify_factorized_source_roots(args.s1_root, args.teacher_root)

    # Load policy intent if 9D
    policy_index = None
    if use_9d:
        verify_sealed_directory(args.policy_intent_root.resolve())
        from gripper_attack.v5_dataset import load_policy_intent_root
        policy_index, _ = load_policy_intent_root(args.policy_intent_root.resolve())

    # Load fold and get val identities
    folds = load_fit_fold_bundle(args.fold_root.resolve())
    fold = [f for f in folds["folds"] if f["fold_id"] == fold_id][0]
    val_ids = set(fold["validation_identities"])

    # Load registry and filter to val identities
    reg = args.registry.resolve()
    rows = list(csv.DictReader(open(reg)))
    fit = [r for r in rows if r.get("split") == "FIT_TRAIN"]
    val_rows = [r for r in fit if r["canonical_parent_key"] in val_ids]

    # Load episodes
    val_eps = load_factorized_episodes(args.s1_root, args.teacher_root, val_rows,
                                        policy_index=policy_index)

    # Build identity -> episode map for later use
    ep_map = {ep.canonical_parent_key: ep for ep in val_eps}

    # Device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(ckpt_dir / "checkpoint.pt", map_location=device)
    model = FactorizedStudent(use_9d=use_9d).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Prepare output
    if out.exists():
        raise SystemExit(f"output exists: {out}")
    staging = out.with_name(f".{out.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    env_info = {
        "python": sys.executable, "python_version": platform.python_version(),
        "torch": torch.__version__, "cuda": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "host": platform.node(),
    }

    # Run predictions
    step_lines = []
    event_lines = []
    total_steps = 0
    supported_steps = 0
    unsupported_episodes = 0

    with torch.no_grad():
        for ep in val_eps:
            T = len(ep.features_25d)
            route = ep.mechanism_route
            route_sup = ep.route_supported

            if not route_sup:
                unsupported_episodes += 1
                # Still write step records with zero probs
                for t in range(T):
                    step_lines.append(json.dumps({
                        "model_type": model_type, "fold_id": fold_id, "seed": seed,
                        "canonical_parent_key": ep.canonical_parent_key,
                        "suite": ep.suite, "task_idx": ep.task_idx, "state_id": ep.state_id,
                        "mechanism_route": route, "route_supported": False,
                        "step_index": t, "event_id": int(ep.event_id[t].item()),
                        "event_ordinal": -1, "is_later_event": False,
                        "event_role": ep.event_role[t],
                        "active_object_name": ep.active_object_name[t],
                        "grasp_logit": -1e4, "grasp_prob": 0.0,
                        "manipulation_logit": -1e4, "manipulation_prob": 0.0,
                        "release_logit": -1e4, "release_prob": 0.0,
                        "grasp_target": bool(ep.grasp_target[t].item()),
                        "grasp_known_mask": bool(ep.grasp_known_mask[t].item()),
                        "manipulation_target": bool(ep.manipulation_target[t].item()),
                        "manipulation_known_mask": bool(ep.manipulation_known_mask[t].item()),
                        "release_target": bool(ep.release_target[t].item()),
                        "release_known_mask": bool(ep.release_known_mask[t].item()),
                        "k10_feasible": bool(ep.k10_feasible[t].item()),
                        "k10_known_mask": bool(ep.k10_known_mask[t].item()),
                    }) + "\n")
                continue

            # Prepare tensors [1, T, *]
            x25 = ((ep.features_25d - mean_25d) / std_25d).unsqueeze(0).to(device)
            m25 = ep.valid_mask.unsqueeze(0).to(device)
            x9 = m9 = None
            if use_9d and ep.policy_intent_9d.numel() > 0:
                x9 = ((ep.policy_intent_9d - mean_9d) / std_9d).unsqueeze(0).to(device)
                m9 = ep.policy_intent_valid_mask.unsqueeze(0).to(device)

            # Forward sequence
            probs = model.forward_sequence(x25, x9, m25, m9, route)

            g_prob = probs["grasp"][0].cpu()     # [T]
            m_prob = probs["manipulation"][0].cpu()
            r_prob = probs["release"][0].cpu()

            # Also get logits for audit
            logits = model.forward_logits(x25, x9, m25, m9, route)
            g_logit = logits["grasp"][0].cpu()
            m_logit = logits["manipulation"][0].cpu()
            r_logit = logits["release"][0].cpu()

            # Build event ordinal map
            eids = ep.event_id
            unique_events = sorted([e.item() for e in eids.unique() if e.item() >= 0])
            eid_to_ordinal = {e: i for i, e in enumerate(unique_events)}
            is_later = {e: (i >= 1) for e, i in eid_to_ordinal.items()}

            supported_steps += T
            total_steps += T

            for t in range(T):
                ev = int(eids[t].item())
                step_lines.append(json.dumps({
                    "model_type": model_type, "fold_id": fold_id, "seed": seed,
                    "canonical_parent_key": ep.canonical_parent_key,
                    "suite": ep.suite, "task_idx": ep.task_idx, "state_id": ep.state_id,
                    "mechanism_route": route, "route_supported": True,
                    "step_index": t, "event_id": ev,
                    "event_ordinal": eid_to_ordinal.get(ev, -1),
                    "is_later_event": is_later.get(ev, False),
                    "event_role": ep.event_role[t],
                    "active_object_name": ep.active_object_name[t],
                    "grasp_logit": round(float(g_logit[t].item()), 8),
                    "grasp_prob": round(float(g_prob[t].item()), 8),
                    "manipulation_logit": round(float(m_logit[t].item()), 8),
                    "manipulation_prob": round(float(m_prob[t].item()), 8),
                    "release_logit": round(float(r_logit[t].item()), 8),
                    "release_prob": round(float(r_prob[t].item()), 8),
                    "grasp_target": bool(ep.grasp_target[t].item()),
                    "grasp_known_mask": bool(ep.grasp_known_mask[t].item()),
                    "manipulation_target": bool(ep.manipulation_target[t].item()),
                    "manipulation_known_mask": bool(ep.manipulation_known_mask[t].item()),
                    "release_target": bool(ep.release_target[t].item()),
                    "release_known_mask": bool(ep.release_known_mask[t].item()),
                    "k10_feasible": bool(ep.k10_feasible[t].item()),
                    "k10_known_mask": bool(ep.k10_known_mask[t].item()),
                }) + "\n")

        # ── Event-level aggregation ──
        # Group step predictions by (identity, event_id)
        step_records = [json.loads(l) for l in step_lines]
        event_groups = defaultdict(list)
        for rec in step_records:
            key = (rec["canonical_parent_key"], rec["event_id"])
            event_groups[key].append(rec)

        for (identity, eid), steps in event_groups.items():
            if eid < 0:
                continue  # IDLE/background, skip for event metrics
            ep = ep_map.get(identity)
            if ep is None:
                continue

            # Known mask for this event
            t_indices = [s["step_index"] for s in steps]
            g_km = [ep.grasp_known_mask[ti].item() for ti in t_indices]
            m_km = [ep.manipulation_known_mask[ti].item() for ti in t_indices]
            r_km = [ep.release_known_mask[ti].item() for ti in t_indices]

            # Aggregate: max probability within known mask for this event
            g_probs_in_event = [s["grasp_prob"] for i, s in enumerate(steps) if g_km[i]]
            m_probs_in_event = [s["manipulation_prob"] for i, s in enumerate(steps) if m_km[i]]
            r_probs_in_event = [s["release_prob"] for i, s in enumerate(steps) if r_km[i]]

            g_max = max(g_probs_in_event) if g_probs_in_event else 0.0
            m_max = max(m_probs_in_event) if m_probs_in_event else 0.0
            r_max = max(r_probs_in_event) if r_probs_in_event else 0.0

            g_mean = sum(g_probs_in_event) / len(g_probs_in_event) if g_probs_in_event else 0.0
            m_mean = sum(m_probs_in_event) / len(m_probs_in_event) if m_probs_in_event else 0.0
            r_mean = sum(r_probs_in_event) / len(r_probs_in_event) if r_probs_in_event else 0.0

            # Teacher targets for this event (any step positive → event positive)
            g_target = any(s["grasp_target"] and s["grasp_known_mask"] for s in steps)
            m_target = any(s["manipulation_target"] and s["manipulation_known_mask"] for s in steps)
            r_target = any(s["release_target"] and s["release_known_mask"] for s in steps)

            route = steps[0]["mechanism_route"]
            event_ordinal = steps[0]["event_ordinal"]
            is_later = steps[0]["is_later_event"]

            event_lines.append(json.dumps({
                "model_type": model_type, "fold_id": fold_id, "seed": seed,
                "canonical_parent_key": identity,
                "mechanism_route": route, "route_supported": True,
                "event_id": eid, "event_ordinal": event_ordinal,
                "is_later_event": is_later,
                "grasp_score_max": round(g_max, 8),
                "grasp_score_mean": round(g_mean, 8),
                "manipulation_score_max": round(m_max, 8),
                "manipulation_score_mean": round(m_mean, 8),
                "release_score_max": round(r_max, 8),
                "release_score_mean": round(r_mean, 8),
                "grasp_emit": g_max >= 0.5,
                "manipulation_emit": m_max >= 0.5,
                "release_emit": r_max >= 0.5,
                "grasp_target": g_target,
                "manipulation_target": m_target,
                "release_target": r_target,
                "steps_in_event": len(steps),
                "known_steps_grasp": sum(g_km),
                "known_steps_manipulation": sum(m_km),
                "known_steps_release": sum(r_km),
            }) + "\n")

    # ── Manifest ──
    manifest = {
        "model_type": model_type, "fold_id": fold_id, "seed": seed,
        "checkpoint_dir": str(ckpt_dir),
        "total_steps": total_steps,
        "supported_steps": supported_steps,
        "unsupported_episodes": unsupported_episodes,
        "total_episodes": len(val_eps),
        "total_events": len(event_lines),
        "step_predictions_file": "heldout_step_predictions.jsonl",
        "event_predictions_file": "heldout_event_predictions.jsonl",
    }

    # ── Write outputs ──
    _atomic_text(staging / "heldout_step_predictions.jsonl", "".join(step_lines))
    _atomic_text(staging / "heldout_event_predictions.jsonl", "".join(event_lines))
    _atomic_text(staging / "prediction_manifest.json", json.dumps(manifest, indent=2))
    _atomic_text(staging / "source_binding.json", json.dumps({
        "checkpoint_dir": str(ckpt_dir),
        "checkpoint_seal": sha256_file(ckpt_dir / "SHA256SUMS"),
        "s1_root": str(args.s1_root),
        "teacher_root": str(args.teacher_root),
        "fold_root": str(args.fold_root),
        "policy_intent_root": str(args.policy_intent_root) if args.policy_intent_root else None,
        "model_type": model_type, "fold_id": fold_id, "seed": seed,
    }, indent=2))
    _atomic_text(staging / "environment.json", json.dumps(env_info, indent=2))
    write_seal(staging)
    os.replace(staging, out)

    print(f"Prediction complete: {out}")
    print(f"  Episodes: {len(val_eps)} (unsupported: {unsupported_episodes})")
    print(f"  Steps: {total_steps} (supported: {supported_steps})")
    print(f"  Events: {len(event_lines)}")


if __name__ == "__main__":
    main()
