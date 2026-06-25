#!/usr/bin/env python3
"""Release trajectory counterfactual: quantify how V2 shared trunk affects release
decisions vs full V1 path. Tests whether V1-injected release head on V2 trunk
produces different emit decisions than on V1 trunk.

Gate: emit disagreement from release differences <= 1/90.
Also measures: release-block disagreement (armed in one but release blocks in other).

Approach: Load both V1 and V2 runtimes from checkpoints, replay each trajectory
through both. At each step, extract raw logits to determine which head (phase,
corridor, or release) caused any arming/emit disagreement.
"""
import csv, json, math, os, sys, numpy as np, torch
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from gripper_attack.sc5_detector_runtime import SC5DetectorRuntime, SC5_FEATURES

TAU_C = 0.3; TAU_R = 0.3; GUARD = 5


def extract_probs(model, x_norm):
    """Get phase/ corridor/ release probabilities from model given normalized input."""
    with torch.no_grad():
        out = model(torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0))
        phase_idx = int(out["phase_logits"].argmax(1).item())
        cp = float(torch.sigmoid(out["corridor_logit"]).item())
        rp = float(torch.sigmoid(out["release_logit"]).item())
        return phase_idx, cp, rp


def counterfactual_replay(rt_v1, rt_v2, rows):
    """Replay one trajectory through both V1 and V2 runtimes.
    At each step, also compute raw probs to attribute any disagreement.
    """
    rt_v1.reset(); rt_v2.reset()

    release_block_events = []
    per_step_deltas = []

    for r in rows:
        feats = {}
        ok = True
        for fn in SC5_FEATURES:
            val = r.get(fn, "")
            if val in ("", "nan", "NaN", None):
                ok = False; break
            try:
                feats[fn] = float(val)
            except (ValueError, TypeError):
                ok = False; break
        if not ok:
            continue

        x = np.array([feats[fn] for fn in SC5_FEATURES], dtype=np.float32)
        if not np.all(np.isfinite(x)):
            continue

        step = int(r.get("step_idx", 0))

        # Raw probs from both models (before state machine update)
        x_v1 = (x - rt_v1.mean) / (rt_v1.std + 1e-8)
        x_v2 = (x - rt_v2.mean) / (rt_v2.std + 1e-8)
        ph_v1, cp_v1, rp_v1 = extract_probs(rt_v1.model, x_v1)
        ph_v2, cp_v2, rp_v2 = extract_probs(rt_v2.model, x_v2)

        pre_v1_armed = rt_v1.state == "ARMED"
        pre_v2_armed = rt_v2.state == "ARMED"

        feat_dict = {fn: float(x[i]) for i, fn in enumerate(SC5_FEATURES)}
        dec_v1 = rt_v1.update(feat_dict, step)
        dec_v2 = rt_v2.update(feat_dict, step)

        post_v1_armed = rt_v1.state == "ARMED"
        post_v2_armed = rt_v2.state == "ARMED"

        # Detect arming disagreement and attribute cause
        if post_v1_armed and not post_v2_armed:
            release_block_events.append({
                "step": step, "blocked_model": "V2",
                "v1_release_p": round(rp_v1, 6), "v2_release_p": round(rp_v2, 6),
                "v1_corridor_p": round(cp_v1, 6), "v2_corridor_p": round(cp_v2, 6),
                "v1_phase": ph_v1, "v2_phase": ph_v2,
                "release_cross": (rp_v1 >= TAU_R) != (rp_v2 >= TAU_R),
                "corridor_cross": (cp_v1 >= TAU_C) != (cp_v2 >= TAU_C),
                "phase_disagree": ph_v1 != ph_v2,
            })
        elif post_v2_armed and not post_v1_armed:
            release_block_events.append({
                "step": step, "blocked_model": "V1",
                "v1_release_p": round(rp_v1, 6), "v2_release_p": round(rp_v2, 6),
                "v1_corridor_p": round(cp_v1, 6), "v2_corridor_p": round(cp_v2, 6),
                "v1_phase": ph_v1, "v2_phase": ph_v2,
                "release_cross": (rp_v1 >= TAU_R) != (rp_v2 >= TAU_R),
                "corridor_cross": (cp_v1 >= TAU_C) != (cp_v2 >= TAU_C),
                "phase_disagree": ph_v1 != ph_v2,
            })

        per_step_deltas.append({
            "step": step,
            "rp_v1": rp_v1, "rp_v2": rp_v2, "rp_delta": rp_v2 - rp_v1,
            "cp_v1": cp_v1, "cp_v2": cp_v2, "cp_delta": cp_v2 - cp_v1,
            "phase_agree": ph_v1 == ph_v2,
        })

        if dec_v1.get("emitted") or dec_v2.get("emitted"):
            break

    rp_deltas = [d["rp_delta"] for d in per_step_deltas]
    cp_deltas = [d["cp_delta"] for d in per_step_deltas]

    return {
        "v1_armed": rt_v1.state == "ARMED", "v1_emitted": rt_v1.emitted,
        "v2_armed": rt_v2.state == "ARMED", "v2_emitted": rt_v2.emitted,
        "v1_emit_step": rt_v1.emit_step, "v2_emit_step": rt_v2.emit_step,
        "release_block_events": release_block_events,
        "n_release_block_events": len(release_block_events),
        "rp_mae": float(np.mean(np.abs(rp_deltas))) if rp_deltas else 0,
        "rp_max_abs": float(np.max(np.abs(rp_deltas))) if rp_deltas else 0,
        "cp_mae": float(np.mean(np.abs(cp_deltas))) if cp_deltas else 0,
        "phase_disagree_count": sum(1 for d in per_step_deltas if not d["phase_agree"]),
        "phase_disagree_frac": sum(1 for d in per_step_deltas if not d["phase_agree"]) / max(len(per_step_deltas), 1),
        "n_steps": len(per_step_deltas),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1_checkpoint", required=True)
    ap.add_argument("--v2_checkpoint", required=True)
    ap.add_argument("--dataset_csv", required=True)
    ap.add_argument("--dev_labels_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--episode_ids", nargs="*", help="Specific episodes (default: all dev)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading runtimes...")
    rt_v1 = SC5DetectorRuntime(args.v1_checkpoint, tau_corridor=TAU_C, tau_release=TAU_R, guard=GUARD)
    rt_v2 = SC5DetectorRuntime(args.v2_checkpoint, tau_corridor=TAU_C, tau_release=TAU_R, guard=GUARD)
    print(f"  V1: {rt_v1.checkpoint_sha256[:16]}")
    print(f"  V2: {rt_v2.checkpoint_sha256[:16]}")

    print("Loading dev labels...")
    dev_labels = {}
    for lr in csv.DictReader(open(args.dev_labels_csv)):
        key = (int(lr["task"]), int(lr["state"]), lr["source"])
        dev_labels[key] = lr
    print(f"  {len(dev_labels)} dev labels")

    print("Loading step dataset...")
    all_rows = list(csv.DictReader(open(args.dataset_csv)))
    val_episodes = defaultdict(list)
    for r in all_rows:
        if r["split"] != "val":
            continue
        val_episodes[r["episode_id"]].append(r)
    print(f"  {len(val_episodes)} val episodes")

    # Filter to dev episodes
    dev_episodes = {}
    for eid, rows in val_episodes.items():
        task = int(rows[0]["task_idx"])
        state = int(rows[0]["parent_state_id"])
        source = rows[0]["source_pool"]
        key = (task, state, source)
        if key in dev_labels:
            dev_episodes[eid] = rows
    print(f"  {len(dev_episodes)} dev episodes with labels")

    if args.episode_ids:
        dev_episodes = {eid: rows for eid, rows in dev_episodes.items() if eid in args.episode_ids}
        print(f"  Filtered to {len(dev_episodes)} specified episodes")

    results = {}
    emit_disagree_eps = []
    release_block_eps = []

    for eid in sorted(dev_episodes.keys()):
        rows = dev_episodes[eid]
        cf = counterfactual_replay(rt_v1, rt_v2, rows)
        results[eid] = cf

        if cf["v1_emitted"] != cf["v2_emitted"]:
            emit_disagree_eps.append(eid)
        if cf["n_release_block_events"] > 0:
            release_block_eps.append(eid)

    # ── Summary ──
    n_emit_disagree = len(emit_disagree_eps)
    n_release_attributable = sum(
        1 for eid in emit_disagree_eps
        if any(ev.get("release_cross") for ev in results[eid]["release_block_events"])
    )

    summary = {
        "gate": "RELEASE_TRAJECTORY_COUNTERFACTUAL",
        "v1_checkpoint_sha256": rt_v1.checkpoint_sha256,
        "v2_checkpoint_sha256": rt_v2.checkpoint_sha256,
        "total_episodes": len(dev_episodes),
        "emit_disagreement": {
            "count": n_emit_disagree,
            "gate": "<= 1/90",
            "pass": n_emit_disagree <= 1,
            "release_attributable": n_release_attributable,
            "episodes": emit_disagree_eps,
        },
        "release_block": {
            "episodes_with_events": len(release_block_eps),
            "total_events": sum(results[eid]["n_release_block_events"] for eid in release_block_eps),
            "release_cross_events": sum(
                1 for eid in release_block_eps
                for ev in results[eid]["release_block_events"]
                if ev.get("release_cross")
            ),
        },
        "aggregate_metrics": {
            "rp_mae_mean": float(np.mean([r["rp_mae"] for r in results.values()])),
            "rp_max_abs_max": float(np.max([r["rp_max_abs"] for r in results.values()])),
            "cp_mae_mean": float(np.mean([r["cp_mae"] for r in results.values()])),
            "phase_disagree_frac_mean": float(np.mean([r["phase_disagree_frac"] for r in results.values()])),
        },
    }

    json_path = os.path.join(args.output_dir, "release_trajectory_counterfactual.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")

    print(f"\n=== RELEASE TRAJECTORY COUNTERFACTUAL ===")
    print(f"  Episodes evaluated: {len(dev_episodes)}")
    print(f"  Emit disagreement: {n_emit_disagree}/90")
    print(f"  Release-attributable emit disagreements: {n_release_attributable}")
    print(f"  Release-block events (arming level): {summary['release_block']['total_events']}")
    print(f"  Release-cross events: {summary['release_block']['release_cross_events']}")
    print(f"  RP MAE (mean): {summary['aggregate_metrics']['rp_mae_mean']:.6f}")
    print(f"  GATE (emit_disagree <= 1): {'PASS' if n_emit_disagree <= 1 else 'FAIL'}")

    # ── Per-episode CSV ──
    csv_path = os.path.join(args.output_dir, "release_counterfactual_per_episode.csv")
    fields = ["episode_id", "v1_armed", "v1_emitted", "v1_emit_step",
              "v2_armed", "v2_emitted", "v2_emit_step",
              "emit_agree", "n_release_block_events", "rp_mae", "rp_max_abs",
              "phase_disagree_frac", "n_steps"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for eid in sorted(results.keys()):
            r = results[eid]
            w.writerow({
                "episode_id": eid,
                "v1_armed": r["v1_armed"], "v1_emitted": r["v1_emitted"], "v1_emit_step": r["v1_emit_step"],
                "v2_armed": r["v2_armed"], "v2_emitted": r["v2_emitted"], "v2_emit_step": r["v2_emit_step"],
                "emit_agree": r["v1_emitted"] == r["v2_emitted"],
                "n_release_block_events": r["n_release_block_events"],
                "rp_mae": round(r["rp_mae"], 6),
                "rp_max_abs": round(r["rp_max_abs"], 6),
                "phase_disagree_frac": round(r["phase_disagree_frac"], 6),
                "n_steps": r["n_steps"],
            })

    # Also save detailed block events for episodes with them
    if release_block_eps:
        block_csv = os.path.join(args.output_dir, "release_block_events_detail.csv")
        block_fields = ["episode_id", "step", "blocked_model",
                        "v1_release_p", "v2_release_p", "v1_corridor_p", "v2_corridor_p",
                        "v1_phase", "v2_phase", "release_cross", "corridor_cross", "phase_disagree"]
        with open(block_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=block_fields)
            w.writeheader()
            for eid in release_block_eps:
                for ev in results[eid]["release_block_events"]:
                    ev["episode_id"] = eid
                    w.writerow(ev)
        print(f"Saved block events detail: {block_csv}")

    print(f"Saved per-episode CSV: {csv_path}")


if __name__ == "__main__":
    main()
