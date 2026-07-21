#!/usr/bin/env python3
"""Held-out prediction runner for the recommended exact-W32 sidecar Student."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import uuid
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.b3_training_protocol import sha256_file, verify_sealed_directory
from gripper_attack.v5_factorized_dataset import (
    load_factorized_episodes,
    verify_factorized_source_roots,
)
from gripper_attack.v5_factorized_student_v2_recommended import (
    RecommendedFactorizedStudentV2,
)
from gripper_attack.v5_factorized_v2_splits import resolve_inner_train_val_ids

OPS = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops")
S1 = OPS / "OFFICIAL_V3_S1_FIT_V1_d31187f"
TEACHER = OPS / "OFFICIAL_V3_DETECTOR_V5_FACTORIZED_TEACHER_V1_de07e1a_20260721"
REGISTRY = OPS / "OFFICIAL_V3_CAMPAIGN_REGISTRY_V1_d31187f/OFFICIAL_V3_FORMAL_REGISTRY_V1.csv"


def _atomic_text(path: Path, value: str) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("x") as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _write_seal(root: Path) -> None:
    excluded = {"SHA256SUMS", "SHA256SUMS.sha256"}
    files = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.name not in excluded),
        key=lambda p: p.relative_to(root).as_posix(),
    )
    content = "".join(
        f"{sha256_file(p)}  {p.relative_to(root).as_posix()}\n" for p in files
    )
    _atomic_text(root / "SHA256SUMS", content)
    _atomic_text(
        root / "SHA256SUMS.sha256",
        f"{sha256_file(root / 'SHA256SUMS')}  SHA256SUMS\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--inner-cv-splits-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    ckpt_dir = args.checkpoint_dir.resolve()
    output = args.output_root.resolve()
    verify_sealed_directory(ckpt_dir)
    verify_sealed_directory(args.inner_cv_splits_root)

    run_config = json.loads((ckpt_dir / "run_config.json").read_text())
    norm = json.loads((ckpt_dir / "normalization.json").read_text())
    candidate = run_config["candidate"]
    outer_fold = int(run_config["outer_fold"])
    inner_fold = int(run_config["inner_fold"])
    seed = int(run_config["seed"])
    hidden_dim = int(run_config["hidden_dim"])
    context_steps = int(run_config["context_steps"])
    dropout = float(run_config["dropout"])

    splits = json.loads((args.inner_cv_splits_root / "inner_cv_splits.json").read_text())
    _, inner_val_ids = resolve_inner_train_val_ids(splits, outer_fold, inner_fold)

    verify_factorized_source_roots(S1, TEACHER)
    rows = list(csv.DictReader(REGISTRY.open()))
    fit_rows = [row for row in rows if row.get("split") == "FIT_TRAIN"]
    id_to_row = {row["canonical_parent_key"]: row for row in fit_rows}
    val_rows = [id_to_row[i] for i in inner_val_ids if i in id_to_row]
    val_eps = load_factorized_episodes(S1, TEACHER, val_rows)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    checkpoint = torch.load(ckpt_dir / "checkpoint.pt", map_location=device)
    model = RecommendedFactorizedStudentV2(
        input_dim_25d=25,
        hidden_dim=hidden_dim,
        context_steps=context_steps,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    if output.exists():
        raise SystemExit(f"OUTPUT EXISTS: {output}")
    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    mean_25d = torch.tensor(norm["mean_25d"])
    std_25d = torch.tensor(norm["std_25d"])
    step_records: list[dict] = []

    with torch.no_grad():
        for episode in val_eps:
            T = len(episode.features_25d)
            route = episode.mechanism_route
            event_ids = episode.event_id
            event_duration = defaultdict(int)
            release_positive_duration = defaultdict(int)
            for t in range(T):
                eid = int(event_ids[t].item())
                if eid >= 0:
                    event_duration[eid] += 1
                    if episode.release_target[t].item() and episode.release_known_mask[t].item():
                        release_positive_duration[eid] += 1
            unique_events = sorted(event_duration)
            ordinal = {eid: i for i, eid in enumerate(unique_events)}

            if episode.route_supported:
                x25 = ((episode.features_25d - mean_25d) / std_25d).unsqueeze(0).to(device)
                mask25 = episode.valid_mask.unsqueeze(0).to(device)
                probs = model.forward_sequence(x25, None, mask25, None, route)
                logits = model.forward_logits(x25, None, mask25, None, route)
                g_prob = probs["grasp"][0].cpu()
                m_prob = probs["manipulation"][0].cpu()
                r_prob = probs["release"][0].cpu()
                g_logit = logits["grasp"][0].cpu()
                m_logit = logits["manipulation"][0].cpu()
                r_logit = logits["release"][0].cpu()
            else:
                g_prob = torch.zeros(T)
                m_prob = torch.zeros(T)
                r_prob = torch.zeros(T)
                g_logit = torch.full((T,), -1e4)
                m_logit = torch.full((T,), -1e4)
                r_logit = torch.full((T,), -1e4)

            for t in range(T):
                eid = int(event_ids[t].item())
                record = {
                    "candidate_id": candidate,
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "seed": seed,
                    "canonical_parent_key": episode.canonical_parent_key,
                    "suite": episode.suite,
                    "task_idx": episode.task_idx,
                    "state_id": episode.state_id,
                    "mechanism_route": route,
                    "route_supported": bool(episode.route_supported),
                    "step_index": t,
                    "event_id": eid,
                    "event_ordinal": ordinal.get(eid, -1),
                    "is_later_event": ordinal.get(eid, -1) >= 1,
                    "event_role": episode.event_role[t],
                    "event_duration": event_duration.get(eid, 0),
                    "release_positive_duration": release_positive_duration.get(eid, 0),
                    "window_id": t // context_steps,
                    "position_in_window": t % context_steps,
                    "encoder_type": "exact_tcn",
                    "window_size": context_steps,
                    "grasp_prob": round(float(g_prob[t].item()), 8),
                    "manipulation_prob": round(float(m_prob[t].item()), 8),
                    "release_prob": round(float(r_prob[t].item()), 8),
                    "grasp_logit": round(float(g_logit[t].item()), 8),
                    "manipulation_logit": round(float(m_logit[t].item()), 8),
                    "release_logit": round(float(r_logit[t].item()), 8),
                    "grasp_target": bool(episode.grasp_target[t].item()),
                    "grasp_known_mask": bool(episode.grasp_known_mask[t].item()),
                    "manipulation_target": bool(episode.manipulation_target[t].item()),
                    "manipulation_known_mask": bool(episode.manipulation_known_mask[t].item()),
                    "release_target": bool(episode.release_target[t].item()),
                    "release_known_mask": bool(episode.release_known_mask[t].item()),
                }
                step_records.append(record)

    event_groups = defaultdict(list)
    for record in step_records:
        if record["event_id"] >= 0 and record["route_supported"]:
            event_groups[(record["canonical_parent_key"], record["event_id"])].append(record)

    event_records: list[dict] = []
    for (identity, eid), records in event_groups.items():
        first = records[0]
        g_known = [r["grasp_known_mask"] for r in records]
        m_known = [r["manipulation_known_mask"] for r in records]
        r_known = [r["release_known_mask"] for r in records]
        g_probs = [r["grasp_prob"] for i, r in enumerate(records) if g_known[i]]
        m_probs = [r["manipulation_prob"] for i, r in enumerate(records) if m_known[i]]
        r_probs = [r["release_prob"] for i, r in enumerate(records) if r_known[i]]

        def first_crossing(values, threshold=0.5):
            for index, value in enumerate(values):
                if value >= threshold:
                    return index
            return -1

        def coverage(values, threshold=0.5):
            return sum(value >= threshold for value in values) / max(1, len(values))

        event_records.append({
            "candidate_id": candidate,
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "seed": seed,
            "canonical_parent_key": identity,
            "event_id": eid,
            "mechanism_route": first["mechanism_route"],
            "event_ordinal": first["event_ordinal"],
            "is_later_event": first["is_later_event"],
            "event_duration": first["event_duration"],
            "release_positive_duration": first["release_positive_duration"],
            "grasp_event_score": round(max(g_probs) if g_probs else 0.0, 8),
            "manipulation_event_score": round(max(m_probs) if m_probs else 0.0, 8),
            "release_event_score": round(max(r_probs) if r_probs else 0.0, 8),
            "grasp_target": any(r["grasp_target"] and r["grasp_known_mask"] for r in records),
            "manipulation_target": any(r["manipulation_target"] and r["manipulation_known_mask"] for r in records),
            "release_target": any(r["release_target"] and r["release_known_mask"] for r in records),
            "grasp_known_steps": sum(g_known),
            "manipulation_known_steps": sum(m_known),
            "release_known_steps": sum(r_known),
            "grasp_first_crossing_05": first_crossing(g_probs),
            "manipulation_first_crossing_05": first_crossing(m_probs),
            "release_first_crossing_05": first_crossing(r_probs),
            "grasp_coverage_05": round(coverage(g_probs), 8),
            "manipulation_coverage_05": round(coverage(m_probs), 8),
            "release_coverage_05": round(coverage(r_probs), 8),
        })

    _atomic_text(
        staging / "heldout_step_predictions.jsonl",
        "".join(json.dumps(record) + "\n" for record in step_records),
    )
    _atomic_text(
        staging / "heldout_event_predictions.jsonl",
        "".join(json.dumps(record) + "\n" for record in event_records),
    )
    _atomic_text(staging / "prediction_manifest.json", json.dumps({
        "candidate": candidate,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "seed": seed,
        "total_steps": len(step_records),
        "total_events": len(event_records),
        "total_episodes": len(val_eps),
        "formal_selection_eligible": False,
    }, indent=2))
    _atomic_text(staging / "source_binding.json", json.dumps({
        "checkpoint_dir": str(ckpt_dir),
        "checkpoint_seal": sha256_file(ckpt_dir / "SHA256SUMS"),
        "inner_cv_splits_root": str(args.inner_cv_splits_root.resolve()),
        "candidate": candidate,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "seed": seed,
        "formal_selection_eligible": False,
    }, indent=2))
    _atomic_text(staging / "environment.json", json.dumps({
        "python_version": platform.python_version(),
        "torch": torch.__version__,
        "host": platform.node(),
    }, indent=2))
    _write_seal(staging)
    os.replace(staging, output)
    print(
        f"Prediction sealed: {output} | steps={len(step_records)} "
        f"events={len(event_records)} eps={len(val_eps)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
