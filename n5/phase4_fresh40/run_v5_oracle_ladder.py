#!/usr/bin/env python3
"""Development-only event oracle ladder for the sealed Fresh40 dev split."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fresh40_v5_pipeline import (
    HEADS,
    _execution_provenance,
    _load_config,
    _load_student_data,
    _publish,
    _seal,
    event_label,
    hash_listed_files,
    sha256_file,
    variant_decision,
)


def critical_event_intervals(rows: list[dict]) -> list[dict]:
    """Build Teacher TRUE runs from the full timeline, before candidate gating."""
    result, start = [], None
    for idx, row in enumerate(rows):
        is_true = bool(row["labels"]["physical_criticality"]["mask"] and row["labels"]["physical_criticality"]["value"] == "TRUE")
        if is_true and start is None:
            start = idx
        elif not is_true and start is not None:
            result.append({"start": start, "end": idx - 1})
            start = None
    if start is not None:
        result.append({"start": start, "end": len(rows) - 1})
    return result


def _overlaps(left: dict, right: dict) -> bool:
    return left["episode_id"] == right["episode_id"] and left["start"] <= right["end"] and right["start"] <= left["end"]


def _event_metrics(events: list[dict], ladder: tuple[str, ...]) -> dict:
    rows = {name: {"positive": 0, "selected_positive": 0, "negative": 0, "selected_negative": 0, "latencies": []} for name in ladder}
    excluded_unknown = 0
    for event in events:
        target = event_label(x["value"] for x in event["labels"])
        if target == "UNKNOWN":
            excluded_unknown += 1
            continue
        positive = target == "TRUE"
        for name in ladder:
            selected = event["selected"][name]
            if positive:
                rows[name]["positive"] += 1
                rows[name]["selected_positive"] += int(selected is not None)
                if selected is not None:
                    rows[name]["latencies"].append(selected - event["start"])
            else:
                rows[name]["negative"] += 1
                rows[name]["selected_negative"] += int(selected is not None)
    result = {"excluded_unknown_events": excluded_unknown}
    for name, value in rows.items():
        result[name] = {
            "positive_events": value["positive"],
            "selected_positive_events": value["selected_positive"],
            "event_recall": value["selected_positive"] / value["positive"] if value["positive"] else None,
            "known_negative_events": value["negative"],
            "selected_negative_events": value["selected_negative"],
            "event_false_positive_rate": value["selected_negative"] / value["negative"] if value["negative"] else None,
            "mean_emit_latency_steps": float(np.mean(value["latencies"])) if value["latencies"] else None,
            "selected_events": value["selected_positive"] + value["selected_negative"],
        }
    return result


def run(dataset_root: Path, checkpoint_root: Path, output_root: Path, config_path: Path) -> dict:
    if output_root.exists():
        raise RuntimeError(f"refusing to overwrite existing root: {output_root}")
    provenance = _execution_provenance()
    hash_listed_files(dataset_root)
    hash_listed_files(checkpoint_root)
    cfg = _load_config(config_path)
    import torch
    from n5_student_model import N5MultiHeadStudent

    checkpoint = torch.load(checkpoint_root / "checkpoint.pt", map_location="cpu", weights_only=False)
    model = N5MultiHeadStudent(input_dim=25, hidden=int(checkpoint["hidden"]), short_rf=int(checkpoint["short_rf"]), long_rf=int(checkpoint["long_rf"]), dropout=0.1)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    dev_eps, _, _ = _load_student_data(dataset_root, "dev")
    ladder = ("candidate_teacher_critical", "predicted_critical", "critical_oracle_aux", "predicted_three_head", "predicted_full_five")
    all_events, teacher_events = [], []
    total_dev_steps = candidate_steps = known_critical_steps = unknown_critical_steps = 0
    teacher_critical_steps = teacher_critical_candidate_steps = 0
    with torch.no_grad():
        for ep in dev_eps:
            rows = ep["rows"]
            total_dev_steps += len(rows)
            for interval in critical_event_intervals(rows):
                teacher_events.append({"episode_id": ep["identity"], **interval})
            x = torch.from_numpy(ep["features"]).unsqueeze(0)
            probabilities = {h: torch.sigmoid(v).squeeze(0).cpu().numpy() for h, v in model(x, timestep_mask=torch.ones((1, x.shape[1]), dtype=torch.bool)).items()}
            current = None
            for idx, row in enumerate(rows):
                candidate = bool(row["candidate_close"])
                candidate_steps += int(candidate)
                target = row["labels"]["physical_criticality"]
                known_critical_steps += int(target["mask"])
                unknown_critical_steps += int(not target["mask"])
                is_teacher_critical = bool(target["mask"] and target["value"] == "TRUE")
                teacher_critical_steps += int(is_teacher_critical)
                teacher_critical_candidate_steps += int(is_teacher_critical and candidate)
                if candidate and current is None:
                    current = {"episode_id": ep["identity"], "start": idx, "end": idx, "labels": [], "selected": {name: None for name in ladder}}
                if not candidate:
                    if current is not None:
                        current["end"] = idx - 1
                        all_events.append(current); current = None
                    continue
                current["end"] = idx
                labels = {h: row["labels"][h] for h in HEADS}
                p = {h: float(probabilities[h][idx]) for h in HEADS}
                current["labels"].append(labels["physical_criticality"])
                predicted_critical = p["physical_criticality"] >= 0.5
                oracle_aux = (
                    labels["instability"]["mask"] and labels["instability"]["value"] == "FALSE"
                    and labels["gripper_closing_state"]["mask"] and labels["gripper_closing_state"]["value"] == "TRUE"
                )
                decisions = {
                    "candidate_teacher_critical": is_teacher_critical,
                    "predicted_critical": bool(predicted_critical),
                    "critical_oracle_aux": bool(predicted_critical and oracle_aux),
                    "predicted_three_head": variant_decision("three_head", True, p),
                    "predicted_full_five": variant_decision("full_five", True, p),
                }
                for name in ladder:
                    if current["selected"][name] is None and decisions[name]:
                        current["selected"][name] = idx
            if current is not None:
                all_events.append(current)
    for event in all_events:
        event["event_label"] = event_label(x["value"] for x in event["labels"])
    candidate_event_counts = {
        "total": len(all_events),
        "known_positive": sum(e["event_label"] == "TRUE" for e in all_events),
        "known_negative": sum(e["event_label"] == "FALSE" for e in all_events),
        "unknown": sum(e["event_label"] == "UNKNOWN" for e in all_events),
    }
    for teacher_event in teacher_events:
        teacher_event["candidate_overlap"] = any(_overlaps(teacher_event, event) for event in all_events)
    overlapped_teacher_events = sum(e["candidate_overlap"] for e in teacher_events)
    overlapped_candidate_events = sum(any(_overlaps(event, teacher_event) for teacher_event in teacher_events) for event in all_events)
    metrics = _event_metrics(all_events, ladder)
    provenance["actual_ended_at"] = datetime.now(timezone.utc).isoformat()
    metrics.update({
        "schema": "FRESH40_V5_ORACLE_LADDER_V1",
        "status": "DEVELOPMENT_NONCONSUMABLE",
        "development_only": True,
        "identity_count": len(dev_eps),
        "total_dev_steps": total_dev_steps,
        "candidate_steps_processed": candidate_steps,
        "known_critical_steps": known_critical_steps,
        "unknown_critical_steps": unknown_critical_steps,
        "teacher_critical_steps": teacher_critical_steps,
        "teacher_critical_candidate_steps": teacher_critical_candidate_steps,
        "candidate_events": candidate_event_counts,
        "teacher_critical_events": {
            "total": len(teacher_events),
            "overlapped_by_candidate": overlapped_teacher_events,
            "missed_by_candidate": len(teacher_events) - overlapped_teacher_events,
            "event_recall_ceiling": overlapped_teacher_events / len(teacher_events) if teacher_events else None,
            "candidate_overlapped_events": overlapped_candidate_events,
        },
        "checkpoint_sha256": sha256_file(checkpoint_root / "checkpoint.pt"),
        "dataset_manifest_sha256": sha256_file(dataset_root / "MANIFEST.json"),
        "oracle_fields_used": True,
        "oracle_fields_runtime_input": False,
        "action_mutation": False,
        "protected_reads": False,
        "attack_enabled": False,
        "formal_selection_eligible": False,
        **provenance,
        "equations": {
            "candidate_teacher_critical": "candidate AND recorded physical_criticality TRUE; diagnostic oracle",
            "predicted_critical": "candidate AND predicted physical_criticality >= 0.5",
            "critical_oracle_aux": "predicted critical AND recorded instability FALSE AND recorded gripper_closing_state TRUE",
            "predicted_three_head": "candidate AND predicted physical_criticality >= 0.5 AND instability < 0.5 AND gripper_closing_state >= 0.5",
            "predicted_full_five": "predicted three-head AND k10 >= 0.5 AND safe_release < 0.5",
        },
        "protocol_sha256": sha256_file(config_path),
    })
    staging = output_root.parent / f".{output_root.name}.staging.{os.getpid()}"
    if staging.exists() or output_root.exists():
        raise RuntimeError(f"staging/output exists: {output_root}")
    staging.mkdir(parents=True)
    (staging / "ladder_records.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in all_events))
    (staging / "teacher_critical_events.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in teacher_events))
    (staging / "evaluation_summary.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    (staging / "MANIFEST.json").write_text(json.dumps({"schema": "FRESH40_V5_ORACLE_LADDER_BUNDLE_V1", "status": "DEVELOPMENT_NONCONSUMABLE", "dataset_manifest_sha256": metrics["dataset_manifest_sha256"], "checkpoint_sha256": metrics["checkpoint_sha256"], "total_dev_steps": total_dev_steps, "candidate_steps_processed": candidate_steps, "candidate_event_count": len(all_events), "teacher_critical_event_count": len(teacher_events), **provenance, "oracle_fields_used": True, "oracle_fields_runtime_input": False, "formal_selection_eligible": False, "protected_reads": False, "attack_enabled": False}, indent=2, sort_keys=True))
    seal = _seal(staging)
    _publish(staging, output_root)
    return {"output_root": str(output_root), "sha256sums_sha256": seal["sha256sums_sha256"], **metrics}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--checkpoint-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "configs" / "FRESH40_V5_DEVELOPMENT_PROTOCOL_V1.json")
    args = ap.parse_args()
    print(json.dumps(run(args.dataset_root.resolve(), args.checkpoint_root.resolve(), args.output_root.resolve(), args.config.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
