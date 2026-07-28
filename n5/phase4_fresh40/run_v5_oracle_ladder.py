#!/usr/bin/env python3
"""Development-only event oracle ladder for the sealed Fresh40 dev split."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from fresh40_v5_pipeline import (
    HEADS,
    _load_config,
    _load_student_data,
    _publish,
    _seal,
    hash_listed_files,
    sha256_file,
    variant_decision,
)


def _event_metrics(events: list[dict], ladder: tuple[str, ...]) -> dict:
    rows = {name: {"positive": 0, "selected_positive": 0, "negative": 0, "selected_negative": 0, "latencies": []} for name in ladder}
    excluded_unknown = 0
    for event in events:
        known = [x for x in event["labels"] if x["mask"]]
        if not known or any(x["value"] == "UNKNOWN" for x in event["labels"]):
            excluded_unknown += 1
            continue
        positive = any(x["value"] == "TRUE" for x in known)
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
    ladder = ("oracle_critical", "predicted_critical", "critical_oracle_aux", "full_predicted")
    all_events, record_count = [], 0
    with torch.no_grad():
        for ep in dev_eps:
            x = torch.from_numpy(ep["features"]).unsqueeze(0)
            probabilities = {h: torch.sigmoid(v).squeeze(0).cpu().numpy() for h, v in model(x, timestep_mask=torch.ones((1, x.shape[1]), dtype=torch.bool)).items()}
            current = None
            for idx, row in enumerate(ep["rows"]):
                candidate = bool(row["candidate_close"])
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
                teacher_critical = labels["physical_criticality"]["mask"] and labels["physical_criticality"]["value"] == "TRUE"
                predicted_critical = p["physical_criticality"] >= 0.5
                oracle_aux = (
                    labels["instability"]["mask"] and labels["instability"]["value"] == "FALSE"
                    and labels["gripper_closing_state"]["mask"] and labels["gripper_closing_state"]["value"] == "TRUE"
                )
                decisions = {
                    "oracle_critical": bool(teacher_critical),
                    "predicted_critical": bool(predicted_critical),
                    "critical_oracle_aux": bool(predicted_critical and oracle_aux),
                    "full_predicted": variant_decision("full_five", True, p),
                }
                for name in ladder:
                    if current["selected"][name] is None and decisions[name]:
                        current["selected"][name] = idx
                record_count += 1
            if current is not None:
                all_events.append(current)
    metrics = _event_metrics(all_events, ladder)
    metrics.update({
        "schema": "FRESH40_V5_ORACLE_LADDER_V1",
        "status": "DEVELOPMENT_NONCONSUMABLE",
        "development_only": True,
        "identity_count": len(dev_eps),
        "step_count": record_count,
        "event_count": len(all_events),
        "checkpoint_sha256": sha256_file(checkpoint_root / "checkpoint.pt"),
        "dataset_manifest_sha256": sha256_file(dataset_root / "MANIFEST.json"),
        "oracle_fields_used": True,
        "oracle_fields_runtime_input": False,
        "action_mutation": False,
        "protected_reads": False,
        "attack_enabled": False,
        "formal_selection_eligible": False,
        "equations": {
            "oracle_critical": "candidate AND recorded physical_criticality TRUE; diagnostic oracle",
            "predicted_critical": "candidate AND predicted physical_criticality >= 0.5",
            "critical_oracle_aux": "predicted critical AND recorded instability FALSE AND recorded gripper_closing_state TRUE",
            "full_predicted": "candidate AND predicted physical_criticality >= 0.5 AND instability < 0.5 AND gripper_closing_state >= 0.5 AND k10 >= 0.5 AND safe_release < 0.5",
        },
        "protocol_sha256": sha256_file(config_path),
    })
    staging = output_root.parent / f".{output_root.name}.staging.{os.getpid()}"
    if staging.exists() or output_root.exists():
        raise RuntimeError(f"staging/output exists: {output_root}")
    staging.mkdir(parents=True)
    (staging / "ladder_records.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in all_events))
    (staging / "evaluation_summary.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    (staging / "MANIFEST.json").write_text(json.dumps({"schema": "FRESH40_V5_ORACLE_LADDER_BUNDLE_V1", "status": "DEVELOPMENT_NONCONSUMABLE", "dataset_manifest_sha256": metrics["dataset_manifest_sha256"], "checkpoint_sha256": metrics["checkpoint_sha256"], "event_count": len(all_events), "oracle_fields_used": True, "oracle_fields_runtime_input": False, "formal_selection_eligible": False, "protected_reads": False, "attack_enabled": False}, indent=2, sort_keys=True))
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
