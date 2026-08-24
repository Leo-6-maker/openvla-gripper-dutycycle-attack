#!/usr/bin/env python3
"""Run and aggregate the frozen clean-only X1 sequential PGD diagnostic."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from scripts.stage_ix.run_stage_ix_f0 import gate_for_row, run_exact_pgd  # noqa: E402


COUNTERS = {
    "protected_reads": 0,
    "eval160_reads": 0,
    "attack_rollouts": 0,
    "vis_pgd_attack_rollouts": 0,
    "env_steps_with_perturbed_action": 0,
}
VALID_LABELS = {"V_PHYS": 1, "NO_PHYSICAL_VULNERABILITY": 0}
DOSES = ("T3", "T5", "T10")
METRICS = ("Q1", "Q2", "Q3")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def label_value(row: dict[str, Any]) -> int | None:
    if row.get("binary_label_consumable") is not True or row.get("control_valid") is not True or row.get("treatment_valid") is not True:
        return None
    if row.get("censoring_class") not in (None, "NONE"):
        return None
    return VALID_LABELS.get(str(row.get("label_class")))


def load_labels(protocol: dict[str, Any]) -> dict[tuple[str, str, str, int, str], int | None]:
    paths = {
        "STAGE_V": Path("/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_M4_CENSOR_AWARE_FORMAL_M4_AGGREGATE_F696F582_20260815T175500Z/M4_ALL_LABELS_V1.jsonl"),
        "STAGE_VI_B2": Path("/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_VI_B2_FORMAL_M4_AGGREGATE_V3_DUAL_SOURCE_20260816T061432Z/B2_ALL_LABELS.jsonl"),
    }
    expected = {
        "STAGE_V": "cf4ece1b0c864289f6dcf318d91ff59fbb3aecbb65d8482f11ad509e3ff59cea",
        "STAGE_VI_B2": "3817648b3a3ac4de236a48a1377dc252ac6dd54d5fdf254f0753392b3b7c7106",
    }
    labels: dict[tuple[str, str, str, int, str], int | None] = {}
    for stage, path in paths.items():
        if sha256_file(path) != expected[stage]:
            raise ValueError(f"LABEL_SHA256_MISMATCH:{stage}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = (stage, str(row["canonical_parent_key"]), str(row["probe_id"]), int(row["probe_step"]), str(row["dose"]))
                value = label_value(row)
                if key in labels and labels[key] != value:
                    raise ValueError(f"DUPLICATE_LABEL_CONFLICT:{key}")
                labels[key] = value
    return labels


def load_audit_rows(audit_root: Path) -> tuple[dict[str, Any], dict[tuple[str, str, int], dict[str, Any]]]:
    summary = load_json(audit_root / "STAGE_X_X1_SEQUENCE_INPUT_AUDIT.json")
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    with (audit_root / "X1_SEQUENCE_ROWS.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (str(row["stage"]), str(row["canonical_parent_key"]), int(row["absolute_step"]))
            if key in rows:
                raise ValueError(f"DUPLICATE_SEQUENCE_FRAME:{key}")
            rows[key] = row
    return summary, rows


def load_f0_reference(protocol: dict[str, Any]) -> tuple[dict[str, dict[str, float | None]], dict[str, str]]:
    root = Path(protocol["stage_ix_negative_reference"]["result_root"])
    split = load_json(root / "F0_SPLIT.json")["assignments"]
    reference: dict[str, dict[str, float | None]] = {}
    with (root / "F0_ROWS.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = "|".join((str(row["stage"]), str(row["canonical_parent_key"]), str(row["probe_id"]), str(int(row["probe_step"]))))
            reference[key] = {"E1": row.get("e1"), "E3": row.get("e3")}
    return reference, {str(key): str(value) for key, value in split.items()}


def sequence_starts(summary: dict[str, Any], frame_rows: dict[tuple[str, str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    starts: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for item in summary["sequence"]["eligible_starts"]["3"]:
        key = (str(item["stage"]), str(item["canonical_parent_key"]), int(item["start_step"]))
        if key in seen:
            continue
        seen.add(key)
        current = frame_rows[key]
        frames = []
        for offset in range(10):
            candidate = frame_rows.get((key[0], key[1], key[2] + offset))
            if candidate is None or candidate.get("exact_current_frame") is not True:
                break
            frames.append(candidate)
        starts.append({
            "sequence_id": "|".join((key[0], key[1], str(current["probe_id"]), str(key[2]))),
            "stage": key[0],
            "canonical_parent_key": key[1],
            "probe_id": str(current["probe_id"]),
            "probe_step": key[2],
            "parent_unit": f"{key[0]}|{key[1]}",
            "frames": frames,
        })
    return starts


def sequence_metrics(success: list[bool]) -> dict[str, float | int | None]:
    metrics: dict[str, float | int | None] = {}
    for d in (3, 5, 10):
        metrics[f"Q1_{d}"] = None if len(success) < d else float(sum(success[:d]) / d)
        if not success:
            metrics[f"Q2_{d}"] = None
        else:
            longest = 0
            current = 0
            for value in success:
                current = current + 1 if value else 0
                longest = max(longest, current)
            metrics[f"Q2_{d}"] = int(min(longest, d))
        metrics[f"Q3_{d}"] = None if len(success) < d else int(all(success[:d]))
    return metrics


def snapshot_action(payload: dict[str, Any], step: int) -> np.ndarray:
    window = payload.get("clean_reference_action_window") or []
    current = next((item for item in window if isinstance(item, dict) and int(item.get("step", -1)) == step), None)
    if not isinstance(current, dict):
        raise ValueError(f"CLEAN_ACTION_MISSING:{step}")
    action = np.asarray(current.get("raw_policy_action", []), dtype=np.float32).reshape(-1)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError(f"CLEAN_ACTION_INVALID:{step}")
    return action


def decode_top(adapter: Any, suite: str, token_id: int) -> tuple[str, float | None]:
    region = adapter.get_gripper_region_by_decoded_action(suite, postprocess_gripper=True)
    action = region.get("token_action_map", {}).get(int(token_id))
    if int(token_id) in set(region.get("open_token_ids", [])):
        return "OPEN", None if action is None else float(action)
    if int(token_id) in set(region.get("close_token_ids", [])):
        return "CLOSE", None if action is None else float(action)
    return "BOUNDARY_OR_UNKNOWN", None if action is None else float(action)


def init_root(args: argparse.Namespace) -> int:
    root = Path(args.output_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"X1_ROOT_NOT_EMPTY:{root}")
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.protocol, root / "STAGE_X_X1_SEQUENTIAL_PGD_PROTOCOL_V1.json")
    shutil.copy2(REPO / "configs" / "STAGE_IX_CANONICAL_PGD_CONTRACT_V1.json", root / "STAGE_IX_CANONICAL_PGD_CONTRACT_V1.json")
    (root / "X1_RUN_STATUS.json").write_text(json.dumps({
        "schema": "STAGE_X_X1_RUN_STATUS_V1",
        "status": "RUNNING",
        "source_commit": git_value("rev-parse", "HEAD"),
        "source_tree": git_value("rev-parse", "HEAD^{tree}"),
        "protected_counters": COUNTERS,
        "physical_intervention": False,
        "env_steps_with_perturbed_action": 0,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def worker(args: argparse.Namespace) -> int:
    import torch
    import yaml
    from gripper_attack.attack_adapter import TokenPrefixPGDAttacker
    from gripper_attack.route_contract import resolve_adapter_class_name, route_config_from_attack_config, validate_attack_request
    from gripper_attack.stage_v_causal_observation_snapshot import assert_primary_observation_exact, load_snapshot
    from scripts.detector_v5.materialize_stage_vii_frozen_embeddings import load_model

    root = Path(args.output_root).resolve()
    protocol = load_json(args.protocol)
    audit_summary, frame_rows = load_audit_rows(Path(args.sequence_audit_root).resolve())
    labels = load_labels(protocol)
    f0_reference, split = load_f0_reference(protocol)
    starts = sequence_starts(audit_summary, frame_rows)
    active_parents = sorted({item["parent_unit"] for item in starts})
    wanted = {parent for index, parent in enumerate(active_parents) if index % args.worker_count == args.worker_index}
    selected = [item for item in starts if item["parent_unit"] in wanted]
    config = yaml.safe_load((REPO / "configs" / "paper_black_bowl_attack.yaml").read_text(encoding="utf-8"))
    config["attack_optimizer"]["strict_route"] = True
    config["attack_optimizer"]["allow_fallback"] = False
    route = route_config_from_attack_config(config)
    if resolve_adapter_class_name(route) != "TokenPrefixPGDAttacker" or route.strict_route is not True or route.allow_fallback is not False:
        raise ValueError("STRICT_TOKEN_PREFIX_ROUTE_NOT_BOUND")
    validate_attack_request(route, target_action_present=True)
    contract = load_json(REPO / "configs" / "STAGE_IX_CANONICAL_PGD_CONTRACT_V1.json")
    device = "cuda:0"
    model = load_model(Path(contract["victim"]["model_path"]), device)
    adapter = TokenPrefixPGDAttacker(model, object(), config, seed=0, preprocess_kwargs={"postprocess_gripper": True}, device=device)
    adapter._freeze_model()
    gpu_uuid = "UNKNOWN"
    try:
        gpu_uuid = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0")], text=True).strip()
    except Exception:
        pass
    output_path = root / f"worker_{args.worker_index:02d}.jsonl"
    with output_path.open("w", encoding="utf-8") as output:
        for number, sequence in enumerate(selected, 1):
            adapter.reset_temporal_state()
            start = sequence["frames"][0]
            first_package = load_snapshot(Path(start["snapshot_path"]).parent, materialize_torch=True)
            first_payload = first_package["payload"]
            assert_primary_observation_exact(first_payload)
            first_action = snapshot_action(first_payload, int(start["absolute_step"]))
            parent_root = Path(start["snapshot_path"]).parents[2]
            trajectory = load_json(parent_root / "CLEAN_TRAJECTORY_V1_4.json")
            gate_row = {"stage": sequence["stage"], "canonical_parent_key": sequence["canonical_parent_key"], "probe_id": sequence["probe_id"], "probe_step": sequence["probe_step"]}
            gate = gate_for_row(gate_row, trajectory, first_action)
            frame_results: list[dict[str, Any]] = []
            if gate["eligible"]:
                for frame in sequence["frames"]:
                    package = first_package if frame is start else load_snapshot(Path(frame["snapshot_path"]).parent, materialize_torch=True)
                    payload = package["payload"]
                    assert_primary_observation_exact(payload)
                    clean_action = snapshot_action(payload, int(frame["absolute_step"]))
                    clean_ids = payload["input_ids"].to(device=device, dtype=torch.long)
                    x0 = payload["pixel_values"].to(device=device)
                    if clean_ids.ndim == 1:
                        clean_ids = clean_ids.unsqueeze(0)
                    if x0.ndim == 3:
                        x0 = x0.unsqueeze(0)
                    target_action = clean_action.copy()
                    target_action[-1] = 1.0
                    suite = sequence["canonical_parent_key"].split("/", 1)[0]
                    target_ids = adapter.action_to_token_ids(target_action, suite)
                    pgd = run_exact_pgd(adapter, clean_ids, x0, target_ids, suite, max_steps=20)
                    decoded_class, decoded_action = decode_top(adapter, suite, int(pgd["final"]["top_token_id"]))
                    frame_results.append({
                        "sequence_offset": int(frame["absolute_step"] - sequence["probe_step"]),
                        "absolute_step": int(frame["absolute_step"]),
                        "snapshot_manifest_sha256": frame["snapshot_manifest_sha256"],
                        "clean_margin": float(pgd["clean"]["margin"]),
                        "attacked_margin": float(pgd["final"]["margin"]),
                        "target_rank": int(pgd["final"]["rank"]),
                        "targeted_OPEN_success": bool(pgd["target_success"]),
                        "decoded_gripper_action": decoded_class,
                        "decoded_gripper_raw": decoded_action,
                        "pixel_linf_norm": float(pgd["pixel_budget_linf"]),
                        "iterations": 20,
                        "prev_delta_used": bool(pgd["prev_delta_used"]),
                        "target_token_id": int(pgd["target_token_id"]),
                        "target_top_token_id": int(pgd["final"]["top_token_id"]),
                    })
                    if package is not first_package:
                        del package, payload
                    del clean_ids, x0, target_ids, pgd
                torch.cuda.empty_cache()
            sequence_row = {
                "schema": "STAGE_X_X1_SEQUENCE_ROW_V1",
                "sequence_id": sequence["sequence_id"],
                "stage": sequence["stage"],
                "canonical_parent_key": sequence["canonical_parent_key"],
                "parent_unit": sequence["parent_unit"],
                "probe_id": sequence["probe_id"],
                "probe_step": int(sequence["probe_step"]),
                "suite": sequence["canonical_parent_key"].split("/", 1)[0],
                "split": split.get(sequence["parent_unit"]),
                "frame_count": len(frame_results),
                "gate": gate,
                "frame_results": frame_results,
                "metrics": sequence_metrics([bool(frame["targeted_OPEN_success"]) for frame in frame_results]),
                "vphys_by_dose": {dose: labels.get((sequence["stage"], sequence["canonical_parent_key"], sequence["probe_id"], int(sequence["probe_step"]), dose)) for dose in DOSES},
                "worker_index": args.worker_index,
                "pid": os.getpid(),
                "gpu_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "gpu_uuid": gpu_uuid,
                "env_step_executed": False,
                "physical_intervention": False,
                "protected_counters": COUNTERS,
            }
            output.write(json.dumps(sequence_row, sort_keys=True, ensure_ascii=False) + "\n")
            output.flush()
            del first_package, first_payload
            if number % 4 == 0:
                print(f"[worker {args.worker_index}] {number}/{len(selected)}", flush=True)
    (root / f"worker_{args.worker_index:02d}.done.json").write_text(json.dumps({
        "schema": "STAGE_X_X1_WORKER_DONE_V1",
        "worker_index": args.worker_index,
        "row_count": len(selected),
        "source_commit": git_value("rev-parse", "HEAD"),
        "source_tree": git_value("rev-parse", "HEAD^{tree}"),
        "protected_counters": COUNTERS,
        "physical_intervention": False,
        "env_steps_with_perturbed_action": 0,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def auc(items: list[dict[str, Any]], score_key: str, label_key: str = "label") -> float | None:
    values = [(float(item[score_key]), int(item[label_key])) for item in items if item.get(score_key) is not None and item.get(label_key) in (0, 1)]
    positives = [score for score, label in values if label == 1]
    negatives = [score for score, label in values if label == 0]
    if not positives or not negatives:
        return None
    return float(sum(1.0 if positive > negative else 0.5 if positive == negative else 0.0 for positive in positives for negative in negatives) / (len(positives) * len(negatives)))


def top_lifts(items: list[dict[str, Any]], score_key: str) -> dict[str, float | None]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        if item.get(score_key) is not None and item.get("label") in (0, 1):
            groups[str(item["parent_unit"])].append(item)
    output: dict[str, float | None] = {}
    for k in (1, 3):
        selected: list[float] = []
        random_rates: list[float] = []
        for group in groups.values():
            if len(group) < k or sum(int(item["label"]) for item in group) == 0:
                continue
            ordered = sorted(group, key=lambda item: (-float(item[score_key]), str(item["probe_id"]), int(item["probe_step"])))
            selected.append(float(sum(int(item["label"]) for item in ordered[:k]) / k))
            random_rates.append(float(sum(int(item["label"]) for item in group) / len(group)))
        output[f"top{k}_lift"] = None if not random_rates or sum(random_rates) == 0 else float((sum(selected) / len(selected)) / (sum(random_rates) / len(random_rates)))
    return output


def average_precision(items: list[dict[str, Any]], score_key: str) -> float | None:
    values = sorted([(float(item[score_key]), int(item["label"])) for item in items if item.get(score_key) is not None and item.get("label") in (0, 1)], key=lambda value: -value[0])
    total_positive = sum(label for _, label in values)
    if total_positive == 0:
        return None
    seen = 0
    total = 0.0
    for index, (_, label) in enumerate(values, 1):
        seen += label
        if label:
            total += seen / index
    return float(total / total_positive)


def metric_bundle(items: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    parent_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    suite_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        if item.get(score_key) is not None and item.get("label") in (0, 1):
            parent_groups[str(item["parent_unit"])].append(item)
            suite_groups[str(item["suite"])].append(item)
    parent_aucs = {key: auc(group, score_key) for key, group in parent_groups.items()}
    parent_aucs = {key: value for key, value in parent_aucs.items() if value is not None}
    suite_aucs = {key: auc(group, score_key) for key, group in suite_groups.items()}
    suite_aucs = {key: value for key, value in suite_aucs.items() if value is not None}
    top = top_lifts(items, score_key)
    return {
        "row_count": sum(len(group) for group in parent_groups.values()),
        "auroc": auc(items, score_key),
        "auprc": average_precision(items, score_key),
        "parent_macro_auc": None if not parent_aucs else float(sum(parent_aucs.values()) / len(parent_aucs)),
        "parent_aucs": parent_aucs,
        "top1_lift": top["top1_lift"],
        "top3_lift": top["top3_lift"],
        "per_suite_auc": suite_aucs,
        "mean_loso_auc": None if not suite_aucs else float(sum(suite_aucs.values()) / len(suite_aucs)),
        "worst_identifiable_suite_auc": None if not suite_aucs else float(min(suite_aucs.values())),
    }


def bootstrap_parent_auc(items: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    bundle = metric_bundle(items, score_key)
    values = list(bundle["parent_aucs"].values())
    if not values:
        return {"replicates": 2000, "seed": 20260817, "ci95": None}
    rng = np.random.RandomState(20260817)
    draws = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(2000)]
    return {"replicates": 2000, "seed": 20260817, "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]}


def aggregate(args: argparse.Namespace) -> int:
    root = Path(args.output_root).resolve()
    protocol = load_json(args.protocol)
    audit_summary, _ = load_audit_rows(Path(args.sequence_audit_root).resolve())
    expected_rows = len(sequence_starts(audit_summary, load_audit_rows(Path(args.sequence_audit_root).resolve())[1]))
    rows: list[dict[str, Any]] = []
    for index in range(args.worker_count):
        path = root / f"worker_{index:02d}.jsonl"
        done_path = root / f"worker_{index:02d}.done.json"
        if not path.is_file() or not done_path.is_file():
            raise ValueError(f"WORKER_INCOMPLETE:{index}")
        done = load_json(done_path)
        if done.get("protected_counters") != COUNTERS or done.get("physical_intervention") is not False:
            raise ValueError(f"WORKER_BOUNDARY_INVALID:{index}")
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if len(rows) != expected_rows:
        raise ValueError(f"X1_ROW_COUNT_INVALID:{len(rows)}:{expected_rows}")
    ids = [row["sequence_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("X1_DUPLICATE_SEQUENCE_ID")
    if any(row.get("protected_counters") != COUNTERS or row.get("physical_intervention") is not False or row.get("env_step_executed") is not False for row in rows):
        raise ValueError("X1_PROTECTED_BOUNDARY_INVALID")

    eval_rows: list[dict[str, Any]] = []
    for row in rows:
        for dose in DOSES:
            for metric in METRICS:
                for d in (3, 5, 10):
                    score = row.get("metrics", {}).get(f"{metric}_{d}")
                    eval_rows.append({
                        "sequence_id": row["sequence_id"],
                        "parent_unit": row["parent_unit"],
                        "probe_id": row["probe_id"],
                        "probe_step": row["probe_step"],
                        "stage": row["stage"],
                        "suite": row["suite"],
                        "split": row.get("split"),
                        "dose": dose,
                        "metric": metric,
                        "d": d,
                        "score": score,
                        "label": row.get("vphys_by_dose", {}).get(dose),
                    })
    reference, split = load_f0_reference(protocol)
    candidates: dict[str, Any] = {}
    for metric in METRICS:
        for d in (3, 5, 10):
            key = f"{metric}_{d}"
            dev = [item for item in eval_rows if item["metric"] == metric and item["d"] == d and item["dose"] == "T5" and item.get("split") == "DEVTEST"]
            bundle = metric_bundle(dev, "score")
            baseline_key = protocol["evaluation"]["baseline_score_map"][metric]
            baseline_items: list[dict[str, Any]] = []
            for item in dev:
                reference_item = reference.get("|".join((item["stage"], item["parent_unit"].split("|", 1)[1], item["probe_id"], str(item["probe_step"]))))
                if reference_item is not None and reference_item.get(baseline_key) is not None and item.get("label") in (0, 1):
                    baseline_items.append({**item, "score": reference_item[baseline_key]})
            baseline_bundle = metric_bundle(baseline_items, "score")
            common_parents = sorted(set(bundle["parent_aucs"]).intersection(baseline_bundle["parent_aucs"]))
            improvement = None if not common_parents else float(sum(bundle["parent_aucs"][parent] - baseline_bundle["parent_aucs"][parent] for parent in common_parents) / len(common_parents))
            passing = bool(
                bundle["parent_macro_auc"] is not None and bundle["parent_macro_auc"] >= 0.65
                and bundle["top1_lift"] is not None and bundle["top1_lift"] >= 1.25
                and bundle["top3_lift"] is not None and bundle["top3_lift"] >= 1.15
                and bundle["worst_identifiable_suite_auc"] is not None and bundle["worst_identifiable_suite_auc"] >= 0.55
                and bundle["mean_loso_auc"] is not None and bundle["mean_loso_auc"] >= 0.60
                and improvement is not None and improvement >= 0.10
            )
            candidates[key] = {"metric": metric, "d": d, "devtest_t5": bundle, "baseline": baseline_bundle, "parent_macro_improvement": improvement, "pass_gate": passing, "bootstrap_parent_macro_auc": bootstrap_parent_auc(dev, "score")}

    passing = [key for key, value in candidates.items() if value["pass_gate"]]
    identifiable = [key for key, value in candidates.items() if value["devtest_t5"]["row_count"] > 0]
    status = "STAGE_X_SEQUENTIAL_PGD_REALIZABILITY_SIGNAL_ESTABLISHED" if passing else "STAGE_X_SEQUENTIAL_PGD_SIGNAL_WEAK" if identifiable else "STAGE_X_NO_SEQUENTIAL_PGD_REALIZABILITY_SIGNAL"
    summary = {
        "schema": "STAGE_X_X1_RESULT_V1",
        "status": status,
        "x1_pgd_executed": True,
        "x1_authorized": bool(passing),
        "x2_physical_pgd_authorized": False,
        "source_commit": git_value("rev-parse", "HEAD"),
        "source_tree": git_value("rev-parse", "HEAD^{tree}"),
        "source_script_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_sha256": sha256_file(args.protocol),
        "sequence_input_audit_root": str(Path(args.sequence_audit_root).resolve()),
        "sequence_input_audit_sha256": sha256_file(Path(args.sequence_audit_root) / "STAGE_X_X1_SEQUENCE_INPUT_AUDIT.json"),
        "sequence_count": len(rows),
        "frame_result_count": sum(len(row.get("frame_results", [])) for row in rows),
        "candidates": candidates,
        "passing_candidates": passing,
        "identifiable_candidates": identifiable,
        "physical_intervention": False,
        "env_steps_with_perturbed_action": 0,
        "protected_counters": COUNTERS,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    }
    (root / "X1_ROWS.jsonl").write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (root / "STAGE_X_X1_RESULT.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    (root / "PROVENANCE.json").write_text(json.dumps({
        "schema": "STAGE_X_X1_RESULT_PROVENANCE_V1",
        "source_commit": summary["source_commit"],
        "source_tree": summary["source_tree"],
        "source_script_sha256": summary["source_script_sha256"],
        "protocol_path": str(args.protocol),
        "protocol_sha256": summary["protocol_sha256"],
        "sequence_input_audit_root": summary["sequence_input_audit_root"],
        "sequence_input_audit_sha256": summary["sequence_input_audit_sha256"],
        "physical_intervention": False,
        "env_steps_with_perturbed_action": 0,
        "protected_counters": COUNTERS,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "X1_RUN_STATUS.json").write_text(json.dumps({"schema": "STAGE_X_X1_RUN_STATUS_V1", "status": status, "protected_counters": COUNTERS, "physical_intervention": False, "env_steps_with_perturbed_action": 0}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    seal_root(root, summary)
    print(json.dumps({"status": status, "sequence_count": len(rows), "frame_result_count": summary["frame_result_count"], "passing_candidates": passing}, sort_keys=True))
    return 0


def seal_root(root: Path, summary: dict[str, Any]) -> None:
    excluded = {"SHA256SUMS", "SHA256SUMS.sha256", "ROOT_SEAL.json", "ROOT_SEAL.sha256"}
    entries = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in sorted(root.rglob("*")) if path.is_file() and path.name not in excluded]
    (root / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")
    sums_sha = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    (root / "ROOT_SEAL.json").write_text(json.dumps({"schema": "STAGE_X_X1_RESULT_ROOT_SEAL_V1", "status": summary["status"], "summary_sha256": sha256_file(root / "STAGE_X_X1_RESULT.json"), "sha256sums_sha256": sums_sha, "physical_intervention": False, "env_steps_with_perturbed_action": 0, "protected_counters": COUNTERS, "eval160": "UNREAD", "protected_evaluation": "UNREAD"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "ROOT_SEAL.sha256").write_text(f"{sha256_file(root / 'ROOT_SEAL.json')}  ROOT_SEAL.json\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--sequence-audit-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.init:
        return init_root(args)
    if args.aggregate:
        return aggregate(args)
    if args.worker_index is None or not (0 <= args.worker_index < args.worker_count):
        parser.error("worker mode requires a valid --worker-index")
    return worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
