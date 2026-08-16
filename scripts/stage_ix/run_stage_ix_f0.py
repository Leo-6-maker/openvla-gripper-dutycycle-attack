#!/usr/bin/env python3
"""Stage IX F0: no-environment white-box gripper-PGD exploitability audit."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
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

COUNTERS = {
    "protected_reads": 0,
    "eval160_reads": 0,
    "attack_rollouts": 0,
    "vis_pgd_attack_rollouts": 0,
    "env_steps_with_perturbed_action": 0,
}
VALID_LABELS = {"V_PHYS": 1, "NO_PHYSICAL_VULNERABILITY": 0}
CONFIG_NAMES = (
    "STAGE_IX_CANONICAL_PGD_CONTRACT_V1.json",
    "STAGE_IX_CLEAN_OPPORTUNITY_GATE_V1.json",
    "STAGE_IX_F0_VIS_EXPLOITABILITY_PROTOCOL_V1.json",
)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def init_root(root: Path) -> None:
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"F0_ROOT_NOT_EMPTY:{root}")
    root.mkdir(parents=True, exist_ok=True)
    for name in CONFIG_NAMES:
        shutil.copy2(REPO / "configs" / name, root / name)
    (root / "F0_PROTECTED_COUNTERS.json").write_text(json.dumps(COUNTERS, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    start = {
        "schema": "STAGE_IX_F0_RUN_STATUS_V1",
        "status": "RUNNING",
        "source_commit": git_value("rev-parse", "HEAD"),
        "source_tree": git_value("rev-parse", "HEAD^{tree}"),
        "protected_counters": dict(COUNTERS),
        "physical_intervention": False,
        "env_step_with_perturbed_action": False,
    }
    (root / "F0_RUN_STATUS.json").write_text(json.dumps(start, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_protocol(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract = load_json(root / CONFIG_NAMES[0])
    gate = load_json(root / CONFIG_NAMES[1])
    protocol = load_json(root / CONFIG_NAMES[2])
    if contract.get("schema") != "STAGE_IX_CANONICAL_PGD_CONTRACT_V1" or contract.get("status") != "FROZEN_BEFORE_F0_EXECUTION":
        raise ValueError("CANONICAL_PGD_CONTRACT_NOT_FROZEN")
    if gate.get("schema") != "STAGE_IX_CLEAN_OPPORTUNITY_GATE_V1" or gate.get("status") != "FROZEN_BEFORE_F0_EXECUTION":
        raise ValueError("OPPORTUNITY_GATE_NOT_FROZEN")
    if protocol.get("schema") != "STAGE_IX_F0_VIS_EXPLOITABILITY_PROTOCOL_V1" or protocol.get("status") != "FROZEN_BEFORE_F0_EXECUTION":
        raise ValueError("F0_PROTOCOL_NOT_FROZEN")
    if contract.get("protected_boundary", {}).get("protected_counters") != COUNTERS or gate.get("protected_boundary", {}).get("protected_counters") != COUNTERS or protocol.get("protected_boundary", {}).get("protected_counters") != COUNTERS:
        raise ValueError("PROTECTED_COUNTER_PROTOCOL_INVALID")
    return contract, gate, protocol


def verify_inputs(protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str, int], int | None]]:
    population = protocol["population"]
    source_rows: list[dict[str, Any]] = []
    label_map: dict[tuple[str, str, str, int], int | None] = {}
    for stage, label_key, snapshot_key in (
        ("STAGE_V", "stage_v_labels_t5", "stage_v_snapshots"),
        ("STAGE_VI_B2", "stage_vi_labels_t5", "stage_vi_snapshots"),
    ):
        label_spec = population["sources"][label_key]
        label_path = Path(label_spec["path"])
        if sha256_file(label_path) != label_spec["sha256"]:
            raise ValueError(f"LABEL_SHA256_MISMATCH:{stage}")
        with label_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("dose") != "T5" or row.get("primary_estimand") != "V_phys@T5":
                    continue
                key = (stage, str(row["canonical_parent_key"]), str(row["probe_id"]), int(row["probe_step"]))
                if row.get("binary_label_consumable") is True and row.get("control_valid") is True and row.get("treatment_valid") is True and row.get("label_class") in VALID_LABELS:
                    value: int | None = VALID_LABELS[str(row["label_class"])]
                else:
                    value = None
                if key in label_map and label_map[key] != value:
                    raise ValueError(f"DUPLICATE_LABEL_CONFLICT:{key}")
                label_map[key] = value

        spec = population["sources"][snapshot_key]
        root = Path(spec["path"])
        sums = Path(spec["sha256sums_path"])
        if sha256_file(sums) != spec["sha256sums_sha256"]:
            raise ValueError(f"SNAPSHOT_SUMS_SHA256_MISMATCH:{stage}")
        manifests = sorted(root.glob("**/CAUSAL_PROBE_SNAPSHOT_V2.json"))
        expected = int(spec["snapshot_count"])
        if len(manifests) != expected:
            raise ValueError(f"SNAPSHOT_COUNT_MISMATCH:{stage}:{len(manifests)}:{expected}")
        for path in manifests:
            manifest = load_json(path)
            binding = manifest.get("binding") or {}
            parent = str(binding.get("parent_key"))
            probe = str(binding.get("probe_id"))
            step = int(binding.get("step"))
            key = (stage, parent, probe, step)
            if key in {tuple(row["key"]) for row in source_rows}:
                raise ValueError(f"DUPLICATE_SNAPSHOT:{key}")
            source_rows.append({
                "key": list(key),
                "stage": stage,
                "canonical_parent_key": parent,
                "probe_id": probe,
                "probe_step": step,
                "suite": parent.split("/", 1)[0],
                "snapshot_path": str(path.parent),
                "parent_root": str(path.parents[2]),
                "snapshot_manifest_sha256": sha256_file(path),
                "label": label_map.get(key),
            })
    if len(source_rows) != 1344:
        raise ValueError(f"GLOBAL_F0_ROW_COUNT_INVALID:{len(source_rows)}")
    source_rows.sort(key=lambda row: (row["stage"], row["suite"], row["canonical_parent_key"], row["probe_step"], row["probe_id"]))
    return source_rows, label_map


def make_split(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, str]:
    split = protocol["split"]
    by_parent: dict[str, tuple[str, str]] = {}
    for row in rows:
        identity = f"{row['stage']}|{row['canonical_parent_key']}"
        by_parent[identity] = (row["stage"], row["suite"])
    assignments: dict[str, str] = {}
    for identity, (stage, suite) in sorted(by_parent.items()):
        digest = hashlib.sha256(f"{split['seed']}|{suite}|{stage}|{identity.split('|', 1)[1]}".encode()).digest()
        fraction = int.from_bytes(digest[:8], "big") / float(1 << 64)
        assignments[identity] = "TRAIN" if fraction < 0.6 else "VAL" if fraction < 0.8 else "DEVTEST"
    return assignments


def gate_for_row(row: dict[str, Any], trajectory: dict[str, Any], clean_action: np.ndarray) -> dict[str, Any]:
    by_step = {int(item["step"]): item for item in trajectory.get("rows", []) if isinstance(item, dict)}
    clean = by_step.get(int(row["probe_step"]))
    if clean is None:
        return {"eligible": False, "reason": "CLEAN_STEP_MISSING"}
    raw = np.asarray(clean.get("raw_action", []), dtype=np.float64).reshape(-1)
    env = np.asarray(clean.get("env_action", []), dtype=np.float64).reshape(-1)
    eef = np.asarray(clean.get("eef_position", []), dtype=np.float64).reshape(-1)
    aperture = clean.get("gripper_aperture")
    finite = raw.size == 7 and env.size == 7 and eef.size == 3 and np.isfinite(raw).all() and np.isfinite(env).all() and np.isfinite(eef).all() and aperture is not None and np.isfinite(float(aperture))
    if clean_action.shape != (7,) or not np.isfinite(clean_action).all():
        return {"eligible": False, "reason": "SNAPSHOT_ACTION_INVALID"}
    if finite and not np.array_equal(raw.astype(np.float32), clean_action.astype(np.float32)):
        raise ValueError(f"CLEAN_ACTION_BINDING_MISMATCH:{row['stage']}:{row['canonical_parent_key']}:{row['probe_id']}")
    eligible = bool(finite and clean.get("clean_record_valid") is True and raw[6] < 0.5 and env[6] > 0.5)
    return {
        "eligible": eligible,
        "reason": "CLEAN_GRIPPER_CLOSED_INTENT" if eligible else "GATE_FALSE",
        "finite_state": bool(finite),
        "clean_record_valid": bool(clean.get("clean_record_valid") is True),
        "raw_gripper": None if raw.size < 7 else float(raw[6]),
        "env_gripper": None if env.size < 7 else float(env[6]),
    }


def target_margin(audit: dict[str, Any]) -> dict[str, Any]:
    rows = audit.get("action_token_logit_audit", [])
    item = next((value for value in rows if int(value.get("dim", -1)) == 6), None)
    if not isinstance(item, dict):
        raise ValueError("GRIPPER_AUDIT_ROW_MISSING")
    return {
        "margin": float(-float(item["top_minus_target_logit"])),
        "rank": int(item["target_rank"]),
        "target_token_id": int(item["target_token_id"]),
        "top_token_id": int(item["top_token_id"]),
        "target_logit": float(item["target_logit"]),
        "top_logit": float(item["top_logit"]),
    }


def run_exact_pgd(adapter: Any, clean_ids: Any, x0_model: Any, target_ids: Any, suite: str, *, max_steps: int) -> dict[str, Any]:
    import torch

    action_dim = int(target_ids.numel())
    full_ids = torch.cat([clean_ids, target_ids.view(1, -1)], dim=1)
    labels = torch.full_like(full_ids, -100)
    labels[:, -action_dim:] = target_ids.view(1, -1)
    gripper_pos = int(labels.shape[1] - 1)
    masked = torch.full_like(labels, -100)
    masked[:, gripper_pos] = labels[:, gripper_pos]
    labels = masked
    x_orig_model = x0_model.detach()
    x_orig = x_orig_model.detach().float()
    prev = adapter._prev_delta
    prev_used = bool(adapter.temporal_init in {"prev_delta", "previous_delta", "carry", "carryover"} and prev is not None and tuple(prev.shape) == tuple(x_orig.shape))
    if prev_used:
        delta = torch.clamp(prev.detach().to(device=x_orig.device, dtype=torch.float32), -adapter.epsilon, adapter.epsilon)
    else:
        delta = torch.zeros_like(x_orig)
    adv = adapter._project_pixel_master(x_orig + delta, x_orig).detach()
    clean_model = adapter._cast_projected_pixel_values(x_orig, x_orig_model)
    clean_audit = adapter._audit_logits(full_ids, labels, clean_model, target_ids, suite, postprocess_gripper=True)
    clean_stats = target_margin(clean_audit)
    margins: dict[int, dict[str, Any]] = {0: clean_stats}
    for index in range(max(int(max_steps), 1)):
        adv = adv.detach().requires_grad_(True)
        adv_for_loss = adapter._cast_projected_pixel_values(adv, x_orig_model)
        loss = adapter._loss(
            full_ids,
            labels,
            adv_for_loss,
            objective="gripper_logit_margin_cw",
            num_action_tokens=action_dim,
            margin=5.0,
        )
        grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
        if not torch.isfinite(grad).all():
            raise ValueError("NONFINITE_PGD_GRADIENT")
        adv = adv.detach() - float(adapter.step_size) * grad.detach().sign()
        adv = adapter._project_pixel_master(adv, x_orig).detach()
        del grad, loss, adv_for_loss
        if index + 1 in {1, 3, int(max_steps)}:
            candidate = adapter._cast_projected_pixel_values(adv, x_orig_model)
            audit = adapter._audit_logits(full_ids, labels, candidate, target_ids, suite, postprocess_gripper=True)
            margins[index + 1] = target_margin(audit)
            del candidate, audit
    final_model = adapter._cast_projected_pixel_values(adv, x_orig_model)
    diff = (final_model.detach().float() - x_orig_model.detach().float()).detach()
    budget = float(diff.abs().max().detach().cpu()) if diff.numel() else 0.0
    if budget > float(adapter.epsilon) + 1e-6:
        raise ValueError(f"PGD_BUDGET_VIOLATION:{budget}:{adapter.epsilon}")
    adapter._prev_delta = diff.detach()
    final_stats = margins[int(max_steps)]
    return {
        "clean": clean_stats,
        "steps": margins,
        "final": final_stats,
        "prev_delta_used": prev_used,
        "pixel_budget_linf": budget,
        "target_success": bool(final_stats["rank"] == 1 and final_stats["margin"] >= 0.0),
        "target_token_id": int(final_stats["target_token_id"]),
    }


def worker(args: argparse.Namespace) -> int:
    import torch
    import yaml
    from gripper_attack.stage_v_causal_observation_snapshot import assert_primary_observation_exact, load_snapshot
    from gripper_attack.attack_adapter import TokenPrefixPGDAttacker
    from gripper_attack.route_contract import resolve_adapter_class_name, route_config_from_attack_config, validate_attack_request
    from scripts.detector_v5.materialize_stage_vii_frozen_embeddings import load_model

    root = Path(args.output_root).resolve()
    _, gate_protocol, protocol = load_protocol(root)
    rows, _ = verify_inputs(protocol)
    assignments = make_split(rows, protocol)
    identities = sorted({f"{row['stage']}|{row['canonical_parent_key']}" for row in rows})
    wanted = {identity for index, identity in enumerate(identities) if index % int(args.worker_count) == int(args.worker_index)}
    selected = [row for row in rows if f"{row['stage']}|{row['canonical_parent_key']}" in wanted]
    if args.limit:
        selected = selected[: int(args.limit)]
    config = yaml.safe_load((REPO / "configs" / "paper_black_bowl_attack.yaml").read_text(encoding="utf-8"))
    config["attack_optimizer"]["strict_route"] = True
    config["attack_optimizer"]["allow_fallback"] = False
    route = route_config_from_attack_config(config)
    if resolve_adapter_class_name(route) != "TokenPrefixPGDAttacker" or route.strict_route is not True or route.allow_fallback is not False:
        raise ValueError("STRICT_TOKEN_PREFIX_ROUTE_NOT_BOUND")
    validate_attack_request(route, target_action_present=True)
    device = "cuda:0"
    model_path = str(protocol["canonical_pgd_contract"] if "canonical_pgd_contract" in protocol else load_json(root / CONFIG_NAMES[0])["victim"]["model_path"])
    if model_path.endswith(".json"):
        model_path = load_json(root / CONFIG_NAMES[0])["victim"]["model_path"]
    model = load_model(Path(model_path), device)
    adapter = TokenPrefixPGDAttacker(model, object(), config, seed=0, preprocess_kwargs={"postprocess_gripper": True}, device=device)
    adapter._freeze_model()
    trajectory_cache: dict[str, dict[str, Any]] = {}
    out_path = root / f"worker_{int(args.worker_index):02d}.jsonl"
    with out_path.open("w", encoding="utf-8") as output:
        for number, row in enumerate(selected, 1):
            identity = f"{row['stage']}|{row['canonical_parent_key']}"
            if row["parent_root"] not in trajectory_cache:
                trajectory_cache[row["parent_root"]] = load_json(Path(row["parent_root"]) / "CLEAN_TRAJECTORY_V1_4.json")
            package = load_snapshot(Path(row["snapshot_path"]), materialize_torch=True)
            payload = package["payload"]
            assert_primary_observation_exact(payload)
            window = payload.get("clean_reference_action_window") or []
            current = next((item for item in window if int(item.get("step", -1)) == int(row["probe_step"])), window[0] if window else None)
            if not isinstance(current, dict):
                raise ValueError(f"CLEAN_ACTION_WINDOW_MISSING:{row['snapshot_path']}")
            clean_action = np.asarray(current.get("raw_policy_action", []), dtype=np.float32).reshape(-1)
            gate_info = gate_for_row(row, trajectory_cache[row["parent_root"]], clean_action)
            result: dict[str, Any] = {
                "stage": row["stage"],
                "canonical_parent_key": row["canonical_parent_key"],
                "probe_id": row["probe_id"],
                "probe_step": int(row["probe_step"]),
                "suite": row["suite"],
                "identity": identity,
                "split": assignments[identity],
                "snapshot_manifest_sha256": row["snapshot_manifest_sha256"],
                "label_vphys_t5": row["label"],
                "gate": gate_info,
                "worker_index": int(args.worker_index),
                "gpu_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "pid": os.getpid(),
                "env_step_executed": False,
                "physical_intervention": False,
                "protected_counters": dict(COUNTERS),
            }
            if not gate_info["eligible"]:
                adapter.reset_temporal_state()
                result.update({"e0": None, "e1": None, "e3": None, "full_pgd_success": None, "u_factorized": None})
            else:
                clean_ids = payload["input_ids"].to(device=device, dtype=torch.long)
                x0 = payload["pixel_values"].to(device=device)
                if clean_ids.ndim == 1:
                    clean_ids = clean_ids.unsqueeze(0)
                if x0.ndim == 3:
                    x0 = x0.unsqueeze(0)
                if tuple(clean_ids.shape[:1]) != (1,) or x0.ndim != 4:
                    raise ValueError(f"SNAPSHOT_TENSOR_SHAPE_INVALID:{row['snapshot_path']}")
                if not bool(torch.all(clean_ids[:, -1] == 29871)):
                    clean_ids = torch.cat([clean_ids, torch.tensor([[29871]], device=device, dtype=torch.long)], dim=1)
                target_action = clean_action.copy()
                target_action[-1] = 1.0
                target_ids = adapter.action_to_token_ids(target_action, row["suite"])
                pgd = run_exact_pgd(adapter, clean_ids, x0, target_ids, row["suite"], max_steps=20)
                clean_margin = float(pgd["clean"]["margin"])
                result.update({
                    "e0": clean_margin,
                    "e1": float(pgd["steps"][1]["margin"] - clean_margin),
                    "e3": float(pgd["steps"][3]["margin"] - clean_margin),
                    "full_pgd_success": bool(pgd["target_success"]),
                    "u_factorized": None if row["label"] is None else bool(row["label"] == 1 and pgd["target_success"]),
                    "target_token_id": int(pgd["target_token_id"]),
                    "target_margin_clean": clean_margin,
                    "target_margin_step1": float(pgd["steps"][1]["margin"]),
                    "target_margin_step3": float(pgd["steps"][3]["margin"]),
                    "target_margin_final": float(pgd["final"]["margin"]),
                    "target_rank_final": int(pgd["final"]["rank"]),
                    "target_top_token_final": int(pgd["final"]["top_token_id"]),
                    "prev_delta_used": bool(pgd["prev_delta_used"]),
                    "pixel_budget_linf": float(pgd["pixel_budget_linf"]),
                    "pgd_steps": 20,
                })
                del clean_ids, x0, target_ids, pgd
            output.write(json.dumps(result, sort_keys=True, ensure_ascii=False) + "\n")
            output.flush()
            del package, payload
            if number % 8 == 0:
                torch.cuda.empty_cache()
                print(f"[worker {args.worker_index}] {number}/{len(selected)}", flush=True)
    done = {
        "schema": "STAGE_IX_F0_WORKER_DONE_V1",
        "worker_index": int(args.worker_index),
        "row_count": len(selected),
        "source_commit": git_value("rev-parse", "HEAD"),
        "source_tree": git_value("rev-parse", "HEAD^{tree}"),
        "protected_counters": dict(COUNTERS),
        "physical_intervention": False,
    }
    (root / f"worker_{int(args.worker_index):02d}.done.json").write_text(json.dumps(done, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def pair_auc(items: list[dict[str, Any]], score_key: str, outcome_key: str) -> float | None:
    values = [(float(item[score_key]), int(item[outcome_key])) for item in items if item.get(score_key) is not None and item.get(outcome_key) is not None and np.isfinite(float(item[score_key]))]
    positives = [score for score, y in values if y == 1]
    negatives = [score for score, y in values if y == 0]
    if not positives or not negatives:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives)
    return float(wins / (len(positives) * len(negatives)))


def average_precision(items: list[dict[str, Any]], score_key: str, outcome_key: str) -> float | None:
    values = [(float(item[score_key]), int(item[outcome_key])) for item in items if item.get(score_key) is not None and item.get(outcome_key) is not None and np.isfinite(float(item[score_key]))]
    positives = sum(y for _, y in values)
    if positives == 0:
        return None
    values.sort(key=lambda item: (-item[0],))
    seen = 0
    total = 0.0
    for index, (_, y) in enumerate(values, 1):
        seen += y
        if y:
            total += seen / index
    return float(total / positives)


def parent_auc_map(items: list[dict[str, Any]], score_key: str, outcome_key: str) -> dict[str, float]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        if item.get(score_key) is not None and item.get(outcome_key) is not None:
            groups[str(item["identity"])].append(item)
    return {key: value for key, group in groups.items() if (value := pair_auc(group, score_key, outcome_key)) is not None}


def top_metrics(items: list[dict[str, Any]], score_key: str, outcome_key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        if item.get(score_key) is not None and item.get(outcome_key) is not None:
            groups[str(item["identity"])].append(item)
    output: dict[str, Any] = {}
    for k in (1, 3):
        selected_rates: list[float] = []
        random_rates: list[float] = []
        regrets: list[float] = []
        for group in groups.values():
            if len(group) < k or sum(int(item[outcome_key]) for item in group) == 0:
                continue
            ordered = sorted(group, key=lambda item: (-float(item[score_key]), str(item["probe_id"]), int(item["probe_step"])))
            y = [int(item[outcome_key]) for item in ordered]
            selected_rates.append(float(np.mean(y[:k])))
            random_rates.append(float(np.mean(y)))
            if k == 1:
                regrets.append(float(y[0] == 0))
        output[f"top{k}_parent_count"] = len(selected_rates)
        output[f"top{k}_selected_rate"] = None if not selected_rates else float(np.mean(selected_rates))
        output[f"top{k}_random_prevalence"] = None if not random_rates else float(np.mean(random_rates))
        output[f"top{k}_lift"] = None if not random_rates or float(np.mean(random_rates)) == 0.0 else float(np.mean(selected_rates) / np.mean(random_rates))
        if k == 1:
            output["zero_regret_rate"] = None if not regrets else float(1.0 - np.mean(regrets))
    return output


def metric_bundle(items: list[dict[str, Any]], score_key: str, outcome_key: str) -> dict[str, Any]:
    parent = parent_auc_map(items, score_key, outcome_key)
    suite_items: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        suite_items[str(item["suite"])].append(item)
    suite = {name: {"auc": pair_auc(group, score_key, outcome_key), "parent_macro_auc": (float(np.mean([parent[key] for key in parent if next(item for item in items if item["identity"] == key)["suite"] == name])) if any(next(item for item in items if item["identity"] == key)["suite"] == name for key in parent) else None)} for name, group in sorted(suite_items.items())}
    suite_aucs = [float(value["auc"]) for value in suite.values() if value["auc"] is not None]
    return {
        "row_count": len([item for item in items if item.get(score_key) is not None and item.get(outcome_key) is not None]),
        "auroc": pair_auc(items, score_key, outcome_key),
        "auprc": average_precision(items, score_key, outcome_key),
        "parent_macro_auc": None if not parent else float(np.mean(list(parent.values))),
        "pair_eligible_parent_count": len(parent),
        "top": top_metrics(items, score_key, outcome_key),
        "per_suite": suite,
        "loso_mean_auc": None if not suite_aucs else float(np.mean(suite_aucs)),
        "loso_worst_auc": None if not suite_aucs else float(np.min(suite_aucs)),
    }


def bootstrap_parent_macro(items: list[dict[str, Any]], score_key: str, outcome_key: str) -> dict[str, Any]:
    parent = parent_auc_map(items, score_key, outcome_key)
    keys = sorted(parent)
    if not keys:
        return {"replicates": 2000, "ci95": None}
    rng = np.random.RandomState(20260817)
    values = [float(np.mean([parent[key] for key in rng.choice(keys, size=len(keys), replace=True)])) for _ in range(2000)]
    return {"replicates": 2000, "seed": 20260817, "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]}


def aggregate(args: argparse.Namespace) -> int:
    root = Path(args.output_root).resolve()
    _, _, protocol = load_protocol(root)
    expected_rows, _ = verify_inputs(protocol)
    rows: list[dict[str, Any]] = []
    for index in range(int(args.worker_count)):
        done_path = root / f"worker_{index:02d}.done.json"
        path = root / f"worker_{index:02d}.jsonl"
        if not done_path.is_file() or not path.is_file():
            raise ValueError(f"WORKER_INCOMPLETE:{index}")
        done = load_json(done_path)
        if done.get("protected_counters") != COUNTERS or done.get("physical_intervention") is not False:
            raise ValueError(f"WORKER_BOUNDARY_INVALID:{index}")
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if len(rows) != len(expected_rows):
        raise ValueError(f"F0_AGGREGATE_ROW_COUNT_INVALID:{len(rows)}:{len(expected_rows)}")
    keys = [(row["stage"], row["canonical_parent_key"], row["probe_id"], int(row["probe_step"])) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("F0_AGGREGATE_DUPLICATE_ROWS")
    if any(row.get("protected_counters") != COUNTERS or row.get("physical_intervention") is not False or row.get("env_step_executed") is not False for row in rows):
        raise ValueError("F0_AGGREGATE_PROTECTED_BOUNDARY_INVALID")
    split_rows = [row for row in rows if row.get("gate", {}).get("eligible") and row.get("full_pgd_success") is not None]
    candidates: dict[str, dict[str, Any]] = {}
    for name, key in (("E0", "e0"), ("E1", "e1"), ("E3", "e3")):
        model = metric_bundle(split_rows, key, "full_pgd_success")
        factor = metric_bundle(split_rows, key, "u_factorized")
        dev = [row for row in split_rows if row.get("split") == "DEVTEST"]
        model_dev = metric_bundle(dev, key, "full_pgd_success")
        factor_dev = metric_bundle(dev, key, "u_factorized")
        suite_auc = [value["auc"] for value in factor_dev["per_suite"].values() if value["auc"] is not None]
        pass_gate = bool(
            model_dev["auroc"] is not None and model_dev["auroc"] >= 0.75 and
            model_dev["top"]["top1_lift"] is not None and model_dev["top"]["top1_lift"] >= 1.40 and
            factor_dev["parent_macro_auc"] is not None and factor_dev["parent_macro_auc"] >= 0.68 and
            factor_dev["top"]["top1_lift"] is not None and factor_dev["top"]["top1_lift"] >= 1.40 and
            factor_dev["top"]["top3_lift"] is not None and factor_dev["top"]["top3_lift"] >= 1.20 and
            all(float(value) >= 0.58 for value in suite_auc) and
            factor_dev["loso_mean_auc"] is not None and factor_dev["loso_mean_auc"] >= 0.60 and
            factor_dev["loso_worst_auc"] is not None and factor_dev["loso_worst_auc"] >= 0.55
        )
        candidates[name] = {
            "score_key": key,
            "all_gate_eligible": {"model_side": model, "factorized": factor},
            "devtest": {"model_side": model_dev, "factorized": factor_dev},
            "bootstrap_report_only": bootstrap_parent_macro(dev, key, "u_factorized"),
            "pass_gate": pass_gate,
        }
    passing = [name for name, value in candidates.items() if value["pass_gate"]]
    selected = None
    if passing:
        selected = sorted(passing, key=lambda name: (
            -(candidates[name]["devtest"]["factorized"]["top"]["top1_lift"] or -float("inf")),
            -(candidates[name]["devtest"]["factorized"]["parent_macro_auc"] or -float("inf")),
            {"E0": 0, "E1": 1, "E3": 3}[name],
        ))[0]
    decision = "PASS_F0_EXPLOITABILITY_SIGNAL" if selected else "STAGE_IX_NO_MODEL_SIDE_TIMING_SIGNAL"
    summary = {
        "schema": "STAGE_IX_F0_VIS_EXPLOITABILITY_RESULT_V1",
        "status": decision,
        "namespace": "STAGE_IX_FACTORIZED_PGD_TIMING_UTILITY",
        "source_commit": git_value("rev-parse", "HEAD"),
        "source_tree": git_value("rev-parse", "HEAD^{tree}"),
        "source_script_sha256": sha256_file(REPO / "scripts/stage_ix/run_stage_ix_f0.py"),
        "protocol_sha256": {name: sha256_file(root / name) for name in CONFIG_NAMES},
        "row_count": len(rows),
        "gate_eligible_row_count": sum(bool(row.get("gate", {}).get("eligible")) for row in rows),
        "label_known_row_count": sum(row.get("label_vphys_t5") is not None for row in rows),
        "full_pgd_success_count": sum(row.get("full_pgd_success") is True for row in rows),
        "factorized_proxy_positive_count": sum(row.get("u_factorized") is True for row in rows),
        "candidates": candidates,
        "passing_candidates": passing,
        "selected_candidate": selected,
        "physical_intervention": False,
        "env_steps_with_perturbed_action": 0,
        "protected_counters": dict(COUNTERS),
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    }
    (root / "F0_SPLIT.json").write_text(json.dumps({"schema": "STAGE_IX_F0_PARENT_GROUPED_SUITE_STRATIFIED_SPLIT_V1", "assignments": make_split(rows, protocol)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (root / "F0_ROWS.jsonl").open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda value: (value["stage"], value["canonical_parent_key"], int(value["probe_step"]), value["probe_id"])):
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    (root / "STAGE_IX_F0_VIS_EXPLOITABILITY.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    provenance = {
        "schema": "STAGE_IX_F0_PROVENANCE_V1",
        "source_commit": summary["source_commit"],
        "source_tree": summary["source_tree"],
        "source_script_sha256": summary["source_script_sha256"],
        "input_roots": protocol["population"]["sources"],
        "canonical_pgd_contract_sha256": sha256_file(root / CONFIG_NAMES[0]),
        "opportunity_gate_sha256": sha256_file(root / CONFIG_NAMES[1]),
        "f0_protocol_sha256": sha256_file(root / CONFIG_NAMES[2]),
        "physical_intervention": False,
        "protected_counters": dict(COUNTERS),
    }
    (root / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    status = "PASS_STAGE_IX_F0_EXPLOITABILITY" if selected else "STAGE_IX_NO_MODEL_SIDE_TIMING_SIGNAL"
    run_status = load_json(root / "F0_RUN_STATUS.json")
    run_status.update({"status": status, "protected_counters": dict(COUNTERS), "physical_intervention": False, "env_step_with_perturbed_action": False, "selected_candidate": selected})
    (root / "F0_RUN_STATUS.json").write_text(json.dumps(run_status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    seal_root(root, summary, status)
    print(json.dumps({"status": status, "selected_candidate": selected, "gate_eligible_rows": summary["gate_eligible_row_count"]}, sort_keys=True))
    return 0 if selected else 3


def seal_root(root: Path, summary: dict[str, Any], status: str) -> None:
    excluded = {"SHA256SUMS", "SHA256SUMS.sha256", "ROOT_SEAL.json", "ROOT_SEAL.sha256"}
    entries = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in sorted(root.rglob("*")) if path.is_file() and path.name not in excluded]
    sums = "\n".join(entries) + "\n"
    (root / "SHA256SUMS").write_text(sums, encoding="utf-8")
    sums_sha = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    seal = {
        "schema": "STAGE_IX_F0_ROOT_SEAL_V1",
        "status": status,
        "summary_sha256": sha256_file(root / "STAGE_IX_F0_VIS_EXPLOITABILITY.json"),
        "sha256sums_sha256": sums_sha,
        "row_count": summary["row_count"],
        "physical_intervention": False,
        "protected_counters": dict(COUNTERS),
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    }
    (root / "ROOT_SEAL.json").write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "ROOT_SEAL.sha256").write_text(f"{sha256_file(root / 'ROOT_SEAL.json')}  ROOT_SEAL.json\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--worker-count", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.init:
        init_root(Path(args.output_root).resolve())
        return 0
    if args.aggregate:
        return aggregate(args)
    if args.worker_index is None or args.worker_index < 0 or args.worker_index >= args.worker_count:
        parser.error("worker mode requires 0 <= --worker-index < --worker-count")
    return worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
