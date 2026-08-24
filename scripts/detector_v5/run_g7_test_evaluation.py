"""G7: ONE-TIME test evaluation. Reads test split exactly once.

Frozen: checkpoint (median MCC seed 20260717), theta_physical=0.30,
scheduler theta=0.25 persistence=5 cooldown=True.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts" / "detector_v5", ROOT / "n5" / "phase3_student"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from audit_r3_contact_input import sha256_file, verify_seal
from gripper_attack.seal_utils import rename_noreplace
from run_r3_heldout_development import (
    _load_g2, _load_splits, _batch, _predict, _head_probability,
    _binary_metrics, _event_metrics, _safe_auc, _safe_auprc,
    _teacher_critical_spans, _candidate_spans, _event_label,
    RISK_DIRECTION, EVENT_DEFINITION,
)
from run_r3_full670_student_development import _load_records, _load_model, _loss

FORBIDDEN = {"cal", "check", "g10", "t2r-d", "protected", "attack"}
THETA_PHYSICAL = 0.30  # frozen from validation
THETA_SCHEDULER = 0.25  # frozen from G6-S
PERSISTENCE = 5
COOLDOWN = True


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _write_seal(p: Path) -> str:
    files = sorted(x for x in p.rglob("*") if x.is_file() and x.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})
    (p / "SHA256SUMS").write_text("".join(f"{sha256_file(x)}  {x.relative_to(p).as_posix()}\n" for x in files), encoding="utf-8")
    d = sha256_file(p / "SHA256SUMS")
    (p / "SHA256SUMS.sha256").write_text(f"{d}  SHA256SUMS\n", encoding="utf-8")
    return d


def compute_scheduler_metrics_test(
    predictions: list[dict[str, Any]], theta: float, persistence: int, cooldown: bool,
) -> dict[str, Any]:
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_episode[row["episode_id"]].append(row)

    teacher_critical_total = 0
    teacher_detected = 0
    false_emits = 0
    total_episodes = 0
    episodes_with_false = 0

    for eid, rows in by_episode.items():
        total_episodes += 1
        steps = len(rows)

        # Teacher critical spans
        tc_spans = _teacher_critical_spans_raw(rows, steps)

        teacher_critical_total += len(tc_spans)

        # Candidate spans
        cand_spans = _candidate_spans_raw(rows, steps)

        for cs, ce in cand_spans:
            prob_above = []
            for i in range(cs, ce + 1):
                prob = rows[i].get("physical_criticality", {}).get("probability", 0)
                prob_above.append(isinstance(prob, (int, float)) and prob >= theta)

            fired_step = -1
            for i in range(len(prob_above) - persistence + 1):
                if all(prob_above[i:i + persistence]):
                    fired_step = cs + i
                    break

            if fired_step < 0:
                continue

            has_tc = any(max(ts, cs) <= min(te, ce) for ts, te in tc_spans)

            if has_tc:
                teacher_detected += 1
            else:
                false_emits += 1
                episodes_with_false += 1

            if cooldown:
                break

    e2e = teacher_detected / teacher_critical_total if teacher_critical_total else None
    return {
        "theta": theta, "persistence": persistence, "cooldown": cooldown,
        "teacher_critical_events": teacher_critical_total,
        "teacher_detected": teacher_detected,
        "end_to_end_recall": e2e,
        "false_emits": false_emits,
        "false_emits_per_episode": false_emits / total_episodes if total_episodes else None,
        "episodes_with_false": episodes_with_false,
        "total_episodes": total_episodes,
    }


def _teacher_critical_spans_raw(rows: list[dict[str, Any]], steps: int) -> list[tuple[int, int]]:
    spans = []
    start = None
    for i, r in enumerate(rows):
        ph = r.get("physical_criticality", {})
        is_true = isinstance(ph, dict) and ph.get("known") and ph.get("target") == 1
        if is_true and start is None:
            start = i
        if not is_true and start is not None:
            spans.append((start, i - 1))
            start = None
    if start is not None:
        spans.append((start, steps - 1))
    return spans


def _candidate_spans_raw(rows: list[dict[str, Any]], steps: int) -> list[tuple[int, int]]:
    spans = []
    start = None
    for i, r in enumerate(rows):
        if r.get("candidate_close"):
            if start is None:
                start = i
        else:
            if start is not None:
                spans.append((start, i - 1))
                start = None
    if start is not None:
        spans.append((start, steps - 1))
    return spans


def run(
    g2_root: Path, checkpoint_root: Path, scheduler_root: Path,
    norm_r2_root: Path, output_root: Path,
) -> dict[str, Any]:
    if subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip():
        raise ValueError("clean checkout required")

    commit = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")

    # Validate inputs
    g2_root = g2_root.resolve(strict=True)
    checkpoint_root = checkpoint_root.resolve(strict=True)
    scheduler_root = scheduler_root.resolve(strict=True)
    norm_r2_root = norm_r2_root.resolve(strict=True)

    g2_seal = verify_seal(g2_root)
    cp_seal = verify_seal(checkpoint_root)
    sc_seal = verify_seal(scheduler_root)
    nm_seal = verify_seal(norm_r2_root)

    # Load G2 and G1
    transition, g2_binding = _load_g2(g2_root)
    scheduler_data = json.loads((scheduler_root / "SCHEDULER_RECEIPT.json").read_text(encoding="utf-8"))
    norm_data = json.loads((norm_r2_root / "NORMALIZATION.json").read_text(encoding="utf-8"))

    # Load TEST split
    family = "episode"
    split_ids, split_meta = _load_splits(
        Path(g2_binding["g1_root"]), family, g2_binding["split_manifests"],
    )
    test_ids = split_ids["episode_test"]
    test_ids_set = set(test_ids)

    # Load records (test + train for normalization reference only)
    print(f"Loading {len(test_ids)} test identities (ONE-TIME READ)...")
    records, record_binding = _load_records(
        Path(transition["t4"]["root"]),
        allow_descendant_snapshot=True,
        identity_allowlist=test_ids_set,
        skip_source_binding=True,
    )
    records_by_id = {r["identity"]: r for r in records}

    # Use G1-R2 normalization
    r2_train = norm_data.get("episode_heldout", {}).get("train", {})
    mean = np.asarray(r2_train["mean"], dtype=np.float64)
    std = np.asarray(r2_train["std"], dtype=np.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    model_cls = _load_model()
    model = model_cls(input_dim=25, hidden=64, short_rf=32, long_rf=128, dropout=0.0).to(device)
    cp = torch.load(str(checkpoint_root / "checkpoint.pt"), map_location=device, weights_only=True)
    model.load_state_dict(cp["model"], strict=True)
    model.eval()

    # Predict
    test_batch = _batch(records_by_id, test_ids, mean, std, device)
    probabilities = _predict(model, test_batch, test_ids, records_by_id, device)

    # Per-head metrics (physical_criticality only)
    head = "physical_criticality"
    head_prob = _head_probability(probabilities, test_ids, head)

    # Event metrics at validation threshold
    event_metrics_test = _event_metrics(records, test_ids, head, head_prob, THETA_PHYSICAL)

    # Step metrics
    ys, ss = [], []
    for item in records:
        mask = np.asarray(item["masks"][head], dtype=bool)
        ys.append(np.asarray(item["targets"][head], dtype=np.int64)[mask])
        ss.append(head_prob[item["identity"]][mask])
    y = np.concatenate(ys) if ys else np.zeros(0, dtype=np.int64)
    s = np.concatenate(ss) if ss else np.zeros(0, dtype=np.float64)
    step_metrics_test = _binary_metrics(y, s, THETA_PHYSICAL)

    # Scheduler metrics
    # Build predictions list in same format as scheduler expects
    pred_list = []
    for item in records:
        eid = item["identity"]
        for step_idx in range(len(item["features"])):
            prob = float(head_prob.get(eid, np.zeros(1))[step_idx]) if eid in head_prob else 0.0
            known = bool(item["masks"][head][step_idx])
            target = int(item["targets"][head][step_idx]) if known else 0
            pred_list.append({
                "episode_id": eid,
                "step": step_idx,
                "split": "episode_test",
                "candidate_close": bool(item["candidate_close"][step_idx]),
                "physical_criticality": {
                    "probability": prob,
                    "known": known,
                    "target": target,
                },
            })
    sched_metrics = compute_scheduler_metrics_test(pred_list, THETA_SCHEDULER, PERSISTENCE, COOLDOWN)

    # G7 gate check
    auc_val = step_metrics_test.get("auroc")
    mcc_val = step_metrics_test.get("mcc")
    bacc_val = step_metrics_test.get("balanced_accuracy")
    e2e_val = event_metrics_test.get("end_to_end_critical_recall")
    ccr_val = event_metrics_test.get("candidate_conditioned_recall")
    ev_rec = event_metrics_test.get("event_recall")

    gates = {
        "auroc_ge_0.85": (auc_val or 0) >= 0.85,
        "mcc_ge_0.50": (mcc_val or -1) >= 0.50,
        "teacher_event_recall_ge_0.65": (ev_rec or 0) >= 0.65,
        "candidate_conditioned_recall_ge_0.60": (ccr_val or 0) >= 0.60,
        "end_to_end_recall_ge_0.50": (e2e_val or 0) >= 0.50,
        "bacc_ge_0.70": (bacc_val or 0) >= 0.70,
    }
    all_gates_pass = all(gates.values())

    # Build test transition
    payload = {
        "schema": "DETECTOR_V2_TEST_TRANSITION_V1",
        "status": "PASS_TEST_CANDIDATE" if all_gates_pass else "FAIL_TEST_GATE",
        "code_snapshot": {"commit": commit, "tree": tree},
        "consumption_count": 1,
        "test_read_once": True,
        "frozen_checkpoint": str(checkpoint_root),
        "frozen_checkpoint_seal": cp_seal["sha256sums_sha256"],
        "frozen_scheduler": str(scheduler_root),
        "frozen_scheduler_seal": sc_seal["sha256sums_sha256"],
        "frozen_theta_physical": THETA_PHYSICAL,
        "frozen_theta_scheduler": THETA_SCHEDULER,
        "frozen_persistence": PERSISTENCE,
        "frozen_cooldown": COOLDOWN,
        "g2_root": str(g2_root),
        "g2_seal": g2_seal["sha256sums_sha256"],
        "normalization_r2_root": str(norm_r2_root),
        "normalization_r2_seal": nm_seal["sha256sums_sha256"],
        "test_split": "episode_test",
        "test_identity_count": len(test_ids),
        "per_head_metrics": {
            "physical_criticality": {
                "theta": THETA_PHYSICAL,
                "step": step_metrics_test,
                "event": event_metrics_test,
            },
        },
        "scheduler_metrics": sched_metrics,
        "gates": gates,
        "all_gates_pass": all_gates_pass,
        "protected_reads": 0,
        "attack_authorized": False,
        "runner_sha256": sha256_file(Path(__file__)),
    }

    # Seal output
    if output_root.exists():
        raise FileExistsError(str(output_root))
    staging = output_root.with_name(f".{output_root.name}.staging.{os.getpid()}")
    staging.mkdir(parents=True)
    try:
        (staging / "G7_TEST_TRANSITION.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "g7_test_predictions.jsonl").write_text(
            "\n".join(json.dumps(r, separators=(",", ":")) for r in pred_list) + "\n", encoding="utf-8",
        )
        torch.save({
            "checkpoint_sha256": sha256_file(checkpoint_root / "checkpoint.pt"),
            "model_config": "physical_only_N5MultiHeadStudent",
            "theta_physical": THETA_PHYSICAL,
            "consumption_count": 1,
        }, staging / "g7_state.pt")
        digest = _write_seal(staging)
        rename_noreplace(staging, output_root)
    except Exception:
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
        raise

    payload["sha256sums_sha256"] = digest
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g2-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True, help="20260717 physical_only root")
    parser.add_argument("--scheduler-root", type=Path, required=True)
    parser.add_argument("--norm-r2-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.g2_root, args.checkpoint_root, args.scheduler_root, args.norm_r2_root, args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result.get("all_gates_pass"):
        print("\n*** G7 TEST GATE FAILED — stopping ***")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
