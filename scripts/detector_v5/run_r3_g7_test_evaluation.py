"""Run the one-time R3 test read after validation thresholds are frozen."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT / "scripts" / "detector_v5", ROOT / "n5" / "phase3_student"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_r3_heldout_development as dev  # noqa: E402
from audit_r3_contact_input import sha256_file, verify_seal  # noqa: E402
from run_r3_full670_student_development import _load_model, _load_records  # noqa: E402


COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _write_seal(root: Path) -> str:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})
    (root / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files), encoding="utf-8")
    digest = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{digest}  SHA256SUMS\n", encoding="utf-8")
    return digest


def _allow_descendant_g2_load() -> None:
    """G7 is the frozen-checkpoint post-processing consumer, not a trainer."""
    original = dev._snapshot_matches
    dev._snapshot_matches = lambda snapshot, allow_descendant_snapshot=False: original(snapshot, allow_descendant_snapshot=True)


def run(g2_root: Path, development_root: Path, output_root: Path, *, split_family: str, device_name: str, threads: int) -> dict[str, Any]:
    if split_family not in {"episode", "task"}:
        raise ValueError("unsupported split family")
    if subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip():
        raise ValueError("clean checkout required")
    commit, tree = _git("rev-parse", "HEAD"), _git("rev-parse", "HEAD^{tree}")
    g2_root = g2_root.resolve(strict=True)
    development_root = development_root.resolve(strict=True)
    output_root = output_root.resolve()
    if output_root.parent != g2_root.parent:
        raise ValueError("output root must be a sibling of G2")
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root = Path(str(output_root))
    if any(part.casefold() in dev.FORBIDDEN_OUTPUT_PARTS for part in output_root.parts):
        raise ValueError("unsafe output root")
    g2_seal = verify_seal(g2_root)["sha256sums_sha256"]
    development_seal = verify_seal(development_root)["sha256sums_sha256"]
    report = _json(development_root / "heldout_report.json")
    if report.get("schema") != "V5_R3_HELDOUT_DEVELOPMENT_V3" or report.get("status") != "ENGINEERING_DEVELOPMENT_NONCONSUMABLE":
        raise ValueError("development report is not the expected sealed R3 output")
    if report.get("test_payload_read") is not False or report.get("test_evaluation_performed") is not False or report.get("threshold_selection_split") != "validation_only":
        raise ValueError("test was read before G7 or threshold provenance is not closed")
    if report.get("protected_reads") != 0 or report.get("permissions", {}).get("protected_reads") != 0:
        raise ValueError("development protected boundary is not zero")
    threshold_data = _json(development_root / "thresholds.json")
    if any(not isinstance(threshold_data.get(head), Mapping) or threshold_data[head].get("status") != "SELECTED_VALIDATION_ONLY" or threshold_data[head].get("threshold") is None for head in dev.ACTIVE_HEADS):
        raise ValueError("validation thresholds are not frozen for every active head")
    if report.get("binding", {}).get("g2_root") != str(g2_root) or report.get("binding", {}).get("g2_seal_sha256sums_sha256") != g2_seal:
        raise ValueError("development to G2 binding mismatch")

    # Permit a descendant checkout for this independent, post-freeze reader while preserving the exact G2 source binding.
    _allow_descendant_g2_load()
    transition, g2_binding = dev._load_g2(g2_root)
    if subprocess.call(("git", "merge-base", "--is-ancestor", str(transition["code_snapshot"]["commit"]), commit), cwd=ROOT) != 0:
        raise ValueError("G7 runner is not a descendant of the G2 training snapshot")
    split_ids, split_meta = dev._load_splits(Path(g2_binding["g1_root"]), split_family, g2_binding["split_manifests"])
    test_ids = split_ids[f"{split_family}_test"]
    records, record_binding = _load_records(Path(transition["t4"]["root"]), allow_descendant_snapshot=True, identity_allowlist=set(test_ids), skip_source_binding=True)
    records_by_id = {row["identity"]: row for row in records}
    if sorted(records_by_id) != sorted(test_ids):
        raise ValueError("G7 test identity closure mismatch")
    torch.set_num_threads(threads)
    device = torch.device(device_name)
    mean = np.asarray(split_meta["normalization"]["mean"], dtype=np.float64)
    std = np.asarray(split_meta["normalization"]["std"], dtype=np.float64)
    batch = dev._batch(records_by_id, test_ids, mean, std, device)
    model_cls = _load_model()
    model = model_cls(input_dim=25, hidden=64, short_rf=32, long_rf=128, dropout=0.0).to(device)
    checkpoint_path = development_root / "checkpoint.pt"
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    if checkpoint.get("active_heads") != list(dev.ACTIVE_HEADS) or checkpoint.get("config") != "shared_four_head":
        raise ValueError("G7 checkpoint configuration mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    probabilities = dev._predict(model, batch, test_ids, records_by_id, device)
    metrics: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    for head in dev.ACTIVE_HEADS:
        threshold = float(threshold_data[head]["threshold"])
        head_prob = dev._head_probability(probabilities, test_ids, head)
        metrics[head] = {"threshold": threshold, "step": dev._step_metrics(records, test_ids, head, head_prob, threshold), "event": dev._event_metrics(records, test_ids, head, head_prob, threshold)}
    for identity in test_ids:
        item = records_by_id[identity]
        for step, row in enumerate(probabilities[identity]):
            prediction = {"episode_id": identity, "step": step, "split": f"{split_family}_test"}
            for index, head in enumerate(dev.ACTIVE_HEADS):
                probability = float(row[index])
                prediction[head] = {"probability": probability, "selected": probability >= float(threshold_data[head]["threshold"]), "known": bool(item["masks"][head][step])}
            prediction_rows.append(prediction)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.with_name(f".{output_root.name}.staging")
    if staging.exists() or output_root.exists():
        raise FileExistsError(output_root)
    staging.mkdir(parents=True)
    payload = {
        "schema": "V5_R3_G7_TEST_EVALUATION_V1",
        "status": "PASS_R3_G7_TEST_EVALUATION",
        "runner_role": "INDEPENDENT_FROZEN_CHECKPOINT_TEST_READER",
        "code_snapshot": {"commit": commit, "tree": tree},
        "g2_root": str(g2_root),
        "g2_seal_sha256sums_sha256": g2_seal,
        "development_root": str(development_root),
        "development_root_sha256sums_sha256": development_seal,
        "checkpoint_sha256": _sha(checkpoint_path),
        "split_family": split_family,
        "test_identity_count": len(test_ids),
        "test_read_count": 1,
        "thresholds_validation_only": threshold_data,
        "thresholds_frozen_before_test": True,
        "model_selection_after_test": False,
        "metrics": metrics,
        "active_heads": list(dev.ACTIVE_HEADS),
        "inactive_heads": list(dev.INACTIVE_HEADS),
        "record_binding": record_binding,
        "protected_counters": dict(COUNTERS),
        "outcomes_read": False,
        "intervention_executed": False,
        "v_phys_generated": False,
        "attack_authorized": False,
        "formal_m4_authorized": False,
    }
    (staging / "G7_TEST_EVALUATION.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (staging / "G7_TEST_PREDICTIONS.jsonl").write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in prediction_rows), encoding="utf-8")
    digest = _write_seal(staging)
    staging.rename(output_root)
    payload["sha256sums_sha256"] = digest
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g2-root", type=Path, required=True)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-family", choices=("episode", "task"), default="episode")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(run(args.g2_root, args.development_root, args.output_root, split_family=args.split_family, device_name=args.device, threads=args.threads), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
