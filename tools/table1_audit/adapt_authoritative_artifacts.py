from __future__ import annotations

import argparse
from pathlib import Path

from tools.table1_audit.common import add_path_arg, canonical_digest, load_json, sha256_file, write_json


def adapt_global_freeze(path: Path) -> dict:
    src = load_json(path)
    return {
        "schema_version": "global_freeze.v1",
        "source_path_metadata": str(path),
        "source_sha256": sha256_file(path),
        "victim_checkpoint_sha256": {},
        "detector_checkpoint_sha256": {f"{c['fold']}|{c['seed']}": c["sha256"] for c in src["checkpoints"]},
        "source_gate": src.get("gate"),
    }


def adapt_state_selection(protocol_path: Path, *, states_per_fold: int = 2) -> dict:
    src = load_json(protocol_path)
    folds = [f for f in sorted(src["fold_matrix"]) if f != "00"]
    return {
        "schema_version": "state_selection.v1",
        "source_path_metadata": str(protocol_path),
        "source_sha256": sha256_file(protocol_path),
        "folds": folds,
        "states_by_fold": {f: [str(i) for i in range(states_per_fold)] for f in folds},
        "tasks_by_fold": {f: [str(src["fold_matrix"][f]["test"])] for f in folds},
        "detector_seeds": src["training"]["seeds"],
        "perturbation_seeds": [0, 1, 2],
    }


def adapt_metric_schema(path: Path) -> dict:
    src = load_json(path)
    return {
        "schema_version": "metric_schema_adapter.v1",
        "source_path_metadata": str(path),
        "source_sha256": sha256_file(path),
        "source_gate": src.get("gate"),
        "canonical_digest": canonical_digest(src),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Adapt read-only authoritative Table 1 artifacts into audit contracts.")
    add_path_arg(ap, "--global-freeze")
    add_path_arg(ap, "--protocol-freeze")
    add_path_arg(ap, "--metric-schema")
    add_path_arg(ap, "--out-dir", required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.global_freeze:
        write_json(args.out_dir / "global_freeze.adapter.json", adapt_global_freeze(args.global_freeze.resolve()))
    if args.protocol_freeze:
        write_json(args.out_dir / "state_selection.adapter.json", adapt_state_selection(args.protocol_freeze.resolve()))
    if args.metric_schema:
        write_json(args.out_dir / "metric_schema.adapter.json", adapt_metric_schema(args.metric_schema.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
