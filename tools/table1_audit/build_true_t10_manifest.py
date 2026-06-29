from __future__ import annotations

import argparse
from pathlib import Path

from tools.table1_audit.common import add_path_arg, canonical_json, job_key, load_json, load_jsonl, parent_key, write_jsonl


def build(args: argparse.Namespace) -> dict:
    bundle = args.clean_bundle.resolve()
    freeze = load_json(bundle / "CONDITION_FREEZE.json")
    if freeze.get("status") != "FROZEN":
        raise SystemExit("refusing TRUE_T10 manifest: clean bundle is not frozen")
    inventory = load_json(bundle / "RESULT_INVENTORY.json")
    rows = inventory.get("rows") or load_jsonl(bundle / "MANIFEST.jsonl")
    if len(rows) != 162:
        raise SystemExit(f"refusing TRUE_T10 manifest: expected 162 rows, got {len(rows)}")
    if len({parent_key(r) for r in rows}) != 54:
        raise SystemExit("refusing TRUE_T10 manifest: expected 54 parents")
    for name, value in {
        "runner_sha256": args.runner_sha256,
        "config_sha256": args.config_sha256,
        "metric_schema_sha256": args.metric_schema_sha256,
    }.items():
        if not value or value in {"MISSING", "UNVERIFIED"}:
            raise SystemExit(f"refusing TRUE_T10 manifest: missing {name}")

    out_root = args.output_root
    manifest_rows = []
    seen = set()
    for r in sorted(rows, key=lambda x: (str(x.get("fold")), int(x.get("state_id")), int(x.get("detector_seed")), int(x.get("perturbation_seed")))):
        out = out_root / f"fold_{r.get('fold')}" / f"state_{r.get('state_id')}" / f"det_seed_{r.get('detector_seed')}" / f"pert_seed_{r.get('perturbation_seed')}"
        if str(out) in seen:
            raise SystemExit("refusing TRUE_T10 manifest: duplicate output root")
        if out.exists():
            raise SystemExit(f"refusing TRUE_T10 manifest: output already exists: {out}")
        seen.add(str(out))
        row = dict(r)
        row.update({
            "condition": args.condition_id,
            "source_clean_job_key": job_key(r),
            "source_clean_manifest_sha256": inventory.get("manifest_sha256"),
            "attack_objective": "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
            "K": 10,
            "no_emission_policy": "ITT_RETAIN",
            "runner_sha256": args.runner_sha256,
            "config_sha256": args.config_sha256,
            "metric_schema_sha256": args.metric_schema_sha256,
            "output_dir": str(out),
        })
        manifest_rows.append(row)
    result = {
        "dry_run": not args.write,
        "condition_id": args.condition_id,
        "row_count": len(manifest_rows),
        "parent_count": len({parent_key(r) for r in manifest_rows}),
        "output_manifest": str(args.output_manifest),
        "preview_first_job": manifest_rows[0],
    }
    if args.write:
        if args.output_manifest.exists():
            raise SystemExit(f"refusing overwrite: {args.output_manifest}")
        write_jsonl(args.output_manifest, manifest_rows)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare TRUE_T10 manifest from a frozen CLEAN bundle.")
    add_path_arg(ap, "--clean-bundle", required=True)
    add_path_arg(ap, "--output-root", required=True)
    add_path_arg(ap, "--output-manifest", required=True)
    ap.add_argument("--condition-id", default="TRUE_T10")
    ap.add_argument("--runner-sha256", required=True)
    ap.add_argument("--config-sha256", required=True)
    ap.add_argument("--metric-schema-sha256", required=True)
    ap.add_argument("--write", action="store_true", help="Actually write manifest. Default is dry-run preview.")
    args = ap.parse_args()
    print(canonical_json(build(args)), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
