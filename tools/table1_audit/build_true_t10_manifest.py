from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from tools.table1_audit.common import (
    add_path_arg,
    atomic_create_text,
    canonical_digest,
    canonical_json,
    ensure_within_root,
    is_valid_sha256,
    job_key,
    load_json,
    load_jsonl,
    parent_key,
    reject_symlink,
    write_jsonl,
)
from tools.table1_audit.contracts import ConditionSpec, load_contract


REQUIRED_BOUND_SHA = {
    "runner_sha256",
    "worker_sha256",
    "bridge_sha256",
    "protocol_sha256",
    "metric_schema_sha256",
    "state_selection_sha256",
    "retry_policy_sha256",
    "detector_global_freeze_sha256",
    "victim_checkpoint_sha256",
}
REQUIRED_ATTACK_FIELDS = {
    "objective_id",
    "objective_semantics_version",
    "epsilon",
    "epsilon_space",
    "step_size",
    "optimization_steps",
    "initialization",
    "K",
    "timing_policy",
    "arm_lock_mode",
    "preprocessing_backend",
    "termination_policy",
    "no_emission_policy",
}
RESERVED_MANIFEST_KEYS = {
    "job_key",
    "condition_id",
    "source_clean_job_key",
    "source_clean_manifest_sha256",
    "source_clean_bundle",
    "output_dir",
}


def _require_frozen_bundle(bundle: Path, verification_path: Path) -> Path:
    final = load_json(bundle / "CONDITION_FREEZE_FINAL.json")
    if final.get("status") != "FROZEN":
        raise SystemExit("refusing TRUE_T10 manifest: clean bundle final status is not FROZEN")
    verification = load_json(verification_path)
    if not verification.get("verification_pass"):
        raise SystemExit("refusing TRUE_T10 manifest: final envelope verification did not pass")
    from tools.table1_audit.common import sha256_file

    if verification.get("final_envelope_sha256") != sha256_file(bundle / "CONDITION_FREEZE_FINAL.json"):
        raise SystemExit("refusing TRUE_T10 manifest: final verification does not bind final envelope")
    candidate = Path(final.get("candidate_bundle", "")).resolve()
    if not candidate.exists():
        raise SystemExit("refusing TRUE_T10 manifest: candidate bundle missing")
    return candidate


def _check_structure(rows: list[dict], spec: dict) -> None:
    folds = {str(f) for f in spec["fields"]["folds"]}
    states = {(str(f), str(s)) for f, ss in spec["fields"]["states_by_fold"].items() for s in ss}
    det = {str(x) for x in spec["fields"]["detector_seeds"]}
    pert = {str(x) for x in spec["fields"]["perturbation_seeds"]}
    parents = Counter(parent_key(r) for r in rows)
    if len(rows) != 162 or len(parents) != 54 or any(v != 3 for v in parents.values()):
        raise SystemExit("refusing TRUE_T10 manifest: CLEAN source is not 162 rows / 54 parents / 3 replicates")
    for r in rows:
        if str(r.get("fold")) not in folds:
            raise SystemExit("refusing TRUE_T10 manifest: wrong fold domain")
        if (str(r.get("fold")), str(r.get("state_id"))) not in states:
            raise SystemExit("refusing TRUE_T10 manifest: wrong state-selection domain")
        if str(r.get("detector_seed")) not in det:
            raise SystemExit("refusing TRUE_T10 manifest: wrong detector seed domain")
        if str(r.get("perturbation_seed")) not in pert:
            raise SystemExit("refusing TRUE_T10 manifest: wrong perturbation seed domain")
    if len({job_key(r) for r in rows}) != 162:
        raise SystemExit("refusing TRUE_T10 manifest: source clean job keys are not unique")


def _output_checks(args: argparse.Namespace, spec: dict, bundle: Path) -> None:
    out_root = args.output_root.resolve()
    if not args.output_root.is_absolute():
        raise SystemExit("refusing TRUE_T10 manifest: output root must be absolute")
    if out_root.exists() or out_root.is_symlink():
        raise SystemExit("refusing TRUE_T10 manifest: output root already exists or is symlink")
    ensure_within_root(out_root, Path(spec["allowed_output_root"]).resolve())
    ensure_within_root(args.output_manifest.resolve(), Path(spec["allowed_manifest_root"]).resolve())
    try:
        out_root.relative_to(bundle)
        raise SystemExit("refusing TRUE_T10 manifest: output root is inside clean bundle")
    except ValueError:
        pass
    for occupied in spec.get("occupied_output_roots", []):
        occ = Path(occupied).resolve()
        if out_root == occ or _is_relative_to(out_root, occ) or _is_relative_to(occ, out_root):
            raise SystemExit("refusing TRUE_T10 manifest: output root overlaps registered root")
    if args.output_manifest.exists() or args.output_manifest.is_symlink():
        raise SystemExit(f"refusing overwrite: {args.output_manifest}")
    reject_symlink(out_root.parent)


def build(args: argparse.Namespace) -> dict:
    final_bundle = args.clean_bundle.resolve()
    bundle = _require_frozen_bundle(final_bundle, args.bundle_verification.resolve())
    spec_loaded = load_contract(args.authorized_condition_spec.resolve(), ConditionSpec)
    spec = spec_loaded.data
    if args.write and spec.get("status") != "AUTHORIZED_FOR_MANIFEST_GENERATION":
        raise SystemExit("refusing TRUE_T10 --write: condition spec is not authorized")
    if spec.get("status") not in {"AUTHORIZED_FOR_MANIFEST_GENERATION", "DRAFT_NOT_AUTHORIZED"}:
        raise SystemExit("refusing TRUE_T10 manifest: unknown condition spec status")
    missing_bound = sorted(REQUIRED_BOUND_SHA - set(spec["bound_contract_sha256"]))
    if missing_bound:
        raise SystemExit(f"refusing TRUE_T10 manifest: missing bound SHA keys {missing_bound}")
    attack_fields = spec["fields"].get("attack", {})
    missing_attack = sorted(REQUIRED_ATTACK_FIELDS - set(attack_fields))
    if missing_attack:
        raise SystemExit(f"refusing TRUE_T10 manifest: missing attack spec keys {missing_attack}")
    reserved_overlap = sorted(RESERVED_MANIFEST_KEYS & set(attack_fields))
    if reserved_overlap:
        raise SystemExit(f"refusing TRUE_T10 manifest: attack spec overrides reserved keys {reserved_overlap}")
    for key, value in spec["bound_contract_sha256"].items():
        if value in {"MISSING", "UNVERIFIED", "SERVER_SNAPSHOT_REQUIRED"} or not is_valid_sha256(value):
            raise SystemExit(f"refusing TRUE_T10 manifest: unresolved or invalid bound SHA {key}")
    _output_checks(args, spec, final_bundle)
    inventory = load_json(bundle / "RESULT_INVENTORY.json")
    rows = inventory.get("rows") or load_jsonl(bundle / "MANIFEST.jsonl")
    _check_structure(rows, spec)

    allow = list(spec["clean_identity_allowlist"])
    deny = set(spec["clean_result_denylist"])
    manifest_rows = []
    removed = set()
    seen_out = set()
    for r in sorted(rows, key=lambda x: (str(x.get("fold")), str(x.get("state_id")), str(x.get("detector_seed")), str(x.get("perturbation_seed")))):
        leaked = sorted(k for k in r if k in deny)
        removed.update(leaked)
        clean_id = {k: r[k] for k in allow if k in r}
        out = args.output_root.resolve() / f"fold_{r.get('fold')}" / f"state_{r.get('state_id')}" / f"det_seed_{r.get('detector_seed')}" / f"pert_seed_{r.get('perturbation_seed')}"
        if str(out) in seen_out:
            raise SystemExit("refusing TRUE_T10 manifest: shared output root")
        seen_out.add(str(out))
        row = {
            **clean_id,
            "job_key": f"{spec['condition_id']}::{job_key(r)}",
            "condition_id": spec["condition_id"],
            "source_clean_job_key": job_key(r),
            "source_clean_manifest_sha256": inventory.get("manifest_sha256"),
            "source_clean_bundle": str(bundle),
            "output_dir": str(out),
            **attack_fields,
            **spec["bound_contract_sha256"],
        }
        manifest_rows.append(row)
    manifest_text = "".join(json.dumps(r, sort_keys=True, ensure_ascii=True) + "\n" for r in manifest_rows)
    manifest_sha = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    result = {
        "dry_run": not args.write,
        "condition_id": spec["condition_id"],
        "would_be_manifest_sha256": manifest_sha,
        "bound_contract_sha256": spec["bound_contract_sha256"],
        "row_count": len(manifest_rows),
        "parent_count": len({parent_key(r) for r in manifest_rows}),
        "field_allowlist": allow,
        "fields_removed_from_clean": sorted(removed),
        "output_root_collision_report": "none",
        "first_row": manifest_rows[0],
        "last_row": manifest_rows[-1],
        "per_fold_counts": dict(sorted(Counter(str(r.get("fold")) for r in manifest_rows).items())),
    }
    if args.write:
        atomic_create_text(args.output_manifest, manifest_text)
    return result


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare TRUE_T10 manifest from a verified frozen CLEAN bundle.")
    add_path_arg(ap, "--clean-bundle", required=True)
    add_path_arg(ap, "--bundle-verification", required=True)
    add_path_arg(ap, "--authorized-condition-spec", required=True)
    add_path_arg(ap, "--output-root", required=True)
    add_path_arg(ap, "--output-manifest", required=True)
    ap.add_argument("--write", action="store_true", help="Actually write manifest. Default is dry-run preview.")
    args = ap.parse_args()
    print(canonical_json(build(args)), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
