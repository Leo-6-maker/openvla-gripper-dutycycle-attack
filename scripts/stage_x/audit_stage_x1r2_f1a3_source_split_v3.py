"""CPU-only F1-A3 source-history reconciliation and population freeze.

The audit is deliberately identity/provenance-only.  It scans the tracked
source plane at every local Git ref plus matching Git-history diffs, records
only identity keys and source metadata, and never reads model scores,
outcomes, simulator payloads, protected data, or Eval160 data.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/STAGE_X_X1R2_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3_20260821"

PI_COMMENT_ID = 5368872540
PI_ATTACHMENT_SHA256 = "240d5ba79354b89f5f98d2e1d81e58dc3ffe4689898675006cf0e980f04c06eb"
SPLIT_SALT = "STAGE_X_X1R2_F1A3_SOURCE_V3_SPLIT_20260821"

SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
ROLES = ("BRIDGE_V3", "DEV_V3", "C_CANARY_V3")
KEY_RE = re.compile(r"libero_(?:10|goal|object|spatial)/task_(?:0\d|1\d)/state_(?:0\d|1\d)")
GIT_GREP_PATTERN = r"libero_(10|goal|object|spatial)/task_[0-9]{2}/state_(0[0-9]|1[0-9])"
KEYS = {
    f"{suite}/task_{task:02d}/state_{state:02d}"
    for suite in SUITES
    for task in range(10)
    for state in range(20)
}

CLASS_PRISTINE = "V3_PRISTINE_NO_RELEVANT_IDENTITY_EXPOSURE"
CLASS_DETECTOR = "V3_DETECTOR_TRAIN_ONLY_ALLOWED_WITH_FLAG"
CLASS_STATIC = "V3_STATIC_MENTION_ONLY_NOT_EXECUTION"
CLASS_HARD = "V3_HARD_EXECUTION_EXPOSURE_EXCLUDE"
CLASS_UNRESOLVED = "V3_UNRESOLVED_EXPOSURE_EXCLUDE"
CLASSES = (CLASS_PRISTINE, CLASS_DETECTOR, CLASS_STATIC, CLASS_HARD, CLASS_UNRESOLVED)


def run_git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def canonical_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(canonical_bytes(path)).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)


def is_audit_artifact(path: str) -> bool:
    lower = path.lower()
    return any(
        marker in lower
        for marker in (
            "stage_x_x1r2_f1a2_source_leakage_and_population_freeze",
            "stage_x_x1r2_f1a3_source_split_and_population_freeze",
        )
    )


def source_class(path: str) -> str:
    """Classify a source path without inspecting outcome values."""
    lower = path.lower()
    if any(marker in lower for marker in ("fec_", "/fec", "g2_canary", "c3_t1d", "g_rec", "grec")):
        return CLASS_HARD
    if "official_v3_detector_v5_takeover" in lower:
        return CLASS_DETECTOR
    if any(
        marker in lower
        for marker in (
            "protocol",
            "taxonomy",
            "static",
            "audit",
            "source_options",
            "handoff",
        )
    ):
        return CLASS_STATIC
    return CLASS_UNRESOLVED


def add_evidence(
    evidence: dict[str, dict[tuple[str, str], dict[str, Any]]],
    key: str,
    path: str,
    authority: str,
    *,
    ref: str | None = None,
    commit: str | None = None,
    line: int | None = None,
    blob_oid: str | None = None,
) -> None:
    source = (authority, path)
    row = evidence[key].setdefault(
        source,
        {
            "authority": authority,
            "path": path,
            "source_class": source_class(path),
            "refs": [],
            "commits": [],
            "lines": [],
            "blob_oids": [],
        },
    )
    if ref and ref not in row["refs"]:
        row["refs"].append(ref)
    if commit and commit not in row["commits"]:
        row["commits"].append(commit)
    if line is not None and line not in row["lines"]:
        row["lines"].append(line)
    if blob_oid and blob_oid not in row["blob_oids"]:
        row["blob_oids"].append(blob_oid)


def scan_ref(ref: str, evidence: dict[str, dict[tuple[str, str], dict[str, Any]]]) -> None:
    try:
        output = subprocess.check_output(
            [
                "git",
                "-C",
                str(ROOT),
                "grep",
                "-I",
                "-n",
                "-E",
                GIT_GREP_PATTERN,
                ref,
                "--",
                "configs",
                "reports",
                "docs/handoffs",
                "paper",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        output = exc.output or ""
    seen: set[tuple[str, str]] = set()
    for line in output.splitlines():
        parts = line.split(":", 3)
        if len(parts) != 4:
            continue
        _, path, line_number, content = parts
        if is_audit_artifact(path):
            continue
        try:
            line_number_int = int(line_number)
        except ValueError:
            line_number_int = None
        for key in sorted(set(KEY_RE.findall(content)) & KEYS):
            # A receipt can contain thousands of repeated identity mentions.
            # One source/ref/identity row is sufficient for provenance and
            # avoids a subprocess per matching line.
            source_key = (path, key)
            if source_key in seen:
                continue
            seen.add(source_key)
            add_evidence(
                evidence,
                key,
                path,
                "GIT_REF_TIP",
                ref=ref,
                line=line_number_int,
            )


def scan_history(evidence: dict[str, dict[tuple[str, str], dict[str, Any]]]) -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "git",
                "-C",
                str(ROOT),
                "log",
                "--all",
                "--full-history",
                "--regexp-ignore-case",
                "-G",
                GIT_GREP_PATTERN,
                "--format=%H",
                "--name-only",
                "--no-renames",
                "--",
                "configs",
                "reports",
                "docs/handoffs",
                "paper",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        output = exc.output or ""

    commits: list[tuple[str, list[str]]] = []
    current_commit: str | None = None
    current_paths: list[str] = []
    for line in output.splitlines():
        if re.fullmatch(r"[0-9a-f]{40}", line):
            if current_commit:
                commits.append((current_commit, current_paths))
            current_commit = line
            current_paths = []
        elif line.strip():
            current_paths.append(line.strip())
    if current_commit:
        commits.append((current_commit, current_paths))

    source_pairs = 0
    hit_commits: set[str] = set()
    for commit, paths in commits:
        for path in sorted(set(paths)):
            if is_audit_artifact(path):
                continue
            try:
                diff = subprocess.check_output(
                    [
                        "git",
                        "-C",
                        str(ROOT),
                        "show",
                        "--format=",
                        "--unified=0",
                        "--no-ext-diff",
                        commit,
                        "--",
                        path,
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError:
                continue
            keys = set(KEY_RE.findall(diff)) & KEYS
            if not keys:
                continue
            source_pairs += 1
            hit_commits.add(commit)
            for key in sorted(keys):
                add_evidence(evidence, key, path, "GIT_HISTORY_DIFF", commit=commit)
    return {
        "commits_with_identity_diffs": len(hit_commits),
        "source_path_commit_pairs_with_identity_diffs": source_pairs,
        "commits": sorted(hit_commits),
    }


def classify(key: str, evidence: dict[str, dict[tuple[str, str], dict[str, Any]]]) -> str:
    classes = {row["source_class"] for row in evidence.get(key, {}).values()}
    if CLASS_HARD in classes:
        return CLASS_HARD
    if CLASS_UNRESOLVED in classes:
        return CLASS_UNRESOLVED
    if CLASS_DETECTOR in classes:
        return CLASS_DETECTOR
    if CLASS_STATIC in classes:
        return CLASS_STATIC
    return CLASS_PRISTINE


def contamination_tier(source_class_name: str) -> str:
    if source_class_name == CLASS_DETECTOR:
        return "DETECTOR_TRAIN_ONLY"
    return "PRISTINE"


def suite_of(key: str) -> str:
    return key.split("/", 1)[0]


def build_rows(evidence: dict[str, dict[tuple[str, str], dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(KEYS):
        suite, task, state = key.split("/")
        source_class_name = classify(key, evidence)
        source_rows = []
        for item in sorted(evidence.get(key, {}).values(), key=lambda row: (row["path"], row["authority"])):
            source_rows.append(
                {
                    **item,
                    "refs": sorted(item["refs"]),
                    "commits": sorted(item["commits"]),
                    "lines": sorted(item["lines"]),
                    "blob_oids": sorted(item["blob_oids"]),
                }
            )
        rows.append(
            {
                "canonical_parent_key": key,
                "suite": suite,
                "task": task,
                "state": int(state.removeprefix("state_")),
                "source_class": source_class_name,
                "contamination_tier": contamination_tier(source_class_name),
                "eligible_for_v3_split": source_class_name not in {CLASS_HARD, CLASS_UNRESOLVED},
                "historical_identity_source_hits": source_rows,
                "historical_outcome_values_read": False,
                "attack_physical_runtime_read": False,
                "protected_or_eval160_read": False,
                "permanent_exclusion_after_freeze": True,
            }
        )
    return rows


def rank_hash(key: str) -> str:
    return hashlib.sha256(f"{SPLIT_SALT}|{key}".encode("utf-8")).hexdigest()


def make_split(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    roles_by_suite: dict[str, dict[str, list[str]]] = {
        suite: {role: [] for role in ROLES} for suite in SUITES
    }
    ranked_by_suite: dict[str, list[dict[str, Any]]] = {}
    for suite in SUITES:
        candidates = [row for row in rows if row["suite"] == suite and row["eligible_for_v3_split"]]
        ranked = sorted(
            candidates,
            key=lambda row: (
                0 if row["contamination_tier"] == "PRISTINE" else 1,
                rank_hash(row["canonical_parent_key"]),
                row["canonical_parent_key"],
            ),
        )
        ranked_by_suite[suite] = [
            {
                "canonical_parent_key": row["canonical_parent_key"],
                "source_class": row["source_class"],
                "contamination_tier": row["contamination_tier"],
                "rank_hash": rank_hash(row["canonical_parent_key"]),
                "rank": index + 1,
            }
            for index, row in enumerate(ranked)
        ]
        roles_by_suite[suite]["BRIDGE_V3"] = [row["canonical_parent_key"] for row in ranked[:5]]
        roles_by_suite[suite]["DEV_V3"] = [row["canonical_parent_key"] for row in ranked[5:11]]
        roles_by_suite[suite]["C_CANARY_V3"] = [row["canonical_parent_key"] for row in ranked[11:13]]

    split_rows: list[dict[str, Any]] = []
    for suite in SUITES:
        for role in ROLES:
            for ordinal, key in enumerate(roles_by_suite[suite][role], start=1):
                row = next(item for item in rows if item["canonical_parent_key"] == key)
                split_rows.append(
                    {
                        "canonical_parent_key": key,
                        "suite": suite,
                        "role": role,
                        "role_ordinal_within_suite": ordinal,
                        "source_class": row["source_class"],
                        "contamination_tier": row["contamination_tier"],
                        "rank_hash": rank_hash(key),
                        "source_domain": "F1_SOURCE_V3",
                        "state_range": "0..19",
                        "identity_only_at_freeze": True,
                        "outcome_read": False,
                        "runtime_read": False,
                        "permanent_exclusion": True,
                    }
                )
    split = {
        "schema": "STAGE_X_X1R2_F1A3_SPLIT_LEDGER_V3",
        "gate": "STAGE_X_X1R2_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3",
        "status": "PENDING_VALIDATION",
        "source_domain": "F1_SOURCE_V3",
        "state_range": "0..19",
        "split_salt": SPLIT_SALT,
        "rank_rule": "contamination_tier(PRISTINE before DETECTOR_TRAIN_ONLY), then sha256(salt|canonical_parent_key), then canonical_parent_key",
        "counts_per_suite": {suite: {role: len(roles_by_suite[suite][role]) for role in ROLES} for suite in SUITES},
        "rows": split_rows,
        "ranked_eligible_by_suite": ranked_by_suite,
        "identity_only_at_freeze": True,
        "outcome_read": False,
        "runtime_read": False,
    }
    return split, roles_by_suite


def paper_v1_binding() -> dict[str, Any]:
    paths = run_git("ls-files", "paper/PAPER_V1_*").splitlines()
    tracked_diff = not any(
        path.startswith("paper/PAPER_V1_")
        for path in run_git("diff", "--name-only", "HEAD", "--", "paper").splitlines()
    )
    staged_diff = not any(
        path.startswith("paper/PAPER_V1_")
        for path in run_git("diff", "--cached", "--name-only", "HEAD", "--", "paper").splitlines()
    )
    # Git pathspec wildcards are not expanded consistently across the
    # Windows/native Git wrappers.  Read the paper tree once, then filter the
    # canonical tracked paths so an empty listing cannot masquerade as a seal.
    tree_listing = "\n".join(
        line for line in run_git("ls-tree", "-r", "HEAD", "--", "paper").splitlines()
        if line.split("\t", 1)[-1].startswith("paper/PAPER_V1_")
    )
    return {
        "tracked_file_count": len(paths),
        "tracked_files": paths,
        "working_tree_unchanged": tracked_diff and staged_diff,
        "paper_v1_tree_listing_sha256": hashlib.sha256(tree_listing.encode("utf-8")).hexdigest(),
    }


def main() -> int:
    evidence: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    refs = run_git("for-each-ref", "--format=%(refname)", "refs/heads", "refs/remotes").splitlines()
    for ref in refs:
        scan_ref(ref, evidence)
    history = scan_history(evidence)
    rows = build_rows(evidence)
    split, roles_by_suite = make_split(rows)

    class_counts = {name: sum(row["source_class"] == name for row in rows) for name in CLASSES}
    eligible_by_suite = {
        suite: sum(row["eligible_for_v3_split"] and row["suite"] == suite for row in rows)
        for suite in SUITES
    }
    selected_rows = split["rows"]
    role_counts = {
        suite: {role: len(roles_by_suite[suite][role]) for role in ROLES} for suite in SUITES
    }
    selected_key_sets = {role: {row["canonical_parent_key"] for row in selected_rows if row["role"] == role} for role in ROLES}
    intersections = {
        f"{left}_x_{right}": len(selected_key_sets[left] & selected_key_sets[right])
        for index, left in enumerate(ROLES)
        for right in ROLES[index + 1 :]
    }
    selected_bad = [
        row["canonical_parent_key"]
        for row in selected_rows
        if row["source_class"] in {CLASS_HARD, CLASS_UNRESOLVED}
    ]
    paper = paper_v1_binding()
    source_commit = run_git("rev-parse", "HEAD")
    source_tree = run_git("rev-parse", "HEAD^{tree}")
    protected_boundary = {
        "gpu": 0,
        "model_inference": 0,
        "simulator": 0,
        "env_step": 0,
        "pgd": 0,
        "vphys": 0,
        "physical_outcome": 0,
        "bridge_runtime": 0,
        "bridge_outcome_read": 0,
        "eval160": "UNREAD",
        "protected": "UNREAD",
    }
    errors: list[str] = []
    if len(rows) != 800:
        errors.append(f"V3_KEY_COUNT:{len(rows)}")
    if any(value < 13 for value in eligible_by_suite.values()):
        errors.append(f"ELIGIBLE_BELOW_13:{eligible_by_suite}")
    if any(role_counts[suite][role] != {"BRIDGE_V3": 5, "DEV_V3": 6, "C_CANARY_V3": 2}[role] for suite in SUITES for role in ROLES):
        errors.append(f"ROLE_COUNT_MISMATCH:{role_counts}")
    if any(intersections.values()):
        errors.append(f"ROLE_INTERSECTION:{intersections}")
    if selected_bad:
        errors.append(f"SELECTED_HARD_OR_UNRESOLVED:{selected_bad}")
    if not paper["working_tree_unchanged"]:
        errors.append("PAPER_V1_WORKTREE_CHANGED")

    status = "PASS_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3" if not errors else "HOLD_F1A3_SOURCE_CONTRACT_NOT_ESTABLISHED"
    split["status"] = status
    split["validation_errors"] = errors

    source_report = {
        "schema": "STAGE_X_X1R2_F1A3_SOURCE_LEAKAGE_CONTRACT_V3",
        "gate": "STAGE_X_X1R2_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3",
        "status": status,
        "pi_authority": {"comment_id": PI_COMMENT_ID, "attachment_sha256": PI_ATTACHMENT_SHA256},
        "source_commit": source_commit,
        "source_tree": source_tree,
        "source_domain": "F1_SOURCE_V3",
        "state_range": "0..19",
        "identity_count": 800,
        "classification_classes": list(CLASSES),
        "classification_precedence": [CLASS_HARD, CLASS_UNRESOLVED, CLASS_DETECTOR, CLASS_STATIC, CLASS_PRISTINE],
        "hard_exclusion_policy": [
            "prior_visual_attack_or_pgd_or_attack_result",
            "prior_command_open_or_physical_or_vphys_or_physical_outcome",
            "prior_q3_q3ar_q3r2_q3r3_e2_e3_f1_engineering_execution",
            "prior_fec_g2_c3_canary_or_runtime_qualification",
            "prior_manual_attack_or_physical_adjudication",
            "protected_or_eval160_membership",
            "unresolved_identity_level_exposure",
        ],
        "detector_only_policy": "V3_DETECTOR_TRAIN_ONLY_ALLOWED_WITH_FLAG only when no hard or unresolved source is present",
        "static_policy": "static protocol/manifest/audit mention is recorded but is not execution",
        "history_authority": {"refs_scanned": refs, **history},
        "audit_artifacts_excluded_from_exposure_authority": [
            "STAGE_X_X1R2_F1A2_SOURCE_LEAKAGE_AND_POPULATION_FREEZE_*",
            "STAGE_X_X1R2_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_*",
        ],
        "class_counts": class_counts,
        "eligible_by_suite": eligible_by_suite,
        "rows": rows,
        "paper_v1_binding": paper,
        "protected_boundary": protected_boundary,
        "outcome_values_inspected": False,
    }
    write_json(OUT / "F1A3_SOURCE_LEAKAGE_CONTRACT_V3.json", source_report)
    write_json(OUT / "F1A3_EXPOSURE_CLASSIFICATION_V3.json", {"schema": "STAGE_X_X1R2_F1A3_EXPOSURE_CLASSIFICATION_V3", "status": status, "rows": rows, "class_counts": class_counts, "history_authority": {"refs_scanned": refs, **history}, "source_commit": source_commit, "source_tree": source_tree})
    write_json(OUT / "F1A3_SPLIT_LEDGER_V3.json", split)
    for role in ROLES:
        rows_for_role = [row for row in selected_rows if row["role"] == role]
        write_json(
            OUT / f"F1A3_{role}_LEDGER_V3.json",
            {
                "schema": f"STAGE_X_X1R2_F1A3_{role}_LEDGER_V3",
                "status": status,
                "role": role,
                "source_domain": "F1_SOURCE_V3",
                "state_range": "0..19",
                "split_salt": SPLIT_SALT,
                "rows": rows_for_role,
                "row_count": len(rows_for_role),
                "identity_only_at_freeze": True,
                "outcome_read": False,
                "runtime_read": False,
                "permanent_exclusion": True,
            },
        )

    artifact_paths = [
        "scripts/stage_x/audit_stage_x1r2_f1a3_source_split_v3.py",
        *(f"reports/{OUT.relative_to(ROOT).as_posix().split('/', 1)[1]}/{name}" for name in (
            "F1A3_SOURCE_LEAKAGE_CONTRACT_V3.json",
            "F1A3_EXPOSURE_CLASSIFICATION_V3.json",
            "F1A3_SPLIT_LEDGER_V3.json",
            "F1A3_BRIDGE_V3_LEDGER_V3.json",
            "F1A3_DEV_V3_LEDGER_V3.json",
            "F1A3_C_CANARY_V3_LEDGER_V3.json",
        )),
    ]
    root_seal = {
        "schema": "STAGE_X_X1R2_F1A3_ROOT_SEAL_V3",
        "status": status,
        "gate": "STAGE_X_X1R2_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3",
        "pi_comment_id": PI_COMMENT_ID,
        "pi_attachment_sha256": PI_ATTACHMENT_SHA256,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "split_salt": SPLIT_SALT,
        "artifact_hashes": {path: sha256(ROOT / path) for path in artifact_paths},
        "class_counts": class_counts,
        "eligible_by_suite": eligible_by_suite,
        "role_counts": role_counts,
        "role_intersections": intersections,
        "selected_hard_or_unresolved_count": len(selected_bad),
        "paper_v1_binding": paper,
        "protected_boundary": protected_boundary,
        "seal_scope_excludes_sidecar": True,
    }
    root_path = OUT / "F1A3_ROOT_SEAL_V3.json"
    write_json(root_path, root_seal)
    (OUT / "F1A3_ROOT_SEAL_V3.sha256").write_text(
        f"{sha256(root_path)}  {root_path.name}\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"status": status, "class_counts": class_counts, "eligible_by_suite": eligible_by_suite, "role_counts": role_counts, "errors": errors, "output": str(OUT.relative_to(ROOT)).replace("\\", "/")}, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
