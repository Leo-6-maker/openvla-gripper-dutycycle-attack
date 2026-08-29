#!/usr/bin/env python3
"""Seal the AC4 three-agent AI adjudication and reconcile it after unblind."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_REL = "reports/STAGE_AC_AC4_NEUTRAL_BLIND_MANIFEST_V1.json"
PACKAGE_SEAL_REL = "reports/STAGE_AC_AC4_NEUTRAL_BLIND_PACKAGE_SEAL_V1.json"
LABELS_REL = "reports/STAGE_AC_AC4_THREE_AGENT_BLINDED_AI_ADJUDICATION_LABELS_V1.json"
LABEL_SEAL_REL = "reports/STAGE_AC_AC4_THREE_AGENT_BLINDED_AI_ADJUDICATION_LABEL_SEAL_V1.json"
INPUT_REL = "reports/STAGE_AC_AC4_THREE_AGENT_BLINDED_AI_ADJUDICATION_INPUT_V1.txt"
BRANCH_INDEX_REL = "reports/STAGE_AC_AC3_G2_BRANCH_RECEIPT_INDEX_V1.json"
G3_REL = "reports/STAGE_AC_AC3_G2R1_G3R1_CENSORING_AWARE_STATISTICS_V2/STAGE_AC_AC3_G2R1_G3R1_CENSORING_AWARE_STATISTICS_V2.json"
SCRIPT_REL = "scripts/stage_ac/reconcile_three_agent_blinded_adjudication.py"
EXPECTED_MANIFEST_SHA = "1b1a0aa3d24bf6aa2e21e83eda2a35b3a4751bd27963c1e4ab8ba5fe712e2c8f"
EXPECTED_PACKAGE_SHA = "0bab4b174942baced11e75d83b913154deedc1c68883a87996df57d6f8f65dfe"
EXPECTED_LABEL_SHA = "47fd5a60bb70a023f2b30122a80c6149511389abc5f8ceb3b2e46ce9bc0f628a"
AGENTS = ("AGENT_A", "AGENT_B", "AGENT_C")
LABELS = (
    "STABLE_GRASP",
    "PREMATURE_APERTURE",
    "CONTACT_LOSS",
    "PREMATURE_RELEASE_OR_DROP",
    "OBJECT_DISPLACEMENT",
    "AMBIGUOUS_OR_OCCLUDED",
    "NOT_IDENTIFIABLE",
)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path, root: Path = ROOT) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(root.resolve())) if path.resolve().is_relative_to(root.resolve()) else str(path),
        "bytes": path.stat().st_size,
        "sha256": sha(path),
    }


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def counts(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def crosstab(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, dict[str, int]]:
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        table[str(row.get(left))][str(row.get(right))] += 1
    return {key: dict(sorted(value.items())) for key, value in sorted(table.items())}


def grouped_counts(rows: list[dict[str, Any]], keys: tuple[str, ...], label_key: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for row in rows:
        group = tuple("<NULL>" if row.get(key) is None else str(row.get(key)) for key in keys)
        groups[group][str(row.get(label_key))] += 1
    return [
        {
            "group": dict(zip(keys, group)),
            "row_count": sum(counter.values()),
            "label_counts": dict(sorted(counter.items())),
        }
        for group, counter in sorted(groups.items())
    ]


def exact_fleiss_kappa(label_rows: list[dict[str, str]]) -> dict[str, Any]:
    n_subjects = len(label_rows)
    n_raters = len(AGENTS)
    category_counts = Counter(row[agent] for row in label_rows for agent in AGENTS)
    p = {label: category_counts[label] / (n_subjects * n_raters) for label in LABELS}
    subject_agreement: list[float] = []
    for row in label_rows:
        per_subject = Counter(row[agent] for agent in AGENTS)
        subject_agreement.append(
            sum(per_subject[label] * (per_subject[label] - 1) for label in LABELS)
            / (n_raters * (n_raters - 1))
        )
    p_bar = sum(subject_agreement) / n_subjects if n_subjects else None
    p_e = sum(value * value for value in p.values())
    kappa = (p_bar - p_e) / (1 - p_e) if p_bar is not None and p_e != 1 else None
    return {
        "subjects": n_subjects,
        "raters": n_raters,
        "categories": list(LABELS),
        "category_marginals": p,
        "mean_observed_agreement": p_bar,
        "expected_agreement": p_e,
        "fleiss_kappa": kappa,
    }


def agreement(label_rows: list[dict[str, str]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    pair_names = (("AGENT_A", "AGENT_B"), ("AGENT_A", "AGENT_C"), ("AGENT_B", "AGENT_C"))
    pairwise: dict[str, Any] = {}
    for left, right in pair_names:
        matches = sum(row[left] == row[right] for row in label_rows)
        pairwise[f"{left}_vs_{right}"] = {
            "matches": matches,
            "subjects": len(label_rows),
            "agreement_rate": rate(matches, len(label_rows)),
        }
    classes = Counter()
    consensus_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for row in label_rows:
        values = [row[agent] for agent in AGENTS]
        value_counts = Counter(values)
        if len(value_counts) == 1:
            agreement_class = "UNANIMOUS_3_OF_3"
            consensus = values[0]
        elif len(value_counts) == 2:
            agreement_class = "MAJORITY_2_OF_3"
            consensus = value_counts.most_common(1)[0][0]
        else:
            agreement_class = "DISAGREEMENT_1_1_1"
            consensus = None
        classes[agreement_class] += 1
        if consensus is not None:
            consensus_counts[consensus] += 1
        rows.append({
            "blinded_video_id": row["blinded_video_id"],
            "agent_labels": {agent: row[agent] for agent in AGENTS},
            "agreement_class": agreement_class,
            "consensus_label": consensus,
        })
    summary = {
        "subjects": len(label_rows),
        "unanimous_3_of_3": classes["UNANIMOUS_3_OF_3"],
        "majority_2_of_3": classes["MAJORITY_2_OF_3"],
        "disagreement_1_1_1": classes["DISAGREEMENT_1_1_1"],
        "unanimous_rate": rate(classes["UNANIMOUS_3_OF_3"], len(label_rows)),
        "majority_rate": rate(classes["MAJORITY_2_OF_3"], len(label_rows)),
        "disagreement_rate": rate(classes["DISAGREEMENT_1_1_1"], len(label_rows)),
        "pairwise_agreement": pairwise,
        "mean_pairwise_agreement_rate": rate(
            sum(item["matches"] for item in pairwise.values()),
            len(pairwise) * len(label_rows),
        ),
        "consensus_or_majority_distribution": dict(sorted(consensus_counts.items())),
    }
    return summary, {"rows": rows}, dict(classes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=Path(r"D:\vla_attack\.ac4_work\ac4_package"))
    parser.add_argument("--private-map", type=Path, default=Path(r"D:\vla_attack\.ac4_work\private\STAGE_AC_AC4_NEUTRAL_BLIND_MAPPING_PRIVATE_V1.json"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/STAGE_AC_AC4_THREE_AGENT_AI_ADJUDICATION_RECONCILIATION_V1")
    args = parser.parse_args()
    failures: list[str] = []

    manifest_path = ROOT / MANIFEST_REL
    package_seal_path = ROOT / PACKAGE_SEAL_REL
    labels_path = ROOT / LABELS_REL
    label_seal_path = ROOT / LABEL_SEAL_REL
    input_path = ROOT / INPUT_REL
    branch_index_path = ROOT / BRANCH_INDEX_REL
    g3_path = ROOT / G3_REL

    # Fail closed before opening the private mapping: labels and their seal are the first authority boundary.
    labels = load(labels_path)
    label_seal = load(label_seal_path)
    actual_label_sha = sha(labels_path)
    if actual_label_sha != EXPECTED_LABEL_SHA or label_seal.get("label_sha256") != actual_label_sha:
        failures.append("LABEL_EXACT_BYTE_OR_SEAL_MISMATCH")
    if label_seal.get("label_bytes") != labels_path.stat().st_size:
        failures.append("LABEL_BYTE_COUNT_MISMATCH")
    if labels.get("status") != "STAGE_AC_AC4_THREE_AGENT_LABELS_SEALED_BEFORE_UNBLIND":
        failures.append("LABEL_STATUS")
    if label_seal.get("status") != "STAGE_AC_AC4_THREE_AGENT_LABELS_SEALED":
        failures.append("LABEL_SEAL_STATUS")
    if label_seal.get("labels_sealed_before_unblind") is not True:
        failures.append("LABEL_ORDER")
    if label_seal.get("hidden_mapping_read_before_label_seal") is not False:
        failures.append("LABEL_MAPPING_ORDER")
    if label_seal.get("reviewer_sessions_mapping_exposure") is not False:
        failures.append("REVIEWER_MAPPING_FIREWALL")
    if label_seal.get("human_review_gate_satisfied") is not False:
        failures.append("HUMAN_GATE_FLAG")
    if failures:
        raise SystemExit(json.dumps({"status": "HOLD_BEFORE_UNBLIND", "failures": failures}, indent=2))

    manifest = load(manifest_path)
    manifest_sha = sha(manifest_path)
    if manifest_sha != EXPECTED_MANIFEST_SHA:
        failures.append("MANIFEST_SHA")
    public_rows = manifest.get("rows", [])
    public_by_id = {row.get("blinded_video_id"): row for row in public_rows}
    present = {key: row for key, row in public_by_id.items() if row.get("availability") == "PRESENT"}
    missing = {key: row for key, row in public_by_id.items() if row.get("availability") != "PRESENT"}
    if len(public_rows) != 96 or len(public_by_id) != 96:
        failures.append("MANIFEST_ROWS")
    if len(present) != 91 or len(missing) != 5:
        failures.append("MANIFEST_PRESENT_MISSING")

    package_seal = load(package_seal_path)
    package_zip = args.package_root / str(package_seal["package"]["path"])
    if package_seal.get("package", {}).get("sha256") != EXPECTED_PACKAGE_SHA:
        failures.append("PACKAGE_SEAL_DECLARED_SHA")
    if package_zip.is_file() and sha(package_zip) != package_seal["package"].get("sha256"):
        failures.append("PACKAGE_ZIP_SHA")
    for video_id, row in present.items():
        video_path = args.package_root / str(row["package_filename"]).replace("/", "\\")
        if not video_path.is_file():
            failures.append(f"VIDEO_MISSING:{video_id}")
        elif video_path.stat().st_size != row.get("bytes") or sha(video_path) != row.get("sha256"):
            failures.append(f"VIDEO_HASH:{video_id}")

    label_by_agent: dict[str, dict[str, dict[str, str]]] = {}
    for agent in AGENTS:
        rows = labels.get("agents", {}).get(agent, [])
        label_by_agent[agent] = {row.get("blinded_video_id"): row for row in rows}
        if len(rows) != 91 or len(label_by_agent[agent]) != 91 or set(label_by_agent[agent]) != set(present):
            failures.append(f"AGENT_ID_SET:{agent}")
        for row in rows:
            if row.get("primary_label") not in LABELS:
                failures.append(f"AGENT_LABEL_VOCABULARY:{agent}:{row.get('blinded_video_id')}")
    if failures:
        raise SystemExit(json.dumps({"status": "HOLD_PRE_UNBLIND_STATIC_VALIDATION", "failures": failures}, indent=2))

    # Only after exact label validation and the pre-unblind seal do we open hidden mapping/outcome-bearing sources.
    private = load(args.private_map)
    branch_index = load(branch_index_path)
    g3 = load(g3_path)
    private_rows = private.get("rows", [])
    private_by_id = {row.get("blinded_video_id"): row for row in private_rows}
    branch_by_id = {row.get("branch_id"): row for row in branch_index.get("rows", [])}
    if private.get("status") != "SEALED_PRIVATE_NOT_FOR_REVIEWER":
        failures.append("PRIVATE_MAP_STATUS")
    if set(private_by_id) != set(public_by_id):
        failures.append("PRIVATE_PUBLIC_ID_SET")

    label_rows: list[dict[str, str]] = []
    panel_rows: list[dict[str, Any]] = []
    for video_id in sorted(present):
        labels_for_id = {agent: label_by_agent[agent][video_id]["primary_label"] for agent in AGENTS}
        label_rows.append({"blinded_video_id": video_id, **labels_for_id})
        private_row = private_by_id.get(video_id)
        source = private_row.get("frozen_sample_row", {}) if private_row else {}
        branch_id = source.get("branch_id")
        branch = branch_by_id.get(branch_id, {})
        validation = branch.get("validation", {})
        if not private_row or not branch:
            failures.append(f"UNBLIND_JOIN:{video_id}")
        panel_rows.append({
            "blinded_video_id": video_id,
            "agent_labels": {
                agent: {
                    "primary_label": label_by_agent[agent][video_id]["primary_label"],
                    "confidence": label_by_agent[agent][video_id].get("confidence"),
                }
                for agent in AGENTS
            },
            "model_family": source.get("model_family") or branch.get("model_family"),
            "suite": source.get("suite") or branch.get("suite"),
            "canonical_parent_key": source.get("canonical_parent_key") or branch.get("canonical_parent_key"),
            "condition": source.get("condition") or branch.get("condition"),
            "dose": source.get("dose") if source.get("dose") is not None else branch.get("dose"),
            "arm": source.get("arm") or branch.get("arm"),
            "branch_id": branch_id,
            "cell_id": source.get("cell_id") or branch.get("cell_id"),
            "automatic_status": branch.get("status"),
            "automatic_physical_class": validation.get("physical_class"),
            "automatic_v_phys_label": validation.get("v_phys_label"),
        })
    if failures:
        raise SystemExit(json.dumps({"status": "HOLD_UNBLIND_JOIN", "failures": failures}, indent=2))

    agreement_summary, agreement_rows, agreement_classes = agreement(label_rows)
    fleiss = exact_fleiss_kappa(label_rows)
    joined_by_id = {row["blinded_video_id"]: row for row in panel_rows}
    for row in agreement_rows["rows"]:
        joined_by_id[row["blinded_video_id"]].update({
            "agreement_class": row["agreement_class"],
            "consensus_label": row["consensus_label"],
        })
    joined = [joined_by_id[key] for key in sorted(joined_by_id)]
    for row in joined:
        for agent in AGENTS:
            row[f"{agent.lower()}_label"] = row["agent_labels"][agent]["primary_label"]
        row["consensus_or_majority_label"] = row.get("consensus_label")

    agent_distributions = {
        agent: counts([row["agent_labels"][agent]["primary_label"] for row in joined])
        for agent in AGENTS
    }
    consensus_rows = [row for row in joined if row.get("consensus_label") is not None]
    auto_vs = {
        "automatic_physical_class_to_agent": {
            agent: crosstab(
                [{"auto": row.get("automatic_physical_class"), "label": row["agent_labels"][agent]["primary_label"]} for row in joined],
                "auto",
                "label",
            )
            for agent in AGENTS
        },
        "automatic_physical_class_to_consensus_or_disagreement": crosstab(
            [{"auto": row.get("automatic_physical_class"), "label": row.get("consensus_label") or row.get("agreement_class")} for row in joined],
            "auto",
            "label",
        ),
        "automatic_v_phys_label_to_agent": {
            agent: crosstab(
                [{"auto": row.get("automatic_v_phys_label"), "label": row["agent_labels"][agent]["primary_label"]} for row in joined],
                "auto",
                "label",
            )
            for agent in AGENTS
        },
        "automatic_v_phys_label_to_consensus_or_disagreement": crosstab(
            [{"auto": row.get("automatic_v_phys_label"), "label": row.get("consensus_label") or row.get("agreement_class")} for row in joined],
            "auto",
            "label",
        ),
    }
    auto_contact = [row for row in joined if row.get("automatic_physical_class") == "GRIPPER_CONTACT_LOSS"]
    auto_vs["gripper_contact_loss"] = {
        "automatic_rows": len(auto_contact),
        "by_agent": {agent: counts([row["agent_labels"][agent]["primary_label"] for row in auto_contact]) for agent in AGENTS},
        "consensus_or_disagreement": counts([row.get("consensus_label") or row.get("agreement_class") for row in auto_contact]),
        "not_identifiable_or_ambiguous_by_agent": {
            agent: sum(row["agent_labels"][agent]["primary_label"] in {"NOT_IDENTIFIABLE", "AMBIGUOUS_OR_OCCLUDED"} for row in auto_contact)
            for agent in AGENTS
        },
        "stable_grasp_by_agent": {
            agent: sum(row["agent_labels"][agent]["primary_label"] == "STABLE_GRASP" for row in auto_contact)
            for agent in AGENTS
        },
        "interpretation": "Diagnostic only; automatic labels and V_phys/abstention denominators are never rewritten.",
    }

    authority = {
        "public_manifest": artifact(manifest_path),
        "package_seal": artifact(package_seal_path),
        "package_zip": artifact(package_zip, root=args.package_root) if package_zip.is_file() else {"path": str(package_zip), "missing": True},
        "input_label_table": artifact(input_path),
        "labels": artifact(labels_path),
        "label_seal": artifact(label_seal_path),
        "private_mapping": {"path": str(args.private_map), "bytes": args.private_map.stat().st_size, "sha256": sha(args.private_map), "not_committed": True},
        "branch_index": artifact(branch_index_path),
        "g3r1_statistics": artifact(g3_path),
        "script": artifact(ROOT / SCRIPT_REL),
    }
    report = {
        "schema": "STAGE_AC_AC4_THREE_AGENT_AI_ADJUDICATION_RECONCILIATION_V1",
        "status": "STAGE_AC_AC4_THREE_AGENT_AI_ADJUDICATION_COMPLETE_STOP_FOR_PI",
        "reviewer_governance": {
            "reviewer_type": "THREE_AGENT_BLINDED_AI_ADJUDICATION",
            "agent_count": 3,
            "agent_row_counts": {agent: len(label_by_agent[agent]) for agent in AGENTS},
            "labels_sealed_before_unblind": True,
            "agent_sessions_mapping_exposure": False,
            "agent_sessions_automatic_label_exposure": False,
            "human_review_gate_satisfied": False,
            "formal_human_review_claim": False,
            "orchestrator_prior_mapping_exposure_before_this_panel": True,
        },
        "counts": {
            "frozen_slots": len(public_rows),
            "present_videos": len(present),
            "missing_frozen_videos": len(missing),
            "label_rows_per_agent": len(present),
            "unblinded_joined_rows": len(joined),
        },
        "agreement": {
            **agreement_summary,
            "fleiss_kappa_nominal": fleiss,
            "agent_marginal_distributions": agent_distributions,
            "agreement_class_counts": agreement_classes,
            "disagreement_rows": [row for row in joined if row["agreement_class"] == "DISAGREEMENT_1_1_1"],
        },
        "automatic_endpoint_reconciliation": auto_vs,
        "unblinded_label_distribution": {
            "overall_by_agent": agent_distributions,
            "consensus_or_majority_only": counts([row["consensus_label"] for row in consensus_rows]),
            "by_model_agent": {
                agent: grouped_counts(joined, ("model_family",), f"{agent.lower()}_label")
                for agent in AGENTS
            },
            "by_suite_agent": {
                agent: grouped_counts(joined, ("suite",), f"{agent.lower()}_label")
                for agent in AGENTS
            },
            "by_model_condition_dose_consensus": grouped_counts(joined, ("model_family", "condition", "dose"), "consensus_or_majority_label"),
        },
        "same_parent_g3r1_authority_preserved": {
            "source": artifact(g3_path),
            "model_summary": g3.get("model_summary"),
            "automatic_statistics_recomputed": False,
            "panel_labels_do_not_rewrite_statistics": True,
        },
        "missing_slots": sorted(missing),
        "authority": authority,
        "joined_rows": joined,
        "scientific_firewall": {
            "new_model_inference": 0,
            "new_env_step": 0,
            "new_open_intervention": 0,
            "new_pgd": 0,
            "new_protected_reads": 0,
            "new_eval160_reads": 0,
            "new_identities": 0,
            "replacement_or_top_up": 0,
            "automatic_labels_rewritten": 0,
            "denominator_changed": 0,
        },
        "claim_boundary": "Three-agent blinded AI adjudication is supplemental endpoint-validity evidence only; it is not human review, does not relabel automatic endpoints or V_phys, and creates no scientific promotion.",
        "next_legal_action": "STOP_FOR_PI_NO_NEW_EXECUTION_NO_PAPER_PROMOTION_NO_BRIDGE",
        "validation_failures": [],
    }

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"OUTPUT_NOT_EMPTY:{output}")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "STAGE_AC_AC4_THREE_AGENT_AI_ADJUDICATION_RECONCILIATION_V1.json"
    report_path.write_bytes((json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    report_ref = artifact(report_path)
    root_payload = {
        "schema": report["schema"],
        "status": report["status"],
        "report": report_ref,
        "counts": report["counts"],
        "agreement": {
            "unanimous_3_of_3": agreement_summary["unanimous_3_of_3"],
            "majority_2_of_3": agreement_summary["majority_2_of_3"],
            "disagreement_1_1_1": agreement_summary["disagreement_1_1_1"],
            "fleiss_kappa": fleiss["fleiss_kappa"],
        },
        "scientific_firewall": report["scientific_firewall"],
    }
    root = {
        "schema": "STAGE_AC_AC4_THREE_AGENT_AI_ADJUDICATION_ROOT_SEAL_V1",
        "status": report["status"],
        "root_payload": root_payload,
        "root_payload_sha256": hashlib.sha256(json.dumps(root_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "artifact_manifest": {
            "entries": [
                authority["public_manifest"],
                authority["package_seal"],
                authority["package_zip"],
                authority["input_label_table"],
                authority["labels"],
                authority["label_seal"],
                authority["branch_index"],
                authority["g3r1_statistics"],
                authority["script"],
                report_ref,
            ],
            "root_seal_excludes_self": True,
        },
        "reviewer_governance": report["reviewer_governance"],
        "scientific_firewall": report["scientific_firewall"],
        "claim_boundary": report["claim_boundary"],
        "next_legal_action": report["next_legal_action"],
        "validation_failures": [],
    }
    root_path = output / "STAGE_AC_AC4_THREE_AGENT_AI_ADJUDICATION_ROOT_SEAL_V1.json"
    root_path.write_bytes((json.dumps(root, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    sidecar = output / "STAGE_AC_AC4_THREE_AGENT_AI_ADJUDICATION_ROOT_SEAL_V1.sha256"
    sidecar.write_bytes(f"{sha(root_path)}  {root_path.name}\n".encode("utf-8"))
    print(json.dumps({
        "status": report["status"],
        "present_videos": len(present),
        "unanimous_3_of_3": agreement_summary["unanimous_3_of_3"],
        "majority_2_of_3": agreement_summary["majority_2_of_3"],
        "disagreement_1_1_1": agreement_summary["disagreement_1_1_1"],
        "fleiss_kappa": fleiss["fleiss_kappa"],
        "report_sha256": sha(report_path),
        "root_sha256": sha(root_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
