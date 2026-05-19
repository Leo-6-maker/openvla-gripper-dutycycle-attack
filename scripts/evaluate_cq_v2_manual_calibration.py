#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def str_bool(value) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_mapping_with_trailing_notes(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return []
    header = rows[0]
    out = []
    for values in rows[1:]:
        fixed = list(values[: len(header)])
        if len(fixed) < len(header):
            fixed.extend([""] * (len(header) - len(fixed)))
        if len(values) > len(header):
            note_idx = header.index("notes") if "notes" in header else len(header) - 1
            extra = " ".join(v.strip() for v in values[len(header) :] if v.strip())
            if extra:
                fixed[note_idx] = " ".join(v for v in [fixed[note_idx].strip(), extra] if v)
        out.append(dict(zip(header, fixed)))
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> dict:
    tp = sum(r["manual_positive"] and r["cq_v2_positive"] for r in rows)
    tn = sum((not r["manual_positive"]) and (not r["cq_v2_positive"]) for r in rows)
    fp = sum((not r["manual_positive"]) and r["cq_v2_positive"] for r in rows)
    fn = sum(r["manual_positive"] and (not r["cq_v2_positive"]) for r in rows)
    total = len(rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "total": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "manual_positive": sum(r["manual_positive"] for r in rows),
        "manual_negative": sum(not r["manual_positive"] for r in rows),
        "cq_v2_positive": sum(r["cq_v2_positive"] for r in rows),
        "cq_v2_negative": sum(not r["cq_v2_positive"] for r in rows),
    }


def build_reports(mapping_path: Path, review_path: Path, cq_v2_path: Path) -> tuple[list[dict], list[dict], dict]:
    mapping = {row["review_id"]: row for row in read_mapping_with_trailing_notes(mapping_path)}
    reviews = {row["review_id"]: row for row in read_rows(review_path)}
    cq_v2 = {row["run_id"]: row for row in read_rows(cq_v2_path)}

    merged = []
    for review_id in sorted(set(mapping) & set(reviews)):
        m = mapping[review_id]
        r = reviews[review_id]
        c = cq_v2.get(m.get("run_id", ""), {})
        manual_positive = str_bool(r.get("manual_contact_quality_failure")) is True
        cq_v2_positive = str_bool(c.get("contact_quality_failure")) is True
        v1_positive = str_bool(m.get("cq_contact_quality_failure")) is True
        merged.append(
            {
                "review_id": review_id,
                "run_id": m.get("run_id", ""),
                "state": m.get("state", ""),
                "seed": m.get("seed", ""),
                "condition": m.get("condition", ""),
                "official_success": m.get("official_success", ""),
                "manual_contact_quality_failure": int(manual_positive),
                "manual_task_success_quality": r.get("manual_task_success_quality", ""),
                "manual_failure_phase": r.get("manual_failure_phase", ""),
                "manual_failure_cause": r.get("manual_failure_cause", ""),
                "uncontrolled_final_drop_manual": r.get("uncontrolled_final_drop", ""),
                "final_placement_quality_manual": r.get("final_placement_quality", ""),
                "auto_cq_v1_contact_quality_failure": int(v1_positive),
                "cq_v2_contact_quality_failure": int(cq_v2_positive),
                "cq_v2_contact_quality_success": c.get("contact_quality_success", ""),
                "cq_v2_sr_cq_mismatch": c.get("sr_cq_mismatch", ""),
                "cq_v2_uncontrolled_final_drop": c.get("uncontrolled_final_drop", ""),
                "cq_v2_stable_controlled_place": c.get("stable_controlled_place", ""),
                "cq_v2_failure_reason": c.get("cq_failure_reason_v2", c.get("failure_reason", "")),
                "cq_v2_confidence": c.get("cq_confidence_v2", c.get("confidence", "")),
                "manual_positive": manual_positive,
                "cq_v2_positive": cq_v2_positive,
                "agreement": manual_positive == cq_v2_positive,
                "error_type": "" if manual_positive == cq_v2_positive else ("false_negative" if manual_positive else "false_positive"),
                "notes_chinese": r.get("notes_chinese", ""),
                "mapping_notes": m.get("notes", ""),
            }
        )

    group_rows = []
    groups = defaultdict(list)
    for row in merged:
        groups[(row["state"], row["condition"])].append(row)
    for (state, condition), rows in sorted(groups.items()):
        s = summarize(rows)
        group_rows.append(
            {
                "state": state,
                "condition": condition,
                "n": s["total"],
                "manual_cq_failure": s["manual_positive"],
                "cq_v2_cq_failure": s["cq_v2_positive"],
                "agreement": s["tp"] + s["tn"],
                "false_positive": s["fp"],
                "false_negative": s["fn"],
                "precision": f"{s['precision']:.6f}",
                "recall": f"{s['recall']:.6f}",
                "f1": f"{s['f1']:.6f}",
            }
        )
    return merged, group_rows, summarize(merged)


def write_summaries(output_dir: Path, metrics: dict, group_rows: list[dict]) -> None:
    passed = metrics["total"] == 20 and metrics["tp"] == 9 and metrics["tn"] == 11 and metrics["fp"] == 0 and metrics["fn"] == 0
    calibration = "Black Bowl calibration pass" if passed else "cq_v2_improved_but_not_fully_calibrated"
    lines = [
        "# Black Bowl Generic v4 Manual Audit Summary 20260517",
        "",
        "- Original auto-CQ conclusion: effect_partial",
        "- Manual-audited conclusion: replication_passed",
        f"- CQ v2 calibrated result: {calibration}",
        "",
        "## CQ v2 vs Manual",
        f"- total: {metrics['total']}",
        f"- true_positive: {metrics['tp']}",
        f"- true_negative: {metrics['tn']}",
        f"- false_positive: {metrics['fp']}",
        f"- false_negative: {metrics['fn']}",
        f"- precision: {metrics['precision']:.3f}",
        f"- recall: {metrics['recall']:.3f}",
        f"- f1: {metrics['f1']:.3f}",
        "",
        "## Group Summary",
    ]
    for row in group_rows:
        lines.append(
            f"- state {row['state']} {row['condition']}: n={row['n']}, "
            f"manual={row['manual_cq_failure']}, cq_v2={row['cq_v2_cq_failure']}, "
            f"fp={row['false_positive']}, fn={row['false_negative']}"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "CQ v2 is calibrated on the Black Bowl manual review set only. This is not LIBERO-wide validation.",
            "No rollout, Moka attack, benchmark, or LIBERO clean scan was run for this report.",
        ]
    )
    (output_dir / "blackbowl_genericv4_manual_audit_summary_20260517.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    final_lines = [
        "# Extended Long-Run Generic v4 LIBERO Summary Manual-Audited 20260517",
        "",
        "Final manual-audited decision: ready_for_cq_v2_blackbowl_calibrated_rerun_before_libero_breadth",
        "",
        "- Original auto-CQ conclusion: effect_partial",
        "- Manual-audited conclusion: replication_passed",
        f"- CQ v2 calibrated result: {calibration}",
        "- LIBERO breadth status: blocked until CQ v2 is rerun/accepted for denominator preparation.",
        "",
        "Do not claim CQ v2 is LIBERO-wide validated.",
    ]
    (output_dir / "extended_longrun_genericv4_libero_summary_manual_audited_20260517.md").write_text(
        "\n".join(final_lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate CQ v2 against blinded manual review labels.")
    parser.add_argument("--mapping_csv", required=True)
    parser.add_argument("--review_csv", required=True)
    parser.add_argument("--cq_v2_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    merged, group_rows, metrics = build_reports(Path(args.mapping_csv), Path(args.review_csv), Path(args.cq_v2_csv))
    write_csv(output_dir / "manual_review_merged_cleaned.csv", merged)
    write_csv(output_dir / "manual_review_group_summary_cleaned.csv", group_rows)
    write_csv(output_dir / "cq_manual_calibration_report_20260517.csv", merged)
    write_summaries(output_dir, metrics, group_rows)
    print(f"wrote {output_dir}")
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
