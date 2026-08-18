#!/usr/bin/env python3
"""Project only safe fields from the private D1M0 mapping to the owner sheet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
MAPPING = REPO / "reports/STAGE_X_X1R_T1D1M0_MANUAL_REVIEW_MAPPING_V1.json"
RENDER = REPO / "reports/STAGE_X_X1R_T1D1M0_REVIEW_RENDER_MANIFEST_V1.json"
D1M0_SUMS = REPO / "reports/STAGE_X_X1R_T1D1M0_SHA256SUMS.txt"
OUT_CSV = REPO / "reports/STAGE_X_X1R_T1D1M0R_HUMAN_REVIEW_SHEET_V1.csv"
OUT_JSON = REPO / "reports/STAGE_X_X1R_T1D1M0R_HUMAN_REVIEW_SHEET_V1.json"
EXPECTED_LEDGER_SHA = "5f1f036b47b1c9a8c1bafe7a400b6be9269cd3e67587691018005c824dc8d89e"
EXPECTED_MAPPING_SHA = "3d7f59a736cc2c7bcb5ecdc49e9e57a7e8b547c9e7554251e88158017366f0fe"
EXPECTED_ORDER_DIGEST = "30a73b0e4ab13e149d8c991906fc9067844797e39113201e9e76a10a8be40d67"
FIELDS = [
    "review_id", "task_instruction", "review_clip_path", "review_clip_sha256",
    "review_frame_strip_path", "review_frame_strip_sha256", "contact_label",
    "reason_code", "reviewer", "review_timestamp", "optional_short_note",
]


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sums_map() -> dict[str, str]:
    result = {}
    for line in D1M0_SUMS.read_text(encoding="utf-8").splitlines():
        digest, _, path = line.partition("  ")
        if digest and path:
            result[path] = digest
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REPO / "reports")
    args = parser.parse_args()
    sums = sums_map()
    if sums.get("reports/STAGE_X_X1R_T1D1M0_MANUAL_REVIEW_MAPPING_V1.json") != EXPECTED_MAPPING_SHA:
        raise SystemExit("STAGE_X_X1R_T1D1M0R_HOLD_D1M0_MAPPING_SHA")
    ledger_path = REPO / "reports/STAGE_X_X1R_T1D1M0_PREMANUAL_ELIGIBLE_LEDGER_V1.json"
    if sums.get("reports/STAGE_X_X1R_T1D1M0_PREMANUAL_ELIGIBLE_LEDGER_V1.json") != EXPECTED_LEDGER_SHA:
        raise SystemExit("STAGE_X_X1R_T1D1M0R_HOLD_D1M0_LEDGER_SHA")
    mapping = load(MAPPING)
    render = load(RENDER)
    if sha(MAPPING) != EXPECTED_MAPPING_SHA or mapping.get("status") != "FROZEN_BLINDED_ORDER_BEFORE_VIDEO_PIXEL_ACCESS":
        raise SystemExit("STAGE_X_X1R_T1D1M0R_HOLD_MAPPING_INVALID")
    if sha(ledger_path) != EXPECTED_LEDGER_SHA or render.get("status") != "PASS_FIXED_REVIEW_COPIES_RENDERED":
        raise SystemExit("STAGE_X_X1R_T1D1M0R_HOLD_D1M0_BINDING_INVALID")
    rows = mapping.get("rows", [])
    render_rows = {row["review_id"]: row for row in render.get("rows", [])}
    if [row.get("review_id") for row in rows] != [f"M{i:03d}" for i in range(1, 15)] or set(render_rows) != {f"M{i:03d}" for i in range(1, 15)}:
        raise SystemExit("STAGE_X_X1R_T1D1M0R_HOLD_ID_ORDER")
    order_digest = hashlib.sha256(json.dumps([{"review_id": row["review_id"], "rank_key": row["rank_key"], "canonical_parent_key": row["canonical_parent_key"]} for row in rows], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if order_digest != EXPECTED_ORDER_DIGEST:
        raise SystemExit("STAGE_X_X1R_T1D1M0R_HOLD_ORDER_DIGEST")

    safe_rows = []
    for mapping_row in rows:
        review_id = mapping_row["review_id"]
        render_row = render_rows[review_id]
        instruction = str(mapping_row["task_instruction"])
        if not instruction:
            raise SystemExit(f"STAGE_X_X1R_T1D1M0R_HOLD_TASK_INSTRUCTION_EMPTY:{review_id}")
        safe_rows.append({
            "review_id": review_id,
            "task_instruction": instruction,
            "review_clip_path": render_row["review_clip_path"],
            "review_clip_sha256": render_row["review_clip_sha256"],
            "review_frame_strip_path": render_row["review_frame_strip_path"],
            "review_frame_strip_sha256": render_row["review_frame_strip_sha256"],
            "contact_label": "",
            "reason_code": "",
            "reviewer": "",
            "review_timestamp": "",
            "optional_short_note": "",
        })

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / OUT_CSV.name
    out_json = out_dir / OUT_JSON.name
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(safe_rows)
    out_json.write_text(json.dumps({"schema": "STAGE_X_X1R_T1D1M0R_HUMAN_REVIEW_SHEET_V1", "status": "BLANK_SAFE_OWNER_SHEET", "fields": FIELDS, "rows": safe_rows, "next_gate": "OWNER_MANUAL_CONTACT_LABELS_REQUIRED"}, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_SAFE_HUMAN_SHEET_PROJECTED", "rows": len(safe_rows), "csv_sha256": sha(out_csv), "json_sha256": sha(out_json)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
