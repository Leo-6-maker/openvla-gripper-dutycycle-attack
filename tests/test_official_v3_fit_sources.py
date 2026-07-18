from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def test_policy_and_privileged_contract_helpers_import() -> None:
    module = importlib.import_module("scripts.detector_v5.audit_official_v3_fit_sources")
    assert module.POLICY_FIELDS[0] == "step"
    assert "object_state" in module.PRIVILEGED_FIELDS
    assert module.finite_list([1.0, 2.0], 2)
    assert not module.finite_list([1.0, float("nan")], 2)


def test_identity_parser_accepts_only_fit_rows(tmp_path: Path) -> None:
    module = importlib.import_module("scripts.detector_v5.audit_official_v3_fit_sources")
    path = tmp_path / "registry.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["canonical_parent_key", "suite", "task_idx", "state_id", "split"])
        writer.writeheader()
        for suite in ("libero_object", "libero_spatial", "libero_goal", "libero_10"):
            for task in range(10):
                for state in range(20):
                    writer.writerow({"canonical_parent_key": f"{suite}/task_{task:02d}/state_{state:02d}", "suite": suite, "task_idx": task, "state_id": state, "split": "FIT_TRAIN"})
        for suite in ("libero_object", "libero_spatial", "libero_goal", "libero_10"):
            for task in range(10):
                for state in range(20, 50):
                    writer.writerow({"canonical_parent_key": f"{suite}/task_{task:02d}/state_{state:02d}", "suite": suite, "task_idx": task, "state_id": state, "split": "OTHER"})
    rows = module.load_fit(path)
    assert len(rows) == 800
    assert rows[0]["state_id"] == "0"
    assert rows[-1]["state_id"] == "19"
