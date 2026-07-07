import csv
import hashlib
import json
from pathlib import Path

from tools.multisuite_detector import select_c6_state_binding_from_parent_v1 as m


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows):
    fields = ["source_path", "line", "source_kind", "locator", "match_reasons", "path_fields", "index_fields", "resolved_path", "resolved_path_exists", "is_concrete_binding", "preview"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def test_state_suffix_index():
    assert m.state_suffix_index("libero_goal/task_01/state_000") == 0
    assert m.state_suffix_index("x/state_049") == 49
    assert m.state_suffix_index("x/no_state") is None


def test_index_handles():
    row = {"index_fields": json.dumps({"state_id": 0, "episode_idx": 7, "noise": 3})}
    assert set(m.index_handles(row)) == {"state_id:0", "episode_idx:7"}


def test_handle_breakdown_ignores_nonconcrete():
    rows = [
        {"is_concrete_binding": True, "index_fields": json.dumps({"state_id": 0})},
        {"is_concrete_binding": False, "index_fields": json.dumps({"state_id": 1})},
    ]
    assert m.handle_breakdown(rows) == {"state_id:0": 1}


def test_selector_passes_on_state_id_zero(tmp_path):
    c6 = tmp_path / "c6_1h.json"
    c6.write_text(json.dumps({"status": "HOLD_AMBIGUOUS_STATE_BINDING", "selected_parent": {"parent_id": "libero_goal/task_01/state_000"}}), encoding="utf-8")
    cand = tmp_path / "candidates.csv"
    write_csv(cand, [{"is_concrete_binding": True, "index_fields": json.dumps({"state_id": 0}), "source_path": "idx.json"}, {"is_concrete_binding": True, "index_fields": json.dumps({"state_id": 1}), "source_path": "idx.json"}])
    out = tmp_path / "out"
    args = type("Args", (), {"input_c6_1h_json": str(c6), "expected_c6_1h_sha256": sha256(c6), "candidate_csv": str(cand), "output_root": str(out), "max_selected_rows": 10, "git_commit": "test", "files_changed": [], "tests": []})()
    assert m.run(args) == 0
    report = json.loads((out / "parent_suffix_state_binding_selector.json").read_text(encoding="utf-8"))
    assert report["status"] == m.PASS
    assert report["selected_handle"] == "state_id:0"


def test_selector_holds_when_target_absent(tmp_path):
    c6 = tmp_path / "c6_1h.json"
    c6.write_text(json.dumps({"status": "HOLD_AMBIGUOUS_STATE_BINDING", "selected_parent": {"parent_id": "libero_goal/task_01/state_000"}}), encoding="utf-8")
    cand = tmp_path / "candidates.csv"
    write_csv(cand, [{"is_concrete_binding": True, "index_fields": json.dumps({"state_id": 2}), "source_path": "idx.json"}])
    out = tmp_path / "out"
    args = type("Args", (), {"input_c6_1h_json": str(c6), "expected_c6_1h_sha256": sha256(c6), "candidate_csv": str(cand), "output_root": str(out), "max_selected_rows": 10, "git_commit": "test", "files_changed": [], "tests": []})()
    assert m.run(args) == 2
    report = json.loads((out / "parent_suffix_state_binding_selector.json").read_text(encoding="utf-8"))
    assert report["status"] == "HOLD_PARENT_SUFFIX_TARGET_HANDLE_NOT_FOUND"
