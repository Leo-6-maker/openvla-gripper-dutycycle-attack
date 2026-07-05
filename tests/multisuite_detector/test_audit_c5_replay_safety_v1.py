import csv
import hashlib
import json
from pathlib import Path

from tools.multisuite_detector.audit_c5_replay_safety_v1 import run


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["suite", "role", "positive_support", "hit_rate", "emission_rate", "safety_false_trigger_rate"])
        writer.writeheader()
        writer.writerows(rows)


def write_sums(root: Path):
    lines = []
    for p in sorted(x for x in root.iterdir() if x.is_file() and x.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}):
        lines.append(f"{sha(p)}  {p.name}")
    sums = root / "SHA256SUMS"
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "SHA256SUMS.sha256").write_text(f"{sha(sums)}  SHA256SUMS\n", encoding="utf-8")


def make_c5(root: Path, object_safety="0.10", l10_safety="0.56"):
    root.mkdir()
    write_json(root / "replay_manifest.json", {"status": "PASS"})
    write_json(root / "metrics_overall.json", {"status": "PASS", "hit_rate": 0.64, "emission_rate": 0.56, "safety_false_trigger_rate": 0.42})
    write_csv(root / "metrics_by_suite.csv", [
        {"suite": "libero_goal", "role": "primary_positive", "positive_support": "100", "hit_rate": "0.64", "emission_rate": "0.31", "safety_false_trigger_rate": "0.095"},
        {"suite": "libero_object", "role": "primary_positive", "positive_support": "100", "hit_rate": "0.56", "emission_rate": "0.75", "safety_false_trigger_rate": object_safety},
        {"suite": "libero_spatial", "role": "primary_positive", "positive_support": "100", "hit_rate": "0.73", "emission_rate": "0.69", "safety_false_trigger_rate": "0.12"},
        {"suite": "libero_10", "role": "diagnostic_only", "positive_support": "0", "hit_rate": "NOT_APPLICABLE", "emission_rate": "0.50", "safety_false_trigger_rate": l10_safety},
    ])
    write_sums(root)


def call(root: Path, out: Path):
    args = type("Args", (), {
        "c5_root": str(root),
        "output_root": str(out),
        "max_primary_safety_false_trigger": 0.15,
        "max_diagnostic_safety_false_trigger": 0.50,
    })()
    return run(args)


def test_c5_safety_triage_primary_ok_diagnostic_hold(tmp_path):
    root = tmp_path / "c5"
    make_c5(root)
    report = call(root, tmp_path / "out")
    assert report["status"] == "PASS_PRIMARY_HOLD_DIAGNOSTIC"
    assert (tmp_path / "out" / "primary_suite_safety_table.csv").is_file()
    assert (tmp_path / "out" / "c6_release_recommendation.json").is_file()


def test_c5_safety_triage_primary_hold(tmp_path):
    root = tmp_path / "c5"
    make_c5(root, object_safety="0.30", l10_safety="0.20")
    report = call(root, tmp_path / "out")
    assert report["status"] == "HOLD_PRIMARY_SAFETY"
