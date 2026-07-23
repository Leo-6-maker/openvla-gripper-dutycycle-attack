"""CPU tests for pilot attack validators."""
from __future__ import annotations

import json, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/student_trigger_calibration"))
sys.path.insert(0, str(ROOT / "analysis/pilot_attack"))

sys.path.insert(0, str(ROOT / "analysis/pilot_attack"))
from pilot_integrity import sha256_file, seal_output_dir

ALL_SPLITS = [f"o{oi}_i{ii}" for oi in range(4) for ii in range(3)]
FEC_IDS = {f"fec_{i}" for i in range(16)}


def _seal_single(root: Path, filename: str, data: dict):
    root.mkdir(parents=True, exist_ok=True)
    (root / filename).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    files = sorted(p for p in root.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (root / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
    (root / "SHA256SUMS.sha256").write_text(f"{sha256_file(root / 'SHA256SUMS')}  SHA256SUMS\n")


# ═══════════════════════════════════════════════════════════════════
# B1: Parent manifest validator tests
# ═══════════════════════════════════════════════════════════════════

def test_parent_in_fec_pass():
    from validate_factorized_pilot_parent_manifest import main as pm_main
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)

        fec_data = {"identities": list(FEC_IDS)}
        _seal_single(dp / "fec", "manifest.json", fec_data)

        parent_data = {"parents": [
            {"parent_id": f"fec_{i}", "suite": "s0", "task": "t0",
             "clean_success": True, "detector_emitted": True,
             "remaining_horizon": 20, "selection_rank": i}
            for i in range(8)
        ]}
        _seal_single(dp / "parents", "manifest.json", parent_data)

        detector = {"paper_authoritative": False, "attack_eval_consumed": False}
        _seal_single(dp / "detector", "config.json", detector)

        for role in ["T", "C", "P", "H", "A"]:
            _seal_single(dp / role, "manifest.json", {"identities": [f"other_{role}_{j}" for j in range(10)]})

        old = sys.argv
        try:
            sys.argv = ["pm", "--pilot-parent-manifest", str(dp / "parents/manifest.json"),
                        "--reserved-fec-manifest", str(dp / "fec/manifest.json"),
                        "--t-manifest", str(dp / "T/manifest.json"),
                        "--c-manifest", str(dp / "C/manifest.json"),
                        "--p-manifest", str(dp / "P/manifest.json"),
                        "--h-manifest", str(dp / "H/manifest.json"),
                        "--a-manifest", str(dp / "A/manifest.json"),
                        "--pilot-detector-config", str(dp / "detector/config.json"),
                        "--output-root", str(dp / "out")]
            rc = pm_main(); assert rc == 0
        finally: sys.argv = old


def test_parent_not_in_fec_rejected():
    from validate_factorized_pilot_parent_manifest import main as pm_main
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        _seal_single(dp / "fec", "manifest.json", {"identities": ["fec_0"]})
        _seal_single(dp / "parents", "manifest.json", {"parents": [
            {"parent_id": "not_fec", "suite": "s0", "task": "t0",
             "clean_success": True, "detector_emitted": True,
             "remaining_horizon": 20, "selection_rank": 0}
        ]})
        _seal_single(dp / "detector", "config.json", {"paper_authoritative": False, "attack_eval_consumed": False})
        for role in ["T", "C", "P", "H", "A"]:
            _seal_single(dp / role, "manifest.json", {"identities": []})
        old = sys.argv
        try:
            sys.argv = ["pm", "--pilot-parent-manifest", str(dp / "parents/manifest.json"),
                        "--reserved-fec-manifest", str(dp / "fec/manifest.json"),
                        "--t-manifest", str(dp / "T/manifest.json"),
                        "--c-manifest", str(dp / "C/manifest.json"),
                        "--p-manifest", str(dp / "P/manifest.json"),
                        "--h-manifest", str(dp / "H/manifest.json"),
                        "--a-manifest", str(dp / "A/manifest.json"),
                        "--pilot-detector-config", str(dp / "detector/config.json"),
                        "--output-root", str(dp / "out")]
            rc = pm_main(); assert rc != 0
        finally: sys.argv = old


def test_parent_in_t_rejected():
    from validate_factorized_pilot_parent_manifest import main as pm_main
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        _seal_single(dp / "fec", "manifest.json", {"identities": ["fec_0"]})
        _seal_single(dp / "parents", "manifest.json", {"parents": [
            {"parent_id": "fec_0", "suite": "s0", "task": "t0",
             "clean_success": True, "detector_emitted": True,
             "remaining_horizon": 20, "selection_rank": 0}
        ]})
        _seal_single(dp / "detector", "config.json", {"paper_authoritative": False, "attack_eval_consumed": False})
        _seal_single(dp / "T", "manifest.json", {"identities": ["fec_0"]})
        for role in ["C", "P", "H", "A"]:
            _seal_single(dp / role, "manifest.json", {"identities": []})
        old = sys.argv
        try:
            sys.argv = ["pm", "--pilot-parent-manifest", str(dp / "parents/manifest.json"),
                        "--reserved-fec-manifest", str(dp / "fec/manifest.json"),
                        "--t-manifest", str(dp / "T/manifest.json"),
                        "--c-manifest", str(dp / "C/manifest.json"),
                        "--p-manifest", str(dp / "P/manifest.json"),
                        "--h-manifest", str(dp / "H/manifest.json"),
                        "--a-manifest", str(dp / "A/manifest.json"),
                        "--pilot-detector-config", str(dp / "detector/config.json"),
                        "--output-root", str(dp / "out")]
            rc = pm_main(); assert rc != 0
        finally: sys.argv = old


# ═══════════════════════════════════════════════════════════════════
# B2: Execution validator tests
# ═══════════════════════════════════════════════════════════════════

def test_k_executed_9_rejected():
    from validate_factorized_attack_pilot_execution import main as ev_main
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        _seal_single(dp / "jobs", "matrix.json", {"entries": []})
        _seal_single(dp / "ledger", "ledger.json", {"runs": [
            {"parent_id": "fec_0", "condition": "CLEAN", "run_index": 0,
             "k_requested": 10, "k_executed": 9, "attack_start_step": 5,
             "attack_end_step": 15, "checkpoint_sha256": "a" * 64,
             "detector_triggered": True, "arm_contact": True,
             "video_path": "v.mp4", "telemetry_path": "t.json"}
        ]})
        _seal_single(dp / "telem", "index.json", {"entries": ["t.json"]})
        _seal_single(dp / "video", "index.json", {"entries": ["v.mp4"]})
        _seal_single(dp / "parents", "manifest.json", {"parents": [
            {"parent_id": "fec_0", "suite": "s0", "task": "t0",
             "clean_success": True, "detector_emitted": True,
             "remaining_horizon": 20, "selection_rank": 0}
        ]})
        old = sys.argv
        try:
            sys.argv = ["ev", "--pilot-job-matrix", str(dp / "jobs/matrix.json"),
                        "--pilot-run-ledger", str(dp / "ledger/ledger.json"),
                        "--pilot-telemetry-index", str(dp / "telem/index.json"),
                        "--pilot-video-index", str(dp / "video/index.json"),
                        "--pilot-parent-manifest", str(dp / "parents/manifest.json"),
                        "--output-root", str(dp / "out")]
            rc = ev_main(); assert rc != 0
        finally: sys.argv = old


def test_missing_video_rejected():
    from validate_factorized_attack_pilot_execution import main as ev_main
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        _seal_single(dp / "jobs", "matrix.json", {"entries": []})
        _seal_single(dp / "ledger", "ledger.json", {"runs": [
            {"parent_id": "fec_0", "condition": "CLEAN", "run_index": 0,
             "k_requested": 10, "k_executed": 10, "attack_start_step": 5,
             "attack_end_step": 15, "checkpoint_sha256": "a" * 64,
             "detector_triggered": True, "arm_contact": True,
             "video_path": "missing.mp4", "telemetry_path": "t.json"}
        ]})
        _seal_single(dp / "telem", "index.json", {"entries": ["t.json"]})
        _seal_single(dp / "video", "index.json", {"entries": ["other.mp4"]})
        _seal_single(dp / "parents", "manifest.json", {"parents": [
            {"parent_id": "fec_0", "suite": "s0", "task": "t0",
             "clean_success": True, "detector_emitted": True,
             "remaining_horizon": 20, "selection_rank": 0}
        ]})
        old = sys.argv
        try:
            sys.argv = ["ev", "--pilot-job-matrix", str(dp / "jobs/matrix.json"),
                        "--pilot-run-ledger", str(dp / "ledger/ledger.json"),
                        "--pilot-telemetry-index", str(dp / "telem/index.json"),
                        "--pilot-video-index", str(dp / "video/index.json"),
                        "--pilot-parent-manifest", str(dp / "parents/manifest.json"),
                        "--output-root", str(dp / "out")]
            rc = ev_main(); assert rc != 0
        finally: sys.argv = old


# ═══════════════════════════════════════════════════════════════════
# B4: Blind review tests
# ═══════════════════════════════════════════════════════════════════

def test_blind_review_hides_condition():
    from build_factorized_pilot_blind_review import main as br_main
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        _seal_single(dp / "ledger", "ledger.json", {"runs": [
            {"parent_id": "fec_0", "condition": "TRUE_T10", "video_path": "v0.mp4"},
            {"parent_id": "fec_1", "condition": "CLEAN", "video_path": "v1.mp4"},
        ]})
        _seal_single(dp / "video", "index.json", {"entries": ["v0.mp4", "v1.mp4"]})
        old = sys.argv
        try:
            sys.argv = ["br", "--pilot-run-ledger", str(dp / "ledger/ledger.json"),
                        "--pilot-video-index", str(dp / "video/index.json"),
                        "--output-root", str(dp / "out")]
            rc = br_main(); assert rc == 0
        finally: sys.argv = old

        # Verify blind review doesn't expose condition
        import csv
        with open(dp / "out/PILOT_BLIND_REVIEW_V0.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                assert "condition" not in row or not row.get("condition")

        # Verify unblinding has condition
        with open(dp / "out/PILOT_UNBLINDING_V0.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                assert "condition" in row


def test_blind_id_unique():
    """Blind IDs should be unique within a review package."""
    from build_factorized_pilot_blind_review import main as br_main
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        runs = [{"parent_id": f"fec_{i}", "condition": f"COND_{i}", "video_path": f"v{i}.mp4"}
                for i in range(50)]
        _seal_single(dp / "ledger", "ledger.json", {"runs": runs})
        _seal_single(dp / "video", "index.json", {"entries": [f"v{i}.mp4" for i in range(50)]})
        old = sys.argv
        try:
            sys.argv = ["br", "--pilot-run-ledger", str(dp / "ledger/ledger.json"),
                        "--pilot-video-index", str(dp / "video/index.json"),
                        "--output-root", str(dp / "out")]
            rc = br_main(); assert rc == 0
        finally: sys.argv = old

        import csv
        blind_ids = set()
        with open(dp / "out/PILOT_BLIND_REVIEW_V0.csv") as f:
            for row in csv.DictReader(f):
                bid = row["blind_id"]
                assert bid not in blind_ids, f"Duplicate blind_id: {bid}"
                blind_ids.add(bid)
