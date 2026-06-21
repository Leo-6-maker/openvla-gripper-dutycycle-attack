import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.stageb.train_provisional_layer2 import run_one  # noqa: E402


def test_run_one_writes_skip_artifact_for_missing_supervised_rows(tmp_path):
    rows = [
        {
            "dataset_split": "train",
            "suite": "libero_spatial",
            "ignore_for_loss": "0",
        }
    ]
    summary = run_one(
        name="M1_in_domain_libero_10",
        dataset_rows=rows,
        train_suites={"libero_10"},
        val_suites={"libero_10"},
        test_suites={"libero_10"},
        output_dir=tmp_path,
        seed=1,
        device="cpu",
        epochs=1,
        dataset_sha="abc",
    )
    metrics_path = tmp_path / "M1_in_domain_libero_10" / "metrics.json"
    assert summary["run_status"] == "SKIPPED_NO_SUPERVISED_ROWS"
    assert "no rows for split=train" in summary["skip_reason"]
    assert summary["checkpoint_sha256"] == ""
    assert metrics_path.exists()
    written = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert written["run_status"] == "SKIPPED_NO_SUPERVISED_ROWS"
    assert written["n_train_rows"] == 0

