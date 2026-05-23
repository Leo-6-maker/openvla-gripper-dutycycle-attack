"""Autowindow runner — command builder and validation helpers.

Provides lightweight utilities for invoking the generic autowindow detector.
Does NOT run the detector here — just builds commands and validates artifacts.
"""
import shlex
from pathlib import Path

from src.utils.autowindow_protocols import TABLE1_GENERIC_AUTOWINDOW_PROTOCOL


def build_generic_autowindow_command(
    input_root,
    output_csv,
    phase_cues_csv=None,
    summary_md=None,
    config=None,
):
    """Build a CLI command list to run the generic autowindow detector.

    Args:
        input_root: Path to the directory containing clean run artifacts.
        output_csv: Path for the detector output CSV.
        phase_cues_csv: Optional path for the phase cues CSV.
        summary_md: Optional path for the summary markdown.
        config: Optional path to the detector YAML config.
            Defaults to TABLE1_GENERIC_AUTOWINDOW_PROTOCOL["detector_config"].

    Returns:
        List of command-line tokens.
    """
    if config is None:
        config = TABLE1_GENERIC_AUTOWINDOW_PROTOCOL["detector_config"]

    script = TABLE1_GENERIC_AUTOWINDOW_PROTOCOL["detector_script"]
    cmd = ["python", script, "--input_root", str(input_root), "--output_csv", str(output_csv), "--config", str(config)]

    if phase_cues_csv:
        cmd += ["--phase_cues_csv", str(phase_cues_csv)]
    if summary_md:
        cmd += ["--summary_md", str(summary_md)]

    return cmd


def build_generic_autowindow_command_string(
    input_root, output_csv, phase_cues_csv=None, summary_md=None, config=None
):
    """Build a shell-safe command string for the generic autowindow detector."""
    return " ".join(
        shlex.quote(tok)
        for tok in build_generic_autowindow_command(
            input_root, output_csv, phase_cues_csv, summary_md, config
        )
    )


def validate_generic_autowindow_artifacts(output_csv, phase_cues_csv=None):
    """Validate that the detector output artifacts exist and are non-empty.

    Returns a dict with status per artifact.
    """
    result = {}
    for label, path in [("output_csv", output_csv), ("phase_cues_csv", phase_cues_csv)]:
        if path is None:
            result[label] = "not_requested"
            continue
        p = Path(path)
        if not p.exists():
            result[label] = "missing"
        elif p.stat().st_size == 0:
            result[label] = "empty"
        else:
            result[label] = "ok"
    return result


def load_autowindow_rows(csv_path):
    """Load detector output rows from a CSV file.

    Returns a list of dicts. Returns an empty list if the file is missing or empty.
    """
    import csv

    path = Path(csv_path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def filter_eligible_rows(rows, min_confidence="medium"):
    """Filter a list of detector rows to those eligible for command-open.

    Requires: window_detected=True, clean_success=True, confidence >= min_confidence,
    detector_mode not in ('failed_no_signal',).

    IMPORTANT: This is a coarse detector-output filter, NOT final command-open
    eligibility. Final rollout gating MUST additionally check:
      - task identity (is_black_bowl_related, suite, runner_task_id)
      - phase label (not post_release, not gripper_open_throughout)
      - window_source is fresh_clean_generic_autowindow
      - detector_config_hash present and matching
      - model suite matches task suite
      - same-seed protocol (matched_seed == clean_seed)
      - protocol validators pass (command_open rho>0, etc.)
    """
    eligible = []
    for row in rows:
        wd = row.get("window_detected")
        if isinstance(wd, str):
            wd = wd.strip().lower() in ("true", "1", "yes")
        if not wd:
            continue

        cs = row.get("clean_success")
        if isinstance(cs, str):
            cs = cs.strip().lower() in ("true", "1", "yes")
        if not cs:
            continue

        conf = str(row.get("confidence", "")).lower()
        mode = str(row.get("detector_mode", ""))

        if mode == "failed_no_signal":
            continue

        if min_confidence == "high":
            if conf != "high":
                continue
        elif min_confidence == "medium":
            if conf not in ("medium", "high"):
                continue

        eligible.append(row)

    return eligible
