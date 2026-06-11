#!/usr/bin/env python3
"""S20e Layer1: Run clean-only contact-window detector on S20d clean traces.
Converts S20d trace CSV to detector-compatible format, then runs detect_window()."""
import csv, json, sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from detect_contact_window_from_clean import (
    detect_window, phase_cues, clamp_window, DetectorConfig, load_config,
    write_csv, available_signals,
)
# Monkey-patch: rename obj_* columns to object_* for detector compatibility
import detect_contact_window_from_clean as dmod

_orig_object_z = dmod.object_z
def _object_z_patched(row):
    # Try standard keys first
    for key in ("object_z_after", "target_object_z_after", "object_z",
                "obj_z_after", "obj_z", "bowl_z_after", "bowl_z"):
        if key in row and row.get(key) not in (None, ""):
            return float(row[key])
    return None
dmod.object_z = _object_z_patched

_orig_eef_z = dmod.eef_z
def _eef_z_patched(row):
    for key in ("eef_z_after", "eef_z", "robot0_eef_pos_z", "proxy_lift_carry_eef_z"):
        if key in row and row.get(key) not in (None, ""):
            return float(row[key])
    return None
dmod.eef_z = _eef_z_patched


def trace_to_steps(csv_path):
    """Convert S20d trace CSV to list of dicts compatible with detector."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    # Add step_idx alias
    for r in rows:
        if "step" in r and "step_idx" not in r:
            r["step_idx"] = r["step"]
    return rows


def main():
    traces = {
        ("ketchup", "1"): "/data/liuyu/outputs/stageb_s20d_v4_official_l3_20260611/smoke/trace_ketchup_s1_w0_10_s20d_clean_seed0_job960001.csv",
        ("tomato_sauce", "3"): "/data/liuyu/outputs/stageb_s20d_v4_official_l3_20260611/smoke/trace_tomato_sauce_s3_w0_10_s20d_clean_seed0_job960003.csv",
        ("tomato_sauce", "5"): "/data/liuyu/outputs/stageb_s20d_v4_official_l3_20260611/smoke/trace_tomato_sauce_s5_w0_10_s20d_clean_seed0_job960004.csv",
    }

    out_csv = "/data/liuyu/outputs/stageb_s20e_mainline_official_closure_20260611/s20e_layer1_candidates.csv"
    cue_csv = "/data/liuyu/outputs/stageb_s20e_mainline_official_closure_20260611/s20e_layer1_phase_cues.csv"
    config_path = "configs/generic_autowindow_detector.yaml"

    cfg, cfg_hash = load_config(Path(config_path) if Path(config_path).exists() else Path("scripts") / ".." / config_path)
    if not isinstance(cfg, DetectorConfig):
        # Config file may not exist locally; use defaults
        cfg = DetectorConfig()
        cfg_hash = "default_v0"

    candidate_rows = []
    cue_rows = []

    for (task, sid), trace_path in traces.items():
        if not Path(trace_path).exists():
            print(f"MISSING: {trace_path}")
            continue

        rows = trace_to_steps(trace_path)
        clean_success = any(r.get("success_primary") == "1" for r in rows)

        # Run detector
        result = detect_window(rows, clean_success, cfg)
        cues = phase_cues(rows, clean_success, cfg)
        signals = available_signals(rows)

        candidate_id = f"{task}_s{sid}_v0_detector"
        result["candidate_id"] = candidate_id
        result["task"] = task
        result["state_id"] = sid
        result["signal_count"] = len(signals)
        result["signals_list"] = ";".join(signals)
        candidate_rows.append(result)

        cues["candidate_id"] = candidate_id
        cues["task"] = task
        cues["state_id"] = sid
        cues["signal_count"] = len(signals)
        cues["signals_list"] = ";".join(signals)
        cue_rows.append(cues)

        print(f"[{task}_s{sid}] detected={result['window_detected']} "
              f"window={result.get('auto_window_start','')}-{result.get('auto_window_end','')} "
              f"mode={result['detector_mode']} confidence={result['confidence']} "
              f"grasp={cues.get('grasp_step','')} lift={cues.get('lift_step','')} "
              f"carry={cues.get('carry_start_step','')} preplace={cues.get('selected_preplace_step','')} "
              f"signals={';'.join(signals)} failure={result.get('failure_reason','')}")

    # Write CSVs
    if candidate_rows:
        fieldnames = ["candidate_id", "task", "state_id", "window_detected",
                      "auto_window_start", "auto_window_end", "auto_window_len",
                      "detector_mode", "confidence", "mechanism_type",
                      "grasp_step", "lift_step", "carry_start_step",
                      "near_target_step", "eef_descent_step", "preplace_step",
                      "release_intent_step", "done_step",
                      "signals_available", "signal_count", "signals_list",
                      "failure_reason"]
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(candidate_rows)

    if cue_rows:
        with open(cue_csv, "w", newline="") as f:
            fieldnames = list(cue_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(cue_rows)

    print(f"\nOutput: {out_csv}")
    print(f"Phase cues: {cue_csv}")


if __name__ == "__main__":
    main()
