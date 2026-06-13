#!/usr/bin/env python3
"""Offline Layer3 objective-transfer audit.

This utility compares surrogate-objective telemetry with actual autoregressive
gripper-token/action telemetry already present in committed or copied artifacts.
It does not load OpenVLA, run rollouts, or perform inference.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "task",
    "state_id",
    "seed",
    "condition",
    "objective",
    "initial_surrogate_loss",
    "final_surrogate_loss",
    "surrogate_improvement",
    "teacher_forced_gripper_margin_before",
    "teacher_forced_gripper_margin_after",
    "generated_prefix_gripper_margin_before",
    "generated_prefix_gripper_margin_after",
    "clean_generated_arm_tokens",
    "perturbed_image_generated_arm_tokens",
    "arm_token_match_rate",
    "final_generated_gripper_token",
    "discrete_index_before_clip",
    "discrete_index_after_clip",
    "clipped_boolean",
    "decoded_raw_gripper",
    "executed_environment_gripper",
    "official_OPEN",
    "native_token_OPEN",
    "clip_mediated_OPEN",
    "diagnostic_category",
    "source_path",
]


def as_float(value: Any, default: float | None = None) -> float | None:
    if value in ("", None):
        return default
    try:
        return float(value)
    except Exception:
        return default


def as_int(value: Any, default: int | None = None) -> int | None:
    if value in ("", None):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in ("", None):
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def json_compact(value: Any) -> str:
    if value in ("", None):
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return [dict(row) for row in obj if isinstance(row, dict)]
    if isinstance(obj, dict):
        return [obj]
    return []


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def iter_artifact_rows(inputs: list[Path]):
    for root in inputs:
        paths: list[Path]
        if root.is_dir():
            paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".json", ".csv"})
        else:
            paths = [root]
        for path in paths:
            try:
                rows = load_json_rows(path) if path.suffix.lower() == ".json" else load_csv_rows(path)
            except Exception as exc:
                yield {"source_path": str(path), "diagnostic_category": "INSUFFICIENT_ARTIFACTS", "load_error": str(exc)}
                continue
            for row in rows:
                row = dict(row)
                row["source_path"] = str(path)
                yield row


def first_present(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] not in ("", None):
            return row[name]
    return ""


def categorize(out: dict[str, Any]) -> str:
    initial = as_float(out["initial_surrogate_loss"])
    final = as_float(out["final_surrogate_loss"])
    gen_before = as_float(out["generated_prefix_gripper_margin_before"])
    gen_after = as_float(out["generated_prefix_gripper_margin_after"])
    tf_before = as_float(out["teacher_forced_gripper_margin_before"])
    tf_after = as_float(out["teacher_forced_gripper_margin_after"])
    official_open = as_bool(out["official_OPEN"])
    native_open = as_bool(out["native_token_OPEN"])
    clipped = as_bool(out["clipped_boolean"])
    arm_match = as_float(out["arm_token_match_rate"])

    if native_open is True:
        return "AR_OPEN_VIA_NATIVE_TOKEN"
    if official_open is True and clipped is True:
        return "AR_OPEN_VIA_OFFICIAL_CLIP"
    if arm_match is not None and arm_match < 1.0:
        return "ARM_PREFIX_DRIFT"
    if None in (initial, final, gen_after, tf_after):
        return "INSUFFICIENT_ARTIFACTS"
    surrogate_improved = final < initial
    tf_improved = tf_after > (tf_before if tf_before is not None else -1e30)
    gen_improved = gen_after > (gen_before if gen_before is not None else -1e30)
    if surrogate_improved and gen_improved:
        return "SURROGATE_AND_AR_AGREE"
    if surrogate_improved and not gen_improved:
        return "SURROGATE_IMPROVES_AR_UNCHANGED"
    if surrogate_improved and tf_improved and gen_after < 0:
        return "SURROGATE_IMPROVES_AR_REVERSES"
    return "INSUFFICIENT_ARTIFACTS"


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    initial = first_present(row, "selected_loss_initial", "target_ce_initial", "initial_surrogate_loss")
    final = first_present(row, "selected_loss_final", "target_ce_final", "final_surrogate_loss")
    initial_f = as_float(initial)
    final_f = as_float(final)
    raw_gripper = first_present(row, "step_executed_gripper_raw", "decoded_raw_gripper")
    env_gripper = first_present(row, "executed_gripper_env", "decoded_open_env", "executed_environment_gripper")
    official_open = first_present(row, "c2o_official", "official_OPEN")
    native_open = first_present(row, "c2o_native_open", "native_token_OPEN")
    clip_open = first_present(row, "c2o_clip_mediated", "clip_mediated_OPEN")
    out = {
        "task": first_present(row, "task", "task_name"),
        "state_id": first_present(row, "state_id"),
        "seed": first_present(row, "attack_seed", "seed"),
        "condition": first_present(row, "condition"),
        "objective": first_present(row, "objective_name", "attack_objective", "step_attack_objective", "objective"),
        "initial_surrogate_loss": initial,
        "final_surrogate_loss": final,
        "surrogate_improvement": "" if initial_f is None or final_f is None else initial_f - final_f,
        "teacher_forced_gripper_margin_before": first_present(row, "teacher_forced_gripper_margin_initial", "teacher_forced_gripper_margin_before"),
        "teacher_forced_gripper_margin_after": first_present(row, "teacher_forced_gripper_margin_final", "teacher_forced_gripper_margin_after", "gripper_logit_margin_after"),
        "generated_prefix_gripper_margin_before": first_present(row, "generated_prefix_gripper_margin_initial", "generated_prefix_gripper_margin_before"),
        "generated_prefix_gripper_margin_after": first_present(row, "generated_prefix_gripper_margin_final", "generated_prefix_gripper_margin_after"),
        "clean_generated_arm_tokens": json_compact(first_present(row, "clean_arm_prefix_token_ids", "clean_generated_arm_tokens")),
        "perturbed_image_generated_arm_tokens": json_compact(first_present(row, "generated_arm_prefix_token_ids", "perturbed_image_generated_arm_tokens")),
        "arm_token_match_rate": first_present(row, "arm_token_match_rate", "token_match_rate"),
        "final_generated_gripper_token": first_present(row, "step_gripper_token_id", "final_generated_gripper_token"),
        "discrete_index_before_clip": first_present(row, "step_gripper_disc_before", "discrete_index_before_clip"),
        "discrete_index_after_clip": first_present(row, "step_gripper_disc_after", "discrete_index_after_clip"),
        "clipped_boolean": first_present(row, "step_gripper_clipped", "clipped_boolean"),
        "decoded_raw_gripper": raw_gripper,
        "executed_environment_gripper": env_gripper,
        "official_OPEN": official_open,
        "native_token_OPEN": native_open,
        "clip_mediated_OPEN": clip_open,
        "diagnostic_category": first_present(row, "diagnostic_category"),
        "source_path": first_present(row, "source_path"),
    }
    if not out["diagnostic_category"]:
        out["diagnostic_category"] = categorize(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Artifact files or directories containing summary JSON / trace CSV files.")
    parser.add_argument("--output-dir", default="outputs/layer3_objective_transfer_audit", help="Directory for CSV/JSON outputs.")
    args = parser.parse_args()

    inputs = [Path(p) for p in args.inputs]
    rows = [normalize_row(row) for row in iter_artifact_rows(inputs)]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "layer3_objective_transfer_audit.csv"
    json_path = out_dir / "layer3_objective_transfer_audit.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in FIELDNAMES})
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[ok] wrote {len(rows)} rows")
    print(f"[ok] csv={csv_path}")
    print(f"[ok] json={json_path}")


if __name__ == "__main__":
    main()
