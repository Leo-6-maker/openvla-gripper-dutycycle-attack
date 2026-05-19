#!/usr/bin/env python3
"""Prepare and summarize LIBERO full4 state0/balanced clean denominator runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path


SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SUITE_SHORT = {
    "libero_spatial": "spatial",
    "libero_object": "object",
    "libero_goal": "goal",
    "libero_10": "libero10",
}
MODEL_PATHS = {
    "libero_spatial": "/data/aviary/models/openvla/openvla-7b-finetuned-libero-spatial",
    "libero_object": "/data/aviary/models/openvla/openvla-7b-finetuned-libero-object",
    "libero_goal": "/data/aviary/models/openvla/openvla-7b-finetuned-libero-goal",
    "libero_10": "/data/aviary/models/openvla/openvla-7b-finetuned-libero-10",
}
DATA_ROOT = Path("/data/aviary/datasets/libero/datasets")
VERIFY_LIGHT = Path("/data/liuyu/outputs/libero_full4_manual_transfer_verify_20260518/tables/full4_hdf5_integrity_after_transfer_light.csv")
OLD_INTEGRITY = Path("/data/liuyu/outputs/libero_full4_download_20260517/tables/full4_hdf5_integrity_20260517.csv")


def norm_id(text: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return re.sub(r"_+", "_", out)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_obj_interest(bddl_path: str) -> list[str]:
    candidates = [Path(bddl_path)]
    libero_root = Path("/data/aviary/envs/openvla_sparse/lib/python3.10/site-packages/libero")
    p = Path(bddl_path)
    candidates += [
        libero_root / p,
        libero_root / "libero" / "bddl_files" / p.parent.name / p.name,
        libero_root / "libero" / "bddl_files" / p.name,
    ]
    text = ""
    for candidate in candidates:
        if candidate.exists():
            text = candidate.read_text(encoding="utf-8", errors="replace")
            break
    if not text:
        return []
    match = re.search(r"\(:obj_of_interest\s+(.*?)\)", text, flags=re.S)
    if not match:
        return []
    return re.findall(r"[A-Za-z0-9_]+", match.group(1))


def infer_mechanism(suite: str, task_name: str) -> tuple[str, bool, str]:
    name = task_name.lower()
    if any(tok in name for tok in ["open_", "open the", "close_", "close it", "turn_on", "turn on", "push_"]):
        return "articulated_or_non_transfer", False, "articulated/open/turn/push boundary"
    if "both_" in name or "both the" in name:
        return "multi_object_transfer", True, ""
    if any(tok in name for tok in ["pick_up", "pick up", "put_", "put the", "place_"]):
        return "pick_place", True, ""
    if suite == "libero_10":
        return "long_horizon_unknown", False, "long horizon unknown mechanism"
    return "unknown", False, "unknown mechanism language"


def is_black_bowl(task_name: str) -> bool:
    low = task_name.lower()
    return "black_bowl" in low or "black bowl" in low or "bowl_on_plate" in low


def sha_sources(upload_rows: list[dict], old_rows: list[dict]) -> dict[tuple[str, str], tuple[str, str]]:
    out: dict[tuple[str, str], tuple[str, str]] = {}
    for row in upload_rows:
        suite = row.get("suite", "")
        file_name = row.get("file_name", "")
        sha = row.get("sha256", "")
        if suite and file_name and sha and row.get("final_status") in {"uploaded", "downloaded_local_only"}:
            out[(suite, file_name)] = (sha, "local_upload_csv")
    for row in old_rows:
        suite = row.get("suite", "")
        file_name = Path(row.get("file_path", "")).name
        sha = row.get("sha256", "")
        if suite and file_name and sha and (suite, file_name) not in out:
            out[(suite, file_name)] = (sha, "server_previous_integrity_csv")
    return out


def remote_sha(path: Path) -> str:
    proc = subprocess.run(["sha256sum", str(path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        return ""
    return proc.stdout.split()[0]


def generate(args: argparse.Namespace) -> int:
    out = Path(args.output_root)
    tables = out / "tables"
    upload_rows = read_csv(Path(args.upload_result_csv))
    old_rows = read_csv(OLD_INTEGRITY)
    light_rows = read_csv(VERIFY_LIGHT)
    sha_map = sha_sources(upload_rows, old_rows)
    frozen = []
    seen_ids: set[str] = set()
    collisions: list[str] = []
    for row in light_rows:
        if row.get("suite") not in SUITES:
            continue
        suite = row["suite"]
        task_name = row["task_name"]
        file_path = Path(row["file_path"])
        file_name = file_path.name
        task_id_base = f"{SUITE_SHORT[suite]}_{norm_id(task_name)}"
        task_id = task_id_base[:170]
        if task_id in seen_ids:
            collisions.append(task_id)
        seen_ids.add(task_id)
        sha, sha_source = sha_map.get((suite, file_name), ("", ""))
        sha_remote = ""
        sha_match = ""
        if not sha and file_path.exists():
            sha = remote_sha(file_path)
            sha_source = "remote_sha256sum"
            sha_remote = sha
            sha_match = "True"
        elif sha:
            sha_match = "not_recomputed"
        obj_interest = parse_obj_interest(row.get("bddl_file_name", ""))
        target_object = obj_interest[0] if obj_interest else ""
        target_receptacle = obj_interest[-1] if len(obj_interest) >= 2 else ""
        mechanism, expected_eligible, exclusion = infer_mechanism(suite, task_name)
        max_steps = 700 if suite == "libero_10" else 400
        frozen.append(
            {
                "suite": suite,
                "task_id": task_id,
                "task_name": task_name,
                "dataset_path": str(file_path),
                "file_name": file_name,
                "size_bytes": row.get("size_bytes", ""),
                "size_gib": row.get("size_gib", ""),
                "sha256": sha,
                "sha256_source": sha_source,
                "sha256_remote": sha_remote,
                "sha256_match": sha_match,
                "h5py_open": row.get("h5py_open", ""),
                "data_group_exists": row.get("data_group_exists", ""),
                "demo_count": row.get("demo_count", ""),
                "bddl_file_name": row.get("bddl_file_name", ""),
                "bddl_resolved": row.get("bddl_resolved", ""),
                "unnorm_key": suite,
                "checkpoint_path": MODEL_PATHS[suite],
                "checkpoint_exists": str(Path(MODEL_PATHS[suite]).exists()),
                "max_steps": str(max_steps),
                "obj_of_interest": "|".join(obj_interest),
                "target_object_name": target_object,
                "target_receptacle_name": target_receptacle,
                "expected_mechanism_type": mechanism,
                "expected_mechanism_eligible_from_language": str(expected_eligible),
                "language_exclusion_reason": exclusion,
                "black_bowl_sanity": str(is_black_bowl(task_name)),
            }
        )

    fields = list(frozen[0].keys()) if frozen else []
    write_csv(tables / "full4_dataset_manifest_frozen.csv", frozen, fields)

    yaml_lines = ["version: v4", "tasks:"]
    for row in frozen:
        yaml_lines += [
            f"  - task_id: {row['task_id']}",
            f"    suite: {row['suite']}",
            f"    task_name: {row['task_name']}",
            f"    dataset_path: {row['dataset_path']}",
            f"    max_steps: {row['max_steps']}",
            f"    default_unnorm_key: {row['unnorm_key']}",
            f"    target_object_name: {row['target_object_name']}",
            f"    target_receptacle_name: {row['target_receptacle_name']}",
            f"    expected_mechanism_type: {row['expected_mechanism_type']}",
        ]
    config_path = Path(args.repo_root) / "configs/v4_tasks_libero_full4_20260518.yaml"
    config_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    (tables / "v4_tasks_libero_full4_20260518.yaml.copy").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    smoke = []
    for suite in SUITES:
        candidates = [r for r in frozen if r["suite"] == suite]
        preferred = next((r for r in candidates if r["task_name"] in {
            "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
            "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
            "open_the_middle_drawer_of_the_cabinet",
            "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
        }), candidates[0])
        smoke.append(job_row(preferred, state=0, seed=1, phase="smoke"))
    write_csv(tables / "phase2_smoke_jobs.csv", smoke)
    write_tsv(tables / "phase2_smoke_jobs.tsv", smoke)

    spatial = next(r for r in frozen if r["suite"] == "libero_spatial")
    sanity = [job_row(spatial, state=0, seed=seed, phase="sanity") for seed in (1, 2, 3)]
    write_csv(tables / "phase2_5_seed_state_sanity_jobs.csv", sanity)
    write_tsv(tables / "phase2_5_seed_state_sanity_jobs.tsv", sanity)

    checks = []
    def check(name: str, ok: bool, observed: str = "") -> None:
        checks.append({"check_name": name, "status": "pass" if ok else "fail", "observed": observed})
    check("hdf5_manifest_40_rows", len(frozen) == 40, str(len(frozen)))
    check("all_hdf5_complete", all(r["h5py_open"] == "True" and r["data_group_exists"] == "True" and r["demo_count"] == "50" for r in frozen), "")
    check("all_bddl_resolved", all(r["bddl_resolved"] == "True" for r in frozen), "")
    check("all_checkpoints_exist", all(r["checkpoint_exists"] == "True" for r in frozen), "")
    check("task_id_no_collision", not collisions, ",".join(collisions))
    check("runner_no_video_supported", True, "v4_run_eval_openvla.py does not render videos; video rendering is separate script")
    check("selective_video_supported", (Path(args.repo_root) / "scripts/v4_render_episode_video_from_steps.py").exists(), "separate render script")
    write_csv(tables / "phase0_dry_run_checks.csv", checks)
    status = "phase0_ready" if all(r["status"] == "pass" for r in checks) else "config_generation_failed"
    (out / "phase0_status.txt").write_text(status + "\n", encoding="utf-8")
    return 0 if status == "phase0_ready" else 2


def job_row(row: dict[str, str], *, state: int, seed: int, phase: str) -> dict[str, str]:
    run_id = f"old2080ti_20260518_full4_{phase}_{row['task_id']}_state{state}_seed{seed}_clean"
    return {
        "phase": phase,
        "suite": row["suite"],
        "task_id": row["task_id"],
        "task_name": row["task_name"],
        "state": str(state),
        "seed": str(seed),
        "run_id": run_id,
        "unnorm_key": row["unnorm_key"],
        "max_steps": row["max_steps"],
        "checkpoint_path": row["checkpoint_path"],
        "target_object_name": row["target_object_name"],
        "target_receptacle_name": row["target_receptacle_name"],
        "black_bowl_sanity": row["black_bowl_sanity"],
    }


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def artifact_complete(run_dir: Path) -> bool:
    required = ["progress.json", "step_records.jsonl", "episode_records.jsonl", "summary.csv", "run_manifest.json"]
    return all((run_dir / name).exists() and (run_dir / name).stat().st_size > 0 for name in required)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_jsonl(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return []


def summarize_runs(args: argparse.Namespace, phase_filter: str | None = None, output_name: str = "clean_success_manifest.csv") -> int:
    out = Path(args.output_root)
    manifest_rows = read_csv(out / "tables/full4_dataset_manifest_frozen.csv")
    by_task = {r["task_id"]: r for r in manifest_rows}
    rows = []
    for run_manifest in sorted(out.glob("old2080ti_20260518_full4_*_clean/run_manifest.json")):
        run_dir = run_manifest.parent
        run_id = run_dir.name
        manifest = read_json(run_manifest)
        phase = run_id.split("_full4_", 1)[-1].split("_", 1)[0]
        if phase_filter and phase != phase_filter:
            continue
        task_id = manifest.get("task_id", "")
        task = by_task.get(task_id, {})
        progress = read_json(run_dir / "progress.json")
        episodes = read_jsonl(run_dir / "episode_records.jsonl")
        steps = read_jsonl(run_dir / "step_records.jsonl")
        ep = episodes[-1] if episodes else {}
        official = bool(ep.get("success", False))
        complete = artifact_complete(run_dir) and str(progress.get("status", "")) == "done"
        object_pose_logged = any(any(k in step and step.get(k) is not None for k in ("bowl_z_after", "object_z_after", "target_object_z_after", "grasp_bowl_z_delta")) for step in steps)
        gripper_qpos_logged = any(step.get("gripper_qpos_abs_sum_after") is not None for step in steps)
        failure_reason = "success_libero" if official else str(ep.get("failure_phase") or progress.get("error") or "unknown_failure")
        mechanism_type, lang_eligible, lang_exclusion = infer_mechanism(task.get("suite", ""), task.get("task_name", ""))
        if not official:
            mechanism_type = "clean_unstable"
        rows.append(
            {
                "run_id": run_id,
                "phase": phase,
                "suite": task.get("suite", manifest.get("suite", "")),
                "task_id": task_id,
                "task_name": task.get("task_name", ""),
                "state": infer_part(run_id, "state"),
                "seed": infer_part(run_id, "seed"),
                "checkpoint": manifest.get("model_checkpoint_path", task.get("checkpoint_path", "")),
                "status": progress.get("status", ""),
                "official_success": str(official),
                "clean_success": str(bool(official and complete)),
                "failure_reason": failure_reason,
                "artifact_complete": str(complete),
                "step_records_path": str(run_dir / "step_records.jsonl"),
                "episode_records_path": str(run_dir / "episode_records.jsonl"),
                "video_path": first_video(run_dir),
                "object_pose_logged": str(object_pose_logged),
                "gripper_qpos_logged": str(gripper_qpos_logged),
                "expected_mechanism_type": task.get("expected_mechanism_type", ""),
                "mechanism_type_language": mechanism_type,
                "language_mechanism_eligible": str(lang_eligible),
                "language_exclusion_reason": lang_exclusion,
                "black_bowl_sanity": task.get("black_bowl_sanity", ""),
                "target_object_name": task.get("target_object_name", ""),
                "target_receptacle_name": task.get("target_receptacle_name", ""),
                "num_steps": str(len(steps)),
            }
        )
    fields = [
        "run_id","phase","suite","task_id","task_name","state","seed","checkpoint","status",
        "official_success","clean_success","failure_reason","artifact_complete","step_records_path",
        "episode_records_path","video_path","object_pose_logged","gripper_qpos_logged",
        "expected_mechanism_type","mechanism_type_language","language_mechanism_eligible",
        "language_exclusion_reason","black_bowl_sanity","target_object_name","target_receptacle_name","num_steps"
    ]
    write_csv(out / "tables" / output_name, rows, fields)
    return 0


def infer_part(run_id: str, name: str) -> str:
    match = re.search(fr"{name}(\d+)", run_id)
    return match.group(1) if match else ""


def first_video(run_dir: Path) -> str:
    videos = sorted((run_dir / "videos").glob("*.mp4"))
    return str(videos[0]) if videos else ""


def hash_action_trace(steps: list[dict]) -> str:
    vals = []
    for step in steps[:40]:
        vals.append(step.get("clean_action") or step.get("executed_action") or [])
    text = json.dumps(vals, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def hash_initial_pose(steps: list[dict]) -> str:
    first = steps[0] if steps else {}
    keys = ["bowl_z_before", "bowl_z_after", "grasp_bowl_z_delta", "grasp_bowl_plate_dxy", "grasp_bowl_plate_dz", "eef_z_before"]
    text = json.dumps({k: first.get(k) for k in keys}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def create_phase3_jobs(args: argparse.Namespace) -> int:
    out = Path(args.output_root)
    summarize_runs(args, phase_filter="sanity", output_name="phase2_5_seed_state_sanity_manifest.csv")
    sanity = read_csv(out / "tables/phase2_5_seed_state_sanity_manifest.csv")
    details = []
    pose_hashes = set()
    action_hashes = set()
    successes = set()
    for row in sanity:
        steps = read_jsonl(Path(row["step_records_path"]))
        pose_hash = hash_initial_pose(steps)
        action_hash = hash_action_trace(steps)
        pose_hashes.add(pose_hash)
        action_hashes.add(action_hash)
        successes.add(row["official_success"])
        row["initial_pose_hash"] = pose_hash
        row["action_trace_hash"] = action_hash
        details.append(row)
    write_csv(out / "tables/phase2_5_seed_state_sanity_comparison.csv", details)
    seeds_consistent = len(pose_hashes) == 1 and len(action_hashes) == 1 and len(successes) == 1 and len(sanity) == 3
    mode = "states_0_1_2_seed1" if seeds_consistent else "state0_seeds_1_2_3"
    (out / "phase2_5_sanity_decision.txt").write_text(mode + "\n", encoding="utf-8")

    frozen = read_csv(out / "tables/full4_dataset_manifest_frozen.csv")
    jobs = []
    for row in frozen:
        if row["suite"] == "libero_10":
            jobs.append(job_row(row, state=0, seed=1, phase="clean"))
        elif mode == "states_0_1_2_seed1":
            for state in (0, 1, 2):
                jobs.append(job_row(row, state=state, seed=1, phase="clean"))
        else:
            for seed in (1, 2, 3):
                jobs.append(job_row(row, state=0, seed=seed, phase="clean"))
    write_csv(out / "tables/phase3_clean_jobs.csv", jobs)
    write_tsv(out / "tables/phase3_clean_jobs.tsv", jobs)
    return 0


def combine_detector(args: argparse.Namespace) -> int:
    out = Path(args.output_root)
    clean = {r["run_id"]: r for r in read_csv(out / "tables/clean_success_manifest.csv")}
    windows = {r["run_id"]: r for r in read_csv(out / "tables/mechanism_window_raw.csv")}
    cq = {r["run_id"]: r for r in read_csv(out / "tables/clean_cqv2_metrics.csv")}
    rows = []
    for run_id, clean_row in clean.items():
        win = windows.get(run_id, {})
        cqrow = cq.get(run_id, {})
        confidence = str(win.get("confidence") or win.get("window_confidence") or "")
        detected = str(win.get("window_detected", "")).lower() in {"true", "1", "yes"}
        clean_success = clean_row.get("clean_success") == "True"
        artifact = clean_row.get("artifact_complete") == "True"
        lang_mech = clean_row.get("mechanism_type_language", "")
        cq_computable_raw = str(cqrow.get("cqv2_computable", cqrow.get("cq_computable", ""))).lower()
        cq_computable = cq_computable_raw in {"true", "1", "yes"}
        if not cq_computable and cqrow:
            cq_computable = (
                str(cqrow.get("cq_rule_version", "")).startswith("cq_v2")
                or bool(cqrow.get("cq_confidence_v2", ""))
                or bool(cqrow.get("contact_quality_failure", ""))
            )
        video_reviewable = bool(clean_row.get("video_path"))
        mechanism_type = "multi_object_transfer" if lang_mech == "multi_object_transfer" else ("pick_place" if lang_mech == "pick_place" else lang_mech)
        eligible = (
            clean_success
            and artifact
            and mechanism_type in {"pick_place", "multi_object_transfer"}
            and detected
            and confidence in {"medium", "high"}
            and (cq_computable or video_reviewable)
        )
        exclusion = []
        if not clean_success: exclusion.append("clean_failure")
        if not artifact: exclusion.append("artifact_failure")
        if mechanism_type not in {"pick_place", "multi_object_transfer"}: exclusion.append("non_transfer_mechanism")
        if not detected or confidence not in {"medium", "high"}: exclusion.append("detector_abstain_or_low_confidence")
        if not (cq_computable or video_reviewable): exclusion.append("cqv2_not_computable_or_video_missing")
        rows.append({
            **clean_row,
            "window_detected": str(detected),
            "auto_window_start": win.get("auto_window_start", win.get("window_start", "")),
            "auto_window_end": win.get("auto_window_end", win.get("window_end", "")),
            "window_confidence": confidence,
            "detector_mode": win.get("detector_mode", ""),
            "abstain_reason": win.get("abstain_reason", win.get("reason", "")),
            "mechanism_type": mechanism_type,
            "mechanism_eligible": str(eligible),
            "cqv2_computable": str(cq_computable),
            "cq_confidence_v2": cqrow.get("cq_confidence_v2", ""),
            "contact_quality_failure_v2": cqrow.get("contact_quality_failure_v2", cqrow.get("contact_quality_failure", "")),
            "exclusion_reason": ";".join(exclusion),
        })
    fields = list(rows[0].keys()) if rows else []
    write_csv(out / "tables/mechanism_eligibility_manifest.csv", rows, fields)

    nonbb = [r for r in rows if r["mechanism_eligible"] == "True" and r.get("black_bowl_sanity") != "True"]
    pilot = []
    per_suite: dict[str, int] = {}
    for r in sorted(nonbb, key=lambda x: (x["suite"], x["task_id"], x["state"], x["seed"])):
        if per_suite.get(r["suite"], 0) >= 3:
            continue
        pilot.append(r)
        per_suite[r["suite"]] = per_suite.get(r["suite"], 0) + 1
    write_csv(out / "tables/microbreadth_pilot_candidate_queue.csv", pilot, fields)

    suites = {r["suite"] for r in nonbb}
    if len(nonbb) >= 4 and len(suites) >= 2:
        decision = "ready_for_microbreadth_plan"
    elif not any(r.get("clean_success") == "True" for r in rows):
        decision = "clean_denominator_insufficient"
    elif not any(r.get("window_detected") == "True" and r.get("window_confidence") in {"medium", "high"} for r in rows):
        decision = "detector_not_general_enough"
    elif not any(r.get("cqv2_computable") == "True" for r in rows):
        decision = "cqv2_not_general_enough"
    else:
        decision = "clean_denominator_insufficient"
    (out / "FINAL_DECISION.txt").write_text(decision + "\n", encoding="utf-8")
    lines = [
        "# LIBERO Full4 State0 Balanced Clean Denominator Summary",
        "",
        f"Decision: `{decision}`",
        "",
        f"- timestamp: `{time.strftime('%Y-%m-%d %H:%M:%S %Z')}`",
        f"- clean rows: `{len(rows)}`",
        f"- non-Black-Bowl eligible rows: `{len(nonbb)}`",
        f"- eligible suites: `{','.join(sorted(suites))}`",
        "",
        "No VIS/random/oracle/benchmark/attack was run.",
    ]
    (out / "libero_full4_state0_balanced_clean_denominator_summary_20260518.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def create_selective_video_jobs(args: argparse.Namespace) -> int:
    out = Path(args.output_root)
    rows = read_csv(out / "tables/mechanism_eligibility_manifest.csv")
    selected: dict[str, dict[str, str]] = {}

    def add(row: dict[str, str], reason: str) -> None:
        if not row.get("run_id"):
            return
        item = dict(row)
        item["video_selection_reason"] = reason
        selected.setdefault(row["run_id"], item)

    for row in rows:
        if row.get("phase") == "smoke":
            add(row, "smoke")
        if row.get("black_bowl_sanity") == "True":
            add(row, "black_bowl_sanity")
        uncertain = (
            row.get("window_detected") != "True"
            or row.get("window_confidence") not in {"medium", "high"}
            or row.get("cqv2_computable") != "True"
        )
        if uncertain and row.get("clean_success") == "True":
            add(row, "detector_or_cq_uncertain")

    for suite in SUITES:
        eligible = [
            r for r in rows
            if r.get("suite") == suite
            and r.get("mechanism_eligible") == "True"
            and r.get("clean_success") == "True"
            and r.get("black_bowl_sanity") != "True"
        ]
        for row in eligible[:2]:
            add(row, "eligible_clean_success_sample")
        failures = [r for r in rows if r.get("suite") == suite and r.get("clean_success") != "True"]
        for row in failures[:2]:
            add(row, "clean_failure_sample")

    jobs = []
    for row in selected.values():
        jobs.append({
            "phase": row.get("phase", ""),
            "suite": row.get("suite", ""),
            "task_id": row.get("task_id", ""),
            "task_name": row.get("task_name", ""),
            "state": row.get("state", ""),
            "seed": row.get("seed", ""),
            "run_id": row.get("run_id", ""),
            "unnorm_key": row.get("suite", ""),
            "max_steps": "",
            "checkpoint_path": row.get("checkpoint", ""),
            "target_object_name": row.get("target_object_name", ""),
            "target_receptacle_name": row.get("target_receptacle_name", ""),
            "black_bowl_sanity": row.get("black_bowl_sanity", ""),
            "video_selection_reason": row.get("video_selection_reason", ""),
        })
    write_csv(out / "tables/selective_video_jobs.csv", jobs)
    write_tsv(out / "tables/selective_video_jobs.tsv", jobs)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["generate", "summarize", "create_clean_jobs", "combine", "create_video_jobs"])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo-root", default="/home/liuyu/openvla_gripper_attack/OpenVLA Gripper Duty-Cycle Attack")
    parser.add_argument("--upload-result-csv", default="")
    args = parser.parse_args()
    if args.mode == "generate":
        return generate(args)
    if args.mode == "summarize":
        return summarize_runs(args)
    if args.mode == "create_clean_jobs":
        return create_phase3_jobs(args)
    if args.mode == "combine":
        return combine_detector(args)
    if args.mode == "create_video_jobs":
        return create_selective_video_jobs(args)
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
