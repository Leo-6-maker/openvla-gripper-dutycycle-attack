# Adaptive window triage for Tier1 negative states. Used to test whether failures were window-specific rather than generic.
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STATE_GROUPS = {
    "S1": [0, 1, 2, 3, 4, 6],
    "S2": [9, 10, 11, 12, 13, 14],
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def as_float(value, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def as_int(value, default: int | None = None) -> int | None:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def step_idx(row: dict) -> int:
    return int(as_int(row.get("step_idx"), 0) or 0)


def object_z_delta(row: dict) -> float:
    vals = [
        as_float(row.get("grasp_bowl_z_delta")),
        as_float(row.get("bowl_z_delta_before")),
        as_float(row.get("grasp_object_z_delta")),
        as_float(row.get("object_z_delta_before")),
    ]
    for value in vals:
        if value is not None:
            return float(value)
    return 0.0


def eef_z_delta(row: dict) -> float:
    vals = [
        as_float(row.get("proxy_lift_carry_eef_z_delta_from_min")),
        as_float(row.get("eef_z_delta_from_min")),
    ]
    for value in vals:
        if value is not None:
            return float(value)
    return 0.0


def eef_object_dist(row: dict) -> float | None:
    return as_float(row.get("grasp_eef_bowl_dist"), as_float(row.get("grasp_eef_object_dist")))


def gripper_closed(row: dict) -> bool:
    # In this codebase positive env gripper means closed during lift.
    env = as_float(row.get("executed_gripper_env"), as_float(row.get("clean_gripper_env")))
    if env is not None:
        return env > 0.5
    raw = as_float(row.get("executed_gripper_raw"), as_float(row.get("clean_gripper_raw")))
    return bool(raw is not None and raw < 0.2)


def maintained_interval(rows: list[dict], value_fn, threshold: float, min_consecutive: int) -> tuple[int | None, int | None, int]:
    best: tuple[int | None, int | None, int] = (None, None, 0)
    cur_start: int | None = None
    cur_end: int | None = None
    cur_count = 0
    for row in sorted(rows, key=step_idx):
        ok = float(value_fn(row)) >= threshold
        idx = step_idx(row)
        if ok:
            if cur_start is None:
                cur_start = idx
            cur_end = idx
            cur_count += 1
        else:
            if cur_count >= min_consecutive and cur_count > best[2]:
                best = (cur_start, cur_end, cur_count)
            cur_start = None
            cur_end = None
            cur_count = 0
    if cur_count >= min_consecutive and cur_count > best[2]:
        best = (cur_start, cur_end, cur_count)
    return best


def count_hits(rows: list[dict], value_fn, threshold: float) -> int:
    return sum(1 for row in rows if float(value_fn(row)) >= threshold)


def infer_first_grasp_step(rows: list[dict]) -> int | None:
    candidates: list[int] = []
    for row in sorted(rows, key=step_idx):
        idx = step_idx(row)
        for key in ["grasp_first_close_step", "grasp_first_gate_step"]:
            value = as_int(row.get(key))
            if value is not None and value > 0:
                candidates.append(value)
        dist = eef_object_dist(row)
        if gripper_closed(row) and dist is not None and dist <= 0.07:
            candidates.append(idx)
    return min(candidates) if candidates else None


def no_clear_lift(rows: list[dict]) -> bool:
    max_z = max([object_z_delta(row) for row in rows] or [0.0])
    max_eef = max([eef_z_delta(row) for row in rows] or [0.0])
    return max_z < 0.015 and max_eef < 0.02


def triage_episode(rows: list[dict], episode: dict, state_id: int, seed: int) -> dict:
    rows = sorted(rows, key=step_idx)
    clean_success = bool(episode.get("success"))
    failure_phase = str(episode.get("failure_phase", ""))
    max_object = max([object_z_delta(row) for row in rows] or [0.0])
    first_grasp = infer_first_grasp_step(rows)
    first_lift_any = next((step_idx(row) for row in rows if object_z_delta(row) >= 0.02), None)

    z_intervals = {
        "z004": maintained_interval(rows, object_z_delta, 0.04, 3),
        "z003": maintained_interval(rows, object_z_delta, 0.03, 3),
        "z002": maintained_interval(rows, object_z_delta, 0.02, 3),
    }
    eef_interval = maintained_interval(rows, eef_z_delta, 0.02, 5)
    hit_counts = {
        "z004": z_intervals["z004"][2],
        "z003": z_intervals["z003"][2],
        "z002": z_intervals["z002"][2],
        "eefrise004": eef_interval[2],
    }

    recommended_gate = "window_unsuitable"
    gate_first = None
    gate_last = None
    gate_count = 0
    triage_label = "window_unsuitable"

    if not clean_success:
        recommended_gate = "skip_no_grasp"
        triage_label = "no_grasp" if failure_phase == "no_grasp" else "clean_failure"
    elif no_clear_lift(rows):
        recommended_gate = "window_unsuitable"
        triage_label = "clean_no_committed_lift"
    else:
        for gate in ["z004", "z003", "z002"]:
            start, end, count = z_intervals[gate]
            if count >= 3:
                recommended_gate = gate
                gate_first = start
                gate_last = end
                gate_count = count
                triage_label = "valid_adaptive_window"
                break
        if recommended_gate == "window_unsuitable":
            start, end, count = eef_interval
            if count >= 5:
                recommended_gate = "eefrise004"
                gate_first = start
                gate_last = end
                gate_count = count
                triage_label = "valid_adaptive_window"

    if first_lift_any is None and recommended_gate == "eefrise004":
        first_lift_any = gate_first

    burst = min(gate_count, 10) if gate_count else 0
    if triage_label == "valid_adaptive_window" and gate_count < 3:
        recommended_gate = "window_unsuitable"
        triage_label = "window_unsuitable"
        burst = 0

    return {
        "state_id": state_id,
        "seed": seed,
        "clean_success": clean_success,
        "failure_phase": failure_phase,
        "max_object_z_delta": max_object,
        "first_grasp_step": first_grasp,
        "first_lift_step": first_lift_any,
        "z004_hit_count": hit_counts["z004"],
        "z003_hit_count": hit_counts["z003"],
        "z002_hit_count": hit_counts["z002"],
        "eefrise004_hit_count": hit_counts["eefrise004"],
        "recommended_gate": recommended_gate,
        "gate_hit_count": gate_count,
        "gate_first_step": gate_first,
        "gate_last_step": gate_last,
        "burst": burst,
        "triage_label": triage_label,
    }


def group_rows_by_episode(rows: list[dict]) -> dict[int, list[dict]]:
    by_ep: dict[int, list[dict]] = {}
    for row in rows:
        eid = int(as_int(row.get("episode_id"), -1) or -1)
        by_ep.setdefault(eid, []).append(row)
    return by_ep


def parse_sources(values: list[str]) -> list[dict]:
    sources = []
    for value in values:
        parts = value.split("=")
        if len(parts) != 4:
            raise ValueError(f"Bad --source {value!r}; expected tag=run_dir=states_csv=seed")
        tag, run_dir, states_csv, seed = parts
        states = [int(x) for x in states_csv.split(",") if x.strip()]
        sources.append({"tag": tag, "run_dir": Path(run_dir), "states": states, "seed": int(seed)})
    return sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True, help="tag=run_dir=states_csv=seed")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    valid_states: list[dict] = []
    for source in parse_sources(args.source):
        run_dir = source["run_dir"]
        steps = read_jsonl(run_dir / "step_records.jsonl")
        episodes = read_jsonl(run_dir / "episode_records.jsonl")
        by_ep = group_rows_by_episode(steps)
        for idx, episode in enumerate(episodes):
            state_id = source["states"][idx] if idx < len(source["states"]) else idx
            eid = int(as_int(episode.get("episode_id"), idx) or idx)
            row = triage_episode(by_ep.get(eid, []), episode, state_id, source["seed"])
            row["source"] = source["tag"]
            row["clean_run_dir"] = str(run_dir)
            all_rows.append(row)
            if row["triage_label"] == "valid_adaptive_window":
                valid_states.append(
                    {
                        "source": source["tag"],
                        "state_id": state_id,
                        "seed": source["seed"],
                        "recommended_gate": row["recommended_gate"],
                        "gate_hit_count": row["gate_hit_count"],
                        "gate_first_step": row["gate_first_step"],
                        "gate_last_step": row["gate_last_step"],
                        "burst": row["burst"],
                    }
                )

    fields = [
        "source",
        "clean_run_dir",
        "state_id",
        "seed",
        "clean_success",
        "failure_phase",
        "max_object_z_delta",
        "first_grasp_step",
        "first_lift_step",
        "z004_hit_count",
        "z003_hit_count",
        "z002_hit_count",
        "eefrise004_hit_count",
        "recommended_gate",
        "gate_hit_count",
        "gate_first_step",
        "gate_last_step",
        "burst",
        "triage_label",
    ]
    with (output_dir / "adaptive_window_triage.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    (output_dir / "adaptive_window_triage.json").write_text(json.dumps(all_rows, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "adaptive_valid_states.json").write_text(json.dumps(valid_states, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {len(all_rows)} triage rows, {len(valid_states)} valid states -> {output_dir}")


if __name__ == "__main__":
    main()
