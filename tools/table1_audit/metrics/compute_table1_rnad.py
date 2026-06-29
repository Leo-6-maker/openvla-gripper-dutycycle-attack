from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


ARM = range(6)
GRIP = 6


def vec(value: str) -> list[float]:
    data = json.loads(value)
    if not isinstance(data, list) or len(data) != 7:
        raise ValueError("action vector must have length 7")
    return [float(x) for x in data]


def mean_abs(a: list[float], b: list[float], idxs) -> float:
    vals = [abs(a[i] - b[i]) for i in idxs]
    return sum(vals) / len(vals)


def compute(path: Path) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            cp, p = vec(row["clean_policy_action_7d"]), vec(row["policy_action_7d"])
            ce, e = vec(row["clean_env_action_7d"]), vec(row["env_action_7d"])
            rec = {
                "attack_active": str(row.get("attack_active", "")).lower() == "true",
                "policy_arm": mean_abs(cp, p, ARM),
                "execution_arm": mean_abs(ce, e, ARM),
                "policy_gripper": mean_abs(cp, p, [GRIP]),
                "execution_gripper": mean_abs(ce, e, [GRIP]),
            }
            groups[row["condition_id"]].append(rec)
    return {"schema_version": "table1_rnad_result.v1", "conditions": {k: summarize(v) for k, v in sorted(groups.items())}}


def avg(rows: list[dict], key: str) -> float:
    return sum(float(r[key]) for r in rows) / len(rows) if rows else 0.0


def summarize(rows: list[dict]) -> dict:
    active = [r for r in rows if r["attack_active"]]
    return {
        "itt_count": len(rows),
        "attack_active_count": len(active),
        "policy_arm_itt": avg(rows, "policy_arm"),
        "execution_arm_itt": avg(rows, "execution_arm"),
        "policy_gripper_itt": avg(rows, "policy_gripper"),
        "execution_gripper_itt": avg(rows, "execution_gripper"),
        "policy_arm_attack_active": avg(active, "policy_arm"),
        "execution_arm_attack_active": avg(active, "execution_arm"),
        "policy_gripper_attack_active": avg(active, "policy_gripper"),
        "execution_gripper_attack_active": avg(active, "execution_gripper"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--telemetry-csv", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()
    args.output_json.write_text(json.dumps(compute(args.telemetry_csv), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
