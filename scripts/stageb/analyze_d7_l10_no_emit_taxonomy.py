#!/usr/bin/env python3
"""D8B — L10 no-emission taxonomy analysis (CPU-only).

Classifies every completed L10 episode by WHY it did or didn't emit.
Inputs (post D7B2 completion):
  - D7C audit CSV
  - episode_summary.json per episode
  - step_telemetry.csv per episode
  - manifest

Output categories:
  NO_EMIT_LOW_EMIT_P          — emit_p never reached threshold
  NO_EMIT_HIGH_SUPPRESS_P     — suppress_p blocked emission
  NO_EMIT_SHORT_EPISODE       — episode too short for W-window
  NO_EMIT_INVALID_FEATURES    — streaming features had invalid/NaN steps
  NO_EMIT_CLEAN_UNSTABLE      — episode terminated before stable phase
  NO_EMIT_TRIGGER_TOO_LATE    — trigger after task already succeeded
  EMIT_BUT_NO_ATTACK_FRAMES   — emitted but attack didn't apply (no_trigger taxonomy)
  EMIT_AND_ATTACKED           — emitted and attack applied
  ORACLE_SENSITIVE            — COMMAND_OPEN_ORACLE caused task failure
  ORACLE_NOT_SENSITIVE        — COMMAND_OPEN_ORACLE didn't affect task
"""

from __future__ import annotations

import argparse, csv, json, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def safe_float(val: Any, default: float = float("nan")) -> float:
    try: return float(val)
    except (ValueError, TypeError): return default


def classify_episode(
    summary: Dict[str, Any],
    telemetry_path: Optional[Path] = None,
) -> str:
    """Classify a single L10 episode into taxonomy bucket."""
    emitted = str(summary.get("detector_emitted", "")).lower() in ("true", "1")
    attack_frames = int(summary.get("attack_frames", 0) or 0)
    n_steps = int(summary.get("n_steps", 0) or 0)
    condition = str(summary.get("condition", ""))
    task_success = str(summary.get("task_success", "")).lower() in ("true", "1")
    error = str(summary.get("error", ""))

    if error:
        return "RUNTIME_ERROR"

    # ── Oracle sensitivity ──
    if condition == "COMMAND_OPEN_ORACLE":
        if not task_success and attack_frames >= 10:
            return "ORACLE_SENSITIVE"
        else:
            return "ORACLE_NOT_SENSITIVE"

    # ── Emission happened ──
    if emitted:
        if attack_frames >= 10:
            return "EMIT_AND_ATTACKED"
        else:
            return "EMIT_BUT_NO_ATTACK_FRAMES"

    # ── No emission — diagnose why ──
    if n_steps < 16:
        return "NO_EMIT_SHORT_EPISODE"

    # Check telemetry for feature validity
    if telemetry_path is not None and telemetry_path.exists():
        try:
            trows = list(csv.DictReader(open(telemetry_path)))
            has_valid = any(
                str(r.get("valid", "")).lower() in ("true", "1")
                for r in trows
            )
            if not has_valid:
                return "NO_EMIT_INVALID_FEATURES"
        except Exception:
            pass

    # Default: low signal (emit_p never hit threshold, or suppress_p blocked)
    return "NO_EMIT_LOW_SIGNAL"


def main():
    ap = argparse.ArgumentParser(description="D8B L10 no-emission taxonomy")
    ap.add_argument("--audit-csv", required=True, help="D7C postrun audit CSV")
    ap.add_argument("--episode-root", required=True, help="D7 rollout root dir")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--suite", default="libero_10", help="Suite to analyze")
    ap.add_argument("--git-commit", required=True)
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    audit_rows = read_csv(args.audit_csv)
    episode_root = Path(args.episode_root)

    # Filter to target suite, completed episodes
    l10_rows = [
        r for r in audit_rows
        if r.get("suite") == args.suite and r.get("completed", "").lower() == "true"
    ]
    print(f"D8B: {len(l10_rows)} completed {args.suite} episodes")

    taxonomy: Dict[str, int] = defaultdict(int)
    rows: List[Dict[str, Any]] = []

    for ar in l10_rows:
        ep_dir = episode_root / ar["suite"] / ar["condition"] / ar["parent_key"]
        summary = read_json(ep_dir / "episode_summary.json")
        telemetry = ep_dir / "step_telemetry.csv"

        bucket = classify_episode(summary, telemetry)
        taxonomy[bucket] += 1

        rows.append({
            "suite": ar["suite"],
            "condition": ar.get("condition", ""),
            "parent_key": ar.get("parent_key", ""),
            "task_success": summary.get("task_success", ""),
            "detector_emitted": summary.get("detector_emitted", ""),
            "attack_frames": summary.get("attack_frames", ""),
            "n_steps": summary.get("n_steps", ""),
            "taxonomy": bucket,
        })

    # Write results
    fields = ["suite", "condition", "parent_key", "task_success",
              "detector_emitted", "attack_frames", "n_steps", "taxonomy"]
    with open(out / "d8b_l10_taxonomy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Summary
    summary = {
        "gate": "D8B_L10_NO_EMIT_TAXONOMY",
        "suite": args.suite,
        "total_episodes": len(l10_rows),
        "taxonomy_counts": dict(taxonomy),
        "created_at_unix": time.time(),
        "runtime_seconds": time.time() - t0,
        "git_commit": args.git_commit,
    }
    (out / "d8b_l10_taxonomy_report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str)
    )

    print(f"D8B taxonomy:")
    for bucket, count in sorted(taxonomy.items(), key=lambda x: -x[1]):
        print(f"  {bucket}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
