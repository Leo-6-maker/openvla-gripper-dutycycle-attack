"""D8-1 Formal G-sensitivity runner.

Loads relation sidecar with correct episode_id mapping (from JSON entry,
NOT from filename), runs formal consolidation for G in {0,1,2,3,5},
and produces per-G statistics with reject taxonomy.

Fixes the episode ID loader bug in /tmp/gsens.py:
  - episode_id read from JSON entry/manifest, not filename
  - fail-closed on duplicate/missing/extra episodes
  - formal mode: relations REQUIRED
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts" / "detector_v5"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from d8_event_consolidator import (
    consolidate_physical_events,
    compute_consolidation_digest,
)
from gripper_attack.seal_utils import rename_noreplace
from audit_r3_contact_input import sha256_file, verify_seal

G_VALUES = [0, 1, 2, 3, 5]
HEAD = "physical_criticality"


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _write_seal(p: Path) -> str:
    files = sorted(
        x for x in p.rglob("*")
        if x.is_file() and x.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}
    )
    (p / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(x)}  {x.relative_to(p).as_posix()}\n" for x in files),
        encoding="utf-8",
    )
    d = sha256_file(p / "SHA256SUMS")
    (p / "SHA256SUMS.sha256").write_text(f"{d}  SHA256SUMS\n", encoding="utf-8")
    return d


def load_sidecar_correct(sidecar_root: Path) -> Dict[str, Dict[int, dict]]:
    """Load sidecar with episode_id from JSON entry (NOT filename).

    Returns: episode_id -> {step: sidecar_entry}

    P0-5: Strict validation:
      - Non-integer step keys raise ValueError
      - Numeric key collisions ("1"/"01") raise ValueError
      - entry.step must equal JSON key
      - All entries must have non-empty episode_id
    """
    ep_dir = sidecar_root / "per_episode"
    if not ep_dir.is_dir():
        raise ValueError(f"per_episode directory not found in {sidecar_root}")

    sidecar: Dict[str, Dict[int, dict]] = {}
    eps_seen: Dict[str, str] = {}  # eid -> first filename

    for fname in sorted(ep_dir.iterdir()):
        if not fname.suffix == ".json":
            continue
        with open(fname, encoding="utf-8") as fh:
            ep_data = json.load(fh)

        eid = None
        ep_steps: Dict[int, dict] = {}
        seen_numeric: Dict[int, str] = {}  # detect "1"/"01" collisions

        for step_key, entry in ep_data.items():
            if not isinstance(entry, dict):
                raise ValueError(
                    f"non-dict entry in {fname.name}: key={step_key} type={type(entry).__name__}"
                )
            # P0-5: Non-integer step key raises (not silently continues)
            try:
                step = int(step_key)
            except (ValueError, TypeError):
                raise ValueError(
                    f"non-integer step key '{step_key}' in {fname.name}"
                ) from None
            # P0-5: String "1" vs "01" collision
            if step_key != str(step):
                raise ValueError(
                    f"step key collision: '{step_key}' normalizes to {step} "
                    f"(possible duplicate with '{step}') in {fname.name}"
                )
            # P0-5: entry.step must match JSON key
            entry_step = entry.get("step")
            if entry_step is not None and int(entry_step) != step:
                raise ValueError(
                    f"entry.step={entry_step} != JSON key={step} in {fname.name}"
                )
            # P0-5: Detect numeric collision
            if step in seen_numeric:
                raise ValueError(
                    f"duplicate step {step} in {fname.name}: "
                    f"keys '{seen_numeric[step]}' and '{step_key}'"
                )
            seen_numeric[step] = step_key

            if eid is None:
                eid = entry.get("episode_id", "")
                if not eid:
                    raise ValueError(
                        f"empty episode_id in entry step={step} in file {fname.name}"
                    )
                _validate_episode_id_format(eid, fname.name)
            # P0-5: All entries must have episode_id
            entry_eid = entry.get("episode_id", "")
            if not entry_eid:
                raise ValueError(
                    f"empty episode_id at step {step} in {fname.name}"
                )
            if entry_eid != eid:
                raise ValueError(
                    f"episode_id mismatch in {fname.name}: "
                    f"step {step} has '{entry_eid}' but expected '{eid}'"
                )
            ep_steps[step] = entry

        if eid is None:
            raise ValueError(f"no valid entries found in {fname.name}")

        # Fail-closed: duplicate episode_id
        if eid in eps_seen:
            raise ValueError(
                f"duplicate episode_id '{eid}' in files "
                f"{eps_seen[eid]} and {fname.name}"
            )
        eps_seen[eid] = fname.name

        sidecar[eid] = ep_steps

    return sidecar


def _validate_episode_id_format(eid: str, source: str):
    """Validate episode_id has the required suite/task/state format."""
    parts = eid.split("/")
    if len(parts) != 3:
        raise ValueError(
            f"malformed episode_id '{eid}' in {source}: "
            f"expected exactly suite/task_XX/state_YY, got {len(parts)} components"
        )
    suite = parts[0]
    if suite not in ("libero_10", "libero_goal", "libero_object", "libero_spatial"):
        raise ValueError(f"unknown suite '{suite}' in episode_id '{eid}' in {source}")
    if not parts[1].startswith("task_"):
        raise ValueError(f"expected task_XX in episode_id '{eid}' in {source}")
    if not parts[2].startswith("state_"):
        raise ValueError(f"expected state_YY in episode_id '{eid}' in {source}")


def load_teacher_labels(teacher_root: Path) -> Tuple[Dict[str, Dict[int, dict]], int, int]:
    """Load teacher labels. P0-5: detect duplicate (eid, step) pairs.

    Returns: (ep_labels, total_steps, total_identities)
    """
    records_path = teacher_root / "teacher_records.jsonl"
    if not records_path.is_file():
        raise ValueError(f"teacher records not found: {records_path}")

    ep_labels: Dict[str, Dict[int, dict]] = defaultdict(dict)
    identities: set = set()
    total = 0
    seen_pairs: set = set()

    with open(records_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            eid = str(row["episode_id"])
            identities.add(eid)
            step = row["step"]
            total += 1
            # P0-5: Detect duplicate (episode_id, step)
            pair = (eid, step)
            if pair in seen_pairs:
                raise ValueError(f"duplicate Teacher record: {eid} step={step}")
            seen_pairs.add(pair)
            pc = row.get("labels", {}).get("physical_criticality", {})
            if isinstance(pc, dict):
                ep_labels[eid][step] = dict(pc)

    return dict(ep_labels), total, len(identities)


def run_formal_g(
    sidecar: Dict[str, Dict[int, dict]],
    ep_labels: Dict[str, Dict[int, dict]],
    G: int,
) -> Dict[str, Any]:
    """Run formal consolidation for a single G value.

    Returns detailed statistics including reject taxonomy.
    """
    # Fail-closed: identity set equality
    sidecar_ids = set(sidecar.keys())
    teacher_ids = set(ep_labels.keys())
    missing = teacher_ids - sidecar_ids
    extra = sidecar_ids - teacher_ids
    if missing:
        raise ValueError(f"G={G}: {len(missing)} episodes in teacher but not sidecar")
    if extra:
        raise ValueError(f"G={G}: {len(extra)} episodes in sidecar but not teacher")

    reject_taxonomy: Dict[str, int] = defaultdict(int)
    global_stats = {
        "raw_true_spans": 0, "consolidated_events": 0,
        "bridged_gaps": 0, "rejected_gaps": 0,
        "multi_fragment_events": 0, "raw_true_steps": 0,
        "unknown_gap_steps": 0, "applicable_identities": 0,
    }
    suite_stats: Dict[str, dict] = defaultdict(lambda: {
        "raw_spans": 0, "cons_events": 0, "bridged": 0, "rejected": 0,
        "gap_lengths": [], "tasks": set(),
    })
    per_episode: Dict[str, dict] = {}
    invariant_violations: List[str] = []

    for eid in sorted(teacher_ids):
        labels = ep_labels.get(eid, {})
        relations = sidecar.get(eid, {})

        # P0-5: Per-episode step set closure
        sc_steps = set(relations.keys())
        t_steps = set(labels.keys())
        if sc_steps != t_steps:
            only_sc = sc_steps - t_steps
            only_t = t_steps - sc_steps
            details = []
            if only_sc: details.append(f"sidecar-only steps: {sorted(only_sc)[:5]}...")
            if only_t: details.append(f"teacher-only steps: {sorted(only_t)[:5]}...")
            raise ValueError(
                f"per-episode step set mismatch in {eid}: {', '.join(details)}"
            )

        result = consolidate_physical_events(eid, labels, relations=relations, G=G)

        is_art = result.get("articulated", False)
        is_applicable = result.get("applicable", True)

        suite = eid.split("/")[0] if "/" in eid else "?"
        task = "/".join(eid.split("/")[:2]) if "/" in eid else eid

        # P0-6: Include ALL 670 episodes in per_episode for A/B comparison
        ep_detail = {
            "suite": suite,
            "task": task,
            "articulated": is_art,
            "applicable": is_applicable,
            "raw_spans": result.get("raw_true_span_count", 0),
            "cons_events": result.get("consolidated_event_count", 0),
            "bridged": result.get("total_bridged_gaps", 0),
            "rejected": result.get("total_rejected_gaps", 0),
            "digest": compute_consolidation_digest(result),
        }
        per_episode[eid] = ep_detail

        if not is_applicable or is_art:
            continue

        global_stats["applicable_identities"] += 1
        global_stats["raw_true_spans"] += result.get("raw_true_span_count", 0)
        global_stats["consolidated_events"] += result.get("consolidated_event_count", 0)
        global_stats["bridged_gaps"] += result.get("total_bridged_gaps", 0)
        global_stats["rejected_gaps"] += result.get("total_rejected_gaps", 0)

        ss = suite_stats[suite]
        ss["raw_spans"] += result.get("raw_true_span_count", 0)
        ss["cons_events"] += result.get("consolidated_event_count", 0)
        ss["bridged"] += result.get("total_bridged_gaps", 0)
        ss["rejected"] += result.get("total_rejected_gaps", 0)
        ss["tasks"].add(task)

        # Collect reject taxonomy
        for group in result.get("event_groups", []):
            for gap in group.get("rejected_gaps", []):
                reason = gap.get("reject_reason", "OTHER")
                reject_taxonomy[reason] += 1

            if group.get("fragment_count", 1) > 1:
                global_stats["multi_fragment_events"] += 1

            global_stats["raw_true_steps"] += group.get("raw_true_step_count", 0)
            global_stats["unknown_gap_steps"] += group.get("unknown_gap_step_count", 0)

            # P0-6: Invariant checks from step_evidence, not reason strings
            for gap in group.get("bridged_gaps", []):
                for ev in gap.get("step_evidence", []):
                    if ev.get("label_reason") == "GEOMETRY_NOT_APPLICABLE":
                        invariant_violations.append(
                            f"{eid} step={ev['step']}: GEOM_NA in bridged gap"
                        )
                    if ev.get("label_value") == "FALSE" and ev.get("mask"):
                        invariant_violations.append(
                            f"{eid} step={ev['step']}: known-FALSE in bridged gap"
                        )
                    if ev.get("right_censored"):
                        invariant_violations.append(
                            f"{eid} step={ev['step']}: RIGHT_CENSORED in bridged gap"
                        )

    # Compile suite stats
    suite_summary = {}
    for suite, ss in suite_stats.items():
        n_tasks = len(ss["tasks"])
        suite_summary[suite] = {
            "tasks": n_tasks,
            "raw_spans": ss["raw_spans"],
            "cons_events": ss["cons_events"],
            "bridged": ss["bridged"],
            "rejected": ss["rejected"],
            "consolidation_ratio": (
                ss["cons_events"] / max(ss["raw_spans"], 1) * 100
            ),
        }

    raw = max(global_stats["raw_true_spans"], 1)
    return {
        "G": G,
        "global": {
            **global_stats,
            "bridge_ratio": global_stats["bridged_gaps"] / max(
                global_stats["bridged_gaps"] + global_stats["rejected_gaps"], 1
            ),
            "consolidation_ratio": global_stats["consolidated_events"] / raw * 100,
        },
        "by_suite": suite_summary,
        "reject_taxonomy": dict(reject_taxonomy),
        "per_episode": per_episode,
        "invariant_violations": invariant_violations,
        "invariants_pass": len(invariant_violations) == 0,
        "episodes_processed": global_stats["applicable_identities"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="D8-1 Formal G-sensitivity")
    parser.add_argument("--sidecar-root", type=Path, required=True,
                        help="Relation sidecar root (Run A or B)")
    parser.add_argument("--teacher-root", type=Path, required=True,
                        help="Teacher records root")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-label", type=str, default="A",
                        help="Sidecar run label (A or B)")
    args = parser.parse_args()

    if subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip():
        return "ERROR: clean checkout required"

    commit = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")

    # Verify inputs
    sidecar_root = args.sidecar_root.resolve(strict=True)
    teacher_root = args.teacher_root.resolve(strict=True)

    sidecar_seal = verify_seal(sidecar_root)
    teacher_seal = verify_seal(teacher_root)

    print(f"Loading sidecar from {sidecar_root}")
    sidecar = load_sidecar_correct(sidecar_root)
    print(f"  Loaded {len(sidecar)} episodes")

    print(f"Loading teacher labels from {teacher_root}")
    ep_labels, teacher_steps, teacher_ids = load_teacher_labels(teacher_root)
    print(f"  Loaded {teacher_ids} identities, {teacher_steps} steps")

    # Identity closure check
    sc_ids = set(sidecar.keys())
    t_ids = set(ep_labels.keys())
    if sc_ids != t_ids:
        print(f"ERROR: identity mismatch: sidecar={len(sc_ids)} teacher={len(t_ids)}")
        print(f"  Missing from sidecar: {sorted(t_ids - sc_ids)[:10]}...")
        print(f"  Extra in sidecar: {sorted(sc_ids - t_ids)[:10]}...")
        return 1

    # Step closure check
    sc_steps = sum(len(v) for v in sidecar.values())
    if sc_steps != teacher_steps:
        print(f"ERROR: step mismatch: sidecar={sc_steps} teacher={teacher_steps}")
        return 1

    print(f"Identity closure: {len(sc_ids)} == {len(t_ids)} PASS")
    print(f"Step closure: {sc_steps} == {teacher_steps} PASS")

    # Run formal G-sensitivity
    if args.output_root.exists():
        return f"ERROR: output root already exists: {args.output_root}"

    staging = args.output_root.with_name(
        f".{args.output_root.name}.staging.{os.getpid()}"
    )
    staging.mkdir(parents=True)

    all_results = {}
    for G in G_VALUES:
        print(f"\nRunning G={G} (formal mode)...")
        result = run_formal_g(sidecar, ep_labels, G)
        all_results[str(G)] = result

        g = result["global"]
        print(f"  raw_spans={g['raw_true_spans']} cons_events={g['consolidated_events']} "
              f"bridged={g['bridged_gaps']} rejected={g['rejected_gaps']} "
              f"ratio={g['consolidation_ratio']:.1f}%")
        if result["invariant_violations"]:
            print(f"  INVARIANT VIOLATIONS: {len(result['invariant_violations'])}")
            for v in result["invariant_violations"][:5]:
                print(f"    {v}")

    # Write per-G results
    for G in G_VALUES:
        gdir = staging / f"G{G}"
        gdir.mkdir()
        (gdir / "result.json").write_text(
            json.dumps(all_results[str(G)], indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    # Write global manifest
    builder_sha = sha256_file(Path(__file__))
    payload = {
        "schema": "DETECTOR_V3_D8_FORMAL_G_SENSITIVITY_V1",
        "status": "PASS_COMPLETED",
        "run_label": args.run_label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_snapshot": {"commit": commit, "tree": tree},
        "sidecar_root": str(sidecar_root),
        "sidecar_seal": sidecar_seal["sha256sums_sha256"],
        "teacher_root": str(teacher_root),
        "teacher_seal": teacher_seal["sha256sums_sha256"],
        "builder_sha256": builder_sha,
        "d8_consolidator_sha256": sha256_file(
            ROOT / "scripts" / "detector_v5" / "d8_event_consolidator.py"
        ),
        "d8_protocol_sha256": sha256_file(
            ROOT / "configs" / "DETECTOR_V3_D8_EVENT_CONSOLIDATION_PROTOCOL.json"
        ),
        "G_values": G_VALUES,
        "summary": {
            str(G): {
                "raw_spans": all_results[str(G)]["global"]["raw_true_spans"],
                "cons_events": all_results[str(G)]["global"]["consolidated_events"],
                "bridged": all_results[str(G)]["global"]["bridged_gaps"],
                "rejected": all_results[str(G)]["global"]["rejected_gaps"],
                "ratio": all_results[str(G)]["global"]["consolidation_ratio"],
                "invariants_pass": all_results[str(G)]["invariants_pass"],
            }
            for G in G_VALUES
        },
        "protected_reads": 0,
        "test_payload_read": 0,
    }

    (staging / "MANIFEST.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    digest = _write_seal(staging)
    rename_noreplace(staging, args.output_root)
    payload["sha256sums_sha256"] = digest
    print(f"\nSealed: {digest}")
    print(f"Output: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
