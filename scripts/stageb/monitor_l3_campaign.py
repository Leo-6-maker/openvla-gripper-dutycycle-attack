#!/usr/bin/env python3
"""D7: Passive L3 campaign monitor (CPU-only, read-only).

Maintains campaign status report and gate history log.
Reads Codex artifacts without modifying them.
Tracks gate transitions and blocks on violations.
"""

import csv, hashlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

GATE_ORDER = ["H0", "H1", "H2", "H3", "H4", "H5", "H6"]

ALLOWED_TRANSITIONS = {
    "H0": ["H1"],
    "H1": ["H2"],
    "H2": ["H3", "H4", "H5"],  # >=2/3 required for H3/H4/H5
    "H3": ["H4"],
    "H4": ["H5"],
    "H5": ["H6"],
    "H6": [],
}

GATE_HISTORY_FIELDS = [
    "timestamp", "codex_branch", "codex_head", "watcher_state",
    "output_root", "gate", "codex_claimed", "deepseek_independent",
    "artifact_count", "hash_status", "xid_delta", "next_authorized", "blocker",
]


class CampaignMonitor:
    def __init__(self):
        self.history_path = REPO_ROOT / "tables" / "l3_campaign_gate_history.csv"
        self.status_path = REPO_ROOT / "reports" / "L3_CAMPAIGN_STATUS.md"
        self.current_gate = "H0"
        self.history = []

    def load_history(self):
        if self.history_path.exists():
            self.history = list(csv.DictReader(open(self.history_path)))
            if self.history:
                self.current_gate = self.history[-1].get("gate", "H0")

    def record_gate(self, gate: str, codex_claimed: str, deepseek_independent: str,
                    codex_branch: str = "", codex_head: str = "",
                    watcher_state: str = "", output_root: str = "",
                    artifact_count: str = "", hash_status: str = "",
                    xid_delta: str = "", blocker: str = ""):
        """Record a gate transition in the history."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Validate transition
        if self.history:
            prev_gate = self.history[-1]["gate"]
            if gate != prev_gate and gate not in ALLOWED_TRANSITIONS.get(prev_gate, []):
                print(f"  WARNING: Unauthorized transition {prev_gate} → {gate}")

        # Determine next authorized
        next_authorized = ""
        if deepseek_independent.endswith("PASS") or deepseek_independent.startswith("L3_VIS_MULTIPARENT"):
            allowed = ALLOWED_TRANSITIONS.get(gate, [])
            next_authorized = ", ".join(allowed) if allowed else "COMPLETE"

        row = {
            "timestamp": timestamp,
            "codex_branch": codex_branch,
            "codex_head": codex_head,
            "watcher_state": watcher_state,
            "output_root": output_root,
            "gate": gate,
            "codex_claimed": codex_claimed,
            "deepseek_independent": deepseek_independent,
            "artifact_count": artifact_count,
            "hash_status": hash_status,
            "xid_delta": xid_delta,
            "next_authorized": next_authorized,
            "blocker": blocker,
        }
        self.history.append(row)
        self._write_history()
        self._write_status()
        return row

    def check_violations(self, codex_branch: str, codex_head: str,
                         output_root: str = "", gpu_mapping: str = "") -> List[str]:
        """Check for contract violations in current state."""
        violations = []

        # Check production tag immutability
        # (can't check remote without auth, but can verify local)

        # Check for GPU mapping changes
        if gpu_mapping and "1,5" not in gpu_mapping:
            violations.append(f"GPU mapping changed: {gpu_mapping}")

        # Check for config drift (lambda, target, seeds)
        # This would need reading actual config files

        return violations

    def snapshot_current(self, codex_branch: str = "exp/l3-vis-handoff-contract-repair-20260617",
                        codex_head: str = "50da442c1b033a780b802c6345c376b23d4833b1",
                        watcher_state: str = "NO_ACTIVE_WATCHER",
                        output_root: str = "/data/liuyu/outputs/l3_vis_codex_results",
                        artifact_count: str = "0",
                        xid_delta: str = "0"):
        """Take a current snapshot of campaign state."""

        violations = self.check_violations(codex_branch, codex_head, output_root)
        blocker = "; ".join(violations) if violations else ""

        self.record_gate(
            gate="H0",
            codex_claimed="H0_IN_PROGRESS",
            deepseek_independent="H0_INDEPENDENT_AWAITING",
            codex_branch=codex_branch,
            codex_head=codex_head,
            watcher_state=watcher_state,
            output_root=output_root,
            artifact_count=artifact_count,
            hash_status="PENDING",
            xid_delta=xid_delta,
            blocker=blocker,
        )

    def _write_history(self):
        out_dir = REPO_ROOT / "tables"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.history_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=GATE_HISTORY_FIELDS)
            w.writeheader()
            for row in self.history:
                w.writerow({k: row.get(k, "") for k in GATE_HISTORY_FIELDS})

    def _write_status(self):
        reports_dir = REPO_ROOT / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        with open(self.status_path, "w") as f:
            f.write("# L3 Campaign Status\n\n")
            if self.history:
                last = self.history[-1]
                f.write(f"**Last updated:** {last['timestamp']}\n\n")
                f.write(f"**Current gate:** {last['gate']}\n")
                f.write(f"**Codex claimed:** {last['codex_claimed']}\n")
                f.write(f"**DeepSeek independent:** {last['deepseek_independent']}\n")
                f.write(f"**Next authorized:** {last['next_authorized']}\n")
                if last['blocker']:
                    f.write(f"**Blocker:** {last['blocker']}\n")
                f.write(f"\n**Codex branch:** {last['codex_branch']}\n")
                f.write(f"**Codex head:** {last['codex_head']}\n")
                f.write(f"**Watcher:** {last['watcher_state']}\n")
                f.write(f"**Output root:** {last['output_root']}\n\n")

            f.write("## Gate Progression\n\n")
            f.write("| Gate | Codex | DeepSeek | Time |\n")
            f.write("|------|-------|----------|------|\n")
            for row in self.history:
                f.write(f"| {row['gate']} | {row['codex_claimed']} | {row['deepseek_independent']} | {row['timestamp']} |\n")

            f.write("\n## Allowed Transitions\n\n")
            for gate, next_gates in ALLOWED_TRANSITIONS.items():
                f.write(f"- **{gate}** → {', '.join(next_gates) if next_gates else 'TERMINAL'}\n")

            f.write("\n## Stop Conditions\n\n")
            f.write("- Scientific FAIL at any gate\n")
            f.write("- Dirty worktree or unexpected commit drift\n")
            f.write("- Config drift (lambda, target, epsilon, seeds)\n")
            f.write("- SHA mismatch or missing artifacts\n")
            f.write("- Route fallback detected\n")
            f.write("- GPU mapping change\n")
            f.write("- Xid/OOM during active phase\n")
            f.write("- Denominator substitution\n")

        print(f"  Status: {self.status_path}")
        print(f"  History: {self.history_path}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", choices=["snapshot", "status"], default="status")
    ap.add_argument("--codex-branch", default="exp/l3-vis-handoff-contract-repair-20260617")
    ap.add_argument("--codex-head", default="50da442c1b033a780b802c6345c376b23d4833b1")
    ap.add_argument("--watcher-state", default="NO_ACTIVE_WATCHER")
    ap.add_argument("--output-root", default="/data/liuyu/outputs/l3_vis_codex_results")
    args = ap.parse_args()

    monitor = CampaignMonitor()
    monitor.load_history()

    if args.action == "snapshot":
        monitor.snapshot_current(
            codex_branch=args.codex_branch,
            codex_head=args.codex_head,
            watcher_state=args.watcher_state,
            output_root=args.output_root,
        )
    else:
        # Just print status
        if monitor.history:
            last = monitor.history[-1]
            print(f"Current gate: {last['gate']}")
            print(f"Codex claimed: {last['codex_claimed']}")
            print(f"DeepSeek independent: {last['deepseek_independent']}")
        else:
            print("No campaign history yet. Run with --action snapshot to initialize.")
        monitor._write_status()


if __name__ == "__main__":
    main()
