#!/usr/bin/env python3
"""D2: Independent fixed-frame result classifier.

Consumes Codex artifacts and recomputes every result independently.
Classifies at frame-seed, frame, parent, and overall levels.
Primary and diagnostic denominators remain separate.
"""

import csv, hashlib, json, os, re, sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

# Frozen contract
FROZEN_TARGET_TOKEN = 31744
FROZEN_LAMBDA = 2.0
FROZEN_SEEDS = [81, 82]
FROZEN_LINF_BUDGET = 6.0 / 255.0
ARM_MIN_MATCH = 5
EXACT_TOKENS = 7
EXPECTED_CANDIDATE_COUNT = 21

# Selected parents
SELECTED_PARENTS = {
    "butter_s11": {
        "task": "butter", "state_id": 11, "timing_class": "exact",
        "teacher_ws": 58, "teacher_anchor": 60, "teacher_we": 68, "d5_emit": 60,
        "primary_frames": [58, 60], "diagnostic_frames": [68],
    },
    "tomato_sauce_s23": {
        "task": "tomato_sauce", "state_id": 23, "timing_class": "early",
        "teacher_ws": 139, "teacher_anchor": 141, "teacher_we": 149, "d5_emit": 69,
        "primary_frames": [139, 141], "diagnostic_frames": [69],
    },
    "salad_dressing_s11": {
        "task": "salad_dressing", "state_id": 11, "timing_class": "late",
        "teacher_ws": 57, "teacher_anchor": 59, "teacher_we": 67, "d5_emit": 128,
        "primary_frames": [57, 59], "diagnostic_frames": [67, 128],
    },
}

PRIMARY_FRAME_SET = set()
DIAGNOSTIC_FRAME_SET = set()
for pid, sel in SELECTED_PARENTS.items():
    for s in sel["primary_frames"]:
        PRIMARY_FRAME_SET.add((pid, s))
    for s in sel["diagnostic_frames"]:
        DIAGNOSTIC_FRAME_SET.add((pid, s))


def sha256_file(p):
    if not os.path.isfile(p): return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


def load_json(p):
    if not os.path.isfile(p): return None
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def load_csv(p):
    if not os.path.isfile(p): return []
    return list(csv.DictReader(open(p)))


class FixedFrameResultClassifier:
    """Independent recomputation of every Codex fixed-frame result."""

    def __init__(self, output_root: str):
        self.output_root = Path(output_root)
        self.failures = []
        self.warnings = []
        self.frame_seed_results = []  # per frame-per-seed
        self.frame_results = []       # per frame (across both seeds)
        self.parent_results = []      # per parent
        self.checks = 0
        self.checks_ok = 0

    def _chk(self, cond, cat, detail):
        self.checks += 1
        if cond:
            self.checks_ok += 1
        else:
            self.failures.append({"category": cat, "detail": detail})
        return bool(cond)

    def discover_results(self) -> List[Dict]:
        """Walk the output root for per-frame-seed result directories."""
        discovered = []
        if not self.output_root.is_dir():
            return discovered

        for d in sorted(self.output_root.iterdir()):
            if not d.is_dir(): continue
            # Match: {parent}_s{state}_step{step}_seed{seed}_attempt{attempt}
            m = re.match(r"(.+)_s(\d+)_step(\d+)_seed(\d+)_attempt(\d+)", d.name)
            if not m:
                # Try alternate: {task}_s{state}_frame_{step}_seed{seed}
                m = re.match(r"(.+)_s(\d+)_frame_(\d+)_seed(\d+)", d.name)
            if not m:
                # Try: {task}_s{state}_step{step}_seed{seed}
                m = re.match(r"(.+)_s(\d+)_step(\d+)_seed(\d+)$", d.name)
            if not m:
                continue

            task = m.group(1)
            state_id = int(m.group(2))
            step = int(m.group(3))
            seed = int(m.group(4))

            # Map task to parent_id
            pid = None
            for p, sel in SELECTED_PARENTS.items():
                if sel["task"] == task and sel["state_id"] == state_id:
                    pid = p; break
            if pid is None:
                self.warnings.append(f"Unknown parent for task={task} state={state_id}")
                continue

            result_dir = d
            cond_csv = result_dir / "m3_step78_condition_results.csv"
            route_csv = result_dir / "m3_step78_route_audit.csv"
            cand_csv = result_dir / "m3_step78_candidate_controls.csv"
            debug_json = result_dir / "m3_step78_canary_debug.json"
            preflight_json = result_dir / "m3_step78_zero_step_preflight.json"
            manifest_csv = result_dir / "m3_step78_manifest.csv"

            discovered.append({
                "parent_id": pid, "task": task, "state_id": state_id,
                "step": step, "seed": seed,
                "result_dir": str(result_dir),
                "condition_csv": str(cond_csv) if cond_csv.exists() else "",
                "route_csv": str(route_csv) if route_csv.exists() else "",
                "cand_csv": str(cand_csv) if cand_csv.exists() else "",
                "debug_json": str(debug_json) if debug_json.exists() else "",
                "preflight_json": str(preflight_json) if preflight_json.exists() else "",
                "manifest_csv": str(manifest_csv) if manifest_csv.exists() else "",
            })
        return discovered

    def classify_frame_seed(self, entry: Dict) -> str:
        """Classify a single frame-seed result.

        Returns: FRAME_SEED_PASS | FRAME_SEED_SCIENTIFIC_FAIL |
                 FRAME_SEED_INFRA_INVALID | CLEAN_CONTEXT_INELIGIBLE
        """
        pid = entry["parent_id"]
        step = entry["step"]
        seed = entry["seed"]

        # Check for required artifact files
        cond_csv = entry["condition_csv"]
        route_csv = entry["route_csv"]
        cand_csv = entry["cand_csv"]

        if not cond_csv:
            return "FRAME_SEED_INFRA_INVALID"
        if not cand_csv:
            return "FRAME_SEED_INFRA_INVALID"

        # Load condition results
        cond_rows = load_csv(cond_csv)
        if not cond_rows:
            return "FRAME_SEED_INFRA_INVALID"

        # Find TRUE_PGD_FINAL condition
        true_row = None
        clean_row = None
        delta0_row = None
        rand_row = None
        shuffled_row = None
        for r in cond_rows:
            c = r.get("condition", "")
            if c == "TRUE_PGD_FINAL": true_row = r
            elif c == "CLEAN": clean_row = r
            elif c == "PGD_DELTA0": delta0_row = r
            elif c == "RAND20": rand_row = r
            elif c == "SHUFFLED_GRAD_PGD20": shuffled_row = r

        if true_row is None:
            return "FRAME_SEED_INFRA_INVALID"

        # 1. Route contract
        route_status = true_row.get("route_status", "")
        if route_status != "PASS":
            return "FRAME_SEED_SCIENTIFIC_FAIL"

        # 2. Exact 7 tokens
        exact_7 = true_row.get("exact_7_tokens", "")
        if exact_7 != "True":
            return "FRAME_SEED_SCIENTIFIC_FAIL"

        # 3. Official gripper token = 31744
        gripper_token = int(true_row.get("official_gripper_token", "0") or 0)
        if gripper_token != FROZEN_TARGET_TOKEN:
            return "FRAME_SEED_SCIENTIFIC_FAIL"

        # 4. Arm prefix match >= 5/6
        arm_match_count = int(true_row.get("arm_prefix_match_count", "0") or 0)
        arm_match_denom = int(true_row.get("arm_prefix_match_denominator", "0") or 0)
        if arm_match_count < ARM_MIN_MATCH:
            return "FRAME_SEED_SCIENTIFIC_FAIL"

        # 5. TRUE margin > RAND margin
        true_margin = float(true_row.get("official_target31744_margin", "0") or 0)
        if rand_row:
            rand_margin = float(rand_row.get("official_target31744_margin", "-inf") or "-inf")
            if true_margin <= rand_margin:
                return "FRAME_SEED_SCIENTIFIC_FAIL"

        # 6. TRUE margin > shuffled margin
        if shuffled_row:
            shuffled_margin = float(shuffled_row.get("official_target31744_margin", "-inf") or "-inf")
            if true_margin <= shuffled_margin:
                return "FRAME_SEED_SCIENTIFIC_FAIL"

        # 7. processor Linf <= 6/255
        processor_linf = float(true_row.get("processor_linf", "999") or 999)
        if processor_linf > FROZEN_LINF_BUDGET + 1e-9:
            return "FRAME_SEED_SCIENTIFIC_FAIL"

        # 8. Strict route, no fallback
        if route_csv:
            route_rows = load_csv(route_csv)
            true_route = next((r for r in route_rows if r.get("condition") == "TRUE_PGD_FINAL"), None)
            if true_route:
                fallback = true_route.get("fallback_used", "")
                if fallback and fallback != "0" and fallback != "False":
                    return "FRAME_SEED_SCIENTIFIC_FAIL"

        # 9. Verify 21 candidates
        cand_rows = load_csv(cand_csv)
        if cand_rows:
            n_selected = sum(1 for r in cand_rows if int(r.get("selected", "0") or 0) == 1)
            n_cands = len(set(r.get("candidate_id", "") for r in cand_rows))
            if n_cands != EXPECTED_CANDIDATE_COUNT:
                self.warnings.append(f"{pid}:{step}:{seed} has {n_cands} candidates, expected {EXPECTED_CANDIDATE_COUNT}")

        # 10. Score invariant check
        score_inv = true_row.get("score_invariant_status", "")
        if score_inv == "FAIL":
            return "FRAME_SEED_SCIENTIFIC_FAIL"

        # 11. Verify preflight (if exists)
        preflight_json = entry.get("preflight_json", "")
        if preflight_json:
            pf = load_json(preflight_json)
            if pf:
                if pf.get("clean_status") == "SURROGATE_OFFICIAL_SCORE_PATH_MISMATCH":
                    return "FRAME_SEED_SCIENTIFIC_FAIL"
                if pf.get("delta0_status") == "SURROGATE_OFFICIAL_SCORE_PATH_MISMATCH":
                    return "FRAME_SEED_SCIENTIFIC_FAIL"

        # Check clean context
        if clean_row:
            clean_gripper = int(clean_row.get("official_gripper_token", "0") or 0)
            if clean_gripper == 0:
                return "CLEAN_CONTEXT_INELIGIBLE"

        return "FRAME_SEED_PASS"

    def run(self, codex_output_root: str):
        self.output_root = Path(codex_output_root)
        print(f"=== D2: Fixed-Frame Independent Classifier ===")
        print(f"  Output root: {self.output_root}")

        entries = self.discover_results()
        if not entries:
            print("  No Codex results discovered yet.")
            print("  Classifier is ready for when Codex produces H2 outputs.")
            self._write_empty_audit()
            return True

        print(f"  Discovered {len(entries)} result directories")

        # Classify each frame-seed
        for entry in entries:
            pid = entry["parent_id"]
            step = entry["step"]
            seed = entry["seed"]
            result_class = self.classify_frame_seed(entry)

            entry["result_class"] = result_class
            self.frame_seed_results.append(entry)
            print(f"  {pid} step{step} seed{seed}: {result_class}")

        # Aggregate per frame
        frame_keys = defaultdict(list)
        for e in self.frame_seed_results:
            frame_keys[(e["parent_id"], e["step"])].append(e)

        for (pid, step), entries in frame_keys.items():
            seeds_seen = {e["seed"] for e in entries}
            classes = [e["result_class"] for e in entries]

            if seeds_seen != set(FROZEN_SEEDS):
                frame_result = "FRAME_INCOMPLETE"
            elif all(c == "FRAME_SEED_PASS" for c in classes):
                frame_result = "FRAME_TWO_SEED_PASS"
            elif any(c == "FRAME_SEED_PASS" for c in classes):
                frame_result = "FRAME_TWO_SEED_FAIL"
            else:
                frame_result = "FRAME_TWO_SEED_FAIL"

            is_primary = (pid, step) in PRIMARY_FRAME_SET
            is_diag = (pid, step) in DIAGNOSTIC_FRAME_SET
            self.frame_results.append({
                "parent_id": pid, "step": step,
                "is_primary": str(is_primary), "is_diagnostic": str(is_diag),
                "seed81": next((e["result_class"] for e in entries if e["seed"] == 81), "MISSING"),
                "seed82": next((e["result_class"] for e in entries if e["seed"] == 82), "MISSING"),
                "frame_result": frame_result,
            })

        # Aggregate per parent (primary frames only)
        for pid, sel in SELECTED_PARENTS.items():
            primary_frames = sel["primary_frames"]
            frame_outcomes = []
            for fr in self.frame_results:
                if fr["parent_id"] == pid and int(fr["step"]) in primary_frames:
                    frame_outcomes.append(fr["frame_result"])

            if not frame_outcomes:
                parent_result = "PARENT_INCOMPLETE"
            elif all(f == "FRAME_TWO_SEED_PASS" for f in frame_outcomes):
                parent_result = "PARENT_PASS"
            else:
                parent_result = "PARENT_FAIL"

            self.parent_results.append({
                "parent_id": pid, "timing_class": sel["timing_class"],
                "primary_frames_checked": str(primary_frames),
                "primary_frame_outcomes": str(frame_outcomes),
                "parent_result": parent_result,
            })

        # Overall classification (primary only)
        parent_outcomes = [p["parent_result"] for p in self.parent_results]
        n_pass = sum(1 for o in parent_outcomes if o == "PARENT_PASS")

        if all(o == "PARENT_PASS" for o in parent_outcomes):
            overall = "L3_VIS_MULTIPARENT_STRONG_PASS"
        elif n_pass >= 2:
            overall = "L3_VIS_MULTIPARENT_PASS"
        elif any(o == "PARENT_INCOMPLETE" for o in parent_outcomes):
            overall = "L3_VIS_INFRA_INCOMPLETE"
        else:
            overall = "L3_VIS_MULTIPARENT_FAIL"

        self.overall = overall

        self._write_outputs()
        self._print_verdict()
        return overall.startswith("L3_VIS_MULTIPARENT_PASS") or overall == "L3_VIS_MULTIPARENT_STRONG_PASS"

    def _write_empty_audit(self):
        out_dir = REPO_ROOT / "tables"
        out_dir.mkdir(parents=True, exist_ok=True)
        reports_dir = REPO_ROOT / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        header = ["parent_id", "step", "seed", "result_class", "result_dir", "condition_csv", "route_csv", "cand_csv"]
        with open(out_dir / "l3_vis_independent_frame_seed_results.csv", "w", newline="") as f:
            csv.DictWriter(f, fieldnames=header).writeheader()

        with open(reports_dir / "L3_VIS_INDEPENDENT_AUDIT.md", "w") as f:
            f.write("# L3 VIS Independent Audit\n\n**Status:** AWAITING_CODEX_RESULTS\n\nClassifier ready — no results discovered yet.\n")

    def _write_outputs(self):
        out_dir = REPO_ROOT / "tables"
        out_dir.mkdir(parents=True, exist_ok=True)
        reports_dir = REPO_ROOT / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Frame-seed results
        fs_fields = ["parent_id", "step", "seed", "result_class", "result_dir"]
        with open(out_dir / "l3_vis_independent_frame_seed_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fs_fields)
            w.writeheader()
            for r in self.frame_seed_results:
                w.writerow({k: r.get(k, "") for k in fs_fields})

        # Frame results
        if self.frame_results:
            fr_fields = list(self.frame_results[0].keys())
            with open(out_dir / "l3_vis_independent_frame_results.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fr_fields)
                w.writeheader(); w.writerows(self.frame_results)

        # Parent results
        if self.parent_results:
            pr_fields = list(self.parent_results[0].keys())
            with open(out_dir / "l3_vis_independent_parent_results.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=pr_fields)
                w.writeheader(); w.writerows(self.parent_results)

        # Report
        with open(reports_dir / "L3_VIS_INDEPENDENT_AUDIT.md", "w") as f:
            f.write("# L3 VIS Independent Audit Report\n\n")
            f.write(f"**Overall:** {self.overall}\n\n")
            f.write(f"**Parent results:**\n\n")
            for p in self.parent_results:
                f.write(f"- **{p['parent_id']}** ({p['timing_class']}): {p['parent_result']} (frames: {p['primary_frame_outcomes']})\n")
            f.write(f"\n**Frame results:**\n\n")
            for fr in self.frame_results:
                f.write(f"- {fr['parent_id']} step{fr['step']}: {fr['frame_result']} (81:{fr['seed81']} 82:{fr['seed82']})\n")
            if self.failures:
                f.write(f"\n**Failures:**\n\n")
                for fail in self.failures:
                    f.write(f"- {fail['category']}: {fail['detail']}\n")

    def _print_verdict(self):
        print(f"\n{'='*60}")
        print(f"  OVERALL: {self.overall}")
        print(f"  Parents PASS: {sum(1 for p in self.parent_results if p['parent_result'] == 'PARENT_PASS')}/{len(self.parent_results)}")
        print(f"{'='*60}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="D2: Independent fixed-frame result classifier")
    ap.add_argument("--codex-output-root", default="/data/liuyu/outputs/l3_vis_codex_results",
                    help="Codex L3 VIS output root directory")
    ap.add_argument("--mode", choices=["classify", "ready"], default="classify",
                    help="classify: run classification; ready: just report schema readiness")
    args = ap.parse_args()

    classifier = FixedFrameResultClassifier(args.codex_output_root)

    if args.mode == "ready":
        print("D2 classifier is ready. Schema:")
        print("  Frame-seed: FRAME_SEED_PASS | FRAME_SEED_SCIENTIFIC_FAIL | FRAME_SEED_INFRA_INVALID | CLEAN_CONTEXT_INELIGIBLE")
        print("  Frame: FRAME_TWO_SEED_PASS | FRAME_TWO_SEED_FAIL | FRAME_INCOMPLETE")
        print("  Parent: PARENT_PASS | PARENT_FAIL | PARENT_INCOMPLETE")
        print("  Overall: L3_VIS_MULTIPARENT_STRONG_PASS | L3_VIS_MULTIPARENT_PASS | L3_VIS_MULTIPARENT_FAIL | L3_VIS_INFRA_INCOMPLETE")
        return

    passed = classifier.run(args.codex_output_root)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
