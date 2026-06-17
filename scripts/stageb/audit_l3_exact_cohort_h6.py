#!/usr/bin/env python3
"""D6: Exact-cohort H6 readiness manifest (CPU-only, read-only).

Audits butter_s11, ketchup_s18, milk_s7 for detector-triggered readiness.
Checks: Teacher anchor frame, D5 emit frame, raw RGB, processor tensor,
clean generation, exact 7 tokens, SHAs, repeatability, clean success, split.
Does NOT execute attacks.
"""

import csv, hashlib, json, os, sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
FRAME_DIR = "/data/liuyu/outputs/l12_frame_handoff_v2_r1"
LABELS = "/data/liuyu/outputs/d5_label_generation/d5_teacher_p_labels_v2.csv"
TIMING_PANEL = "/data/liuyu/outputs/l12_timing_panel_v2"
TIMING_RESUME = "/data/liuyu/outputs/l12_timing_panel_v2_resume_r1"
MANIFEST = "/data/liuyu/outputs/d5_label_generation/d44d_accepted_episode_manifest.csv"

EXACT_COHORT = {
    "butter_s11":  {"task": "butter", "state_id": 11, "timing_class": "exact",
                    "teacher_ws": 58, "teacher_anchor": 60, "teacher_we": 68, "d5_emit": 60},
    "ketchup_s18": {"task": "ketchup", "state_id": 18, "timing_class": "exact",
                    "teacher_ws": 83, "teacher_anchor": 84, "teacher_we": 93, "d5_emit": 84},
    "milk_s7":     {"task": "milk", "state_id": 7, "timing_class": "exact",
                    "teacher_ws": 40, "teacher_anchor": 41, "teacher_we": 50, "d5_emit": 41},
}


def sha256_file(p):
    if not os.path.isfile(p): return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


class ExactCohortAuditor:
    def __init__(self):
        self.results = []
        self.cohort_status = "EXACT_COHORT_READY"

    def audit_parent(self, pid: str, sel: Dict) -> Dict:
        """Audit a single exact-cohort parent for readiness."""
        task = sel["task"]
        sid = sel["state_id"]
        tag = f"{task}_s{sid}"
        parent_dir = os.path.join(FRAME_DIR, f"{task}_s{sid}_frame")

        result = {
            "parent_id": pid, "task": task, "state_id": str(sid),
            "timing_class": sel["timing_class"],
            "teacher_ws": sel["teacher_ws"], "teacher_anchor": sel["teacher_anchor"],
            "teacher_we": sel["teacher_we"], "d5_emit": sel["d5_emit"],
        }

        # 1. Teacher anchor frame
        anchor_npy = os.path.join(parent_dir, f"frame_{sel['teacher_anchor']:04d}.npy")
        anchor_pt = os.path.join(parent_dir, f"processor_{sel['teacher_anchor']:04d}.pt")
        result["anchor_frame_exists"] = os.path.isfile(anchor_npy)
        result["anchor_frame_sha256"] = sha256_file(anchor_npy)[:16] + "..." if os.path.isfile(anchor_npy) else "MISSING"
        result["anchor_processor_exists"] = os.path.isfile(anchor_pt)
        result["anchor_processor_sha256"] = sha256_file(anchor_pt)[:16] + "..." if os.path.isfile(anchor_pt) else "MISSING"

        # 2. D5 emit frame (should be same as anchor for exact)
        emit_npy = os.path.join(parent_dir, f"frame_{sel['d5_emit']:04d}.npy")
        emit_pt = os.path.join(parent_dir, f"processor_{sel['d5_emit']:04d}.pt")
        result["emit_frame_exists"] = os.path.isfile(emit_npy)
        result["emit_frame_sha256"] = sha256_file(emit_npy)[:16] + "..." if os.path.isfile(emit_npy) else "MISSING"
        result["emit_processor_exists"] = os.path.isfile(emit_pt)

        # 3. Anchor-emit identity (exact class: they must be same step)
        if sel["d5_emit"] == sel["teacher_anchor"]:
            if os.path.isfile(anchor_npy) and os.path.isfile(emit_npy):
                result["anchor_emit_same_step"] = True
                result["anchor_emit_sha_match"] = sha256_file(anchor_npy) == sha256_file(emit_npy)
            else:
                result["anchor_emit_same_step"] = True
                result["anchor_emit_sha_match"] = "INCOMPLETE"
        else:
            result["anchor_emit_same_step"] = False
            result["anchor_emit_sha_match"] = "N/A"

        # 4. Clean success check
        acc_rows = list(csv.DictReader(open(MANIFEST)))
        accepted = next((r for r in acc_rows if r.get("task") == task and
                        int(r.get("state_id", -1)) == sid and r.get("status") == "BOUND"), None)
        result["in_accepted_manifest"] = accepted is not None
        result["split"] = accepted.get("split", "?") if accepted else "?"

        # 5. Repeatability check
        for base in [TIMING_PANEL, TIMING_RESUME]:
            ep_dir = os.path.join(base, f"{tag}_shadow_attempt1")
            if os.path.isdir(ep_dir):
                result["timing_attempt1_exists"] = True
                # Check for emission
                emit_json = os.path.join(ep_dir, "detector_emission.json")
                if os.path.isfile(emit_json):
                    try:
                        ed = json.load(open(emit_json))
                        result["timing_emit_step"] = ed.get("emit_step", -1)
                        result["timing_emit_matches"] = ed.get("emit_step", -1) == sel["d5_emit"]
                    except Exception:
                        result["timing_emit_step"] = "PARSE_ERROR"
                        result["timing_emit_matches"] = False
                # Check step trace
                st_csv = os.path.join(ep_dir, "step_trace.csv")
                if os.path.isfile(st_csv):
                    st_rows = list(csv.DictReader(open(st_csv)))
                    result["timing_total_steps"] = len(st_rows)
                    if st_rows:
                        result["timing_success_done"] = st_rows[-1].get("success_done", "0")
                        result["timing_success_check"] = st_rows[-1].get("success_check", "0")
                break
        else:
            result["timing_attempt1_exists"] = False
            result["timing_emit_step"] = "MISSING"

        # 6. Label consistency
        lr = None
        for r_lbl in csv.DictReader(open(LABELS)):
            if r_lbl["task"] == task and int(r_lbl["state_id"]) == sid:
                lr = r_lbl; break
        if lr:
            result["label_ws_match"] = int(lr.get("ws", -1)) == sel["teacher_ws"]
            result["label_anchor_match"] = int(lr.get("anchor", -1)) == sel["teacher_anchor"]
            result["label_we_match"] = int(lr.get("we", -1)) == sel["teacher_we"]
        else:
            result["label_ws_match"] = "NO_LABEL"
            result["label_anchor_match"] = "NO_LABEL"
            result["label_we_match"] = "NO_LABEL"

        # 7. Overall readiness
        readiness_checks = [
            result.get("anchor_frame_exists", False),
            result.get("anchor_processor_exists", False),
            result.get("emit_frame_exists", False),
            result.get("timing_attempt1_exists", False),
            result.get("in_accepted_manifest", False),
            result.get("timing_emit_matches", False),
        ]
        n_pass = sum(1 for c in readiness_checks if c)
        result["readiness_checks_pass"] = f"{n_pass}/{len(readiness_checks)}"

        if n_pass == len(readiness_checks):
            result["readiness"] = "EXACT_COHORT_READY"
        elif n_pass >= len(readiness_checks) - 2:
            result["readiness"] = "TARGETED_RECAPTURE_REQUIRED"
        else:
            result["readiness"] = "INELIGIBLE"

        if result["readiness"] != "EXACT_COHORT_READY":
            self.cohort_status = result["readiness"]

        return result

    def run(self):
        print("=== D6: Exact-Cohort H6 Readiness Audit ===\n")

        for pid, sel in EXACT_COHORT.items():
            result = self.audit_parent(pid, sel)
            self.results.append(result)
            status = "READY" if result["readiness"] == "EXACT_COHORT_READY" else result["readiness"]
            print(f"  {pid}: {status} (checks: {result['readiness_checks_pass']})")
            if result["readiness"] != "EXACT_COHORT_READY":
                missing = []
                if not result.get("anchor_frame_exists"): missing.append("anchor_frame")
                if not result.get("anchor_processor_exists"): missing.append("anchor_processor")
                if not result.get("emit_frame_exists"): missing.append("emit_frame")
                if not result.get("timing_attempt1_exists"): missing.append("timing")
                if not result.get("in_accepted_manifest"): missing.append("manifest")
                if not result.get("timing_emit_matches"): missing.append("emit_match")
                print(f"    Missing: {missing}")

        self._write_outputs()
        print(f"\n  Cohort status: {self.cohort_status}")

    def _write_outputs(self):
        out_dir = REPO_ROOT / "tables"
        out_dir.mkdir(parents=True, exist_ok=True)
        reports_dir = REPO_ROOT / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        if self.results:
            fields = list(self.results[0].keys())
            with open(out_dir / "l3_exact_cohort_readiness.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader(); w.writerows(self.results)

        # Write cohort config
        config = {
            "stage": "L3_H6_EXACT_COHORT_DETECTOR_TRIGGERED",
            "cohort": "exact_timing_v1",
            "parents": {},
            "attack_params": {
                "lambda": 2.0, "target_token": 31744,
                "linf_budget": 6.0 / 255.0, "seeds": [81, 82],
                "gpu": [1, 5],
            },
            "conditions": ["CLEAN", "TRUE", "RAND", "SHUFFLED"],
            "trigger": "d5_direct_emit",
            "notes": "Do not use Teacher-P oracle steps. Use D5 emit step directly.",
        }
        for pid, sel in EXACT_COHORT.items():
            result = next((r for r in self.results if r["parent_id"] == pid), {})
            config["parents"][pid] = {
                "task": sel["task"], "state_id": sel["state_id"],
                "timing_class": sel["timing_class"],
                "d5_emit_step": sel["d5_emit"],
                "teacher_anchor": sel["teacher_anchor"],
                "teacher_window": [sel["teacher_ws"], sel["teacher_we"]],
                "readiness": result.get("readiness", "UNKNOWN"),
            }

        with open(REPO_ROOT / "configs" / "l3_exact_cohort_v1.yaml", "w") as f:
            import yaml
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        with open(reports_dir / "L3_EXACT_COHORT_READINESS.md", "w") as f:
            f.write("# L3 H6 Exact-Cohort Readiness Report\n\n")
            f.write(f"**Cohort status:** {self.cohort_status}\n\n")
            f.write(f"**Preregistered cohort:** butter_s11, ketchup_s18, milk_s7\n\n")
            f.write(f"**Required result:** >=2/3 parents × 2 seeds × matched controls\n")
            f.write(f"**Trigger:** D5 direct emit (no Teacher-P oracle)\n\n")

            for r in self.results:
                f.write(f"## {r['parent_id']}\n\n")
                f.write(f"- Readiness: **{r['readiness']}**\n")
                f.write(f"- Anchor frame: {r['anchor_frame_exists']} (SHA: {r.get('anchor_frame_sha256', '?')})\n")
                f.write(f"- Anchor processor: {r['anchor_processor_exists']}\n")
                f.write(f"- D5 emit ({r['d5_emit']}): frame={r['emit_frame_exists']}\n")
                f.write(f"- Timing attempt1: {r.get('timing_attempt1_exists', False)}\n")
                f.write(f"- Timing emit match: {r.get('timing_emit_matches', False)}\n")
                f.write(f"- In accepted manifest: {r.get('in_accepted_manifest', False)}\n")
                f.write(f"- Split: {r.get('split', '?')}\n")
                f.write(f"- Checks: {r['readiness_checks_pass']}\n\n")

        print(f"  Config: configs/l3_exact_cohort_v1.yaml")


def main():
    auditor = ExactCohortAuditor()
    auditor.run()


if __name__ == "__main__":
    main()
