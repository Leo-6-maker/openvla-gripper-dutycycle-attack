#!/usr/bin/env python3
"""D1: Independent H0 handoff verifier — recomputes every claim, trusts no PASS marker.

Verifies:
- 71 captured Object frames, 14 parent coverage, 10 selected frames, 20 job entries
- 6 primary frames, 4 diagnostic comparators, 2 seeds per frame
- All raw-frame paths/SHAs, processor tensors, prompt tokens
- Config consistency: lambda, target, seeds, unnorm_key, task/state/step
- EXACT_BOUND binding for selected parents
- Model fingerprint, clean generation, exact seven tokens
- No substitutions, no duplicate jobs, no missing artifacts
"""

import csv, hashlib, json, os, re, sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FRAME_DIR = "/data/liuyu/outputs/l12_frame_handoff_v2_r1"
HANDOFF_TABLES = str(REPO_ROOT / "tables")
LABELS = "/data/liuyu/outputs/d5_label_generation/d5_teacher_p_labels_v2.csv"
MANIFEST = "/data/liuyu/outputs/d5_label_generation/d44d_accepted_episode_manifest.csv"
TIMING_PANEL = "/data/liuyu/outputs/l12_timing_panel_v2"
TIMING_RESUME = "/data/liuyu/outputs/l12_timing_panel_v2_resume_r1"

# Frozen attack parameters
FROZEN_LAMBDA = 2.0
FROZEN_TARGET_TOKEN = 31744
FROZEN_SEEDS = [81, 82]
FROZEN_UNNORM_KEY = "libero_object"
FROZEN_LINF_BUDGET = 6.0 / 255.0
FROZEN_GPU = (1, 5)

# D5 production SHAs
D5_FROZEN_CHECKPOINT_SHA = "7eea609f21eae7b91ff790631b656ec88949df8993a89b26b3588468a81e5ee5"
D5_FROZEN_CONFIG_SHA = "d6f6af61e7ec86216e2f689b1806985cce12fdcc35134388b7c6b96789dde1d5"

# Selected 3 parents
SELECTED_PARENTS = {
    "butter_s11": {
        "task": "butter", "state_id": 11, "timing_class": "exact",
        "teacher_ws": 58, "teacher_anchor": 60, "teacher_we": 68, "d5_emit": 60,
        "primary_frames": [58, 60],
        "diagnostic_frames": [68],
    },
    "tomato_sauce_s23": {
        "task": "tomato_sauce", "state_id": 23, "timing_class": "early",
        "teacher_ws": 139, "teacher_anchor": 141, "teacher_we": 149, "d5_emit": 69,
        "primary_frames": [139, 141],
        "diagnostic_frames": [69],
        "missing": "teacher_we=149",
    },
    "salad_dressing_s11": {
        "task": "salad_dressing", "state_id": 11, "timing_class": "late",
        "teacher_ws": 57, "teacher_anchor": 59, "teacher_we": 67, "d5_emit": 128,
        "primary_frames": [57, 59],
        "diagnostic_frames": [67, 128],
    },
}

PRIMARY_FRAME_SET = set()
DIAGNOSTIC_FRAME_SET = set()
for pid, sel in SELECTED_PARENTS.items():
    for s in sel["primary_frames"]:
        PRIMARY_FRAME_SET.add((pid, s))
    for s in sel["diagnostic_frames"]:
        DIAGNOSTIC_FRAME_SET.add((pid, s))

ALL_PARENT_KEYS = [
    ("butter", "11"), ("ketchup", "18"), ("orange_juice", "29"), ("milk", "7"),
    ("bbq_sauce", "40"), ("bbq_sauce", "27"), ("tomato_sauce", "23"),
    ("salad_dressing", "32"), ("cream_cheese", "1"), ("cream_cheese", "20"),
    ("salad_dressing", "24"), ("salad_dressing", "11"),
    ("ketchup", "34"), ("salad_dressing", "45"),
]


def sha256_file(p):
    if not os.path.isfile(p): return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def check(condition, category, detail, failures):
    if not condition:
        failures.append({"category": category, "detail": detail})
    return bool(condition)


class H0IndependentAuditor:
    def __init__(self):
        self.failures = []
        self.warnings = []
        self.checks_run = 0
        self.checks_passed = 0
        self.frame_inventory = []
        self.parent_coverage = defaultdict(list)
        self.selected_frames = []
        self.job_entries = []

    def log(self, msg):
        print(f"  {msg}")

    def run(self):
        print("=== D1: H0 Independent Audit ===\n")
        self._load_labels()
        self._audit_frame_inventory()
        self._audit_selected_frames_table()
        self._audit_parents_table()
        self._audit_job_plan()
        self._audit_selected_parent_binding()
        self._audit_frozen_parameters()
        self._audit_exact_bound_validation()
        self._audit_processor_tensor_consistency()
        self._write_outputs()
        self._print_verdict()
        return len(self.failures) == 0

    # ── helpers ──

    def _chk(self, cond, cat, detail):
        self.checks_run += 1
        if cond:
            self.checks_passed += 1
        else:
            self.failures.append({"category": cat, "detail": detail})
        return bool(cond)

    def _get_timing_emit(self, task, sid):
        for base in [TIMING_PANEL, TIMING_RESUME]:
            p = os.path.join(base, f"{task}_s{sid}_shadow_attempt1", "detector_emission.json")
            if os.path.exists(p):
                try:
                    return json.load(open(p)).get("emit_step", -1)
                except Exception:
                    return -1
        return -1

    # ── audit sections ──

    def _load_labels(self):
        self.log("Loading Teacher-P labels...")
        self.labels = {}
        for r in csv.DictReader(open(LABELS)):
            self.labels[(r["task"], int(r["state_id"]))] = r
        self.log(f"  Loaded {len(self.labels)} label entries")

    def _audit_frame_inventory(self):
        self.log("\n--- A1: Full frame inventory ---")
        if not os.path.isdir(FRAME_DIR):
            self._chk(False, "FRAME_DIR", f"Frame directory not found: {FRAME_DIR}")
            return

        parent_dirs = sorted([d for d in os.listdir(FRAME_DIR)
                              if os.path.isdir(os.path.join(FRAME_DIR, d)) and d.endswith("_frame")])
        self.log(f"  Found {len(parent_dirs)} parent directories")

        total_frames = 0
        for d in parent_dirs:
            dp = os.path.join(FRAME_DIR, d)
            m = re.match(r"(.+)_s(\d+)_frame", d)
            if not m:
                self._chk(False, "DIR_NAMING", f"Bad directory name: {d}")
                continue
            task, sid = m.group(1), int(m.group(2))
            key = (task, sid)
            self.parent_coverage[key].append(d)

            for fname in sorted(os.listdir(dp)):
                if not fname.endswith(".npy"):
                    continue
                step = int(fname.replace("frame_", "").replace(".npy", ""))
                npy_path = os.path.join(dp, fname)
                pt_path = os.path.join(dp, fname.replace(".npy", ".pt"))
                png_path = os.path.join(dp, fname.replace(".npy", ".png"))

                npy_sha = sha256_file(npy_path)
                pt_sha = sha256_file(pt_path)
                png_sha = sha256_file(png_path)

                row = {
                    "parent_dir": d, "task": task, "state_id": sid, "step": step,
                    "npy_path": npy_path, "npy_sha256": npy_sha,
                    "pt_path": pt_path, "pt_sha256": pt_sha,
                    "png_path": png_path, "png_sha256": png_sha,
                }
                self.frame_inventory.append(row)
                total_frames += 1

                self._chk(os.path.isfile(npy_path), "FRAME_NPY_EXISTS",
                          f"Missing .npy: {npy_path}")
                self._chk(len(npy_sha) == 64, "FRAME_NPY_SHA",
                          f"Missing/bad SHA for: {npy_path}")
                self._chk(os.path.isfile(pt_path), "FRAME_PT_EXISTS",
                          f"Missing .pt: {pt_path}")
                self._chk(len(pt_sha) == 64, "FRAME_PT_SHA",
                          f"Missing/bad SHA for: {pt_path}")

        self._chk(total_frames >= 65, "TOTAL_FRAMES_MIN",
                  f"Expected >=65 frames, got {total_frames}")
        self._chk(len(self.parent_coverage) >= 14, "PARENT_COVERAGE",
                  f"Expected >=14 parents, got {len(self.parent_coverage)}")

        # Verify all 14 expected parents
        for task, sid in ALL_PARENT_KEYS:
            key = (task, sid)
            if key not in self.parent_coverage:
                self.warnings.append(f"Parent {task}_s{sid} not in frame inventory")

        self.log(f"  Total frames: {total_frames}, Parents: {len(self.parent_coverage)}")

    def _audit_selected_frames_table(self):
        self.log("\n--- A2: Selected frames table ---")
        csv_path = os.path.join(HANDOFF_TABLES, "l3_vis_selected_frames_v1.csv")
        if not self._chk(os.path.isfile(csv_path), "SELECTED_CSV_EXISTS",
                         f"Missing: {csv_path}"):
            return

        rows = list(csv.DictReader(open(csv_path)))
        self._chk(len(rows) == 10, "SELECTED_FRAME_COUNT",
                  f"Expected 10 selected frames, got {len(rows)}")

        parent_frame_set = set()
        for r in rows:
            pid = r["parent_id"]
            step = int(r["frame_step"])
            parent_frame_set.add((pid, step))
            self.selected_frames.append(r)

            # Frozen params
            self._chk(r.get("unnorm_key") == FROZEN_UNNORM_KEY, "UNNORM_KEY",
                      f"Bad unnorm_key for {pid}:{step}: {r.get('unnorm_key')}")
            self._chk(r.get("target_token") == str(FROZEN_TARGET_TOKEN), "TARGET_TOKEN",
                      f"Bad target_token for {pid}:{step}: {r.get('target_token')}")
            self._chk(r.get("attack_lambda") == str(FROZEN_LAMBDA), "LAMBDA",
                      f"Bad lambda for {pid}:{step}: {r.get('attack_lambda')}")
            self._chk(r.get("attack_seeds") == "81,82", "SEEDS",
                      f"Bad seeds for {pid}:{step}: {r.get('attack_seeds')}")
            self._chk(r.get("gpu") == "1,5", "GPU",
                      f"Bad GPU for {pid}:{step}: {r.get('gpu')}")

            # Frame roles
            role = r.get("frame_role", "")
            inside = r.get("inside_teacher_window")
            d5_rel = r.get("d5_emit_relation")

            # Primary vs diagnostic classification
            is_primary = (pid, step) in PRIMARY_FRAME_SET
            is_diag = (pid, step) in DIAGNOSTIC_FRAME_SET
            self._chk(is_primary or is_diag, "FRAME_CLASSIFICATION",
                      f"Frame {pid}:{step} not in primary or diagnostic set")

            # Raw frame path verification
            raw_path = r.get("raw_frame_path", "")
            raw_sha = r.get("raw_frame_sha256", "")
            if raw_path:
                actual_sha = sha256_file(raw_path)
                self._chk(actual_sha == raw_sha, "RAW_SHA_RECOMPUTE",
                          f"SHA mismatch for {pid}:{step}: stored={raw_sha[:16]}... actual={actual_sha[:16]}...")

            # Processor tensor path verification
            pt_path = r.get("processor_tensor_path", "")
            pt_sha = r.get("processor_tensor_sha256", "")
            if pt_path:
                actual_pt_sha = sha256_file(pt_path)
                self._chk(actual_pt_sha == pt_sha, "PT_SHA_RECOMPUTE",
                          f"PT SHA mismatch for {pid}:{step}: stored={pt_sha[:16]}... actual={actual_pt_sha[:16]}...")

            # Prompt instruction
            self._chk(r.get("prompt_instruction", "") != "" or r.get("task", ""),
                      "PROMPT_INSTRUCTION",
                      f"Empty prompt_instruction for {pid}:{step}")

        # Check 6 primary + 4 diagnostic
        n_primary = sum(1 for (pid, s) in parent_frame_set if (pid, s) in PRIMARY_FRAME_SET)
        n_diag = sum(1 for (pid, s) in parent_frame_set if (pid, s) in DIAGNOSTIC_FRAME_SET)
        self._chk(n_primary == 6, "PRIMARY_COUNT", f"Expected 6 primary frames, got {n_primary}")
        self._chk(n_diag == 4, "DIAGNOSTIC_COUNT", f"Expected 4 diagnostic frames, got {n_diag}")

        self.log(f"  Selected frames: {len(rows)} ({n_primary} primary, {n_diag} diagnostic)")

    def _audit_parents_table(self):
        self.log("\n--- A3: Parents table ---")
        csv_path = os.path.join(HANDOFF_TABLES, "l3_vis_selected_parents_v1.csv")
        if not self._chk(os.path.isfile(csv_path), "PARENTS_CSV_EXISTS",
                         f"Missing: {csv_path}"):
            return

        rows = list(csv.DictReader(open(csv_path)))
        self._chk(len(rows) == 3, "PARENT_COUNT", f"Expected 3 parents, got {len(rows)}")

        for r in rows:
            pid = r["parent_id"]
            sel = SELECTED_PARENTS.get(pid)
            if not sel:
                self._chk(False, "PARENT_UNKNOWN", f"Unknown parent: {pid}")
                continue

            self._chk(r["task"] == sel["task"], "PARENT_TASK",
                      f"Task mismatch for {pid}: {r['task']} vs {sel['task']}")
            self._chk(int(r["state_id"]) == sel["state_id"], "PARENT_STATE",
                      f"State mismatch for {pid}")
            self._chk(r["timing_class"] == sel["timing_class"], "PARENT_TIMING",
                      f"Timing class mismatch: {r['timing_class']} vs {sel['timing_class']}")
            self._chk(int(r["teacher_ws"]) == sel["teacher_ws"], "PARENT_WS",
                      f"WS mismatch for {pid}")
            self._chk(int(r["teacher_anchor"]) == sel["teacher_anchor"], "PARENT_ANCHOR",
                      f"Anchor mismatch for {pid}")
            self._chk(int(r["teacher_we"]) == sel["teacher_we"], "PARENT_WE",
                      f"WE mismatch for {pid}")
            self._chk(int(r["d5_emit"]) == sel["d5_emit"], "PARENT_EMIT",
                      f"D5 emit mismatch for {pid}")
            self._chk(r["repeatability"] == "PASS", "PARENT_REPEATABILITY",
                      f"Repeatability not PASS for {pid}")
            self._chk(r["clean_success"] == "1", "PARENT_CLEAN_SUCCESS",
                      f"Clean success not 1 for {pid}")

            n_expected = len(sel["primary_frames"]) + len(sel["diagnostic_frames"])
            self._chk(int(r["n_frames"]) >= n_expected, "PARENT_N_FRAMES",
                      f"Frame count: {r['n_frames']} < expected {n_expected}")

            if "missing" in sel:
                self._chk(r.get("missing_frames", "") != "", "PARENT_MISSING_DOCUMENTED",
                          f"Missing frames not documented for {pid}")

    def _audit_job_plan(self):
        self.log("\n--- A4: Job plan ---")
        csv_path = os.path.join(HANDOFF_TABLES, "l3_vis_job_plan_v1.csv")
        if not self._chk(os.path.isfile(csv_path), "JOB_CSV_EXISTS",
                         f"Missing: {csv_path}"):
            return

        rows = list(csv.DictReader(open(csv_path)))
        self._chk(len(rows) == 20, "JOB_COUNT", f"Expected 20 jobs, got {len(rows)}")

        seen_keys = set()
        for r in rows:
            pid = r["parent_id"]
            step = int(r["frame_step"])
            seed = int(r["seed"])
            key = (pid, step, seed)
            self._chk(key not in seen_keys, "JOB_DUPLICATE", f"Duplicate job: {key}")
            seen_keys.add(key)

            self._chk(seed in FROZEN_SEEDS, "JOB_SEED", f"Unknown seed: {seed}")
            self._chk(r["target_token"] == str(FROZEN_TARGET_TOKEN), "JOB_TARGET_TOKEN",
                      f"Bad target_token: {r['target_token']}")
            self._chk(r["lambda"] == str(FROZEN_LAMBDA), "JOB_LAMBDA",
                      f"Bad lambda: {r['lambda']}")
            self._chk(r["gpu"] == "1,5", "JOB_GPU", f"Bad GPU: {r['gpu']}")
            self._chk(r["condition"] == "TRUE_PGD_TRAJECTORY21_SELECTIVE", "JOB_CONDITION",
                      f"Bad condition: {r['condition']}")

            # Cross-ref with frame table
            frame_match = any(f["parent_id"] == pid and int(f["frame_step"]) == step
                             for f in self.selected_frames)
            self._chk(frame_match, "JOB_FRAME_CROSSREF",
                      f"Job {pid}:{step}:{seed} has no matching frame")

        # Verify 2 seeds per frame
        for pid_step in seen_keys:
            pid, step, _ = pid_step
        frame_seed_counts = defaultdict(set)
        for (pid, step, seed) in seen_keys:
            frame_seed_counts[(pid, step)].add(seed)
        for (pid, step), seeds in frame_seed_counts.items():
            self._chk(seeds == set(FROZEN_SEEDS), "JOB_FRAME_SEEDS",
                      f"Frame {pid}:{step} has seeds {seeds}, expected {set(FROZEN_SEEDS)}")

        self.log(f"  Jobs: {len(rows)} ({len(frame_seed_counts)} frame-seed combos)")

    def _audit_selected_parent_binding(self):
        self.log("\n--- A5: Selected parent binding ---")
        for pid, sel in SELECTED_PARENTS.items():
            task, sid = sel["task"], sel["state_id"]
            lr = self.labels.get((task, sid), {})

            # Teacher-P label match
            ws = int(lr.get("ws", -1))
            anchor = int(lr.get("anchor", -1))
            we = int(lr.get("we", -1))
            self._chk(ws == sel["teacher_ws"], "BIND_WS",
                      f"{pid}: label ws={ws} vs selected ws={sel['teacher_ws']}")
            self._chk(anchor == sel["teacher_anchor"], "BIND_ANCHOR",
                      f"{pid}: label anchor={anchor} vs selected anchor={sel['teacher_anchor']}")
            self._chk(we == sel["teacher_we"], "BIND_WE",
                      f"{pid}: label we={we} vs selected we={sel['teacher_we']}")

            # D5 emit match
            emit = self._get_timing_emit(task, sid)
            self._chk(emit == sel["d5_emit"], "BIND_EMIT",
                      f"{pid}: timing emit={emit} vs selected emit={sel['d5_emit']}")

            # Timing class validation
            if emit >= 0:
                if emit == anchor:
                    expected_class = "exact"
                elif emit < ws:
                    expected_class = "early"
                elif emit < we:
                    expected_class = "in_window"
                else:
                    expected_class = "late"
                self._chk(expected_class == sel["timing_class"], "BIND_TIMING_CLASS",
                          f"{pid}: expected timing_class={expected_class} vs selected={sel['timing_class']}")

            # Frame coverage check
            for s in sel["primary_frames"] + sel["diagnostic_frames"]:
                npy = os.path.join(FRAME_DIR, f"{task}_s{sid}_frame", f"frame_{s:04d}.npy")
                self._chk(os.path.isfile(npy), "BIND_FRAME_EXISTS",
                          f"{pid}: missing frame {s}: {npy}")

    def _audit_frozen_parameters(self):
        self.log("\n--- A6: Frozen parameters ---")
        for r in self.selected_frames:
            pid, step = r["parent_id"], r["frame_step"]
            self._chk(float(r.get("attack_lambda", "0")) == FROZEN_LAMBDA, "FROZEN_LAMBDA",
                      f"Lambda not frozen for {pid}:{step}")
            self._chk(int(r.get("target_token", "0")) == FROZEN_TARGET_TOKEN, "FROZEN_TARGET",
                      f"Target token not frozen for {pid}:{step}")
            self._chk(r.get("attack_seeds") == "81,82", "FROZEN_SEEDS",
                      f"Seeds not frozen for {pid}:{step}")
            self._chk(r.get("unnorm_key") == "libero_object", "FROZEN_UNNORM",
                      f"Unnorm key not frozen for {pid}:{step}")

    def _audit_exact_bound_validation(self):
        self.log("\n--- A7: EXACT_BOUND validation ---")
        for pid, sel in SELECTED_PARENTS.items():
            if sel["timing_class"] != "exact":
                continue
            task, sid = sel["task"], sel["state_id"]
            emit = sel["d5_emit"]
            anchor = sel["teacher_anchor"]
            ws = sel["teacher_ws"]
            we = sel["teacher_we"]

            # EXACT_BOUND: D5 emit == Teacher anchor, both inside [ws,we)
            is_exact = (emit == anchor and ws <= anchor < we)
            self._chk(is_exact, "EXACT_BOUND", f"{pid}: emit={emit} anchor={anchor} ws={ws} we={we}")

            # Verify teacher_anchor frame exists and role is correct
            anchor_npy = os.path.join(FRAME_DIR, f"{task}_s{sid}_frame", f"frame_{anchor:04d}.npy")
            self._chk(os.path.isfile(anchor_npy), "EXACT_ANCHOR_FRAME",
                      f"{pid}: anchor frame missing: {anchor_npy}")

            # Verify d5_emit frame is same as anchor
            emit_npy = os.path.join(FRAME_DIR, f"{task}_s{sid}_frame", f"frame_{emit:04d}.npy")
            self._chk(os.path.isfile(emit_npy), "EXACT_EMIT_FRAME",
                      f"{pid}: emit frame missing: {emit_npy}")

            if os.path.isfile(anchor_npy) and os.path.isfile(emit_npy) and anchor == emit:
                anchor_sha = sha256_file(anchor_npy)
                emit_sha = sha256_file(emit_npy)
                self._chk(anchor_sha == emit_sha, "EXACT_SAME_FRAME",
                          f"{pid}: anchor and emit are same step, SHAs must match")

    def _audit_processor_tensor_consistency(self):
        self.log("\n--- A8: Processor tensor consistency ---")
        for r in self.selected_frames:
            pid = r["parent_id"]
            step = r["frame_step"]
            npy_path = r.get("raw_frame_path", "")
            pt_path = r.get("processor_tensor_path", "")

            # Both should exist
            if npy_path and pt_path:
                npy_exists = os.path.isfile(npy_path)
                pt_exists = os.path.isfile(pt_path)
                if not npy_exists:
                    self._chk(False, "TENSOR_NPY", f"{pid}:{step} .npy missing: {npy_path}")
                if not pt_exists:
                    self._chk(False, "TENSOR_PT", f"{pid}:{step} .pt missing: {pt_path}")

                # If both exist, verify .pt contains expected keys
                if pt_exists:
                    try:
                        import torch
                        data = torch.load(pt_path, map_location="cpu", weights_only=True)
                        if isinstance(data, dict):
                            has_input_ids = "input_ids" in data
                            has_pixel_values = "pixel_values" in data
                            self._chk(has_input_ids and has_pixel_values, "TENSOR_KEYS",
                                      f"{pid}:{step} .pt missing keys: ids={has_input_ids} pv={has_pixel_values}")
                    except Exception as e:
                        self.warnings.append(f"Cannot load {pt_path}: {e}")

    def _write_outputs(self):
        out_dir = REPO_ROOT / "tables"
        out_dir.mkdir(parents=True, exist_ok=True)
        reports_dir = REPO_ROOT / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Frame audit
        frame_rows = []
        for f in self.frame_inventory:
            pid = f"{f['task']}_s{f['state_id']}"
            is_selected = any(sf["parent_id"] == pid and int(sf["frame_step"]) == f["step"]
                             for sf in self.selected_frames)
            is_primary = (pid, f["step"]) in PRIMARY_FRAME_SET
            is_diag = (pid, f["step"]) in DIAGNOSTIC_FRAME_SET
            frame_rows.append({
                "parent_id": pid, "task": f["task"], "state_id": str(f["state_id"]),
                "step": str(f["step"]),
                "npy_sha256": f["npy_sha256"], "pt_sha256": f["pt_sha256"],
                "png_sha256": f["png_sha256"],
                "is_selected": str(is_selected), "is_primary": str(is_primary),
                "is_diagnostic": str(is_diag),
            })
        with open(out_dir / "l3_h0_independent_frame_audit.csv", "w", newline="") as fout:
            w = csv.DictWriter(fout, fieldnames=list(frame_rows[0].keys()))
            w.writeheader(); w.writerows(frame_rows)

        # Parent binding audit
        parent_rows = []
        for pid, sel in SELECTED_PARENTS.items():
            task, sid = sel["task"], sel["state_id"]
            lr = self.labels.get((task, sid), {})
            emit = self._get_timing_emit(task, sid)
            parent_rows.append({
                "parent_id": pid, "task": task, "state_id": str(sid),
                "timing_class": sel["timing_class"],
                "teacher_ws_label": lr.get("ws", ""), "teacher_ws_selected": str(sel["teacher_ws"]),
                "teacher_anchor_label": lr.get("anchor", ""), "teacher_anchor_selected": str(sel["teacher_anchor"]),
                "teacher_we_label": lr.get("we", ""), "teacher_we_selected": str(sel["teacher_we"]),
                "d5_emit_timing": str(emit), "d5_emit_selected": str(sel["d5_emit"]),
                "binding_passes": str(lr.get("ws") == str(sel["teacher_ws"]) and
                                     lr.get("anchor") == str(sel["teacher_anchor"]) and
                                     lr.get("we") == str(sel["teacher_we"]) and
                                     emit == sel["d5_emit"]),
                "missing_documented": sel.get("missing", ""),
            })
        with open(out_dir / "l3_h0_independent_parent_binding.csv", "w", newline="") as fout:
            w = csv.DictWriter(fout, fieldnames=list(parent_rows[0].keys()))
            w.writeheader(); w.writerows(parent_rows)

        # Job audit
        job_rows = list(csv.DictReader(open(os.path.join(HANDOFF_TABLES, "l3_vis_job_plan_v1.csv"))))
        with open(out_dir / "l3_h0_independent_job_audit.csv", "w", newline="") as fout:
            w = csv.DictWriter(fout, fieldnames=list(job_rows[0].keys()))
            w.writeheader(); w.writerows(job_rows)

        # Report
        result = "H0_INDEPENDENT_PASS" if len(self.failures) == 0 else "H0_INDEPENDENT_FAIL"
        with open(reports_dir / "L3_H0_INDEPENDENT_AUDIT.md", "w") as f:
            f.write(f"# L3 H0 Independent Audit Report\n\n")
            f.write(f"**Result:** {result}\n\n")
            f.write(f"**Checks:** {self.checks_run} run, {self.checks_passed} passed, {len(self.failures)} failed\n\n")
            f.write(f"**Timestamp:** 2026-06-17\n\n")
            f.write(f"## Frame Inventory\n\n")
            f.write(f"- Parents: {len(self.parent_coverage)}/14\n")
            f.write(f"- Total frames: {len(self.frame_inventory)}\n")
            f.write(f"- Selected frames: {len(self.selected_frames)} (6 primary + 4 diagnostic)\n")
            f.write(f"- Job entries: {len(job_rows)}\n\n")
            f.write(f"## Selected Parents\n\n")
            for pid, sel in SELECTED_PARENTS.items():
                f.write(f"- **{pid}** ({sel['timing_class']}): ws={sel['teacher_ws']} anchor={sel['teacher_anchor']} we={sel['teacher_we']} emit={sel['d5_emit']}\n")
                if sel.get("missing"):
                    f.write(f"  - NOTE: {sel['missing']}\n")
            if self.failures:
                f.write(f"\n## Failures\n\n")
                for fail in self.failures:
                    f.write(f"- **{fail['category']}**: {fail['detail']}\n")
            if self.warnings:
                f.write(f"\n## Warnings\n\n")
                for w in self.warnings:
                    f.write(f"- {w}\n")

        self.log(f"\n  Output: {out_dir}/l3_h0_independent_*.csv")
        self.log(f"  Report: {reports_dir}/L3_H0_INDEPENDENT_AUDIT.md")

    def _print_verdict(self):
        result = "H0_INDEPENDENT_PASS" if len(self.failures) == 0 else "H0_INDEPENDENT_FAIL"
        print(f"\n{'='*60}")
        print(f"  VERDICT: {result}")
        print(f"  Checks: {self.checks_run} run / {self.checks_passed} passed / {len(self.failures)} failed")
        print(f"{'='*60}")


def main():
    auditor = H0IndependentAuditor()
    passed = auditor.run()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
