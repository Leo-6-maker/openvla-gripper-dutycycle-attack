"""Mutation tests for R5 Canonical A/B Comparator.

Creates fake sealed roots and verifies the comparator correctly detects:
  - Different episode count
  - Different entity pose
  - Different action
  - Missing file in SHA256SUMS
  - Bad sidecar
  - Non-finite values
  - script_sha256 mismatch (after fix)
"""
import json, os, sys, hashlib, math, tempfile, shutil, unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_r5_canonical import (
    full_seal_check, compare_r5e, compare_r5f, compare_c1,
    canonical_sha, VARIANT_MANIFEST_KEYS,
)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def make_sealed_root(tmp, structure):
    """Create a sealed directory from a structure dict.

    WARNING: Use write_bytes() not write_text() on Windows to avoid
    \n → \r\n translation that corrupts SHA hashes.

    structure keys:
      - ".jsonl" suffixed: list of dicts, written as newline-delimited JSON (JSONL)
      - ".json" suffixed: dict, written as JSON
      - other keys: subdirectory, values are dicts of files within
      - "SHA256SUMS" and "SHA256SUMS.sha256": auto-computed
    """
    root = Path(tmp)
    root.mkdir(parents=True, exist_ok=True)

    for name, content in structure.items():
        if name in ("SHA256SUMS", "SHA256SUMS.sha256"):
            continue
        if name.endswith(".jsonl") and isinstance(content, list):
            lines = "\n".join(
                line if isinstance(line, str) else json.dumps(line, sort_keys=True)
                for line in content
            )
            (root / name).write_bytes((lines + "\n").encode("utf-8"))
        elif name.endswith(".json"):
            (root / name).write_bytes(
                json.dumps(content, sort_keys=True).encode("utf-8"))
        elif isinstance(content, dict):
            # Subdirectory (not a .json file)
            subdir = root / name
            subdir.mkdir(exist_ok=True)
            _write_subdir(subdir, content)
        elif isinstance(content, list):
            (root / name).write_bytes(
                json.dumps(content, sort_keys=True).encode("utf-8"))
        else:
            (root / name).write_text(str(content), encoding="utf-8")

    sums_lines = []
    for fpath in sorted(root.rglob("*")):
        if fpath.is_file() and fpath.name not in ("SHA256SUMS", "SHA256SUMS.sha256"):
            rel = fpath.relative_to(root).as_posix()
            h = hashlib.sha256(fpath.read_bytes()).hexdigest()
            sums_lines.append(f"{h}  {rel}")
    sums_data = ("\n".join(sums_lines) + "\n").encode("utf-8")
    (root / "SHA256SUMS").write_bytes(sums_data)
    sums_sha = hashlib.sha256(sums_data).hexdigest()
    (root / "SHA256SUMS.sha256").write_bytes(
        f"{sums_sha}  SHA256SUMS\n".encode("utf-8"))
    return root


def _write_subdir(subdir, content):
    for name, value in content.items():
        path = subdir / name
        if isinstance(value, dict):
            path.write_bytes(json.dumps(value, sort_keys=True).encode("utf-8"))
        else:
            path.write_bytes(str(value).encode("utf-8"))


def make_r5e_manifest(status="SAME_LIVE_GATE_PASS", **overrides):
    m = {
        "gate": "R5-E_FORMAL_SAME_LIVE_GATE",
        "schema": "G_REC_SAME_LIVE_GATE_V2",
        "status": status,
        "mode": "formal",
        "consumer_eligible": True,
        "start_time": "2026-07-27T00:00:00Z",
        "end_time": "2026-07-27T00:01:00Z",
        "elapsed_s": 60.0,
        "source_commit": "abc123",
        "source_tree": "def456",
        "script_sha256": "aaa",
        "n_tasks_tested": 40,
        "n_tasks_expected": 40,
        "n_tasks_skipped": 0,
        "total_records": 320,
        "BC_pos_fail": 0,
        "BC_rot_fail": 0,
        "source_mutations": 0,
        "nonfinite": 0,
        "entity_closure_ok": True,
        "seed": 20260717,
        "state_id": 0,
        "steps_per_task": 10,
        "executable": sys.executable,
        "command": sys.argv,
        "python_version": sys.version,
    }
    m.update(overrides)
    return m


def make_r5e_record(suite="libero_10", task_idx=0, state_id=0, step=0,
                    entity_type="body", entity_id=1, **overrides):
    r = {
        "suite": suite, "task_idx": task_idx, "state_id": state_id,
        "step": step, "entity_type": entity_type, "entity_id": entity_id,
        "entity_name": "test_body",
        "semantic_role": "MANIPULATED_OBJECT",
        "resolution": "EXACT_BODY",
        "AB_pos_Linf": 0.001, "AB_rot_err": 0.0,
        "BC_pos_Linf": 0.0, "BC_rot_err": 0.0,
        "BC_pos_pass": True, "BC_rot_pass": True,
        "AB_stale": True,
        "pos_limit": 1e-8, "rot_limit": 1e-7,
        "source_mutated_fwd1": False,
        "source_mutated_fwd2": False,
        "nonfinite_pose": False,
        "nonfinite_source": False,
        "fwd1_qpos_drift": 0.0, "fwd1_qvel_drift": 0.0,
        "fwd1_act_drift": 0.0, "fwd1_time_drift": 0.0,
        "fwd2_qpos_drift": 0.0, "fwd2_qvel_drift": 0.0,
        "fwd2_act_drift": 0.0, "fwd2_time_drift": 0.0,
    }
    r.update(overrides)
    return r


def make_r5e_summary(task_key="libero_10/task_00", **overrides):
    s = {
        "task_key": task_key, "status": "PASS",
        "n_entities": 4, "n_records": 8,
        "BC_pos_fail": 0, "BC_rot_fail": 0,
        "AB_stale_count": 8,
        "source_mutations": 0, "nonfinite": 0,
        "entity_closure_ok": True,
    }
    s.update(overrides)
    return s


class TestComparatorMutations(unittest.TestCase):
    """Mutation tests for the R5 Canonical A/B Comparator."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _r5e_sealed(self, manifest_overrides=None, records=None, summaries=None, label=""):
        records = records or [make_r5e_record() for _ in range(8)]
        summaries = summaries or [make_r5e_summary()]
        m = make_r5e_manifest(**(manifest_overrides or {}))
        return make_sealed_root(
            os.path.join(self._tmpdir, f"root_{label}"),
            {
                "MANIFEST.json": m,
                "case_records.jsonl": [json.dumps(r, sort_keys=True) for r in records],
                "per_task_summary.jsonl": [json.dumps(s, sort_keys=True) for s in summaries],
            })

    # ── Seal-level mutations ──

    def test_seal_ok_passes(self):
        root = self._r5e_sealed(label="A")
        ok, result = full_seal_check(root)
        self.assertTrue(ok)

    def test_missing_file_in_seal_fails(self):
        root = self._r5e_sealed()
        # Remove a sealed file
        (root / "case_records.jsonl").unlink()
        ok, result = full_seal_check(root)
        self.assertFalse(ok)
        self.assertIn("missing", str(result))

    def test_tampered_file_fails(self):
        root = self._r5e_sealed()
        # Modify a sealed file
        (root / "MANIFEST.json").write_text("{}")
        ok, result = full_seal_check(root)
        self.assertFalse(ok)
        self.assertIn("mismatch", str(result))

    def test_bad_sidecar_fails(self):
        root = self._r5e_sealed()
        # Corrupt SHA256SUMS.sha256
        (root / "SHA256SUMS.sha256").write_text("0000  SHA256SUMS\n")
        ok, result = full_seal_check(root)
        self.assertFalse(ok)
        self.assertIn("sidecar", str(result))

    # ── R5-E comparison mutations ──

    def test_different_record_count_detected(self):
        root_a = self._r5e_sealed(records=[make_r5e_record() for _ in range(8)], label="A")
        root_b = self._r5e_sealed(records=[make_r5e_record() for _ in range(6)], label="B")
        issues = compare_r5e(root_a, root_b)
        self.assertTrue(any("length" in str(i).lower() for i in issues),
                        f"No length issue found: {issues}")

    def test_different_entity_pose_detected(self):
        rec = make_r5e_record(BC_pos_Linf=0.0, BC_pos_pass=True)
        rec_b = dict(rec)
        rec_b["BC_pos_Linf"] = 0.5
        rec_b["BC_pos_pass"] = False
        root_a = self._r5e_sealed(records=[rec], label="A")
        root_b = self._r5e_sealed(records=[rec_b], label="B")
        issues = compare_r5e(root_a, root_b)
        self.assertTrue(any("BC_pos_Linf" in str(i) for i in issues),
                        f"No pose diff found: {issues}")

    def test_different_action_detected(self):
        rec = make_r5e_record(fwd1_act_drift=0.0)
        rec_b = dict(rec)
        rec_b["fwd1_act_drift"] = 0.1
        root_a = self._r5e_sealed(records=[rec], label="A")
        root_b = self._r5e_sealed(records=[rec_b], label="B")
        issues = compare_r5e(root_a, root_b)
        self.assertTrue(any("fwd1_act_drift" in str(i) for i in issues))

    def test_script_sha256_difference_detected(self):
        """After VARIANT_MANIFEST_KEYS fix, script_sha256 must be identical."""
        root_a = self._r5e_sealed(manifest_overrides={"script_sha256": "aaa"}, label="A")
        root_b = self._r5e_sealed(manifest_overrides={"script_sha256": "bbb"}, label="B")
        issues = compare_r5e(root_a, root_b)
        self.assertTrue(any("canonical" in str(i).lower() and "sha" in str(i).lower()
                           for i in issues),
                       f"No script_sha256 diff detected: {issues}")

    def test_seed_difference_detected(self):
        """seed is NOT in variant keys, so different seeds must fail."""
        root_a = self._r5e_sealed(manifest_overrides={"seed": 20260717}, label="A")
        root_b = self._r5e_sealed(manifest_overrides={"seed": 99999999}, label="B")
        issues = compare_r5e(root_a, root_b)
        self.assertTrue(any("canonical" in str(i).lower() and "sha" in str(i).lower()
                           for i in issues),
                       f"No seed diff detected: {issues}")

    def test_nonfinite_float_causes_error(self):
        """Non-finite float in canonical payload must raise ValueError."""
        rec = make_r5e_record(BC_pos_Linf=float('nan'))
        with self.assertRaises((ValueError,)):
            canonical_sha(rec)

    def test_inf_float_causes_error(self):
        rec = make_r5e_record(BC_pos_Linf=float('inf'))
        with self.assertRaises((ValueError,)):
            canonical_sha(rec)

    def test_nonfinite_record_detected_by_assert_finite(self):
        """compare_r5e should detect non-finite values via assert_finite."""
        from compare_r5_canonical import assert_finite
        rec = make_r5e_record(BC_pos_Linf=float('nan'))
        with self.assertRaises(ValueError):
            assert_finite(rec)

    def test_different_entity_name_detected(self):
        rec = make_r5e_record(entity_name="body_A")
        rec_b = dict(rec)
        rec_b["entity_name"] = "body_B"
        root_a = self._r5e_sealed(records=[rec], label="A")
        root_b = self._r5e_sealed(records=[rec_b], label="B")
        issues = compare_r5e(root_a, root_b)
        self.assertTrue(any("entity_name" in str(i) for i in issues))

    def test_different_summary_status_detected(self):
        sum_a = make_r5e_summary(status="PASS")
        sum_b = make_r5e_summary(status="FAIL", BC_pos_fail=3)
        root_a = self._r5e_sealed(summaries=[sum_a], label="A")
        root_b = self._r5e_sealed(summaries=[sum_b], label="B")
        issues = compare_r5e(root_a, root_b)
        self.assertTrue(any("status" in str(i) for i in issues))

    # ── C1 comparison mutations ──

    def _c1_sealed(self, per_task_files=None, label=""):
        pt = {}
        for fn, content in (per_task_files or {}).items():
            pt[fn] = content
        manifest = {
            "gate": "T2R-C1-V2_PER_TASK_REGISTRY",
            "status": "PASS",
            "version": "C1-V2",
            "timestamp": "2026-07-27T00:00:00Z",
            "script_sha256": "aaa",
        }
        return make_sealed_root(
            os.path.join(self._tmpdir, f"c1_root_{label}"),
            {
                "MANIFEST.json": manifest,
                "per_task": pt,
            })

    def test_c1_different_resolution_detected(self):
        rel_a = {
            "predicate": "In",
            "object_bddl": "obj", "target_bddl": "tgt",
            "object_resolution": {"resolution": "EXACT_BODY",
                                   "entity_type": "body", "entity_id": 1},
            "target_resolution": {"resolution": "EXACT_SITE",
                                   "entity_type": "site", "entity_id": 10},
            "relation_ok": True,
        }
        rel_b = {
            "predicate": "In",
            "object_bddl": "obj", "target_bddl": "tgt",
            "object_resolution": {"resolution": "EXACT_GEOM",
                                   "entity_type": "geom", "entity_id": 5},
            "target_resolution": {"resolution": "EXACT_SITE",
                                   "entity_type": "site", "entity_id": 10},
            "relation_ok": True,
        }
        task_a = {"status": "OK", "resolution_counts": {"object_ok": 1, "target_ok": 1},
                  "relations": [rel_a]}
        task_b = {"status": "OK", "resolution_counts": {"object_ok": 1, "target_ok": 1},
                  "relations": [rel_b]}
        root_a = self._c1_sealed({"libero_10_task_00.json": {"legacy": task_a}}, label="A")
        root_b = self._c1_sealed({"libero_10_task_00.json": {"legacy": task_b}}, label="B")
        issues = compare_c1(root_a, root_b)
        self.assertTrue(any("entity_type" in str(i) for i in issues),
                       f"No resolution diff detected: {issues}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
