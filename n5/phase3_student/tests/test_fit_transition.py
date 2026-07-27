"""[DeepSeek] FIT-INFERENCE Transition Negative Tests (v2).

All rejections must occur BEFORE model loading.
"""
import json, os, sys, hashlib, shutil, tempfile, unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'phase2_labels'))
from fit_transition import (
    verify_transition, TransitionRejected, sha256_file, full_seal_check,
    compute_model_tree_fingerprint, validate_identity_allowlist, FROZEN_R5E,
)

EXEC_COMMIT = "e" * 40
SCRIPT_SHA = "a" * 64
GPU = 6
OUTPUT_ROOT = str(Path("/tmp/test_r5f_output").resolve())

VALID_MANIFEST_BASE = {
    "gate": "FIT-INFERENCE_TRANSITION",
    "schema": "FIT_INFERENCE_TRANSITION_V1",
    "status": "FROZEN_BEFORE_EXECUTION",
    "created_at": "2026-07-28T00:00:00Z",
    **FROZEN_R5E,
    "r5e_comparison_sha256": "c" * 64,
    "r5f_execution_source_commit": EXEC_COMMIT,
    "r5f_script_sha256": SCRIPT_SHA,
    "model_tree_sha256": "",  # filled by helper
    "processor_sha256": "",
    "official_worker_sha256": "",
    "pilot_manifest_sha256": "",
    "registry_summary_sha256": "",
    "alias_ledger_sha256": "",
    "upstream_commit": "u" * 40,
    "libero_fingerprint": "l" * 64,
    "identity_allowlist_digest": "",
    "identity_set_digest": "",
    "authorized_identities": 40,
    "allowed_gpus": [6, 7],
    "allowed_output_roots": [OUTPUT_ROOT],
    "openvla_inference_authorized": True,
    "clean_action_only": True,
    "forward_before_capture": True,
    "max_episodes": 40,
    "identity_set_frozen": True,
    "teacher_labels_authorized": False,
    "student_training_authorized": False,
    "detector_load_authorized": False,
    "attack_authorized": False,
    "protected_payload_read": False,
}


def _make_temp_file(content, suffix=".json"):
    p = Path(tempfile.mktemp(suffix=suffix))
    p.write_text(content)
    return p

def _make_pilot():
    records = []
    for suite in ["libero_10", "libero_goal", "libero_object", "libero_spatial"]:
        for tid in range(10):
            records.append({
                "episode_id": f"{suite}/task_{tid:02d}/state_0",
                "suite": suite, "task_id": tid, "state_id": 0,
                "collection_seed": 20260717,
                "initial_state_sha256": "0" * 64,
            })
    return json.dumps({"protected_payload_read": False, "no_attack": True,
                        "records": records})

def _make_allowlist():
    identities = []
    for suite in ["libero_10", "libero_goal", "libero_object", "libero_spatial"]:
        for tid in range(10):
            identities.append({
                "episode_id": f"{suite}/task_{tid:02d}/state_0",
                "suite": suite, "task_id": tid, "state_id": 0,
                "collection_seed": 20260717,
                "initial_state_sha256": "0" * 64,
            })
    return json.dumps({"gate": "FIT-INFERENCE_IDENTITY_ALLOWLIST",
                        "n_identities": 40, "identities": identities})

def _make_model_dir():
    d = Path(tempfile.mkdtemp(prefix="test_model_"))
    (d / "config.json").write_text('{"test": true}')
    (d / "preprocessor_config.json").write_text('{"test": true}')
    return d

def _sealed_transition(manifest_overrides=None, tamper=None, extra_file=None):
    """Create a sealed transition receipt with all files."""
    root = Path(tempfile.mkdtemp(prefix="fit_test_transition_"))
    # Create dummy model for fingerprint
    model_dir = _make_model_dir()
    worker_file = _make_temp_file("# test worker")
    pilot_file = _make_temp_file(_make_pilot())
    registry_file = _make_temp_file('{"status":"PASS"}')
    alias_file = _make_temp_file('{"n_aliases":5}')

    manifest = dict(VALID_MANIFEST_BASE)
    manifest["model_tree_sha256"] = compute_model_tree_fingerprint(model_dir)
    manifest["processor_sha256"] = sha256_file(model_dir / "preprocessor_config.json")
    manifest["official_worker_sha256"] = sha256_file(worker_file)
    manifest["pilot_manifest_sha256"] = sha256_file(pilot_file)
    manifest["registry_summary_sha256"] = sha256_file(registry_file)
    manifest["alias_ledger_sha256"] = sha256_file(alias_file)

    # Identity allowlist
    allowlist_content = _make_allowlist()
    allowlist_path = root / "IDENTITY_ALLOWLIST.json"
    allowlist_path.write_text(allowlist_content)
    manifest["identity_allowlist_digest"] = sha256_file(allowlist_path)
    manifest["identity_set_digest"] = hashlib.sha256(
        json.dumps(json.loads(allowlist_content)["identities"], sort_keys=True).encode()
    ).hexdigest()

    if manifest_overrides:
        manifest.update(manifest_overrides)

    (root / "TRANSITION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True))

    if extra_file:
        (root / extra_file).write_text("UNSEALED")

    # Seal
    payload = sorted(p for p in root.rglob("*") if p.is_file())
    sums = "\n".join(
        f"{sha256_file(p)}  {p.relative_to(root).as_posix()}" for p in payload) + "\n"
    (root / "SHA256SUMS").write_text(sums)
    sums_sha = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n")

    if tamper:
        target = root / tamper
        target.write_text(target.read_text() + "TAMPERED")

    return root, model_dir, worker_file, pilot_file, registry_file, alias_file


def _verify(root, model_dir, worker, pilot, registry, alias, **kw):
    return verify_transition(
        root, kw.get("commit", EXEC_COMMIT), kw.get("script", SCRIPT_SHA),
        model_dir, worker, pilot, registry, alias,
        kw.get("output", OUTPUT_ROOT), kw.get("gpu", GPU))


class TestTransitionSealRejects(unittest.TestCase):
    def test_01_missing_receipt(self):
        with self.assertRaises((TransitionRejected, SystemExit, FileNotFoundError)):
            _verify("/nonexistent", Path("/tmp/m"), Path("/tmp/w"),
                    Path("/tmp/p"), Path("/tmp/r"), Path("/tmp/a"))

    def test_02_unsealed_root(self):
        root = Path(tempfile.mkdtemp())
        (root / "TRANSITION_MANIFEST.json").write_text("{}")
        with self.assertRaises(TransitionRejected):
            _verify(root, Path("/tmp/m"), Path("/tmp/w"),
                    Path("/tmp/p"), Path("/tmp/r"), Path("/tmp/a"))
        shutil.rmtree(root, ignore_errors=True)

    def test_03_tampered_seal(self):
        root, md, wk, pl, rg, al = _sealed_transition(tamper="TRANSITION_MANIFEST.json")
        try:
            with self.assertRaises(TransitionRejected):
                _verify(root, md, wk, pl, rg, al)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_04_extra_unsealed_file(self):
        root, md, wk, pl, rg, al = _sealed_transition(extra_file="EXTRA_FILE.txt")
        try:
            with self.assertRaises(TransitionRejected):
                _verify(root, md, wk, pl, rg, al)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_05_path_traversal_in_manifest(self):
        root = Path(tempfile.mkdtemp())
        sums = f"{'0'*64}  ../etc/passwd\n"
        (root / "SHA256SUMS").write_text(sums)
        (root / "SHA256SUMS.sha256").write_text(f"{sha256_file(root / 'SHA256SUMS')}  SHA256SUMS\n")
        (root / "TRANSITION_MANIFEST.json").write_text("{}")
        with self.assertRaises(TransitionRejected):
            _verify(root, Path("/tmp/m"), Path("/tmp/w"),
                    Path("/tmp/p"), Path("/tmp/r"), Path("/tmp/a"))
        shutil.rmtree(root, ignore_errors=True)


class TestTransitionFrozenBindings(unittest.TestCase):
    def test_06_wrong_c1_digest(self):
        root, md, wk, pl, rg, al = _sealed_transition(
            manifest_overrides={"c1_canonical_digest": "0" * 64})
        try:
            with self.assertRaises(TransitionRejected):
                _verify(root, md, wk, pl, rg, al)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_07_wrong_r5e_digest(self):
        root, md, wk, pl, rg, al = _sealed_transition(
            manifest_overrides={"r5e_run_a_sha256sums": "0" * 64})
        try:
            with self.assertRaises(TransitionRejected):
                _verify(root, md, wk, pl, rg, al)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_08_missing_comparison_sha(self):
        root, md, wk, pl, rg, al = _sealed_transition(
            manifest_overrides={"r5e_comparison_sha256": ""})
        try:
            with self.assertRaises(TransitionRejected):
                _verify(root, md, wk, pl, rg, al)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_09_wrong_source_commit(self):
        root, md, wk, pl, rg, al = _sealed_transition()
        try:
            with self.assertRaises(TransitionRejected):
                _verify(root, md, wk, pl, rg, al, commit="0" * 40)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_10_wrong_script_sha(self):
        root, md, wk, pl, rg, al = _sealed_transition()
        try:
            with self.assertRaises(TransitionRejected):
                _verify(root, md, wk, pl, rg, al, script="0" * 64)
        finally:
            shutil.rmtree(root, ignore_errors=True)


class TestTransitionModelBinding(unittest.TestCase):
    def test_11_wrong_model_tree(self):
        root, md, wk, pl, rg, al = _sealed_transition(
            manifest_overrides={"model_tree_sha256": "0" * 64})
        try:
            with self.assertRaises(TransitionRejected):
                _verify(root, md, wk, pl, rg, al)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_12_wrong_processor_config(self):
        root, md, wk, pl, rg, al = _sealed_transition(
            manifest_overrides={"processor_sha256": "0" * 64})
        try:
            with self.assertRaises(TransitionRejected):
                _verify(root, md, wk, pl, rg, al)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_13_wrong_worker_sha(self):
        root, md, wk, pl, rg, al = _sealed_transition()
        # Create different worker file
        wrong_worker = _make_temp_file("# wrong worker")
        try:
            with self.assertRaises(TransitionRejected):
                _verify(root, md, wrong_worker, pl, rg, al)
        finally:
            shutil.rmtree(root, ignore_errors=True)


class TestTransitionPermissions(unittest.TestCase):
    def test_14_teacher_authorized(self):
        root, md, wk, pl, rg, al = _sealed_transition(
            manifest_overrides={"teacher_labels_authorized": True})
        try:
            with self.assertRaises(TransitionRejected):
                _verify(root, md, wk, pl, rg, al)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_15_attack_authorized(self):
        root, md, wk, pl, rg, al = _sealed_transition(
            manifest_overrides={"attack_authorized": True})
        try:
            with self.assertRaises(TransitionRejected):
                _verify(root, md, wk, pl, rg, al)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_16_student_training_authorized(self):
        root, md, wk, pl, rg, al = _sealed_transition(
            manifest_overrides={"student_training_authorized": True})
        try:
            with self.assertRaises(TransitionRejected):
                _verify(root, md, wk, pl, rg, al)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_17_protected_payload_read(self):
        root, md, wk, pl, rg, al = _sealed_transition(
            manifest_overrides={"protected_payload_read": True})
        try:
            with self.assertRaises(TransitionRejected):
                _verify(root, md, wk, pl, rg, al)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_18_detector_load(self):
        root, md, wk, pl, rg, al = _sealed_transition(
            manifest_overrides={"detector_load_authorized": True})
        try:
            with self.assertRaises(TransitionRejected):
                _verify(root, md, wk, pl, rg, al)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_19_missing_clean_action_only(self):
        root, md, wk, pl, rg, al = _sealed_transition(
            manifest_overrides={"clean_action_only": False})
        try:
            with self.assertRaises(TransitionRejected):
                _verify(root, md, wk, pl, rg, al)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_20_max_episodes_wrong(self):
        root, md, wk, pl, rg, al = _sealed_transition(
            manifest_overrides={"max_episodes": 100})
        try:
            with self.assertRaises(TransitionRejected):
                _verify(root, md, wk, pl, rg, al)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_21_unauthorized_gpu(self):
        root, md, wk, pl, rg, al = _sealed_transition()
        try:
            with self.assertRaises(TransitionRejected):
                _verify(root, md, wk, pl, rg, al, gpu=99)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_22_wrong_output_root(self):
        root, md, wk, pl, rg, al = _sealed_transition()
        try:
            with self.assertRaises(TransitionRejected):
                _verify(root, md, wk, pl, rg, al, output="/wrong/path")
        finally:
            shutil.rmtree(root, ignore_errors=True)


class TestTransitionPositive(unittest.TestCase):
    def test_23_valid_transition_passes(self):
        """Positive path: valid transition receipt with real model dir."""
        # Create registry FIRST so we can bind its SHA into the manifest
        reg_dir = Path(tempfile.mkdtemp(prefix="test_registry_"))
        per_task = reg_dir / "per_task"
        per_task.mkdir()
        (reg_dir / "ENTITY_REGISTRY_V2_SUMMARY.json").write_text(
            json.dumps({"status": "PASS"}))
        registry_sha = sha256_file(reg_dir / "ENTITY_REGISTRY_V2_SUMMARY.json")

        root, md, wk, pl, rg, al = _sealed_transition(
            manifest_overrides={"registry_summary_sha256": registry_sha})
        try:
            result = _verify(root, md, wk, pl, str(per_task), al)
            self.assertIsNotNone(result)
            self.assertEqual(result["gate"], "FIT-INFERENCE_TRANSITION")
        finally:
            shutil.rmtree(root, ignore_errors=True)
            shutil.rmtree(md, ignore_errors=True)
            shutil.rmtree(reg_dir, ignore_errors=True)
            os.remove(wk); os.remove(pl); os.remove(rg); os.remove(al)


class TestIdentityAllowlist(unittest.TestCase):
    def test_24_pilot_allowlist_mismatch_rejected(self):
        """Allowlist that doesn't match pilot rebuild must be rejected."""
        root = Path(tempfile.mkdtemp())
        pilot = _make_temp_file(_make_pilot())
        # Create an allowlist with wrong task_id
        wrong_ids = []
        for suite in ["libero_10", "libero_goal", "libero_object", "libero_spatial"]:
            for tid in range(10):
                wrong_ids.append({
                    "episode_id": f"{suite}/task_{tid:02d}/state_0",
                    "suite": suite, "task_id": tid, "state_id": 0,
                    "collection_seed": 20260717,
                    "initial_state_sha256": "X" * 64,  # WRONG
                })
        (root / "IDENTITY_ALLOWLIST.json").write_text(json.dumps(
            {"gate": "FIT-INFERENCE_IDENTITY_ALLOWLIST", "identities": wrong_ids}))
        with self.assertRaises(TransitionRejected):
            validate_identity_allowlist(root / "IDENTITY_ALLOWLIST.json", pilot)
        shutil.rmtree(root, ignore_errors=True)

    def test_25_missing_seed_rejected(self):
        """Missing collection_seed in pilot must be rejected."""
        pilot_data = json.loads(_make_pilot())
        del pilot_data["records"][0]["collection_seed"]
        pilot = _make_temp_file(json.dumps(pilot_data))
        root = Path(tempfile.mkdtemp())
        with self.assertRaises(TransitionRejected):
            validate_identity_allowlist(root / "nonexistent", pilot)
        shutil.rmtree(root, ignore_errors=True)
        os.remove(pilot)


class TestModelTreeFingerprint(unittest.TestCase):
    def test_26_different_model_trees_differ(self):
        d1 = _make_model_dir()
        d2 = _make_model_dir()
        (d2 / "extra_file.txt").write_text("different")
        fp1 = compute_model_tree_fingerprint(d1)
        fp2 = compute_model_tree_fingerprint(d2)
        self.assertNotEqual(fp1, fp2)
        shutil.rmtree(d1, ignore_errors=True)
        shutil.rmtree(d2, ignore_errors=True)

    def test_27_symlink_rejected(self):
        d = Path(tempfile.mkdtemp(prefix="test_symlink_"))
        (d / "real.txt").write_text("real")
        os.symlink(d / "real.txt", d / "link.txt")
        try:
            with self.assertRaises(TransitionRejected):
                compute_model_tree_fingerprint(d)
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
