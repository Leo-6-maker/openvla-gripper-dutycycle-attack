from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.table1_audit.build_condition_freeze_bundle import build as build_bundle
from tools.table1_audit.build_true_t10_manifest import build as build_true_t10
from tools.table1_audit.adapt_authoritative_artifacts import adapt_global_freeze, adapt_metric_schema, adapt_state_selection
from tools.table1_audit.common import canonical_json, load_json, load_jsonl, sha256_file, write_json, write_jsonl
from tools.table1_audit.validate_batch_a_registry import validate as validate_registry
from tools.table1_audit.validate_formal_clean_closure import validate
from tools.table1_audit.verify_condition_freeze_bundle import finalize, verify as verify_bundle, verify_final
from tools.table1_audit.verify_server_runtime_snapshot import verify as verify_snapshot
from tools.table1_audit.validate_worker_runtime_binding import evaluate as evaluate_worker_binding


H = "a" * 64


def _write_contracts(root: Path):
    contract = root / "contracts"
    contract.mkdir(parents=True)
    folds = [f"{i:02d}" for i in range(1, 10)]
    state = {
        "schema_version": "state_selection.v1",
        "folds": folds,
        "states_by_fold": {f: ["0", "1"] for f in folds},
        "tasks_by_fold": {f: [f"task_{f}"] for f in folds},
        "detector_seeds": [1, 2, 3],
        "perturbation_seeds": [0, 1, 2],
    }
    global_freeze = {
        "schema_version": "global_freeze.v1",
        "victim_checkpoint_sha256": {f"{f}|{d}": f"{int(f):02d}{d}" + "b" * 61 for f in folds for d in [1, 2, 3]},
        "detector_checkpoint_sha256": {f"{f}|{d}": f"{int(f):02d}{d}" + "c" * 61 for f in folds for d in [1, 2, 3]},
    }
    runtime = {
        "schema_version": "runtime_lock.v1",
        "required_fields": ["runner_sha256", "worker_sha256", "bridge_sha256", "protocol_sha256", "metric_schema_sha256"],
        "required_sha256": {
            "runner_sha256": "1" * 64,
            "worker_sha256": "2" * 64,
            "bridge_sha256": "3" * 64,
            "protocol_sha256": "4" * 64,
            "metric_schema_sha256": "5" * 64,
        },
    }
    retry = {
        "schema_version": "retry_policy.v1",
        "legal_terminal_invalid_statuses": ["SCIENTIFIC_INVALID"],
        "terminal_reasons": ["MAX_RETRY_EXHAUSTED"],
        "max_attempts": 3,
    }
    artifacts = {
        "schema_version": "required_artifact_schema.v1",
        "complete": {"required_files": ["episode_summary.json", "step_telemetry.csv"]},
        "terminal_invalid": {"required_files": ["terminal_ledger.json"]},
    }
    paths = {}
    for name, data in [("state", state), ("freeze", global_freeze), ("runtime", runtime), ("retry", retry), ("artifacts", artifacts)]:
        p = contract / f"{name}.json"
        write_json(p, data)
        paths[name] = p
    return paths, state, global_freeze, runtime


def make_clean_tree(root: Path, *, mutate=None):
    paths, state, global_freeze, runtime = _write_contracts(root)
    condition = root / "CLEAN"
    rows = []
    for fold in state["folds"]:
        for state_id in ["0", "1"]:
            for det in [1, 2, 3]:
                for pert in [0, 1, 2]:
                    out = condition / f"fold_{fold}" / f"state_{state_id}" / f"det_{det}" / f"pert_{pert}"
                    out.mkdir(parents=True, exist_ok=True)
                    row = {
                        "job_key": f"f{fold}_s{state_id}_d{det}_p{pert}",
                        "fold": fold,
                        "task_id": f"task_{fold}",
                        "state_id": state_id,
                        "detector_seed": det,
                        "perturbation_seed": pert,
                        "condition_id": "CLEAN",
                        "checkpoint_sha256": global_freeze["victim_checkpoint_sha256"][f"{fold}|{det}"],
                        "detector_checkpoint_sha256": global_freeze["detector_checkpoint_sha256"][f"{fold}|{det}"],
                        "output_dir": str(out),
                    }
                    rows.append(row)
                    (out / "step_telemetry.csv").write_text("step\n0\n", encoding="utf-8")
    if mutate:
        mutate(rows, condition, paths)
    manifest = condition / "MANIFEST.jsonl"
    write_jsonl(manifest, rows)
    manifest_sha = sha256_file(manifest)
    for row in rows:
        out = Path(row["output_dir"])
        if (out / "episode_summary.json").exists() or (out / "terminal_ledger.json").exists():
            continue
        summary = {
            "task_success": True,
            "state_id": row["state_id"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "detector_checkpoint_sha256": row["detector_checkpoint_sha256"],
            "manifest_sha256": manifest_sha,
            "state_selection_sha256": sha256_file(paths["state"]),
            "global_freeze_sha256": sha256_file(paths["freeze"]),
            **runtime["required_sha256"],
        }
        write_json(out / "episode_summary.json", summary)
    return manifest, condition, paths


def run_validate(manifest: Path, condition: Path, paths: dict):
    return validate(SimpleNamespace(
        manifest=manifest,
        condition_root=condition,
        state_selection=paths["state"],
        global_freeze=paths["freeze"],
        runtime_lock=paths["runtime"],
        retry_policy=paths["retry"],
        required_artifact_schema=paths["artifacts"],
        expected_rows=162,
        expected_parents=54,
        expected_replicates=3,
    ))


def make_bundle(root: Path):
    manifest, condition, paths = make_clean_tree(root)
    validation = run_validate(manifest, condition, paths)
    validation_json = root / "validation.json"
    write_json(validation_json, validation)
    freeze_root = root / "freezes"
    dest = freeze_root / "clean_candidate"
    args = SimpleNamespace(
        validator_json=validation_json,
        manifest=manifest,
        condition_root=condition,
        state_selection=paths["state"],
        global_freeze=paths["freeze"],
        runtime_lock=paths["runtime"],
        retry_policy=paths["retry"],
        required_artifact_schema=paths["artifacts"],
        freeze_root=freeze_root,
        dest=dest,
        condition_id="CLEAN",
        dry_run=False,
    )
    build_bundle(args)
    verification_json = root / "verification.json"
    verification = verify_bundle(SimpleNamespace(bundle=dest))
    write_json(verification_json, verification)
    final_dir = root / "final" / "clean"
    finalize(SimpleNamespace(bundle=dest, bundle_verification=verification_json, final_dir=final_dir))
    final_verification_json = root / "final_verification.json"
    write_json(final_verification_json, verify_final(SimpleNamespace(bundle=final_dir)))
    return final_dir, final_verification_json, paths


class FormalCleanValidatorTests(unittest.TestCase):
    def assertProblem(self, result, name):
        self.assertIn(name, {p["class"] for p in result["problems"]})

    def test_valid_162_rows_54_parents_3_replicates(self):
        with tempfile.TemporaryDirectory() as d:
            manifest, condition, paths = make_clean_tree(Path(d))
            result = run_validate(manifest, condition, paths)
            self.assertTrue(result["closure_pass"])
            self.assertEqual(result["accepted_count"], 162)

    def test_duplicate_job_key_fails(self):
        def mutate(rows, *_):
            rows[1]["job_key"] = rows[0]["job_key"]
        with tempfile.TemporaryDirectory() as d:
            result = run_validate(*make_clean_tree(Path(d), mutate=mutate))
            self.assertFalse(result["closure_pass"])
            self.assertProblem(result, "duplicate_job_key")

    def test_empty_job_key_fails(self):
        def mutate(rows, *_):
            rows[0]["job_key"] = ""
        with tempfile.TemporaryDirectory() as d:
            result = run_validate(*make_clean_tree(Path(d), mutate=mutate))
            self.assertProblem(result, "empty_job_key")

    def test_missing_required_manifest_field_fails(self):
        def mutate(rows, *_):
            rows[0].pop("task_id")
        with tempfile.TemporaryDirectory() as d:
            result = run_validate(*make_clean_tree(Path(d), mutate=mutate))
            self.assertProblem(result, "missing_required_manifest_field")

    def test_wrong_exact_fold_set_fails(self):
        def mutate(rows, *_):
            rows[0]["fold"] = "99"
        with tempfile.TemporaryDirectory() as d:
            result = run_validate(*make_clean_tree(Path(d), mutate=mutate))
            self.assertProblem(result, "wrong_fold_set")

    def test_wrong_seed_values_fails(self):
        def mutate(rows, *_):
            rows[0]["detector_seed"] = 9
        with tempfile.TemporaryDirectory() as d:
            result = run_validate(*make_clean_tree(Path(d), mutate=mutate))
            self.assertProblem(result, "wrong_detector_seed_domain")

    def test_three_replicates_with_wrong_ids_fails(self):
        def mutate(rows, *_):
            rows[0]["perturbation_seed"] = 99
        with tempfile.TemporaryDirectory() as d:
            result = run_validate(*make_clean_tree(Path(d), mutate=mutate))
            self.assertProblem(result, "wrong_perturbation_seed_domain")
            self.assertProblem(result, "replicate_count")

    def test_output_dir_outside_condition_root_fails(self):
        def mutate(rows, condition, *_):
            outside = condition.parent / "outside"
            outside.mkdir()
            rows[0]["output_dir"] = str(outside)
        with tempfile.TemporaryDirectory() as d:
            result = run_validate(*make_clean_tree(Path(d), mutate=mutate))
            self.assertProblem(result, "unsafe_output_dir")

    def test_output_dir_via_symlink_escape_fails(self):
        def mutate(rows, condition, *_):
            outside = condition.parent / "outside"
            outside.mkdir()
            link = condition / "link"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except OSError as exc:
                raise unittest.SkipTest(f"symlink unavailable: {exc}")
            rows[0]["output_dir"] = str(link)
        with tempfile.TemporaryDirectory() as d:
            result = run_validate(*make_clean_tree(Path(d), mutate=mutate))
            self.assertProblem(result, "unsafe_output_dir")

    def test_missing_required_telemetry_fails(self):
        def mutate(rows, *_):
            Path(rows[0]["output_dir"], "step_telemetry.csv").unlink()
        with tempfile.TemporaryDirectory() as d:
            result = run_validate(*make_clean_tree(Path(d), mutate=mutate))
            self.assertProblem(result, "missing_required_artifact")

    def test_manifest_sha_recorded_consistently_but_wrong_fails(self):
        def mutate(rows, condition, *_):
            pass
        with tempfile.TemporaryDirectory() as d:
            manifest, condition, paths = make_clean_tree(Path(d), mutate=mutate)
            for p in condition.rglob("episode_summary.json"):
                data = load_json(p)
                data["manifest_sha256"] = "9" * 64
                write_json(p, data)
            result = run_validate(manifest, condition, paths)
            self.assertProblem(result, "manifest_sha_mismatch")

    def test_missing_required_provenance_field_fails(self):
        with tempfile.TemporaryDirectory() as d:
            manifest, condition, paths = make_clean_tree(Path(d))
            p = next(condition.rglob("episode_summary.json"))
            data = load_json(p)
            data.pop("runner_sha256")
            write_json(p, data)
            result = run_validate(manifest, condition, paths)
            self.assertProblem(result, "missing_required_provenance_field")

    def test_observed_runtime_sha_does_not_fallback_to_manifest_expected(self):
        with tempfile.TemporaryDirectory() as d:
            manifest, condition, paths = make_clean_tree(Path(d))
            rows = load_jsonl(manifest)
            rows[0]["runner_sha256"] = "1" * 64
            write_jsonl(manifest, rows)
            p = Path(rows[0]["output_dir"], "episode_summary.json")
            data = load_json(p)
            data.pop("runner_sha256")
            write_json(p, data)
            result = run_validate(manifest, condition, paths)
            self.assertProblem(result, "missing_required_provenance_field")

    def test_checkpoint_does_not_match_global_freeze_fails(self):
        with tempfile.TemporaryDirectory() as d:
            manifest, condition, paths = make_clean_tree(Path(d))
            p = next(condition.rglob("episode_summary.json"))
            data = load_json(p)
            data["checkpoint_sha256"] = "8" * 64
            write_json(p, data)
            result = run_validate(manifest, condition, paths)
            self.assertProblem(result, "global_freeze_checkpoint_mismatch")

    def test_global_freeze_zero_and_multiple_matches_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, condition, paths = make_clean_tree(root)
            freeze = load_json(paths["freeze"])
            freeze["victim_checkpoint_sha256"].pop("01|1")
            freeze["detector_checkpoint_sha256"]["01"] = "7" * 64
            write_json(paths["freeze"], freeze)
            result = run_validate(manifest, condition, paths)
            self.assertProblem(result, "global_freeze_checkpoint_match_count")

    def test_attempt_zero_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            manifest, condition, paths = make_clean_tree(Path(d))
            rows = load_jsonl(manifest)
            rows[0]["attempt"] = 0
            write_jsonl(manifest, rows)
            result = run_validate(manifest, condition, paths)
            self.assertEqual(result["rows"][0]["retry_attempt"], 0)

    def test_legal_terminal_invalid_without_policy_evidence_fails(self):
        def mutate(rows, *_):
            out = Path(rows[0]["output_dir"])
            write_json(out / "episode_summary.json", {"terminal_status": "SCIENTIFIC_INVALID"})
        with tempfile.TemporaryDirectory() as d:
            result = run_validate(*make_clean_tree(Path(d), mutate=mutate))
            self.assertProblem(result, "terminal_ledger_job_key_mismatch")

    def test_retry_duplicate_attempt_fails(self):
        def mutate(rows, *_):
            out = Path(rows[0]["output_dir"])
            write_json(out / "terminal_ledger.json", {
                "job_key": rows[0]["job_key"],
                "terminal_status": "SCIENTIFIC_INVALID",
                "terminal_reason": "MAX_RETRY_EXHAUSTED",
                "no_retry_remaining": True,
                "attempt_history": [
                    {"attempt": 0, "accepted": False, "quarantined": True},
                    {"attempt": 0, "accepted": True},
                ],
            })
        with tempfile.TemporaryDirectory() as d:
            result = run_validate(*make_clean_tree(Path(d), mutate=mutate))
            self.assertProblem(result, "duplicate_retry_attempt")

    def test_malformed_manifest_row_does_not_crash_validator(self):
        with tempfile.TemporaryDirectory() as d:
            manifest, condition, paths = make_clean_tree(Path(d))
            manifest.write_text(manifest.read_text(encoding="utf-8") + "{bad\n", encoding="utf-8")
            result = run_validate(manifest, condition, paths)
            self.assertProblem(result, "malformed_manifest_row")


class FreezeBuilderVerifierTests(unittest.TestCase):
    def test_freeze_builder_and_verifier_candidate_flow(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, condition, paths = make_clean_tree(root)
            validation = run_validate(manifest, condition, paths)
            vjson = root / "validation.json"
            write_json(vjson, validation)
            dest = root / "freezes" / "candidate"
            build_bundle(SimpleNamespace(validator_json=vjson, manifest=manifest, condition_root=condition, state_selection=paths["state"], global_freeze=paths["freeze"], runtime_lock=paths["runtime"], retry_policy=paths["retry"], required_artifact_schema=paths["artifacts"], freeze_root=root / "freezes", dest=dest, condition_id="CLEAN", dry_run=False))
            self.assertEqual(load_json(dest / "CONDITION_FREEZE.json")["status"], "FREEZE_CANDIDATE")
            self.assertTrue(verify_bundle(SimpleNamespace(bundle=dest))["verification_pass"])
            self.assertNotIn(str(condition), (dest / "ARTIFACT_SHA256SUMS.txt").read_text(encoding="utf-8"))

    def test_validator_report_belongs_to_another_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            m1, c1, p1 = make_clean_tree(root / "a")
            m2, c2, p2 = make_clean_tree(root / "b")
            vjson = root / "v.json"
            write_json(vjson, run_validate(m1, c1, p1))
            with self.assertRaises(SystemExit):
                build_bundle(SimpleNamespace(validator_json=vjson, manifest=m2, condition_root=c2, state_selection=p2["state"], global_freeze=p2["freeze"], runtime_lock=p2["runtime"], retry_policy=p2["retry"], required_artifact_schema=p2["artifacts"], freeze_root=root / "freezes", dest=root / "freezes" / "x", condition_id="CLEAN", dry_run=False))

    def test_condition_root_differs_from_validator_report_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, condition, paths = make_clean_tree(root)
            other = root / "other"
            other.mkdir()
            vjson = root / "v.json"
            write_json(vjson, run_validate(manifest, condition, paths))
            with self.assertRaises(SystemExit):
                build_bundle(SimpleNamespace(validator_json=vjson, manifest=manifest, condition_root=other, state_selection=paths["state"], global_freeze=paths["freeze"], runtime_lock=paths["runtime"], retry_policy=paths["retry"], required_artifact_schema=paths["artifacts"], freeze_root=root / "freezes", dest=root / "freezes" / "x", condition_id="CLEAN", dry_run=False))

    def test_accepted_job_key_count_below_162_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, condition, paths = make_clean_tree(root)
            validation = run_validate(manifest, condition, paths)
            validation["accepted_job_keys"] = validation["accepted_job_keys"][:-1]
            vjson = root / "v.json"
            write_json(vjson, validation)
            with self.assertRaises(SystemExit):
                build_bundle(SimpleNamespace(validator_json=vjson, manifest=manifest, condition_root=condition, state_selection=paths["state"], global_freeze=paths["freeze"], runtime_lock=paths["runtime"], retry_policy=paths["retry"], required_artifact_schema=paths["artifacts"], freeze_root=root / "freezes", dest=root / "freezes" / "x", condition_id="CLEAN", dry_run=False))

    def test_manifest_modified_after_validation_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, condition, paths = make_clean_tree(root)
            vjson = root / "v.json"
            write_json(vjson, run_validate(manifest, condition, paths))
            manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                build_bundle(SimpleNamespace(validator_json=vjson, manifest=manifest, condition_root=condition, state_selection=paths["state"], global_freeze=paths["freeze"], runtime_lock=paths["runtime"], retry_policy=paths["retry"], required_artifact_schema=paths["artifacts"], freeze_root=root / "freezes", dest=root / "freezes" / "x", condition_id="CLEAN", dry_run=False))

    def test_source_artifact_modified_after_validation_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, condition, paths = make_clean_tree(root)
            vjson = root / "v.json"
            write_json(vjson, run_validate(manifest, condition, paths))
            next(condition.rglob("step_telemetry.csv")).write_text("step\n0\n1\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                build_bundle(SimpleNamespace(validator_json=vjson, manifest=manifest, condition_root=condition, state_selection=paths["state"], global_freeze=paths["freeze"], runtime_lock=paths["runtime"], retry_policy=paths["retry"], required_artifact_schema=paths["artifacts"], freeze_root=root / "freezes", dest=root / "freezes" / "x", condition_id="CLEAN", dry_run=False))

    def test_destination_exists_empty_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, condition, paths = make_clean_tree(root)
            vjson = root / "v.json"
            write_json(vjson, run_validate(manifest, condition, paths))
            dest = root / "freezes" / "x"
            dest.mkdir(parents=True)
            with self.assertRaises(SystemExit):
                build_bundle(SimpleNamespace(validator_json=vjson, manifest=manifest, condition_root=condition, state_selection=paths["state"], global_freeze=paths["freeze"], runtime_lock=paths["runtime"], retry_policy=paths["retry"], required_artifact_schema=paths["artifacts"], freeze_root=root / "freezes", dest=dest, condition_id="CLEAN", dry_run=False))

    def test_destination_outside_freeze_root_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, condition, paths = make_clean_tree(root)
            vjson = root / "v.json"
            write_json(vjson, run_validate(manifest, condition, paths))
            with self.assertRaises(SystemExit):
                build_bundle(SimpleNamespace(validator_json=vjson, manifest=manifest, condition_root=condition, state_selection=paths["state"], global_freeze=paths["freeze"], runtime_lock=paths["runtime"], retry_policy=paths["retry"], required_artifact_schema=paths["artifacts"], freeze_root=root / "freezes", dest=root / "outside", condition_id="CLEAN", dry_run=False))

    def test_nested_artifact_is_included_and_relative(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, condition, paths = make_clean_tree(root)
            vjson = root / "v.json"
            write_json(vjson, run_validate(manifest, condition, paths))
            dest = root / "freezes" / "candidate"
            build_bundle(SimpleNamespace(validator_json=vjson, manifest=manifest, condition_root=condition, state_selection=paths["state"], global_freeze=paths["freeze"], runtime_lock=paths["runtime"], retry_policy=paths["retry"], required_artifact_schema=paths["artifacts"], freeze_root=root / "freezes", dest=dest, condition_id="CLEAN", dry_run=False))
            sums = (dest / "ARTIFACT_SHA256SUMS.txt").read_text(encoding="utf-8")
            self.assertIn("fold_01/state_0/det_1/pert_0/episode_summary.json", sums)
            self.assertNotIn(str(condition), sums)

    def test_candidate_bundle_cannot_claim_frozen(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, condition, paths = make_clean_tree(root)
            vjson = root / "v.json"
            write_json(vjson, run_validate(manifest, condition, paths))
            dest = root / "freezes" / "candidate"
            build_bundle(SimpleNamespace(validator_json=vjson, manifest=manifest, condition_root=condition, state_selection=paths["state"], global_freeze=paths["freeze"], runtime_lock=paths["runtime"], retry_policy=paths["retry"], required_artifact_schema=paths["artifacts"], freeze_root=root / "freezes", dest=dest, condition_id="CLEAN", dry_run=False))
            data = load_json(dest / "CONDITION_FREEZE.json")
            data["status"] = "FROZEN"
            write_json(dest / "CONDITION_FREEZE.json", data)
            result = verify_bundle(SimpleNamespace(bundle=dest))
            self.assertIn("candidate_bundle_claims_frozen", {p["class"] for p in result["problems"]})

    def test_bundle_checksum_tampering_detected(self):
        with tempfile.TemporaryDirectory() as d:
            final_dir, _, _ = make_bundle(Path(d))
            candidate = Path(load_json(final_dir / "CONDITION_FREEZE_FINAL.json")["candidate_bundle"])
            (candidate / "accepted_job_keys.txt").write_text("tampered\n", encoding="utf-8")
            result = verify_bundle(SimpleNamespace(bundle=candidate))
            self.assertFalse(result["verification_pass"])
            self.assertIn("bundle_checksum_mismatch", {p["class"] for p in result["problems"]})


class TrueT10Tests(unittest.TestCase):
    def _spec(self, root: Path, paths: dict, *, status="AUTHORIZED_FOR_MANIFEST_GENERATION"):
        spec = {
            "schema_version": "condition_spec.v1",
            "status": status,
            "condition_id": "TRUE_T10",
            "allowed_output_root": str(root / "allowed"),
            "allowed_manifest_root": str(root / "manifests"),
            "clean_identity_allowlist": ["fold", "task_id", "state_id", "detector_seed", "perturbation_seed", "checkpoint_sha256", "detector_checkpoint_sha256"],
            "clean_result_denylist": ["task_success", "failure", "status", "terminal_status", "result_status", "reward", "done", "output_dir"],
            "bound_contract_sha256": {
                "runner_sha256": "1" * 64,
                "worker_sha256": "2" * 64,
                "bridge_sha256": "3" * 64,
                "protocol_sha256": "4" * 64,
                "metric_schema_sha256": "5" * 64,
                "victim_checkpoint_sha256": "6" * 64,
                "state_selection_sha256": sha256_file(paths["state"]),
                "retry_policy_sha256": sha256_file(paths["retry"]),
                "detector_global_freeze_sha256": sha256_file(paths["freeze"]),
            },
            "fields": {
                "folds": [f"{i:02d}" for i in range(1, 10)],
                "states_by_fold": {f"{i:02d}": ["0", "1"] for i in range(1, 10)},
                "detector_seeds": [1, 2, 3],
                "perturbation_seeds": [0, 1, 2],
                "attack": {
                    "objective_id": "prefix_log_ratio_open",
                    "objective_semantics_version": "v1",
                    "epsilon": "synthetic",
                    "epsilon_space": "processor_linf",
                    "step_size": "synthetic",
                    "optimization_steps": 10,
                    "initialization": "zero",
                    "K": 10,
                    "timing_policy": "Student trigger",
                    "arm_lock_mode": "PRESERVE_ARM_QPOS",
                    "preprocessing_backend": "synthetic",
                    "termination_policy": "fixed_steps",
                    "no_emission_policy": "ITT_RETAIN",
                },
            },
        }
        p = root / "spec.json"
        write_json(p, spec)
        return p

    def test_authorized_synthetic_condition_produces_deterministic_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bundle, verification, paths = make_bundle(root)
            spec = self._spec(root, paths)
            out_root = root / "allowed" / "true_t10"
            out = root / "manifests" / "manifest.jsonl"
            a = build_true_t10(SimpleNamespace(clean_bundle=bundle, bundle_verification=verification, authorized_condition_spec=spec, output_root=out_root, output_manifest=out, write=False))
            b = build_true_t10(SimpleNamespace(clean_bundle=bundle, bundle_verification=verification, authorized_condition_spec=spec, output_root=out_root, output_manifest=out, write=False))
            self.assertEqual(a["row_count"], 162)
            self.assertEqual(a["would_be_manifest_sha256"], b["would_be_manifest_sha256"])
            build_true_t10(SimpleNamespace(clean_bundle=bundle, bundle_verification=verification, authorized_condition_spec=spec, output_root=out_root, output_manifest=out, write=True))
            rows = load_jsonl(out)
            self.assertNotIn("task_success", rows[0])
            self.assertNotIn("output_dir", rows[0]["source_clean_job_key"])

    def test_draft_condition_spec_rejects_write(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bundle, verification, paths = make_bundle(root)
            spec = self._spec(root, paths, status="DRAFT_NOT_AUTHORIZED")
            with self.assertRaises(SystemExit):
                build_true_t10(SimpleNamespace(clean_bundle=bundle, bundle_verification=verification, authorized_condition_spec=spec, output_root=root / "allowed" / "x", output_manifest=root / "manifests" / "x.jsonl", write=True))

    def test_non_hex_64_character_sha_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bundle, verification, paths = make_bundle(root)
            spec = load_json(self._spec(root, paths))
            spec["bound_contract_sha256"]["runner_sha256"] = "g" * 64
            spec_path = root / "bad_spec.json"
            write_json(spec_path, spec)
            with self.assertRaises(ValueError):
                build_true_t10(SimpleNamespace(clean_bundle=bundle, bundle_verification=verification, authorized_condition_spec=spec_path, output_root=root / "allowed" / "x", output_manifest=root / "manifests" / "x.jsonl", write=False))

    def test_attack_spec_cannot_override_reserved_manifest_key(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bundle, verification, paths = make_bundle(root)
            spec_path = self._spec(root, paths)
            spec = load_json(spec_path)
            spec["fields"]["attack"]["job_key"] = "bad"
            write_json(spec_path, spec)
            with self.assertRaises(SystemExit):
                build_true_t10(SimpleNamespace(clean_bundle=bundle, bundle_verification=verification, authorized_condition_spec=spec_path, output_root=root / "allowed" / "x", output_manifest=root / "manifests" / "x.jsonl", write=False))

    def test_relative_output_root_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bundle, verification, paths = make_bundle(root)
            spec = self._spec(root, paths)
            with self.assertRaises(SystemExit):
                build_true_t10(SimpleNamespace(clean_bundle=bundle, bundle_verification=verification, authorized_condition_spec=spec, output_root=Path("relative"), output_manifest=root / "manifests" / "x.jsonl", write=False))

    def test_fake_frozen_json_without_verifier_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bundle = root / "bundle"
            bundle.mkdir()
            write_json(bundle / "CONDITION_FREEZE_FINAL.json", {"status": "FROZEN", "verifier_report_sha256": "0" * 64})
            verification = root / "verification.json"
            write_json(verification, {"verification_pass": True})
            spec = self._spec(root, _write_contracts(root)[0])
            with self.assertRaises(SystemExit):
                build_true_t10(SimpleNamespace(clean_bundle=bundle, bundle_verification=verification, authorized_condition_spec=spec, output_root=root / "allowed" / "x", output_manifest=root / "manifests" / "x.jsonl", write=False))

    def test_shared_output_root_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bundle, verification, paths = make_bundle(root)
            spec_path = self._spec(root, paths)
            spec = load_json(spec_path)
            out_root = root / "allowed" / "x"
            spec["occupied_output_roots"] = [str(out_root)]
            write_json(spec_path, spec)
            with self.assertRaises(SystemExit):
                build_true_t10(SimpleNamespace(clean_bundle=bundle, bundle_verification=verification, authorized_condition_spec=spec_path, output_root=out_root, output_manifest=root / "manifests" / "x.jsonl", write=False))

    def test_condition_spec_change_changes_manifest_digest(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bundle, verification, paths = make_bundle(root)
            spec_a = self._spec(root, paths)
            spec_data = load_json(spec_a)
            spec_data["fields"]["attack"]["K"] = 11
            spec_b = root / "spec_b.json"
            write_json(spec_b, spec_data)
            a = build_true_t10(SimpleNamespace(clean_bundle=bundle, bundle_verification=verification, authorized_condition_spec=spec_a, output_root=root / "allowed" / "a", output_manifest=root / "manifests" / "a.jsonl", write=False))
            b = build_true_t10(SimpleNamespace(clean_bundle=bundle, bundle_verification=verification, authorized_condition_spec=spec_b, output_root=root / "allowed" / "b", output_manifest=root / "manifests" / "b.jsonl", write=False))
            self.assertNotEqual(a["would_be_manifest_sha256"], b["would_be_manifest_sha256"])

    def test_reported_manifest_sha_is_actual_jsonl_bytes_sha(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bundle, verification, paths = make_bundle(root)
            spec = self._spec(root, paths)
            out = root / "manifests" / "manifest.jsonl"
            result = build_true_t10(SimpleNamespace(clean_bundle=bundle, bundle_verification=verification, authorized_condition_spec=spec, output_root=root / "allowed" / "x", output_manifest=out, write=True))
            self.assertEqual(result["would_be_manifest_sha256"], sha256_file(out))

    def test_authoritative_artifact_adapters_bind_source_sha(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            freeze = root / "freeze.json"
            protocol = root / "protocol.json"
            metric = root / "metric.json"
            write_json(freeze, {"gate": "G", "checkpoints": [{"fold": "01", "seed": 1, "sha256": "1" * 64}]})
            write_json(protocol, {"fold_matrix": {"01": {"test": 7}}, "training": {"seeds": [1, 2, 3]}})
            write_json(metric, {"gate": "M", "x": 1})
            self.assertEqual(adapt_global_freeze(freeze)["source_sha256"], sha256_file(freeze))
            self.assertEqual(adapt_state_selection(protocol)["source_sha256"], sha256_file(protocol))
            self.assertEqual(adapt_metric_schema(metric)["source_sha256"], sha256_file(metric))


class SnapshotAndRegistryTests(unittest.TestCase):
    def make_snapshot(self, root: Path):
        snap = root / "snap"
        gh = root / "gh"
        (snap / "scripts").mkdir(parents=True)
        (gh / "scripts").mkdir(parents=True)
        (snap / "scripts" / "worker.py").write_text("print('server')\n", encoding="utf-8")
        (gh / "scripts" / "worker.py").write_text("print('github')\n", encoding="utf-8")
        write_json(snap / "SNAPSHOT_METADATA.json", {"server_hostname_identifier": "h", "original_repo_path": "repo", "branch": "b", "HEAD": H, "dirty_status": "clean", "snapshot_utc_timestamp": "2026-06-29T00:00:00Z", "original_relative_paths": ["scripts/worker.py"], "snapshot_creator_command_version": "v1"})
        write_json(snap / "FILES.json", {"files": [{"relative_path": "scripts/worker.py"}]})
        (snap / "SHA256SUMS.txt").write_text(f"{sha256_file(snap / 'scripts' / 'worker.py')}  scripts/worker.py\n", encoding="utf-8")
        return snap, gh

    def test_snapshot_verifier_pass_and_diff(self):
        with tempfile.TemporaryDirectory() as d:
            snap, gh = self.make_snapshot(Path(d))
            result = verify_snapshot(SimpleNamespace(snapshot_root=snap, github_root=gh, max_file_bytes=1000000))
            self.assertTrue(result["verification_pass"])
            self.assertFalse(result["rows"][0]["byte_identical"])
            self.assertIn("unified_diff", result["rows"][0])

    def test_snapshot_extra_file_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            snap, gh = self.make_snapshot(Path(d))
            (snap / "extra.py").write_text("x\n", encoding="utf-8")
            result = verify_snapshot(SimpleNamespace(snapshot_root=snap, github_root=gh, max_file_bytes=1000000))
            self.assertFalse(result["verification_pass"])
            self.assertIn("snapshot_extra_file", {p["class"] for p in result["problems"]})

    def test_snapshot_checksum_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            snap, gh = self.make_snapshot(Path(d))
            (snap / "SHA256SUMS.txt").write_text(f"{'0' * 64}  scripts/worker.py\n", encoding="utf-8")
            result = verify_snapshot(SimpleNamespace(snapshot_root=snap, github_root=gh, max_file_bytes=1000000))
            self.assertIn("snapshot_checksum_mismatch", {p["class"] for p in result["problems"]})

    def test_snapshot_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            snap, gh = self.make_snapshot(Path(d))
            write_json(snap / "FILES.json", {"files": [{"relative_path": "../escape.py"}]})
            result = verify_snapshot(SimpleNamespace(snapshot_root=snap, github_root=gh, max_file_bytes=1000000))
            self.assertIn("snapshot_traversal", {p["class"] for p in result["problems"]})

    def test_registry_authorized_row_with_unresolved_field_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "registry.csv"
            p.write_text("condition_id,authorized,launch_gate,victim_checkpoint_sha256,detector_global_freeze_sha256,state_selection_sha256,protocol_sha256,runner_sha256,worker_sha256,bridge_sha256,metric_schema_sha256,retry_policy_sha256,condition_spec_sha256,manifest_sha256,arm_lock_mode,output_root\nX,true,GO,UNVERIFIED,{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},PRESERVE_ARM_QPOS,out\n".format("1" * 64), encoding="utf-8")
            result = validate_registry(p)
            self.assertFalse(result["validation_pass"])
            self.assertIn("authorized_unresolved_field", {p["class"] for p in result["problems"]})

    def test_registry_optional_arm_lock_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "registry.csv"
            p.write_text("condition_id,authorized,launch_gate,victim_checkpoint_sha256,detector_global_freeze_sha256,state_selection_sha256,protocol_sha256,runner_sha256,worker_sha256,bridge_sha256,metric_schema_sha256,retry_policy_sha256,condition_spec_sha256,manifest_sha256,arm_lock_mode,output_root\nX,false,HOLD,UNVERIFIED,UNVERIFIED,UNVERIFIED,UNVERIFIED,UNVERIFIED,UNVERIFIED,UNVERIFIED,UNVERIFIED,UNVERIFIED,UNVERIFIED,UNVERIFIED,optional,out\n", encoding="utf-8")
            result = validate_registry(p)
            self.assertIn("invalid_arm_lock", {p["class"] for p in result["problems"]})

    def test_registry_duplicate_output_roots_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "registry.csv"
            p.write_text("condition_id,authorized,launch_gate,victim_checkpoint_sha256,detector_global_freeze_sha256,state_selection_sha256,protocol_sha256,runner_sha256,worker_sha256,bridge_sha256,metric_schema_sha256,retry_policy_sha256,condition_spec_sha256,manifest_sha256,arm_lock_mode,output_root\nA,false,HOLD,UNVERIFIED,UNVERIFIED,UNVERIFIED,UNVERIFIED,UNVERIFIED,UNVERIFIED,UNVERIFIED,UNVERIFIED,UNVERIFIED,UNVERIFIED,UNVERIFIED,NO_ARM_LOCK,out\nB,false,HOLD,UNVERIFIED,UNVERIFIED,UNVERIFIED,UNVERIFIED,UNVERIFIED,UNVERIFIED,UNVERIFIED,UNVERIFIED,UNVERIFIED,UNVERIFIED,UNVERIFIED,NO_ARM_LOCK,out\n", encoding="utf-8")
            result = validate_registry(p)
            self.assertIn("output_root_overlap", {p["class"] for p in result["problems"]})


class WorkerRuntimeBindingTests(unittest.TestCase):
    SPEC = "4" * 64
    DISK = "e" * 64

    def row(self, job_key: str, sha: str) -> dict:
        return {
            "job_key": job_key,
            "attempt": 0,
            "pid": 123,
            "start_time": "2026-06-29T00:00:00Z",
            "actual_loaded_worker_sha": sha,
            "bridge_sha": "b" * 64,
            "manifest_sha": "c" * 64,
            "provenance_source": "episode_summary",
        }

    def test_spec_bound_worker_is_case_a(self):
        result = evaluate_worker_binding([self.row("a", self.SPEC)], spec_worker_sha=self.SPEC, disk_worker_sha=self.DISK, expected_jobs=1)
        self.assertEqual(result["worker_binding_status"], "CASE_A_SPEC_BOUND_WORKER")

    def test_disk_worker_without_equivalence_stays_p0_hold(self):
        result = evaluate_worker_binding([self.row("a", self.DISK)], spec_worker_sha=self.SPEC, disk_worker_sha=self.DISK, expected_jobs=1)
        self.assertEqual(result["worker_binding_status"], "RUNTIME_BINDING_P0_HOLD")
        self.assertIn("disk_worker_without_valid_row_equivalence", {p["class"] for p in result["problems"]})

    def test_disk_worker_with_equivalence_is_case_b_review(self):
        result = evaluate_worker_binding([self.row("a", self.DISK)], spec_worker_sha=self.SPEC, disk_worker_sha=self.DISK, expected_jobs=1, equivalence={"valid_row_equivalence_pass": True})
        self.assertEqual(result["worker_binding_status"], "CASE_B_POSTLAUNCH_RUNTIME_DEVIATION_REVIEW")

    def test_mixed_workers_quarantine(self):
        result = evaluate_worker_binding([self.row("a", self.SPEC), self.row("b", self.DISK)], spec_worker_sha=self.SPEC, disk_worker_sha=self.DISK, expected_jobs=2)
        self.assertEqual(result["worker_binding_status"], "VIS_RUNTIME_QUARANTINE_HOLD")


if __name__ == "__main__":
    unittest.main()
