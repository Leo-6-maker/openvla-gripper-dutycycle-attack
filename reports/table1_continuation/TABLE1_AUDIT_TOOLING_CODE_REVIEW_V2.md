# Table 1 Audit Tooling Code Review V2

Status: `CODE_HARDENING_IN_PR`

Scope: local GitHub-side audit tooling only. No server artifact was accessed and no live validator, freeze bundle, formal TRUE_T10 manifest, rollout, or attack was launched.

| Risk | Severity | Existing behavior | Required behavior | Test that proves the fix |
|---|---|---|---|---|
| stale validator report binding | P0 | Builder trusted `closure_pass=True`. | Builder rechecks manifest, root identity, and every contract SHA. | `test_validator_report_belongs_to_another_manifest_rejected` |
| manifest changed after validation | P0 | Builder did not compare actual manifest SHA to report. | Refuse when actual manifest SHA differs. | `test_manifest_modified_after_validation_rejected` |
| condition-root mismatch | P0 | Builder accepted a different root. | Refuse when condition root identity differs. | `test_condition_root_differs_from_validator_report_rejected` |
| hard-coded terminal-invalid policy | P0 | Validator used Python constant for legal invalid statuses. | Legal terminal-invalid states come only from retry-policy contract. | `test_legal_terminal_invalid_without_policy_evidence_fails` |
| missing job-key uniqueness | P0 | Accepted keys could duplicate. | Empty or duplicate `job_key` is HOLD. | `test_duplicate_job_key_fails`, `test_empty_job_key_fails` |
| missing exact seed-domain validation | P0 | Counts could pass with wrong folds/seeds. | Fold/state/detector/perturbation domains must match state-selection exactly. | `test_wrong_exact_fold_set_fails`, `test_wrong_seed_values_fails`, `test_three_replicates_with_wrong_ids_fails` |
| output path escape | P0 | `output_dir` could resolve outside root. | All output dirs must resolve inside condition root and not equal it. | `test_output_dir_outside_condition_root_fails` |
| symlink escape | P0 | Symlinked dirs/files were not rejected. | Any path component or artifact symlink is HOLD. | `test_output_dir_via_symlink_escape_fails` |
| missing required artifact enforcement | P0 | `episode_summary.json` alone was enough. | Required artifacts are contract-driven per outcome. | `test_missing_required_telemetry_fails` |
| missing provenance field enforcement | P0 | Missing provenance fields could pass if values looked uniform. | Required provenance fields must be present and SHA-formatted. | `test_missing_required_provenance_field_fails` |
| checkpoint/global-freeze mismatch | P0 | Checkpoint was treated like a global scalar. | Checkpoint identity is matched by Global Freeze mapping. | `test_checkpoint_does_not_match_global_freeze_fails` |
| absolute-path checksums | P1 | Artifact checksum file used absolute paths. | Checksums use paths relative to condition root. | `test_nested_artifact_is_included_and_relative` |
| non-recursive artifact inventory | P1 | Builder only globbed one level. | Inventory is recursive. | `test_nested_artifact_is_included_and_relative` |
| premature FROZEN status | P0 | Builder wrote `status=FROZEN`. | Builder writes `FREEZE_CANDIDATE`; verifier/finalize creates final frozen record. | `test_candidate_bundle_cannot_claim_frozen` |
| TRUE_T10 clean-outcome field leakage | P0 | Generator copied `dict(clean_row)`. | Generator uses an allowlist and denylist guard. | `test_authorized_synthetic_condition_produces_deterministic_manifest` |
| hard-coded objective/K before prereg freeze | P0 | Objective, K, and no-emission policy were literals. | Values must come from authorized condition spec. | `test_condition_spec_change_changes_manifest_digest` |
| incomplete runtime hash binding | P0 | TRUE_T10 took only a few CLI hashes. | Condition spec binds runner/worker/bridge/protocol/metric/checkpoint/state/retry hashes. | `test_non_hex_64_character_sha_rejected` |
| invalid SHA256 format acceptance | P0 | Non-hex 64-character values could pass. | SHA fields must be 64 lowercase hex. | `test_non_hex_64_character_sha_rejected` |
| shared output-root risk | P0 | Output root reuse was only partially checked. | Output root must be absolute, under allowed root, unused, not inside bundle, and not registry-occupied. | `test_shared_output_root_rejected`, `test_relative_output_root_rejected` |
