# Label V2 Implementation-Only Authorization V1

Status: PLANNING_ONLY_AUTHORIZATION_RECORD

```text
GATE_A1_LABEL_SPEC = PASS
LABEL_V2_IMPLEMENTATION_AUTHORIZATION = AUTHORIZED_CPU_ONLY
LABEL_V2_BUILD_EXECUTION_AUTHORIZATION = NOT_AUTHORIZED
EXPERIMENT_AUTHORIZATION_STATUS = NOT_AUTHORIZED
```

| Field | Value |
|---|---|
| authorization_basis_commit_sha | `8c8226959a7007968568f22d69da2c80e0abfcf0` |
| input_manifest_path | `tables/server_freeze/clean2000_teacher_source_availability.csv` |
| input_manifest_git_blob_sha1 | `22d54409bb01db489d5b2edc0640efafcb6a6408` |
| input_manifest_repo_content_sha256_lf | `268ec095aae19a5aca62141b162c0719706b885c96c84122174fe425493426e4` |
| source_roots | no CLEAN2000 source access under this authorization |
| output_root | synthetic fixture output under the test temp directory only |
| allowed_commands | create `tools/multisuite_detector/build_clean2000_label_v2.py`; run `--help`; run synthetic fixture `--dry-run`; local CPU pytest/lint |
| cpu_limit | local CPU only |
| gpu_limit | 0 GPUs |
| maximum_jobs | 1 local process |
| maximum_runtime | 30 minutes |
| maximum_storage | 100 MB synthetic/test output |
| retry_eligibility | unrestricted on synthetic fixtures only |
| terminal_failure_rule | any attempt to read CLEAN2000 source or write formal Label V2 artifact invalidates this authorization |
| abort_conditions | CLEAN2000 live/backup reads, formal 2000-row output, OpenVLA inference, detector training, rollout, attack code invocation, GPU allocation |
| authorization_expiry | next protocol revision or 2026-07-10, whichever comes first |
| authorization_scope | builder implementation, unit tests, synthetic fixture dry-run, manual audit sampler planning |

Not authorized: reading CLEAN2000 live or backup source, generating formal
2000-row Label V2 artifact, detector training, OpenVLA inference, simulator
rollout, attack execution, GPU jobs, source artifact mutation.

Build-execution authorization must be a later record binding builder code
commit, builder file SHA256, exact source host:path, exact output host:path,
exact command, and validator command.
