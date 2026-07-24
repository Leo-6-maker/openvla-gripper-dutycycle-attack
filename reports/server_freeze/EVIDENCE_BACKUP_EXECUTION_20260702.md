# Evidence Backup Execution 2026-07-02

Backup target: `/data/liuyu/openvla_gripper_freeze/20260702_codex_verified_v3`.

Transfer notes:

- `v1` partial: interrupted slow `scp -3 -r` attempt; preserved and not used for PASS.
- `v2` partial: interrupted failed tar-stream attempt; preserved and not used for PASS.
- `v3`: verified backup target used for final status.

Verification summary:

```json
{
  "clean2000": {
    "destination_file_count": 16,
    "destination_total_bytes": 579585829,
    "relative_path_set_equal": true,
    "sha256_all_match": true,
    "source_file_count": 16,
    "source_total_bytes": 579585829,
    "status": "PASS"
  },
  "object": {
    "destination_file_count": 7777,
    "destination_total_bytes": 504814688,
    "relative_path_set_equal": true,
    "sha256_all_match": true,
    "source_file_count": 7777,
    "source_total_bytes": 504814688,
    "status": "PASS"
  },
  "runtime_provenance": {
    "destination_file_count": 33,
    "destination_total_bytes": 6272266,
    "relative_path_set_equal": true,
    "sha256_all_match": true,
    "source_file_count": 33,
    "source_total_bytes": 6272266,
    "status": "PASS"
  },
  "source_quiescence": {
    "after_rows": 7793,
    "before_rows": 7793,
    "status": "PASS"
  },
  "target_root": "/data/liuyu/openvla_gripper_freeze/20260702_codex_verified_v3"
}
```

PASS criteria applied:

- source file count equals destination file count;
- source total bytes equals destination total bytes;
- relative path set equal;
- every destination SHA256 equals source/staging SHA256;
- source Object/CLEAN2000 before and after manifests are identical.

Final status:

- OBJECT_BACKUP = PASS
- CLEAN2000_BACKUP = PASS
- RUNTIME_PROVENANCE_BACKUP = PASS
- BACKUP_VERIFIED = PASS

No source scientific artifact was modified.
