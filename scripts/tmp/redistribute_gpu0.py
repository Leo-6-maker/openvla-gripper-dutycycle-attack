import json, sys

manifest = "/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1/TRUE_T10/launch/manifest_gpu0_w0.jsonl"
retry_dir = "/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1/TRUE_T10/launch/retry_gpu3"

with open(manifest) as f:
    all_jobs = [json.loads(line) for line in f]

# Jobs 1-4 completed (0-indexed: 0-3), remaining: 4-17
remaining = all_jobs[4:]

# Read existing retry manifests
retry_jobs = []
for wi in [0, 1, 2]:
    with open(f"{retry_dir}/manifest_gpu3_w{wi}.jsonl") as f:
        retry_jobs.extend([json.loads(line) for line in f])

combined = retry_jobs + remaining

# Round-robin across 3 workers
for wi in range(3):
    split = combined[wi::3]
    path = f"{retry_dir}/manifest_gpu3_w{wi}.jsonl"
    with open(path, "w") as f:
        for j in split:
            f.write(json.dumps(j) + "\n")
    print(f"w{wi}: {len(split)} jobs -> {path}")

print(f"GPU0 remaining: {len(remaining)}")
print(f"Retry existing: {len(retry_jobs)}")
print(f"Combined total: {len(combined)}")
