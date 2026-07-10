#!/usr/bin/env bash
# Five-episode C2F Track A smoke. Do not use for full Goal/Object matrices.
set -uo pipefail

CODE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${A800_PY:-/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python}"
PARENT_CSV="${PARENT_CSV:-/mnt/sdc/dty_user/openvla_attack_evidence/condition_matrix/d7_table1_manifest/d7_table1_preregistered_parent_keys.csv}"
CHECKPOINT="${CHECKPOINT:-/mnt/sdc/dty_user/openvla_attack_evidence/c2f/siglip_full_final/train_D/c2f_rgb_lang_temporal_detector_v0.pt}"
GOAL_MODEL_MANIFEST="${GOAL_MODEL_MANIFEST:-$CODE_REPO/artifacts/goal_model_manifest.json}"

cd "$CODE_REPO" || exit 1
COMMIT=$(git rev-parse HEAD)
EXPECTED_GIT_COMMIT="${EXPECTED_GIT_COMMIT:-$COMMIT}"
if [ "$COMMIT" != "$EXPECTED_GIT_COMMIT" ]; then
  echo "Expected $EXPECTED_GIT_COMMIT, got $COMMIT" >&2
  exit 2
fi
if [ -n "$(git status --porcelain)" ]; then
  echo "Refusing smoke from dirty worktree" >&2
  exit 2
fi

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="${RUN_ROOT:-/mnt/sdc/dty_user/openvla_attack_evidence/c2f/c2f_track_a_smoke_${COMMIT:0:7}_${STAMP}}"
OUT="$RUN_ROOT/output"
mkdir -p "$OUT" "$RUN_ROOT/invalid_attempts"

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

"$VENV" - <<PY
import json, pathlib, sys
manifest = json.loads(pathlib.Path("$GOAL_MODEL_MANIFEST").read_text())
if manifest.get("status") != "PASS_C2F_GOAL_MODEL_INTEGRITY_AUDITED" or manifest.get("unnorm_key") != "libero_goal":
    raise SystemExit("Goal manifest not valid for smoke")
parents = [json.loads(json.dumps(r)) for r in __import__("csv").DictReader(pathlib.Path("$PARENT_CSV").open())]
goal = [p for p in parents if p.get("suite") == "libero_goal"][:2]
obj = [p for p in parents if p.get("suite") == "libero_object"][:1]
if len(goal) != 2 or len(obj) != 1:
    raise SystemExit("Not enough Goal/Object parents for smoke")
rows = [(p["parent_key"], "CLEAN") for p in goal]
rows += [(obj[0]["parent_key"], c) for c in ["CLEAN", "TRUE_CMDOPEN_T10_C2F", "RAND_ACTION_NOISE_T10_C2F"]]
root = pathlib.Path("$RUN_ROOT")
with (root / "parent_manifest.jsonl").open("w") as f:
    for p in goal + obj:
        f.write(json.dumps(p, sort_keys=True) + "\n")
with (root / "smoke_jobs.txt").open("w") as f:
    for parent, cond in rows:
        f.write(parent + "|" + cond + "\n")
print("smoke_jobs", len(rows))
PY

cat > "$RUN_ROOT/launch_env.json" <<JSON
{
  "protocol_name": "C2F_TRACK_A_CMDOPEN_ACTION_SPACE",
  "protocol_version": "2026-07-10.v2",
  "attack_space": "action_space_command_intervention",
  "code_repo": "$CODE_REPO",
  "python": "$VENV",
  "commit": "$COMMIT",
  "expected_git_commit": "$EXPECTED_GIT_COMMIT",
  "checkpoint": "$CHECKPOINT",
  "goal_model_manifest": "$GOAL_MODEL_MANIFEST",
  "episodes": 5,
  "conditions": ["CLEAN", "TRUE_CMDOPEN_T10_C2F", "RAND_ACTION_NOISE_T10_C2F"],
  "direct_command_override": "TRUE_CMDOPEN_T10_C2F sets raw gripper command to 1.0",
  "action_noise": "RAND_ACTION_NOISE_T10_C2F uses deterministic sha256-seeded action-space noise"
}
JSON

while IFS='|' read -r parent_key cond; do
  [ -z "$parent_key" ] && continue
  meta="$OUT/${parent_key}/${cond}/episode_metadata.json"
  if [ -s "$meta" ] && "$VENV" scripts/stageb/audit_c2f_track_a_run.py --metadata-complete "$meta" >/dev/null 2>&1; then
    echo "SKIP $parent_key $cond" | tee -a "$RUN_ROOT/launch.log"
    continue
  fi
  [ -s "$meta" ] && "$VENV" scripts/stageb/audit_c2f_track_a_run.py --archive-invalid-output-root "$OUT" --parent-key "$parent_key" --condition "$cond" --invalid-archive-root "$RUN_ROOT/invalid_attempts"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONPATH="$CODE_REPO:$CODE_REPO/src:$CODE_REPO/scripts" \
    "$VENV" scripts/stageb/run_c2f_canary_worker.py \
    --parent-key "$parent_key" --condition "$cond" --checkpoint "$CHECKPOINT" \
    --output-dir "$OUT" --expected-git-commit "$COMMIT" --policy-model-manifest "$GOAL_MODEL_MANIFEST" \
    > "$RUN_ROOT/log_${parent_key//\//_}_${cond}.log" 2>&1
  rc=$?
  [ "$rc" -eq 0 ] || echo "FAIL rc=$rc $parent_key $cond" | tee -a "$RUN_ROOT/launch.log"
done < "$RUN_ROOT/smoke_jobs.txt"

"$VENV" scripts/stageb/audit_c2f_track_a_run.py \
  --run-root "$RUN_ROOT" --output-root "$OUT" --parent-manifest "$RUN_ROOT/parent_manifest.jsonl" \
  --jobs-file "$RUN_ROOT/smoke_jobs.txt"
echo "RUN_ROOT=$RUN_ROOT"
