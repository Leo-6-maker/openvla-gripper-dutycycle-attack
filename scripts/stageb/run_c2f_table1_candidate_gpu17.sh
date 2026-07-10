#!/usr/bin/env bash
# C2f Table1-candidate online experiment — GPU1 + GPU7, 144 episodes
# D7 Table1 remains FROZEN. This validates C2f as secondary/upgrade candidate.
set -uo pipefail

CODE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${A800_PY:-/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python}"
PARENT_CSV=/mnt/sdc/dty_user/openvla_attack_evidence/condition_matrix/d7_table1_manifest/d7_table1_preregistered_parent_keys.csv
CHECKPOINT=/mnt/sdc/dty_user/openvla_attack_evidence/c2f/siglip_full_final/train_D/c2f_rgb_lang_temporal_detector_v0.pt
GOAL_MODEL_MANIFEST="$CODE_REPO/artifacts/goal_model_manifest.json"

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT=/mnt/sdc/dty_user/openvla_attack_evidence/c2f/table1_candidate_gpu17_${STAMP}
OUT=$RUN_ROOT/output
mkdir -p "$OUT"

cd "$CODE_REPO" || exit 1

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

COMMIT=$(git rev-parse HEAD)

echo "=== C2f Table1 Candidate GPU1+GPU7 ===" | tee "$RUN_ROOT/launch.log"
echo "RUN_ROOT=$RUN_ROOT" | tee -a "$RUN_ROOT/launch.log"
echo "COMMIT=$COMMIT" | tee -a "$RUN_ROOT/launch.log"
echo "START=$(date)" | tee -a "$RUN_ROOT/launch.log"

git status --short > "$RUN_ROOT/git_status_short.txt"
sha256sum "$CHECKPOINT" > "$RUN_ROOT/checkpoint.sha256"
sha256sum "$PARENT_CSV" > "$RUN_ROOT/parent_csv.sha256"
sha256sum scripts/stageb/run_c2f_canary_worker.py > "$RUN_ROOT/worker.sha256"
sha256sum src/gripper_attack/c2f_siglip_detector_runtime.py > "$RUN_ROOT/runtime.sha256"
nvidia-smi -i 1,7 > "$RUN_ROOT/nvidia_smi_start.txt"

cat > "$RUN_ROOT/launch_env.json" <<JSON
{
  "code_repo": "$CODE_REPO", "python": "$VENV", "commit": "$COMMIT",
  "checkpoint": "$CHECKPOINT", "parent_csv": "$PARENT_CSV",
  "gpus": [1, 7], "conditions": ["CLEAN", "TRUE_CMDOPEN_T10_C2F", "RAND_ACTION_NOISE_T10_C2F"],
  "n_per_suite": 12, "seed": 42,
  "tau_emit": 0.33, "tau_suppress": 0.67, "tau_abstain": 0.5, "tau_primary": 0.5,
  "attack_horizon": 10, "attack_space": "action_space_command_intervention",
  "direct_command_override": "TRUE_CMDOPEN_T10_C2F sets raw gripper command to 1.0",
  "action_noise": "RAND_ACTION_NOISE_T10_C2F uses deterministic sha256-seeded action-space noise",
  "epsilon": "6/255"
}
JSON

echo "[1/5] Freeze parent manifest" | tee -a "$RUN_ROOT/launch.log"
"$VENV" - <<PY
import csv, json, random, hashlib, pathlib
parent_csv = pathlib.Path("$PARENT_CSV")
run_root = pathlib.Path("$RUN_ROOT")
random.seed(42)
parents = list(csv.DictReader(parent_csv.open()))
suites = ["libero_object", "libero_10", "libero_goal", "libero_spatial"]
selected = []
for suite in suites:
    pool = [p for p in parents if p.get("suite") == suite]
    if len(pool) < 12:
        raise RuntimeError(f"Not enough parents for {suite}: {len(pool)}")
    chosen = random.sample(pool, 12)
    selected.extend(chosen)
manifest = run_root / "parent_manifest.jsonl"
with manifest.open("w") as f:
    for p in selected:
        f.write(json.dumps(p, sort_keys=True) + "\n")
csv_sha = hashlib.sha256(parent_csv.read_bytes()).hexdigest()
prov = {"source_csv": str(parent_csv), "csv_sha256": csv_sha, "seed": 42,
        "suites": suites, "n_per_suite": 12, "total_parents": len(selected)}
(run_root / "parent_selection_provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
print(f"Frozen {len(selected)} parents")
PY

echo "[2/5] Generate job queues" | tee -a "$RUN_ROOT/launch.log"
rm -f "$RUN_ROOT"/jobs_gpu*.txt "$RUN_ROOT/jobs_all.txt"
CONDITIONS=("CLEAN" "TRUE_CMDOPEN_T10_C2F" "RAND_ACTION_NOISE_T10_C2F")
GPUS=(1 7)
idx=0
while IFS= read -r line; do
  parent_key=$(echo "$line" | "$VENV" -c "import json,sys; print(json.loads(sys.stdin.read())['parent_key'])")
  for cond in "${CONDITIONS[@]}"; do
    gpu=${GPUS[$((idx % 2))]}
    echo "${parent_key}|${cond}|${gpu}" >> "$RUN_ROOT/jobs_gpu${gpu}.txt"
    echo "${parent_key}|${cond}|${gpu}" >> "$RUN_ROOT/jobs_all.txt"
    idx=$((idx + 1))
  done
done < "$RUN_ROOT/parent_manifest.jsonl"
echo "Total jobs: $idx" | tee -a "$RUN_ROOT/launch.log"
wc -l "$RUN_ROOT"/jobs_gpu*.txt | tee -a "$RUN_ROOT/launch.log"

touch "$RUN_ROOT/failed_jobs.txt" "$RUN_ROOT/completed_jobs.txt"
mkdir -p "$RUN_ROOT/invalid_attempts"

run_queue() {
  local gpu=$1
  local jobfile="$RUN_ROOT/jobs_gpu${gpu}.txt"
  echo "[GPU${gpu}] queue start $(date)" | tee -a "$RUN_ROOT/launch.log"
  while IFS='|' read -r parent_key cond assigned_gpu; do
    [ -z "$parent_key" ] && continue
    local meta="$OUT/${parent_key}/${cond}/episode_metadata.json"
    local steps="$OUT/${parent_key}/${cond}/step_records.jsonl"
    local logname="${parent_key//\//_}_${cond}.log"
    local logfile="$RUN_ROOT/log_${logname}"
    if [ -s "$meta" ] && "$VENV" scripts/stageb/audit_c2f_track_a_run.py --metadata-complete "$meta" >/dev/null 2>&1; then
      echo "[GPU${gpu}] SKIP ${parent_key} ${cond}" | tee -a "$RUN_ROOT/launch.log"
      echo "${parent_key}|${cond}|${gpu}|SKIP" >> "$RUN_ROOT/completed_jobs.txt"
      continue
    fi
    if [ -s "$meta" ]; then
      "$VENV" scripts/stageb/audit_c2f_track_a_run.py \
        --archive-invalid-output-root "$OUT" --parent-key "$parent_key" --condition "$cond" \
        --invalid-archive-root "$RUN_ROOT/invalid_attempts" >> "$RUN_ROOT/launch.log"
    fi
    echo "[GPU${gpu}] START ${parent_key} ${cond} $(date +%H:%M:%S)" | tee -a "$RUN_ROOT/launch.log"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$CODE_REPO:$CODE_REPO/src:$CODE_REPO/scripts \
      "$VENV" scripts/stageb/run_c2f_canary_worker.py \
      --parent-key "$parent_key" --condition "$cond" \
      --checkpoint "$CHECKPOINT" --gpu 0 --output-dir "$OUT" \
      --expected-git-commit "$COMMIT" --policy-model-manifest "$GOAL_MODEL_MANIFEST" > "$logfile" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ] && [ -s "$meta" ] && "$VENV" scripts/stageb/audit_c2f_track_a_run.py --metadata-complete "$meta" >/dev/null 2>&1 && [ -s "$steps" ]; then
      echo "[GPU${gpu}] OK ${parent_key} ${cond} $(date +%H:%M:%S)" | tee -a "$RUN_ROOT/launch.log"
      echo "${parent_key}|${cond}|${gpu}|OK" >> "$RUN_ROOT/completed_jobs.txt"
    else
      echo "[GPU${gpu}] FAIL rc=${rc} ${parent_key} ${cond}" | tee -a "$RUN_ROOT/launch.log"
      echo "${parent_key}|${cond}|${gpu}|FAIL|rc=${rc}" >> "$RUN_ROOT/failed_jobs.txt"
    fi
    nvidia-smi -i "$gpu" >> "$RUN_ROOT/nvidia_smi_gpu${gpu}.log" 2>&1 || true
  done < "$jobfile"
  # ── Retry failed jobs at tail end ──
  echo "[GPU${gpu}] retry phase $(date)" | tee -a "$RUN_ROOT/launch.log"
  local retry_file="$RUN_ROOT/retry_gpu${gpu}.txt"
  grep "|${gpu}|FAIL" "$RUN_ROOT/failed_jobs.txt" 2>/dev/null | cut -d'|' -f1,2 | while IFS='|' read -r parent_key cond; do
    [ -z "$parent_key" ] && continue
    echo "${parent_key}|${cond}|${gpu}" >> "$retry_file"
  done
  if [ -s "$retry_file" ]; then
    while IFS='|' read -r parent_key cond assigned_gpu; do
      [ -z "$parent_key" ] && continue
      local meta="$OUT/${parent_key}/${cond}/episode_metadata.json"
      local steps="$OUT/${parent_key}/${cond}/step_records.jsonl"
      if [ -s "$meta" ] && "$VENV" scripts/stageb/audit_c2f_track_a_run.py --metadata-complete "$meta" >/dev/null 2>&1 && [ -s "$steps" ]; then continue; fi
      if [ -s "$meta" ]; then
        "$VENV" scripts/stageb/audit_c2f_track_a_run.py \
          --archive-invalid-output-root "$OUT" --parent-key "$parent_key" --condition "$cond" \
          --invalid-archive-root "$RUN_ROOT/invalid_attempts" >> "$RUN_ROOT/launch.log"
      fi
      echo "[GPU${gpu}] RETRY ${parent_key} ${cond} $(date +%H:%M:%S)" | tee -a "$RUN_ROOT/launch.log"
      CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$CODE_REPO:$CODE_REPO/src:$CODE_REPO/scripts \
        "$VENV" scripts/stageb/run_c2f_canary_worker.py \
        --parent-key "$parent_key" --condition "$cond" \
        --checkpoint "$CHECKPOINT" --gpu 0 --output-dir "$OUT" \
        --expected-git-commit "$COMMIT" --policy-model-manifest "$GOAL_MODEL_MANIFEST" > "$RUN_ROOT/log_retry_${parent_key//\//_}_${cond}.log" 2>&1
      rc=$?
      if [ "$rc" -eq 0 ] && [ -s "$meta" ] && "$VENV" scripts/stageb/audit_c2f_track_a_run.py --metadata-complete "$meta" >/dev/null 2>&1 && [ -s "$steps" ]; then
        echo "[GPU${gpu}] RETRY OK ${parent_key} ${cond}" | tee -a "$RUN_ROOT/launch.log"
        sed -i "\|${parent_key}|${cond}|${gpu}|FAIL|d" "$RUN_ROOT/failed_jobs.txt" 2>/dev/null || true
      else
        echo "[GPU${gpu}] RETRY STILL FAIL ${parent_key} ${cond}" | tee -a "$RUN_ROOT/launch.log"
      fi
    done < "$retry_file"
  fi
  echo "[GPU${gpu}] queue done $(date)" | tee -a "$RUN_ROOT/launch.log"
}

echo "[3/5] Launch GPU queues" | tee -a "$RUN_ROOT/launch.log"
run_queue 1 &
PID1=$!
run_queue 7 &
PID7=$!
wait $PID1; wait $PID7

echo "[4/5] Post-run audit" | tee -a "$RUN_ROOT/launch.log"
if ! "$VENV" scripts/stageb/audit_c2f_track_a_run.py \
  --run-root "$RUN_ROOT" --output-root "$OUT" --parent-manifest "$RUN_ROOT/parent_manifest.jsonl"; then
  echo "[4/5] Post-run audit HOLD: missing or runtime-invalid episodes remain" | tee -a "$RUN_ROOT/launch.log"
  exit 2
fi
echo "[5/5] DONE $(date)" | tee -a "$RUN_ROOT/launch.log"
echo "RUN_ROOT=$RUN_ROOT"
