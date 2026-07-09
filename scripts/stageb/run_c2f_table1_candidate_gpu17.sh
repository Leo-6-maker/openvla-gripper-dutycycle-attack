#!/usr/bin/env bash
# C2f Table1-candidate online experiment — GPU1 + GPU7, 144 episodes
# D7 Table1 remains FROZEN. This validates C2f as secondary/upgrade candidate.
set -uo pipefail

REPO=/mnt/sdc/dty_user/openvla_attack
VENV=$REPO/envs/openvla-official-a800/bin/python
PARENT_CSV=/mnt/sdc/dty_user/openvla_attack_evidence/condition_matrix/d7_table1_manifest/d7_table1_preregistered_parent_keys.csv
CHECKPOINT=/mnt/sdc/dty_user/openvla_attack_evidence/c2f/siglip_full_final/train_D/c2f_rgb_lang_temporal_detector_v0.pt

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT=/mnt/sdc/dty_user/openvla_attack_evidence/c2f/table1_candidate_gpu17_${STAMP}
OUT=$RUN_ROOT/output
mkdir -p "$OUT"

cd "$REPO" || exit 1

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

COMMIT=$(git rev-parse HEAD)
SHORT=$(git rev-parse --short HEAD)

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
  "repo": "$REPO", "commit": "$COMMIT",
  "checkpoint": "$CHECKPOINT", "parent_csv": "$PARENT_CSV",
  "gpus": [1, 7], "conditions": ["CLEAN", "TRUE_T10", "RAND_T10"],
  "n_per_suite": 12, "seed": 42,
  "tau_emit": 0.33, "tau_suppress": 0.67, "tau_abstain": 0.5, "tau_primary": 0.5,
  "attack_horizon": 10, "pgd_steps": 10, "epsilon": "6/255"
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
CONDITIONS=("CLEAN" "TRUE_T10" "RAND_T10")
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
    if [ -s "$meta" ] && [ -s "$steps" ]; then
      echo "[GPU${gpu}] SKIP ${parent_key} ${cond}" | tee -a "$RUN_ROOT/launch.log"
      echo "${parent_key}|${cond}|${gpu}|SKIP" >> "$RUN_ROOT/completed_jobs.txt"
      continue
    fi
    echo "[GPU${gpu}] START ${parent_key} ${cond} $(date +%H:%M:%S)" | tee -a "$RUN_ROOT/launch.log"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$REPO:$REPO/src:$REPO/scripts \
      "$VENV" scripts/stageb/run_c2f_canary_worker.py \
      --parent-key "$parent_key" --condition "$cond" \
      --checkpoint "$CHECKPOINT" --gpu 0 --output-dir "$OUT" \
      --git-commit "$SHORT" > "$logfile" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ] && [ -s "$meta" ] && [ -s "$steps" ]; then
      echo "[GPU${gpu}] OK ${parent_key} ${cond} $(date +%H:%M:%S)" | tee -a "$RUN_ROOT/launch.log"
      echo "${parent_key}|${cond}|${gpu}|OK" >> "$RUN_ROOT/completed_jobs.txt"
    else
      echo "[GPU${gpu}] FAIL rc=${rc} ${parent_key} ${cond}" | tee -a "$RUN_ROOT/launch.log"
      echo "${parent_key}|${cond}|${gpu}|FAIL|rc=${rc}" >> "$RUN_ROOT/failed_jobs.txt"
    fi
    nvidia-smi -i "$gpu" >> "$RUN_ROOT/nvidia_smi_gpu${gpu}.log" 2>&1 || true
  done < "$jobfile"
  echo "[GPU${gpu}] queue done $(date)" | tee -a "$RUN_ROOT/launch.log"
}

echo "[3/5] Launch GPU queues" | tee -a "$RUN_ROOT/launch.log"
run_queue 1 &
PID1=$!
run_queue 7 &
PID7=$!
wait $PID1; wait $PID7

echo "[4/5] Post-run audit" | tee -a "$RUN_ROOT/launch.log"
"$VENV" - <<PY
import json, pathlib, glob, hashlib
from collections import Counter, defaultdict
run_root = pathlib.Path("$RUN_ROOT")
out = run_root / "output"
reports = sorted(out.glob("**/episode_metadata.json"))
steps = sorted(out.glob("**/step_records.jsonl"))
metas = []
for rp in reports:
    try:
        m = json.loads(rp.read_text())
        m["_path"] = str(rp)
        metas.append(m)
    except Exception as e:
        print("BAD_META", rp, e)
expected_parents = [json.loads(l)["parent_key"] for l in (run_root / "parent_manifest.jsonl").read_text().splitlines()]
expected_conditions = ["CLEAN", "TRUE_T10", "RAND_T10"]
by_cond = Counter(m.get("condition") for m in metas)
by_suite = Counter(m.get("suite") for m in metas)
parents_done = defaultdict(set)
for m in metas:
    parents_done[m["parent_key"]].add(m["condition"])
missing = []
for p in expected_parents:
    for c in expected_conditions:
        if c not in parents_done[p]:
            missing.append((p, c))
pairs = [p for p in expected_parents if "TRUE_T10" in parents_done[p] and "RAND_T10" in parents_done[p]]
delivery = []
no_emit = []
for m in metas:
    if m.get("condition") in ["TRUE_T10", "RAND_T10"]:
        dc = int(m.get("delivery_count", 0))
        delivery.append(dc)
        if int(m.get("attack_window_start", -1)) < 0:
            no_emit.append((m["parent_key"], m["condition"]))
audit = {
    "expected_parents": len(expected_parents),
    "expected_episodes": len(expected_parents) * len(expected_conditions),
    "metadata_count": len(reports),
    "step_records_count": len(steps),
    "by_condition": dict(by_cond),
    "by_suite": dict(by_suite),
    "paired_true_rand_parents": len(pairs),
    "missing": missing,
    "no_emit": no_emit,
    "delivery_count_min": min(delivery) if delivery else None,
    "delivery_count_max": max(delivery) if delivery else None,
    "delivery_count_mean": sum(delivery) / len(delivery) if delivery else None,
    "failed_jobs_file_nonempty": (run_root / "failed_jobs.txt").stat().st_size > 0,
}
(run_root / "postrun_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
print(json.dumps(audit, indent=2, sort_keys=True))
rows = ["parent_key,suite,task_index,state_id,condition,success,total_steps,attack_window_start,attack_window_end,delivery_count"]
for m in metas:
    rows.append(",".join(str(m.get(k, "")) for k in [
        "parent_key","suite","task_index","state_id","condition","success","total_steps",
        "attack_window_start","attack_window_end","delivery_count"
    ]))
(run_root / "summary_table.csv").write_text("\n".join(rows) + "\n")
readme = f"""# C2f Table1 Candidate GPU1+GPU7
Run root: {run_root}
Expected: {audit['expected_episodes']} | Done: {audit['metadata_count']} | Missing: {len(audit['missing'])}
By condition: {audit['by_condition']}
By suite: {audit['by_suite']}
Paired TRUE/RAND: {audit['paired_true_rand_parents']}
No-emit: {len(audit['no_emit'])}
Delivery mean: {audit['delivery_count_mean']}
D7 Table1 FROZEN. COMMAND_OPEN not run.
"""
(run_root / "README_STATUS.md").write_text(readme)
PY
echo "[5/5] DONE $(date)" | tee -a "$RUN_ROOT/launch.log"
echo "RUN_ROOT=$RUN_ROOT"
