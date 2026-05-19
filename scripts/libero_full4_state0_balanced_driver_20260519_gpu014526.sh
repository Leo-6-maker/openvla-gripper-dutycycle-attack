#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENVLA_ATTACK_REPO_ROOT:-/home/liuyu/openvla_gripper_attack/OpenVLA Gripper Duty-Cycle Attack}"
PY="${OPENVLA_COMPAT_PYTHON:-/data/aviary/envs/openvla_compat/bin/python}"
OUT="${FULL4_DENOMINATOR_OUTPUT_ROOT:-/data/liuyu/outputs/libero_full4_state0_balanced_clean_20260519_gpu014526}"
TABLES="$OUT/tables"
LOGS="$OUT/logs"
FAILED="$OUT/failed_runs"
PREP="${FULL4_PREP_SCRIPT:-scripts/libero_full4_state0_balanced_prepare_20260518.py}"
UPLOAD_CSV="$TABLES/local_missing_download_upload_result_20260518.csv"
LOCK="$OUT/RUNNING.lock"
STATUS_MD="$OUT/DENOMINATOR_STATUS_20260519.md"
CONFIG="${FULL4_TASKS_CONFIG:-configs/v4_tasks_libero_full4_20260518.yaml}"
SCRATCH="${FULL4_VIDEO_SCRATCH:-/home/liuyu/root_scratch/libero_clean_video_scratch_20260519_gpu014526}"

mkdir -p "$OUT" "$TABLES" "$LOGS" "$FAILED" "$SCRATCH"

if [[ -f "$LOCK" ]]; then
  old_pid="$(awk -F= '/^pid=/{print $2}' "$LOCK" 2>/dev/null || true)"
  if [[ -n "${old_pid:-}" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "RUNNING.lock exists with live pid=$old_pid; exiting"
    exit 0
  fi
fi
{
  echo "pid=$$"
  echo "host=$(hostname)"
  echo "started_at=$(date '+%F %T %Z')"
  echo "task=state0_balanced_clean_denominator_only"
} >"$LOCK"

cd "$ROOT"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export TMPDIR="/data/liuyu/tmp/libero_full4_state0_balanced_20260519_gpu014526"
export HF_HOME="/data/liuyu/.cache/huggingface"
export HF_HUB_CACHE="/data/liuyu/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="/data/liuyu/.cache/huggingface/transformers"
export XDG_CACHE_HOME="/data/liuyu/.cache"
export LIBERO_DATA_ROOT="/data/aviary/datasets/libero/datasets"
export OPENVLA_MODEL_ROOT="/data/aviary/models/openvla"
export OPENVLA_BASE_MODEL_DIR="/data/aviary/models/openvla/openvla-7b"
mkdir -p "$TMPDIR" "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME"

SLOT_CUDA=("0,1" "4,5" "2,6")
SLOT_RENDER=(0 4 2)

log() {
  printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "$LOGS/driver.log"
}

free_gib() {
  df -BG --output=avail "$1" | tail -1 | tr -dc '0-9'
}

scratch_gib() {
  du -sBG "$SCRATCH" 2>/dev/null | awk '{gsub(/G/,"",$1); print $1+0}'
}

write_status() {
  local decision="$1"
  local phase="$2"
  local notes="${3:-}"
  printf '%s\n' "$decision" >"$OUT/FINAL_DECISION.txt"
  {
    echo "# LIBERO Full4 State0 Balanced Denominator Status"
    echo
    echo "Decision: \`$decision\`"
    echo
    echo "- phase: \`$phase\`"
    echo "- updated_at: \`$(date '+%F %T %Z')\`"
    echo "- host: \`$(hostname)\`"
    echo "- output_root: \`$OUT\`"
    echo "- notes: $notes"
    echo
    echo "No VIS/random/oracle/benchmark/attack was run."
  } >"$STATUS_MD"
}

record_df() {
  {
    echo "## df snapshot $(date '+%F %T %Z')"
    echo '```text'
    df -h / /data
    echo '```'
  } >>"$STATUS_MD"
}

guard_rollout_start() {
  local data_free root_free
  data_free="$(free_gib /data)"
  root_free="$(free_gib /)"
  if (( data_free < 30 )); then
    write_status "disk_blocked" "disk_guard" "/data free ${data_free}G <30G"
    return 2
  fi
  if (( data_free < 40 )); then
    write_status "disk_blocked" "disk_guard" "/data free ${data_free}G <40G; no new rollout"
    return 1
  fi
  if (( root_free < 50 )); then
    write_status "disk_blocked" "disk_guard" "/ free ${root_free}G <50G; no new rollout"
    return 1
  fi
  return 0
}

guard_video_start() {
  local data_free root_free scratch_used
  data_free="$(free_gib /data)"
  root_free="$(free_gib /)"
  scratch_used="$(scratch_gib)"
  if (( data_free < 45 )); then
    return 1
  fi
  if (( root_free < 60 )); then
    return 1
  fi
  if (( scratch_used > 25 )); then
    return 1
  fi
  return 0
}

artifact_done() {
  local run_id="$1"
  "$PY" - "$OUT/$run_id" <<'PY' >/dev/null 2>&1
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
required = ["progress.json", "step_records.jsonl", "episode_records.jsonl", "summary.csv", "run_manifest.json"]
if not all((d / x).exists() and (d / x).stat().st_size > 0 for x in required):
    raise SystemExit(1)
try:
    progress = json.loads((d / "progress.json").read_text())
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if progress.get("status") == "done" else 1)
PY
}

prepare_run_dir() {
  local run_id="$1"
  local dir="$OUT/$run_id"
  if [[ -d "$dir" ]]; then
    if artifact_done "$run_id"; then
      log "reuse completed run $run_id"
      return 0
    fi
    local ts
    ts="$(date '+%Y%m%d_%H%M%S')"
    mkdir -p "$FAILED/$ts"
    mv "$dir" "$FAILED/$ts/$run_id"
    log "moved incomplete run to failed_runs/$ts/$run_id"
  fi
}

render_one() {
  local task_id="$1" run_id="$2" cuda="$3" render="$4" phase="$5"
  if compgen -G "$OUT/$run_id/videos/*.mp4" >/dev/null; then
    return 0
  fi
  if ! guard_video_start; then
    log "skip video by guard run=$run_id phase=$phase"
    if [[ "$phase" == "smoke" ]]; then
      write_status "video_guard_blocked" "smoke_video" "video guard blocked required smoke video"
      return 2
    fi
    return 0
  fi
  log "render video run=$run_id cuda=$cuda render=$render"
  env CUDA_VISIBLE_DEVICES="$cuda" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
    "$PY" scripts/v4_render_episode_video_from_steps.py \
      --tasks_config "$CONFIG" \
      --task_id "$task_id" \
      --run_dir "$OUT/$run_id" \
      --episode_ids auto \
      --render_gpu_device_id "$render" \
      --image_size 256 \
      --frame_stride 4 \
      --fps 20 >"$LOGS/${run_id}_render.log" 2>&1
}

run_rollout_one() {
  local phase="$1" suite="$2" task_id="$3" task_name="$4" state="$5" seed="$6" run_id="$7" unnorm="$8" max_steps="$9" checkpoint="${10}" target_object="${11}" target_receptacle="${12}" cuda="${13}" render="${14}" do_video="${15}"
  if ! guard_rollout_start; then
    return 3
  fi
  prepare_run_dir "$run_id"
  if ! artifact_done "$run_id"; then
    log "launch clean phase=$phase suite=$suite task=$task_id state=$state seed=$seed cuda=$cuda render=$render"
    env CUDA_VISIBLE_DEVICES="$cuda" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
      V4_TARGET_OBJECT_NAME="${target_object:-akita_black_bowl_1}" \
      V4_TARGET_RECEPTACLE_NAME="${target_receptacle:-plate_1}" \
      "$PY" scripts/v4_run_eval_openvla.py \
        --tasks_config "$CONFIG" \
        --attack_config configs/paper_black_bowl_attack.yaml \
        --directions_config configs/v4_directions.yaml \
        --task_id "$task_id" \
        --trigger clean \
        --rho 0.0 \
        --seed "$seed" \
        --episodes 1 \
        --max_steps_override "$max_steps" \
        --output_root "$OUT" \
        --run_id "$run_id" \
        --model_path "$checkpoint" \
        --base_model_code_dir "$OPENVLA_BASE_MODEL_DIR" \
        --unnorm_key "$unnorm" \
        --camera_obs_key agentview_image \
        --model_gpu_device_id -1 \
        --render_gpu_device_id "$render" \
        --image_size 256 \
        --openvla_resize_size 224 \
        --success_metric done \
        --auto_patch_compat \
        --epsilon 0.10 \
        --step_size 0.020 \
        --attack_steps 20 \
        --temporal_init none \
        --action_clamp_mode none \
        --libero_official_preprocess \
        --center_crop \
        --postprocess_gripper \
        --deterministic_init_states \
        --state_ids "$state" >"$LOGS/${run_id}.log" 2>&1 || true
  fi
  if [[ "$do_video" == "yes" ]]; then
    render_one "$task_id" "$run_id" "$cuda" "$render" "$phase"
  fi
}

run_jobs_tsv() {
  local jobs="$1" do_video="$2" mode="${3:-rollout}"
  local idx=0
  local pids=()
  local failures=0
  while IFS=$'\t' read -r phase suite task_id task_name state seed run_id unnorm max_steps checkpoint target_object target_receptacle black_bowl_sanity rest; do
    [[ -z "${phase:-}" || "$phase" == "phase" ]] && continue
    local slot=$((idx % ${#SLOT_CUDA[@]}))
    if [[ "$mode" == "render" ]]; then
      ( render_one "$task_id" "$run_id" "${SLOT_CUDA[$slot]}" "${SLOT_RENDER[$slot]}" "$phase" ) &
    else
      ( run_rollout_one "$phase" "$suite" "$task_id" "$task_name" "$state" "$seed" "$run_id" "$unnorm" "$max_steps" "$checkpoint" "$target_object" "$target_receptacle" "${SLOT_CUDA[$slot]}" "${SLOT_RENDER[$slot]}" "$do_video" ) &
    fi
    pids+=("$!")
    idx=$((idx + 1))
    if (( ${#pids[@]} >= ${#SLOT_CUDA[@]} )); then
      for pid in "${pids[@]}"; do
        wait "$pid" || failures=$((failures + 1))
      done
      pids=()
    fi
  done <"$jobs"
  for pid in "${pids[@]}"; do
    wait "$pid" || failures=$((failures + 1))
  done
  return 0
}

all_rows_complete() {
  local csv="$1" expected="$2"
  "$PY" - "$csv" "$expected" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8"))) if sys.argv[1] else []
expected = int(sys.argv[2])
ok = len(rows) == expected and all(r.get("artifact_complete") == "True" for r in rows)
if expected == 4:
    ok = ok and all(bool(r.get("video_path")) for r in rows)
raise SystemExit(0 if ok else 1)
PY
}

write_status "config_generation_failed" "starting" "driver started; final decision pending"
record_df
log "Phase 0: dry-run, manifest, BDDL/checkpoint/config checks"
nvidia-smi >"$LOGS/nvidia_smi_start.log" 2>&1 || true
df -h / /data >"$LOGS/df_start.log"
"$PY" "$PREP" generate --output-root "$OUT" --repo-root "$ROOT" --upload-result-csv "$UPLOAD_CSV" >"$LOGS/phase0_generate.log" 2>&1 || {
  write_status "config_generation_failed" "phase0" "dry-run/config generation failed"
  exit 2
}
write_status "config_generation_failed" "phase0_complete" "phase0 passed; smoke next"

log "Phase 2: four-suite smoke, state0 seed1 clean with videos"
run_jobs_tsv "$TABLES/phase2_smoke_jobs.tsv" "yes" "rollout"
"$PY" "$PREP" summarize --output-root "$OUT" >"$LOGS/phase2_summarize_smoke_all.log" 2>&1 || true
"$PY" - "$OUT" <<'PY'
import argparse, importlib.util, pathlib, sys
out = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("prep", out / "full4_state0_balanced_prepare_20260518.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
args = argparse.Namespace(output_root=str(out))
mod.summarize_runs(args, phase_filter="smoke", output_name="phase2_smoke_manifest.csv")
PY
if ! all_rows_complete "$TABLES/phase2_smoke_manifest.csv" 4; then
  existing_decision="$(cat "$OUT/FINAL_DECISION.txt" 2>/dev/null || true)"
  if [[ "$existing_decision" == "video_guard_blocked" || "$existing_decision" == "disk_blocked" ]]; then
    write_status "$existing_decision" "phase2" "required smoke artifacts were blocked by guard"
  else
    write_status "smoke_failed" "phase2" "not all 4 smoke runs completed with video artifacts"
  fi
  record_df
  exit 3
fi
write_status "config_generation_failed" "phase2_complete" "smoke 4/4 artifact complete; seed/state sanity next"

log "Phase 2.5: Spatial seed/state sanity, no-video"
run_jobs_tsv "$TABLES/phase2_5_seed_state_sanity_jobs.tsv" "no" "rollout"
"$PY" "$PREP" create_clean_jobs --output-root "$OUT" >"$LOGS/phase2_5_create_clean_jobs.log" 2>&1 || {
  write_status "config_generation_failed" "phase2_5" "failed to summarize sanity/create phase3 jobs"
  exit 4
}
sanity_mode="$(cat "$OUT/phase2_5_sanity_decision.txt" 2>/dev/null || true)"
write_status "config_generation_failed" "phase2_5_complete" "sanity mode: ${sanity_mode:-unknown}; clean denominator next"

log "Phase 3: clean denominator, no-video"
if ! guard_rollout_start; then
  write_status "disk_blocked" "phase3_precheck" "disk guard prevented clean denominator launch"
  record_df
  exit 0
else
  run_jobs_tsv "$TABLES/phase3_clean_jobs.tsv" "no" "rollout"
fi
"$PY" "$PREP" summarize --output-root "$OUT" >"$LOGS/phase3_summarize_clean.log" 2>&1 || true
if [[ "$(cat "$OUT/FINAL_DECISION.txt" 2>/dev/null || true)" == "disk_blocked" ]]; then
  record_df
  exit 0
fi

log "Phase 4: clean-only generic-v4 detector and CQ-v2"
if "$PY" scripts/detect_contact_window_from_clean.py \
    -input_root "$OUT" \
    -output_csv "$TABLES/mechanism_window_raw.csv" \
    -config configs/generic_autowindow_detector.yaml \
    -summary_md "$OUT/generic_v4_window_summary_20260518.md" \
    -phase_cues_csv "$TABLES/generic_v4_phase_cues_20260518.csv" >"$LOGS/phase4_detector.log" 2>&1; then
  log "detector complete"
else
  log "detector returned nonzero; combine will classify if possible"
fi
if "$PY" scripts/extract_contact_quality_metrics.py \
    --input_root "$OUT" \
    --output_csv "$TABLES/clean_cqv2_metrics.csv" >"$LOGS/phase4_cqv2.log" 2>&1; then
  log "CQ-v2 extraction complete"
else
  log "CQ-v2 extraction returned nonzero; combine will classify if possible"
fi
"$PY" "$PREP" combine --output-root "$OUT" >"$LOGS/phase4_combine.log" 2>&1 || {
  write_status "config_generation_failed" "phase4" "failed to combine clean/detector/CQ outputs"
  exit 5
}

log "Phase 4 video queue: selective videos only if guards allow"
"$PY" "$PREP" create_video_jobs --output-root "$OUT" >"$LOGS/phase4_create_video_jobs.log" 2>&1 || true
if guard_video_start; then
  run_jobs_tsv "$TABLES/selective_video_jobs.tsv" "no" "render"
else
  log "selective video rendering skipped by disk/video guard"
fi

decision="$(cat "$OUT/FINAL_DECISION.txt" 2>/dev/null || echo clean_denominator_insufficient)"
write_status "$decision" "complete" "clean denominator construction finished"
record_df
df -h / /data >"$LOGS/df_end.log"
log "done decision=$decision"
