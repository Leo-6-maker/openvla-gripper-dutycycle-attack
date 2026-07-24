#!/usr/bin/env bash
set -u

REPO=/mnt/sdc/dty_user/openvla_attack_codex_r9q_final_20260713
PY=/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python3.10
HEAD=8ddbd03503c66e7efca9fcc84ad7d49974af33e0
BUNDLE=/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r9q_final_detector_bundle_5576d46_20260713_v1
BASE=/mnt/sdc/dty_user/openvla_attack_evidence/c2g
CAMPAIGN=$BASE/c2g_r9q_attack_campaign_8ddbd035_20260713_v1
CANARY_PLAN=$BASE/c2g_r9q_attack_canary_8ddbd035_20260713_v1
PANEL_PLAN=$BASE/c2g_r9q_attack_panel_8ddbd035_20260713_v1
FULL_PLAN=$BASE/c2g_r9q_attack_main_plan_8ddbd035_20260713_v1

test "$(git -C "$REPO" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$REPO" status --short)"
mkdir -p "$CAMPAIGN"
exec >>"$CAMPAIGN/campaign.log" 2>&1
printf 'CAMPAIGN_START=%s\nHEAD=%s\nCAP=2_PER_GPU\n' "$(date -Is)" "$HEAD"

deadline=$(( $(date +%s) + 18000 ))
state() { printf '%s %s\n' "$(date -Is)" "$1" | tee -a "$CAMPAIGN/state.log"; }

resource_ok() {
  local line6 line7 free6 free7 util6 util7
  line6=$(nvidia-smi -i 6 --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | head -n 1)
  line7=$(nvidia-smi -i 7 --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | head -n 1)
  free6=$(echo "$line6" | awk -F, '{gsub(/ /,"",$1); print $1}')
  util6=$(echo "$line6" | awk -F, '{gsub(/ /,"",$2); print $2}')
  free7=$(echo "$line7" | awk -F, '{gsub(/ /,"",$1); print $1}')
  util7=$(echo "$line7" | awk -F, '{gsub(/ /,"",$2); print $2}')
  printf '%s cap=2 required=44000 GPU6 free=%s util=%s GPU7 free=%s util=%s\n' "$(date -Is)" "$free6" "$util6" "$free7" "$util7" >>"$CAMPAIGN/resource.log"
  [ "${free6:-0}" -ge 44000 ] && [ "${free7:-0}" -ge 44000 ] && [ "${util6:-100}" -le 40 ] && [ "${util7:-100}" -le 40 ]
}

wait_resource() {
  while ! resource_ok; do
    if [ "$(date +%s)" -ge "$deadline" ]; then state HOLD_GPU_RESOURCE_TIMEOUT; exit 2; fi
    sleep 30
  done
  state GPU_RESOURCE_ADMITTED_2_RESIDENT_PER_GPU
}

run_stage() {
  local name=$1 plan=$2 run=$3 audit=$4 cells=$5
  wait_resource
  state "${name}_SCHEDULER_START_CAP_2"
  if [ -e "$run" ]; then state "${name}_RUN_ROOT_EXISTS"; exit 3; fi
  set +e
  "$PY" "$REPO/scripts/stageb/run_c2g_r9q_attack_scheduler.py" \
    --mode run \
    --plan-root "$plan" \
    --detector-bundle "$BUNDLE" \
    --output-root "$run" \
    --expected-git-commit "$HEAD" \
    --worker-budget-mib 18000 \
    --gpu-reserve-mib 8000 \
    --max-resident-workers-per-gpu 2 \
    --poll-seconds 20
  local rc=$?
  set -e
  printf '%s %s_SCHEDULER_RC=%s\n' "$(date -Is)" "$name" "$rc" >>"$CAMPAIGN/state.log"
  [ "$rc" -eq 0 ] || { state "HOLD_${name}_SCHEDULER"; exit 4; }
  state "${name}_AUDIT_START"
  set +e
  "$PY" "$REPO/tools/multisuite_detector/audit_c2g_r9q_attack_run.py" \
    --manifest "$plan/r9q_attack_manifest.jsonl" \
    --run-root "$run" \
    --output-root "$audit" \
    --expected-git-commit "$HEAD" \
    --mode "$name" \
    --expected-cells "$cells"
  rc=$?
  set -e
  printf '%s %s_AUDIT_RC=%s\n' "$(date -Is)" "$name" "$rc" >>"$CAMPAIGN/state.log"
  [ "$rc" -eq 0 ] || { state "HOLD_${name}_AUDIT"; exit 5; }
  state "${name}_PASS"
}

run_stage canary "$CANARY_PLAN" "$CAMPAIGN/canary_run" "$CAMPAIGN/canary_audit" 32
run_stage panel "$PANEL_PLAN" "$CAMPAIGN/panel_run" "$CAMPAIGN/panel_audit" 160
run_stage full "$FULL_PLAN" "$CAMPAIGN/full_run" "$CAMPAIGN/full_audit" 716
state PASS_C2G_R9Q_ATTACK_CAMPAIGN_COMPLETE
