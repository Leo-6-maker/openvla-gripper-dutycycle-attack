#!/usr/bin/env bash
set -euo pipefail

OUT="${FULL4_DENOMINATOR_OUTPUT_ROOT:-/data/liuyu/outputs/libero_full4_state0_balanced_clean_20260519_gpu014526}"
LOGS="$OUT/logs"
TABLES="$OUT/tables"
SERVICE="${FULL4_DENOMINATOR_SERVICE:-libero-full4-denom-20260519-gpu014526.service}"
STATUS_MD="$OUT/DENOMINATOR_WATCHER_STATUS_20260519.md"
WATCH_CSV="$LOGS/denom_watch_20260519.csv"

mkdir -p "$LOGS" "$TABLES"

if [[ ! -f "$WATCH_CSV" ]]; then
  echo "timestamp,service_active,service_substate,root_avail_gib,data_avail_gib,gpu_mem_used_mib,gpu_util_pct,rollout_processes,final_decision,status_phase,recent_driver_log" >"$WATCH_CSV"
fi

free_gib() {
  df -BG --output=avail "$1" | tail -1 | tr -dc '0-9'
}

csv_escape() {
  local s="${1:-}"
  s="${s//$'\n'/ | }"
  s="${s//\"/\"\"}"
  printf '"%s"' "$s"
}

service_field() {
  systemctl show "$SERVICE" -p "$1" --value --no-pager 2>/dev/null || true
}

while true; do
  ts="$(date '+%F %T %Z')"
  active="$(service_field ActiveState)"
  substate="$(service_field SubState)"
  root_free="$(free_gib /)"
  data_free="$(free_gib /data)"
  gpu_mem="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';' || true)"
  gpu_util="$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';' || true)"
  procs="$(pgrep -af 'v4_run_eval_openvla.py|full4_state0_balanced_driver_20260519_gpu014526' 2>/dev/null | wc -l | tr -dc '0-9')"
  decision="$(cat "$OUT/FINAL_DECISION.txt" 2>/dev/null | head -1 || true)"
  phase="$(grep -m1 '^- phase:' "$OUT/DENOMINATOR_STATUS_20260519.md" 2>/dev/null | sed 's/^- phase: `//; s/`//g' || true)"
  recent="$(tail -20 "$LOGS/driver.log" 2>/dev/null || true)"

  {
    printf '%s,%s,%s,%s,%s,' "$ts" "$active" "$substate" "$root_free" "$data_free"
    csv_escape "$gpu_mem"; printf ','
    csv_escape "$gpu_util"; printf ',%s,' "$procs"
    csv_escape "$decision"; printf ','
    csv_escape "$phase"; printf ','
    csv_escape "$recent"; printf '\n'
  } >>"$WATCH_CSV"

  {
    echo "# LIBERO Full4 Denominator Watcher"
    echo
    echo "- updated_at: \`$ts\`"
    echo "- service: \`$SERVICE\`"
    echo "- service_active: \`${active:-unknown}\`"
    echo "- service_substate: \`${substate:-unknown}\`"
    echo "- final_decision: \`${decision:-pending}\`"
    echo "- status_phase: \`${phase:-pending}\`"
    echo "- root_free_gib: \`${root_free}\`"
    echo "- data_free_gib: \`${data_free}\`"
    echo "- rollout_processes: \`${procs}\`"
    echo
    echo "## GPU"
    echo '```text'
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true
    echo '```'
    echo
    echo "## Recent Driver Log"
    echo '```text'
    tail -40 "$LOGS/driver.log" 2>/dev/null || true
    echo '```'
    echo
    echo "No VIS/random/oracle/benchmark/attack is launched by this watcher."
  } >"$STATUS_MD"

  if [[ "${active:-}" != "active" && -n "${active:-}" ]]; then
    break
  fi
  sleep 60
done
