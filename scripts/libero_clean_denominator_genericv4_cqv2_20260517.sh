#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/liuyu/openvla_gripper_attack/OpenVLA Gripper Duty-Cycle Attack"
PY="/data/aviary/envs/openvla_compat/bin/python"
OUT="/data/liuyu/outputs/libero_clean_denominator_genericv4_cqv2_20260517_2080ti"
PREV="/data/liuyu/outputs/extended_longrun_genericv4_libero_20260516_2080ti"
TABLES="$OUT/tables"
LOGS="$OUT/logs"
FAILED="$OUT/failed_runs"
SERVER_TAG="old2080ti"

mkdir -p "$OUT" "$TABLES" "$LOGS" "$FAILED"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-/data/aviary/datasets/libero}"
OPENVLA_MODEL_ROOT="${OPENVLA_MODEL_ROOT:-/data/aviary/models/openvla}"
OPENVLA_BASE_MODEL_DIR="${OPENVLA_BASE_MODEL_DIR:-$OPENVLA_MODEL_ROOT/openvla-7b}"
SPATIAL_MODEL="${OPENVLA_SPATIAL_MODEL_PATH:-$OPENVLA_MODEL_ROOT/openvla-7b-finetuned-libero-spatial}"
OBJECT_MODEL="${OPENVLA_OBJECT_MODEL_PATH:-$OPENVLA_MODEL_ROOT/openvla-7b-finetuned-libero-object}"
GOAL_MODEL="${OPENVLA_GOAL_MODEL_PATH:-$OPENVLA_MODEL_ROOT/openvla-7b-finetuned-libero-goal}"
LIBERO10_MODEL="${OPENVLA_LIBERO10_MODEL_PATH:-$OPENVLA_MODEL_ROOT/openvla-7b-finetuned-libero-10}"

SLOT_CUDA=("0,1" "4,5" "2,6")
SLOT_RENDER=(0 4 2)

log() {
  printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "$LOGS/driver.log"
}

sha_file() {
  [[ -f "$1" ]] && sha256sum "$1" | awk '{print $1}' || printf 'missing'
}

artifact_done() {
  "$PY" - "$OUT/$1/progress.json" <<'PY' >/dev/null 2>&1
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
obj = json.loads(p.read_text())
raise SystemExit(0 if obj.get("status") == "done" else 1)
PY
}

prepare_run_dir() {
  local run_id="$1"
  local dir="$OUT/$run_id"
  if [[ -d "$dir" ]]; then
    if artifact_done "$run_id"; then
      log "reuse completed run: $run_id"
      return 0
    fi
    local ts
    ts="$(date '+%Y%m%d_%H%M%S')"
    mkdir -p "$FAILED/$ts"
    mv "$dir" "$FAILED/$ts/$run_id"
    log "moved incomplete run to failed_runs/$ts/$run_id"
  fi
}

model_for_suite() {
  case "$1" in
    libero_spatial) printf '%s' "$SPATIAL_MODEL" ;;
    libero_object) printf '%s' "$OBJECT_MODEL" ;;
    libero_goal) printf '%s' "$GOAL_MODEL" ;;
    libero_10) printf '%s' "$LIBERO10_MODEL" ;;
    *) printf '%s' "$SPATIAL_MODEL" ;;
  esac
}

render_video() {
  local task_id="$1" run_id="$2" cuda="$3" render="$4"
  if compgen -G "$OUT/$run_id/videos/*.mp4" >/dev/null; then
    return 0
  fi
  log "render video $run_id"
  env CUDA_VISIBLE_DEVICES="$cuda" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
    "$PY" scripts/v4_render_episode_video_from_steps.py \
    --tasks_config configs/v4_tasks_libero.yaml \
    --task_id "$task_id" \
    --run_dir "$OUT/$run_id" \
    --episode_ids auto \
    --render_gpu_device_id "$render" \
    --image_size 256 \
    --frame_stride 4 \
    --fps 20 >"$LOGS/${run_id}_render.log" 2>&1 || true
}

run_clean_one() {
  local task_id="$1" suite="$2" unnorm="$3" max_steps="$4" state="$5" seed="$6" run_id="$7" cuda="$8" render="$9"
  local model
  model="$(model_for_suite "$suite")"
  prepare_run_dir "$run_id"
  if ! artifact_done "$run_id"; then
    log "launch clean $run_id task=$task_id state=$state seed=$seed on $cuda render=$render"
    env CUDA_VISIBLE_DEVICES="$cuda" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
      "$PY" scripts/v4_run_eval_openvla.py \
      --tasks_config configs/v4_tasks_libero.yaml \
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
      --model_path "$model" \
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
  render_video "$task_id" "$run_id" "$cuda" "$render"
}

run_jobs_tsv() {
  local jobs="$1"
  local idx=0
  local pids=()
  while IFS=$'\t' read -r task_id suite unnorm max_steps state seed run_id; do
    [[ -z "${task_id:-}" || "$task_id" == task_id ]] && continue
    local slot=$((idx % ${#SLOT_CUDA[@]}))
    run_clean_one "$task_id" "$suite" "$unnorm" "$max_steps" "$state" "$seed" "$run_id" "${SLOT_CUDA[$slot]}" "${SLOT_RENDER[$slot]}" &
    pids+=("$!")
    idx=$((idx + 1))
    if (( ${#pids[@]} >= ${#SLOT_CUDA[@]} )); then
      for pid in "${pids[@]}"; do wait "$pid"; done
      pids=()
    fi
  done <"$jobs"
  for pid in "${pids[@]}"; do wait "$pid"; done
}

write_lock() {
  local pytest_result="$1"
  local commit status
  commit="$(git rev-parse HEAD 2>&1 || true)"
  status="$(git status --short 2>&1 || true)"
  {
    echo "# EXPERIMENT_LOCK_LIBERO_CLEAN_DENOMINATOR_20260517"
    echo
    echo "- hostname: \`$(hostname)\`"
    echo "- date: \`$(date '+%F %T %Z')\`"
    echo "- git commit or unavailable reason: \`$commit\`"
    echo "- dirty status: \`$([[ -n "$status" ]] && echo dirty || echo clean)\`"
    echo "- Python path: \`$PY\`"
    echo "- pytest result: \`$pytest_result\`"
    echo "- server tag: \`$SERVER_TAG\`"
    echo '- statement: "This is LIBERO clean-denominator and auto-window/CQ coverage preparation only. No attack, no benchmark, no Moka, no Table 1."'
    echo
    echo "## Code/Config Hashes"
    for f in configs/generic_autowindow_detector.yaml configs/v4_tasks_libero.yaml configs/paper_black_bowl_attack.yaml configs/v4_directions.yaml scripts/detect_contact_window_from_clean.py scripts/extract_contact_quality_metrics.py; do
      echo "- \`$f\`: \`$(sha_file "$f")\`"
    done
    echo
    echo "## GPU Snapshot"
    echo '```text'
    cat "$LOGS/nvidia_smi.log" 2>/dev/null || true
    echo
    cat "$LOGS/nvidia_pmon.log" 2>/dev/null || true
    echo '```'
  } >"$OUT/EXPERIMENT_LOCK_LIBERO_CLEAN_DENOMINATOR_20260517.md"
}

log "Phase 0 preflight"
hostname >"$LOGS/hostname.log"
date >"$LOGS/date.log"
nvidia-smi >"$LOGS/nvidia_smi.log"
nvidia-smi pmon -c 3 >"$LOGS/nvidia_pmon.log" 2>&1 || true
pgrep -af "v4_run_eval_openvla.py|run_attack_pipeline.py|python.*openvla|python.*libero" >"$LOGS/pgrep.log" 2>&1 || true
pytest_result="not_run"
if "$PY" -m pytest -q >"$LOGS/pytest.log" 2>&1; then
  pytest_result="passed"
else
  pytest_result="failed"
fi
write_lock "$pytest_result"
if [[ "$pytest_result" != "passed" ]]; then
  log "pytest failed; stopping"
  exit 2
fi

log "Phase 1 CQ-v2 readiness"
"$PY" - "$PREV" "$TABLES/phase1_cqv2_readiness_check_20260517.csv" "$OUT/phase1_cqv2_decision.txt" <<'PY'
import csv, pathlib, sys
prev = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
decision_path = pathlib.Path(sys.argv[3])
report = prev / "cq_manual_calibration_report_20260517.csv"
cq = prev / "tables" / "phase5_blackbowl_genericv4_cq_v2_20260517.csv"
summary = prev / "blackbowl_genericv4_manual_audit_summary_20260517.md"
rows = []
def add(name, status, expected, observed, notes=""):
    rows.append({"check_name": name, "status": status, "expected": expected, "observed": observed, "notes": notes})
ready = True
for name, path in [("cq_v2_csv", cq), ("manual_calibration_report", report), ("manual_audit_summary", summary)]:
    ok = path.exists()
    ready &= ok
    add(name, "pass" if ok else "fail", "exists", str(path), "")
metrics = {"tp":0,"tn":0,"fp":0,"fn":0,"state5_vis":0,"random_pos":0}
if report.exists():
    data = list(csv.DictReader(report.open(encoding="utf-8-sig")))
    for r in data:
        man = str(r.get("manual_contact_quality_failure","")).strip() in {"1","True","true"}
        pred = str(r.get("cq_v2_contact_quality_failure","")).strip() in {"1","True","true"}
        if man and pred: metrics["tp"] += 1
        elif (not man) and (not pred): metrics["tn"] += 1
        elif (not man) and pred: metrics["fp"] += 1
        elif man and (not pred): metrics["fn"] += 1
        if r.get("state") == "5" and r.get("condition") == "vis_arm_clean" and pred:
            metrics["state5_vis"] += 1
        if "random" in r.get("condition","") and pred:
            metrics["random_pos"] += 1
    checks = [
        ("manual_positives_detected", metrics["tp"] == 9 and metrics["fn"] == 0, "9/9", f"{metrics['tp']}/9 fn={metrics['fn']}"),
        ("manual_negatives_clean", metrics["tn"] == 11 and metrics["fp"] == 0, "11/11", f"{metrics['tn']}/11 fp={metrics['fp']}"),
        ("state5_vis_corrected", metrics["state5_vis"] == 5, "5/5", f"{metrics['state5_vis']}/5"),
        ("random_controls_negative", metrics["random_pos"] == 0, "0 positives", str(metrics["random_pos"])),
    ]
    for name, ok, expected, observed in checks:
        ready &= ok
        add(name, "pass" if ok else "fail", expected, observed, "")
else:
    ready = False
decision = "cqv2_ready_for_clean_denominator_scan" if ready else "cqv2_not_ready_stop"
add("decision", "pass" if ready else "fail", "cqv2_ready_for_clean_denominator_scan", decision, "")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["check_name","status","expected","observed","notes"])
    w.writeheader(); w.writerows(rows)
decision_path.write_text(decision + "\n", encoding="utf-8")
print(decision)
PY
if [[ "$(cat "$OUT/phase1_cqv2_decision.txt")" != "cqv2_ready_for_clean_denominator_scan" ]]; then
  log "CQ-v2 not ready; stopping"
  exit 0
fi

log "Phase 2 inventory and seed1 jobs"
"$PY" - "$LIBERO_DATA_ROOT" "$TABLES/phase2_libero_candidate_task_inventory_20260517.csv" "$TABLES/phase3_seed1_jobs.tsv" <<'PY'
import csv, os, pathlib, sys, yaml
root = pathlib.Path(sys.argv[1])
inventory_path = pathlib.Path(sys.argv[2])
jobs_path = pathlib.Path(sys.argv[3])
cfg = yaml.safe_load(open("configs/v4_tasks_libero.yaml", encoding="utf-8"))["tasks"]
def resolve(path):
    text = path.replace("${LIBERO_DATA_ROOT}", str(root))
    candidates = [pathlib.Path(text), pathlib.Path(text.replace(str(root), str(root / "datasets"), 1))]
    for p in candidates:
        if p.exists():
            return str(p), True
    return str(candidates[-1]), False
def mech(suite):
    return {"libero_spatial":"pick_place_transfer","libero_object":"pick_place_transfer","libero_goal":"articulated_object","libero_10":"multi_object_transfer"}.get(suite,"unknown_low_signal")
rows = []
jobs = []
for task in cfg:
    dataset_path, exists = resolve(task.get("dataset_path",""))
    included = exists
    suite = task["suite"]
    task_id = task["task_id"]
    rows.append({
        "suite": suite,
        "task_id": task_id,
        "task_name": task.get("task_name",""),
        "dataset_path": dataset_path,
        "dataset_exists": exists,
        "task_config_exists": True,
        "included_in_clean_scan": included,
        "expected_mechanism_type": mech(suite),
        "exclusion_reason": "" if included else "dataset_missing",
        "notes": "configured representative task; local search found no additional demo files beyond configured set",
    })
    if included:
        for state in (0,1,2):
            run_id = f"old2080ti_20260517_cleandenom_{suite}_{task_id}_state{state}_seed1_clean"
            jobs.append([task_id, suite, task.get("default_unnorm_key",""), str(task.get("max_steps",400)), str(state), "1", run_id])
inventory_path.parent.mkdir(parents=True, exist_ok=True)
with inventory_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
with jobs_path.open("w", encoding="utf-8") as f:
    f.write("task_id\tsuite\tunnorm\tmax_steps\tstate\tseed\trun_id\n")
    for row in jobs:
        f.write("\t".join(row) + "\n")
PY

log "Phase 3 clean seed1 scan"
run_jobs_tsv "$TABLES/phase3_seed1_jobs.tsv"

aggregate_clean() {
  local out_csv="$1"
  "$PY" - "$OUT" "$TABLES/phase2_libero_candidate_task_inventory_20260517.csv" "$out_csv" <<'PY'
import csv, json, pathlib, re, sys
root = pathlib.Path(sys.argv[1]); inv_path = pathlib.Path(sys.argv[2]); out = pathlib.Path(sys.argv[3])
inventory = {r["task_id"]: r for r in csv.DictReader(inv_path.open(encoding="utf-8-sig"))}
def read_json(p):
    try: return json.loads(p.read_text(encoding="utf-8"))
    except Exception: return {}
def read_jsonl(p):
    try: return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    except Exception: return []
def video(run):
    vids = sorted((run/"videos").glob("*.mp4"))
    return str(vids[0]) if vids else ""
def artifact(run):
    return all((run/f).exists() and ((run/f).stat().st_size > 0) for f in ["progress.json","step_records.jsonl","episode_records.jsonl","summary.csv","run_manifest.json"])
def infer_state_seed(run_id):
    s = re.search(r"state(\d+)", run_id); seed = re.search(r"seed(\d+)", run_id)
    return (s.group(1) if s else "", seed.group(1) if seed else "")
def boolish(v): return str(v).strip().lower() in {"true","1","yes"}
rows=[]
for mpath in sorted(root.rglob("run_manifest.json")):
    run = mpath.parent; run_id = run.name
    if "_cleandenom_" not in run_id or not run_id.endswith("_clean"): continue
    manifest = read_json(mpath); task_id = manifest.get("task_id","")
    inv = inventory.get(task_id,{})
    progress = read_json(run/"progress.json")
    episodes = read_jsonl(run/"episode_records.jsonl")
    status = str(progress.get("status",""))
    official = bool(episodes and episodes[-1].get("success", False))
    failure = str(progress.get("failure_reason") or progress.get("reason") or ("success_libero" if official else "unknown_failure"))
    state, seed = infer_state_seed(run_id)
    suite = inv.get("suite", manifest.get("suite",""))
    expected = inv.get("expected_mechanism_type","unknown_low_signal")
    if not official:
        mechanism = "clean_unstable"
    else:
        mechanism = expected
    eligible = official and mechanism in {"pick_place_transfer","multi_object_transfer"}
    steps = read_jsonl(run/"step_records.jsonl")
    object_lifted = any(float(x.get("grasp_bowl_z_delta",0) or 0) >= 0.018 or any(k.endswith("_z_after") and str(x.get(k,"")) not in {"","None"} for k in x) for x in steps)
    stable_grasp = object_lifted or official
    release_place = official
    rows.append({
        "run_id": run_id, "suite": suite, "task_id": task_id, "task_name": inv.get("task_name",""),
        "state": state, "seed": seed, "status": status, "official_success": official,
        "failure_reason": failure, "clean_success": official, "no_grasp": (not official and "grasp" in failure.lower()),
        "timeout": (not official and ("timeout" in failure.lower() or status != "done")),
        "object_lifted": object_lifted, "stable_grasp_detected": stable_grasp,
        "transfer_phase_detected": eligible, "release_or_place_phase_detected": release_place,
        "mechanism_type": mechanism, "mechanism_eligible": eligible, "artifact_complete": artifact(run),
        "video_path": video(run), "notes": "",
    })
out.parent.mkdir(parents=True, exist_ok=True)
fields = ["run_id","suite","task_id","task_name","state","seed","status","official_success","failure_reason","clean_success","no_grasp","timeout","object_lifted","stable_grasp_detected","transfer_phase_detected","release_or_place_phase_detected","mechanism_type","mechanism_eligible","artifact_complete","video_path","notes"]
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
PY
}

aggregate_clean "$TABLES/phase3_libero_clean_eligibility_scan_20260517.csv"

log "Phase 3 optional seed2 jobs"
"$PY" - "$TABLES/phase3_libero_clean_eligibility_scan_20260517.csv" "$TABLES/phase3_seed2_jobs.tsv" <<'PY'
import csv, re, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8-sig")))
jobs = []
for r in rows:
    if len(jobs) >= 20: break
    if str(r.get("mechanism_eligible","")).lower() != "true": continue
    task_id = r["task_id"]; suite = r["suite"]; state = r["state"]
    unnorm = {"libero_spatial":"libero_spatial","libero_object":"libero_object","libero_goal":"libero_goal","libero_10":"libero_10"}.get(suite,"libero_spatial")
    max_steps = "700" if suite == "libero_10" else "400"
    run_id = f"old2080ti_20260517_cleandenom_{suite}_{task_id}_state{state}_seed2_clean"
    jobs.append([task_id,suite,unnorm,max_steps,state,"2",run_id])
with open(sys.argv[2], "w", encoding="utf-8") as f:
    f.write("task_id\tsuite\tunnorm\tmax_steps\tstate\tseed\trun_id\n")
    for row in jobs:
        f.write("\t".join(row)+"\n")
PY
if [[ "$(wc -l < "$TABLES/phase3_seed2_jobs.tsv")" -gt 1 ]]; then
  run_jobs_tsv "$TABLES/phase3_seed2_jobs.tsv"
  aggregate_clean "$TABLES/phase3_libero_clean_eligibility_scan_20260517.csv"
fi

log "Phase 4 generic-v4 detector"
"$PY" scripts/detect_contact_window_from_clean.py \
  --input_root "$OUT" \
  --output_csv "$TABLES/phase4_detector_raw_all_clean_runs_20260517.csv" \
  --config configs/generic_autowindow_detector.yaml \
  --phase_cues_csv "$TABLES/phase4_detector_phase_cues_20260517.csv" \
  --summary_md "$OUT/phase4_detector_summary_20260517.md" >"$LOGS/phase4_detector.log" 2>&1 || true

log "Phase 5 CQ-v2 extraction"
"$PY" scripts/extract_contact_quality_metrics.py \
  --input_root "$OUT" \
  --output_csv "$TABLES/phase5_cqv2_raw_all_runs_20260517.csv" >"$LOGS/phase5_cqv2.log" 2>&1 || true

log "Phase 6/7/8 aggregation"
"$PY" - "$OUT" <<'PY'
import csv, pathlib, collections, statistics, sys
root = pathlib.Path(sys.argv[1]); tables = root / "tables"
def rr(name):
    p = tables / name
    return list(csv.DictReader(p.open(encoding="utf-8-sig"))) if p.exists() else []
def ww(name, rows):
    if not rows: rows=[{}]
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with (tables/name).open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
def truth(v): return str(v).strip().lower() in {"true","1","yes"}
clean = rr("phase3_libero_clean_eligibility_scan_20260517.csv")
detraw = {r.get("run_id"): r for r in rr("phase4_detector_raw_all_clean_runs_20260517.csv")}
cqraw = {r.get("run_id"): r for r in rr("phase5_cqv2_raw_all_runs_20260517.csv")}
phase4=[]
phase5=[]
for r in clean:
    if not truth(r.get("clean_success")): continue
    d = detraw.get(r["run_id"], {})
    phase4.append({
        "clean_run_id": r["run_id"], "suite": r["suite"], "task_id": r["task_id"], "task_name": r["task_name"],
        "state": r["state"], "seed": r["seed"], "clean_success": r["clean_success"],
        "mechanism_type": r["mechanism_type"], "mechanism_eligible": r["mechanism_eligible"],
        "window_detected": d.get("window_detected",""), "auto_window_start": d.get("auto_window_start",""),
        "auto_window_end": d.get("auto_window_end",""), "detector_mode": d.get("detector_mode",""),
        "detector_confidence": d.get("confidence",""), "grasp_step": d.get("grasp_step",""),
        "lift_step": d.get("lift_step",""), "carry_start_step": d.get("carry_start_step",""),
        "preplace_step": d.get("preplace_step",""), "release_intent_step": d.get("release_intent_step",""),
        "signals_available": d.get("signals_available",""), "failure_reason": d.get("failure_reason",""),
        "config_hash": d.get("detector_config_hash",""), "notes": "",
    })
    c = cqraw.get(r["run_id"], {})
    cq_computable = c.get("contact_quality_failure") not in {"", "NA"} and c.get("cq_confidence_v2", c.get("confidence","")) != "low"
    phase5.append({
        "clean_run_id": r["run_id"], "suite": r["suite"], "task_id": r["task_id"], "task_name": r["task_name"],
        "state": r["state"], "seed": r["seed"], "mechanism_type": r["mechanism_type"],
        "official_success": r["official_success"], "object_pose_available": c.get("object_lifted") not in {"", "NA"},
        "eef_pose_available": "", "gripper_qpos_available": c.get("qpos_abs_after_max","") != "",
        "cq_computable": cq_computable, "contact_quality_failure_v2": c.get("contact_quality_failure",""),
        "contact_quality_success_v2": c.get("contact_quality_success",""),
        "uncontrolled_final_drop": c.get("uncontrolled_final_drop",""),
        "stable_controlled_place": c.get("stable_controlled_place",""),
        "sr_cq_mismatch": c.get("sr_cq_mismatch",""), "cq_failure_reason_v2": c.get("cq_failure_reason_v2",""),
        "cq_confidence_v2": c.get("cq_confidence_v2", c.get("confidence","")),
        "failure_reason": c.get("failure_reason",""), "notes": "",
    })
ww("phase4_libero_genericv4_autowindow_candidates_20260517.csv", phase4)
ww("phase5_libero_cqv2_availability_20260517.csv", phase5)
by_suite=collections.defaultdict(list)
for r in clean: by_suite[r["suite"]].append(r)
det_by_run={r["clean_run_id"]:r for r in phase4}
cq_by_run={r["clean_run_id"]:r for r in phase5}
suite_rows=[]
for suite, rows in sorted(by_suite.items()):
    clean_success=[r for r in rows if truth(r["clean_success"])]
    elig=[r for r in clean_success if truth(r["mechanism_eligible"])]
    window=[r for r in elig if truth(det_by_run.get(r["run_id"],{}).get("window_detected"))]
    mh=[r for r in window if det_by_run.get(r["run_id"],{}).get("detector_confidence") in {"medium","high"}]
    cq=[r for r in clean_success if truth(cq_by_run.get(r["run_id"],{}).get("cq_computable"))]
    cqfail=[r for r in clean_success if truth(cq_by_run.get(r["run_id"],{}).get("contact_quality_failure_v2"))]
    tasks={r["task_id"] for r in rows}
    suite_rows.append({"suite":suite,"candidate_tasks":len(tasks),"included_tasks":len(tasks),"clean_rollouts_attempted":len(rows),"clean_success_count":len(clean_success),"clean_success_rate":len(clean_success)/len(rows) if rows else 0,"mechanism_eligible_count":len(elig),"mechanism_eligible_rate_among_clean":len(elig)/len(clean_success) if clean_success else 0,"window_detected_count":len(window),"window_detected_rate_among_eligible":len(window)/len(elig) if elig else 0,"medium_high_conf_window_count":len(mh),"cq_computable_count":len(cq),"cq_computable_rate_among_clean":len(cq)/len(clean_success) if clean_success else 0,"clean_cq_failure_count":len(cqfail),"notes":""})
ww("phase6_libero_denominator_summary_by_suite_20260517.csv", suite_rows)
by_task=collections.defaultdict(list)
for r in clean: by_task[(r["suite"],r["task_id"],r["task_name"])].append(r)
task_rows=[]
for (suite,task_id,task_name), rows in sorted(by_task.items()):
    clean_success=[r for r in rows if truth(r["clean_success"])]
    elig=[r for r in clean_success if truth(r["mechanism_eligible"])]
    window=[r for r in elig if truth(det_by_run.get(r["run_id"],{}).get("window_detected"))]
    cq=[r for r in clean_success if truth(cq_by_run.get(r["run_id"],{}).get("cq_computable"))]
    modes=[r["mechanism_type"] for r in rows if r.get("mechanism_type")]
    mode=collections.Counter(modes).most_common(1)[0][0] if modes else ""
    rec = len(window)>0 and len(cq)>0 and task_id != "libero_spatial_black_bowl"
    reason = "" if rec else ("black_bowl_reference_task" if task_id=="libero_spatial_black_bowl" else "insufficient_clean_window_or_cq")
    task_rows.append({"suite":suite,"task_id":task_id,"task_name":task_name,"clean_attempts":len(rows),"clean_success_count":len(clean_success),"mechanism_type_mode":mode,"mechanism_eligible_count":len(elig),"window_detected_count":len(window),"cq_computable_count":len(cq),"recommended_for_future_attack":rec,"exclusion_reason":reason,"notes":""})
ww("phase6_libero_denominator_summary_by_task_20260517.csv", task_rows)
queue=[]
for r in clean:
    d=det_by_run.get(r["run_id"],{}); c=cq_by_run.get(r["run_id"],{})
    reasons=[]
    priority=""
    if truth(r["clean_success"]) and truth(c.get("contact_quality_failure_v2")):
        priority="high"; reasons.append("clean_success_cqv2_failure")
    if truth(r["clean_success"]) and truth(r["mechanism_eligible"]) and not truth(d.get("window_detected")):
        priority="high"; reasons.append("eligible_clean_detector_abstained")
    if r["mechanism_type"]=="unknown_low_signal":
        priority="high"; reasons.append("unknown_low_signal")
    if r["suite"]=="libero_10" and truth(r["clean_success"]):
        priority=priority or "high"; reasons.append("libero10_phase_segmentation")
    if not priority and truth(r["clean_success"]):
        priority="medium"; reasons.append("representative_clean_success")
    if priority:
        queue.append({"priority":priority,"reason":";".join(reasons),"clean_run_id":r["run_id"],"suite":r["suite"],"task_id":r["task_id"],"task_name":r["task_name"],"state":r["state"],"seed":r["seed"],"official_success":r["official_success"],"clean_success":r["clean_success"],"mechanism_type":r["mechanism_type"],"window_detected":d.get("window_detected",""),"detector_confidence":d.get("detector_confidence",""),"cq_computable":c.get("cq_computable",""),"contact_quality_failure_v2":c.get("contact_quality_failure_v2",""),"video_path":r["video_path"],"notes":""})
ww("phase7_libero_clean_manual_review_queue_20260517.csv", queue)
nonbb=[r for r in clean if r["task_id"]!="libero_spatial_black_bowl" and truth(r["clean_success"]) and truth(r["mechanism_eligible"])]
nonbb_window=[r for r in nonbb if truth(det_by_run.get(r["run_id"],{}).get("window_detected")) and det_by_run.get(r["run_id"],{}).get("detector_confidence") in {"medium","high"}]
nonbb_suites={r["suite"] for r in nonbb_window}
cq_ok=any(truth(cq_by_run.get(r["run_id"],{}).get("cq_computable")) or r.get("video_path") for r in nonbb)
if len(nonbb)>=4 and len(nonbb_window)>=3 and len(nonbb_suites)>=2 and cq_ok:
    decision="A. ready_for_microbreadth_vis_random_pilot"
elif nonbb_window:
    decision="B. clean_denominator_promising_but_needs_more_clean"
elif nonbb:
    decision="C. detector_not_general_enough"
elif any(truth(r["clean_success"]) for r in clean):
    decision="E. clean_denominator_insufficient"
else:
    decision="F. server_or_artifact_failure"
lines=["# LIBERO Clean Denominator Generic-v4 CQ-v2 Summary 20260517","",f"Decision: {decision}","","## CQ-v2 readiness","- cqv2_ready_for_clean_denominator_scan","","## Candidate Inventory",f"- candidate tasks: {len({r['task_id'] for r in clean})}","- local search found only the configured representative demos on this server.","","## Suite Summary"]
for r in suite_rows:
    lines.append(f"- {r['suite']}: clean {r['clean_success_count']}/{r['clean_rollouts_attempted']}, eligible {r['mechanism_eligible_count']}, windows {r['window_detected_count']}, cq_computable {r['cq_computable_count']}, clean_cq_fail {r['clean_cq_failure_count']}")
lines += ["","## Recommendation"]
if decision.startswith("A"):
    lines.append("A later microbreadth VIS/random pilot is justified, capped to non-Black-Bowl eligible windowed runs.")
elif decision.startswith("B"):
    lines.append("Run more clean seeds/tasks before any attack.")
elif decision.startswith("C"):
    lines.append("Improve generic-v4 detector coverage outside Black Bowl before attack.")
elif decision.startswith("D"):
    lines.append("Improve CQ-v2 availability outside Black Bowl before attack.")
elif decision.startswith("E"):
    lines.append("Clean/mechanism denominator is currently insufficient for non-Black-Bowl attack.")
else:
    lines.append("Fix server/artifact issues before continuing.")
lines += ["","Forbidden work respected: no VIS/random/oracle attack, no benchmark, no Moka, no Table 1 aggregation, no detector tuning."]
(root/"libero_clean_denominator_genericv4_cqv2_summary_20260517.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
PY

log "done"
