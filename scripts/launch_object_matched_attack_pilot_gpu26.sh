#!/bin/bash
# Official Object Matched Attack Pilot — GPU2,6 only
# 2 tasks × 5 states × 4 conditions = 40 rollouts
# Same detector trigger for ALL conditions
set -e

REPO=/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524
PY=/data/aviary/envs/openvla_official_libero_20260525/bin/python
OUT=/data/liuyu/outputs/milestone_2f_object_detector_matched_attack_pilot_20260529
DET=/data/liuyu/outputs/milestone_2e3_object100_visual_proprio_no_step_20260527/models/ProprioNoStep_baseline.pt
MODEL=/data/aviary/models/openvla/openvla-7b-finetuned-libero-object

DET_SHA256=4b3f3d479d6bbb92b2bd15cffec0be587bf221dc81663aaff93e44afdd9c7b1f
RUNNER_COMMIT=da4b297

export CUDA_VISIBLE_DEVICES=2,6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

mkdir -p ${OUT}/{logs,tables,reports}

echo "=== Object Matched Attack Pilot ==="
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Detector: $DET (sha256: $DET_SHA256)"
echo "Runner commit: $RUNNER_COMMIT"
echo "Output: $OUT"
echo ""

# Preflight
nvidia-smi --query-gpu=index,pci.bus_id,utilization.gpu,memory.used --format=csv,noheader -i 2,6
echo ""
dmesg -T 2>/dev/null | grep -i "Xid" | tail -3
echo ""

# Record preflight
cat > ${OUT}/reports/PREFLIGHT_GPU26.txt << PREFLIGHT
GPU2,6 attack pilot preflight — $(date)
Detector: ${DET} (sha256: ${DET_SHA256})
Runner commit: ${RUNNER_COMMIT}
PREFLIGHT
nvidia-smi -i 2,6 >> ${OUT}/reports/PREFLIGHT_GPU26.txt

# cream_cheese=task1, milk=task7
# 4 conditions, 5 states each
for COND in clean oracle_open random_control VIS_targeted; do
  for TID in 1 7; do
    WORKER="atk26_${COND}_t${TID}"
    echo "=== ${COND} task_start=${TID} $(date) ==="
    cd ${REPO}
    ${PY} -u scripts/run_official_eval_artifact_rich.py \
      --model_path ${MODEL} \
      --task_suite_name libero_object \
      --task_start ${TID} --task_count 1 \
      --num_trials_per_task 5 \
      --worker_id ${WORKER} \
      --save_rgb --save_step_records --save_privileged_teacher_state \
      --cuda_visible_devices 2,6 --render_gpu_device_id 2 \
      --output_root ${OUT} \
      --run_id_prefix "atk26_${COND}" \
      --detector_path ${DET} \
      --detector_hazard_threshold 0.1 --detector_trigger_duration 5 \
      --detector_cooldown 0 \
      --attack_condition ${COND}
    echo "  Done: ${COND} task_start=${TID} $(date)"
  done
done

echo "=== ATTACK PILOT COMPLETE $(date) ==="
nvidia-smi -i 2,6 >> ${OUT}/reports/PREFLIGHT_GPU26.txt
dmesg -T 2>/dev/null | grep -i "Xid" | tail -5 >> ${OUT}/reports/PREFLIGHT_GPU26.txt
