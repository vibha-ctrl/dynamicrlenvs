#!/usr/bin/env bash

set -u

NUM_ENVS="${1:-4096}"
MAX_ITERATIONS="${2:-15000}"
RUN_TAG="${3:-$(date +%Y%m%d_%H%M%S)}"

ISAAC_ROOT="/home/apple/IsaacLab"
ENV_ROOT="/home/apple/conveyor_belt"
TRAIN_SCRIPT="${ENV_ROOT}/conveyor_belt/train.py"
VENV_ACTIVATE="${ISAAC_ROOT}/vibhaenv/bin/activate"

OUT_LOG="${ENV_ROOT}/train_output_${RUN_TAG}.log"
META_LOG="${ENV_ROOT}/train_meta_${RUN_TAG}.log"
RESOURCE_LOG="${ENV_ROOT}/train_resources_${RUN_TAG}.log"
PID_FILE="${ENV_ROOT}/train_${RUN_TAG}.pid"
LATEST_INFO="${ENV_ROOT}/train_latest_run.txt"

START_TS="$(date --iso-8601=seconds)"

{
  echo "run_tag=${RUN_TAG}"
  echo "start_time=${START_TS}"
  echo "num_envs=${NUM_ENVS}"
  echo "max_iterations=${MAX_ITERATIONS}"
  echo "train_script=${TRAIN_SCRIPT}"
  echo "out_log=${OUT_LOG}"
  echo "resource_log=${RESOURCE_LOG}"
  echo "pid_file=${PID_FILE}"
} > "${META_LOG}"

(
  cd "${ISAAC_ROOT}" || exit 201
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}" || exit 202
  cd "${ENV_ROOT}" || exit 203
  exec python -u "${TRAIN_SCRIPT}" \
    --num_envs "${NUM_ENVS}" \
    --max_iterations "${MAX_ITERATIONS}" \
    --headless
) >> "${OUT_LOG}" 2>&1 &

TRAIN_PID=$!
echo "${TRAIN_PID}" > "${PID_FILE}"
echo "${TRAIN_PID}" > "${ENV_ROOT}/train_active.pid"

{
  echo "pid=${TRAIN_PID}"
  echo "status=started"
} >> "${META_LOG}"

{
  echo "run_tag=${RUN_TAG}"
  echo "pid=${TRAIN_PID}"
  echo "start_time=${START_TS}"
  echo "meta_log=${META_LOG}"
  echo "out_log=${OUT_LOG}"
  echo "resource_log=${RESOURCE_LOG}"
  echo "pid_file=${PID_FILE}"
} > "${LATEST_INFO}"

while kill -0 "${TRAIN_PID}" 2>/dev/null; do
  TS="$(date --iso-8601=seconds)"
  MEM_LINE="$(free -m | awk 'NR==2{printf("mem_used_mb=%s mem_free_mb=%s mem_avail_mb=%s",$3,$4,$7)}')"
  SWAP_LINE="$(free -m | awk 'NR==3{printf("swap_used_mb=%s swap_free_mb=%s",$3,$4)}')"
  GPU_LINE="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null || echo "gpu_unavailable")"
  echo "${TS} | ${MEM_LINE} | ${SWAP_LINE} | gpu=${GPU_LINE}" >> "${RESOURCE_LOG}"
  sleep 30
done

wait "${TRAIN_PID}"
EXIT_CODE=$?
END_TS="$(date --iso-8601=seconds)"

if (( EXIT_CODE > 128 )); then
  EXIT_SIGNAL="$((EXIT_CODE - 128))"
else
  EXIT_SIGNAL="none"
fi

{
  echo "end_time=${END_TS}"
  echo "exit_code=${EXIT_CODE}"
  echo "exit_signal=${EXIT_SIGNAL}"
  echo "status=finished"
} >> "${META_LOG}"

echo "Training monitor finished for run_tag=${RUN_TAG} pid=${TRAIN_PID} exit_code=${EXIT_CODE} exit_signal=${EXIT_SIGNAL}"
