
#!/usr/bin/env bash
set -euo pipefail

export EXP_ROOT="results/EASE"
LOG_ROOT="${EXP_ROOT}/log"
mkdir -p "${LOG_ROOT}"

SEEDS=(1111 1112 1113)
GPUS=(0 1 2) 


cleanup () {
  echo "[train.sh] Interrupted. Killing child processes..."
  pkill -P $$ || true
  exit 130
}
trap cleanup INT TERM

run_dataset () {
  local name=$1
  local cfg=$2

  echo "============================================================"
  echo "[dataset] ${name}  cfg=${cfg}  start=$(date)"
  echo "============================================================"

  for idx in 0 1 2; do
    local seed="${SEEDS[$idx]}"
    local gpu="${GPUS[$idx]}"
    local logf="${LOG_ROOT}/${name}_seed${seed}_gpu${gpu}.txt"

    echo "[launch] ${name} seed=${seed} gpu=${gpu} -> ${logf}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
      python -u train_EMHC.py \
      --config_file "${cfg}" \
      --seed "${seed}" \
      > "${logf}" 2>&1 &

    echo "[pid] ${name} seed=${seed} gpu=${gpu} pid=$!"
  done

  echo "[dataset] ${name} waiting all seeds..."
  wait
  echo "[dataset] ${name} done=$(date)"
}

run_dataset "mosi"  "configs/train_mosi.yaml"
run_dataset "sims"  "configs/train_sims.yaml"
run_dataset "mosei" "configs/train_mosei.yaml"

echo "[train.sh] ALL DONE at $(date)"