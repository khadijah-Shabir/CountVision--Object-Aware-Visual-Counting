#!/usr/bin/env bash
# Usage: run_jobs.sh MODEL JOBS OUT_PREFIX "GPU_LIST" [GPUS_PER_REPLICA] [BATCH_SIZE] [SAMPLING]
set -euo pipefail

model=${1:?model directory required}
jobs=${2:?job manifest required}
out=${3:?output prefix required}
read -r -a gpus <<< "${4:?GPU list required}"
per_replica=${5:-1}
batch=${6:-16}
sampling=${7:-default}
lv_python=${PY_LV:-../.venv_lvlm/bin/python}

if [[ ! $per_replica =~ ^[1-9][0-9]*$ ]] ||
   (( ${#gpus[@]} == 0 || ${#gpus[@]} % per_replica != 0 )); then
  echo "GPU list must divide evenly into positive replica groups" >&2
  exit 2
fi

replicas=$(( ${#gpus[@]} / per_replica ))
script_dir=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$(dirname "$out")"
pids=()
for ((i=0; i<replicas; i++)); do
  group=$(IFS=,; echo "${gpus[*]:$((i * per_replica)):$per_replica}")
  CUDA_VISIBLE_DEVICES=$group "$lv_python" "$script_dir/lvlm_worker.py" \
      --model "$model" --jobs "$jobs" --out "${out}_$i.jsonl" \
      --shard "$i/$replicas" --batch "$batch" --sampling "$sampling" > "${out}_$i.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if (( failed )); then
  echo "One or more replicas failed. See ${out}_*.log" >&2
  exit 1
fi
echo "Completed $out"
