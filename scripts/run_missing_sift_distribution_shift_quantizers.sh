#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:-sift-128-euclidean}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/benchmark/results/distribution_shift}"
QUANTIZER_DIR="${QUANTIZER_DIR:-$REPO_ROOT/benchmark/algorithms/quantizer}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/distribution_shift}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$LOG_DIR"

mapfile -t ALL_ALGOS < <(find "$QUANTIZER_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
mapfile -t DONE_ALGOS < <(
  find "$RESULTS_DIR" -maxdepth 1 -type f -name "${DATASET}_*.json" -printf '%f\n' \
    | sed -E "s/^${DATASET}_(.*)\.json$/\1/" \
    | sort -u
)

missing_algos=()
for algo in "${ALL_ALGOS[@]}"; do
  found=0
  for done in "${DONE_ALGOS[@]:-}"; do
    if [[ "$algo" == "$done" ]]; then
      found=1
      break
    fi
  done
  if [[ "$found" -eq 0 ]]; then
    missing_algos+=("$algo")
  fi
done

if [[ ${#missing_algos[@]} -eq 0 ]]; then
  echo "No missing distribution-shift quantizer runs found for dataset '$DATASET'."
  exit 0
fi

echo "Dataset: $DATASET"
echo "Missing quantizer algorithms: ${missing_algos[*]}"
echo

for algo in "${missing_algos[@]}"; do
  stamp="$(date +%Y%m%d_%H%M%S)"
  build_log="$LOG_DIR/${DATASET}_${algo}_build_${stamp}.log"
  run_log="$LOG_DIR/${DATASET}_${algo}_distribution_shift_${stamp}.log"

  echo "============================================================"
  echo "[$(date '+%F %T')] Rebuilding image for $algo"
  echo "Build log: $build_log"
  echo "============================================================"
  "$PYTHON_BIN" "$REPO_ROOT/run.py" \
    --algorithm "$algo" \
    --build-images \
    --force-rebuild 2>&1 | tee "$build_log"

  echo
  echo "============================================================"
  echo "[$(date '+%F %T')] Running distribution shift benchmark for $algo"
  echo "Run log: $run_log"
  echo "============================================================"
  "$PYTHON_BIN" "$REPO_ROOT/run.py" \
    --dataset "$DATASET" \
    --algorithm "$algo" \
    --distribution-shift-test 2>&1 | tee "$run_log"

  echo
  echo "Finished $algo"
  echo

done

echo "All missing quantizer distribution-shift runs completed for dataset '$DATASET'."
