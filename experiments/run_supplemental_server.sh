#!/usr/bin/env bash
# Run from any directory. The default only prints data-free experiment plans.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
CIFAR_ROOT="${CIFAR_ROOT:-$REPO_ROOT}"
DTD_ROOT="${DTD_ROOT:-$REPO_ROOT/experiments/data/dtd_hf}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/experiments/results/empirical}"
THREADS="${THREADS:-1}"
FEATURE_BATCH="${FEATURE_BATCH:-256}"
KNN_BATCH="${KNN_BATCH:-128}"
NATURAL_DRIVER="$REPO_ROOT/experiments/code/empirical/natural_moment_experiment.py"
TEXTURE_DRIVER="$REPO_ROOT/experiments/code/empirical/texture_rebound_sweep.py"

COMMON=(--seeds 30 31 32 --epochs 5000 --batch-size 256 --lr .002
        --device "$DEVICE" --threads "$THREADS" --feature-batch "$FEATURE_BATCH"
        --knn-batch "$KNN_BATCH")

run_natural() {
    local dataset="$1" phase="$2" until="${3:-}"
    local data_root="$CIFAR_ROOT"
    if [[ "$dataset" == dtd ]]; then data_root="$DTD_ROOT"; fi
    local command=("$PYTHON_BIN" -B "$NATURAL_DRIVER" "$phase"
                   --dataset "$dataset" --data-root "$data_root"
                   --output "$RESULT_ROOT/natural_moments_$dataset" "${COMMON[@]}")
    if [[ -n "$until" ]]; then command+=(--until-epoch "$until"); fi
    "${command[@]}"
}

finish_natural() {
    local dataset="$1" until="$2"
    run_natural "$dataset" train "$until"
    run_natural "$dataset" evaluate "$until"
    # Summarize in its own process, without loading Torch first.
    run_natural "$dataset" summarize "$until"
}

stage="${1:-plan}"
if [[ $# -gt 1 ]]; then
    printf 'Usage: bash experiments/run_supplemental_server.sh [plan|first|long|dtd-long|textures|summary|smoke]\n' >&2
    exit 2
fi
case "$stage" in
    plan)
        run_natural cifar10 plan 200
        run_natural dtd plan 200
        "$PYTHON_BIN" -B "$TEXTURE_DRIVER" --stage plan --output "$RESULT_ROOT/texture_rebound_50_seeds"
        ;;
    first)
        finish_natural cifar10 200
        finish_natural dtd 200
        ;;
    long)
        finish_natural cifar10 5000
        ;;
    dtd-long)
        finish_natural dtd 5000
        ;;
    textures)
        "$PYTHON_BIN" -B "$TEXTURE_DRIVER" --stage all --output "$RESULT_ROOT/texture_rebound_50_seeds"
        ;;
    summary)
        run_natural cifar10 summarize
        run_natural dtd summarize
        "$PYTHON_BIN" -B "$TEXTURE_DRIVER" --stage summary --output "$RESULT_ROOT/texture_rebound_50_seeds"
        ;;
    smoke)
        "$PYTHON_BIN" -B "$NATURAL_DRIVER" smoke --device cpu
        ;;
    *)
        printf 'Unknown stage: %s\n' "$stage" >&2
        printf 'Choose plan, first, long, dtd-long, textures, summary, or smoke.\n' >&2
        exit 2
        ;;
esac
