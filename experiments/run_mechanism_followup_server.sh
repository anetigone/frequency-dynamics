#!/usr/bin/env bash
# Default: data-free plans. Formal work runs only in the selected server stage.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
CIFAR_ROOT="${CIFAR_ROOT:-$REPO_ROOT}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/experiments/results/empirical}"
SOURCE_ROOT="${SOURCE_ROOT:-$RESULT_ROOT/natural_moments_cifar10}"
THREADS="${THREADS:-1}"
FEATURE_BATCH="${FEATURE_BATCH:-256}"
DRIVERS="$REPO_ROOT/experiments/code/empirical"

COMMON=(--data-root "$CIFAR_ROOT" --device "$DEVICE" --threads "$THREADS"
        --feature-batch "$FEATURE_BATCH")

confirmation() {
    "$PYTHON_BIN" -B "$DRIVERS/natural_rebound_confirmation.py" "$1" \
        --output "$RESULT_ROOT/natural_confirmation_cifar10" \
        --seeds 40 41 42 43 44 "${COMMON[@]}"
}

metric() {
    "$PYTHON_BIN" -B "$DRIVERS/natural_metric_intervention.py" "$1" \
        --source "$SOURCE_ROOT" --output "$RESULT_ROOT/natural_metric_cifar10" \
        --seeds 30 31 32 "${COMMON[@]}"
}

heads() {
    "$PYTHON_BIN" -B "$DRIVERS/natural_head_counterfactual.py" "$1" \
        --source "$SOURCE_ROOT" --output "$RESULT_ROOT/natural_head_counterfactual_cifar10" \
        --seeds 30 31 32 --epochs 3000 5000 "${COMMON[@]}"
}

finish_confirmation() {
    confirmation train
    confirmation evaluate
    confirmation summarize
}

finish_metric() {
    metric train
    metric evaluate
    metric summarize
}

finish_heads() {
    heads evaluate
    heads summarize
}

stage="${1:-plan}"
if [[ $# -gt 1 ]]; then
    printf 'Usage: bash experiments/run_mechanism_followup_server.sh [plan|confirm|metric|heads|all|summary|events|geometry-plan|geometry|smoke]\n' >&2
    exit 2
fi
case "$stage" in
    plan)
        confirmation plan
        metric plan
        heads plan
        ;;
    confirm) finish_confirmation ;;
    metric) finish_metric ;;
    heads) finish_heads ;;
    all)
        finish_heads
        finish_metric
        finish_confirmation
        ;;
    summary)
        confirmation summarize
        metric summarize
        heads summarize
        ;;
    events) confirmation analyze ;;
    geometry-plan) confirmation geometry-plan ;;
    geometry)
        confirmation geometry
        confirmation geometry-summary
        ;;
    smoke)
        confirmation smoke
        metric smoke
        heads smoke
        ;;
    *)
        printf 'Unknown stage: %s\nChoose plan, confirm, metric, heads, all, summary, events, geometry-plan, geometry, or smoke.\n' "$stage" >&2
        exit 2
        ;;
esac
