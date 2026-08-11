#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs}"
BASELINE_RUN_ID="${BASELINE_RUN_ID:-grpo_baseline}"
Q0_RUN_ID="${Q0_RUN_ID:-q0_grpo}"
MOPD_RUN_ID="${MOPD_RUN_ID:-mopd}"
EVAL_OUTPUT="${EVAL_OUTPUT:-experiments/q0_mopd/results/eval_results.json}"

# the trainers do sibling imports (import train_grpo / import eval); those libs now live
# in core/, so put it on the path before we launch anything
export PYTHONPATH="$PWD/core:${PYTHONPATH:-}"

# extra arguments are whitespace-separated and are never evaluated as shell code.
read -r -a baseline_extra_args <<< "${BASELINE_EXTRA_ARGS:-}"
read -r -a q0_extra_args <<< "${Q0_EXTRA_ARGS:-}"
read -r -a mopd_extra_args <<< "${MOPD_EXTRA_ARGS:-}"
read -r -a eval_extra_args <<< "${EVAL_EXTRA_ARGS:-}"

CHILD_PID=""

# stop and reap only the child started by this script.
cleanup_current_child() {
    if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
        kill -TERM "$CHILD_PID" 2>/dev/null || true
        wait "$CHILD_PID" || true
    fi
    CHILD_PID=""
}

on_signal() {
    cleanup_current_child
    trap - EXIT
    exit 130
}

trap cleanup_current_child EXIT
trap on_signal INT TERM

show_gpu_processes() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
    else
        printf 'run_all: nvidia-smi is unavailable; skipping gpu visibility check.\n'
    fi
}

run_child() {
    local name="$1"
    local is_training="$2"
    local status
    shift 2

    printf 'run_all: starting %s\n' "$name"
    "$@" &
    CHILD_PID=$!
    if wait "$CHILD_PID"; then
        status=0
    else
        status=$?
    fi
    CHILD_PID=""

    if [[ "$is_training" == "true" ]]; then
        printf 'run_all: %s exited; its process allocations are released.\n' "$name"
        show_gpu_processes
    fi

    if (( status != 0 )); then
        printf 'run_all: %s failed with status %d.\n' "$name" "$status" >&2
        exit "$status"
    fi
}

run_child "baseline grpo training" true \
    "$PYTHON" -u core/train_grpo.py \
    --output-dir "$OUTPUT_ROOT" \
    --run-id "$BASELINE_RUN_ID" \
    "${baseline_extra_args[@]}"

run_child "q0 grpo training" true \
    "$PYTHON" -u experiments/q0_mopd/train_q0_grpo.py \
    --output-dir "$OUTPUT_ROOT" \
    --run-id "$Q0_RUN_ID" \
    "${q0_extra_args[@]}"

run_child "snapshot distillation" true \
    "$PYTHON" -u experiments/q0_mopd/train_mopd.py \
    --output-dir "$OUTPUT_ROOT" \
    --run-id "$MOPD_RUN_ID" \
    --q0-run-dir "$OUTPUT_ROOT/$Q0_RUN_ID" \
    "${mopd_extra_args[@]}"

run_child "evaluation" false \
    "$PYTHON" -u core/eval.py \
    --baseline-run-dir "$OUTPUT_ROOT/$BASELINE_RUN_ID" \
    --q0-run-dir "$OUTPUT_ROOT/$Q0_RUN_ID" \
    --mopd-run-dir "$OUTPUT_ROOT/$MOPD_RUN_ID" \
    --output "$EVAL_OUTPUT" \
    "${eval_extra_args[@]}"

printf 'run_all: all stages completed; results are in %s.\n' "$EVAL_OUTPUT"
