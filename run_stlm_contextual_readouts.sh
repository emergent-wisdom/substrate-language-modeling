#!/usr/bin/env bash

# Re-evaluate the three published comparison conditions, train fresh causal
# token readouts (TR-C) over both frozen SP state sequences, and collect the
# fixed five-condition comparison. The older trace runner remains available
# only for reproducing the superseded TR-H residual diagnostic.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_DIR="$SCRIPT_DIR/stlm_prototype"
PY="$SCRIPT_DIR/.venv/bin/python"

if [ ! -x "$PY" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PY="python3"
    else
        PY="python"
    fi
fi

export STLM_EXPERIMENT_TYPE="contextual_readout"
export STLM_SOURCE_RUN_ID="${STLM_SOURCE_RUN_ID:-data5x_seed42}"
export STLM_DIRECT_BASELINE_RUN_ID="${STLM_DIRECT_BASELINE_RUN_ID:-baseline5x_seed42}"
export STLM_RUN_ID="${STLM_RUN_ID:-tr_c_replication_seed42_t128}"
export STLM_SEED="${STLM_SEED:-42}"
export STLM_MAX_SENTENCES="${STLM_MAX_SENTENCES:-250000}"
export STLM_CONTEXTUAL_TRAIN_WINDOWS="${STLM_CONTEXTUAL_TRAIN_WINDOWS:-100000}"
export STLM_READOUT_EVAL_TARGETS="${STLM_READOUT_EVAL_TARGETS:-1024}"
export STLM_CONTEXTUAL_PIXEL_BATCH="${STLM_CONTEXTUAL_PIXEL_BATCH:-8}"
export STLM_CONTEXTUAL_DESCRIPTION_BATCH="${STLM_CONTEXTUAL_DESCRIPTION_BATCH:-8}"
export STLM_CONTEXTUAL_PIXEL_MICROBATCH="${STLM_CONTEXTUAL_PIXEL_MICROBATCH:-8}"
export STLM_CONTEXTUAL_DESCRIPTION_MICROBATCH="${STLM_CONTEXTUAL_DESCRIPTION_MICROBATCH:-8}"
export STLM_CONTEXTUAL_LR="${STLM_CONTEXTUAL_LR:-0.0003}"
export STLM_CONTEXTUAL_SAVE_EVERY_STEPS="${STLM_CONTEXTUAL_SAVE_EVERY_STEPS:-250}"
export STLM_CONTEXTUAL_D_MODEL="${STLM_CONTEXTUAL_D_MODEL:-256}"
export STLM_CONTEXTUAL_HEADS="${STLM_CONTEXTUAL_HEADS:-8}"
export STLM_CONTEXTUAL_LAYERS="${STLM_CONTEXTUAL_LAYERS:-2}"
export STLM_CONTEXTUAL_FF_DIM="${STLM_CONTEXTUAL_FF_DIM:-1024}"
export STLM_CONTEXTUAL_PIXEL_POOL_GRID="${STLM_CONTEXTUAL_PIXEL_POOL_GRID:-8}"
export STLM_CONTEXTUAL_DESCRIPTION_POOL_BINS="${STLM_CONTEXTUAL_DESCRIPTION_POOL_BINS:-8}"
export PYTHONUNBUFFERED=1

if [ "$STLM_CONTEXTUAL_PIXEL_BATCH" -ne \
        "$STLM_CONTEXTUAL_DESCRIPTION_BATCH" ]; then
    printf '%s\n' \
        'pixel and description logical batches must match so both media see the same sampled windows' \
        >&2
    exit 2
fi

RUN_DIR="$PROTO_DIR/runs/$STLM_RUN_ID"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/run_contextual_readouts.log"

run_logged() {
    label="$1"
    shift
    log="$LOG_DIR/$label.log"
    printf -- '\n--- starting %s ---\n' "$label" | tee -a "$RUN_LOG"
    "$@" 2>&1 | tee -a "$log"
    status=${PIPESTATUS[0]}
    if [ "$status" -ne 0 ]; then
        printf -- '--- %s stopped with exit code %s; rerun to resume ---\n' \
            "$label" "$status" | tee -a "$RUN_LOG"
        return "$status"
    fi
    printf -- '--- %s complete ---\n' "$label" | tee -a "$RUN_LOG"
}

run_logged manifest "$PY" "$PROTO_DIR/write_run_manifest.py" || exit $?

printf '\n=== five-condition TR-C run started %s ===\n' "$(date)" | tee -a "$RUN_LOG"
printf 'derived run ID: %s\n' "$STLM_RUN_ID" | tee -a "$RUN_LOG"
printf 'frozen source run ID: %s\n' "$STLM_SOURCE_RUN_ID" | tee -a "$RUN_LOG"
printf 'direct baseline run ID: %s\n' \
    "$STLM_DIRECT_BASELINE_RUN_ID" | tee -a "$RUN_LOG"
printf 'seed: %s\n' "$STLM_SEED" | tee -a "$RUN_LOG"
printf 'training: %s windows x 128 aligned positions\n' \
    "$STLM_CONTEXTUAL_TRAIN_WINDOWS" | tee -a "$RUN_LOG"
printf 'evaluation: %s fixed final-position contexts\n' \
    "$STLM_READOUT_EVAL_TARGETS" | tee -a "$RUN_LOG"
printf 'logical batches: description=%s, pixel=%s\n' \
    "$STLM_CONTEXTUAL_DESCRIPTION_BATCH" \
    "$STLM_CONTEXTUAL_PIXEL_BATCH" | tee -a "$RUN_LOG"
printf 'microbatches: description=%s, pixel=%s\n' \
    "$STLM_CONTEXTUAL_DESCRIPTION_MICROBATCH" \
    "$STLM_CONTEXTUAL_PIXEL_MICROBATCH" | tee -a "$RUN_LOG"
printf 'TR-C: d_model=%s, heads=%s, layers=%s, ff=%s, lr=%s\n' \
    "$STLM_CONTEXTUAL_D_MODEL" "$STLM_CONTEXTUAL_HEADS" \
    "$STLM_CONTEXTUAL_LAYERS" "$STLM_CONTEXTUAL_FF_DIM" \
    "$STLM_CONTEXTUAL_LR" | tee -a "$RUN_LOG"

# Re-evaluate the three immutable published conditions rather than copying
# their recorded numbers without checking the checkpoints and holdout.
run_logged verify_source_conditions \
    "$PY" "$PROTO_DIR/verify_published_results.py" \
    "$STLM_SOURCE_RUN_ID" --medium both || exit $?
run_logged verify_direct_token \
    "$PY" "$PROTO_DIR/verify_published_results.py" \
    "$STLM_DIRECT_BASELINE_RUN_ID" || exit $?

# Each trainer resumes from its own derived checkpoint. Only the source SPs
# are loaded, frozen, and used to construct the causal state sequences.
run_logged description_tr_c \
    "$PY" "$PROTO_DIR/phase2_tr_c_description.py" || exit $?
run_logged pixel_tr_c \
    "$PY" "$PROTO_DIR/phase2_tr_c_pixel.py" || exit $?

# Detect any accidental mutation of the immutable source before promotion.
run_logged verify_source_unchanged \
    "$PY" "$PROTO_DIR/verify_contextual_source.py" || exit $?
run_logged collect_five_conditions \
    "$PY" "$PROTO_DIR/collect_tr_c_five_condition_results.py" || exit $?

printf '\n=== five-condition TR-C run complete %s ===\n' "$(date)" | tee -a "$RUN_LOG"
printf 'combined results: %s\n' \
    "$RUN_DIR/results/five_condition_comparison.json" | tee -a "$RUN_LOG"
