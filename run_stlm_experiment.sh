#!/usr/bin/env bash

# Run one reproducible STLM experiment in dependency order. Each run ID has
# isolated checkpoints, logs, results, and completion markers.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_DIR="$SCRIPT_DIR/stlm_prototype"
DATA_DIR="$PROTO_DIR/data"
PY="$SCRIPT_DIR/.venv/bin/python"

if [ ! -x "$PY" ]; then
    PY="python"
fi

export STLM_RUN_ID="${STLM_RUN_ID:-replication_seed42_e1_n2-100k}"
export STLM_SEED="${STLM_SEED:-42}"
export STLM_MAX_SENTENCES="${STLM_MAX_SENTENCES:-50000}"
export STLM_TEXT_NN1_EPOCHS="${STLM_TEXT_NN1_EPOCHS:-1}"
export STLM_PIXEL_NN1_EPOCHS="${STLM_PIXEL_NN1_EPOCHS:-1}"
export STLM_NN2_TRAIN_TARGETS="${STLM_NN2_TRAIN_TARGETS:-100000}"
export STLM_NN2_EVAL_TARGETS="${STLM_NN2_EVAL_TARGETS:-1024}"
export PYTHONUNBUFFERED=1

RUN_DIR="$PROTO_DIR/runs/$STLM_RUN_ID"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/run_all.log"

"$PY" "$PROTO_DIR/write_run_manifest.py" || exit $?

printf '\n=== STLM run started %s ===\n' "$(date)" | tee -a "$RUN_LOG"
printf 'run ID: %s\n' "$STLM_RUN_ID" | tee -a "$RUN_LOG"
printf 'seed: %s\n' "$STLM_SEED" | tee -a "$RUN_LOG"
printf 'repo: %s\n' "$SCRIPT_DIR" | tee -a "$RUN_LOG"
printf 'python: %s\n' "$PY" | tee -a "$RUN_LOG"
printf 'max sentences: %s\n' "$STLM_MAX_SENTENCES" | tee -a "$RUN_LOG"
printf 'NN1 total epochs: text=%s, pixel=%s\n' \
    "$STLM_TEXT_NN1_EPOCHS" "$STLM_PIXEL_NN1_EPOCHS" | tee -a "$RUN_LOG"
printf 'matched NN2 targets: train=%s, eval=%s\n' \
    "$STLM_NN2_TRAIN_TARGETS" "$STLM_NN2_EVAL_TARGETS" | tee -a "$RUN_LOG"

run_stage() {
    stage="$1"
    shift
    marker="$LOG_DIR/$stage.done"
    log="$LOG_DIR/$stage.log"

    if [ -f "$marker" ]; then
        printf -- '--- %s already complete; skipping ---\n' "$stage" | tee -a "$RUN_LOG"
        return 0
    fi

    printf -- '\n--- starting %s ---\n' "$stage" | tee -a "$RUN_LOG"
    "$@" 2>&1 | tee -a "$log"
    status=${PIPESTATUS[0]}
    if [ "$status" -ne 0 ]; then
        printf -- '--- %s stopped with exit code %s; rerun to resume ---\n' \
            "$stage" "$status" | tee -a "$RUN_LOG"
        return "$status"
    fi

    touch "$marker"
    printf -- '--- %s complete ---\n' "$stage" | tee -a "$RUN_LOG"
}

if [ -f "$DATA_DIR/canonicals.npz" ]; then
    touch "$LOG_DIR/phase0_pixel.done"
    printf '%s\n' 'phase0_pixel: canonicals.npz already exists; skipping render' | tee -a "$RUN_LOG"
else
    run_stage phase0_pixel "$PY" "$PROTO_DIR/phase0_canonicals.py" --native 512 || exit $?
fi

if [ -f "$DATA_DIR/text_token_canonicals.npz" ]; then
    touch "$LOG_DIR/phase0_text.done"
    printf '%s\n' 'phase0_text: text canonicals already exist; skipping encoder pass' | tee -a "$RUN_LOG"
else
    run_stage phase0_text "$PY" "$PROTO_DIR/phase0_text_canonicals.py" || exit $?
fi

run_stage text_nn1 "$PY" "$PROTO_DIR/phase1_nn1_text.py" || exit $?
run_stage text_nn2 "$PY" "$PROTO_DIR/phase2_nn2_text.py" || exit $?
run_stage text_demo "$PY" "$PROTO_DIR/generate_text.py" \
    "the dog ran into the yard" || exit $?
run_stage pixel_nn1 "$PY" "$PROTO_DIR/phase1_nn1.py" || exit $?
run_stage pixel_nn2 "$PY" "$PROTO_DIR/phase2_nn2.py" || exit $?
run_stage pixel_demo "$PY" "$PROTO_DIR/h7_demo.py" \
    "the dog ran into the yard" || exit $?

printf '\n=== STLM run complete %s ===\n' "$(date)" | tee -a "$RUN_LOG"
printf 'run artifacts: %s\n' "$RUN_DIR" | tee -a "$RUN_LOG"
printf 'pixel H7 strip: %s\n' "$RUN_DIR/results/pixel_strip.png" | tee -a "$RUN_LOG"
