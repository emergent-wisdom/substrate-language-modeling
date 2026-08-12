#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Render all 1024 Sana-Sprint visual canonicals for the STLM pixel substrate.
#
#   - Resumable: skips any word already rendered in data/512/
#   - Self-restarting: if a render crashes/stops, it resumes automatically
#   - Stays awake: re-execs itself under caffeinate so the Mac will not nap
#
# USAGE:
#   bash "render_canonicals.sh"          # just run it; it wraps itself in caffeinate
#
# RECOMMENDED first (so it can run unattended overnight): plug into power, then
#   sudo pmset -c disablesleep 1         # disable idle sleep on AC
#   ...                                  # (undo later with: sudo pmset -c disablesleep 0)
# ---------------------------------------------------------------------------
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR="$SCRIPT_DIR/stlm_prototype"
PY="$SCRIPT_DIR/.venv/bin/python"

if [ ! -x "$PY" ]; then
  PY="python"
fi

# Re-exec under caffeinate (display/idle/disk/system sleep all held) once.
if [ -z "${UNDER_CAFFEINATE:-}" ]; then
  exec env UNDER_CAFFEINATE=1 caffeinate -dimsu bash "$0" "$@"
fi

cd "$DIR" || { echo "cannot cd to $DIR"; exit 1; }

while [ ! -f data/canonicals.npz ]; do
  n=$(ls data/512/*.png 2>/dev/null | wc -l | tr -d ' ')
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $n/1024 rendered - (re)starting render..."
  "$PY" phase0_canonicals.py --native 512
  echo "[$(date '+%H:%M:%S')] render exited (code $?); resuming in 5s..."
  sleep 5
done

echo "DONE - all 1024 rendered; data/canonicals.npz written."
