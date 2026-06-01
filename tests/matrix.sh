#!/usr/bin/env bash
# matrix.sh — orchestrates the channel×load detection matrix.
#
# Runs channel_probe.js across the grid:
#   channels: read (chromium), cache (chromium), flush (firefox)
#   loads:    dd, burn
#
# Each cell's output is copied to /tmp/matrix-<channel>-<load>.{csv,json}.
# Idempotent: skips a cell if both output files already exist and are non-empty.
#
# Usage:
#   cd /home/bram/work/opfs-ssd-timing
#   bash tests/matrix.sh
#
# Full run takes ~6 playwright sessions (~3-4 min each) = ~20-25 min total.

set -euo pipefail

SKILL_DIR="/home/bram/.claude/skills/playwright-skill"
PROBE="$(cd "$(dirname "$0")" && pwd)/channel_probe.js"
N_WIN=1

run_cell() {
  local channel="$1"
  local load="$2"
  local browser="$3"
  local out_csv="/tmp/matrix-${channel}-${load}.csv"
  local out_json="/tmp/matrix-${channel}-${load}.json"

  if [[ -s "$out_csv" && -s "$out_json" ]]; then
    echo "[SKIP] ${channel}×${load} — outputs already exist at ${out_csv}"
    return 0
  fi

  echo "[RUN ] ${channel}×${load} (browser=${browser}, N_WIN=${N_WIN})"
  (
    cd "$SKILL_DIR"
    CHANNEL="$channel" LOAD="$load" BROWSER="$browser" N_WIN="$N_WIN" \
      node run.js "$PROBE"
  )

  # channel_probe.js writes to fixed paths; copy before next run overwrites them
  cp "/tmp/frost-${channel}.csv"        "$out_csv"
  cp "/tmp/frost-${channel}-marks.json" "$out_json"
  echo "[DONE] saved ${out_csv} and ${out_json}"
}

echo "=== FROST channel×load matrix ==="
echo "probe: $PROBE"
echo "skill: $SKILL_DIR"
echo ""

# read channel: SSD direct-read timing — meaningful under disk (dd) and CPU (burn)
run_cell read dd   chromium
run_cell read burn chromium

# cache channel: LLC occupancy timing — meaningful under CPU/LLC (burn) and (maybe) dd
run_cell cache dd   chromium
run_cell cache burn chromium

# flush channel: write-flush timing — only a real disk channel on Firefox
run_cell flush dd   firefox
run_cell flush burn firefox

echo ""
echo "=== Matrix complete. Run eval_matrix.py to analyze: ==="
echo "python3 $(cd "$(dirname "$0")" && pwd)/eval_matrix.py"
