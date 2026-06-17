#!/usr/bin/env bash
# Launch 8 parallel R-MDH walk-forward quarters for STRICT_LO_SETUP60_RMDH.
# Each quarter is independent ($10k reset). Approximate runtime: 60-120 min
# per quarter; running in parallel takes ~60-120 min wall-clock.
#
# Usage:
#   bash scripts/audit/run_rmdh_8q.sh [config_name]
# config_name: 'baseline_only', 'rmdh', 'rmdh_r4'
#
# Produces JSON in logs/strat_audit/paper_trade_replay_v3_*_RMDH_*.json
set -uo pipefail
cd "$(dirname "$0")/../.."

BLEND=REGIME_ROUTER_STRICT_LO_SETUP60
UNIVERSE=u100
RMDH_HOLD=3
RMDH_TP=0.15
RMDH_TRAIL=0.03
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT_DIR="runs/rmdh_walkforward_${STAMP}"
mkdir -p "$OUT_DIR"

declare -a WINDOWS=(
  "24Q1:2024-01-01:90"
  "24Q2:2024-04-01:91"
  "24Q3:2024-07-01:92"
  "24Q4:2024-10-01:92"
  "25Q1:2025-01-01:90"
  "25Q2:2025-04-01:91"
  "25Q3:2025-07-01:92"
  "25Q4:2025-10-01:92"
)

pids=()
for w in "${WINDOWS[@]}"; do
  IFS=':' read -r q start days <<< "$w"
  # end-date is start + days - 1 (rough; v3 uses days parameter to walk)
  end_date=$(python -c "import datetime; print((datetime.date.fromisoformat('$start') + datetime.timedelta(days=$days-1)).isoformat())")
  log="$OUT_DIR/${BLEND}_RMDH-${q}_${STAMP}.stdout.log"
  echo "Launching ${q}: start=${start} end=${end_date} days=${days} log=${log}"
  PYTHONIOENCODING=utf-8 python -X utf8 scripts/strat_audit/paper_trade_replay_v3.py \
      --blend "$BLEND" --universe "$UNIVERSE" \
      --days "$days" --end-date "$end_date" --reset-state \
      --rmdh-min-hold-bars "$RMDH_HOLD" \
      --rmdh-take-profit "$RMDH_TP" \
      --rmdh-trail-stop "$RMDH_TRAIL" \
      > "$log" 2>&1 &
  pids+=($!)
done

echo
echo "Launched ${#pids[@]} parallel runs. PIDs: ${pids[@]}"
echo "Output dir: $OUT_DIR"
echo
echo "Waiting for completion..."
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=$((fail+1))
  fi
done

echo
echo "Done. Failed: $fail / ${#pids[@]}"
echo "Outputs in: $OUT_DIR"
echo "JSONs in: logs/strat_audit/"
ls -la logs/strat_audit/ | grep "_RMDH_" | tail -10
