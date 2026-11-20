#!/usr/bin/env bash
# R-MDH + bear-cash 5% + regime_flip threshold -3% combined
# Same 8 quarters as run_rmdh_8q.sh but with bear-onset protection.
set -uo pipefail
cd "$(dirname "$0")/../.."

BLEND=REGIME_ROUTER_STRICT_LO_SETUP60
UNIVERSE=u100
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT_DIR="runs/rmdh_bearcash_walkforward_${STAMP}"
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
  end_date=$(python -c "import datetime; print((datetime.date.fromisoformat('$start') + datetime.timedelta(days=$days-1)).isoformat())")
  log="$OUT_DIR/${BLEND}_RMDH_BEARCASH-${q}_${STAMP}.stdout.log"
  echo "Launching ${q}: ${start} -> ${end_date} ${days}d  log=${log}"
  PYTHONIOENCODING=utf-8 python -X utf8 scripts/strat_audit/paper_trade_replay_v3.py \
      --blend "$BLEND" --universe "$UNIVERSE" \
      --days "$days" --end-date "$end_date" --reset-state \
      --rmdh-min-hold-bars 3 \
      --rmdh-take-profit 0.15 \
      --rmdh-trail-stop 0.03 \
      --v3-bear-cash 0.05 \
      --regime-flip-threshold -0.03 \
      > "$log" 2>&1 &
  pids+=($!)
done

echo
echo "Launched ${#pids[@]} parallel R-MDH+bear-cash runs. PIDs: ${pids[@]}"
echo "Waiting for completion..."
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=$((fail+1))
  fi
done
echo
echo "Done. Failed: $fail / ${#pids[@]}"
ls -la logs/strat_audit/ | grep -E "_(2024|2025)[0-9]{4}_(2024|2025)[0-9]{4}.json" | tail -10
