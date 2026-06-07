#!/bin/bash
# ============================================================
# Dukascopy 1-Hour Downloader — Full Forex Basket
# ============================================================
set -e

DIR="data"
FROM="2020-01-01"
TO="2024-12-31"

PAIRS=(
    "eurusd" "gbpusd" "usdjpy" "usdchf" "audusd" "nzdusd" "usdcad"
    "eurgbp" "audnzd" "audcad" "eurchf" "gbpchf" "eurnok" "eursek" "noksek"
    "nzdcad" "audchf" "nzdchf" "cadchf" "euraud" "eurcad" "gbpaud"
    "chfjpy" "cadjpy" "gbpcad" "gbpnzd"
)

echo "=============================================="
echo " Dukascopy 1-Hour Download — ${#PAIRS[@]} Pairs"
echo " Range: $FROM → $TO  |  Timeframe: h1"
echo "=============================================="
echo ""

SUCCESS=0
FAIL=0

for PAIR in "${PAIRS[@]}"; do
    UPPER=$(echo "$PAIR" | tr '[:lower:]' '[:upper:]')
    echo "📡 Downloading $UPPER (1H) ..."
    
    # Skip if file already exists and has size > 10KB
    if [ -s "$DIR/${PAIR}_1h.csv" ] && [ $(stat -c %s "$DIR/${PAIR}_1h.csv") -gt 10000 ]; then
        echo "  ✓ ${PAIR}_1h.csv already exists. Skipping."
        SUCCESS=$((SUCCESS + 1))
        continue
    fi

    if npx dukascopy-node \
        -i "$PAIR" \
        -t h1 \
        -from "$FROM" \
        -to "$TO" \
        -v true \
        -f csv \
        -dir "$DIR" \
        -fn "${PAIR}_1h" \
        -s 2>/dev/null; then
        echo "  ✓ ${PAIR}_1h.csv saved"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "  ✗ FAILED to download $UPPER"
        FAIL=$((FAIL + 1))
    fi
    
    sleep 1
    echo ""
done

echo "=============================================="
echo " Download Complete"
echo " Success: $SUCCESS / ${#PAIRS[@]}"
echo " Failed:  $FAIL / ${#PAIRS[@]}"
echo "=============================================="
