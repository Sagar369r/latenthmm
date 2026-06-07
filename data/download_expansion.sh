#!/bin/bash
# ============================================================
# Dukascopy Expansion Basket Downloader
# ============================================================
set -e

DIR="data"
FROM="2020-01-01"
TO="2024-12-31"

PAIRS=(
    "eurcad" "eurnzd" "eurchf"
    "gbpcad" "gbpnzd" "gbpaud"
    "audcad" "audnzd" "nzdcad"
    "chfjpy" "cadjpy" "nzdchf"
)

echo "=============================================="
echo " Downloading Expansion Basket — ${#PAIRS[@]} Pairs"
echo " Range: $FROM → $TO  |  Timeframe: d1"
echo "=============================================="

for PAIR in "${PAIRS[@]}"; do
    UPPER=$(echo "$PAIR" | tr '[:lower:]' '[:upper:]')
    echo "📡 Downloading $UPPER ..."
    
    npx dukascopy-node \
        -i "$PAIR" \
        -t d1 \
        -from "$FROM" \
        -to "$TO" \
        -v true \
        -f csv \
        -dir "$DIR" \
        -fn "${PAIR}_daily" \
        -s 2>/dev/null || echo "  ✗ FAILED to download $UPPER"
    
    sleep 1
done

echo "=============================================="
echo " Download Complete"
echo "=============================================="
