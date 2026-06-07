#!/bin/bash
# ============================================================
# Dukascopy Basket Downloader — 15-Pair Mean-Reversion Ensemble
# ============================================================
# Downloads Daily (d1) OHLCV data with volumes for all 15
# institutional cross-pairs from 2020-01-01 to 2024-12-31.
# ============================================================

set -e

DIR="data"
FROM="2020-01-01"
TO="2024-12-31"

PAIRS=(
    # Tier 1: Ultimate Rubber Bands
    "eurgbp"
    "audnzd"
    "audcad"
    "eurchf"
    # Tier 2: European Crosses
    "gbpchf"
    "eurnok"
    "eursek"
    "noksek"
    # Tier 3: Commodity Crosses
    "nzdcad"
    "audchf"
    "nzdchf"
    "cadchf"
    # Tier 4: Heavy Crosses
    "euraud"
    "eurcad"
    "gbpaud"
)

echo "=============================================="
echo " Dukascopy Basket Download — ${#PAIRS[@]} Pairs"
echo " Range: $FROM → $TO  |  Timeframe: d1"
echo "=============================================="
echo ""

SUCCESS=0
FAIL=0

for PAIR in "${PAIRS[@]}"; do
    UPPER=$(echo "$PAIR" | tr '[:lower:]' '[:upper:]')
    echo "📡 Downloading $UPPER ..."
    
    if npx dukascopy-node \
        -i "$PAIR" \
        -t d1 \
        -from "$FROM" \
        -to "$TO" \
        -v true \
        -f csv \
        -dir "$DIR" \
        -fn "${PAIR}_daily" \
        -s 2>/dev/null; then
        echo "  ✓ ${PAIR}_daily.csv saved"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "  ✗ FAILED to download $UPPER"
        FAIL=$((FAIL + 1))
    fi
    
    # Brief pause between downloads to avoid rate limiting
    sleep 1
    echo ""
done

echo "=============================================="
echo " Download Complete"
echo " Success: $SUCCESS / ${#PAIRS[@]}"
echo " Failed:  $FAIL / ${#PAIRS[@]}"
echo "=============================================="
echo ""
echo "Files saved to ./$DIR/"
ls -lh "$DIR"/*_daily.csv 2>/dev/null || echo "No files found."
