#!/bin/bash
# Machine 3
# Eval 5b locus: 5 seeds, locus recombination (fast baseline)
# Eval 4: famine epoch, 5 seeds, locus

set -e
cd "$(dirname "$0")/../../.."

BASE_URL="http://localhost:11434/v1"
MODEL="gemma4:e2b"
TICKS=10000
CREATURES=30
OUT="examples/evolutionary_ecosystem/eval/results"
mkdir -p "$OUT"

echo "=== MACHINE 3: Eval 5b locus + Eval 4 (famine) ==="

# Eval 5b — locus baseline condition
echo "--- Eval 5b: locus baseline (5 seeds) ---"
for SEED in 42 1042 2042 3042 4042; do
    echo "  seed=$SEED"
    python -m examples.evolutionary_ecosystem.server.app \
        --seed $SEED \
        --ticks $TICKS \
        --creatures $CREATURES \
        --recombination locus \
        --headless \
        --base-url $BASE_URL \
        --model $MODEL \
        --output "$OUT/eval5b_locus_seed${SEED}.json"
done

# Eval 4 — famine epoch
echo "--- Eval 4: famine epoch (5 seeds, locus) ---"
for SEED in 42 1042 2042 3042 4042; do
    echo "  seed=$SEED"
    python -m examples.evolutionary_ecosystem.server.app \
        --seed $SEED \
        --lock-epoch famine \
        --ticks $TICKS \
        --creatures $CREATURES \
        --recombination locus \
        --headless \
        --base-url $BASE_URL \
        --model $MODEL \
        --output "$OUT/eval4_famine_seed${SEED}.json"
done

echo "=== Machine 3 complete ==="
