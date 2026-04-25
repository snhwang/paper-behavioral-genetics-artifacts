#!/bin/bash
# Machine 1 (RTX Pro 6000)
# Eval 3: 5 seeds, abundance epoch, locus recombination
# Eval 4: abundance + ice_age epochs, 5 seeds each, locus recombination

set -e
cd "$(dirname "$0")/../../.."

BASE_URL="http://192.168.1.175:11434/v1"
MODEL="gemma4:e2b"
TICKS=10000
CREATURES=30
OUT="examples/evolutionary_ecosystem/eval/results"
mkdir -p "$OUT"

echo "=== MACHINE 1: Eval 3 + Eval 4 (abundance, ice_age) ==="

# Eval 3 — inheritance fidelity (any epoch, we use abundance)
echo "--- Eval 3: inheritance fidelity (5 seeds, abundance, locus) ---"
for SEED in 42 1042 2042 3042 4042; do
    echo "  seed=$SEED"
    python -m examples.evolutionary_ecosystem.server.app \
        --seed $SEED \
        --lock-epoch abundance \
        --ticks $TICKS \
        --creatures $CREATURES \
        --recombination locus \
        --headless \
        --base-url $BASE_URL \
        --model $MODEL \
        --output "$OUT/eval3_seed${SEED}.json"
done

# Eval 4 — abundance epoch
echo "--- Eval 4: abundance epoch (5 seeds, locus) ---"
for SEED in 42 1042 2042 3042 4042; do
    echo "  seed=$SEED"
    python -m examples.evolutionary_ecosystem.server.app \
        --seed $SEED \
        --lock-epoch abundance \
        --ticks $TICKS \
        --creatures $CREATURES \
        --recombination locus \
        --headless \
        --base-url $BASE_URL \
        --model $MODEL \
        --output "$OUT/eval4_abundance_seed${SEED}.json"
done

# Eval 4 — ice_age epoch
echo "--- Eval 4: ice_age epoch (5 seeds, locus) ---"
for SEED in 42 1042 2042 3042 4042; do
    echo "  seed=$SEED"
    python -m examples.evolutionary_ecosystem.server.app \
        --seed $SEED \
        --lock-epoch ice_age \
        --ticks $TICKS \
        --creatures $CREATURES \
        --recombination locus \
        --headless \
        --base-url $BASE_URL \
        --model $MODEL \
        --output "$OUT/eval4_ice_age_seed${SEED}.json"
done

echo "=== Machine 1 complete ==="
