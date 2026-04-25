#!/bin/bash
# Machine 2 (GB10 Spark)
# Eval 5b blend: 5 seeds, LLM recombination (slow — needs Ollama)
# Eval 4: predator_bloom + expansion epochs, 5 seeds each, locus

set -e
cd "$(dirname "$0")/../../.."
export PYTHONPATH="examples/evolutionary_ecosystem:$PYTHONPATH"

BASE_URL="http://localhost:11434/v1"
MODEL="gemma4:e2b"
TICKS=10000
CREATURES=30
OUT="examples/evolutionary_ecosystem/eval/results"
mkdir -p "$OUT"

echo "=== MACHINE 2: Eval 5b blend + Eval 4 (predator_bloom, expansion) ==="

# Eval 5b — LLM blend condition
echo "--- Eval 5b: LLM blend (5 seeds) ---"
for SEED in 42 1042 2042 3042 4042; do
    echo "  seed=$SEED"
    python -m examples.evolutionary_ecosystem.server.app \
        --seed $SEED \
        --ticks $TICKS \
        --creatures $CREATURES \
        --recombination blend \
        --headless \
        --base-url $BASE_URL \
        --model $MODEL \
        --output "$OUT/eval5b_blend_seed${SEED}.json"
done

# Eval 4 — predator_bloom epoch
echo "--- Eval 4: predator_bloom epoch (5 seeds, locus) ---"
for SEED in 42 1042 2042 3042 4042; do
    echo "  seed=$SEED"
    python -m examples.evolutionary_ecosystem.server.app \
        --seed $SEED \
        --lock-epoch predator_bloom \
        --ticks $TICKS \
        --creatures $CREATURES \
        --recombination locus \
        --headless \
        --base-url $BASE_URL \
        --model $MODEL \
        --output "$OUT/eval4_predator_bloom_seed${SEED}.json"
done

# Eval 4 — expansion epoch
echo "--- Eval 4: expansion epoch (5 seeds, locus) ---"
for SEED in 42 1042 2042 3042 4042; do
    echo "  seed=$SEED"
    python -m examples.evolutionary_ecosystem.server.app \
        --seed $SEED \
        --lock-epoch expansion \
        --ticks $TICKS \
        --creatures $CREATURES \
        --recombination locus \
        --headless \
        --base-url $BASE_URL \
        --model $MODEL \
        --output "$OUT/eval4_expansion_seed${SEED}.json"
done

echo "=== Machine 2 complete ==="
