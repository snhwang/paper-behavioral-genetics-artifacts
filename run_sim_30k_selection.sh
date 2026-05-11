#!/usr/bin/env bash
# =============================================================================
# run_sim_30k_selection.sh — Selection-pressure demonstration sims
#
# Runs short 30k-tick simulations with mutation_rate=0 across 5 seeds × 2
# ploidies (haploid, diploid_codominant). Mutation_rate=0 preserves founder
# action markers so selection has stable substrate to act on.
#
# Companion to the long 100k mutation_rate=0.15 runs, which demonstrate
# LLM-mediated mutation gradually purges action markers from the gene pool.
# Together the two sets of sims show:
#   - With mutation, markers decay -- selection has nothing to act on.
#   - Without mutation, selection visibly favors adaptive markers.
#
# Output: sim_log_selection_{ploidy}_seed{N}_NN.json chunks.
#
# Requires: bear v0.1.8+ installed, vLLM running gemma-4-e2b on
# localhost:8355.
# =============================================================================

set -e
cd "$(dirname "$0")"

BASE_URL="${BASE_URL:-http://127.0.0.1:8355/v1}"
MODEL="${MODEL:-gemma-4-e2b}"
TICKS="${TICKS:-30000}"
CREATURES="${CREATURES:-30}"
CHUNK_SIZE="${CHUNK_SIZE:-10000}"

SEEDS=(42 142 242 342 442)
PLOIDIES=(haploid diploid_codominant)

for ploidy in "${PLOIDIES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        out="sim_log_selection_${ploidy}_seed${seed}.json"
        echo "=========================================="
        echo "  ${ploidy} / seed ${seed} -> ${out}"
        echo "=========================================="
        python3 evolutionary_ecosystem/run.py \
            --ticks       "$TICKS" \
            --creatures   "$CREATURES" \
            --seed        "$seed" \
            --recombination locus \
            --ploidy      "$ploidy" \
            --mutation-rate 0 \
            --base-url    "$BASE_URL" \
            --model       "$MODEL" \
            --output      "$out" \
            --chunk-size  "$CHUNK_SIZE"
    done
done

echo ""
echo "=== Selection-pressure sims complete ==="
ls -lh sim_log_selection_*_seed*_*.json 2>/dev/null
