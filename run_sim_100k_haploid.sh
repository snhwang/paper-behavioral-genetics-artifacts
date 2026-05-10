#!/usr/bin/env bash
# =============================================================================
# run_sim_100k_haploid.sh — Behavioral Genetics paper figure data
#
# Runs the 100k-tick haploid simulation that anchors §sec:eval-diploid in the
# paper. Saves to chunked sim_log files (each <50MB so GitHub-friendly).
#
# Requires: bear v0.1.8+ installed, vLLM running gemma4:e2b on localhost:8355.
#
# Usage:
#   ./run_sim_100k_haploid.sh                                  # defaults
#   ./run_sim_100k_haploid.sh --base-url http://host:port/v1   # remote LLM
# =============================================================================

set -e
cd "$(dirname "$0")"

BASE_URL="${BASE_URL:-http://127.0.0.1:8355/v1}"
MODEL="${MODEL:-gemma4:e2b}"
SEED="${SEED:-42}"
TICKS="${TICKS:-100000}"
CREATURES="${CREATURES:-30}"
CHUNK_SIZE="${CHUNK_SIZE:-20000}"
OUTPUT="${OUTPUT:-sim_log_haploid.json}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-url) BASE_URL="$2"; shift 2 ;;
        --model)    MODEL="$2"; shift 2 ;;
        --seed)     SEED="$2"; shift 2 ;;
        --ticks)    TICKS="$2"; shift 2 ;;
        --output)   OUTPUT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=========================================="
echo "  100k Haploid Simulation"
echo "=========================================="
echo "  base-url:   $BASE_URL"
echo "  model:      $MODEL"
echo "  seed:       $SEED"
echo "  ticks:      $TICKS"
echo "  creatures:  $CREATURES"
echo "  chunk-size: $CHUNK_SIZE"
echo "  output:     $OUTPUT (chunked)"
echo "=========================================="

python3 evolutionary_ecosystem/run.py \
    --headless \
    --ticks       "$TICKS" \
    --creatures   "$CREATURES" \
    --seed        "$SEED" \
    --recombination locus \
    --ploidy      haploid \
    --base-url    "$BASE_URL" \
    --model       "$MODEL" \
    --output      "$OUTPUT" \
    --chunk-size  "$CHUNK_SIZE"

echo ""
echo "=== Haploid sim complete ==="
ls -lh "${OUTPUT%.json}"_*.json 2>/dev/null || ls -lh "$OUTPUT" 2>/dev/null
