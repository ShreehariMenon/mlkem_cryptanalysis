#!/bin/bash
# Run the full ML-KEM data generation pipeline
# Usage: bash scripts/generate_datasets.sh [--samples N] [--batch B]

set -e  # exit on any error

SAMPLES=${1:-100000}
BATCH=${2:-5000}
OUTDIR="data/raw"
KAT_DIR="data/kat_vectors"

echo "Starting ML-KEM dataset generation..."
echo "Samples per variant : $SAMPLES"
echo "Batch size          : $BATCH"

python3 src/generation/mlkem_profiling_pipeline.py \
    --samples "$SAMPLES" \
    --batch   "$BATCH" \
    --outdir  "$OUTDIR"

# Move KAT artifacts to their own directory
mv "$OUTDIR"/*.hex  "$KAT_DIR/" 2>/dev/null || true
mv "$OUTDIR"/manifest.json "$KAT_DIR/" 2>/dev/null || true

echo "Done. Raw datasets in: $OUTDIR"
echo "KAT artifacts in    : $KAT_DIR"
