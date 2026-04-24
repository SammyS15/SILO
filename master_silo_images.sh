#!/bin/bash
# Master script: submit one SILO job per image in IMAGE_DIR as a SLURM array.
#
# Run from the SILO/ directory:
#   bash master_silo_images.sh
#
# Override any parameter via env var:
#   NUM_SAMPLES=50 PROMPT="a photo of a room" bash master_silo_images.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Parameters (override via env vars) ───────────────────────────────────────
IMAGE_DIR="${IMAGE_DIR:-/lustre/fsn1/projects/rech/ynx/uxl64xr/latent_model_test_images}"
SILO_ROOT="${SILO_ROOT:-/lustre/fsn1/projects/rech/ynx/uxl64xr/SILO}"
CKPT="${CKPT:-$SILO_ROOT/training/runs/7pumca5l_silo_operator/ckpts/checkpoint_step_30000.pt}"
PRETRAINED="${PRETRAINED:-/lustre/fswork/projects/rech/ynx/uxl64xr/models/sd15}"
TASK_CONFIG="${TASK_CONFIG:-configs/latino_pro_inpainting_config.yaml}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
SCALE="${SCALE:-1}"
PROMPT="${PROMPT:-a high quality photo}"
SAVE_DIR="${SAVE_DIR:-$SILO_ROOT/sample_results}"
MAX_CONCURRENT="${MAX_CONCURRENT:-10}"

# ── Count images ─────────────────────────────────────────────────────────────
N_IMAGES=$(find "$IMAGE_DIR" -maxdepth 1 -name "*.png" | wc -l)

if [[ $N_IMAGES -eq 0 ]]; then
    echo "ERROR: no .png files found in $IMAGE_DIR" >&2
    exit 1
fi

ARRAY_LAST=$((N_IMAGES - 1))

mkdir -p "$SCRIPT_DIR/logs"

echo "============================================================"
echo " SILO  Image Array"
echo "  image_dir:    $IMAGE_DIR  ($N_IMAGES images)"
echo "  save_dir:     $SAVE_DIR"
echo "  num_samples:  $NUM_SAMPLES  per image"
echo "  prompt:       $PROMPT"
echo "  max_concurrent: $MAX_CONCURRENT"
echo "============================================================"

# ── Retry wrapper (mirrors ReSample) ─────────────────────────────────────────
sbatch_with_retry() {
    local max_attempts=5
    local wait=30
    local attempt job_id
    for attempt in $(seq 1 $max_attempts); do
        job_id=$(sbatch --parsable "$@" 2>&1)
        if [[ $? -eq 0 ]]; then
            echo "$job_id"
            return 0
        fi
        echo "$job_id" >&2
        echo "  sbatch attempt $attempt/$max_attempts failed. Retrying in ${wait}s..." >&2
        sleep "$wait"
        wait=$((wait * 2))
    done
    echo "ERROR: sbatch failed after $max_attempts attempts" >&2
    return 1
}

# ── Submit array ──────────────────────────────────────────────────────────────
ARRAY_JOB_ID=$(sbatch_with_retry \
    --array="0-${ARRAY_LAST}%${MAX_CONCURRENT}" \
    --export=ALL,\
IMAGE_DIR="$IMAGE_DIR",\
SILO_ROOT="$SILO_ROOT",\
CKPT="$CKPT",\
PRETRAINED="$PRETRAINED",\
TASK_CONFIG="$TASK_CONFIG",\
NUM_SAMPLES="$NUM_SAMPLES",\
SCALE="$SCALE",\
PROMPT="$PROMPT",\
SAVE_DIR="$SAVE_DIR" \
    "$SCRIPT_DIR/run_silo_images_array.sbatch")

echo ""
echo "Submitted array job: $ARRAY_JOB_ID  (tasks 0–${ARRAY_LAST}, ≤${MAX_CONCURRENT} concurrent)"
echo ""
echo "Monitor:  squeue -j $ARRAY_JOB_ID"
echo "Logs:     $SCRIPT_DIR/logs/silo_inpaint_${ARRAY_JOB_ID}_*.out"
echo "Results:  $SAVE_DIR/inpainting_<image_name>/"