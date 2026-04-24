#!/bin/bash
#SBATCH --job-name=silo_inpaint_arr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=1
#SBATCH --account=ynx@v100
#SBATCH -C v100-32g
#SBATCH --output=./logs/silo_inpaint_%A_%a.out
#SBATCH --error=./logs/silo_inpaint_%A_%a.err
#SBATCH --mail-user="sammy.sharief@umontreal.ca"
#SBATCH --mail-type=ALL

# set -euo pipefail

# ── Pick image for this array task ───────────────────────────────────────────
IMAGE_DIR="${IMAGE_DIR:-/lustre/fsn1/projects/rech/ynx/uxl64xr/latent_model_test_images}"
# Build sorted list of .png files and select by task index
mapfile -t IMAGES < <(find "$IMAGE_DIR" -maxdepth 1 -name "*.png" | sort)
if [[ $SLURM_ARRAY_TASK_ID -ge ${#IMAGES[@]} ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID >= ${#IMAGES[@]} images" >&2
    exit 1
fi

IMAGE_PATH="${IMAGES[$SLURM_ARRAY_TASK_ID]}"
IMAGE_TAG="$(basename "${IMAGE_PATH%.*}")"

# ── Parameters (override via env vars from master script) ────────────────────
SILO_ROOT="${SILO_ROOT:-/lustre/fsn1/projects/rech/ynx/uxl64xr/SILO}"
CKPT="${CKPT:-$SILO_ROOT/training/runs/7pumca5l_silo_operator/ckpts/checkpoint_step_30000.pt}"
PRETRAINED="${PRETRAINED:-/lustre/fswork/projects/rech/ynx/uxl64xr/models/sd15}"
TASK_CONFIG="${TASK_CONFIG:-configs/latino_pro_inpainting_config.yaml}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
SCALE="${SCALE:-1}"
PROMPT="${PROMPT:-a high quality photo}"
SAVE_DIR="${SAVE_DIR:-$SILO_ROOT/sample_results}"
OUT_FOLDER="$SAVE_DIR/inpainting_${IMAGE_TAG}"

echo "=================================================="
echo " SILO  task $SLURM_ARRAY_TASK_ID / image: $IMAGE_TAG"
echo "  image:       $IMAGE_PATH"
echo "  out_folder:  $OUT_FOLDER"
echo "  num_samples: $NUM_SAMPLES"
echo "  prompt:      $PROMPT"
echo "=================================================="

# ── Environment ──────────────────────────────────────────────────────────────
module purge
module load gcc
module load opencv
module load ffmpeg
export PATH="/lustre/fsn1/projects/rech/ynx/uxl64xr/conda_envs/mamba/bin:$PATH"
eval "$(mamba shell hook --shell bash)"
mamba activate silo

export HF_HOME=/lustre/fsn1/projects/rech/ynx/uxl64xr/hf_cache
export TORCH_HOME=/lustre/fsn1/projects/rech/ynx/uxl64xr/torch_cache
export HF_DATASETS_CACHE="/lustre/fswork/projects/rech/ynx/uxl64xr/datasets/hf_cache"
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

# ── Run ───────────────────────────────────────────────────────────────────────

cd "$SILO_ROOT"
python silo_sample.py \
    --start_from_ckpt="$CKPT" \
    --pretrained_model_name_or_path="$PRETRAINED" \
    --diff_model=sd1.5 \
    --task_config="$TASK_CONFIG" \
    --sigma_condition \
    --clamp \
    --image_path="$IMAGE_PATH" \
    --explicit_name="$IMAGE_TAG" \
    --prompt="$PROMPT" \
    --scale="$SCALE" \
    --save_mode=both \
    --folder_path="$OUT_FOLDER" \
    --disable_bar \
    --num_samples="$NUM_SAMPLES"