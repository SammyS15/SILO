#!/bin/bash
#SBATCH --job-name=silo_rg_sd15
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=10:50:00
#SBATCH --gpus-per-node=1
#SBATCH --account=ynx@v100
#SBATCH -C v100-32g
#SBATCH --output=./logs/silo_rg_sd15_slurm_%j.out
#SBATCH --error=./logs/silo_rg_sd15_slurm_%j.err
#SBATCH --mail-user="sammy.sharief@umontreal.ca"
#SBATCH --mail-type=ALL

# set -eo pipefail

SILO_ROOT="/lustre/fsn1/projects/rech/ynx/uxl64xr/SILO"
mkdir -p "$SILO_ROOT/logs"

module purge

# only keep system-level GPU/compiler tools if needed
module load gcc
module load opencv
module load ffmpeg

export PATH="/lustre/fsn1/projects/rech/ynx/uxl64xr/conda_envs/mamba/bin:$PATH"


# ---- conda/mamba setup ----
eval "$(mamba shell hook --shell bash)"
mamba activate silo

export HF_HOME="$SCRATCH/hf_cache"
mkdir -p "$HF_HOME"

export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

export HF_HOME=/lustre/fsn1/projects/rech/ynx/uxl64xr/hf_cache
export TORCH_HOME=/lustre/fsn1/projects/rech/ynx/uxl64xr/torch_cache
export HF_DATASETS_CACHE="/lustre/fswork/projects/rech/ynx/uxl64xr/datasets/hf_cache"

cd "$SILO_ROOT/training"

TASK_CONFIG="${TASK_CONFIG:-../configs/latino_pro_inpainting_config.yaml}"
DATASET="${DATASET:-ffhq}"
DATASET_PATH="${DATASET_PATH:-/lustre/fswork/projects/rech/ynx/uxl64xr/datasets/ffhq-hf}"
NOTES="${NOTES:-rg_sd15_scratch}"

python silo_rg_train.py \
  --config_path=train_rg_silo_config.yaml \
  --task_config="$TASK_CONFIG" \
  --dataset_name="$DATASET" \
  --dataset_path="$DATASET_PATH" \
  --train_batch_size=4 \
  --train_with_noise \
  --l1_loss \
  --clamp \
  --num_workers=6 \
  --wandb_mode=offline \
  --notes="$NOTES"
