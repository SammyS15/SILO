#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=11:59:00
#SBATCH --account=rrg-lplevass
#SBATCH --gres=gpu:1
#SBATCH --job-name=silo_rg_sd15
#SBATCH --output=/home/sammys15/links/scratch/Latent_Posterior_Sampling_Method_Comparsion/SILO/logs/slurm_%j.out
#SBATCH --error=/home/sammys15/links/scratch/Latent_Posterior_Sampling_Method_Comparsion/SILO/logs/slurm_%j.err
#SBATCH --mail-user="sammy.sharief@umontreal.ca"
#SBATCH --mail-type=ALL

set -euo pipefail

SILO_ROOT="/home/sammys15/links/scratch/Latent_Posterior_Sampling_Method_Comparsion/SILO"
# mkdir -p "$SILO_ROOT/logs"

module purge
module load python/3.11
module load gcc
module load arrow/23.0.1
module load opencv
module load ffmpeg

# Reuse the original SILO env — dependencies are unchanged
source "$HOME/links/scratch/SILO/silo_env/bin/activate"

export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

cd "$SILO_ROOT/training"

# ---------------------------------------------------------------------------
# Train the RG operator from scratch on SD 1.5.
#
# Override via env vars, e.g.:
#   TASK_CONFIG=../configs/super_resolution_4_config.yaml DATASET=ffhq sbatch run_rg_finetune_sd_1_5.sbatch
#
# No --start_from_ckpt: existing operators were trained on RV5.1 features;
# SD 1.5 UNet internals differ enough that warm-starting would be misleading.
# Training from scratch takes ~30 000 steps (~2-3 h on a single GPU).
# ---------------------------------------------------------------------------

TASK_CONFIG="${TASK_CONFIG:-../configs/gaussian_deblur_config.yaml}"
DATASET="${DATASET:-ffhq}"
NOTES="${NOTES:-rg_sd15_scratch}"

python silo_rg_train.py \
  --config_path=train_rg_silo_config.yaml \
  --task_config="$TASK_CONFIG" \
  --dataset_name="$DATASET" \
  --train_batch_size=4 \
  --train_with_noise \
  --l1_loss \
  --clamp \
  --num_workers=6 \
  --wandb_mode=offline \
  --notes="$NOTES"
