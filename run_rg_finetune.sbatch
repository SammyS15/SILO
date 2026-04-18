#!/bin/bash
#SBATCH --job-name=silo_rg_finetune
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=0:50:00
#SBATCH --gpus-per-node=1
#SBATCH --account=ynx@v100
#SBATCH -C v100-32g
#SBATCH --output=./logs/silo_rg_finetune_slurm_%j.out
#SBATCH --error=./logs/silo_rg_finetune_slurm_%j.err
#SBATCH --mail-user="sammy.sharief@umontreal.ca"
#SBATCH --mail-type=ALL

set -euo pipefail

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

cd "$SILO_ROOT/training"

# ---------------------------------------------------------------------------
# Fine-tune the COCO RG inpainting operator for the LATINO-PRO mask shape.
#
# Only the aggregation network trains; the UNet and VAE are frozen.
# Starting from ip_rg_coco_operator.pt means the network already understands
# indoor scenes — it just needs to learn the new masked-region target.
# Expect convergence well before max_steps=30000 (vs 200k from scratch).
#
# Checkpoints are saved under training/operator_ckpt/<wandb-run-id>/
# Monitor wandb (offline: run `wandb sync`) to find the best step.
# ---------------------------------------------------------------------------

python silo_rg_train.py \
  --config_path=train_rg_silo_config.yaml \
  --start_from_ckpt=../trained_ckpt/RG/ip_rg_coco_operator.pt \
  --task_config=../configs/latino_pro_inpainting_config.yaml \
  --dataset_name=coco \
  --train_batch_size=4 \
  --train_with_noise \
  --l1_loss \
  --clamp \
  --num_workers=6 \
  --wandb_mode=offline \
  --notes="finetune_coco_latino_pro_mask"
