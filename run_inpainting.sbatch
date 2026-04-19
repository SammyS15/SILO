#!/bin/bash
#SBATCH --job-name=silo_inpaint
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=0:50:00
#SBATCH --gpus-per-node=1
#SBATCH --account=ynx@v100
#SBATCH -C v100-32g
#SBATCH --output=./logs/silo_inpaint_slurm_%j.out
#SBATCH --error=./logs/silo_inpaint_slurm_%j.err
#SBATCH --mail-user="sammy.sharief@umontreal.ca"
#SBATCH --mail-type=ALL

set -euo pipefail
export SILO_ROOT="/lustre/fsn1/projects/rech/ynx/uxl64xr/SILO"
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

# export HF_HOME="$SCRATCH/hf_cache"
# mkdir -p "$HF_HOME"

export PYTHONUNBUFFERED=1

export HF_HOME=/lustre/fsn1/projects/rech/ynx/uxl64xr/hf_cache
export TORCH_HOME=/lustre/fsn1/projects/rech/ynx/uxl64xr/torch_cache

cd "$SILO_ROOT"

# ---------------------------------------------------------------------------
# Inpainting inference — no training required.
#
# Pre-trained Readout Guidance (RG) checkpoints are already included in
# trained_ckpt/RG/ for both center-box and random-box inpainting on FFHQ
# and COCO. No CNN training is needed for inpainting; CNN checkpoints only
# exist for super-resolution tasks.
#
# Variants (swap the two lines below to switch):
#   Center-box inpainting on FFHQ (default):
#     --start_from_ckpt=trained_ckpt/RG/ip_rg_ffhq_operator.pt
#     --task_config=configs/center_inpainting_config.yaml
#   Random-box inpainting on FFHQ:
#     --start_from_ckpt=trained_ckpt/RG/rand_ip_rg_ffhq_operator.pt
#     --task_config=configs/random_box_inpainting_config.yaml
#   Center-box inpainting on COCO:
#     --start_from_ckpt=trained_ckpt/RG/ip_rg_coco_operator.pt
#     --task_config=configs/center_inpainting_config.yaml
#     --dataset_name=coco
#
# Key arguments:
#   --idx_image   : index of the image to restore from the dataset (0-based)
#   --start/--end : run a batch range, e.g. --start=0 --end=10
#   --scale       : gradient step size η (paper default = 1 for inpainting)
#   --save_mode   : "grid" | "image" | "both"
#   --folder_path : where to write results
# ---------------------------------------------------------------------------

python silo_sample.py \
  --start_from_ckpt=trained_ckpt/RG/ip_rg_coco_operator.pt \
  --pretrained_model_name_or_path=$HF_HOME/models/realistic-vision-v51
  --task_config=configs/latino_pro_inpainting_config.yaml \
  --sigma_condition \
  --clamp \
  --image_path=../latent_model_test_images/apt.png \
  --explicit_name=apt \
  --prompt="a photo of a living room" \
  --scale=1 \
  --save_mode=both \
  --folder_path=./sample_results/inpainting_apt \
  --disable_bar \
  --num_samples=1

# ---------------------------------------------------------------------------
# Optional: train your own CNN operator (only needed if you want to use the
# CNN-based method instead of Readout Guidance, or for a custom dataset/task).
#
# Example: train a CNN inpainting operator on FFHQ
# ---------------------------------------------------------------------------
# python training/silo_cnn_train.py \
#   --pretrained_model_name_or_path="botp/stable-diffusion-v1-5" \
#   --dataset_name=ffhq \
#   --task_config=configs/center_inpainting_config.yaml \
#   --sigma_condition \
#   --train_with_noise \
#   --l1_loss \
#   --clamp \
#   --learning_rate=1e-4 \
#   --train_batch_size=4 \
#   --wandb_mode=disabled
