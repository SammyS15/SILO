#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:40:00
#SBATCH --account=rrg-lplevass
#SBATCH --gres=gpu:1
#SBATCH --job-name=silo_inpaint
#SBATCH --output=/home/sammys15/links/scratch/SILO/logs/slurm_%j.out
#SBATCH --error=/home/sammys15/links/scratch/SILO/logs/slurm_%j.err
#SBATCH --mail-user="sammy.sharief@umontreal.ca"
#SBATCH --mail-type=ALL

set -euo pipefail

SILO_ROOT="$HOME/links/scratch/SILO"
mkdir -p "$SILO_ROOT/logs"

module purge
module load python/3.11
module load gcc
module load arrow/23.0.1
module load opencv
module load ffmpeg

source "$SILO_ROOT/silo_env/bin/activate"

export PYTHONUNBUFFERED=1

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
  --task_config=configs/latino_pro_inpainting_config.yaml \
  --sigma_condition \
  --clamp \
  --image_path=/home/sammys15/links/scratch/latent_model_test_images/apt.png \
  --explicit_name=apt \
  --prompt="a photo of a living room" \
  --scale=1 \
  --save_mode=both \
  --folder_path=./sample_results/inpainting_apt \
  --disable_bar \
  --num_samples=5

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
