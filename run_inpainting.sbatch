#!/bin/bash
#SBATCH --job-name=silo_inpaint
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=1
#SBATCH --account=ynx@v100
#SBATCH -C v100-32g
#SBATCH --output=./logs/silo_inpaint_slurm_%j.out
#SBATCH --error=./logs/silo_inpaint_slurm_%j.err
#SBATCH --mail-user="sammy.sharief@umontreal.ca"
#SBATCH --mail-type=ALL

# set -euo pipefail

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

cd "$SILO_ROOT"

python silo_sample.py \
  --start_from_ckpt=/lustre/fsn1/projects/rech/ynx/uxl64xr/SILO/training/runs/7pumca5l_silo_operator/ckpts/checkpoint_step_30000.pt \
    --pretrained_model_name_or_path=/lustre/fswork/projects/rech/ynx/uxl64xr/models/sd15 \
    --diff_model=sd1.5 \
  --task_config=configs/latino_pro_inpainting_config.yaml \
  --sigma_condition \
  --clamp \
  --image_path=/lustre/fsn1/projects/rech/ynx/uxl64xr/latent_model_test_images/apt.png \
  --explicit_name=apt \
  --prompt="a photo of a living room" \
  --scale=1 \
  --save_mode=both \
  --folder_path=./sample_results/inpainting_apt \
  --disable_bar \
  --num_samples=100